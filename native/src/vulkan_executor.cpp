#include "executor.h"
#include "encoder.h"
#include "dpt.h"
#include "gpu_preprocess.h"
#include "gpu_model.h"
#include "image.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace dad {
namespace {

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kGpuSlotCount = 3;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

ComPtr<ID3D12Device> matching_d3d12_device(
    std::uint64_t adapter_luid) {
    if (adapter_luid == 0) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated =
            factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(enumerated, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(
            adapter->GetDesc1(&description),
            "IDXGIAdapter1::GetDesc1");
        std::uint64_t candidate = 0;
        static_assert(
            sizeof(candidate) == sizeof(description.AdapterLuid));
        std::memcpy(
            &candidate,
            &description.AdapterLuid,
            sizeof(candidate));
        if (candidate != adapter_luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

struct SharedD3D12Output {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;

    SharedD3D12Output() = default;
    SharedD3D12Output(const SharedD3D12Output&) = delete;
    SharedD3D12Output& operator=(const SharedD3D12Output&) = delete;
    SharedD3D12Output(SharedD3D12Output&& other) noexcept
        : resource(std::move(other.resource)),
          fence(std::move(other.fence)),
          resource_handle(
              std::exchange(other.resource_handle, nullptr)),
          fence_handle(
              std::exchange(other.fence_handle, nullptr)) {}
    SharedD3D12Output& operator=(
        SharedD3D12Output&& other) noexcept {
        if (this != &other) {
            if (resource_handle != nullptr) CloseHandle(resource_handle);
            if (fence_handle != nullptr) CloseHandle(fence_handle);
            resource = std::move(other.resource);
            fence = std::move(other.fence);
            resource_handle =
                std::exchange(other.resource_handle, nullptr);
            fence_handle =
                std::exchange(other.fence_handle, nullptr);
        }
        return *this;
    }

    ~SharedD3D12Output() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

struct GpuSlot {
    std::atomic<bool> occupied{false};
    SharedD3D12Output shared;
    GpuOutputKind kind = GpuOutputKind::buffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t fence_value = 0;
};

SharedD3D12Output create_shared_output(
    ID3D12Device* device,
    std::uint64_t bytes) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        bytes,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    SharedD3D12Output result;
    check_hresult(
        device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &description,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(DAD output)");
    check_hresult(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.fence)),
        "CreateFence(DAD output)");
    check_hresult(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.resource_handle),
        "CreateSharedHandle(DAD output)");
    check_hresult(
        device->CreateSharedHandle(
            result.fence.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.fence_handle),
        "CreateSharedHandle(DAD output fence)");
    return result;
}

SharedD3D12Output create_shared_texture_output(
    ID3D12Device* device,
    std::uint32_t width,
    std::uint32_t height) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0,
        width,
        height,
        1,
        1,
        DXGI_FORMAT_R32_FLOAT,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    SharedD3D12Output result;
    check_hresult(
        device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &description,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(DAD texture output)");
    check_hresult(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.fence)),
        "CreateFence(DAD texture output)");
    check_hresult(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.resource_handle),
        "CreateSharedHandle(DAD texture output)");
    check_hresult(
        device->CreateSharedHandle(
            result.fence.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.fence_handle),
        "CreateSharedHandle(DAD texture output fence)");
    return result;
}

std::shared_ptr<GpuSlot> acquire_gpu_slot(
    const std::array<
        std::shared_ptr<GpuSlot>, kGpuSlotCount>& slots,
    std::atomic<std::uint32_t>& next_slot) {
    const std::uint32_t first =
        next_slot.fetch_add(1, std::memory_order_relaxed) %
        kGpuSlotCount;
    for (std::uint32_t offset = 0;
         offset < kGpuSlotCount;
         ++offset) {
        const std::shared_ptr<GpuSlot>& slot =
            slots[(first + offset) % kGpuSlotCount];
        bool expected = false;
        if (slot->occupied.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            return slot;
        }
    }
    throw GpuSlotsExhausted();
}

VulkanBuffer prepare_buffer_output(
    GpuSlot& slot,
    ID3D12Device* device,
    VulkanContext& context,
    std::uint32_t width,
    std::uint32_t height) {
    if (slot.shared.resource != nullptr &&
        slot.kind == GpuOutputKind::buffer &&
        slot.width == width &&
        slot.height == height) {
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(width) *
            height * sizeof(float);
        return context.import_d3d12_buffer(
            slot.shared.resource_handle, bytes);
    }
    slot.shared = SharedD3D12Output{};
    slot.width = 0;
    slot.height = 0;
    slot.fence_value = 0;
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(width) *
        height * sizeof(float);
    slot.shared = create_shared_output(device, bytes);
    VulkanBuffer output = context.import_d3d12_buffer(
        slot.shared.resource_handle, bytes);
    slot.kind = GpuOutputKind::buffer;
    slot.width = width;
    slot.height = height;
    return output;
}

VulkanImage prepare_texture_output(
    GpuSlot& slot,
    ID3D12Device* device,
    VulkanContext& context,
    std::uint32_t width,
    std::uint32_t height) {
    if (slot.shared.resource != nullptr &&
        slot.kind == GpuOutputKind::texture &&
        slot.width == width &&
        slot.height == height) {
        return context.import_d3d12_image(
            slot.shared.resource_handle,
            width,
            height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
    slot.shared = SharedD3D12Output{};
    slot.width = 0;
    slot.height = 0;
    slot.fence_value = 0;
    slot.shared =
        create_shared_texture_output(device, width, height);
    VulkanImage output = context.import_d3d12_image(
        slot.shared.resource_handle,
        width,
        height,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    slot.kind = GpuOutputKind::texture;
    slot.width = width;
    slot.height = height;
    return output;
}

void validate_shared_texture(
    ID3D12Device* device,
    std::uintptr_t handle,
    std::uint32_t width,
    std::uint32_t height,
    dad_gpu_pixel_format format) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(handle),
            IID_PPV_ARGS(&resource)),
        "OpenSharedHandle(DAD texture input)");
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    const DXGI_FORMAT expected =
        format == DAD_GPU_PIXEL_BGRA8
        ? DXGI_FORMAT_B8G8R8A8_UNORM
        : DXGI_FORMAT_R8G8B8A8_UNORM;
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Width != width ||
        description.Height != height ||
        description.DepthOrArraySize != 1 ||
        description.MipLevels != 1 ||
        description.SampleDesc.Count != 1 ||
        description.Format != expected) {
        throw std::invalid_argument(
            "shared D3D12 texture does not match its descriptor");
    }
}

GpuCapabilities d3d12_capabilities(
    const VulkanContext& context,
    ID3D12Device* device) {
    GpuCapabilities result;
    const VulkanExternalCapabilities& external =
        context.external_capabilities();
    if (external.d3d12_resource_import &&
        external.d3d12_fence_import &&
        device != nullptr) {
        result.flags =
            DAD_GPU_CAP_D3D12_SHARED_INPUT |
            DAD_GPU_CAP_D3D12_FENCE_WAIT |
            DAD_GPU_CAP_D3D12_SHARED_OUTPUT |
            DAD_GPU_CAP_D3D12_FENCE_SIGNAL |
            DAD_GPU_CAP_ASYNC_SUBMIT |
            DAD_GPU_CAP_CANCELLATION |
            DAD_GPU_CAP_NO_HOST_PIXEL_STAGING |
            DAD_GPU_CAP_NO_HOST_DEPTH_STAGING;
        result.adapter_luid = context.adapter_luid();
        result.maximum_in_flight_jobs = kGpuSlotCount;
        if (external.d3d12_bgra8_sampled_image_import &&
            external.d3d12_rgba8_sampled_image_import &&
            external.d3d12_r32_storage_image_import) {
            result.flags |=
                DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
                DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
        }
    }
    return result;
}

class VulkanGpuJob final : public GpuJob {
public:
    VulkanGpuJob(
        std::shared_ptr<GpuSlot> slot,
        VulkanBuffer output_buffer,
        VulkanImage input_image,
        VulkanImage output_image,
        VulkanSubmission submission,
        std::uint32_t width,
        std::uint32_t height,
        std::uint64_t fence_value,
        std::uint64_t source_frame_id,
        std::uint64_t timestamp_ns)
        : slot_(std::move(slot)),
          output_buffer_(std::move(output_buffer)),
          input_image_(std::move(input_image)),
          output_image_(std::move(output_image)),
          submission_(std::move(submission)),
          width_(width),
          height_(height),
          fence_value_(fence_value),
          source_frame_id_(source_frame_id),
          timestamp_ns_(timestamp_ns) {}

    ~VulkanGpuJob() override {
        try {
            submission_.wait();
        } catch (...) {
        }
        // Release per-submission descriptors and the imported input before
        // another job can claim and record work against this output slot.
        submission_ = VulkanSubmission{};
        output_image_ = VulkanImage{};
        input_image_ = VulkanImage{};
        output_buffer_ = VulkanBuffer{};
        slot_->occupied.store(false, std::memory_order_release);
    }

    dad_gpu_job_state state() const override {
        if (cancelled_.load(std::memory_order_acquire)) {
            return DAD_GPU_JOB_CANCELLED;
        }
        return submission_.ready()
            ? DAD_GPU_JOB_COMPLETE
            : DAD_GPU_JOB_RUNNING;
    }

    void cancel() override {
        cancelled_.store(true, std::memory_order_release);
    }

    GpuOutput output() const override {
        if (cancelled_.load(std::memory_order_acquire)) {
            throw std::runtime_error("GPU job was cancelled");
        }
        GpuOutput result;
        result.kind = slot_->kind;
        result.width = width_;
        result.height = height_;
        result.row_stride_bytes = width_ * sizeof(float);
        result.byte_size =
            static_cast<std::uint64_t>(result.row_stride_bytes) * height_;
        result.shared_resource_handle =
            reinterpret_cast<std::uintptr_t>(
                slot_->shared.resource_handle);
        result.ready_fence_handle =
            reinterpret_cast<std::uintptr_t>(
                slot_->shared.fence_handle);
        result.ready_fence_value = fence_value_;
        result.source_frame_id = source_frame_id_;
        result.timestamp_ns = timestamp_ns_;
        return result;
    }

private:
    std::shared_ptr<GpuSlot> slot_;
    // Per-job Vulkan imports outlive the submission that references them.
    VulkanBuffer output_buffer_;
    VulkanImage input_image_;
    VulkanImage output_image_;
    VulkanSubmission submission_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint64_t fence_value_ = 0;
    std::uint64_t source_frame_id_ = 0;
    std::uint64_t timestamp_ns_ = 0;
    std::atomic<bool> cancelled_{false};
};
#endif

class VulkanExecutor final : public Executor {
public:
    VulkanExecutor(
        const std::string& model_path,
        dad_encoder encoder,
        int vulkan_device_index)
        : model_(model_path, encoder),
          context_(
              static_cast<std::uint32_t>(vulkan_device_index),
              encoder != DAD_ENCODER_VITL),
          weights_(model_, context_),
          operators_(context_),
          preprocessor_(context_),
          encoder_(encoder, context_, weights_, operators_),
          dpt_(encoder, context_, weights_, operators_)
#if defined(_WIN32)
          , d3d12_device_(
              matching_d3d12_device(context_.adapter_luid())),
          gpu_slots_{
              std::make_shared<GpuSlot>(),
              std::make_shared<GpuSlot>(),
              std::make_shared<GpuSlot>()}
#endif
          {}

    void infer(const float* input, int width, int height, float* output) override {
        infer_resized(input, width, height, output, width, height);
    }

    void infer_resized(
        const float* input,
        int width,
        int height,
        float* output,
        int output_width,
        int output_height) override {
        const std::size_t input_elements =
            static_cast<std::size_t>(width) * height * 3;
        VulkanBuffer image =
            context_.create_device_buffer(input_elements * sizeof(float));
        context_.upload(
            image, input, input_elements * sizeof(float));
        FeatureMap depth;
        const auto run_graph = [&] {
            EncoderOutput encoded = encoder_.forward(
                image,
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height));
            depth = dpt_.forward(std::move(encoded));
            if (output_width != width || output_height != height) {
                VulkanBuffer resized = context_.create_device_buffer(
                    static_cast<std::size_t>(output_width) *
                    output_height * sizeof(float));
                operators_.bilinear_align_false(
                    resized,
                    depth.buffer,
                    static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height),
                    static_cast<std::uint32_t>(output_width),
                    static_cast<std::uint32_t>(output_height),
                    1);
                depth.buffer = std::move(resized);
            }
        };
        const std::uint64_t tokens =
            std::uint64_t(width / 14) * (height / 14) + 1;
        if (tokens <= 2000) {
            encoder_.prepare(
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height));
            context_.batch(run_graph);
        } else {
            run_graph();
        }
        context_.download(
            depth.buffer,
            output,
            static_cast<std::size_t>(output_width) *
                output_height * sizeof(float));
    }

    GpuCapabilities gpu_capabilities() const override {
#if defined(_WIN32)
        return d3d12_capabilities(
            context_, d3d12_device_.Get());
#else
        return {};
#endif
    }

    std::unique_ptr<GpuJob> submit_gpu(
        const GpuSubmitRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error(
            "D3D12 GPU submission is only available on Windows");
#else
        const GpuCapabilities capabilities = gpu_capabilities();
        if (capabilities.flags == 0) {
            throw std::runtime_error(
                "complete D3D12/Vulkan GPU interop is unavailable");
        }
        if (request.width == 0 || request.height == 0 ||
            request.row_stride_bytes < request.width * 4u ||
            request.shared_resource_handle == 0 ||
            request.wait_fence_handle == 0 ||
            (request.pixel_format != DAD_GPU_PIXEL_BGRA8 &&
             request.pixel_format != DAD_GPU_PIXEL_RGBA8)) {
            throw std::invalid_argument(
                "invalid D3D12 GPU inference request");
        }
        const std::uint64_t required_input =
            static_cast<std::uint64_t>(request.row_stride_bytes) *
            request.height;
        if (required_input > request.resource_byte_size ||
            request.width >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            request.height >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(
                "D3D12 GPU input dimensions exceed the resource");
        }
        std::shared_ptr<GpuSlot> slot =
            acquire_gpu_slot(gpu_slots_, next_gpu_slot_);
        try {
            const ImageShape shape = network_shape(
                static_cast<int>(request.width),
                static_cast<int>(request.height),
                request.input_size);
            VulkanBuffer output = prepare_buffer_output(
                *slot,
                d3d12_device_.Get(),
                context_,
                request.width,
                request.height);
            const std::uint64_t signal_value =
                ++slot->fence_value;
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(
                    request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
            VulkanSubmission submission = context_.batch_async(
                std::move(wait),
                std::move(signal),
                [&] {
                    VulkanBuffer source =
                        context_.import_d3d12_buffer(
                            reinterpret_cast<void*>(
                                request.shared_resource_handle),
                            request.resource_byte_size);
                    VulkanBuffer image =
                        context_.create_device_buffer(
                            static_cast<std::uint64_t>(
                                shape.width) *
                            shape.height * 3u * sizeof(float));
                    context_.acquire_external_buffer(
                        source, VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_buffer(
                        output,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    preprocessor_.run(
                        image,
                        source,
                        request.width,
                        request.height,
                        request.row_stride_bytes,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height),
                        request.pixel_format ==
                                DAD_GPU_PIXEL_BGRA8
                            ? GpuPixelOrder::bgra
                            : GpuPixelOrder::rgba);
                    EncoderOutput encoded = encoder_.forward(
                        image,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    FeatureMap depth =
                        dpt_.forward(std::move(encoded));
                    operators_.bilinear_align_false(
                        output,
                        depth.buffer,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height),
                        request.width,
                        request.height,
                        1);
                    context_.release_external_buffer(
                        source, VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_buffer(
                        output,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_unique<VulkanGpuJob>(
                slot,
                std::move(output),
                VulkanImage{},
                VulkanImage{},
                std::move(submission),
                request.width,
                request.height,
                signal_value,
                request.source_frame_id,
                request.timestamp_ns);
        } catch (...) {
            slot->occupied.store(
                false, std::memory_order_release);
            throw;
        }
#endif
    }

    std::unique_ptr<GpuJob> submit_gpu_texture(
        const GpuTextureSubmitRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error(
            "D3D12 GPU texture submission is only available on Windows");
#else
        const GpuCapabilities capabilities = gpu_capabilities();
        const std::uint64_t required =
            DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
            DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
        if ((capabilities.flags & required) != required) {
            throw std::runtime_error(
                "complete D3D12/Vulkan texture interop is unavailable");
        }
        if (request.width == 0 || request.height == 0 ||
            request.shared_texture_handle == 0 ||
            request.wait_fence_handle == 0 ||
            (request.pixel_format != DAD_GPU_PIXEL_BGRA8 &&
             request.pixel_format != DAD_GPU_PIXEL_RGBA8) ||
            request.width >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            request.height >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(
                "invalid D3D12 GPU texture inference request");
        }
        validate_shared_texture(
            d3d12_device_.Get(),
            request.shared_texture_handle,
            request.width,
            request.height,
            request.pixel_format);
        std::shared_ptr<GpuSlot> slot =
            acquire_gpu_slot(gpu_slots_, next_gpu_slot_);
        try {
            const ImageShape shape = network_shape(
                static_cast<int>(request.width),
                static_cast<int>(request.height),
                request.input_size);
            VulkanImage output = prepare_texture_output(
                *slot,
                d3d12_device_.Get(),
                context_,
                request.width,
                request.height);
            const std::uint64_t signal_value =
                ++slot->fence_value;
            const VkFormat input_format =
                request.pixel_format == DAD_GPU_PIXEL_BGRA8
                ? VK_FORMAT_B8G8R8A8_UNORM
                : VK_FORMAT_R8G8B8A8_UNORM;
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(
                    request.shared_texture_handle),
                request.width,
                request.height,
                input_format,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(
                    request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
            VulkanSubmission submission = context_.batch_async(
                std::move(wait),
                std::move(signal),
                [&] {
                    VulkanBuffer image =
                        context_.create_device_buffer(
                            static_cast<std::uint64_t>(
                                shape.width) *
                            shape.height * 3u * sizeof(float));
                    context_.acquire_external_image(
                        input,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    preprocessor_.run_texture(
                        image,
                        input,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    EncoderOutput encoded = encoder_.forward(
                        image,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    FeatureMap depth =
                        dpt_.forward(std::move(encoded));
                    operators_.bilinear_align_false_image(
                        output,
                        depth.buffer,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height),
                        request.width,
                        request.height);
                    context_.release_external_image(
                        input,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_unique<VulkanGpuJob>(
                slot,
                VulkanBuffer{},
                std::move(input),
                std::move(output),
                std::move(submission),
                request.width,
                request.height,
                signal_value,
                request.source_frame_id,
                request.timestamp_ns);
        } catch (...) {
            slot->occupied.store(
                false, std::memory_order_release);
            throw;
        }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }

private:
    ModelFile model_;
    VulkanContext context_;
    GpuModel weights_;
    VulkanOperators operators_;
    GpuPreprocessor preprocessor_;
    DinoEncoder encoder_;
    DptHead dpt_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_device_;
    std::array<
        std::shared_ptr<GpuSlot>, kGpuSlotCount> gpu_slots_;
    std::atomic<std::uint32_t> next_gpu_slot_{0};
#endif
};

}  // namespace

std::unique_ptr<Executor> create_executor(
    const std::string& model_path,
    dad_encoder encoder,
    int vulkan_device_index) {
    if (vulkan_device_index < 0) {
        throw std::invalid_argument("vulkan_device_index must be non-negative");
    }
    return std::make_unique<VulkanExecutor>(
        model_path, encoder, vulkan_device_index);
}

GpuCapabilities probe_gpu_capabilities(
    int vulkan_device_index) {
    if (vulkan_device_index < 0) {
        throw std::invalid_argument(
            "vulkan_device_index must be non-negative");
    }
#if defined(_WIN32)
    VulkanContext context(
        static_cast<std::uint32_t>(vulkan_device_index));
    ComPtr<ID3D12Device> device =
        matching_d3d12_device(context.adapter_luid());
    return d3d12_capabilities(context, device.Get());
#else
    return {};
#endif
}

}  // namespace dad
