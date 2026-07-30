#include "gpu_preprocess.h"
#include "image.h"
#include "vulkan.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

ComPtr<ID3D12Device> matching_device(std::uint64_t luid) {
    ComPtr<IDXGIFactory6> factory;
    check(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated =
            factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        check(enumerated, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0;
        static_assert(sizeof(candidate) == sizeof(description.AdapterLuid));
        std::memcpy(
            &candidate,
            &description.AdapterLuid,
            sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check(
            D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    throw std::runtime_error(
        "no D3D12 adapter matches the Vulkan device LUID");
}

struct SharedResource {
    ComPtr<ID3D12Resource> resource;
    HANDLE handle = nullptr;

    SharedResource() = default;
    SharedResource(const SharedResource&) = delete;
    SharedResource& operator=(const SharedResource&) = delete;
    SharedResource(SharedResource&& other) noexcept
        : resource(std::move(other.resource)),
          handle(std::exchange(other.handle, nullptr)) {}

    ~SharedResource() {
        if (handle != nullptr) CloseHandle(handle);
    }
};

SharedResource make_shared_upload(
    ID3D12Device* device,
    const std::vector<std::uint8_t>& bytes) {
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        bytes.size(),
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    SharedResource result;
    check(
        device->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_SHARED,
            &description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(shared)");
    ComPtr<ID3D12Resource> upload;
    check(
        device->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&upload)),
        "CreateCommittedResource(upload)");
    void* mapped = nullptr;
    check(upload->Map(0, nullptr, &mapped), "Map");
    std::memcpy(mapped, bytes.data(), bytes.size());
    upload->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC queue_description{
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        0,
        D3D12_COMMAND_QUEUE_FLAG_NONE,
        0,
    };
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(
        device->CreateCommandQueue(
            &queue_description,
            IID_PPV_ARGS(&queue)),
        "CreateCommandQueue");
    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&commands)),
        "CreateCommandList");
    commands->CopyBufferRegion(
        result.resource.Get(), 0, upload.Get(), 0, bytes.size());
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                result.resource.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    commands->ResourceBarrier(1, &barrier);
    check(commands->Close(), "Close(command list)");
    ID3D12CommandList* submitted[] = {commands.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> complete;
    check(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&complete)),
        "CreateFence(copy)");
    check(queue->Signal(complete.Get(), 1), "Signal(copy)");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }
    check(
        complete->SetEventOnCompletion(1, event),
        "SetEventOnCompletion(copy)");
    if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(event);
        throw std::runtime_error("copy wait failed");
    }
    CloseHandle(event);

    check(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.handle),
        "CreateSharedHandle(resource)");
    return result;
}

std::vector<float> expected(
    const std::vector<std::uint8_t>& pixels,
    std::uint32_t width,
    std::uint32_t height,
    bool rgba,
    dad::ImageShape destination) {
    std::vector<std::uint8_t> bgr(
        static_cast<std::size_t>(width) * height * 3);
    for (std::size_t index = 0; index < bgr.size() / 3; ++index) {
        const std::uint8_t* source = pixels.data() + index * 4;
        std::uint8_t* target = bgr.data() + index * 3;
        target[0] = rgba ? source[2] : source[0];
        target[1] = source[1];
        target[2] = rgba ? source[0] : source[2];
    }
    std::vector<float> result;
    dad::preprocess_bgr8(
        bgr.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<std::ptrdiff_t>(width * 3),
        destination,
        result);
    return result;
}

void run_case(
    dad::VulkanContext& context,
    dad::GpuPreprocessor& preprocessor,
    ID3D12Device* device,
    const dad::VulkanSemaphore& wait,
    bool rgba) {
    constexpr std::uint32_t width = 19;
    constexpr std::uint32_t height = 13;
    const dad::ImageShape destination{28, 28};
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width) * height * 4);
    for (std::size_t index = 0; index < pixels.size() / 4; ++index) {
        const std::uint8_t red =
            static_cast<std::uint8_t>((index * 17 + 11) % 251);
        const std::uint8_t green =
            static_cast<std::uint8_t>((index * 29 + 7) % 253);
        const std::uint8_t blue =
            static_cast<std::uint8_t>((index * 43 + 3) % 255);
        std::uint8_t* pixel = pixels.data() + index * 4;
        pixel[0] = rgba ? red : blue;
        pixel[1] = green;
        pixel[2] = rgba ? blue : red;
        pixel[3] = 255;
    }

    dad::VulkanBuffer imported;
    {
        SharedResource shared =
            make_shared_upload(device, pixels);
        imported = context.import_d3d12_buffer(
            shared.handle, pixels.size());
    }
    dad::VulkanBuffer output = context.create_device_buffer(
        static_cast<std::size_t>(destination.width) *
        destination.height * 3 * sizeof(float));
    preprocessor.run(
        output,
        imported,
        width,
        height,
        width * 4,
        destination.width,
        destination.height,
        rgba ? dad::GpuPixelOrder::rgba
             : dad::GpuPixelOrder::bgra,
        &wait);

    std::vector<float> actual(
        static_cast<std::size_t>(destination.width) *
        destination.height * 3);
    context.download(
        output, actual.data(), actual.size() * sizeof(float));
    const std::vector<float> reference =
        expected(pixels, width, height, rgba, destination);
    float maximum = 0.0f;
    double absolute = 0.0;
    double reference_absolute = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float difference =
            std::abs(actual[index] - reference[index]);
        maximum = std::max(maximum, difference);
        absolute += difference;
        reference_absolute += std::abs(reference[index]);
    }
    if (maximum > 2.0e-4f ||
        absolute / reference_absolute > 2.0e-5) {
        throw std::runtime_error(
            "GPU preprocessing accuracy gate failed");
    }
    std::cout << (rgba ? "RGBA" : "BGRA")
              << " max_abs=" << maximum
              << " relative_l1="
              << absolute / reference_absolute << '\n';
}

}  // namespace

int main() {
    dad::VulkanContext context(0);
    const auto& capabilities = context.external_capabilities();
    std::cout
        << "D3D12 resource import="
        << capabilities.d3d12_resource_import
        << " fence import=" << capabilities.d3d12_fence_import
        << " BGRA8 sampled image import="
        << capabilities.d3d12_bgra8_sampled_image_import
        << " RGBA8 sampled image import="
        << capabilities.d3d12_rgba8_sampled_image_import
        << " R32 storage image import="
        << capabilities.d3d12_r32_storage_image_import
        << '\n';
    if (!capabilities.d3d12_resource_import ||
        !capabilities.d3d12_fence_import ||
        context.adapter_luid() == 0) {
        std::cout << "D3D12 external import unavailable\n";
        return 77;
    }
    ComPtr<ID3D12Device> device =
        matching_device(context.adapter_luid());
    ComPtr<ID3D12Fence> fence;
    check(
        device->CreateFence(
            1,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&fence)),
        "CreateFence");
    HANDLE fence_handle = nullptr;
    check(
        device->CreateSharedHandle(
            fence.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &fence_handle),
        "CreateSharedHandle(fence)");
    dad::VulkanSemaphore wait =
        context.import_d3d12_fence(fence_handle, 1);
    CloseHandle(fence_handle);
    fence.Reset();

    dad::GpuPreprocessor preprocessor(context);
    run_case(context, preprocessor, device.Get(), wait, false);
    run_case(context, preprocessor, device.Get(), wait, true);
    std::cout << context.device_name()
              << " D3D12 import and GPU preprocessing passed\n";
    return 0;
}
