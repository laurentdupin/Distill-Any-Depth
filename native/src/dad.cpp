#include "distill_any_depth.h"

#include "executor.h"
#include "image.h"

#include <atomic>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct dad_context {
    std::shared_ptr<dad::Executor> executor;
    dad::ImageScratch image_scratch;
    std::vector<float> image_input;
};

struct dad_gpu_job {
    std::atomic<uint32_t> references{1};
    std::shared_ptr<dad::Executor> executor;
    std::unique_ptr<dad::GpuJob> implementation;
    uint64_t source_frame_id = 0;
};

struct dad_gpu_output_lease {
    dad_gpu_job* job = nullptr;
};

namespace {

thread_local std::string last_error;

class ApiError final : public std::runtime_error {
public:
    ApiError(dad_status status, const char* message)
        : std::runtime_error(message), status_(status) {}
    dad_status status() const { return status_; }

private:
    dad_status status_;
};

dad_status fail(dad_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

bool supported_encoder(dad_encoder encoder) {
    return encoder == DAD_ENCODER_VITS ||
        encoder == DAD_ENCODER_VITB;
}

void retain_gpu_job(dad_gpu_job* job) {
    job->references.fetch_add(1, std::memory_order_relaxed);
}

void release_gpu_job(dad_gpu_job* job) {
    if (job != nullptr &&
        job->references.fetch_sub(
            1, std::memory_order_acq_rel) == 1) {
        delete job;
    }
}

template <typename Function>
dad_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DAD_STATUS_OK;
    } catch (const ApiError& error) {
        return fail(error.status(), error.what());
    } catch (const std::bad_alloc&) {
        return fail(DAD_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const dad::GpuSlotsExhausted& error) {
        return fail(DAD_STATUS_INVALID_STATE, error.what());
    } catch (const std::exception& error) {
        return fail(DAD_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(DAD_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}

}  // namespace

#if !defined(DAD_WITH_VULKAN)
namespace dad {
std::unique_ptr<Executor> create_executor(
    const std::string&,
    dad_encoder,
    int) {
    throw std::runtime_error(
        "this DLL was built without Vulkan");
}
GpuCapabilities probe_gpu_capabilities(int) {
    return {};
}
}  // namespace dad
#endif

extern "C" {

uint32_t DAD_CALL dad_abi_version(void) {
    return DAD_ABI_VERSION;
}

const char* DAD_CALL dad_version_string(void) {
    return "0.1.0";
}

const char* DAD_CALL dad_status_string(dad_status status) {
    switch (status) {
        case DAD_STATUS_OK: return "ok";
        case DAD_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DAD_STATUS_MODEL_IO: return "model I/O error";
        case DAD_STATUS_MODEL_FORMAT: return "invalid model format";
        case DAD_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case DAD_STATUS_OUT_OF_MEMORY: return "out of memory";
        case DAD_STATUS_INFERENCE_FAILED: return "inference failed";
        case DAD_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case DAD_STATUS_UNSUPPORTED: return "unsupported";
        case DAD_STATUS_INTERNAL_ERROR: return "internal error";
        case DAD_STATUS_INVALID_STATE: return "invalid state";
        case DAD_STATUS_CANCELLED: return "cancelled";
        default: return "unknown status";
    }
}

const char* DAD_CALL dad_last_error(void) {
    return last_error.c_str();
}

dad_status DAD_CALL dad_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dad_image_shape* network_shape) {
    if (network_shape == nullptr) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "network_shape is null");
    }
    return protect([&] {
        const dad::ImageShape shape =
            dad::network_shape(image_width, image_height, input_size);
        network_shape->width = shape.width;
        network_shape->height = shape.height;
    });
}

dad_status DAD_CALL dad_create(
    const char* model_path_utf8,
    const dad_create_options* options,
    dad_context** context) {
    if (context == nullptr) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "model_path_utf8 is empty");
    }
    if (options == nullptr || options->struct_size < sizeof(dad_create_options)) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
    if (options->abi_version != DAD_ABI_VERSION) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "ABI version mismatch");
    }
    if (!supported_encoder(options->encoder)) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "unsupported encoder");
    }
    return protect([&] {
        auto result = std::make_unique<dad_context>();
        result->executor = std::shared_ptr<dad::Executor>(
            dad::create_executor(
                model_path_utf8,
                options->encoder,
                options->vulkan_device_index));
        *context = result.release();
    });
}

void DAD_CALL dad_destroy(dad_context* context) {
    delete context;
}

dad_status DAD_CALL dad_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dad_gpu_capabilities* capabilities) {
    if (capabilities == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU capabilities are null");
    }
    if (capabilities->struct_size < sizeof(*capabilities) ||
        vulkan_device_index < 0) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "invalid GPU capability probe");
    }
    return protect([&] {
        const dad::GpuCapabilities available =
            dad::probe_gpu_capabilities(vulkan_device_index);
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAD_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dad_status DAD_CALL dad_get_gpu_capabilities(
    const dad_context* context,
    dad_gpu_capabilities* capabilities) {
    if (context == nullptr || capabilities == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null GPU capability argument");
    }
    if (capabilities->struct_size < sizeof(*capabilities)) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU capabilities struct is too small");
    }
    return protect([&] {
        const dad::GpuCapabilities available =
            context->executor->gpu_capabilities();
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAD_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dad_status DAD_CALL dad_get_transfer_counters(
    const dad_context* context,
    dad_transfer_counters* counters) {
    if (context == nullptr || counters == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null transfer counter argument");
    }
    if (counters->struct_size < sizeof(*counters)) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "transfer counters struct is too small");
    }
    return protect([&] {
        std::uint64_t upload = 0;
        std::uint64_t download = 0;
        context->executor->transfer_counters(upload, download);
        *counters = {};
        counters->struct_size = sizeof(*counters);
        counters->tensor_upload_bytes = upload;
        counters->tensor_download_bytes = download;
    });
}

dad_status DAD_CALL dad_submit_d3d12(
    dad_context* context,
    const dad_d3d12_submit_request* request,
    dad_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null D3D12 submit argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAD_ABI_VERSION) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 submit request");
    }
    return protect([&] {
        if (context->executor->gpu_capabilities().flags == 0) {
            throw ApiError(
                DAD_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan GPU interop is unavailable");
        }
        dad::GpuSubmitRequest native;
        native.shared_resource_handle =
            static_cast<std::uintptr_t>(
                request->shared_resource_handle);
        native.resource_byte_size = request->resource_byte_size;
        native.width = request->width;
        native.height = request->height;
        native.row_stride_bytes = request->row_stride_bytes;
        native.pixel_format =
            static_cast<dad_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dad_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu(native);
        *job = result.release();
    });
}

dad_status DAD_CALL dad_submit_d3d12_texture(
    dad_context* context,
    const dad_d3d12_texture_submit_request* request,
    dad_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null D3D12 texture submit argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAD_ABI_VERSION) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 texture submit request");
    }
    return protect([&] {
        const std::uint64_t required =
            DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
            DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
        if ((context->executor->gpu_capabilities().flags &
             required) != required) {
            throw ApiError(
                DAD_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan texture interop is unavailable");
        }
        dad::GpuTextureSubmitRequest native;
        native.shared_texture_handle =
            static_cast<std::uintptr_t>(
                request->shared_texture_handle);
        native.width = request->width;
        native.height = request->height;
        native.pixel_format =
            static_cast<dad_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dad_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu_texture(native);
        *job = result.release();
    });
}

dad_status DAD_CALL dad_gpu_job_poll(
    const dad_gpu_job* job,
    dad_gpu_job_status* status) {
    if (job == nullptr || status == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null GPU job poll argument");
    }
    if (status->struct_size < sizeof(*status)) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU job status struct is too small");
    }
    return protect([&] {
        const dad_gpu_job_state state =
            job->implementation->state();
        *status = {};
        status->struct_size = sizeof(*status);
        status->state = state;
        status->output_count =
            state == DAD_GPU_JOB_CANCELLED ? 0u : 1u;
        status->source_frame_id = job->source_frame_id;
    });
}

dad_status DAD_CALL dad_gpu_job_cancel(dad_gpu_job* job) {
    if (job == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU job is null");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAD_GPU_JOB_COMPLETE) {
            throw ApiError(
                DAD_STATUS_INVALID_STATE,
                "completed GPU job cannot be cancelled");
        }
        job->implementation->cancel();
    });
}

void DAD_CALL dad_gpu_job_release(dad_gpu_job* job) {
    release_gpu_job(job);
}

dad_status DAD_CALL dad_gpu_output_acquire(
    dad_gpu_job* job,
    uint32_t output_index,
    dad_d3d12_output_descriptor* descriptor,
    dad_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null GPU output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAD_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAD_STATUS_CANCELLED,
                "cancelled GPU job has no output");
        }
        const dad::GpuOutput output =
            job->implementation->output();
        if (output.kind != dad::GpuOutputKind::buffer) {
            throw ApiError(
                DAD_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 texture");
        }
        auto result =
            std::make_unique<dad_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAD_ABI_VERSION;
        descriptor->pixel_format =
            DAD_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->row_stride_bytes =
            output.row_stride_bytes;
        descriptor->byte_size = output.byte_size;
        descriptor->shared_resource_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

dad_status DAD_CALL dad_gpu_texture_output_acquire(
    dad_gpu_job* job,
    uint32_t output_index,
    dad_d3d12_texture_output_descriptor* descriptor,
    dad_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "null GPU texture output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU texture output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "GPU texture output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAD_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAD_STATUS_CANCELLED,
                "cancelled GPU job has no texture output");
        }
        const dad::GpuOutput output =
            job->implementation->output();
        if (output.kind != dad::GpuOutputKind::texture) {
            throw ApiError(
                DAD_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 buffer");
        }
        auto result =
            std::make_unique<dad_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAD_ABI_VERSION;
        descriptor->pixel_format =
            DAD_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->shared_texture_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

void DAD_CALL dad_gpu_output_release(
    dad_gpu_output_lease* lease) {
    if (lease == nullptr) return;
    release_gpu_job(lease->job);
    delete lease;
}

dad_status DAD_CALL dad_infer_tensor_f32(
    dad_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || normalized_rgb_chw == nullptr || output_depth == nullptr) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    if (network_width <= 0 || network_height <= 0 ||
        network_width % 14 != 0 || network_height % 14 != 0) {
        return fail(
            DAD_STATUS_INVALID_ARGUMENT,
            "network dimensions must be positive multiples of 14");
    }
    const size_t required =
        static_cast<size_t>(network_width) * static_cast<size_t>(network_height);
    if (output_count < required) {
        return fail(DAD_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        context->executor->infer(
            normalized_rgb_chw, network_width, network_height, output_depth);
    });
}

dad_status DAD_CALL dad_infer_bgr8(
    dad_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || bgr == nullptr || output_depth == nullptr) {
        return fail(DAD_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    const size_t required =
        image_width > 0 && image_height > 0
        ? static_cast<size_t>(image_width) * static_cast<size_t>(image_height)
        : 0;
    if (output_count < required) {
        return fail(DAD_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        const dad::ImageShape shape =
            dad::network_shape(image_width, image_height, input_size);
        dad::preprocess_bgr8(
            bgr,
            image_width,
            image_height,
            row_stride_bytes,
            shape,
            context->image_scratch,
            context->image_input);
        context->executor->infer_resized(
            context->image_input.data(),
            shape.width,
            shape.height,
            output_depth,
            image_width,
            image_height);
    });
}

}  // extern "C"
