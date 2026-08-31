#include "inferbridge_harness.h"

#include "distill_any_depth.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(__APPLE__)
#define DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES 1
#endif

class DadGpuWorker;
struct DadGpuAdmission;

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    dad_context* context = nullptr;
    uint32_t input_size = 280u;
    std::mutex submit_mutex;
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    std::shared_ptr<DadGpuWorker> gpu_worker;
    std::shared_ptr<std::atomic<uint32_t>> gpu_admissions =
        std::make_shared<std::atomic<uint32_t>>(0u);
#endif
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
    mutable std::mutex gpu_mutex;
    dad_gpu_job* gpu_job = nullptr;
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    std::shared_ptr<DadGpuAdmission> gpu_admission;
    std::weak_ptr<DadGpuWorker> gpu_worker;
    std::atomic<uint32_t> gpu_state{IBRH_JOB_COMPLETE};
    std::atomic<bool> cancel_requested{false};
    std::string gpu_error;
    uintptr_t input_texture_handle = 0u;
    uintptr_t input_texture_identity = 0u;
    uintptr_t input_fence_handle = 0u;
    uint64_t input_fence_value = 0u;
    uint32_t input_pixel_format = DAD_GPU_PIXEL_BGRA8;
    uintptr_t output_texture_handle = 0u;
    uintptr_t output_texture_identity = 0u;
    uintptr_t output_fence_handle = 0u;
    uint64_t output_fence_value = 0u;
    int32_t input_size = 0;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t state = IBRH_JOB_COMPLETE;
    std::vector<float> depth;
    ~ibrh_job() {
        if (gpu_job != nullptr) dad_gpu_job_release(gpu_job);
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
        gpu_admission.reset();
#endif
    }
};

namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.distill-any-depth.native";
constexpr char kHarnessVersion[] = "1.2.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}
std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_uint(
    const std::string& json, const std::string& key, uint32_t& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    if (json[position] == '"') ++position;
    size_t end = position;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
    if (end == position) return false;
    uint64_t parsed = 0u;
    for (size_t index = position; index < end; ++index) {
        parsed = parsed * 10u + static_cast<uint32_t>(json[index] - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f')
            return character - 'a' + 10;
        if (character >= 'A' && character <= 'F')
            return character - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t index = 0u; index < 8u; ++index) {
        const int upper = nibble(value[index * 2u]);
        const int lower = nibble(value[index * 2u + 1u]);
        if (upper < 0 || lower < 0) return false;
        bytes[index] = static_cast<uint8_t>((upper << 4u) | lower);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
    for (int32_t index = 0; index < 32; ++index) {
        dad_gpu_capabilities capabilities{
            sizeof(capabilities), DAD_ABI_VERSION, 0u, 0u, 0u, 0u};
        const dad_status status =
            dad_probe_gpu_capabilities(index, &capabilities);
        if (status != DAD_STATUS_OK) {
            if (index == 0) return false;
            break;
        }
        if (capabilities.adapter_luid == luid) {
            device_index = index;
            return true;
        }
    }
    return false;
}

bool input_size(
    const std::string& json, uint32_t fallback, uint32_t& value) {
    value = fallback;
    uint32_t parsed = 0u;
    if (!json_uint(json, "Size", parsed)) return true;
    if (parsed == 0u || parsed > 4096u) return false;
    value = parsed;
    return true;
}

dad_encoder encoder(
    const std::string& model_path, const std::string& parameters) {
    std::string name;
    (void)json_string(parameters, "Encoder", name);
    const std::string& source = name.empty() ? model_path : name;
    if (source.find("vitl") != std::string::npos ||
        source.find("large") != std::string::npos)
        return DAD_ENCODER_VITL;
    if (source.find("vitb") != std::string::npos ||
        source.find("base") != std::string::npos)
        return DAD_ENCODER_VITB;
    return DAD_ENCODER_VITS;
}

ibrh_result status_result(dad_status status) {
    switch (status) {
        case DAD_STATUS_OK: return IBRH_OK;
        case DAD_STATUS_INVALID_ARGUMENT:
        case DAD_STATUS_BUFFER_TOO_SMALL:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case DAD_STATUS_VULKAN_UNAVAILABLE:
        case DAD_STATUS_UNSUPPORTED:
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        case DAD_STATUS_INVALID_STATE:
            return IBRH_ERROR_INVALID_STATE;
        case DAD_STATUS_CANCELLED:
            return IBRH_ERROR_CANCELLED;
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) delete job;
}

}  // namespace

#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
struct DadGpuAdmission {
    explicit DadGpuAdmission(std::shared_ptr<std::atomic<uint32_t>> value)
        : count(std::move(value)) {}
    ~DadGpuAdmission() { count->fetch_sub(1u); }
    std::shared_ptr<std::atomic<uint32_t>> count;
};

class DadGpuWorker {
public:
    explicit DadGpuWorker(dad_context* context)
        : context_(context), thread_([this] { run(); }) {}
    ~DadGpuWorker() { stop(); }

    void enqueue(ibrh_job* job) {
        retain_job(job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                release_job(job);
                throw std::runtime_error("DAD GPU worker is stopping");
            }
            queue_.push_back(job);
        }
        condition_.notify_one();
    }

    bool cancel_queued(ibrh_job* job) noexcept {
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto queued = std::find(queue_.begin(), queue_.end(), job);
            if (queued != queue_.end()) {
                queue_.erase(queued);
                removed = true;
            }
        }
        if (removed) {
            job->cancel_requested.store(true);
            job->gpu_state.store(IBRH_JOB_CANCELLED);
            release_job(job);
        }
        return removed;
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() noexcept {
        for (;;) {
            ibrh_job* job = nullptr;
            bool stop_requested = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] {
                    return stopping_ || !queue_.empty();
                });
                if (queue_.empty()) {
                    if (stopping_) return;
                    continue;
                }
                job = queue_.front();
                queue_.pop_front();
                stop_requested = stopping_;
            }
            if (stop_requested || job->cancel_requested.load()) {
                job->gpu_state.store(IBRH_JOB_CANCELLED);
                release_job(job);
                continue;
            }
            dad_gpu_job* native_job = nullptr;
#if defined(_WIN32)
            const dad_d3d12_texture_binding_request request{
                sizeof(request), DAD_ABI_VERSION,
                job->input_texture_handle, job->width, job->height,
                job->input_pixel_format, job->input_size,
                job->input_fence_handle, job->input_fence_value,
                job->output_texture_handle, job->width, job->height,
                job->output_fence_handle, job->output_fence_value,
                job->source_frame_id, job->timestamp_ns,
                job->input_texture_identity, job->output_texture_identity};
            const dad_status result = dad_submit_d3d12_texture_binding(
                context_, &request, &native_job);
#elif defined(__APPLE__)
            const dad_metal_texture_binding_request request{
                sizeof(request), DAD_ABI_VERSION,
                job->input_texture_handle, job->width, job->height,
                job->input_pixel_format, job->input_size,
                job->input_fence_handle, job->input_fence_value,
                job->output_texture_handle, job->width, job->height,
                job->output_fence_handle, job->output_fence_value,
                job->source_frame_id, job->timestamp_ns};
            const dad_status result = dad_submit_metal_texture_binding(
                context_, &request, &native_job);
#endif
            if (result == DAD_STATUS_OK && native_job != nullptr) {
                if (job->cancel_requested.load())
                    (void)dad_gpu_job_cancel(native_job);
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_job = native_job;
                }
                job->gpu_state.store(
                    job->cancel_requested.load() ?
                        IBRH_JOB_CANCELLED : IBRH_JOB_RUNNING);
            } else {
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_error = std::string(dad_status_string(result)) +
                        ": " + dad_last_error();
                }
                job->gpu_state.store(
                    job->cancel_requested.load() ?
                        IBRH_JOB_CANCELLED : IBRH_JOB_FAILED);
            }
            release_job(job);
        }
    }

    dad_context* context_ = nullptr;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<ibrh_job*> queue_;
    bool stopping_ = false;
    std::thread thread_;
};
#else
struct DadGpuAdmission {};
#endif

namespace {

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 1u;
#if defined(_WIN32)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    capabilities->output_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_D3D12_FENCE;
    capabilities->maximum_in_flight_jobs = 3u;
#elif defined(__APPLE__)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->output_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_METAL_SHARED_EVENT;
    capabilities->maximum_in_flight_jobs = 3u;
#endif
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    const std::string device = copy_string(request->requested_device_json);
    uint32_t index = 0u;
    if (json_uint(device, "index", index)) {
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "DAD requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                "DAD could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "DAD model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    inferbridge::native::Precision precision;
    try {
        precision = inferbridge::native::precision_from_parameters_json(parameters);
    } catch (const std::exception& error) {
        return fail(runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
    }
    const inferbridge::native::ScopedPrecisionRequest precision_scope(precision);
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    if (!input_size(parameters, model->input_size, model->input_size)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "DAD Size must be an integer from 1 to 4096");
    }
    uint32_t create_flags = 0u;
    if (precision == inferbridge::native::Precision::fp16)
        create_flags |= DAD_CREATE_FORCE_FP16;
    else if (precision == inferbridge::native::Precision::int8)
        create_flags |= DAD_CREATE_FORCE_INT8;
    const dad_create_options options{
        sizeof(options), DAD_ABI_VERSION, encoder(path, parameters),
        runtime->vulkan_device_index, create_flags};
    const dad_status status =
        dad_create(path.c_str(), &options, &model->context);
    if (status != DAD_STATUS_OK) {
        const std::string message =
            std::string("DAD model load failed: ") + dad_last_error();
        delete model;
        return fail(runtime, status_result(status), message);
    }
    if (runtime->adapter_luid != 0u) {
        dad_gpu_capabilities capabilities{
            sizeof(capabilities), DAD_ABI_VERSION, 0u, 0u, 0u, 0u};
        const dad_status capability_status =
            dad_get_gpu_capabilities(model->context, &capabilities);
        if (capability_status != DAD_STATUS_OK ||
            capabilities.adapter_luid != runtime->adapter_luid) {
            dad_destroy(model->context);
            delete model;
            return fail(
                runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                "DAD loaded on a GPU other than the requested device");
        }
    }
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    try {
        model->gpu_worker = std::make_shared<DadGpuWorker>(model->context);
    } catch (...) {
        dad_destroy(model->context);
        delete model;
        return IBRH_ERROR_INTERNAL;
    }
#endif
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    if (model->gpu_worker) model->gpu_worker->stop();
    model->gpu_worker.reset();
#endif
    dad_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(
    const ibrh_model* model, size_t descriptor_size,
    ibrh_model_io_descriptor* descriptor) {
    if (!model || !descriptor) return IBRH_ERROR_INVALID_ARGUMENT;
    if (descriptor_size < sizeof(*descriptor)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *descriptor = {};
    descriptor->struct_size = sizeof(*descriptor);
    descriptor->api_version = IBRH_CURRENT_API_VERSION;
    descriptor->input_count = 1u;
    descriptor->output_count = 1u;
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_get_port(
    const ibrh_model* model, uint32_t direction, uint32_t index,
    size_t descriptor_size, ibrh_port_descriptor* descriptor) {
    if (!model || !descriptor) return IBRH_ERROR_INVALID_ARGUMENT;
    if (descriptor_size < sizeof(*descriptor)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (index != 0u || (direction != IBRH_PORT_INPUT && direction != IBRH_PORT_OUTPUT))
        return IBRH_ERROR_NOT_FOUND;
    *descriptor = {};
    descriptor->struct_size = sizeof(*descriptor);
    descriptor->api_version = IBRH_CURRENT_API_VERSION;
    descriptor->index = 0u;
    descriptor->direction = direction;
    descriptor->semantic = direction == IBRH_PORT_INPUT ? IBRH_SEMANTIC_IMAGE : IBRH_SEMANTIC_DEPTH;
    descriptor->payload_type = direction == IBRH_PORT_INPUT ? IBRH_PIXEL_BGRA8 : IBRH_PIXEL_DEPTH_FLOAT32;
    descriptor->pixel_format = descriptor->payload_type;
    descriptor->accepted_pixel_format_mask = direction == IBRH_PORT_INPUT
        ? (1ull << IBRH_PIXEL_BGRA8) | (1ull << IBRH_PIXEL_RGBA8)
        : (1ull << IBRH_PIXEL_DEPTH_FLOAT32);
    descriptor->resource_kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    descriptor->depth = 1u;
    descriptor->flags = IBRH_DESCRIPTOR_DYNAMIC_WIDTH | IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_plan_outputs(
    const ibrh_model* model, size_t request_size,
    const ibrh_output_plan_request* request, uint32_t capacity,
    ibrh_port_descriptor* outputs) {
    if (!model || !request || !outputs) return IBRH_ERROR_INVALID_ARGUMENT;
    if (request_size < sizeof(*request) || request->struct_size < sizeof(*request) || capacity < 1u)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || !request->inputs ||
        !request->inputs[0].width || !request->inputs[0].height)
        return IBRH_ERROR_INVALID_ARGUMENT;
    const auto result = model_get_port(model, IBRH_PORT_OUTPUT, 0u, sizeof(outputs[0]), &outputs[0]);
    if (result != IBRH_OK) return result;
    outputs[0].width = request->inputs[0].width;
    outputs[0].height = request->inputs[0].height;
    outputs[0].flags = 0u;
    return IBRH_OK;
}

ibrh_result IBRH_CALL submit(
    ibrh_model* model, size_t request_size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if (model == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || request->inputs == nullptr ||
        request->output_count != 1u || request->outputs == nullptr)
        return fail(
            model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "DAD requires exactly one BGRA8 input");
    const ibrh_transfer_binding& input_binding = request->inputs[0];
    const ibrh_transfer_binding& output_binding = request->outputs[0];
    if (input_binding.struct_size < sizeof(input_binding) ||
        output_binding.struct_size < sizeof(output_binding))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    const ibrh_resource& input = input_binding.resource;
    const ibrh_resource& output_resource = output_binding.resource;
    uint32_t size = model->input_size;
    if (!input_size(copy_string(request->parameters_json), size, size))
        return fail(
            model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "DAD Size must be an integer from 1 to 4096");
    const bool d3d12_texture =
        input.domain == IBRH_RESOURCE_DOMAIN_D3D12 &&
        input.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED;
    const bool metal_texture =
        input.domain == IBRH_RESOURCE_DOMAIN_METAL &&
        input.native_handle_type == IBRH_NATIVE_HANDLE_METAL_TEXTURE;
    if ((d3d12_texture || metal_texture) &&
        input.kind == IBRH_RESOURCE_KIND_IMAGE_2D) {
        if ((input.pixel_format != IBRH_PIXEL_BGRA8 &&
             input.pixel_format != IBRH_PIXEL_RGBA8) ||
            input.native_handle == 0u || input.width == 0u ||
            input.height == 0u ||
            input.width > static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max()) ||
            input.height > static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max()))
            return fail(
                model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                "DAD native GPU texture input is invalid");
        const auto& wait = input_binding.synchronization;
        const auto& signal = output_binding.synchronization;
        const bool common_output =
            output_resource.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
            output_resource.pixel_format == IBRH_PIXEL_DEPTH_FLOAT32 &&
            output_resource.width == input.width &&
            output_resource.height == input.height &&
            output_resource.native_handle != 0u;
        const bool valid_d3d12 = d3d12_texture &&
            wait.kind == IBRH_SYNC_D3D12_FENCE &&
            wait.operation == IBRH_SYNC_WAIT &&
            wait.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED &&
            wait.native_handle != 0u &&
            output_resource.domain == IBRH_RESOURCE_DOMAIN_D3D12 &&
            output_resource.native_handle_type ==
                IBRH_NATIVE_HANDLE_WIN32_SHARED &&
            signal.kind == IBRH_SYNC_D3D12_FENCE &&
            signal.operation == IBRH_SYNC_SIGNAL &&
            signal.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED &&
            signal.native_handle != 0u;
        const bool no_metal_wait = wait.kind == IBRH_SYNC_NONE &&
            wait.native_handle == 0u;
        const bool metal_event_wait =
            wait.kind == IBRH_SYNC_METAL_SHARED_EVENT &&
            wait.operation == IBRH_SYNC_WAIT &&
            wait.native_handle_type == IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT &&
            wait.native_handle != 0u;
        const bool valid_metal = metal_texture &&
            (no_metal_wait || metal_event_wait) &&
            output_resource.domain == IBRH_RESOURCE_DOMAIN_METAL &&
            output_resource.native_handle_type == IBRH_NATIVE_HANDLE_METAL_TEXTURE &&
            signal.kind == IBRH_SYNC_METAL_SHARED_EVENT &&
            signal.operation == IBRH_SYNC_SIGNAL &&
            signal.native_handle_type == IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT &&
            signal.native_handle != 0u && signal.value != 0u;
        if (!common_output || (!valid_d3d12 && !valid_metal))
            return fail(
                model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                "DAD native GPU transfer bindings are invalid");
#if !defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#else
#if defined(_WIN32)
        if (!valid_d3d12) return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#elif defined(__APPLE__)
        if (!valid_metal) return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#endif
        uint32_t admitted = model->gpu_admissions->load();
        while (admitted < 3u &&
               !model->gpu_admissions->compare_exchange_weak(
                   admitted, admitted + 1u)) {}
        if (admitted >= 3u) {
            return fail(
                model->runtime, IBRH_ERROR_INVALID_STATE,
                "all DAD GPU job admissions are occupied");
        }
        auto* job = new (std::nothrow) ibrh_job();
        if (job == nullptr) {
            model->gpu_admissions->fetch_sub(1u);
            return IBRH_ERROR_INTERNAL;
        }
        try {
            job->gpu_admission = std::make_shared<DadGpuAdmission>(
                model->gpu_admissions);
        } catch (...) {
            model->gpu_admissions->fetch_sub(1u);
            delete job;
            return IBRH_ERROR_INTERNAL;
        }
        job->input_texture_handle = static_cast<uintptr_t>(input.native_handle);
        job->input_texture_identity = static_cast<uintptr_t>(
            input.auxiliary_handle != 0u
                ? input.auxiliary_handle : input.native_handle);
        job->input_fence_handle = wait.kind == IBRH_SYNC_NONE
            ? 0u : static_cast<uintptr_t>(wait.native_handle);
        job->input_fence_value = wait.value;
        job->input_pixel_format = input.pixel_format == IBRH_PIXEL_BGRA8
            ? DAD_GPU_PIXEL_BGRA8 : DAD_GPU_PIXEL_RGBA8;
        job->output_texture_handle =
            static_cast<uintptr_t>(output_resource.native_handle);
        job->output_texture_identity = static_cast<uintptr_t>(
            output_resource.auxiliary_handle != 0u
                ? output_resource.auxiliary_handle
                : output_resource.native_handle);
        job->output_fence_handle = static_cast<uintptr_t>(signal.native_handle);
        job->output_fence_value = signal.value;
        job->input_size = static_cast<int32_t>(size);
        job->source_frame_id = request->source_frame_id;
        job->timestamp_ns = request->timestamp_ns;
        job->width = input.width;
        job->height = input.height;
        job->state = IBRH_JOB_QUEUED;
        job->gpu_state.store(IBRH_JOB_QUEUED);
        try {
            std::lock_guard<std::mutex> lock(model->submit_mutex);
            if (!model->gpu_worker)
                throw std::runtime_error("DAD GPU worker is unavailable");
            job->gpu_worker = model->gpu_worker;
            model->gpu_worker->enqueue(job);
        } catch (const std::exception& error) {
            delete job;
            return fail(
                model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                error.what());
        }
        *output = job;
        return IBRH_OK;
#endif
    }
    if (input.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        input.kind != IBRH_RESOURCE_KIND_IMAGE_2D ||
        input.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        (input.pixel_format != IBRH_PIXEL_BGRA8 &&
         input.pixel_format != IBRH_PIXEL_RGBA8) ||
        input.native_handle == 0u || input.width == 0u ||
        input.height == 0u || input.width > UINT32_MAX / 4u ||
        input.row_stride_bytes < input.width * 4u ||
        input.byte_offset > input.byte_size ||
        input.byte_size - input.byte_offset <
            static_cast<uint64_t>(input.row_stride_bytes) * input.height) {
        return fail(model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY, "DAD harness requires a valid host BGRA8 image");
    }
    dad_image_shape shape{};
    dad_status status = dad_get_inferbridge_shape(static_cast<int32_t>(input.width), static_cast<int32_t>(input.height), static_cast<int32_t>(size), &shape);
    if (status != DAD_STATUS_OK) return fail(model->runtime, status_result(status), dad_last_error());
    if (output_resource.domain != IBRH_RESOURCE_DOMAIN_HOST || output_resource.kind != IBRH_RESOURCE_KIND_IMAGE_2D ||
        output_resource.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        output_resource.pixel_format != IBRH_PIXEL_DEPTH_FLOAT32 ||
        output_resource.width != input.width ||
        output_resource.height != input.height ||
        output_resource.row_stride_bytes < input.width * sizeof(float) || !output_resource.native_handle)
        return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT, "DAD host output binding is invalid");
    auto* job = new (std::nothrow) ibrh_job();
    if (job == nullptr) return IBRH_ERROR_INTERNAL;
    job->source_frame_id = request->source_frame_id;
    job->timestamp_ns = request->timestamp_ns;
    job->width = static_cast<uint32_t>(shape.width);
    job->height = static_cast<uint32_t>(shape.height);
    try {
        job->depth.resize(
            static_cast<size_t>(job->width) * job->height);
    } catch (...) {
        delete job;
        return IBRH_ERROR_INTERNAL;
    }
    const auto* source_pixels = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(input.native_handle)) + input.byte_offset;
    std::vector<uint8_t> converted;
    const uint8_t* bgra = source_pixels;
    ptrdiff_t bgra_stride = input.row_stride_bytes;
    if (input.pixel_format == IBRH_PIXEL_RGBA8) {
        converted.resize(static_cast<size_t>(input.width) * input.height * 4u);
        for (uint32_t y = 0; y < input.height; ++y)
            for (uint32_t x = 0; x < input.width; ++x) {
                const auto* source = source_pixels + static_cast<size_t>(y) * input.row_stride_bytes + x * 4u;
                auto* target = converted.data() + (static_cast<size_t>(y) * input.width + x) * 4u;
                target[0] = source[2]; target[1] = source[1]; target[2] = source[0]; target[3] = source[3];
            }
        bgra = converted.data();
        bgra_stride = static_cast<ptrdiff_t>(input.width) * 4;
    }
    {
        std::lock_guard<std::mutex> lock(model->submit_mutex);
        status = dad_inferbridge_bgra8_f32(
            model->context, bgra,
            static_cast<int32_t>(input.width),
            static_cast<int32_t>(input.height),
            bgra_stride,
            static_cast<int32_t>(size),
            job->depth.data(), job->depth.size());
    }
    if (status != DAD_STATUS_OK) { const std::string message = std::string("DAD inference failed: ") + dad_last_error(); delete job; return fail(model->runtime, status_result(status), message); }
    auto* target = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_resource.native_handle)) + output_resource.byte_offset;
    for (uint32_t y = 0; y < input.height; ++y) {
        auto* row = reinterpret_cast<float*>(
            target + static_cast<size_t>(y) * output_resource.row_stride_bytes);
        const float source_y =
            (static_cast<float>(y) + 0.5f) * job->height / input.height - 0.5f;
        const float floor_y = std::floor(source_y);
        const int y0 = std::clamp(
            static_cast<int>(floor_y), 0, static_cast<int>(job->height) - 1);
        const int y1 = std::min(y0 + 1, static_cast<int>(job->height) - 1);
        const float wy = std::clamp(source_y - floor_y, 0.0f, 1.0f);
        for (uint32_t x = 0; x < input.width; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * job->width / input.width - 0.5f;
            const float floor_x = std::floor(source_x);
            const int x0 = std::clamp(
                static_cast<int>(floor_x), 0, static_cast<int>(job->width) - 1);
            const int x1 = std::min(x0 + 1, static_cast<int>(job->width) - 1);
            const float wx = std::clamp(source_x - floor_x, 0.0f, 1.0f);
            const float top =
                job->depth[static_cast<size_t>(y0) * job->width + x0] * (1.0f - wx) +
                job->depth[static_cast<size_t>(y0) * job->width + x1] * wx;
            const float bottom =
                job->depth[static_cast<size_t>(y1) * job->width + x0] * (1.0f - wx) +
                job->depth[static_cast<size_t>(y1) * job->width + x1] * wx;
            row[x] = top * (1.0f - wy) + bottom * wy;
        }
    }
    *output = job; return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
    dad_gpu_job* gpu_job = nullptr;
    {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        gpu_job = job->gpu_job;
    }
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    if (job->gpu_admission && gpu_job == nullptr) {
        status->state = job->gpu_state.load();
    } else
#endif
    if (gpu_job != nullptr) {
        dad_gpu_job_status native_status{
            sizeof(native_status), DAD_GPU_JOB_QUEUED, 0u, 0u, 0u};
        const dad_status result =
            dad_gpu_job_poll(gpu_job, &native_status);
        if (result != DAD_STATUS_OK) return status_result(result);
        switch (native_status.state) {
            case DAD_GPU_JOB_QUEUED: status->state = IBRH_JOB_QUEUED; break;
            case DAD_GPU_JOB_RUNNING: status->state = IBRH_JOB_RUNNING; break;
            case DAD_GPU_JOB_COMPLETE: status->state = IBRH_JOB_COMPLETE; break;
            case DAD_GPU_JOB_FAILED: status->state = IBRH_JOB_FAILED; break;
            case DAD_GPU_JOB_CANCELLED: status->state = IBRH_JOB_CANCELLED; break;
            default: return IBRH_ERROR_INTERNAL;
        }
    } else {
        status->state = job->state;
    }
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    job->cancel_requested.store(true);
    if (auto worker = job->gpu_worker.lock();
        worker && worker->cancel_queued(job))
        return IBRH_OK;
#endif
    dad_gpu_job* gpu_job = nullptr;
    {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        gpu_job = job->gpu_job;
    }
    if (gpu_job != nullptr) {
        const ibrh_result result = status_result(dad_gpu_job_cancel(gpu_job));
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
        if (result == IBRH_OK) job->gpu_state.store(IBRH_JOB_CANCELLED);
#endif
        return result;
    }
#if defined(DAD_INFERBRIDGE_NATIVE_GPU_TEXTURES)
    if (job->gpu_admission) {
        job->gpu_state.store(IBRH_JOB_CANCELLED);
        return IBRH_OK;
    }
#endif
    return job->state == IBRH_JOB_COMPLETE ?
        IBRH_ERROR_INVALID_STATE : IBRH_ERROR_UNSUPPORTED_CAPABILITY;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->model_describe_io = model_describe_io;
    api->model_get_port = model_get_port;
    api->model_plan_outputs = model_plan_outputs;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
