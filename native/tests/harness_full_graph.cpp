#include "inferbridge_harness.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

ibrh_string_view view(const std::string& value) {
    return {value.data(), value.size()};
}

bool check(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    const char* path = std::getenv("DAD_VITS_MODEL");
    if (path == nullptr || *path == '\0') {
        std::cout << "DAD_VITS_MODEL is not set; skipping\n";
        return 77;
    }
    ibrh_api api{};
    if (!check(
            ibrh_get_api(
                IBRH_CURRENT_API_VERSION, sizeof(api), &api) == IBRH_OK,
            "API negotiation failed"))
        return 1;

    const std::string backend = "native";
    const std::string device = "{\"index\":0}";
    ibrh_runtime_create_request runtime_request{
        sizeof(runtime_request), IBRH_CURRENT_API_VERSION,
        view(backend), view(device), {}, nullptr, nullptr};
    ibrh_runtime* runtime = nullptr;
    if (!check(
            api.runtime_create(
                sizeof(runtime_request), &runtime_request, &runtime) ==
                IBRH_OK &&
                runtime != nullptr,
            "runtime creation failed"))
        return 2;

    const std::string model_path = path;
    const std::string parameters =
        "{\"Encoder\":\"vits\",\"Size\":\"140\"}";
    ibrh_model_load_request load_request{
        sizeof(load_request), IBRH_CURRENT_API_VERSION,
        view(model_path), view(parameters)};
    ibrh_model* model = nullptr;
    if (!check(
            api.model_load(
                runtime, sizeof(load_request), &load_request, &model) ==
                IBRH_OK &&
                model != nullptr,
            "model load failed")) {
        api.runtime_destroy(runtime);
        return 3;
    }

    constexpr uint32_t width = 37u;
    constexpr uint32_t height = 23u;
    std::vector<uint8_t> pixels(width * height * 4u);
    for (uint32_t y = 0u; y < height; ++y) {
        for (uint32_t x = 0u; x < width; ++x) {
            const size_t index =
                (static_cast<size_t>(y) * width + x) * 4u;
            pixels[index] = static_cast<uint8_t>((x * 7u + y) & 255u);
            pixels[index + 1u] =
                static_cast<uint8_t>((x + y * 11u) & 255u);
            pixels[index + 2u] =
                static_cast<uint8_t>((x * 3u + y * 5u) & 255u);
            pixels[index + 3u] = 255u;
        }
    }
    ibrh_resource input{};
    input.struct_size = sizeof(input);
    input.api_version = IBRH_CURRENT_API_VERSION;
    input.domain = IBRH_RESOURCE_DOMAIN_HOST;
    input.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    input.access = IBRH_RESOURCE_ACCESS_READ;
    input.pixel_format = IBRH_PIXEL_BGRA8;
    input.width = width;
    input.height = height;
    input.depth = 1u;
    input.row_stride_bytes = width * 4u;
    input.byte_size = pixels.size();
    input.native_handle_type = IBRH_NATIVE_HANDLE_HOST_POINTER;
    input.native_handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(pixels.data()));
    ibrh_submit_request submit_request{
        sizeof(submit_request), IBRH_CURRENT_API_VERSION,
        &input, 1u, nullptr, 0u, 123456u, 987654321u, {}};
    ibrh_job* job = nullptr;
    if (!check(
            api.submit(
                model, sizeof(submit_request), &submit_request, &job) ==
                IBRH_OK &&
                job != nullptr,
            "host submission failed")) {
        api.model_unload(model);
        api.runtime_destroy(runtime);
        return 4;
    }

    ibrh_job_status status{};
    if (!check(
            api.job_poll(job, sizeof(status), &status) == IBRH_OK &&
                status.state == IBRH_JOB_COMPLETE &&
                status.output_count == 1u &&
                status.source_frame_id == submit_request.source_frame_id,
            "job status/correlation failed"))
        return 5;

    ibrh_output_descriptor descriptor{};
    ibrh_output_lease* lease = nullptr;
    if (!check(
            api.output_acquire(
                job, 0u, sizeof(descriptor), &descriptor, &lease) ==
                IBRH_OK &&
                lease != nullptr,
            "output acquire failed"))
        return 6;
    api.job_release(job);
    job = nullptr;
    if (!check(
            descriptor.payload_type == IBRH_PIXEL_DEPTH_FLOAT32 &&
                descriptor.source_frame_id ==
                    submit_request.source_frame_id &&
                descriptor.timestamp_ns == submit_request.timestamp_ns &&
                descriptor.resource.domain ==
                    IBRH_RESOURCE_DOMAIN_HOST &&
                descriptor.resource.pixel_format ==
                    IBRH_PIXEL_DEPTH_FLOAT32 &&
                descriptor.resource.width != 0u &&
                descriptor.resource.height != 0u &&
                descriptor.resource.native_handle != 0u &&
                descriptor.resource.byte_size ==
                    static_cast<uint64_t>(descriptor.resource.width) *
                        descriptor.resource.height * sizeof(float),
            "output descriptor failed"))
        return 7;
    const auto* depth = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(descriptor.resource.native_handle));
    float minimum = 1.0f;
    float maximum = 0.0f;
    const uint64_t count =
        descriptor.resource.byte_size / sizeof(float);
    for (uint64_t index = 0u; index < count; ++index) {
        minimum = std::min(minimum, depth[index]);
        maximum = std::max(maximum, depth[index]);
    }
    if (!check(
            minimum == 0.0f && maximum == 1.0f,
            "output normalization failed"))
        return 8;

    api.output_release(lease);
    api.model_unload(model);
    api.runtime_destroy(runtime);
    std::cout << "InferBridge host lifecycle and lease passed\n";
    return 0;
}
