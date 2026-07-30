#pragma once

#include "distill_any_depth.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace dad {

class GpuSlotsExhausted final : public std::runtime_error {
public:
    GpuSlotsExhausted()
        : std::runtime_error(
              "all DAD GPU output slots are retained") {}
};

struct GpuCapabilities {
    std::uint64_t flags = 0;
    std::uint64_t adapter_luid = 0;
    std::uint32_t maximum_in_flight_jobs = 0;
};

struct GpuSubmitRequest {
    std::uintptr_t shared_resource_handle = 0;
    std::uint64_t resource_byte_size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t row_stride_bytes = 0;
    dad_gpu_pixel_format pixel_format = DAD_GPU_PIXEL_BGRA8;
    std::int32_t input_size = 0;
    std::uintptr_t wait_fence_handle = 0;
    std::uint64_t wait_fence_value = 0;
    std::uint64_t source_frame_id = 0;
    std::uint64_t timestamp_ns = 0;
};

struct GpuTextureSubmitRequest {
    std::uintptr_t shared_texture_handle = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    dad_gpu_pixel_format pixel_format = DAD_GPU_PIXEL_BGRA8;
    std::int32_t input_size = 0;
    std::uintptr_t wait_fence_handle = 0;
    std::uint64_t wait_fence_value = 0;
    std::uint64_t source_frame_id = 0;
    std::uint64_t timestamp_ns = 0;
};

enum class GpuOutputKind {
    buffer,
    texture,
};

struct GpuOutput {
    GpuOutputKind kind = GpuOutputKind::buffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t row_stride_bytes = 0;
    std::uint64_t byte_size = 0;
    std::uintptr_t shared_resource_handle = 0;
    std::uintptr_t ready_fence_handle = 0;
    std::uint64_t ready_fence_value = 0;
    std::uint64_t source_frame_id = 0;
    std::uint64_t timestamp_ns = 0;
};

class GpuJob {
public:
    virtual ~GpuJob() = default;
    virtual dad_gpu_job_state state() const = 0;
    virtual void cancel() = 0;
    virtual GpuOutput output() const = 0;
};

class Executor {
public:
    virtual ~Executor() = default;
    virtual void infer(
        const float* normalized_rgb_chw,
        int width,
        int height,
        float* depth) = 0;
    virtual void infer_resized(
        const float* normalized_rgb_chw,
        int width,
        int height,
        float* depth,
        int output_width,
        int output_height) = 0;
    virtual GpuCapabilities gpu_capabilities() const = 0;
    virtual std::unique_ptr<GpuJob> submit_gpu(
        const GpuSubmitRequest& request) = 0;
    virtual std::unique_ptr<GpuJob> submit_gpu_texture(
        const GpuTextureSubmitRequest& request) = 0;
    virtual void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const = 0;
};

std::unique_ptr<Executor> create_executor(
    const std::string& model_path,
    dad_encoder encoder,
    int vulkan_device_index);
GpuCapabilities probe_gpu_capabilities(int vulkan_device_index);

}  // namespace dad
