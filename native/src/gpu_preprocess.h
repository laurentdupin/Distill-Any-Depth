#pragma once

#include "vulkan.h"

#include <cstdint>

namespace dad {

enum class GpuPixelOrder : std::uint32_t {
    bgra = 0,
    rgba = 1,
};

class GpuPreprocessor {
public:
    explicit GpuPreprocessor(VulkanContext& context);

    void run(
        VulkanBuffer& destination,
        const VulkanBuffer& source,
        std::uint32_t source_width,
        std::uint32_t source_height,
        std::uint32_t source_row_stride_bytes,
        std::uint32_t destination_width,
        std::uint32_t destination_height,
        GpuPixelOrder order,
        const VulkanSemaphore* wait = nullptr);
    void run_texture(
        VulkanBuffer& destination,
        const VulkanImage& source,
        std::uint32_t destination_width,
        std::uint32_t destination_height);

private:
    VulkanContext& context_;
    VulkanPipeline pipeline_;
    VulkanPipeline texture_pipeline_;
};

}  // namespace dad
