#include "gpu_preprocess.h"

#include "preprocess_rgba_spv.h"
#include "preprocess_texture_spv.h"

#include <limits>
#include <stdexcept>

namespace dad {

GpuPreprocessor::GpuPreprocessor(VulkanContext& context)
    : context_(context),
      pipeline_(context.create_pipeline(
          dad_preprocess_rgba_spv,
          dad_preprocess_rgba_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          {
              VK_ACCESS_SHADER_READ_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          6 * sizeof(std::uint32_t))),
      texture_pipeline_(context.create_pipeline(
          dad_preprocess_texture_spv,
          dad_preprocess_texture_spv_size,
          {
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          {
              VK_ACCESS_SHADER_READ_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          4 * sizeof(std::uint32_t))) {
    pipeline_.set_debug_name("preprocess_rgba");
    texture_pipeline_.set_debug_name("preprocess_texture");
}

void GpuPreprocessor::run(
    VulkanBuffer& destination,
    const VulkanBuffer& source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t source_row_stride_bytes,
    std::uint32_t destination_width,
    std::uint32_t destination_height,
    GpuPixelOrder order,
    const VulkanSemaphore* wait) {
    if (source_width == 0 || source_height == 0 ||
        destination_width == 0 || destination_height == 0 ||
        source_row_stride_bytes % sizeof(std::uint32_t) != 0 ||
        source_row_stride_bytes <
            source_width * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            "invalid GPU preprocessing dimensions");
    }
    const std::uint64_t source_bytes =
        static_cast<std::uint64_t>(source_row_stride_bytes) *
        source_height;
    const std::uint64_t destination_bytes =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3 * sizeof(float);
    if (source_bytes > source.size() ||
        destination_bytes > destination.size()) {
        throw std::invalid_argument(
            "GPU preprocessing buffer is too small");
    }
    if (source_width >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
        source_height >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "GPU preprocessing dimensions are too large");
    }
    struct Parameters {
        std::uint32_t source_width;
        std::uint32_t source_height;
        std::uint32_t source_row_stride_pixels;
        std::uint32_t destination_width;
        std::uint32_t destination_height;
        std::uint32_t rgba_order;
    } parameters{
        source_width,
        source_height,
        static_cast<std::uint32_t>(
            source_row_stride_bytes / sizeof(std::uint32_t)),
        destination_width,
        destination_height,
        static_cast<std::uint32_t>(order),
    };
    context_.dispatch(
        pipeline_,
        {&source, &destination},
        &parameters,
        sizeof(parameters),
        (destination_width + 7) / 8,
        (destination_height + 7) / 8,
        1,
        wait);
}

void GpuPreprocessor::run_texture(
    VulkanBuffer& destination,
    const VulkanImage& source,
    std::uint32_t destination_width,
    std::uint32_t destination_height) {
    if (source.width() == 0 || source.height() == 0 ||
        destination_width == 0 || destination_height == 0) {
        throw std::invalid_argument(
            "invalid GPU texture preprocessing dimensions");
    }
    const std::uint64_t destination_bytes =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3 * sizeof(float);
    if (destination_bytes > destination.size()) {
        throw std::invalid_argument(
            "GPU texture preprocessing buffer is too small");
    }
    struct Parameters {
        std::uint32_t source_width;
        std::uint32_t source_height;
        std::uint32_t destination_width;
        std::uint32_t destination_height;
    } parameters{
        source.width(),
        source.height(),
        destination_width,
        destination_height,
    };
    context_.dispatch_image_to_buffer(
        texture_pipeline_,
        source,
        destination,
        &parameters,
        sizeof(parameters),
        (destination_width + 7) / 8,
        (destination_height + 7) / 8);
}

}  // namespace dad
