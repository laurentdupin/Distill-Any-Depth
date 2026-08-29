#pragma once

#include "model.h"
#include "vulkan.h"
#include <inferbridge/native_harness_precision.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace dad {

std::uint32_t crc32(const void* data, std::size_t bytes);

struct GpuTensor {
    VulkanBuffer buffer;
    VulkanBuffer half_buffer;
    VulkanBuffer int8_buffer;
    VulkanBuffer int8_scales;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class GpuModel {
public:
    GpuModel(const ModelFile& model, VulkanContext& context);

    const GpuTensor& tensor(std::string_view name) const;
    void retain_transformer_precision(bool half_weight);
    void retain_dpt_precision(bool half_weight);
    std::size_t tensor_count() const { return tensors_.size(); }
    bool select_fp16(bool automatic_choice) const {
        if (precision_ == inferbridge::native::Precision::int8) return false;
        return inferbridge::native::select_fp16_weights(
            precision_, automatic_choice);
    }
    bool uses_int8_weights() const {
        return precision_ == inferbridge::native::Precision::int8;
    }

private:
    VulkanContext& context_;
    std::unordered_map<std::string_view, GpuTensor> tensors_;
    inferbridge::native::Precision precision_ =
        inferbridge::native::Precision::automatic;
};

}  // namespace dad
