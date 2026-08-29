#pragma once

#include "distill_any_depth.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dad {

struct EncoderOutput {
    std::vector<VulkanBuffer> features;
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::uint32_t tokens = 0;
    std::uint32_t embedding = 0;
};

class DinoEncoder {
public:
    DinoEncoder(
        dad_encoder encoder,
        VulkanContext& context,
        GpuModel& weights,
        VulkanOperators& operators);

    void prepare(std::uint32_t width, std::uint32_t height);

    EncoderOutput forward(
        const VulkanBuffer& image,
        std::uint32_t width,
        std::uint32_t height);

private:
    void select_linear_tile(std::uint32_t rows);
    bool select_half_attention(
        const VulkanBuffer& current,
        VulkanBuffer& normalized,
        VulkanBuffer& qkv,
        VulkanBuffer& attention,
        std::uint32_t tokens);
    const VulkanBuffer& linear_weight(const std::string& name) const;
    void linear(
        VulkanBuffer& output, const VulkanBuffer& input,
        const std::string& weight_name, const std::string& bias_name,
        std::uint32_t rows, std::uint32_t input_columns,
        std::uint32_t output_columns, bool gelu = false);

    dad_encoder encoder_;
    VulkanContext& context_;
    GpuModel& weights_;
    VulkanOperators& operators_;
    std::uint32_t embedding_ = 0;
    std::uint32_t heads_ = 0;
    std::uint32_t blocks_ = 0;
    std::uint32_t capture_[4]{};
    bool linear_tile_selected_ = false;
    bool linear_block16_ = false;
    std::uint32_t linear_vector_tile_ = 8;
    bool linear_half_weight_ = false;
    std::unordered_map<std::uint32_t, bool> half_attention_by_tokens_;
};

}  // namespace dad
