#pragma once

#include "vulkan.h"

#include <cstdint>

namespace dad {

class VulkanOperators {
public:
    explicit VulkanOperators(VulkanContext& context);

    void linear(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu,
        bool block16 = false,
        bool half_weight = false);

    void layer_norm(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t columns,
        float epsilon);

    void add_scaled(
        VulkanBuffer& output,
        const VulkanBuffer& residual,
        const VulkanBuffer& addend,
        const VulkanBuffer& scale,
        std::uint32_t count,
        std::uint32_t columns);

    void attention_head64(
        VulkanBuffer& output,
        const VulkanBuffer& qkv,
        std::uint32_t tokens,
        std::uint32_t heads,
        VulkanBuffer* score_scratch = nullptr,
        bool half_scores = false);

    void prepare_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& image,
        const VulkanBuffer& patch_weight,
        const VulkanBuffer& patch_bias,
        const VulkanBuffer& class_token,
        const VulkanBuffer& position,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t embedding);

    void project_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& tokens,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t embedding,
        std::uint32_t output_channels,
        bool half_weight = false);

    void conv2d(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        std::uint32_t stride,
        std::uint32_t padding,
        bool has_bias,
        bool block8 = false,
        bool half_weight = false,
        bool tiled = false);

    void conv_transpose_nonoverlap(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        bool half_weight = false);

    void bilinear_align_true(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t channels);
    void bilinear_align_true_image(
        VulkanImage& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height);
    void bilinear_align_false(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t channels);
    void bilinear_align_false_image(
        VulkanImage& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height);

    void relu(VulkanBuffer& output, const VulkanBuffer& input, std::uint32_t count);

    void add(
        VulkanBuffer& output,
        const VulkanBuffer& left,
        const VulkanBuffer& right,
        std::uint32_t count);

private:
    VulkanContext& context_;
    VulkanPipeline linear_;
    VulkanPipeline linear16_;
    VulkanPipeline linear_half_;
    VulkanPipeline linear16_half_;
    VulkanPipeline linear_vec8_;
    VulkanPipeline gelu_;
    VulkanPipeline layer_norm_;
    VulkanPipeline add_scaled_;
    VulkanPipeline bmm_;
    VulkanPipeline bmm_score_half_;
    VulkanPipeline bmm_value_half_;
    VulkanPipeline softmax_lastdim_;
    VulkanPipeline softmax_lastdim_half_;
    VulkanPipeline prepare_tokens_;
    VulkanPipeline position_bicubic_;
    VulkanPipeline add_position_;
    VulkanPipeline add_;
    VulkanPipeline project_tokens_;
    VulkanPipeline project_tokens_half_;
    VulkanPipeline conv2d_;
    VulkanPipeline conv2d8_;
    VulkanPipeline conv2d_half_;
    VulkanPipeline conv2d8_half_;
    VulkanPipeline conv2d_tiled_;
    VulkanPipeline conv_transpose_nonoverlap_;
    VulkanPipeline conv_transpose_nonoverlap_half_;
    VulkanPipeline bilinear_align_false_;
    VulkanPipeline bilinear_align_false_image_;
    VulkanPipeline bilinear_align_true_;
    VulkanPipeline bilinear_align_true_image_;
    VulkanPipeline relu_;
};

}  // namespace dad
