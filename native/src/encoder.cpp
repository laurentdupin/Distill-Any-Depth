#include "encoder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace dad {
namespace {

const VulkanBuffer& buffer(
    const GpuModel& weights,
    const std::string& name) {
    return weights.tensor(name).buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return "pretrained.blocks." + std::to_string(block) + suffix;
}

}  // namespace

DinoEncoder::DinoEncoder(
    dad_encoder encoder,
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators)
    : encoder_(encoder),
      context_(context),
      weights_(weights),
      operators_(operators) {
    switch (encoder_) {
        case DAD_ENCODER_VITS:
            embedding_ = 384;
            heads_ = 6;
            blocks_ = 12;
            capture_[0] = 2;
            capture_[1] = 5;
            capture_[2] = 8;
            capture_[3] = 11;
            break;
        case DAD_ENCODER_VITB:
            embedding_ = 768;
            heads_ = 12;
            blocks_ = 12;
            capture_[0] = 2;
            capture_[1] = 5;
            capture_[2] = 8;
            capture_[3] = 11;
            break;
        case DAD_ENCODER_VITL:
            embedding_ = 1024;
            heads_ = 16;
            blocks_ = 24;
            capture_[0] = 4;
            capture_[1] = 11;
            capture_[2] = 17;
            capture_[3] = 23;
            break;
        default:
            throw std::invalid_argument("unsupported DINOv2 encoder");
    }
    if (weights_.tensor("pretrained.cls_token").elements != embedding_ ||
        weights_.tensor("pretrained.pos_embed").elements !=
            std::uint64_t(1370) * embedding_) {
        throw std::runtime_error("encoder tensor dimensions do not match");
    }
}

void DinoEncoder::select_linear_tile(std::uint32_t rows) {
    linear_block16_ = false;
    linear_half_weight_ = false;
    const VkDeviceSize work_bytes =
        std::uint64_t(rows) * embedding_ * 4 * sizeof(float);
    VulkanBuffer left = context_.create_device_buffer(work_bytes);
    VulkanBuffer right = context_.create_device_buffer(work_bytes);
    const auto run = [&](std::uint32_t vector_tile) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            operators_.linear(
                right,
                left,
                buffer(weights_, block_name(0, ".attn.qkv.weight")),
                buffer(weights_, block_name(0, ".attn.qkv.bias")),
                rows,
                embedding_,
                embedding_ * 3,
                false,
                false,
                false,
                vector_tile);
            operators_.linear(
                left,
                right,
                buffer(weights_, block_name(0, ".attn.proj.weight")),
                buffer(weights_, block_name(0, ".attn.proj.bias")),
                rows,
                embedding_,
                embedding_,
                false,
                false,
                false,
                vector_tile);
            operators_.linear(
                right,
                left,
                buffer(weights_, block_name(0, ".mlp.fc1.weight")),
                buffer(weights_, block_name(0, ".mlp.fc1.bias")),
                rows,
                embedding_,
                embedding_ * 4,
                true,
                false,
                false,
                vector_tile);
            operators_.linear(
                left,
                right,
                buffer(weights_, block_name(0, ".mlp.fc2.weight")),
                buffer(weights_, block_name(0, ".mlp.fc2.bias")),
                rows,
                embedding_ * 4,
                embedding_,
                false,
                false,
                false,
                vector_tile);
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct Candidate {
        std::uint32_t tile;
        std::array<double, 3> samples{};
    };
    std::array<Candidate, 3> candidates{{
        {4, {}},
        {8, {}},
        {16, {}},
    }};
    for (Candidate& candidate : candidates) run(candidate.tile);
    for (std::size_t sample = 0;
         sample < candidates[0].samples.size();
         ++sample) {
        if ((sample & 1u) == 0) {
            for (Candidate& candidate : candidates)
                candidate.samples[sample] = run(candidate.tile);
        } else {
            for (auto candidate = candidates.rbegin();
                 candidate != candidates.rend();
                 ++candidate)
                candidate->samples[sample] = run(candidate->tile);
        }
    }
    Candidate* best = nullptr;
    double best_time = 0.0;
    for (Candidate& candidate : candidates) {
        std::sort(candidate.samples.begin(), candidate.samples.end());
        const double median = candidate.samples[1];
        if (best == nullptr || median < best_time) {
            best = &candidate;
            best_time = median;
        }
    }
    linear_vector_tile_ = best->tile;
    weights_.retain_transformer_precision(false);
    linear_tile_selected_ = true;
}

const VulkanBuffer& DinoEncoder::linear_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    return linear_half_weight_ ? tensor.half_buffer : tensor.buffer;
}

void DinoEncoder::prepare(
    std::uint32_t width,
    std::uint32_t height) {
    if (width == 0 || height == 0 || width % 14 != 0 ||
        height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    if (!linear_tile_selected_) {
        select_linear_tile((width / 14) * (height / 14) + 1);
    }
}

bool DinoEncoder::select_half_attention(
    const VulkanBuffer& current,
    VulkanBuffer& normalized,
    VulkanBuffer& qkv,
    VulkanBuffer& attention,
    std::uint32_t tokens) {
    context_.batch([&] {
        operators_.layer_norm(
            normalized,
            current,
            buffer(weights_, block_name(0, ".norm1.weight")),
            buffer(weights_, block_name(0, ".norm1.bias")),
            tokens,
            embedding_,
            1.0e-6f);
        operators_.linear(
            qkv,
            normalized,
            linear_weight(block_name(0, ".attn.qkv.weight")),
            buffer(weights_, block_name(0, ".attn.qkv.bias")),
            tokens,
            embedding_,
            embedding_ * 3,
            false,
            linear_block16_,
            linear_half_weight_,
            linear_vector_tile_);
    });

    VulkanBuffer fp32_scratch = context_.create_device_buffer(
        std::uint64_t(heads_) * tokens * tokens * sizeof(float));
    VulkanBuffer half_scratch = context_.create_device_buffer(
        std::uint64_t(heads_) * tokens *
        ((std::uint64_t(tokens) + 1) / 2) *
        sizeof(std::uint32_t));
    const std::uint32_t repetitions =
        tokens < 256 ? 8 : (tokens < 768 ? 4 : 2);
    const auto run = [&](bool half_scores) {
        VulkanBuffer& scratch =
            half_scores ? half_scratch : fp32_scratch;
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            for (std::uint32_t repetition = 0;
                 repetition < repetitions;
                 ++repetition) {
                operators_.attention_head64(
                    attention,
                    qkv,
                    tokens,
                    heads_,
                    &scratch,
                    half_scores);
            }
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    run(false);
    run(true);
    std::array<double, 3> fp32_samples{};
    std::array<double, 3> half_samples{};
    for (std::size_t sample = 0; sample < fp32_samples.size(); ++sample) {
        if ((sample & 1u) == 0) {
            fp32_samples[sample] = run(false);
            half_samples[sample] = run(true);
        } else {
            half_samples[sample] = run(true);
            fp32_samples[sample] = run(false);
        }
    }
    std::sort(fp32_samples.begin(), fp32_samples.end());
    std::sort(half_samples.begin(), half_samples.end());
    return half_samples[half_samples.size() / 2] <
        fp32_samples[fp32_samples.size() / 2] * 0.99;
}

EncoderOutput DinoEncoder::forward(
    const VulkanBuffer& image,
    std::uint32_t width,
    std::uint32_t height) {
    if (width == 0 || height == 0 || width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    prepare(width, height);
    const std::uint64_t token_elements =
        std::uint64_t(tokens) * embedding_;
    const VkDeviceSize token_bytes = token_elements * sizeof(float);

    VulkanBuffer current = context_.create_device_buffer(token_bytes);
    VulkanBuffer next = context_.create_device_buffer(token_bytes);
    VulkanBuffer normalized = context_.create_device_buffer(token_bytes);
    VulkanBuffer query = context_.create_device_buffer(token_bytes);
    VulkanBuffer attention = context_.create_device_buffer(token_bytes);
    VulkanBuffer qkv =
        context_.create_device_buffer(token_bytes * 3);
    VulkanBuffer hidden =
        context_.create_device_buffer(token_bytes * 4);
    context_.batch([&] {
        operators_.prepare_tokens(
            current,
            image,
            buffer(weights_, "pretrained.patch_embed.proj.weight"),
            buffer(weights_, "pretrained.patch_embed.proj.bias"),
            buffer(weights_, "pretrained.cls_token"),
            buffer(weights_, "pretrained.pos_embed"),
            width,
            height,
            embedding_);
    });
    const bool half_attention = false;
    const VkDeviceSize attention_score_bytes = half_attention
        ? std::uint64_t(heads_) * tokens *
            ((std::uint64_t(tokens) + 1) / 2) *
            sizeof(std::uint32_t)
        : std::uint64_t(heads_) * tokens * tokens * sizeof(float);
    VulkanBuffer attention_scores =
        context_.create_device_buffer(attention_score_bytes);

    EncoderOutput result;
    result.features.reserve(4);
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.tokens = tokens;
    result.embedding = embedding_;
    std::uint32_t capture_index = 0;

    const std::uint32_t blocks_per_submission =
        tokens > 2000 ? 4 : blocks_;
    for (std::uint32_t block_begin = 0;
         block_begin < blocks_;
         block_begin += blocks_per_submission) {
        const std::uint32_t block_end =
            std::min(blocks_, block_begin + blocks_per_submission);
        context_.batch([&, block_begin, block_end] {
        for (std::uint32_t block = block_begin;
             block < block_end;
             ++block) {
            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm1.weight")),
                buffer(weights_, block_name(block, ".norm1.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                qkv,
                normalized,
                linear_weight(block_name(block, ".attn.qkv.weight")),
                buffer(weights_, block_name(block, ".attn.qkv.bias")),
                tokens,
                embedding_,
                embedding_ * 3,
                false,
                linear_block16_,
                linear_half_weight_,
                linear_vector_tile_);
            operators_.attention_head64(
                attention,
                qkv,
                tokens,
                heads_,
                &attention_scores,
                half_attention);
            if (!linear_half_weight_) {
                operators_.linear_residual(
                    next,
                    attention,
                    linear_weight(
                        block_name(block, ".attn.proj.weight")),
                    buffer(
                        weights_, block_name(block, ".attn.proj.bias")),
                    current,
                    buffer(weights_, block_name(block, ".ls1.gamma")),
                    tokens,
                    embedding_,
                    embedding_,
                    linear_vector_tile_);
            } else {
                operators_.linear(
                    query,
                    attention,
                    linear_weight(
                        block_name(block, ".attn.proj.weight")),
                    buffer(
                        weights_, block_name(block, ".attn.proj.bias")),
                    tokens,
                    embedding_,
                    embedding_,
                    false,
                    linear_block16_,
                    linear_half_weight_,
                    linear_vector_tile_);
                operators_.add_scaled(
                    next,
                    current,
                    query,
                    buffer(weights_, block_name(block, ".ls1.gamma")),
                    static_cast<std::uint32_t>(token_elements),
                    embedding_);
            }
            std::swap(current, next);

            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm2.weight")),
                buffer(weights_, block_name(block, ".norm2.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                hidden,
                normalized,
                linear_weight(block_name(block, ".mlp.fc1.weight")),
                buffer(weights_, block_name(block, ".mlp.fc1.bias")),
                tokens,
                embedding_,
                embedding_ * 4,
                true,
                linear_block16_,
                linear_half_weight_,
                linear_vector_tile_);
            if (!linear_half_weight_) {
                operators_.linear_residual(
                    next,
                    hidden,
                    linear_weight(block_name(block, ".mlp.fc2.weight")),
                    buffer(weights_, block_name(block, ".mlp.fc2.bias")),
                    current,
                    buffer(weights_, block_name(block, ".ls2.gamma")),
                    tokens,
                    embedding_ * 4,
                    embedding_,
                    linear_vector_tile_);
            } else {
                operators_.linear(
                    query,
                    hidden,
                    linear_weight(block_name(block, ".mlp.fc2.weight")),
                    buffer(weights_, block_name(block, ".mlp.fc2.bias")),
                    tokens,
                    embedding_ * 4,
                    embedding_,
                    false,
                    linear_block16_,
                    linear_half_weight_,
                    linear_vector_tile_);
                operators_.add_scaled(
                    next,
                    current,
                    query,
                    buffer(weights_, block_name(block, ".ls2.gamma")),
                    static_cast<std::uint32_t>(token_elements),
                    embedding_);
            }
            std::swap(current, next);

            if (capture_index < 4 && block == capture_[capture_index]) {
                VulkanBuffer feature =
                    context_.create_device_buffer(token_bytes);
                operators_.layer_norm(
                    feature,
                    current,
                    buffer(weights_, "pretrained.norm.weight"),
                    buffer(weights_, "pretrained.norm.bias"),
                    tokens,
                    embedding_,
                    1.0e-6f);
                result.features.push_back(std::move(feature));
                ++capture_index;
            }
        }
        });
    }
    if (result.features.size() != 4) {
        throw std::runtime_error("encoder did not produce four features");
    }
    return result;
}

}  // namespace dad
