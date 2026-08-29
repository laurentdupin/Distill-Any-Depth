#include "dpt.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace dad {
namespace {

std::uint64_t elements(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t channels) {
    return std::uint64_t(width) * height * channels;
}

const VulkanBuffer& weight(
    const GpuModel& model,
    const std::string& name) {
    return model.tensor(name).buffer;
}

}  // namespace

DptHead::DptHead(
    dad_encoder encoder,
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators)
    : context_(context),
      weights_(weights),
      operators_(operators),
      zero_bias_(context.create_device_buffer(sizeof(float))) {
    const float zero = 0.0f;
    context_.upload(zero_bias_, &zero, sizeof(zero));
    switch (encoder) {
        case DAD_ENCODER_VITS:
            embedding_ = 384;
            features_ = 64;
            project_channels_[0] = 48;
            project_channels_[1] = 96;
            project_channels_[2] = 192;
            project_channels_[3] = 384;
            break;
        case DAD_ENCODER_VITB:
            embedding_ = 768;
            features_ = 128;
            project_channels_[0] = 96;
            project_channels_[1] = 192;
            project_channels_[2] = 384;
            project_channels_[3] = 768;
            break;
        case DAD_ENCODER_VITL:
            embedding_ = 1024;
            features_ = 256;
            project_channels_[0] = 256;
            project_channels_[1] = 512;
            project_channels_[2] = 1024;
            project_channels_[3] = 1024;
            break;
        default:
            throw std::invalid_argument("unsupported DPT encoder");
    }
    select_convolution_block();
}

void DptHead::select_convolution_block() {
    convolution_block8_ = false;
    convolution_half_weight_ = weights_.select_fp16(false);
    constexpr std::uint32_t side = 16;
    const VkDeviceSize bytes =
        elements(side, side, features_) * sizeof(float);
    VulkanBuffer input = context_.create_device_buffer(bytes);
    VulkanBuffer output = context_.create_device_buffer(bytes);
    const VulkanBuffer& convolution_weight = selected_weight(
        "depth_head.scratch.refinenet1.resConfUnit2.conv1.weight");
    const VulkanBuffer& convolution_bias = weight(
        weights_,
        "depth_head.scratch.refinenet1.resConfUnit2.conv1.bias");
    const auto run = [&](bool tiled) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            for (int repetition = 0; repetition < 8; ++repetition) {
                operators_.conv2d(
                    output, input, convolution_weight, convolution_bias,
                    side, side, features_, features_, 3, 1, 1, true,
                    false, convolution_half_weight_, tiled);
            }
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    run(false);
    run(true);
    std::array<double, 3> direct{};
    std::array<double, 3> tiled{};
    for (std::size_t sample = 0; sample < direct.size(); ++sample) {
        direct[sample] = run(false);
        tiled[sample] = run(true);
    }
    std::sort(direct.begin(), direct.end());
    std::sort(tiled.begin(), tiled.end());
    convolution_tiled_ = tiled[1] < direct[1] * 0.95;
    weights_.retain_dpt_precision(convolution_half_weight_);
    convolution_block_selected_ = true;
}

const VulkanBuffer& DptHead::selected_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    return convolution_half_weight_
        ? tensor.half_buffer
        : tensor.buffer;
}

FeatureMap DptHead::conv(
    FeatureMap&& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    const std::uint32_t output_width =
        (input.width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input.height + 2 * padding - kernel) / stride + 1;
    FeatureMap output{
        context_.create_device_buffer(
            elements(output_width, output_height, output_channels) *
            sizeof(float)),
        output_width,
        output_height,
        output_channels,
    };
    const GpuTensor& convolution_weight =
        weights_.tensor(weight_name);
    operators_.conv2d(
        output.buffer,
        input.buffer,
        convolution_half_weight_
            ? convolution_weight.half_buffer
            : convolution_weight.buffer,
        has_bias ? weight(weights_, bias_name) : zero_bias_,
        input.width,
        input.height,
        input.channels,
        output_channels,
        kernel,
        stride,
        padding,
        has_bias,
        convolution_block8_,
        convolution_half_weight_,
        convolution_tiled_);
    return output;
}

FeatureMap DptHead::residual_unit(
    FeatureMap&& input,
    const std::string& prefix) {
    const std::uint32_t count =
        static_cast<std::uint32_t>(
            elements(input.width, input.height, input.channels));
    FeatureMap activated{
        context_.create_device_buffer(std::uint64_t(count) * sizeof(float)),
        input.width,
        input.height,
        input.channels,
    };
    operators_.relu(activated.buffer, input.buffer, count);
    FeatureMap first = conv(
        std::move(activated),
        prefix + ".conv1.weight",
        prefix + ".conv1.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.relu(first.buffer, first.buffer, count);
    FeatureMap second = conv(
        std::move(first),
        prefix + ".conv2.weight",
        prefix + ".conv2.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.add(
        second.buffer, second.buffer, input.buffer, count);
    return second;
}

FeatureMap DptHead::fusion(
    FeatureMap&& path,
    FeatureMap&& skip,
    const std::string& prefix,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (skip.buffer.handle() != VK_NULL_HANDLE) {
        FeatureMap processed_skip =
            residual_unit(std::move(skip), prefix + ".resConfUnit1");
        const std::uint32_t count =
            static_cast<std::uint32_t>(
                elements(path.width, path.height, path.channels));
        operators_.add(
            path.buffer, path.buffer, processed_skip.buffer, count);
    }
    path = residual_unit(
        std::move(path), prefix + ".resConfUnit2");
    FeatureMap resized{
        context_.create_device_buffer(
            elements(output_width, output_height, features_) * sizeof(float)),
        output_width,
        output_height,
        features_,
    };
    operators_.bilinear_align_true(
        resized.buffer,
        path.buffer,
        path.width,
        path.height,
        output_width,
        output_height,
        features_);
    return conv(
        std::move(resized),
        prefix + ".out_conv.weight",
        prefix + ".out_conv.bias",
        features_,
        1,
        1,
        0,
        true);
}

FeatureMap DptHead::forward(EncoderOutput&& encoded) {
    if (encoded.features.size() != 4 ||
        encoded.embedding != embedding_) {
        throw std::invalid_argument("invalid DPT encoder output");
    }
    if (!convolution_block_selected_) {
        select_convolution_block();
    }
    FeatureMap layers[4];
    const auto run_projections = [&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            context_.batch([&] {
            FeatureMap projected{
                context_.create_device_buffer(
                    elements(
                        encoded.patch_width,
                        encoded.patch_height,
                        project_channels_[index]) *
                    sizeof(float)),
                encoded.patch_width,
                encoded.patch_height,
                project_channels_[index],
            };
            const std::string prefix =
                "depth_head.projects." + std::to_string(index);
            operators_.project_tokens(
                projected.buffer,
                encoded.features[index],
                selected_weight(prefix + ".weight"),
                weight(weights_, prefix + ".bias"),
                encoded.patch_width,
                encoded.patch_height,
                embedding_,
                project_channels_[index],
                convolution_half_weight_);
            if (index < 2) {
                const std::uint32_t kernel = index == 0 ? 4 : 2;
                FeatureMap resized{
                    context_.create_device_buffer(
                        elements(
                            projected.width * kernel,
                            projected.height * kernel,
                            projected.channels) *
                        sizeof(float)),
                    projected.width * kernel,
                    projected.height * kernel,
                    projected.channels,
                };
                const std::string resize =
                    "depth_head.resize_layers." + std::to_string(index);
                operators_.conv_transpose_nonoverlap(
                    resized.buffer,
                    projected.buffer,
                    selected_weight(resize + ".weight"),
                    weight(weights_, resize + ".bias"),
                    projected.width,
                    projected.height,
                    projected.channels,
                    projected.channels,
                    kernel,
                    convolution_half_weight_);
                layers[index] = std::move(resized);
            } else if (index == 2) {
                layers[index] = std::move(projected);
            } else {
                layers[index] = conv(
                    std::move(projected),
                    "depth_head.resize_layers.3.weight",
                    "depth_head.resize_layers.3.bias",
                    project_channels_[3],
                    3,
                    2,
                    1,
                    true);
            }
            });
        }
    };
    if (encoded.tokens > 2000) {
        run_projections();
    } else {
        context_.batch(run_projections);
    }

    FeatureMap refined[4];
    FeatureMap path;
    FeatureMap depth;
    const auto run_head = [&] {
    context_.batch([&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            const std::string prefix =
                "depth_head.scratch.layer" + std::to_string(index + 1) +
                "_rn.weight";
            refined[index] = conv(
                std::move(layers[index]),
                prefix,
                "",
                features_,
                3,
                1,
                1,
                false);
        }
    });

    context_.batch([&] {
        path = fusion(
            std::move(refined[3]),
            FeatureMap{},
            "depth_head.scratch.refinenet4",
            refined[2].width,
            refined[2].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[2]),
            "depth_head.scratch.refinenet3",
            refined[1].width,
            refined[1].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[1]),
            "depth_head.scratch.refinenet2",
            refined[0].width,
            refined[0].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[0]),
            "depth_head.scratch.refinenet1",
            refined[0].width * 2,
            refined[0].height * 2);
    });

    context_.batch([&] {
        path = conv(
            std::move(path),
            "depth_head.scratch.output_conv1.weight",
            "depth_head.scratch.output_conv1.bias",
            features_ / 2,
            3,
            1,
            1,
            true);
        FeatureMap full{
            context_.create_device_buffer(
                elements(
                    encoded.patch_width * 14,
                    encoded.patch_height * 14,
                    features_ / 2) *
                sizeof(float)),
            encoded.patch_width * 14,
            encoded.patch_height * 14,
            features_ / 2,
        };
        operators_.bilinear_align_true(
            full.buffer,
            path.buffer,
            path.width,
            path.height,
            full.width,
            full.height,
            full.channels);
        full = conv(
            std::move(full),
            "depth_head.scratch.output_conv2.0.weight",
            "depth_head.scratch.output_conv2.0.bias",
            32,
            3,
            1,
            1,
            true);
        operators_.relu(
            full.buffer,
            full.buffer,
            static_cast<std::uint32_t>(
                elements(full.width, full.height, full.channels)));
        depth = conv(
            std::move(full),
            "depth_head.scratch.output_conv2.2.weight",
            "depth_head.scratch.output_conv2.2.bias",
            1,
            1,
            1,
            0,
            true);
        operators_.relu(
            depth.buffer,
            depth.buffer,
            depth.width * depth.height);
    });
    };
    if (encoded.tokens > 2000) {
        run_head();
    } else {
        context_.batch(run_head);
    }
    return depth;
}

}  // namespace dad
