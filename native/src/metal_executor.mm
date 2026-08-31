#include "executor.h"
#include "image.h"
#include "model.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dad {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> dimensions) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:dimensions.size()];
    for (NSInteger value : dimensions) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t index = 0; index < tensor.rank; ++index) {
        [result addObject:@(static_cast<unsigned long long>(
            tensor.dimensions[index]))];
    }
    return result;
}

NSString* name(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

std::string hexadecimal(const std::array<std::uint8_t, 32>& bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2u, '0');
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result[index * 2u] = digits[bytes[index] >> 4u];
        result[index * 2u + 1u] = digits[bytes[index] & 0x0fu];
    }
    return result;
}

float cubic_convolution1(float x) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

float cubic_convolution2(float x) {
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

std::array<float, 4> cubic_coefficients(float t) {
    return {
        cubic_convolution2(t + 1.0f),
        cubic_convolution1(t),
        cubic_convolution1(1.0f - t),
        cubic_convolution2(2.0f - t),
    };
}

std::vector<float> position_embedding(
    const TensorView& source, int patch_height, int patch_width,
    int embedding) {
    if (source.rank != 3u || source.dimensions[0] != 1u ||
        source.dimensions[1] != 1370u ||
        source.dimensions[2] != static_cast<std::uint64_t>(embedding)) {
        throw std::runtime_error("invalid DINO position embedding");
    }
    const int patches = patch_height * patch_width;
    std::vector<float> result(
        static_cast<std::size_t>(patches + 1) * embedding);
    std::copy_n(source.data, embedding, result.data());

    const float scale_y = 37.0f / (static_cast<float>(patch_height) + 0.1f);
    const float scale_x = 37.0f / (static_cast<float>(patch_width) + 0.1f);
    for (int y = 0; y < patch_height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int base_y = static_cast<int>(std::floor(source_y));
        const auto cy = cubic_coefficients(source_y - base_y);
        for (int x = 0; x < patch_width; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int base_x = static_cast<int>(std::floor(source_x));
            const auto cx = cubic_coefficients(source_x - base_x);
            float* destination = result.data() +
                static_cast<std::size_t>(y * patch_width + x + 1) * embedding;
            for (int channel = 0; channel < embedding; ++channel) {
                float value = 0.0f;
                for (int ky = 0; ky < 4; ++ky) {
                    const int sy = std::clamp(base_y - 1 + ky, 0, 36);
                    float row = 0.0f;
                    for (int kx = 0; kx < 4; ++kx) {
                        const int sx = std::clamp(base_x - 1 + kx, 0, 36);
                        const std::size_t offset =
                            (static_cast<std::size_t>(1 + sy * 37 + sx) *
                             embedding) + channel;
                        row += source.data[offset] * cx[kx];
                    }
                    value += row * cy[ky];
                }
                destination[channel] = value;
            }
        }
    }
    return result;
}

struct EncoderConfiguration {
    int embedding = 0;
    int heads = 0;
    int blocks = 0;
    int features = 0;
    std::array<int, 4> captures{};
    std::array<int, 4> project_channels{};
};

EncoderConfiguration configuration(dad_encoder encoder) {
    switch (encoder) {
        case DAD_ENCODER_VITS:
            return {384, 6, 12, 64, {2, 5, 8, 11}, {48, 96, 192, 384}};
        case DAD_ENCODER_VITB:
            return {768, 12, 12, 128, {2, 5, 8, 11}, {96, 192, 384, 768}};
        case DAD_ENCODER_VITL:
            return {1024, 16, 24, 256, {4, 11, 17, 23},
                    {256, 512, 1024, 1024}};
        default:
            throw std::invalid_argument("unsupported DINOv2 encoder");
    }
}

class MetalGraphBuilder {
public:
    MetalGraphBuilder(
        const ModelFile& model, dad_encoder encoder,
        int width, int height, int output_width, int output_height,
        bool fp16)
        : model_(model), config_(configuration(encoder)),
          width_(width), height_(height),
          patch_width_(width / 14), patch_height_(height / 14),
          tokens_(patch_width_ * patch_height_ + 1),
          output_width_(output_width), output_height_(output_height),
          fp16_(fp16),
          graph_([MPSGraph new]) {}

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }

    void build(float metric_max_depth) {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
                                         dataType:MPSDataTypeFloat32
                                             name:@"normalized_rgb_chw"];
        MPSGraphTensor* graph_input = fp16_
            ? [graph_ castTensor:input_ toType:MPSDataTypeFloat16 name:nil]
            : input_;
        MPSGraphTensor* current = conv(
            graph_input, "pretrained.patch_embed.proj", 14, 0);
        current = [graph_ reshapeTensor:current
                              withShape:shape({1, config_.embedding,
                                               patch_height_ * patch_width_})
                                   name:nil];
        current = [graph_ transposeTensor:current
                                dimension:1
                            withDimension:2
                                     name:nil];
        MPSGraphTensor* class_token = constant("pretrained.cls_token");
        current = [graph_ concatTensors:@[class_token, current]
                               dimension:1
                                    name:nil];
        const std::vector<float> positions = position_embedding(
            model_.tensor("pretrained.pos_embed"), patch_height_,
            patch_width_, config_.embedding);
        MPSGraphTensor* position = owned_constant(
            positions, shape({1, tokens_, config_.embedding}));
        current = add(current, position);

        std::array<MPSGraphTensor*, 4> captured{};
        int capture_index = 0;
        for (int block = 0; block < config_.blocks; ++block) {
            const std::string prefix =
                "pretrained.blocks." + std::to_string(block);
            MPSGraphTensor* normalized = layer_norm(
                current, prefix + ".norm1", 1.0e-6f);
            MPSGraphTensor* qkv = linear(
                normalized, prefix + ".attn.qkv");
            qkv = [graph_ reshapeTensor:qkv
                              withShape:shape({1, tokens_, 3, config_.heads, 64})
                                   name:nil];
            qkv = [graph_ transposeTensor:qkv
                              permutation:@[@2, @0, @3, @1, @4]
                                     name:nil];
            MPSGraphTensor* query = slice(qkv, 0, 0, 1);
            MPSGraphTensor* key = slice(qkv, 0, 1, 1);
            MPSGraphTensor* value = slice(qkv, 0, 2, 1);
            query = [graph_ reshapeTensor:query
                               withShape:shape({1, config_.heads, tokens_, 64})
                                    name:nil];
            key = [graph_ reshapeTensor:key
                             withShape:shape({1, config_.heads, tokens_, 64})
                                  name:nil];
            value = [graph_ reshapeTensor:value
                               withShape:shape({1, config_.heads, tokens_, 64})
                                    name:nil];
            query = multiply(query, scalar(0.125f));
            key = [graph_ transposeTensor:key
                                dimension:2
                            withDimension:3
                                     name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:query
                secondaryTensor:key
                name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attention = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:value
                name:nil];
            attention = [graph_ transposeTensor:attention
                                      dimension:1
                                  withDimension:2
                                           name:nil];
            attention = [graph_ reshapeTensor:attention
                                    withShape:shape({1, tokens_, config_.embedding})
                                         name:nil];
            MPSGraphTensor* projected = linear(
                attention, prefix + ".attn.proj");
            projected = multiply(
                projected, constant(prefix + ".ls1.gamma"));
            current = add(current, projected);

            normalized = layer_norm(
                current, prefix + ".norm2", 1.0e-6f);
            MPSGraphTensor* hidden = linear(
                normalized, prefix + ".mlp.fc1");
            hidden = gelu(hidden);
            hidden = linear(hidden, prefix + ".mlp.fc2");
            hidden = multiply(hidden, constant(prefix + ".ls2.gamma"));
            current = add(current, hidden);

            if (capture_index < 4 &&
                block == config_.captures[capture_index]) {
                captured[capture_index] = layer_norm(
                    current, "pretrained.norm", 1.0e-6f);
                ++capture_index;
            }
        }
        if (capture_index != 4) {
            throw std::runtime_error("Metal encoder did not capture four features");
        }
        output_ = build_dpt(captured, metric_max_depth);
    }

    void build_presentation(float metric_max_depth) {
        build(metric_max_depth);
        NSArray<NSNumber*>* axes = @[@0, @1, @2, @3];
        MPSGraphTensor* minimum = [graph_
            reductionMinimumWithTensor:output_ axes:axes name:nil];
        MPSGraphTensor* span = nil;
        if (metric_max_depth > 0.0f) {
            span = [graph_ subtractionWithPrimaryTensor:scalar(25.0f)
                                        secondaryTensor:minimum
                                                   name:nil];
        } else {
            MPSGraphTensor* maximum = [graph_
                reductionMaximumWithTensor:output_ axes:axes name:nil];
            span = [graph_ subtractionWithPrimaryTensor:maximum
                                        secondaryTensor:minimum
                                                   name:nil];
        }
        span = [graph_ maximumWithPrimaryTensor:span
                                secondaryTensor:scalar(1.0e-12f)
                                           name:nil];
        MPSGraphTensor* centered = [graph_
            subtractionWithPrimaryTensor:output_
                          secondaryTensor:minimum
                                     name:nil];
        output_ = [graph_ divisionWithPrimaryTensor:centered
                                    secondaryTensor:span
                                               name:nil];
        output_ = [graph_ clampWithTensor:output_
                           minValueTensor:scalar(0.0f)
                           maxValueTensor:scalar(1.0f)
                                     name:nil];
        // Match the Vulkan path: normalize at network resolution, then let a
        // small backend kernel perform the full-resolution aligned resize.
        // Keeping presentation dimensions out of MPSGraph avoids recompiling
        // the model for every source size and cuts first-use compilation time.
        cast_output_float32();
    }

    void cast_output_float32() {
        if (fp16_ && output_.dataType != MPSDataTypeFloat32) {
            output_ = [graph_ castTensor:output_
                                  toType:MPSDataTypeFloat32
                                    name:@"depth_float32"];
        }
    }

private:
    MPSGraphTensor* constant(const std::string& tensor_name) {
        const TensorView& tensor = model_.tensor(tensor_name);
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<float*>(tensor.data)
            length:static_cast<NSUInteger>(tensor.elements * sizeof(float))
            freeWhenDone:NO];
        MPSGraphTensor* value = [graph_ constantWithData:data
                                                   shape:shape(tensor)
                                                dataType:MPSDataTypeFloat32];
        return fp16_
            ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
            : value;
    }

    MPSGraphTensor* owned_constant(
        const std::vector<float>& values, MPSShape* dimensions) {
        NSData* data = [NSData dataWithBytes:values.data()
                                      length:values.size() * sizeof(float)];
        MPSGraphTensor* value = [graph_ constantWithData:data
                                                   shape:dimensions
                                                dataType:MPSDataTypeFloat32];
        return fp16_
            ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
            : value;
    }

    MPSGraphTensor* scalar(float value) {
        MPSGraphTensor* result = [graph_ constantWithScalar:value
                                                   dataType:MPSDataTypeFloat32];
        return fp16_
            ? [graph_ castTensor:result toType:MPSDataTypeFloat16 name:nil]
            : result;
    }

    MPSGraphTensor* add(MPSGraphTensor* left, MPSGraphTensor* right) {
        return [graph_ additionWithPrimaryTensor:left
                                secondaryTensor:right
                                           name:nil];
    }

    MPSGraphTensor* multiply(MPSGraphTensor* left, MPSGraphTensor* right) {
        return [graph_ multiplicationWithPrimaryTensor:left
                                      secondaryTensor:right
                                                 name:nil];
    }

    MPSGraphTensor* slice(
        MPSGraphTensor* tensor, NSUInteger dimension,
        NSInteger start, NSInteger length) {
        return [graph_ sliceTensor:tensor dimension:dimension
                             start:start length:length name:nil];
    }

    MPSGraphTensor* linear(
        MPSGraphTensor* tensor, const std::string& prefix) {
        MPSGraphTensor* weight = constant(prefix + ".weight");
        weight = [graph_ transposeTensor:weight
                               dimension:0
                           withDimension:1
                                    name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:tensor
            secondaryTensor:weight
            name:name(prefix)];
        if (model_.contains(prefix + ".bias")) {
            result = add(result, constant(prefix + ".bias"));
        }
        return result;
    }

    MPSGraphTensor* layer_norm(
        MPSGraphTensor* tensor, const std::string& prefix, float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:tensor axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:tensor
                                                meanTensor:mean
                                                      axes:axes
                                                      name:nil];
        return [graph_ normalizationWithTensor:tensor
                                    meanTensor:mean
                                varianceTensor:variance
                                   gammaTensor:constant(prefix + ".weight")
                                    betaTensor:constant(prefix + ".bias")
                                       epsilon:epsilon
                                          name:name(prefix)];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* tensor) {
        MPSGraphTensor* scaled = multiply(
            tensor, scalar(static_cast<float>(M_SQRT1_2)));
        MPSGraphTensor* error = [graph_ erfWithTensor:scaled name:nil];
        return multiply(
            multiply(tensor, scalar(0.5f)), add(error, scalar(1.0f)));
    }

    MPSGraphConvolution2DOpDescriptor* convolution_descriptor(
        int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride
            strideInY:stride
            dilationRateInX:1
            dilationRateInY:1
            groups:1
            paddingLeft:padding
            paddingRight:padding
            paddingTop:padding
            paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }

    MPSGraphTensor* conv(
        MPSGraphTensor* tensor, const std::string& prefix,
        int stride, int padding, bool bias = true) {
        MPSGraphTensor* result = [graph_
            convolution2DWithSourceTensor:tensor
            weightsTensor:constant(prefix + ".weight")
            descriptor:convolution_descriptor(stride, padding)
            name:name(prefix)];
        if (bias && model_.contains(prefix + ".bias")) {
            const TensorView& bias_tensor = model_.tensor(prefix + ".bias");
            MPSGraphTensor* bias_value = [graph_ reshapeTensor:
                constant(prefix + ".bias")
                withShape:shape({1, static_cast<NSInteger>(bias_tensor.elements), 1, 1})
                name:nil];
            result = add(result, bias_value);
        }
        return result;
    }

    MPSGraphTensor* conv_transpose(
        MPSGraphTensor* tensor, const std::string& prefix,
        int channels, int source_height, int source_width, int stride) {
        MPSGraphTensor* result = [graph_
            convolutionTranspose2DWithSourceTensor:tensor
            weightsTensor:constant(prefix + ".weight")
            outputShape:shape({1, channels, source_height * stride,
                               source_width * stride})
            descriptor:convolution_descriptor(stride, 0)
            name:name(prefix)];
        MPSGraphTensor* bias = [graph_ reshapeTensor:
            constant(prefix + ".bias")
            withShape:shape({1, channels, 1, 1})
            name:nil];
        return add(result, bias);
    }

    MPSGraphTensor* resize(
        MPSGraphTensor* tensor, int height, int width,
        MPSGraphResizeMode mode, bool align_corners) {
        return [graph_ resizeTensor:tensor
                              size:shape({height, width})
                              mode:mode
                      centerResult:align_corners ? NO : YES
                      alignCorners:align_corners ? YES : NO
                            layout:MPSGraphTensorNamedDataLayoutNCHW
                              name:nil];
    }

    MPSGraphTensor* residual_unit(
        MPSGraphTensor* tensor, const std::string& prefix) {
        MPSGraphTensor* result = [graph_ reLUWithTensor:tensor name:nil];
        result = conv(result, prefix + ".conv1", 1, 1);
        result = [graph_ reLUWithTensor:result name:nil];
        result = conv(result, prefix + ".conv2", 1, 1);
        return add(tensor, result);
    }

    MPSGraphTensor* fusion(
        MPSGraphTensor* path, MPSGraphTensor* skip,
        const std::string& prefix, int target_height, int target_width) {
        if (skip != nil) {
            path = add(path, residual_unit(
                skip, prefix + ".resConfUnit1"));
        }
        path = residual_unit(path, prefix + ".resConfUnit2");
        path = resize(
            path, target_height, target_width,
            MPSGraphResizeBilinear, true);
        return conv(path, prefix + ".out_conv", 1, 0);
    }

    MPSGraphTensor* build_dpt(
        const std::array<MPSGraphTensor*, 4>& features,
        float metric_max_depth) {
        std::array<MPSGraphTensor*, 4> layers{};
        std::array<int, 4> layer_heights{
            patch_height_ * 4, patch_height_ * 2,
            patch_height_, (patch_height_ + 1) / 2};
        std::array<int, 4> layer_widths{
            patch_width_ * 4, patch_width_ * 2,
            patch_width_, (patch_width_ + 1) / 2};
        for (int index = 0; index < 4; ++index) {
            MPSGraphTensor* patches = slice(
                features[index], 1, 1, patch_width_ * patch_height_);
            patches = [graph_ transposeTensor:patches
                                    dimension:1
                                withDimension:2
                                         name:nil];
            patches = [graph_ reshapeTensor:patches
                                  withShape:shape({1, config_.embedding,
                                                   patch_height_, patch_width_})
                                       name:nil];
            const std::string project =
                "depth_head.projects." + std::to_string(index);
            patches = conv(patches, project, 1, 0);
            if (index < 2) {
                const int scale = index == 0 ? 4 : 2;
                patches = conv_transpose(
                    patches,
                    "depth_head.resize_layers." + std::to_string(index),
                    config_.project_channels[index],
                    patch_height_, patch_width_, scale);
            } else if (index == 3) {
                patches = conv(
                    patches, "depth_head.resize_layers.3", 2, 1);
            }
            layers[index] = conv(
                patches,
                "depth_head.scratch.layer" + std::to_string(index + 1) +
                    "_rn",
                1, 1, false);
        }

        MPSGraphTensor* path = fusion(
            layers[3], nil, "depth_head.scratch.refinenet4",
            layer_heights[2], layer_widths[2]);
        path = fusion(
            path, layers[2], "depth_head.scratch.refinenet3",
            layer_heights[1], layer_widths[1]);
        path = fusion(
            path, layers[1], "depth_head.scratch.refinenet2",
            layer_heights[0], layer_widths[0]);
        path = fusion(
            path, layers[0], "depth_head.scratch.refinenet1",
            layer_heights[0] * 2, layer_widths[0] * 2);
        path = conv(
            path, "depth_head.scratch.output_conv1", 1, 1);
        path = resize(
            path, height_, width_, MPSGraphResizeBilinear, true);
        path = conv(
            path, "depth_head.scratch.output_conv2.0", 1, 1);
        path = [graph_ reLUWithTensor:path name:nil];
        path = conv(
            path, "depth_head.scratch.output_conv2.2", 1, 0);
        if (metric_max_depth > 0.0f) {
            path = [graph_ sigmoidWithTensor:path name:nil];
            path = multiply(path, scalar(metric_max_depth));
        } else {
            path = [graph_ reLUWithTensor:path name:nil];
        }
        if (output_height_ != height_ || output_width_ != width_) {
            path = resize(
                path, output_height_, output_width_,
                MPSGraphResizeBilinear, true);
        }
        return path;
    }

    const ModelFile& model_;
    EncoderConfiguration config_;
    int width_;
    int height_;
    int patch_width_;
    int patch_height_;
    int tokens_;
    int output_width_;
    int output_height_;
    bool fp16_ = false;
    MPSGraph* graph_;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
};

struct PlanKey {
    int width;
    int height;
    int output_width;
    int output_height;
    bool presentation = false;

    bool operator==(const PlanKey& other) const {
        return width == other.width && height == other.height &&
            output_width == other.output_width &&
            output_height == other.output_height &&
            presentation == other.presentation;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& value) const {
        std::size_t result = static_cast<std::size_t>(value.width);
        result = result * 1315423911u + static_cast<std::size_t>(value.height);
        result = result * 1315423911u +
            static_cast<std::size_t>(value.output_width);
        result = result * 1315423911u +
            static_cast<std::size_t>(value.output_height);
        return result * 1315423911u +
            static_cast<std::size_t>(value.presentation);
    }
};

struct MetalPlan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphTensor* output = nil;
    MPSGraphExecutable* executable = nil;
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
    bool presentation = false;
};

constexpr char kMetalTransferKernels[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct PreprocessParameters {
    uint source_width;
    uint source_height;
    uint destination_width;
    uint destination_height;
};

float cubic1(float value) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * value - (a + 3.0f)) *
        value * value + 1.0f;
}

float cubic2(float value) {
    constexpr float a = -0.75f;
    return ((a * value - 5.0f * a) * value + 8.0f * a) *
        value - 4.0f * a;
}

float coefficient(int tap, float fraction) {
    if (tap == 0) return cubic2(fraction + 1.0f);
    if (tap == 1) return cubic1(fraction);
    if (tap == 2) return cubic1(1.0f - fraction);
    return cubic2(2.0f - fraction);
}

kernel void preprocess_texture(
    texture2d<float, access::read> source [[texture(0)]],
    device float* destination [[buffer(0)]],
    constant PreprocessParameters& parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
    if (position.x >= parameters.destination_width ||
        position.y >= parameters.destination_height) return;
    const float source_x =
        (float(position.x) + 0.5f) * float(parameters.source_width) /
            float(parameters.destination_width) - 0.5f;
    const float source_y =
        (float(position.y) + 0.5f) * float(parameters.source_height) /
            float(parameters.destination_height) - 0.5f;
    const int base_x = int(floor(source_x));
    const int base_y = int(floor(source_y));
    const float fraction_x = source_x - float(base_x);
    const float fraction_y = source_y - float(base_y);
    constexpr float means[3] = {0.485f, 0.456f, 0.406f};
    constexpr float deviations[3] = {0.229f, 0.224f, 0.225f};
    const uint plane =
        parameters.destination_width * parameters.destination_height;
    const uint index =
        position.y * parameters.destination_width + position.x;
    for (uint channel = 0; channel < 3; ++channel) {
        float resized = 0.0f;
        for (int tap_y = 0; tap_y < 4; ++tap_y) {
            const int y = clamp(
                base_y - 1 + tap_y, 0,
                int(parameters.source_height) - 1);
            float row = 0.0f;
            for (int tap_x = 0; tap_x < 4; ++tap_x) {
                const int x = clamp(
                    base_x - 1 + tap_x, 0,
                    int(parameters.source_width) - 1);
                row += source.read(uint2(x, y))[channel] *
                    coefficient(tap_x, fraction_x);
            }
            resized += row * coefficient(tap_y, fraction_y);
        }
        destination[channel * plane + index] =
            (resized - means[channel]) / deviations[channel];
    }
}

struct PresentationParameters {
    uint source_width;
    uint source_height;
    uint destination_width;
    uint destination_height;
};

kernel void resize_depth_to_texture(
    device const float* source [[buffer(0)]],
    texture2d<float, access::write> destination [[texture(0)]],
    constant PresentationParameters& parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
    if (position.x >= parameters.destination_width ||
        position.y >= parameters.destination_height) return;
    const float source_x = parameters.destination_width > 1
        ? float(position.x) * float(parameters.source_width - 1) /
            float(parameters.destination_width - 1)
        : 0.0f;
    const float source_y = parameters.destination_height > 1
        ? float(position.y) * float(parameters.source_height - 1) /
            float(parameters.destination_height - 1)
        : 0.0f;
    const uint x0 = uint(floor(source_x));
    const uint y0 = uint(floor(source_y));
    const uint x1 = min(x0 + 1, parameters.source_width - 1);
    const uint y1 = min(y0 + 1, parameters.source_height - 1);
    const float x_fraction = source_x - float(x0);
    const float y_fraction = source_y - float(y0);
    const float top = mix(
        source[y0 * parameters.source_width + x0],
        source[y0 * parameters.source_width + x1], x_fraction);
    const float bottom = mix(
        source[y1 * parameters.source_width + x0],
        source[y1 * parameters.source_width + x1], x_fraction);
    destination.write(float4(mix(top, bottom, y_fraction)), position);
}
)METAL";

void dispatch_2d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger width, NSUInteger height) {
    const NSUInteger thread_width = pipeline.threadExecutionWidth;
    const NSUInteger thread_height = std::max<NSUInteger>(
        1u, pipeline.maxTotalThreadsPerThreadgroup / thread_width);
    [encoder dispatchThreads:MTLSizeMake(width, height, 1u)
        threadsPerThreadgroup:MTLSizeMake(thread_width, thread_height, 1u)];
}

class MetalGpuJob final : public GpuJob {
public:
    MetalGpuJob(
        id<MTLTexture> input_texture,
        id<MTLSharedEvent> input_event,
        id<MTLTexture> output_texture,
        id<MTLSharedEvent> output_event,
        id<MTLBuffer> normalized_input,
        id<MTLBuffer> presentation_output,
        MPSGraphTensorData* input_data,
        MPSGraphTensorData* output_data,
        id<MTLCommandBuffer> completion,
        std::uint64_t signal_value)
        : input_texture_(input_texture), input_event_(input_event),
          output_texture_(output_texture), output_event_(output_event),
          normalized_input_(normalized_input),
          presentation_output_(presentation_output),
          input_data_(input_data), output_data_(output_data),
          completion_(completion), signal_value_(signal_value) {}

    ~MetalGpuJob() override {
        if (completion_ != nil &&
            completion_.status < MTLCommandBufferStatusCompleted)
            [completion_ waitUntilCompleted];
    }

    dad_gpu_job_state state() const override {
        if (cancelled_.load(std::memory_order_acquire))
            return DAD_GPU_JOB_CANCELLED;
        if (completion_ != nil &&
            completion_.status == MTLCommandBufferStatusError) {
            const char* description =
                completion_.error.localizedDescription.UTF8String;
            throw std::runtime_error(description == nullptr
                ? "Metal inference command failed"
                : std::string("Metal inference command failed: ") + description);
        }
        return output_event_.signaledValue >= signal_value_
            ? DAD_GPU_JOB_COMPLETE : DAD_GPU_JOB_RUNNING;
    }

    void cancel() override {
        cancelled_.store(true, std::memory_order_release);
    }

    GpuOutput output() const override {
        throw std::runtime_error(
            "InferBridge-owned Metal output has no standalone output lease");
    }

private:
    id<MTLTexture> input_texture_ = nil;
    id<MTLSharedEvent> input_event_ = nil;
    id<MTLTexture> output_texture_ = nil;
    id<MTLSharedEvent> output_event_ = nil;
    id<MTLBuffer> normalized_input_ = nil;
    id<MTLBuffer> presentation_output_ = nil;
    MPSGraphTensorData* input_data_ = nil;
    MPSGraphTensorData* output_data_ = nil;
    id<MTLCommandBuffer> completion_ = nil;
    std::uint64_t signal_value_ = 0u;
    std::atomic<bool> cancelled_{false};
};

class MetalExecutor final : public Executor {
public:
    MetalExecutor(
        const std::string& model_path, dad_encoder encoder,
        std::uint32_t flags, const std::string& cache_path)
        : model_(model_path, encoder), encoder_(encoder),
          metric_max_depth_(0.0f),
          cache_path_(cache_path) {
        if ((flags & DAD_CREATE_FORCE_INT8) != 0u) {
            throw std::invalid_argument(
                "the Metal executor does not support INT8 yet");
        }
        fp16_ = (flags & DAD_CREATE_FORCE_FP16) != 0u;
        configure_device(MTLCreateSystemDefaultDevice());
    }

    void prepare(int image_width, int image_height, int input_size) override {
        const ImageShape network =
            network_shape(image_width, image_height, input_size);
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            (void)get_presentation_plan(network.width, network.height);
        }
    }

    void infer(
        const float* input, int width, int height, float* depth) override {
        infer_resized(input, width, height, depth, width, height);
    }

    void infer_resized(
        const float* input, int width, int height, float* depth,
        int output_width, int output_height) override {
        if (input == nullptr || depth == nullptr || width <= 0 || height <= 0 ||
            output_width <= 0 || output_height <= 0 ||
            width % 14 != 0 || height % 14 != 0) {
            throw std::invalid_argument("invalid Metal inference dimensions");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            MetalPlan& plan = get_plan(
                width, height, output_width, output_height);
            const NSUInteger input_bytes = static_cast<NSUInteger>(
                std::uint64_t(width) * height * 3u * sizeof(float));
            id<MTLBuffer> input_buffer = [device_
                newBufferWithBytes:input
                length:input_bytes
                options:MTLResourceStorageModeShared];
            if (input_buffer == nil) {
                throw std::bad_alloc();
            }
            MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:input_buffer
                shape:shape({1, 3, height, width})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_
                inputsArray:@[input_data]
                resultsArray:nil
                executionDescriptor:execution];
            if (results.count != 1u) {
                throw std::runtime_error("Metal graph returned no depth tensor");
            }
            [results[0].mpsndarray readBytes:depth strideBytes:nil];
            upload_bytes_.fetch_add(input_bytes, std::memory_order_relaxed);
            download_bytes_.fetch_add(
                std::uint64_t(output_width) * output_height * sizeof(float),
                std::memory_order_relaxed);
        }
    }

    GpuCapabilities gpu_capabilities() const override {
        GpuCapabilities result;
        result.flags =
            DAD_GPU_CAP_METAL_TEXTURE_INPUT |
            DAD_GPU_CAP_METAL_SHARED_EVENT_WAIT |
            DAD_GPU_CAP_METAL_TEXTURE_OUTPUT |
            DAD_GPU_CAP_METAL_SHARED_EVENT_SIGNAL |
            DAD_GPU_CAP_ASYNC_SUBMIT |
            DAD_GPU_CAP_CANCELLATION |
            DAD_GPU_CAP_NO_HOST_PIXEL_STAGING |
            DAD_GPU_CAP_NO_HOST_DEPTH_STAGING;
        result.adapter_luid = device_.registryID;
        result.maximum_in_flight_jobs = 3u;
        return result;
    }

    std::unique_ptr<GpuJob> submit_gpu(
        const GpuSubmitRequest&) override {
        throw std::runtime_error(
            "Metal external-buffer submission is not implemented yet");
    }

    std::unique_ptr<GpuJob> submit_gpu_texture(
        const GpuTextureSubmitRequest& request) override {
        if (request.width == 0u || request.height == 0u ||
            request.width > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            request.height > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            request.shared_texture_handle == 0u ||
            request.output_texture_handle == 0u ||
            request.output_width != request.width ||
            request.output_height != request.height ||
            request.signal_fence_handle == 0u ||
            request.signal_fence_value == 0u ||
            (request.pixel_format != DAD_GPU_PIXEL_BGRA8 &&
             request.pixel_format != DAD_GPU_PIXEL_RGBA8)) {
            throw std::invalid_argument(
                "invalid Metal GPU texture inference request");
        }
        id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.shared_texture_handle));
        id<MTLTexture> output_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.output_texture_handle));
        id<MTLSharedEvent> wait_event = request.wait_fence_handle == 0u
            ? nil : (__bridge id<MTLSharedEvent>)(
                reinterpret_cast<void*>(request.wait_fence_handle));
        id<MTLSharedEvent> signal_event =
            (__bridge id<MTLSharedEvent>)(
                reinterpret_cast<void*>(request.signal_fence_handle));
        const MTLPixelFormat expected_input =
            request.pixel_format == DAD_GPU_PIXEL_BGRA8
                ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm;
        if (input_texture.textureType != MTLTextureType2D ||
            input_texture.width != request.width ||
            input_texture.height != request.height ||
            input_texture.pixelFormat != expected_input)
            throw std::invalid_argument(
                "Metal input texture does not match the inference descriptor");
        if (output_texture.textureType != MTLTextureType2D ||
            output_texture.width != request.output_width ||
            output_texture.height != request.output_height ||
            output_texture.pixelFormat != MTLPixelFormatR32Float) {
            throw std::invalid_argument(
                "Metal output texture does not match the inference descriptor");
        }
        const ImageShape network = network_shape(
            static_cast<int>(request.width),
            static_cast<int>(request.height), request.input_size);
        const std::uint64_t normalized_bytes =
            static_cast<std::uint64_t>(network.width) * network.height *
            3u * sizeof(float);
        const std::uint64_t presentation_bytes =
            static_cast<std::uint64_t>(network.width) * network.height *
            sizeof(float);
        if (normalized_bytes > std::numeric_limits<NSUInteger>::max() ||
            presentation_bytes > std::numeric_limits<NSUInteger>::max())
            throw std::overflow_error("Metal inference tensors are too large");

        std::lock_guard<std::mutex> guard(mutex_);
        if (!external_device_adopted_) {
            if (input_texture.device.registryID != device_.registryID)
                throw std::invalid_argument(
                    "Metal input belongs to a different physical device");
            configure_device(input_texture.device);
            external_device_adopted_ = true;
        }
        if (output_texture.device.registryID != device_.registryID)
            throw std::invalid_argument(
                "Metal transfer resources belong to a different physical device "
                "(executor=" + std::to_string(device_.registryID) +
                ", input=" + std::to_string(input_texture.device.registryID) +
                ", output=" + std::to_string(output_texture.device.registryID) +
                ")");
        @autoreleasepool {
            MetalPlan& plan = get_presentation_plan(
                network.width, network.height);
            id<MTLBuffer> normalized = [device_
                newBufferWithLength:static_cast<NSUInteger>(normalized_bytes)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> presentation = [device_
                newBufferWithLength:static_cast<NSUInteger>(presentation_bytes)
                options:MTLResourceStorageModePrivate];
            if (normalized == nil || presentation == nil)
                throw std::bad_alloc();

            id<MTLCommandBuffer> preprocess_command = [queue_ commandBuffer];
            if (wait_event != nil) {
                [preprocess_command encodeWaitForEvent:wait_event
                                                 value:request.wait_fence_value];
            }
            id<MTLComputeCommandEncoder> preprocess_encoder =
                [preprocess_command computeCommandEncoder];
            if (preprocess_encoder == nil)
                throw std::runtime_error(
                    "could not encode DAD Metal preprocessing");
            struct PreprocessParameters {
                std::uint32_t source_width;
                std::uint32_t source_height;
                std::uint32_t destination_width;
                std::uint32_t destination_height;
            } parameters{
                request.width, request.height,
                static_cast<std::uint32_t>(network.width),
                static_cast<std::uint32_t>(network.height)};
            [preprocess_encoder setComputePipelineState:preprocess_pipeline_];
            [preprocess_encoder setTexture:input_texture atIndex:0u];
            [preprocess_encoder setBuffer:normalized offset:0u atIndex:0u];
            [preprocess_encoder setBytes:&parameters
                                  length:sizeof(parameters)
                                 atIndex:1u];
            dispatch_2d(
                preprocess_encoder, preprocess_pipeline_,
                network.width, network.height);
            [preprocess_encoder endEncoding];
            [preprocess_command commit];

            MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:normalized
                shape:shape({1, 3, network.height, network.width})
                dataType:MPSDataTypeFloat32];
            MPSGraphTensorData* output_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:presentation
                shape:shape({1, 1, network.height, network.width})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = NO;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runAsyncWithMTLCommandQueue:queue_
                inputsArray:@[input_data]
                resultsArray:@[output_data]
                executionDescriptor:execution];
            if (results.count != 1u)
                throw std::runtime_error(
                    "Metal graph did not bind its presentation output");

            id<MTLCommandBuffer> completion = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> copy_encoder =
                [completion computeCommandEncoder];
            if (copy_encoder == nil)
                throw std::runtime_error(
                    "could not encode DAD Metal output copy");
            [copy_encoder setComputePipelineState:copy_pipeline_];
            [copy_encoder setBuffer:presentation offset:0u atIndex:0u];
            [copy_encoder setTexture:output_texture atIndex:0u];
            struct PresentationParameters {
                std::uint32_t source_width;
                std::uint32_t source_height;
                std::uint32_t destination_width;
                std::uint32_t destination_height;
            } presentation_parameters{
                static_cast<std::uint32_t>(network.width),
                static_cast<std::uint32_t>(network.height),
                request.width, request.height};
            [copy_encoder setBytes:&presentation_parameters
                            length:sizeof(presentation_parameters)
                           atIndex:1u];
            dispatch_2d(
                copy_encoder, copy_pipeline_,
                request.width, request.height);
            [copy_encoder endEncoding];
            [completion encodeSignalEvent:signal_event
                                    value:request.signal_fence_value];
            [completion commit];
            return std::make_unique<MetalGpuJob>(
                input_texture, wait_event, output_texture, signal_event,
                normalized, presentation, input_data, output_data,
                completion, request.signal_fence_value);
        }
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        upload_bytes = upload_bytes_.load(std::memory_order_relaxed);
        download_bytes = download_bytes_.load(std::memory_order_relaxed);
    }

private:
    void configure_device(id<MTLDevice> device) {
        if (device == nil)
            throw std::runtime_error("Metal is unavailable on this Mac");
        device_ = device;
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize the Metal executor");

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kMetalTransferKernels];
        id<MTLLibrary> library = [device_
            newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            const char* description = error.localizedDescription.UTF8String;
            throw std::runtime_error(description == nullptr
                ? "could not compile DAD Metal transfer kernels"
                : std::string("could not compile DAD Metal transfer kernels: ") +
                    description);
        }
        id<MTLFunction> preprocess =
            [library newFunctionWithName:@"preprocess_texture"];
        id<MTLFunction> copy =
            [library newFunctionWithName:@"resize_depth_to_texture"];
        preprocess_pipeline_ = [device_
            newComputePipelineStateWithFunction:preprocess error:&error];
        if (preprocess_pipeline_ == nil) {
            const char* description = error.localizedDescription.UTF8String;
            throw std::runtime_error(description == nullptr
                ? "could not create DAD Metal preprocess pipeline"
                : std::string("could not create DAD Metal preprocess pipeline: ") +
                    description);
        }
        copy_pipeline_ = [device_
            newComputePipelineStateWithFunction:copy error:&error];
        if (copy_pipeline_ == nil) {
            const char* description = error.localizedDescription.UTF8String;
            throw std::runtime_error(description == nullptr
                ? "could not create DAD Metal output pipeline"
                : std::string("could not create DAD Metal output pipeline: ") +
                    description);
        }
        plans_.clear();
    }

    MetalPlan& get_plan(
        int width, int height, int output_width, int output_height) {
        const PlanKey key{
            width, height, output_width, output_height, false};
        auto existing = plans_.find(key);
        if (existing != plans_.end()) return existing->second;

        MetalGraphBuilder builder(
            model_, encoder_, width, height, output_width, output_height,
            fp16_);
        builder.build(metric_max_depth_);
        builder.cast_output_float32();
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* compilation =
            [MPSGraphCompilationDescriptor new];
        compilation.optimizationLevel = MPSGraphOptimizationLevel1;
        compilation.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph()
            compileWithDevice:graph_device_
            feeds:@{builder.input(): input_type}
            targetTensors:@[builder.output()]
            targetOperations:nil
            compilationDescriptor:compilation];
        if (executable == nil) {
            throw std::runtime_error("failed to compile the Metal graph");
        }
        executable.options = MPSGraphOptionsSynchronizeResults;
        MetalPlan plan;
        plan.graph = builder.graph();
        plan.input = builder.input();
        plan.output = builder.output();
        plan.executable = executable;
        plan.width = width;
        plan.height = height;
        plan.output_width = output_width;
        plan.output_height = output_height;
        return plans_.emplace(key, std::move(plan)).first->second;
    }

    MetalPlan& get_presentation_plan(int width, int height) {
        const PlanKey key{width, height, width, height, true};
        auto existing = plans_.find(key);
        if (existing != plans_.end()) return existing->second;

        MetalGraphBuilder builder(
            model_, encoder_, width, height, width, height, fp16_);
        builder.build_presentation(metric_max_depth_);
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* compilation =
            [MPSGraphCompilationDescriptor new];
        compilation.optimizationLevel = MPSGraphOptimizationLevel1;
        compilation.waitForCompilationCompletion = YES;
        NSURL* cache_url = presentation_cache_url(width, height);
        MPSGraphExecutable* executable = nil;
        if (@available(macOS 14.0, *)) {
            if (cache_url != nil && [[NSFileManager defaultManager]
                    fileExistsAtPath:cache_url.path]) {
                @try {
                    executable = [[MPSGraphExecutable alloc]
                        initWithMPSGraphPackageAtURL:cache_url
                        compilationDescriptor:compilation];
                } @catch (NSException*) {
                    [[NSFileManager defaultManager]
                        removeItemAtURL:cache_url error:nil];
                    executable = nil;
                }
            }
        }
        if (executable == nil) {
            executable = [builder.graph()
                compileWithDevice:graph_device_
                feeds:@{builder.input(): input_type}
                targetTensors:@[builder.output()]
                targetOperations:nil
                compilationDescriptor:compilation];
            if (@available(macOS 14.0, *)) {
                if (executable != nil && cache_url != nil) {
                    @try {
                        [executable serializeToMPSGraphPackageAtURL:cache_url
                                                         descriptor:nil];
                    } @catch (NSException*) {
                        [[NSFileManager defaultManager]
                            removeItemAtURL:cache_url error:nil];
                    }
                }
            }
        }
        if (executable == nil)
            throw std::runtime_error(
                "failed to compile the Metal presentation graph");
        executable.options = MPSGraphOptionsNone;
        MetalPlan plan;
        plan.graph = builder.graph();
        plan.input = builder.input();
        plan.output = builder.output();
        plan.executable = executable;
        plan.width = width;
        plan.height = height;
        plan.output_width = width;
        plan.output_height = height;
        plan.presentation = true;
        return plans_.emplace(key, std::move(plan)).first->second;
    }

    NSURL* presentation_cache_url(int width, int height) const {
        if (@available(macOS 14.0, *)) {
            const ModelDerivation& derivation = model_.derivation();
            if (!derivation.present) return nil;
            if (cache_path_.empty()) return nil;
            NSString* directory = [[NSString
                stringWithUTF8String:cache_path_.c_str()]
                stringByAppendingPathComponent:@"DADMetalGraphCache-v1"];
            NSError* error = nil;
            if (![[NSFileManager defaultManager]
                    createDirectoryAtPath:directory
                    withIntermediateDirectories:YES
                    attributes:nil error:&error]) {
                return nil;
            }
            const NSOperatingSystemVersion os =
                NSProcessInfo.processInfo.operatingSystemVersion;
            const std::string key = hexadecimal(
                derivation.canonical_sha256) + "-" +
                std::to_string(static_cast<unsigned>(encoder_)) + "-" +
                std::to_string(width) + "x" + std::to_string(height) + "-" +
                (fp16_ ? "fp16" : "fp32") + "-" +
                std::to_string(device_.registryID) + "-macos" +
                std::to_string(os.majorVersion) + "." +
                std::to_string(os.minorVersion) + ".mpsgraphpackage";
            return [NSURL fileURLWithPath:[directory
                stringByAppendingPathComponent:name(key)]];
        }
        return nil;
    }

    ModelFile model_;
    dad_encoder encoder_;
    float metric_max_depth_ = 0.0f;
    std::string cache_path_;
    bool fp16_ = false;
    bool external_device_adopted_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> preprocess_pipeline_ = nil;
    id<MTLComputePipelineState> copy_pipeline_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, MetalPlan, PlanKeyHash> plans_;
    std::mutex mutex_;
    std::atomic<std::uint64_t> upload_bytes_{0u};
    std::atomic<std::uint64_t> download_bytes_{0u};
};

}  // namespace

std::unique_ptr<Executor> create_metal_executor(
    const std::string& model_path,
    dad_encoder encoder,
    std::uint32_t flags,
    const std::string& cache_path) {
    return std::make_unique<MetalExecutor>(
        model_path, encoder, flags, cache_path);
}

}  // namespace dad
