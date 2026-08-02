#include "elementwise_spv.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void expect_near(float actual, float expected, float tolerance) {
    assert(std::abs(actual - expected) <= tolerance);
}

float cubic1(float x) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

float cubic2(float x) {
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

}  // namespace

int main() {
    static constexpr char crc_text[] = "123456789";
    assert(dad::crc32(crc_text, 9) == 0xcbf43926u);

    dad::VulkanContext context(0);
    const std::vector<float> input{
        -4.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 7.0f, 12.0f,
    };
    auto gpu_input = context.create_device_buffer(input.size() * sizeof(float));
    auto gpu_output = context.create_device_buffer(input.size() * sizeof(float));
    context.upload(gpu_input, input.data(), input.size() * sizeof(float));

    auto pipeline = context.create_pipeline(
        dad_elementwise_spv,
        dad_elementwise_spv_size,
        {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        },
        {
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
        },
        16);
    struct Parameters {
        std::uint32_t count;
        float scale;
        float bias;
        std::uint32_t relu;
    } parameters{
        static_cast<std::uint32_t>(input.size()),
        1.5f,
        -0.25f,
        1,
    };
    context.dispatch(
        pipeline,
        {&gpu_input, &gpu_output},
        &parameters,
        sizeof(parameters),
        1);

    std::vector<float> output(input.size());
    context.download(
        gpu_output, output.data(), output.size() * sizeof(float));
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected =
            std::max(input[index] * parameters.scale + parameters.bias, 0.0f);
        assert(std::abs(output[index] - expected) == 0.0f);
    }

    dad::VulkanOperators operators(context);
    {
        const std::vector<float> raw_depth{2.0f, 3.0f, 5.0f, 10.0f};
        auto depth = context.create_device_buffer(raw_depth.size() * sizeof(float));
        auto range = context.create_device_buffer(2u * sizeof(float));
        context.upload(depth, raw_depth.data(), raw_depth.size() * sizeof(float));
        operators.reduce_minmax(depth, range, static_cast<std::uint32_t>(raw_depth.size()));
        operators.normalize_relative(depth, range, static_cast<std::uint32_t>(raw_depth.size()));
        std::vector<float> normalized(raw_depth.size());
        context.download(depth, normalized.data(), normalized.size() * sizeof(float));
        const std::vector<float> expected{0.0f, 0.125f, 0.375f, 1.0f};
        for (std::size_t index = 0; index < expected.size(); ++index)
            expect_near(normalized[index], expected[index], 1.0e-6f);
    }
    const std::uint32_t rows = 3;
    const std::uint32_t input_columns = 19;
    const std::uint32_t output_columns = 7;
    std::vector<float> matrix_input(rows * input_columns);
    std::vector<float> matrix_weight(output_columns * input_columns);
    std::vector<float> matrix_bias(output_columns);
    for (std::size_t index = 0; index < matrix_input.size(); ++index) {
        matrix_input[index] =
            (static_cast<int>(index % 13) - 6) * 0.03125f;
    }
    for (std::size_t index = 0; index < matrix_weight.size(); ++index) {
        matrix_weight[index] =
            (static_cast<int>(index % 11) - 5) * 0.015625f;
    }
    for (std::size_t index = 0; index < matrix_bias.size(); ++index) {
        matrix_bias[index] = static_cast<float>(index) * 0.0625f;
    }
    auto gpu_matrix_input =
        context.create_device_buffer(matrix_input.size() * sizeof(float));
    auto gpu_matrix_weight =
        context.create_device_buffer(matrix_weight.size() * sizeof(float));
    auto gpu_matrix_bias =
        context.create_device_buffer(matrix_bias.size() * sizeof(float));
    auto gpu_matrix_output = context.create_device_buffer(
        rows * output_columns * sizeof(float));
    context.upload(
        gpu_matrix_input,
        matrix_input.data(),
        matrix_input.size() * sizeof(float));
    context.upload(
        gpu_matrix_weight,
        matrix_weight.data(),
        matrix_weight.size() * sizeof(float));
    context.upload(
        gpu_matrix_bias,
        matrix_bias.data(),
        matrix_bias.size() * sizeof(float));
    operators.linear(
        gpu_matrix_output,
        gpu_matrix_input,
        gpu_matrix_weight,
        gpu_matrix_bias,
        rows,
        input_columns,
        output_columns,
        false);
    std::vector<float> matrix_output(rows * output_columns);
    context.download(
        gpu_matrix_output,
        matrix_output.data(),
        matrix_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < output_columns; ++column) {
            float expected = 0.0f;
            for (std::uint32_t inner = 0; inner < input_columns; ++inner) {
                expected += matrix_input[row * input_columns + inner] *
                    matrix_weight[column * input_columns + inner];
            }
            expected += matrix_bias[column];
            expect_near(
                matrix_output[row * output_columns + column],
                expected,
                1.0e-6f);
        }
    }
    operators.linear(
        gpu_matrix_output,
        gpu_matrix_input,
        gpu_matrix_weight,
        gpu_matrix_bias,
        rows,
        input_columns,
        output_columns,
        true);
    context.download(
        gpu_matrix_output,
        matrix_output.data(),
        matrix_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < output_columns; ++column) {
            float expected = 0.0f;
            for (std::uint32_t inner = 0; inner < input_columns; ++inner) {
                expected += matrix_input[row * input_columns + inner] *
                    matrix_weight[column * input_columns + inner];
            }
            expected += matrix_bias[column];
            const float cube = expected * expected * expected;
            const float inner = 0.7978845608028654f *
                (expected + 0.044715f * cube);
            expected = 0.5f * expected *
                (1.0f + std::tanh(std::clamp(inner, -15.0f, 15.0f)));
            expect_near(
                matrix_output[row * output_columns + column],
                expected,
                1.0e-6f);
        }
    }

    const std::uint32_t norm_rows = 4;
    const std::uint32_t norm_columns = 17;
    std::vector<float> norm_input(norm_rows * norm_columns);
    std::vector<float> norm_weight(norm_columns);
    std::vector<float> norm_bias(norm_columns);
    for (std::size_t index = 0; index < norm_input.size(); ++index) {
        norm_input[index] =
            (static_cast<int>(index % 23) - 10) * 0.125f;
    }
    std::fill(norm_weight.begin(), norm_weight.end(), 1.0f);
    std::fill(norm_bias.begin(), norm_bias.end(), 0.0f);
    auto gpu_norm_input =
        context.create_device_buffer(norm_input.size() * sizeof(float));
    auto gpu_norm_weight =
        context.create_device_buffer(norm_weight.size() * sizeof(float));
    auto gpu_norm_bias =
        context.create_device_buffer(norm_bias.size() * sizeof(float));
    auto gpu_norm_output =
        context.create_device_buffer(norm_input.size() * sizeof(float));
    context.upload(
        gpu_norm_input, norm_input.data(), norm_input.size() * sizeof(float));
    context.upload(
        gpu_norm_weight,
        norm_weight.data(),
        norm_weight.size() * sizeof(float));
    context.upload(
        gpu_norm_bias, norm_bias.data(), norm_bias.size() * sizeof(float));
    operators.layer_norm(
        gpu_norm_output,
        gpu_norm_input,
        gpu_norm_weight,
        gpu_norm_bias,
        norm_rows,
        norm_columns,
        1.0e-6f);
    std::vector<float> norm_output(norm_input.size());
    context.download(
        gpu_norm_output,
        norm_output.data(),
        norm_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < norm_rows; ++row) {
        float sum = 0.0f;
        float sum_of_squares = 0.0f;
        for (std::uint32_t column = 0; column < norm_columns; ++column) {
            const float value = norm_input[row * norm_columns + column];
            sum += value;
            sum_of_squares += value * value;
        }
        const float mean = sum / norm_columns;
        const float variance =
            std::max(sum_of_squares / norm_columns - mean * mean, 0.0f);
        const float inverse = 1.0f / std::sqrt(variance + 1.0e-6f);
        for (std::uint32_t column = 0; column < norm_columns; ++column) {
            expect_near(
                norm_output[row * norm_columns + column],
                (norm_input[row * norm_columns + column] - mean) * inverse,
                1.0e-5f);
        }
    }

    const std::uint32_t scaled_columns = 5;
    const std::uint32_t scaled_count = 15;
    std::vector<float> residual(scaled_count);
    std::vector<float> addend(scaled_count);
    std::vector<float> scale(scaled_columns);
    for (std::uint32_t index = 0; index < scaled_count; ++index) {
        residual[index] = static_cast<float>(index) * 0.125f;
        addend[index] = static_cast<float>(index + 2) * -0.0625f;
    }
    for (std::uint32_t index = 0; index < scaled_columns; ++index) {
        scale[index] = static_cast<float>(index + 1) * 0.25f;
    }
    auto gpu_residual =
        context.create_device_buffer(residual.size() * sizeof(float));
    auto gpu_addend =
        context.create_device_buffer(addend.size() * sizeof(float));
    auto gpu_scale =
        context.create_device_buffer(scale.size() * sizeof(float));
    auto gpu_scaled_output =
        context.create_device_buffer(residual.size() * sizeof(float));
    context.upload(
        gpu_residual, residual.data(), residual.size() * sizeof(float));
    context.upload(gpu_addend, addend.data(), addend.size() * sizeof(float));
    context.upload(gpu_scale, scale.data(), scale.size() * sizeof(float));
    operators.add_scaled(
        gpu_scaled_output,
        gpu_residual,
        gpu_addend,
        gpu_scale,
        scaled_count,
        scaled_columns);
    std::vector<float> scaled_output(scaled_count);
    context.download(
        gpu_scaled_output,
        scaled_output.data(),
        scaled_output.size() * sizeof(float));
    for (std::uint32_t index = 0; index < scaled_count; ++index) {
        expect_near(
            scaled_output[index],
            residual[index] +
                addend[index] * scale[index % scaled_columns],
            0.0f);
    }

    const std::uint32_t attention_tokens = 5;
    const std::uint32_t attention_heads = 2;
    const std::uint32_t attention_embedding = attention_heads * 64;
    std::vector<float> qkv(
        attention_tokens * attention_embedding * 3);
    for (std::size_t index = 0; index < qkv.size(); ++index) {
        qkv[index] =
            (static_cast<int>(index % 29) - 14) * 0.0078125f;
    }
    const std::size_t attention_elements =
        attention_tokens * attention_embedding;
    auto gpu_qkv =
        context.create_device_buffer(qkv.size() * sizeof(float));
    auto gpu_attention =
        context.create_device_buffer(attention_elements * sizeof(float));
    context.upload(gpu_qkv, qkv.data(), qkv.size() * sizeof(float));
    operators.attention_head64(
        gpu_attention,
        gpu_qkv,
        attention_tokens,
        attention_heads);
    std::vector<float> merged(attention_elements);
    context.download(
        gpu_attention, merged.data(), merged.size() * sizeof(float));
    const std::vector<float> fp32_attention = merged;
    for (std::uint32_t token = 0; token < attention_tokens; ++token) {
        for (std::uint32_t head = 0; head < attention_heads; ++head) {
            std::vector<float> scores(attention_tokens);
            float maximum = -std::numeric_limits<float>::max();
            for (std::uint32_t source = 0;
                 source < attention_tokens;
                 ++source) {
                float score = 0.0f;
                for (std::uint32_t feature = 0; feature < 64; ++feature) {
                    const std::size_t query_index =
                        token * attention_embedding * 3 +
                        head * 64 + feature;
                    const std::size_t key_index =
                        source * attention_embedding * 3 +
                        attention_embedding + head * 64 + feature;
                    score += qkv[query_index] * 0.125f *
                        qkv[key_index];
                }
                scores[source] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::uint32_t feature = 0; feature < 64; ++feature) {
                float expected = 0.0f;
                for (std::uint32_t source = 0;
                     source < attention_tokens;
                     ++source) {
                    const std::size_t value_index =
                        source * attention_embedding * 3 +
                        attention_embedding * 2 +
                        head * 64 + feature;
                    expected += scores[source] / denominator *
                        qkv[value_index];
                }
                expect_near(
                    merged[token * attention_embedding +
                        head * 64 + feature],
                    expected,
                    2.0e-6f);
            }
        }
    }

    operators.attention_head64(
        gpu_attention,
        gpu_qkv,
        attention_tokens,
        attention_heads,
        nullptr,
        true);
    context.download(
        gpu_attention, merged.data(), merged.size() * sizeof(float));
    for (std::size_t index = 0; index < merged.size(); ++index) {
        expect_near(
            merged[index],
            fp32_attention[index],
            2.0e-4f);
    }

    const std::uint32_t token_width = 42;
    const std::uint32_t token_height = 28;
    const std::uint32_t token_embedding = 3;
    std::vector<float> token_image(
        token_width * token_height * 3, 0.0f);
    std::vector<float> patch_weight(
        token_embedding * 3 * 14 * 14, 0.0f);
    std::vector<float> patch_bias{0.25f, -0.5f, 0.75f};
    std::vector<float> class_token{1.0f, 2.0f, 3.0f};
    std::vector<float> position(1370 * token_embedding);
    for (std::size_t index = 0; index < position.size(); ++index) {
        position[index] =
            (static_cast<int>(index % 31) - 15) * 0.015625f;
    }
    auto gpu_token_image =
        context.create_device_buffer(token_image.size() * sizeof(float));
    auto gpu_patch_weight =
        context.create_device_buffer(patch_weight.size() * sizeof(float));
    auto gpu_patch_bias =
        context.create_device_buffer(patch_bias.size() * sizeof(float));
    auto gpu_class_token =
        context.create_device_buffer(class_token.size() * sizeof(float));
    auto gpu_position =
        context.create_device_buffer(position.size() * sizeof(float));
    auto gpu_tokens = context.create_device_buffer(
        (1 + (token_width / 14) * (token_height / 14)) *
        token_embedding * sizeof(float));
    context.upload(
        gpu_token_image,
        token_image.data(),
        token_image.size() * sizeof(float));
    context.upload(
        gpu_patch_weight,
        patch_weight.data(),
        patch_weight.size() * sizeof(float));
    context.upload(
        gpu_patch_bias,
        patch_bias.data(),
        patch_bias.size() * sizeof(float));
    context.upload(
        gpu_class_token,
        class_token.data(),
        class_token.size() * sizeof(float));
    context.upload(
        gpu_position,
        position.data(),
        position.size() * sizeof(float));
    operators.prepare_tokens(
        gpu_tokens,
        gpu_token_image,
        gpu_patch_weight,
        gpu_patch_bias,
        gpu_class_token,
        gpu_position,
        token_width,
        token_height,
        token_embedding);
    std::vector<float> tokens(7 * token_embedding);
    context.download(
        gpu_tokens, tokens.data(), tokens.size() * sizeof(float));
    for (std::uint32_t feature = 0; feature < token_embedding; ++feature) {
        expect_near(
            tokens[feature],
            class_token[feature] + position[feature],
            0.0f);
    }
    const std::uint32_t patch_width = token_width / 14;
    const std::uint32_t patch_height = token_height / 14;
    for (std::uint32_t patch_y = 0; patch_y < patch_height; ++patch_y) {
        for (std::uint32_t patch_x = 0; patch_x < patch_width; ++patch_x) {
            const float source_x =
                (patch_x + 0.5f) * (37.0f / (patch_width + 0.1f)) -
                0.5f;
            const float source_y =
                (patch_y + 0.5f) * (37.0f / (patch_height + 0.1f)) -
                0.5f;
            const int base_x = static_cast<int>(std::floor(source_x));
            const int base_y = static_cast<int>(std::floor(source_y));
            const float fraction_x = source_x - base_x;
            const float fraction_y = source_y - base_y;
            const float x_coefficients[4] = {
                cubic2(fraction_x + 1.0f),
                cubic1(fraction_x),
                cubic1(1.0f - fraction_x),
                cubic2(2.0f - fraction_x),
            };
            const float y_coefficients[4] = {
                cubic2(fraction_y + 1.0f),
                cubic1(fraction_y),
                cubic1(1.0f - fraction_y),
                cubic2(2.0f - fraction_y),
            };
            for (std::uint32_t feature = 0;
                 feature < token_embedding;
                 ++feature) {
                float rows[4]{};
                for (int row = 0; row < 4; ++row) {
                    const int y = std::clamp(base_y - 1 + row, 0, 36);
                    for (int column = 0; column < 4; ++column) {
                        const int x =
                            std::clamp(base_x - 1 + column, 0, 36);
                        rows[row] +=
                            position[
                                (1 + y * 37 + x) * token_embedding +
                                feature] *
                            x_coefficients[column];
                    }
                }
                float expected = patch_bias[feature];
                for (int row = 0; row < 4; ++row) {
                    expected += rows[row] * y_coefficients[row];
                }
                const std::uint32_t patch =
                    patch_y * patch_width + patch_x;
                expect_near(
                    tokens[(patch + 1) * token_embedding + feature],
                    expected,
                    2.0e-6f);
            }
        }
    }

    const std::vector<float> resize_input{
        1.0f, 3.0f,
        5.0f, 7.0f,
    };
    auto gpu_resize_input =
        context.create_device_buffer(resize_input.size() * sizeof(float));
    auto gpu_resize_output =
        context.create_device_buffer(9 * sizeof(float));
    context.upload(
        gpu_resize_input,
        resize_input.data(),
        resize_input.size() * sizeof(float));
    operators.bilinear_align_false(
        gpu_resize_output,
        gpu_resize_input,
        2,
        2,
        3,
        3,
        1);
    std::vector<float> resize_output(9);
    context.download(
        gpu_resize_output,
        resize_output.data(),
        resize_output.size() * sizeof(float));
    const std::vector<float> resize_expected{
        1.0f, 2.0f, 3.0f,
        3.0f, 4.0f, 5.0f,
        5.0f, 6.0f, 7.0f,
    };
    for (std::size_t index = 0; index < resize_output.size(); ++index) {
        expect_near(resize_output[index], resize_expected[index], 1.0e-6f);
    }

    std::cout << context.device_name() << '\n';
    return 0;
}
