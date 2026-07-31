#version 450 core

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly restrict buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly restrict buffer Input {
    vec4 data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly restrict buffer Weight {
    vec4 data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly restrict buffer Bias {
    float data[];
} bias_buffer;
layout(set = 0, binding = 4, std430) readonly restrict buffer Residual {
    float data[];
} residual_buffer;
layout(set = 0, binding = 5, std430) readonly restrict buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
    uint gelu;
    uint residual;
} parameters;

#define K_VECTORS 8
#define K_STRIDE (K_VECTORS + 1)
shared vec4 input_tile[40 * K_STRIDE];
shared vec4 weight_tile[64 * K_STRIDE];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x;
    const uint row_base =
        gl_WorkGroupID.y * 40 + gl_LocalInvocationID.y * 5;
    float sums[5][4];
    for (uint row = 0; row < 5; ++row)
        for (uint column = 0; column < 4; ++column)
            sums[row][column] = 0.0;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    const uint input_vectors = parameters.input_columns / 4;
    for (uint inner_base = 0;
         inner_base < input_vectors;
         inner_base += K_VECTORS) {
        for (uint index = lane; index < 40 * K_VECTORS; index += 128) {
            const uint tile_row = index / K_VECTORS;
            const uint inner = inner_base + index % K_VECTORS;
            const uint output_row =
                gl_WorkGroupID.y * 40 + tile_row;
            input_tile[tile_row * K_STRIDE +
                       index % K_VECTORS] =
                output_row < parameters.rows && inner < input_vectors
                ? input_buffer.data[
                      output_row * input_vectors + inner]
                : vec4(0.0);
        }
        for (uint index = lane; index < 64 * K_VECTORS; index += 128) {
            const uint tile_column = index / K_VECTORS;
            const uint inner = inner_base + index % K_VECTORS;
            const uint output_column =
                gl_WorkGroupID.x * 64 + tile_column;
            weight_tile[tile_column * K_STRIDE +
                        index % K_VECTORS] =
                output_column < parameters.output_columns &&
                    inner < input_vectors
                ? weight_buffer.data[
                      output_column * input_vectors + inner]
                : vec4(0.0);
        }
        barrier();
        const uint count =
            min(K_VECTORS, input_vectors - inner_base);
        for (uint inner = 0; inner < count; ++inner) {
            vec4 input_values[5];
            vec4 weight_values[4];
            for (uint row = 0; row < 5; ++row)
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * 5 + row) * K_STRIDE +
                    inner];
            for (uint column = 0; column < 4; ++column)
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x + column * 16) * K_STRIDE +
                    inner];
            for (uint row = 0; row < 5; ++row)
                for (uint column = 0; column < 4; ++column)
                    sums[row][column] +=
                        dot(input_values[row], weight_values[column]);
        }
        barrier();
    }
    for (uint row = 0; row < 5; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) continue;
        for (uint column = 0; column < 4; ++column) {
            const uint output_column =
                column_base + column * 16;
            if (output_column < parameters.output_columns) {
                float value =
                    sums[row][column] + bias_buffer.data[output_column];
                if (parameters.gelu != 0) {
                    const float scaled =
                        abs(value) * 0.7071067811865476;
                    const float t =
                        1.0 / (1.0 + 0.3275911 * scaled);
                    float polynomial = 1.061405429 * t - 1.453152027;
                    polynomial = polynomial * t + 1.421413741;
                    polynomial = polynomial * t - 0.284496736;
                    polynomial = polynomial * t + 0.254829592;
                    polynomial *= t;
                    const float erf_magnitude =
                        1.0 - polynomial * exp(-scaled * scaled);
                    const float erf_value =
                        value < 0.0 ? -erf_magnitude : erf_magnitude;
                    value = 0.5 * value * (1.0 + erf_value);
                }
                if (parameters.residual != 0) {
                    value =
                        residual_buffer.data[
                            output_row * parameters.output_columns +
                            output_column] +
                        value * scale_buffer.data[output_column];
                }
                output_buffer.data[
                    output_row * parameters.output_columns + output_column] =
                    value;
            }
        }
    }
}
