#version 450 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
    float epsilon;
} parameters;

void main() {
    const uint row = gl_GlobalInvocationID.x;
    if (row >= parameters.rows) {
        return;
    }

    const uint base = row * parameters.columns;
    float sum = 0.0;
    float sum_of_squares = 0.0;
    for (uint column = 0; column < parameters.columns; ++column) {
        const float value = input_buffer.data[base + column];
        sum += value;
        sum_of_squares += value * value;
    }
    const float denominator = max(float(parameters.columns), 1.0);
    const float mean = sum / denominator;
    const float variance =
        max(sum_of_squares / denominator - mean * mean, 0.0);
    const float inverse_deviation =
        inversesqrt(variance + parameters.epsilon);
    for (uint column = 0; column < parameters.columns; ++column) {
        output_buffer.data[base + column] =
            (input_buffer.data[base + column] - mean) *
                inverse_deviation * weight_buffer.data[column] +
            bias_buffer.data[column];
    }
}
