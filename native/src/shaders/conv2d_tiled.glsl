#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
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
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel; uint stride; int padding; uint has_bias;
} parameters;
shared float spatial_tile[100];
shared float kernel_tile[36];

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * 4;
    const bool valid =
        output_x < parameters.output_width &&
        output_y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[4] = float[4](0.0, 0.0, 0.0, 0.0);
    const uint lane = gl_LocalInvocationID.y * 8 +
        gl_LocalInvocationID.x;
    const int origin_x = int(gl_WorkGroupID.x * 8) - 1;
    const int origin_y = int(gl_WorkGroupID.y * 8) - 1;
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        for (uint index = lane; index < 100; index += 64) {
            const int x = origin_x + int(index % 10);
            const int y = origin_y + int(index / 10);
            spatial_tile[index] =
                x >= 0 && x < int(parameters.input_width) &&
                y >= 0 && y < int(parameters.input_height)
                ? input_buffer.data[
                      (input_channel * parameters.input_height + uint(y)) *
                      parameters.input_width + uint(x)]
                : 0.0;
        }
        for (uint index = lane; index < 36; index += 64) {
            const uint offset = index / 9;
            const uint channel = channel_base + offset;
            kernel_tile[index] =
                channel < parameters.output_channels
                ? weight_buffer.data[
                      (channel * parameters.input_channels + input_channel) *
                      9 + index % 9]
                : 0.0;
        }
        barrier();
        if (valid) {
            for (uint ky = 0; ky < 3; ++ky)
                for (uint kx = 0; kx < 3; ++kx) {
                    const float value = spatial_tile[
                        (gl_LocalInvocationID.y + ky) * 10 +
                        gl_LocalInvocationID.x + kx];
                    const uint kernel_index = ky * 3 + kx;
                    for (uint offset = 0; offset < 4; ++offset)
                        sums[offset] += value *
                            kernel_tile[offset * 9 + kernel_index];
                }
        }
        barrier();
    }
    if (!valid) return;
    for (uint offset = 0; offset < 4; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels)
            output_buffer.data[
                (channel * parameters.output_height + output_y) *
                parameters.output_width + output_x] =
                sums[offset] + (parameters.has_bias != 0
                    ? bias_buffer.data[channel] : 0.0);
    }
}
