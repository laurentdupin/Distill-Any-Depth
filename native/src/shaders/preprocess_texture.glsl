#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D source_texture;

layout(std430, binding = 1) writeonly buffer Destination {
    float values[];
} destination_data;

layout(push_constant) uniform Parameters {
    uint source_width;
    uint source_height;
    uint destination_width;
    uint destination_height;
} parameters;

float cubic1(float value) {
    const float a = -0.75;
    return ((a + 2.0) * value - (a + 3.0)) *
        value * value + 1.0;
}

float cubic2(float value) {
    const float a = -0.75;
    return ((a * value - 5.0 * a) * value + 8.0 * a) *
        value - 4.0 * a;
}

float coefficient(int tap, float fraction) {
    if (tap == 0) return cubic2(fraction + 1.0);
    if (tap == 1) return cubic1(fraction);
    if (tap == 2) return cubic1(1.0 - fraction);
    return cubic2(2.0 - fraction);
}

void main() {
    const uint destination_x = gl_GlobalInvocationID.x;
    const uint destination_y = gl_GlobalInvocationID.y;
    if (destination_x >= parameters.destination_width ||
        destination_y >= parameters.destination_height) {
        return;
    }

    const float source_x =
        (float(destination_x) + 0.5) *
            float(parameters.source_width) /
            float(parameters.destination_width) -
        0.5;
    const float source_y =
        (float(destination_y) + 0.5) *
            float(parameters.source_height) /
            float(parameters.destination_height) -
        0.5;
    const int base_x = int(floor(source_x));
    const int base_y = int(floor(source_y));
    const float fraction_x = source_x - float(base_x);
    const float fraction_y = source_y - float(base_y);
    const uint destination_plane =
        parameters.destination_width * parameters.destination_height;
    const uint destination_index =
        destination_y * parameters.destination_width + destination_x;
    const float means[3] = float[3](0.485, 0.456, 0.406);
    const float deviations[3] = float[3](0.229, 0.224, 0.225);

    for (uint channel = 0; channel < 3; ++channel) {
        float resized = 0.0;
        for (int tap_y = 0; tap_y < 4; ++tap_y) {
            const int source_iy = clamp(
                base_y - 1 + tap_y,
                0,
                int(parameters.source_height) - 1);
            float row = 0.0;
            for (int tap_x = 0; tap_x < 4; ++tap_x) {
                const int source_ix = clamp(
                    base_x - 1 + tap_x,
                    0,
                    int(parameters.source_width) - 1);
                const vec4 pixel = texelFetch(
                    source_texture,
                    ivec2(source_ix, source_iy),
                    0);
                row += pixel[channel] *
                    coefficient(tap_x, fraction_x);
            }
            resized += row * coefficient(tap_y, fraction_y);
        }
        destination_data.values[
            channel * destination_plane + destination_index] =
            (resized - means[channel]) / deviations[channel];
    }
}
