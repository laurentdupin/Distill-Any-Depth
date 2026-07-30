#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dad {

struct ImageShape {
    int width;
    int height;
};

struct ImageScratch {
    std::vector<int> x_indices;
    std::vector<float> x_coefficients;
    std::vector<double> horizontal;
};

ImageShape network_shape(int width, int height, int input_size);

void preprocess_bgr8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    ImageScratch& scratch,
    std::vector<float>& rgb_chw);

void preprocess_bgr8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    std::vector<float>& rgb_chw);

void preprocess_inferbridge_bgra8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    std::vector<float>& channels_chw);

}  // namespace dad
