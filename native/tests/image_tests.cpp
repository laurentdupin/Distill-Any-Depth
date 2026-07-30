#include "image.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    using dad::ImageShape;
    const auto landscape = dad::network_shape(640, 480, 518);
    assert(landscape.width == 686 && landscape.height == 518);
    const auto portrait = dad::network_shape(480, 640, 518);
    assert(portrait.width == 518 && portrait.height == 686);
    const auto square = dad::network_shape(518, 518, 518);
    assert(square.width == 518 && square.height == 518);
    const auto hd = dad::network_shape(1920, 1080, 280);
    assert(hd.width == 504 && hd.height == 280);

    const std::uint8_t bgr[] = {0, 0, 255};
    std::vector<float> chw;
    dad::preprocess_bgr8(bgr, 1, 1, 3, {1, 1}, chw);
    assert(chw.size() == 3);
    assert(std::abs(chw[0] - static_cast<float>((1.0 - 0.485) / 0.229)) < 1e-6f);
    assert(std::abs(chw[1] - static_cast<float>((0.0 - 0.456) / 0.224)) < 1e-6f);
    assert(std::abs(chw[2] - static_cast<float>((0.0 - 0.406) / 0.225)) < 1e-6f);

    const std::uint8_t bgra[] = {
        0, 10, 255, 99,
        255, 20, 0, 88,
    };
    dad::preprocess_inferbridge_bgra8(
        bgra, 2, 1, 8, {4, 1}, chw);
    assert(chw.size() == 12);
    assert(
        std::abs(chw[0] - (0.0f - 0.485f) / 0.229f) < 1e-6f);
    assert(
        std::abs(chw[1] - (0.0f - 0.485f) / 0.229f) < 1e-6f);
    assert(
        std::abs(chw[2] - (1.0f - 0.485f) / 0.229f) < 1e-6f);
    assert(
        std::abs(chw[3] - (1.0f - 0.485f) / 0.229f) < 1e-6f);
    assert(
        std::abs(chw[8] - (1.0f - 0.406f) / 0.225f) < 1e-6f);
    assert(
        std::abs(chw[11] - (0.0f - 0.406f) / 0.225f) < 1e-6f);

    std::vector<std::uint8_t> tall_bgra(425u * 4u, 0u);
    tall_bgra[254u * 4u] = 17u;
    tall_bgra[255u * 4u] = 231u;
    dad::preprocess_inferbridge_bgra8(
        tall_bgra.data(), 1, 425, 4, {1, 210}, chw);
    const float expected_legacy_nearest =
        (17.0f / 255.0f - 0.485f) / 0.229f;
    assert(std::abs(chw[126] - expected_legacy_nearest) < 1e-6f);

    return 0;
}
