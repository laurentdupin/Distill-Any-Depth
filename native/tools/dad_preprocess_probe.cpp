#include "image.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int positive(const char* text) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::runtime_error("dimensions must be positive");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr
            << "usage: dad_preprocess_probe WIDTH HEIGHT NETWORK_WIDTH "
               "NETWORK_HEIGHT INPUT_BGR8 OUTPUT_F32\n";
        return 2;
    }
    try {
        const int width = positive(argv[1]);
        const int height = positive(argv[2]);
        const int network_width = positive(argv[3]);
        const int network_height = positive(argv[4]);
        const std::size_t input_size =
            static_cast<std::size_t>(width) * height * 3;

        std::vector<std::uint8_t> input(input_size);
        std::ifstream source(argv[5], std::ios::binary);
        source.read(
            reinterpret_cast<char*>(input.data()),
            static_cast<std::streamsize>(input.size()));
        if (source.gcount() != static_cast<std::streamsize>(input.size()) ||
            source.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("input file has the wrong size");
        }

        std::vector<float> output;
        dad::preprocess_bgr8(
            input.data(),
            width,
            height,
            static_cast<std::ptrdiff_t>(width) * 3,
            {network_width, network_height},
            output);
        std::ofstream destination(argv[6], std::ios::binary);
        destination.write(
            reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!destination) {
            throw std::runtime_error("failed to write output");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
