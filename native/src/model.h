#pragma once

#include "distill_any_depth.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace dad {

struct TensorView {
    const float* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
    std::uint32_t crc32 = 0;
};

struct ModelDerivation {
    bool present = false;
    std::array<std::uint8_t, 32> canonical_sha256{};
    std::string converter;
    std::uint32_t format_version = 0;
    dad_encoder encoder = DAD_ENCODER_VITS;
};

class ModelFile {
public:
    ModelFile(const std::string& path_utf8, dad_encoder expected_encoder);
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ~ModelFile();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    const std::vector<std::string_view>& tensor_names() const {
        return tensor_names_;
    }
    const ModelDerivation& derivation() const {
        return derivation_;
    }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::unordered_map<std::string_view, TensorView> tensors_;
    std::vector<std::string_view> tensor_names_;
    ModelDerivation derivation_;
};

}  // namespace dad
