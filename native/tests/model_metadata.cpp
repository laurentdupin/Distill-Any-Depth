#include "model.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian;
    std::uint32_t encoder;
    std::uint32_t tensor_count;
    std::uint64_t directory_offset;
    std::uint64_t directory_bytes;
    std::uint64_t data_offset;
    std::uint64_t file_bytes;
    std::uint64_t metadata_offset;
};

struct TensorRecord {
    char name[112];
    std::uint32_t dtype;
    std::uint32_t rank;
    std::uint64_t dimensions[4];
    std::uint64_t data_offset;
    std::uint64_t data_bytes;
    std::uint64_t element_count;
    std::uint32_t crc32;
    std::uint32_t flags;
    std::uint64_t reserved;
};

struct DerivationMetadata {
    char magic[8];
    std::uint32_t version;
    std::uint32_t bytes;
    std::uint32_t model_format_version;
    std::uint32_t encoder;
    std::uint32_t flags;
    std::uint32_t reserved;
    std::uint8_t canonical_sha256[32];
    char converter[64];
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 64);
static_assert(sizeof(TensorRecord) == 192);
static_assert(sizeof(DerivationMetadata) == 128);

void write_fixture(
    const std::filesystem::path& path,
    bool include_metadata) {
    constexpr std::uint64_t directory_offset = sizeof(FileHeader);
    constexpr std::uint64_t directory_bytes = sizeof(TensorRecord);
    constexpr std::uint64_t metadata_offset =
        directory_offset + directory_bytes;
    const std::uint64_t data_offset =
        include_metadata ? 384 : 256;
    const std::uint64_t file_bytes = data_offset + sizeof(float);
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(file_bytes));

    FileHeader header{};
    std::memcpy(header.magic, "DAD1MOD", 8);
    header.version = 1;
    header.endian = 0x01020304;
    header.encoder = DAD_ENCODER_VITS;
    header.tensor_count = 1;
    header.directory_offset = directory_offset;
    header.directory_bytes = directory_bytes;
    header.data_offset = data_offset;
    header.file_bytes = file_bytes;
    header.metadata_offset =
        include_metadata ? metadata_offset : 0;
    std::memcpy(bytes.data(), &header, sizeof(header));

    TensorRecord record{};
    std::memcpy(record.name, "weight", 7);
    record.dtype = 1;
    record.rank = 1;
    record.dimensions[0] = 1;
    record.data_offset = data_offset;
    record.data_bytes = sizeof(float);
    record.element_count = 1;
    std::memcpy(
        bytes.data() + directory_offset,
        &record,
        sizeof(record));

    if (include_metadata) {
        DerivationMetadata metadata{};
        std::memcpy(metadata.magic, "DAD1META", 8);
        metadata.version = 1;
        metadata.bytes = sizeof(metadata);
        metadata.model_format_version = 1;
        metadata.encoder = DAD_ENCODER_VITS;
        for (std::size_t index = 0;
             index < sizeof(metadata.canonical_sha256);
             ++index) {
            metadata.canonical_sha256[index] =
                static_cast<std::uint8_t>(index);
        }
        std::memcpy(
            metadata.converter,
            "dad-export-safetensors-v1",
            sizeof("dad-export-safetensors-v1"));
        std::memcpy(
            bytes.data() + metadata_offset,
            &metadata,
            sizeof(metadata));
    }
    const float value = 42.0f;
    std::memcpy(bytes.data() + data_offset, &value, sizeof(value));

    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path();
    const std::string prefix =
        "dad-model-metadata-" +
        std::to_string(GetCurrentProcessId());
    const std::filesystem::path derived =
        root / (prefix + "-derived.dad");
    const std::filesystem::path legacy =
        root / (prefix + "-legacy.dad");
    try {
        {
            write_fixture(derived, true);
            dad::ModelFile derived_model(
                derived.u8string(), DAD_ENCODER_VITS);
            assert(derived_model.derivation().present);
            assert(
                derived_model.derivation().converter ==
                "dad-export-safetensors-v1");
            assert(derived_model.derivation().format_version == 1);
            assert(
                derived_model.derivation().encoder ==
                DAD_ENCODER_VITS);
            for (std::size_t index = 0; index < 32; ++index) {
                assert(
                    derived_model.derivation()
                        .canonical_sha256[index] == index);
            }
            assert(derived_model.tensor("weight").data[0] == 42.0f);
        }

        {
            write_fixture(legacy, false);
            dad::ModelFile legacy_model(
                legacy.u8string(), DAD_ENCODER_VITS);
            assert(!legacy_model.derivation().present);
            assert(legacy_model.tensor("weight").data[0] == 42.0f);
        }
    } catch (...) {
        std::filesystem::remove(derived);
        std::filesystem::remove(legacy);
        throw;
    }
    std::filesystem::remove(derived);
    std::filesystem::remove(legacy);
    return 0;
}
