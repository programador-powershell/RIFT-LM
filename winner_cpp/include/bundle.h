/**
 * WINNER Bundle format (mmap-native)
 * Logical Page vs Physical Extent separation (RIFT heritage)
 * Accepts containers from SPECTRA / AETHER / CASCADE / RIFT via magic detection.
 */

#ifndef WINNER_BUNDLE_H
#define WINNER_BUNDLE_H

#include "winner.h"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace winner {

#pragma pack(push, 1)
struct BundleHeader {
    char     magic[4];          // "WINR" | "SPCT" | "AETH" | "CASC" | "RIFT"
    uint16_t version;           // 0x0100
    uint16_t flags;             // bit0=HQR, bit1=P-IO, bit2=TADDS, bit3=Fused...
    uint32_t reserved0;
    uint32_t header_size;       // usually 128
    uint64_t op_count;
    uint64_t op_table_offset;
    uint64_t stage_count;
    uint64_t stage_table_offset;
    uint64_t payload_offset;
    uint64_t file_size;
    uint64_t checksum;
    char     reserved[56];
};
#pragma pack(pop)

static_assert(sizeof(BundleHeader) == 128, "BundleHeader must be 128 bytes");

struct StagePageMeta {
    uint32_t stage_id;
    uint32_t operation_id;
    uint16_t stage_index;
    uint8_t  stage_type;
    uint8_t  residency_hint;
    uint32_t codec_id;
    uint64_t file_offset;
    uint64_t payload_bytes;
    uint32_t dependency_stage_id;
    uint32_t checksum;
};

struct TensorDesc {
    uint32_t id;
    char     name[64];
    uint8_t  dtype;
    uint8_t  rank;
    uint32_t dims[8];
    uint32_t semantic_role;
};

class Bundle {
public:
    Bundle() = default;
    ~Bundle();

    bool load(const std::string& path);
    bool is_valid() const { return valid_; }

    const BundleHeader& header() const { return header_; }
    OptimizerId optimizer() const { return optimizer_; }
    const std::vector<StagePageMeta>& stages() const { return stages_; }

    const void* map_stage(uint32_t stage_id, size_t* out_bytes) const;
    const void* base_weights() const { return base_ptr_; }
    size_t      base_bytes()  const { return base_bytes_; }

private:
    bool valid_ = false;
    BundleHeader header_{};
    OptimizerId  optimizer_ = OptimizerId::UNKNOWN;

    std::vector<StagePageMeta> stages_;
    std::vector<TensorDesc>    tensors_;

    void*  mmap_ptr_   = nullptr;
    size_t mmap_size_  = 0;
    int    fd_         = -1;
    std::vector<uint8_t> owned_bytes_;

    const void* base_ptr_  = nullptr;
    size_t      base_bytes_ = 0;

    void reset();
    bool parse_tables();
    void detect_optimizer();
};

} // namespace winner

#endif // WINNER_BUNDLE_H
