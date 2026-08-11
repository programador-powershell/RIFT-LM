/**
 * WINNER Bundle format (mmap-native)
 * Logical Page vs Physical Extent separation (RIFT heritage)
 * Accepts containers from SPECTRA / AETHER / CASCADE / RIFT via magic detection.
 * Also parses the real CASCADE container "CSCD" v0x0003 written by
 * cascade/compiler/bundle_writer.py (128-byte header, 24-byte stage entries,
 * JSON stage meta, CRC32-in-u64 checksum over everything after the header).
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

/** CSCD v0x0003 constants (cascade/compiler/bundle_writer.py) */
constexpr uint16_t CSCD_CONTAINER_VERSION = 0x0003;
constexpr uint32_t CSCD_HEADER_SIZE = 128;
constexpr uint32_t CSCD_STAGE_ENTRY_SIZE = 24;   // offset u64 + size u64 + stage_id u32 + flags u32
constexpr uint32_t CSCD_MAX_STAGES = 1024;       // sanity cap (writer emits 2)
constexpr uint32_t CSCD_MAX_META_BYTES = 1u << 20;

/** One raw tensor inside a CSCD stage payload (points into the mapped file). */
struct CscdTensorView {
    std::string dtype;               // "uint8" | "float32" | ...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;   // valid while the Bundle is alive
    size_t bytes = 0;
    size_t elem_count = 0;
};

/** One parsed CSCD stage (F0 INT4_GROUP or F1 FP32_LOWRANK) with its JSON meta. */
struct CscdStageView {
    uint32_t stage_id = 0;
    uint32_t flags = 0;
    uint64_t file_offset = 0;        // stage blob offset in the file
    uint64_t payload_bytes = 0;      // stage blob size (meta + tensors)
    std::string stage_type;          // "BASE_STAGE" | "RESIDUAL_LOWRANK"
    std::string codec;               // "INT4_GROUP" | "FP32_LOWRANK"
    int group_size = 0;
    int out_features = 0;
    int in_features = 0;
    int rank = 0;
    std::vector<CscdTensorView> tensors;
};

class Bundle {
public:
    Bundle() = default;
    ~Bundle();

    // Non-copyable / non-movable: the destructor releases mmap_ptr_/fd_ and
    // CscdTensorView::data points into this object's mapping/buffer, so any
    // copy or move would double-free or dangle (mirrors BundleView in
    // cascade/runtime/cpp/mmap_bundle.hpp).
    Bundle(const Bundle&) = delete;
    Bundle& operator=(const Bundle&) = delete;
    Bundle(Bundle&&) = delete;
    Bundle& operator=(Bundle&&) = delete;

    bool load(const std::string& path);
    bool is_valid() const { return valid_; }

    const BundleHeader& header() const { return header_; }
    OptimizerId optimizer() const { return optimizer_; }
    const std::vector<StagePageMeta>& stages() const { return stages_; }

    const void* map_stage(uint32_t stage_id, size_t* out_bytes) const;
    const void* base_weights() const { return base_ptr_; }
    size_t      base_bytes()  const { return base_bytes_; }

    /** true when the loaded file is a real CASCADE "CSCD" v0x0003 container */
    bool is_cscd() const { return cscd_; }
    const std::vector<CscdStageView>& cscd_stages() const { return cscd_stages_; }
    /** Gate meta from the bundle (ACTIVATION_SCORE_PERCENTILE_V0); -1 when absent */
    double cscd_gate_percentile() const { return cscd_gate_percentile_; }
    const std::string& cscd_gate_type() const { return cscd_gate_type_; }

private:
    bool valid_ = false;
    BundleHeader header_{};
    OptimizerId  optimizer_ = OptimizerId::UNKNOWN;

    std::vector<StagePageMeta> stages_;
    std::vector<TensorDesc>    tensors_;

    bool cscd_ = false;
    std::vector<CscdStageView> cscd_stages_;
    double cscd_gate_percentile_ = -1.0;
    std::string cscd_gate_type_;

    void*  mmap_ptr_   = nullptr;
    size_t mmap_size_  = 0;
    int    fd_         = -1;
    std::vector<uint8_t> owned_bytes_;

    const void* base_ptr_  = nullptr;
    size_t      base_bytes_ = 0;

    void reset();
    bool parse_tables();
    bool parse_cscd();
    void detect_optimizer();
};

} // namespace winner

#endif // WINNER_BUNDLE_H
