/**
 * WINNER Bundle loader (mmap-native, Phase 1 reference)
 */

#include "bundle.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#if defined(_WIN32)
#  include <vector>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace winner {

namespace {

// ---- little-endian field readers (memcpy avoids unaligned UB) ----
inline uint16_t rd_u16(const uint8_t* p) { uint16_t v; memcpy(&v, p, sizeof v); return v; }
inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, sizeof v); return v; }
inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, sizeof v); return v; }

// ---- CRC32 (zlib polynomial 0xEDB88320), matches Python zlib.crc32 ----
struct Crc32Table {
    uint32_t t[256];
    Crc32Table() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
    }
};

uint32_t crc32_bytes(const uint8_t* buf, size_t len) {
    static const Crc32Table table;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = table.t[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ---- minimal extractors for the compact machine-generated JSON metas ----
bool json_find_key(const std::string& j, const char* key, size_t& val_pos) {
    std::string pat = std::string("\"") + key + "\":";
    const size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    val_pos = p + pat.size();
    return val_pos < j.size();
}

bool json_get_string(const std::string& j, const char* key, std::string& out) {
    size_t p = 0;
    if (!json_find_key(j, key, p)) return false;
    while (p < j.size() && j[p] == ' ') ++p;
    if (p >= j.size() || j[p] != '"') return false;
    ++p;
    size_t e = p;
    while (e < j.size() && j[e] != '"') {
        if (j[e] == '\\') return false;  // escaped values are not expected here
        ++e;
    }
    if (e >= j.size()) return false;
    out.assign(j, p, e - p);
    return true;
}

bool json_get_i64(const std::string& j, const char* key, long long& out) {
    size_t p = 0;
    if (!json_find_key(j, key, p)) return false;
    while (p < j.size() && j[p] == ' ') ++p;
    if (p >= j.size()) return false;
    errno = 0;
    const char* s = j.c_str() + p;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || errno != 0) return false;
    out = v;
    return true;
}

bool json_get_f64(const std::string& j, const char* key, double& out) {
    size_t p = 0;
    if (!json_find_key(j, key, p)) return false;
    while (p < j.size() && j[p] == ' ') ++p;
    if (p >= j.size()) return false;
    errno = 0;
    const char* s = j.c_str() + p;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || errno != 0) return false;
    out = v;
    return true;
}

bool json_get_i64_array(const std::string& j, const char* key, std::vector<long long>& out) {
    size_t p = 0;
    if (!json_find_key(j, key, p)) return false;
    while (p < j.size() && j[p] == ' ') ++p;
    if (p >= j.size() || j[p] != '[') return false;
    ++p;
    out.clear();
    while (p < j.size()) {
        while (p < j.size() && (j[p] == ' ' || j[p] == ',')) ++p;
        if (p < j.size() && j[p] == ']') return true;
        errno = 0;
        const char* s = j.c_str() + p;
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || errno != 0) return false;
        out.push_back(v);
        p = size_t(end - j.c_str());
    }
    return false;
}

size_t dtype_elem_size(const std::string& dt) {
    if (dt == "uint8" || dt == "int8" || dt == "bool") return 1;
    if (dt == "float16" || dt == "bfloat16" || dt == "int16" || dt == "uint16") return 2;
    if (dt == "float32" || dt == "int32" || dt == "uint32") return 4;
    if (dt == "float64" || dt == "int64" || dt == "uint64") return 8;
    return 0;
}

} // namespace

Bundle::~Bundle() {
    reset();
}

void Bundle::reset() {
#if !defined(_WIN32)
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
    mmap_ptr_ = nullptr;
    fd_ = -1;
    mmap_size_ = 0;
    valid_ = false;
    header_ = {};
    optimizer_ = OptimizerId::UNKNOWN;
    stages_.clear();
    tensors_.clear();
    base_ptr_ = nullptr;
    base_bytes_ = 0;
    owned_bytes_.clear();
    cscd_ = false;
    cscd_stages_.clear();
    cscd_gate_percentile_ = -1.0;
    cscd_gate_type_.clear();
}

bool Bundle::load(const std::string& path) {
    reset();
#if defined(_WIN32)
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        fprintf(stderr, "[WINNER] cannot open %s\n", path.c_str());
        return false;
    }
    // Compare in unsigned space: casting SIZE_MAX to the signed streamoff
    // wraps to -1 on 64-bit targets, which would reject every file. The
    // upper-bound check only matters where size_t is narrower than streamoff
    // (32-bit builds); on 64-bit it is a no-op.
    const std::streamoff end = static_cast<std::streamoff>(stream.tellg());
    if (end < static_cast<std::streamoff>(sizeof(BundleHeader)) ||
        static_cast<unsigned long long>(end) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        fprintf(stderr, "[WINNER] invalid bundle size\n");
        return false;
    }
    mmap_size_ = static_cast<size_t>(end);
    owned_bytes_.resize(mmap_size_);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(owned_bytes_.data()), static_cast<std::streamsize>(mmap_size_))) {
        fprintf(stderr, "[WINNER] cannot read bundle\n");
        reset();
        return false;
    }
    mmap_ptr_ = owned_bytes_.data();
#else
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "[WINNER] cannot open %s\n", path.c_str());
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    if (st.st_size < static_cast<off_t>(sizeof(BundleHeader))) {
        fprintf(stderr, "[WINNER] file too small\n");
        reset();
        return false;
    }
    mmap_size_ = static_cast<size_t>(st.st_size);

    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        fprintf(stderr, "[WINNER] mmap failed\n");
        reset();
        return false;
    }
#endif

    memcpy(&header_, mmap_ptr_, sizeof(BundleHeader));

    // Real CASCADE container (cascade/compiler/bundle_writer.py): magic CSCD,
    // version 0x0003, own header layout — parsed and CRC-verified separately.
    if (memcmp(header_.magic, "CSCD", 4) == 0) {
        if (!parse_cscd()) {
            reset();
            return false;
        }
        detect_optimizer();
        valid_ = true;
        return true;
    }

    const bool known_magic = memcmp(header_.magic, "WINR", 4) == 0 ||
        memcmp(header_.magic, "SPCT", 4) == 0 ||
        memcmp(header_.magic, "AETH", 4) == 0 ||
        memcmp(header_.magic, "CASC", 4) == 0 ||
        memcmp(header_.magic, "RIFT", 4) == 0;
    if (!known_magic || header_.version != WINNER_CONTAINER_VERSION ||
        header_.header_size < sizeof(BundleHeader) || header_.header_size > mmap_size_ ||
        header_.payload_offset < header_.header_size || header_.payload_offset > mmap_size_ ||
        (header_.file_size != 0 && header_.file_size != mmap_size_)) {
        fprintf(stderr, "[WINNER] invalid or unsupported bundle header\n");
        reset();
        return false;
    }

    if (!parse_tables()) {
        reset();
        return false;
    }
    detect_optimizer();
    valid_ = true;
    return true;
}

bool Bundle::parse_tables() {
    base_ptr_   = static_cast<const char*>(mmap_ptr_) + header_.payload_offset;
    base_bytes_ = mmap_size_ - static_cast<size_t>(header_.payload_offset);

    StagePageMeta f0{};
    f0.stage_id     = 0;
    f0.stage_index  = 0;
    f0.stage_type   = static_cast<uint8_t>(StageType::BASE_STAGE);
    f0.residency_hint = 0;
    f0.file_offset  = header_.payload_offset;
    f0.payload_bytes = base_bytes_;
    stages_.push_back(f0);
    return true;
}

/**
 * CSCD v0x0003 layout (little-endian, from cascade/compiler/bundle_writer.py):
 *   header (128 B): magic "CSCD" | version u16 | flags u16 | header_size u32 |
 *     n_stages u32 | ir_offset u64 | stage_table_offset u64 | gate_table_offset u64 |
 *     payload_offset u64 | file_size u64 | checksum u64 (zlib CRC32 of bytes
 *     [128, file_size)) | reserved u64 | pad 56 B
 *   IR blob at ir_offset:      u32 len + JSON (+ zero pad to stage table)
 *   stage table:               n_stages × { offset u64, size u64, stage_id u32, flags u32 }
 *   gate blob at gate_offset:  u32 len + JSON (+ zero pad to payload)
 *   stage blob at each offset: u32 meta_len + meta JSON + tensors, where each
 *     tensor is u32 tmeta_len + JSON {"dtype","shape"} + raw body; stage blobs
 *     are 64-byte aligned in the file (padding lives between blobs, never inside).
 * Every offset/size is bounds-checked against the file length BEFORE any access.
 */
bool Bundle::parse_cscd() {
    const uint8_t* base = static_cast<const uint8_t*>(mmap_ptr_);
    const uint64_t fsize = mmap_size_;
    if (fsize < CSCD_HEADER_SIZE) {
        fprintf(stderr, "[WINNER] CSCD bundle smaller than its 128-byte header\n");
        return false;
    }
    const uint16_t version            = rd_u16(base + 4);
    const uint16_t flags              = rd_u16(base + 6);
    const uint32_t header_size        = rd_u32(base + 8);
    const uint32_t n_stages           = rd_u32(base + 12);
    const uint64_t ir_offset          = rd_u64(base + 16);
    const uint64_t stage_table_offset = rd_u64(base + 24);
    const uint64_t gate_table_offset  = rd_u64(base + 32);
    const uint64_t payload_offset     = rd_u64(base + 40);
    const uint64_t file_size          = rd_u64(base + 48);
    const uint64_t checksum           = rd_u64(base + 56);

    if (version != CSCD_CONTAINER_VERSION) {
        fprintf(stderr, "[WINNER] unsupported CSCD version 0x%04x (expected 0x%04x)\n",
                unsigned(version), unsigned(CSCD_CONTAINER_VERSION));
        return false;
    }
    if (header_size != CSCD_HEADER_SIZE) {
        fprintf(stderr, "[WINNER] CSCD header_size %u != 128\n", header_size);
        return false;
    }
    if (file_size != fsize) {
        fprintf(stderr, "[WINNER] CSCD file_size field (%llu) != actual file size (%llu)\n",
                (unsigned long long)file_size, (unsigned long long)fsize);
        return false;
    }
    if (n_stages == 0 || n_stages > CSCD_MAX_STAGES) {
        fprintf(stderr, "[WINNER] CSCD stage count %u out of range [1, %u]\n",
                n_stages, CSCD_MAX_STAGES);
        return false;
    }
    // CRC verified over the whole body BEFORE trusting any table.
    const uint64_t crc = crc32_bytes(base + CSCD_HEADER_SIZE, size_t(fsize - CSCD_HEADER_SIZE));
    if (crc != checksum) {
        fprintf(stderr, "[WINNER] CSCD checksum mismatch (stored=%llu computed=%llu) — rejecting corrupted bundle\n",
                (unsigned long long)checksum, (unsigned long long)crc);
        return false;
    }
    // Section offsets, in writer order: header < ir < stage table < gate < payload.
    if (ir_offset < CSCD_HEADER_SIZE || ir_offset > fsize || fsize - ir_offset < 4) {
        fprintf(stderr, "[WINNER] CSCD ir_offset out of bounds\n");
        return false;
    }
    if (stage_table_offset < ir_offset + 4 || stage_table_offset > fsize ||
        (fsize - stage_table_offset) / CSCD_STAGE_ENTRY_SIZE < n_stages) {
        fprintf(stderr, "[WINNER] CSCD stage table out of bounds\n");
        return false;
    }
    const uint64_t stage_table_end = stage_table_offset + uint64_t(n_stages) * CSCD_STAGE_ENTRY_SIZE;
    if (gate_table_offset < stage_table_end || gate_table_offset > fsize ||
        fsize - gate_table_offset < 4) {
        fprintf(stderr, "[WINNER] CSCD gate table out of bounds\n");
        return false;
    }
    if (payload_offset < gate_table_offset + 4 || payload_offset > fsize) {
        fprintf(stderr, "[WINNER] CSCD payload_offset out of bounds\n");
        return false;
    }
    const uint32_t ir_len = rd_u32(base + ir_offset);
    if (ir_len > CSCD_MAX_META_BYTES || uint64_t(ir_len) > stage_table_offset - ir_offset - 4) {
        fprintf(stderr, "[WINNER] CSCD IR blob out of bounds\n");
        return false;
    }
    const uint32_t gate_len = rd_u32(base + gate_table_offset);
    if (gate_len > CSCD_MAX_META_BYTES || uint64_t(gate_len) > payload_offset - gate_table_offset - 4) {
        fprintf(stderr, "[WINNER] CSCD gate blob out of bounds\n");
        return false;
    }
    if (gate_len > 0) {
        const std::string gate_json(reinterpret_cast<const char*>(base + gate_table_offset + 4), gate_len);
        std::string gtype;
        double pct = -1.0;
        if (json_get_string(gate_json, "type", gtype)) cscd_gate_type_ = gtype;
        if (json_get_f64(gate_json, "percentile", pct)) cscd_gate_percentile_ = pct;
    }

    std::vector<CscdStageView> parsed;
    parsed.reserve(n_stages);
    for (uint32_t i = 0; i < n_stages; ++i) {
        const uint8_t* e = base + stage_table_offset + uint64_t(i) * CSCD_STAGE_ENTRY_SIZE;
        CscdStageView sv;
        sv.file_offset   = rd_u64(e);
        sv.payload_bytes = rd_u64(e + 8);
        sv.stage_id      = rd_u32(e + 16);
        sv.flags         = rd_u32(e + 20);
        if (sv.file_offset < payload_offset || sv.file_offset > fsize ||
            sv.payload_bytes < 4 || sv.payload_bytes > fsize - sv.file_offset) {
            fprintf(stderr, "[WINNER] CSCD stage %u payload out of bounds\n", i);
            return false;
        }
        const uint32_t meta_len = rd_u32(base + sv.file_offset);
        if (meta_len > CSCD_MAX_META_BYTES || uint64_t(meta_len) > sv.payload_bytes - 4) {
            fprintf(stderr, "[WINNER] CSCD stage %u meta out of bounds\n", i);
            return false;
        }
        const std::string meta(reinterpret_cast<const char*>(base + sv.file_offset + 4), meta_len);
        json_get_string(meta, "stage_type", sv.stage_type);
        json_get_string(meta, "codec", sv.codec);
        long long v = 0;
        if (json_get_i64(meta, "group_size", v) && v > 0 && v <= (1 << 16)) sv.group_size = int(v);
        if (json_get_i64(meta, "out_features", v) && v > 0 && v <= (1 << 24)) sv.out_features = int(v);
        if (json_get_i64(meta, "in_features", v) && v > 0 && v <= (1 << 24)) sv.in_features = int(v);
        if (json_get_i64(meta, "rank", v) && v > 0 && v <= (1 << 16)) sv.rank = int(v);

        // Tensor sub-blobs fill the rest of the stage blob exactly.
        uint64_t pos = sv.file_offset + 4 + meta_len;
        const uint64_t end = sv.file_offset + sv.payload_bytes;
        while (pos < end) {
            if (end - pos < 4) {
                fprintf(stderr, "[WINNER] CSCD stage %u truncated tensor header\n", i);
                return false;
            }
            const uint32_t tlen = rd_u32(base + pos);
            if (tlen > CSCD_MAX_META_BYTES || uint64_t(tlen) > end - pos - 4) {
                fprintf(stderr, "[WINNER] CSCD stage %u tensor meta out of bounds\n", i);
                return false;
            }
            const std::string tmeta(reinterpret_cast<const char*>(base + pos + 4), tlen);
            CscdTensorView tv;
            std::vector<long long> shp;
            if (!json_get_string(tmeta, "dtype", tv.dtype) ||
                !json_get_i64_array(tmeta, "shape", shp)) {
                fprintf(stderr, "[WINNER] CSCD stage %u tensor meta missing dtype/shape\n", i);
                return false;
            }
            const size_t esz = dtype_elem_size(tv.dtype);
            if (esz == 0) {
                fprintf(stderr, "[WINNER] CSCD stage %u unsupported dtype '%s'\n", i, tv.dtype.c_str());
                return false;
            }
            uint64_t count = 1;
            for (long long d : shp) {
                if (d < 0 || (d != 0 && count > UINT64_MAX / uint64_t(d))) {
                    fprintf(stderr, "[WINNER] CSCD stage %u invalid tensor shape\n", i);
                    return false;
                }
                count *= uint64_t(d);
                tv.shape.push_back(int64_t(d));
            }
            const uint64_t remaining = end - pos - 4 - tlen;
            if (count > remaining / esz) {
                fprintf(stderr, "[WINNER] CSCD stage %u tensor body out of bounds\n", i);
                return false;
            }
            tv.elem_count = size_t(count);
            tv.bytes = size_t(count * esz);
            tv.data = base + pos + 4 + tlen;
            pos += 4 + uint64_t(tlen) + count * esz;
            sv.tensors.push_back(std::move(tv));
        }
        if (pos != end) {
            fprintf(stderr, "[WINNER] CSCD stage %u payload has trailing bytes\n", i);
            return false;
        }
        parsed.push_back(std::move(sv));
    }

    // Legacy-compatible view of the same stages (map_stage / stages()).
    stages_.clear();
    for (uint32_t i = 0; i < n_stages; ++i) {
        const CscdStageView& sv = parsed[i];
        StagePageMeta m{};
        m.stage_id = sv.stage_id;
        m.stage_index = uint16_t(i);
        m.stage_type = static_cast<uint8_t>(
            sv.stage_type == "RESIDUAL_LOWRANK" ? StageType::RESIDUAL_LOWRANK : StageType::BASE_STAGE);
        m.file_offset = sv.file_offset;
        m.payload_bytes = sv.payload_bytes;
        stages_.push_back(m);
    }
    cscd_stages_ = std::move(parsed);
    cscd_ = true;

    // Rebuild header_ with the CSCD fields mapped onto the legacy struct so
    // header() consumers see coherent values (layouts differ on disk).
    BundleHeader h{};
    memcpy(h.magic, "CSCD", 4);
    h.version = CSCD_CONTAINER_VERSION;
    h.flags = flags;
    h.header_size = CSCD_HEADER_SIZE;
    h.op_count = 0;
    h.op_table_offset = ir_offset;      // IR JSON blob
    h.stage_count = n_stages;
    h.stage_table_offset = stage_table_offset;
    h.payload_offset = payload_offset;
    h.file_size = file_size;
    h.checksum = checksum;
    header_ = h;

    base_ptr_ = reinterpret_cast<const char*>(base) + payload_offset;
    base_bytes_ = size_t(fsize - payload_offset);
    return true;
}

void Bundle::detect_optimizer() {
    if (memcmp(header_.magic, "CSCD", 4) == 0) {
        optimizer_ = OptimizerId::CASCADE;
        return;
    }
    if (memcmp(header_.magic, "WINR", 4) == 0 || memcmp(header_.magic, "SPCT", 4) == 0)
        optimizer_ = OptimizerId::SPECTRA;
    else if (memcmp(header_.magic, "AETH", 4) == 0)
        optimizer_ = OptimizerId::AETHER;
    else if (memcmp(header_.magic, "CASC", 4) == 0)
        optimizer_ = OptimizerId::CASCADE;
    else if (memcmp(header_.magic, "RIFT", 4) == 0)
        optimizer_ = OptimizerId::RIFT;
    else
        optimizer_ = OptimizerId::UNKNOWN;
}

const void* Bundle::map_stage(uint32_t stage_id, size_t* out_bytes) const {
    for (const auto& s : stages_) {
        if (s.stage_id == stage_id) {
            if (s.file_offset > mmap_size_ || s.payload_bytes > mmap_size_ - s.file_offset) {
                if (out_bytes) *out_bytes = 0;
                return nullptr;
            }
            if (out_bytes) *out_bytes = s.payload_bytes;
            return static_cast<const char*>(mmap_ptr_) + s.file_offset;
        }
    }
    if (out_bytes) *out_bytes = 0;
    return nullptr;
}

} // namespace winner
