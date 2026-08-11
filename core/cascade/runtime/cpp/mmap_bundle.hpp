#pragma once
// CASCADE Bundle M0 — mmap reader (C++17, POSIX-only; Windows builds skip it)
// Validates every offset/size against the mapped length BEFORE any access and
// verifies the CRC32 checksum (fail closed with std::runtime_error).
#if !defined(_WIN32)

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cascade {

static constexpr char kMagic[4] = {'C','S','C','D'};
static constexpr uint16_t kVersion = 0x0003;
static constexpr size_t kHeaderSize = 128;
static constexpr size_t kStageEntrySize = 24;   // offset u64 + size u64 + stage_id u32 + flags u32
static constexpr uint32_t kMaxStages = 1024;    // sanity cap (writer emits 2)

namespace detail {

inline uint16_t rd_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, sizeof v); return v; }
inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, sizeof v); return v; }
inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, sizeof v); return v; }

// CRC32 (zlib polynomial 0xEDB88320) — matches Python zlib.crc32
inline uint32_t crc32_bytes(const uint8_t* buf, size_t len) {
  struct Table {
    uint32_t t[256];
    Table() {
      for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[i] = c;
      }
    }
  };
  static const Table table;
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i)
    crc = table.t[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

} // namespace detail

struct StageEntry {
  uint64_t offset;
  uint64_t size;
  uint32_t stage_id;
  uint32_t flags;
};

struct BundleView {
  const uint8_t* data = nullptr;
  size_t length = 0;
  int fd = -1;
  bool owns_map = false;

  uint16_t version = 0;
  uint32_t n_stages = 0;
  uint64_t ir_offset = 0;
  uint64_t stage_table_offset = 0;
  uint64_t gate_table_offset = 0;
  uint64_t payload_offset = 0;
  uint64_t file_size = 0;
  uint64_t checksum = 0;
  std::vector<StageEntry> stages;

  BundleView() = default;
  BundleView(const BundleView&) = delete;
  BundleView& operator=(const BundleView&) = delete;
  BundleView(BundleView&& o) noexcept { *this = std::move(o); }
  BundleView& operator=(BundleView&& o) noexcept {
    if (this != &o) {
      close();
      data = o.data; length = o.length; fd = o.fd; owns_map = o.owns_map;
      version = o.version; n_stages = o.n_stages;
      ir_offset = o.ir_offset; stage_table_offset = o.stage_table_offset;
      gate_table_offset = o.gate_table_offset; payload_offset = o.payload_offset;
      file_size = o.file_size; checksum = o.checksum;
      stages = std::move(o.stages);
      o.data = nullptr; o.length = 0; o.fd = -1; o.owns_map = false;
    }
    return *this;
  }

  ~BundleView() { close(); }

  void close() {
    if (owns_map && data && data != MAP_FAILED) {
      munmap(const_cast<uint8_t*>(data), length);
    }
    data = nullptr;
    if (fd >= 0) { ::close(fd); fd = -1; }
    owns_map = false;
  }

  static BundleView open_mmap(const std::string& path) {
    BundleView b;
    b.fd = ::open(path.c_str(), O_RDONLY);
    if (b.fd < 0) throw std::runtime_error("open failed: " + path);
    struct stat st{};
    if (fstat(b.fd, &st) != 0) throw std::runtime_error("fstat failed");
    if (st.st_size < static_cast<off_t>(kHeaderSize))
      throw std::runtime_error("file too small");
    b.length = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, b.length, PROT_READ, MAP_PRIVATE, b.fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
    b.data = static_cast<const uint8_t*>(p);
    b.owns_map = true;
    b.parse_header();
    return b;
  }

  void parse_header() {
    using detail::rd_u16;
    using detail::rd_u32;
    using detail::rd_u64;
    if (length < kHeaderSize) throw std::runtime_error("file too small");
    if (std::memcmp(data, kMagic, 4) != 0) throw std::runtime_error("bad magic");
    // layout: 4s H H I I Q Q Q Q Q Q Q 56s  (little-endian)
    version            = rd_u16(data + 4);
    const uint32_t hdr = rd_u32(data + 8);
    n_stages           = rd_u32(data + 12);
    ir_offset          = rd_u64(data + 16);
    stage_table_offset = rd_u64(data + 24);
    gate_table_offset  = rd_u64(data + 32);
    payload_offset     = rd_u64(data + 40);
    file_size          = rd_u64(data + 48);
    checksum           = rd_u64(data + 56);

    if (version != kVersion) throw std::runtime_error("unsupported CSCD version");
    if (hdr != kHeaderSize) throw std::runtime_error("bad header size");
    if (file_size != length) throw std::runtime_error("file_size field mismatch");
    if (n_stages == 0 || n_stages > kMaxStages) throw std::runtime_error("stage count out of range");

    // Checksum verified over everything after the header BEFORE trusting tables.
    const uint64_t crc = detail::crc32_bytes(data + kHeaderSize, length - kHeaderSize);
    if (crc != checksum) throw std::runtime_error("checksum mismatch");

    if (ir_offset < kHeaderSize || ir_offset > length)
      throw std::runtime_error("ir_offset out of bounds");
    if (stage_table_offset < kHeaderSize || stage_table_offset > length ||
        (length - stage_table_offset) / kStageEntrySize < n_stages)
      throw std::runtime_error("stage table out of bounds");
    if (gate_table_offset < kHeaderSize || gate_table_offset > length)
      throw std::runtime_error("gate table out of bounds");
    if (payload_offset < kHeaderSize || payload_offset > length)
      throw std::runtime_error("payload offset out of bounds");

    stages.resize(n_stages);
    const uint8_t* st = data + stage_table_offset;
    for (uint32_t i = 0; i < n_stages; ++i) {
      stages[i].offset = rd_u64(st); st += 8;
      stages[i].size = rd_u64(st); st += 8;
      stages[i].stage_id = rd_u32(st); st += 4;
      stages[i].flags = rd_u32(st); st += 4;
      if (stages[i].offset < payload_offset || stages[i].offset > length ||
          stages[i].size > length - stages[i].offset)
        throw std::runtime_error("stage payload out of bounds");
    }
  }

  const uint8_t* stage_ptr(uint32_t idx) const {
    if (idx >= stages.size()) throw std::runtime_error("stage idx");
    // offset/size already validated against `length` in parse_header()
    return data + stages[idx].offset;
  }
};

} // namespace cascade

#endif // !defined(_WIN32)
