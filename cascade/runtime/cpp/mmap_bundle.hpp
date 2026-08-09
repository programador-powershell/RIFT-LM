#pragma once
// CASCADE Bundle M0 — mmap reader (C++20)
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cascade {

static constexpr char kMagic[4] = {'C','S','C','D'};
static constexpr uint16_t kVersion = 0x0003;
static constexpr size_t kHeaderSize = 128;

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
    b.length = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, b.length, PROT_READ, MAP_PRIVATE, b.fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
    b.data = static_cast<const uint8_t*>(p);
    b.owns_map = true;
    b.parse_header();
    return b;
  }

  void parse_header() {
    if (length < kHeaderSize) throw std::runtime_error("file too small");
    if (std::memcmp(data, kMagic, 4) != 0) throw std::runtime_error("bad magic");
    // layout: 4s H H I I Q Q Q Q Q Q Q 56s  (little-endian)
    const uint8_t* p = data;
    p += 4;
    version = *reinterpret_cast<const uint16_t*>(p); p += 2;
    p += 2; // flags
    uint32_t hdr = *reinterpret_cast<const uint32_t*>(p); p += 4;
    n_stages = *reinterpret_cast<const uint32_t*>(p); p += 4;
    ir_offset = *reinterpret_cast<const uint64_t*>(p); p += 8;
    stage_table_offset = *reinterpret_cast<const uint64_t*>(p); p += 8;
    gate_table_offset = *reinterpret_cast<const uint64_t*>(p); p += 8;
    payload_offset = *reinterpret_cast<const uint64_t*>(p); p += 8;
    file_size = *reinterpret_cast<const uint64_t*>(p); p += 8;
    checksum = *reinterpret_cast<const uint64_t*>(p); p += 8;
    (void)hdr;
    stages.resize(n_stages);
    const uint8_t* st = data + stage_table_offset;
    for (uint32_t i = 0; i < n_stages; ++i) {
      stages[i].offset = *reinterpret_cast<const uint64_t*>(st); st += 8;
      stages[i].size = *reinterpret_cast<const uint64_t*>(st); st += 8;
      stages[i].stage_id = *reinterpret_cast<const uint32_t*>(st); st += 4;
      stages[i].flags = *reinterpret_cast<const uint32_t*>(st); st += 4;
    }
  }

  const uint8_t* stage_ptr(uint32_t idx) const {
    if (idx >= stages.size()) throw std::runtime_error("stage idx");
    return data + stages[idx].offset;
  }
};

} // namespace cascade
