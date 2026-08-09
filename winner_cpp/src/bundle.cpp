/**
 * WINNER Bundle loader (mmap-native, Phase 1 reference)
 */

#include "bundle.h"
#include <cstdio>
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
}

bool Bundle::load(const std::string& path) {
    reset();
#if defined(_WIN32)
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        fprintf(stderr, "[WINNER] cannot open %s\n", path.c_str());
        return false;
    }
    const auto end = stream.tellg();
    if (end < static_cast<std::streamoff>(sizeof(BundleHeader)) ||
        end > static_cast<std::streamoff>(std::numeric_limits<size_t>::max())) {
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

void Bundle::detect_optimizer() {
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
