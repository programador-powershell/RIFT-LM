#pragma once
/**
 * Zero-copy I/O + async weight prefetch (io_uring / fallback)
 * - O_DIRECT to bypass page cache when safe
 * - Background prefetch of layers L+1, L+2 while computing L
 */
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace winner {

struct PrefetchRequest {
    int      fd = -1;
    uint64_t offset = 0;
    size_t   length = 0;
    void*    dest = nullptr;     // must be aligned for O_DIRECT
    int      layer_id = -1;
    std::function<void(bool ok)> on_complete;
};

class IoUringPrefetch {
public:
    ~IoUringPrefetch();
    bool init(size_t ring_entries = 64, bool use_odirect = true);
    void shutdown();

    // Submit async read (returns immediately)
    bool submit(const PrefetchRequest& req);

    // Wait for at least min_complete completions (0 = poll)
    int  wait_completions(int min_complete = 1, int timeout_ms = -1);

    // High-level: prefetch next layers while current runs
    void prefetch_layers(int current_layer, int lookahead,
                         int model_fd,
                         const std::vector<uint64_t>& layer_offsets,
                         const std::vector<size_t>& layer_sizes,
                         void** dest_buffers);

    bool ready(int layer_id) const;
    size_t pending() const { return pending_.load(); }

private:
    bool use_odirect_ = true;
    void* ring_ = nullptr;          // io_uring* when available
    std::atomic<size_t> pending_{0};
    // Fallback thread-pool path when io_uring not available
    bool use_fallback_ = true;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable completion_cv_;
    std::queue<PrefetchRequest> q_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::vector<uint8_t> layer_ready_;
    size_t completed_ = 0;
    void worker_loop();
};

// Open weights file with O_DIRECT when possible
int open_weights_direct(const std::string& path, bool odirect = true);

// Aligned allocation for O_DIRECT (typically 4096)
void* aligned_alloc_pages(size_t bytes, size_t align = 4096);
void  aligned_free_pages(void* p);

} // namespace winner
