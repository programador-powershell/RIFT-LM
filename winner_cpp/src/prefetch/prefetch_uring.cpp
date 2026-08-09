#include "prefetch_uring.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace winner {

IoUringPrefetch::~IoUringPrefetch() { shutdown(); }

int open_weights_direct(const std::string& path, bool odirect) {
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (odirect) flags |= O_DIRECT;
#else
    (void)odirect;
#endif
    int fd = ::open(path.c_str(), flags);
    if (fd < 0 && odirect) {
        // fallback without O_DIRECT
        fd = ::open(path.c_str(), O_RDONLY);
    }
    return fd;
}

void* aligned_alloc_pages(size_t bytes, size_t align) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}

void aligned_free_pages(void* p) { free(p); }

bool IoUringPrefetch::init(size_t /*ring_entries*/, bool use_odirect) {
    shutdown();
    use_odirect_ = use_odirect;
    use_fallback_ = true; // full io_uring needs liburing; fallback always works
    stop_ = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        layer_ready_.assign(256, 0);
        completed_ = 0;
    }
    const unsigned reported = std::max(1u, std::thread::hardware_concurrency());
    const int n = static_cast<int>(std::min(8u, std::max(1u, reported / 2)));
    for (int i = 0; i < n; ++i)
        workers_.emplace_back(&IoUringPrefetch::worker_loop, this);
    return true;
}

void IoUringPrefetch::shutdown() {
    if (workers_.empty()) return;
    stop_.store(true);
    cv_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
    workers_.clear();
    {
        std::lock_guard<std::mutex> lock(mu_);
        while (!q_.empty()) q_.pop();
        pending_.store(0);
    }
    completion_cv_.notify_all();
}

void IoUringPrefetch::worker_loop() {
    while (true) {
        PrefetchRequest req;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) return;
            if (q_.empty()) continue;
            req = std::move(q_.front());
            q_.pop();
        }
        bool ok = false;
        if (req.fd >= 0 && req.dest && req.length > 0) {
            ssize_t n = pread(req.fd, req.dest, req.length, static_cast<off_t>(req.offset));
            ok = (n == static_cast<ssize_t>(req.length));
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (req.layer_id >= 0) {
                if (static_cast<size_t>(req.layer_id) >= layer_ready_.size()) {
                    layer_ready_.resize(static_cast<size_t>(req.layer_id) + 1, 0);
                }
                layer_ready_[static_cast<size_t>(req.layer_id)] = ok ? 1 : 0;
            }
            pending_.fetch_sub(1);
            ++completed_;
        }
        completion_cv_.notify_all();
        if (req.on_complete) req.on_complete(ok);
    }
}

bool IoUringPrefetch::submit(const PrefetchRequest& req) {
    if (stop_.load() || req.fd < 0 || !req.dest || req.length == 0) return false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(req);
        pending_++;
    }
    cv_.notify_one();
    return true;
}

int IoUringPrefetch::wait_completions(int min_complete, int timeout_ms) {
    if (min_complete <= 0) return 0;
    std::unique_lock<std::mutex> lock(mu_);
    const size_t start = completed_;
    const auto ready = [&] {
        return completed_ - start >= static_cast<size_t>(min_complete) || pending_.load() == 0;
    };
    if (timeout_ms < 0) completion_cv_.wait(lock, ready);
    else completion_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
    return static_cast<int>(std::min(completed_ - start, static_cast<size_t>(min_complete)));
}

void IoUringPrefetch::prefetch_layers(int current_layer, int lookahead,
                                      int model_fd,
                                      const std::vector<uint64_t>& layer_offsets,
                                      const std::vector<size_t>& layer_sizes,
                                      void** dest_buffers) {
    for (int k = 1; k <= lookahead; ++k) {
        int L = current_layer + k;
        if (L < 0 || L >= static_cast<int>(layer_offsets.size()) ||
            L >= static_cast<int>(layer_sizes.size())) break;
        PrefetchRequest r;
        r.fd = model_fd;
        r.offset = layer_offsets[L];
        r.length = layer_sizes[L];
        r.dest = dest_buffers ? dest_buffers[L] : nullptr;
        r.layer_id = L;
        submit(r);
    }
}

bool IoUringPrefetch::ready(int layer_id) const {
    if (layer_id < 0) return false;
    std::lock_guard<std::mutex> lock(mu_);
    if (layer_id >= static_cast<int>(layer_ready_.size())) return false;
    return layer_ready_[static_cast<size_t>(layer_id)] != 0;
}

} // namespace winner
