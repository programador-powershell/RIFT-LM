#pragma once
/**
 * MoE Expert Paging — LRU/LFU hot cache + routing-grouped batching
 * Expected gain (Kimi/Mixtral class): +40% to +70% tok/s
 */
#include "kernels/residual_ls.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <memory>

namespace winner {
namespace moe {

struct ExpertId {
    int layer = 0;
    int expert = 0;
    bool operator==(const ExpertId& o) const { return layer == o.layer && expert == o.expert; }
};
struct ExpertIdHash {
    size_t operator()(const ExpertId& e) const {
        return (size_t(e.layer) << 20) ^ size_t(e.expert);
    }
};

struct ExpertPage {
    ExpertId id;
    kernels::TernaryMatrix f0;
    kernels::LowRankResidual residual;
    bool residual_loaded = false;
    int last_used_token = -1;
    uint32_t hit_count = 0;   // LFU counter
    uint64_t last_ns = 0;
};

struct MoEConfig {
    int n_layers = 93;
    int n_experts = 896;
    int top_k = 16;
    int n_shared = 2;
    int latent_dim = 3584;
    int expert_inter = 3072;
    int hidden = 7168;
    int residual_rank = 64;
    int ring_capacity = 128;      // more hot slots
    bool use_lfu = true;          // LFU over pure LRU when true
    bool group_by_expert = true;  // routing-grouped batching
    std::string expert_dir;
};

/** Tokens routed to the same expert — one fat GEMM instead of N tiny ones */
struct ExpertBatch {
    ExpertId id;
    std::vector<int> token_indices;  // positions in the micro-batch
    std::vector<float*> activations; // pointers into activation buffer
};

class ExpertPager {
public:
    void init(const MoEConfig& cfg);
    std::shared_ptr<ExpertPage> pin(ExpertId id, bool want_residual, int token_idx);
    void unpin_cold(int token_idx, int keep_recent = 64);
    size_t rss_bytes() const;
    size_t disk_bytes_estimate() const;
    const MoEConfig& config() const { return cfg_; }
    std::vector<int> route(int layer, const float* hidden, int dim);

    /** Group tokens by expert for batched GEMM (routing grouping) */
    std::vector<ExpertBatch> group_by_expert(
        int layer, const std::vector<std::vector<int>>& routes_per_token,
        std::vector<float*>& act_ptrs);

    float hit_rate() const;
    int hits() const { return hits_.load(); }
    int misses() const { return misses_.load(); }

private:
    MoEConfig cfg_;
    std::unordered_map<ExpertId, std::shared_ptr<ExpertPage>, ExpertIdHash> hot_;
    mutable std::mutex mu_;
    std::atomic<int> hits_{0}, misses_{0};
    ExpertPage load_from_disk_or_synth(ExpertId id);
    ExpertId pick_evict_candidate() const;
};

struct K3MemoryEstimate {
    double naive_gguf_gb = 594;
    double winner_disk_tb = 0.54;
    double winner_rss_gb = 5.0;
    int experts_touched_per_token = 1656;
};
K3MemoryEstimate estimate_k3_memory(const MoEConfig& cfg);

} // namespace moe
} // namespace winner
