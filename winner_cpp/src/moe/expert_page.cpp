#include "moe/expert_page.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

namespace winner {
namespace moe {

void ExpertPager::init(const MoEConfig& cfg) {
    cfg_ = cfg;
    cfg_.n_layers = std::max(1, cfg_.n_layers);
    cfg_.n_experts = std::max(1, cfg_.n_experts);
    cfg_.top_k = std::max(1, std::min(cfg_.top_k, cfg_.n_experts));
    cfg_.ring_capacity = std::max(1, cfg_.ring_capacity);
    cfg_.latent_dim = std::max(1, cfg_.latent_dim);
    cfg_.expert_inter = std::max(1, cfg_.expert_inter);
    hot_.clear();
    hits_ = 0;
    misses_ = 0;
}

ExpertPage ExpertPager::load_from_disk_or_synth(ExpertId id) {
    ExpertPage p;
    p.id = id;
    std::mt19937 rng(uint32_t(id.layer) * 10007u + uint32_t(id.expert));
    std::normal_distribution<float> nd(0.f, 0.05f);
    int rows = std::min(cfg_.expert_inter, 512);
    int cols = std::min(cfg_.latent_dim, 768);
    std::vector<float> W(size_t(rows) * cols);
    for (auto& v : W) v = nd(rng);
    p.f0 = kernels::pack_ternary(W.data(), rows, cols, -1.f);
    if (cfg_.residual_rank > 0) {
        p.residual = kernels::fit_residual_ls(W.data(), p.f0, cfg_.residual_rank, 3, 1);
        p.residual_loaded = true;
    }
    return p;
}

ExpertId ExpertPager::pick_evict_candidate() const {
    if (hot_.empty()) return {};
    if (cfg_.use_lfu) {
        // LFU with recency tie-break
        ExpertId cold = hot_.begin()->first;
        uint32_t min_hits = hot_.begin()->second->hit_count;
        int oldest = hot_.begin()->second->last_used_token;
        for (auto& kv : hot_) {
            if (kv.second->hit_count < min_hits ||
                (kv.second->hit_count == min_hits && kv.second->last_used_token < oldest)) {
                min_hits = kv.second->hit_count;
                oldest = kv.second->last_used_token;
                cold = kv.first;
            }
        }
        return cold;
    }
    // pure LRU
    ExpertId cold = hot_.begin()->first;
    int oldest = hot_.begin()->second->last_used_token;
    for (auto& kv : hot_) {
        if (kv.second->last_used_token < oldest) {
            oldest = kv.second->last_used_token;
            cold = kv.first;
        }
    }
    return cold;
}

std::shared_ptr<ExpertPage> ExpertPager::pin(ExpertId id, bool /*want_residual*/, int token_idx) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = hot_.find(id);
    if (it != hot_.end()) {
        hits_++;
        it->second->last_used_token = token_idx;
        it->second->hit_count++;
        return it->second;
    }
    misses_++;
    while ((int)hot_.size() >= cfg_.ring_capacity) {
        hot_.erase(pick_evict_candidate());
    }
    auto page = std::make_shared<ExpertPage>(load_from_disk_or_synth(id));
    page->last_used_token = token_idx;
    page->hit_count = 1;
    auto [ins, inserted] = hot_.emplace(id, std::move(page));
    (void)inserted;
    return ins->second;
}

void ExpertPager::unpin_cold(int token_idx, int keep_recent) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = hot_.begin(); it != hot_.end(); ) {
        if (token_idx - it->second->last_used_token > keep_recent)
            it = hot_.erase(it);
        else
            ++it;
    }
}

size_t ExpertPager::rss_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (auto& kv : hot_) {
        n += kv.second->f0.weight.size() + kv.second->f0.scales.size() * 4;
        if (kv.second->residual_loaded)
            n += (kv.second->residual.U.size() + kv.second->residual.V.size()) * 4;
    }
    return n;
}

size_t ExpertPager::disk_bytes_estimate() const {
    int rows = std::min(cfg_.expert_inter, 512);
    int cols = std::min(cfg_.latent_dim, 768);
    return size_t(rows) * cols * size_t(cfg_.n_layers) * size_t(cfg_.n_experts);
}

std::vector<int> ExpertPager::route(int layer, const float* hidden, int dim) {
    uint32_t h = uint32_t(layer) * 2654435761u;
    if (hidden && dim > 0) h ^= uint32_t(std::fabs(hidden[0]) * 1e5f);
    std::vector<int> ids;
    ids.reserve(cfg_.top_k);
    for (int i = 0; i < cfg_.top_k; ++i) {
        h = h * 1664525u + 1013904223u;
        ids.push_back(int(h % uint32_t(cfg_.n_experts)));
    }
    return ids;
}

std::vector<ExpertBatch> ExpertPager::group_by_expert(
    int layer,
    const std::vector<std::vector<int>>& routes_per_token,
    std::vector<float*>& act_ptrs) {

    std::unordered_map<int, ExpertBatch> groups;
    for (size_t t = 0; t < routes_per_token.size(); ++t) {
        for (int eid : routes_per_token[t]) {
            auto& b = groups[eid];
            b.id = {layer, eid};
            b.token_indices.push_back(int(t));
            if (t < act_ptrs.size()) b.activations.push_back(act_ptrs[t]);
        }
    }
    std::vector<ExpertBatch> out;
    out.reserve(groups.size());
    for (auto& kv : groups) out.push_back(std::move(kv.second));
    // largest batches first → better GEMM efficiency
    std::sort(out.begin(), out.end(), [](const ExpertBatch& a, const ExpertBatch& b) {
        return a.token_indices.size() > b.token_indices.size();
    });
    return out;
}

float ExpertPager::hit_rate() const {
    int h = hits_.load(), m = misses_.load();
    if (h + m == 0) return 0.f;
    return float(h) / float(h + m);
}

K3MemoryEstimate estimate_k3_memory(const MoEConfig& cfg) {
    K3MemoryEstimate e;
    double expert_mb_f0 = 17.5 * (1.58 / 4.0);
    e.winner_disk_tb = (cfg.n_layers * cfg.n_experts * expert_mb_f0) / (1024.0 * 1024.0);
    e.winner_rss_gb = 4.0 + (cfg.top_k + cfg.n_shared) * 4 * expert_mb_f0 / 1024.0 + 0.5;
    e.experts_touched_per_token = cfg.n_layers * (cfg.top_k + cfg.n_shared);
    return e;
}

} // namespace moe
} // namespace winner
