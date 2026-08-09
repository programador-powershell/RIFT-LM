#include "attention/page_table.h"
#include <cmath>
#include <algorithm>
#if defined(__AVX2__)
#  include <immintrin.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#  define WINNER_PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#else
#  define WINNER_PREFETCH(addr) ((void)0)
#endif

namespace winner {
namespace attention {

void PageTable::init(int n_layers, int n_heads, int head_dim, int max_blocks) {
    n_layers_ = std::max(0, n_layers);
    n_heads_ = std::max(0, n_heads);
    head_dim_ = std::max(0, head_dim);
    max_blocks_ = std::max(0, max_blocks);
    layers_.clear();
    layers_.resize(static_cast<size_t>(n_layers_));
    free_list_.clear();
}

KVBlock* PageTable::alloc_block(int layer) {
    if (layer < 0 || layer >= n_layers_) return nullptr;
    KVBlock* b = nullptr;
    const auto reusable = std::find_if(free_list_.begin(), free_list_.end(),
        [&](const KVBlock* block) { return block && block->layer == layer; });
    if (reusable != free_list_.end()) {
        b = *reusable;
        free_list_.erase(reusable);
        b->n_tokens = 0;
    } else {
        size_t allocated = 0;
        for (const auto& blocks : layers_) allocated += blocks.size();
        if (max_blocks_ > 0 && allocated >= static_cast<size_t>(max_blocks_)) return nullptr;
        layers_[layer].emplace_back();
        b = &layers_[layer].back();
        b->keys.assign(KVBlock::kBlockTokens * head_dim_, 0.f);
        b->values.assign(KVBlock::kBlockTokens * head_dim_, 0.f);
        b->layer = layer;
    }
    return b;
}

void PageTable::free_block(KVBlock* b) {
    if (!b) return;
    const bool belongs = std::any_of(layers_.begin(), layers_.end(), [&](const auto& blocks) {
        return std::any_of(blocks.begin(), blocks.end(), [&](const KVBlock& block) { return &block == b; });
    });
    if (!belongs || std::find(free_list_.begin(), free_list_.end(), b) != free_list_.end()) return;
    b->n_tokens = 0;
    free_list_.push_back(b);
}

void PageTable::prefetch_block(const KVBlock* b) {
    if (!b || b->keys.empty()) return;
    WINNER_PREFETCH(b->keys.data());
    WINNER_PREFETCH(b->keys.data() + b->keys.size() / 2);
    WINNER_PREFETCH(b->values.data());
}

void PageTable::flash_decode_scores(int layer, const float* query, int head_dim,
                                    float* out_scores, int n_scores_cap) {
    if (layer < 0 || layer >= n_layers_ || !query || !out_scores ||
        head_dim <= 0 || head_dim != head_dim_ || n_scores_cap <= 0) return;
    int written = 0;
    auto& blocks = layers_[layer];
    if (!blocks.empty()) prefetch_block(&blocks[0]);
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const KVBlock& b = blocks[bi];
        if (bi + 1 < blocks.size()) prefetch_block(&blocks[bi + 1]);
        for (int t = 0; t < b.n_tokens && written < n_scores_cap; ++t) {
            const float* k = b.keys.data() + t * head_dim;
            float dot = 0.f;
            int d = 0;
#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; d + 8 <= head_dim; d += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(query + d), _mm256_loadu_ps(k + d), acc);
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            dot = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#endif
            for (; d < head_dim; ++d) dot += query[d] * k[d];
            out_scores[written++] = dot / std::sqrt(float(head_dim));
        }
    }
}

int PageTable::n_blocks(int layer) const {
    if (layer < 0 || layer >= n_layers_) return 0;
    return int(layers_[layer].size());
}

size_t PageTable::bytes_used() const {
    size_t n = 0;
    for (auto& L : layers_)
        for (auto& b : L)
            n += (b.keys.size() + b.values.size()) * sizeof(float);
    return n;
}

} // namespace attention
} // namespace winner
