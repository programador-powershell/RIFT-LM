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

namespace {
void quantize_vec(const float* src, int n, int8_t* dst, float& scale) {
    float amax = 0.f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    scale = amax > 1e-12f ? amax / 127.f : 1.f;
    const float inv = 1.f / scale;
    for (int i = 0; i < n; ++i) {
        const int v = static_cast<int>(std::lrintf(src[i] * inv));
        dst[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
    }
}
} // namespace

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
        layers_[static_cast<size_t>(layer)].emplace_back();
        b = &layers_[static_cast<size_t>(layer)].back();
        b->keys_q.assign(static_cast<size_t>(KVBlock::kBlockTokens * head_dim_), 0);
        b->values_q.assign(static_cast<size_t>(KVBlock::kBlockTokens * head_dim_), 0);
        b->layer = layer;
        b->n_tokens = 0;
        b->key_scale = 1.f;
        b->value_scale = 1.f;
    }
    return b;
}

void PageTable::free_block(KVBlock* b) {
    if (!b) return;
    b->n_tokens = 0;
    free_list_.push_back(b);
}

bool PageTable::append_kv(KVBlock* b, const float* key, const float* value, int head_dim) {
    if (!b || !key || !value || b->n_tokens >= KVBlock::kBlockTokens) return false;
    const int slot = b->n_tokens * head_dim;
    // Recompute block scales cheaply from this token (running max)
    float kmax = b->key_scale * 127.f, vmax = b->value_scale * 127.f;
    for (int i = 0; i < head_dim; ++i) {
        kmax = std::max(kmax, std::fabs(key[i]));
        vmax = std::max(vmax, std::fabs(value[i]));
    }
    b->key_scale = kmax > 1e-12f ? kmax / 127.f : 1.f;
    b->value_scale = vmax > 1e-12f ? vmax / 127.f : 1.f;
    const float kinv = 1.f / b->key_scale;
    const float vinv = 1.f / b->value_scale;
    for (int i = 0; i < head_dim; ++i) {
        int kv = static_cast<int>(std::lrintf(key[i] * kinv));
        int vv = static_cast<int>(std::lrintf(value[i] * vinv));
        b->keys_q[static_cast<size_t>(slot + i)] = static_cast<int8_t>(std::max(-127, std::min(127, kv)));
        b->values_q[static_cast<size_t>(slot + i)] = static_cast<int8_t>(std::max(-127, std::min(127, vv)));
    }
    b->n_tokens += 1;
    return true;
}

void PageTable::prefetch_block(const KVBlock* b) {
    if (!b) return;
    if (!b->keys_q.empty()) WINNER_PREFETCH(b->keys_q.data());
    if (!b->values_q.empty()) WINNER_PREFETCH(b->values_q.data());
}

void PageTable::flash_decode_scores(int layer, const float* query, int head_dim,
                                    float* out_scores, int n_scores_cap) {
    if (layer < 0 || layer >= n_layers_ || !query || !out_scores || n_scores_cap <= 0) return;
    int written = 0;
    for (const auto& block : layers_[static_cast<size_t>(layer)]) {
        prefetch_block(&block);
        for (int t = 0; t < block.n_tokens && written < n_scores_cap; ++t) {
            const int8_t* k = block.keys_q.data() + t * head_dim;
            float score = 0.f;
            for (int d = 0; d < head_dim; ++d)
                score += query[d] * (float(k[d]) * block.key_scale);
            out_scores[written++] = score;
        }
    }
    for (; written < n_scores_cap; ++written) out_scores[written] = 0.f;
}

int PageTable::n_blocks(int layer) const {
    if (layer < 0 || layer >= n_layers_) return 0;
    return static_cast<int>(layers_[static_cast<size_t>(layer)].size());
}

size_t PageTable::bytes_used() const {
    size_t n = 0;
    for (const auto& layer : layers_) {
        for (const auto& b : layer) {
            n += b.keys_q.size() * sizeof(int8_t);
            n += b.values_q.size() * sizeof(int8_t);
            n += sizeof(float) * 2;
        }
    }
    return n;
}

} // namespace attention
} // namespace winner
