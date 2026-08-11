#pragma once
/**
 * Paged KV + FlashDecoding-style parallel attention over blocks.
 * KV content stored as INT8 + per-block scale to cut bandwidth/footprint ~2×
 * without increasing weight RSS.
 */
#include <cstdint>
#include <vector>
#include <deque>
#include <cstring>

namespace winner {
namespace attention {

struct KVBlock {
    static constexpr int kBlockTokens = 16;
    std::vector<int8_t> keys_q;    // quantized keys
    std::vector<int8_t> values_q;  // quantized values
    float key_scale = 1.f;
    float value_scale = 1.f;
    int n_tokens = 0;
    int layer = 0;
};

class PageTable {
public:
    void init(int n_layers, int n_heads, int head_dim, int max_blocks);
    KVBlock* alloc_block(int layer);
    void free_block(KVBlock* b);
    /** Quantize and append one token's key/value (head_dim floats each). */
    bool append_kv(KVBlock* b, const float* key, const float* value, int head_dim);
    void flash_decode_scores(int layer, const float* query, int head_dim,
                             float* out_scores, int n_scores_cap);
    void prefetch_block(const KVBlock* b);
    int n_blocks(int layer) const;
    size_t bytes_used() const;

private:
    int n_layers_ = 0, n_heads_ = 0, head_dim_ = 0, max_blocks_ = 0;
    std::vector<std::deque<KVBlock>> layers_;
    std::vector<KVBlock*> free_list_;
};

} // namespace attention
} // namespace winner
