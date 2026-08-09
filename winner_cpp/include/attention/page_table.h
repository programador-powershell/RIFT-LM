#pragma once
/**
 * Paged KV + FlashDecoding-style parallel attention over blocks
 * Expected: +30–50% tok/s on long context
 */
#include <cstdint>
#include <vector>
#include <deque>
#include <cstring>

namespace winner {
namespace attention {

struct KVBlock {
    static constexpr int kBlockTokens = 16;
    std::vector<float> keys;
    std::vector<float> values;
    int n_tokens = 0;
    int layer = 0;
};

class PageTable {
public:
    void init(int n_layers, int n_heads, int head_dim, int max_blocks);
    KVBlock* alloc_block(int layer);
    void free_block(KVBlock* b);
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
