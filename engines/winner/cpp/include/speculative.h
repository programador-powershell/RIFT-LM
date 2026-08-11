#pragma once
/**
 * Tree-based speculative decoding
 * Expected: +80% to +150% tok/s depending on draft quality
 *
 * Draft model expands a tree (top-k branches) instead of a linear chain.
 * Target model verifies the whole tree in one forward, accepting the
 * longest valid path → higher tokens accepted per iteration.
 */
#include <cstdint>
#include <utility>
#include <vector>
#include <functional>

namespace winner {

struct SpecNode {
    int32_t token = 0;
    float logprob = 0.f;
    int parent = -1;
    int depth = 0;
    std::vector<int> children;
};

struct SpecTree {
    std::vector<SpecNode> nodes; // nodes[0] = root (last confirmed token)
    int max_depth = 0;
    int max_branch = 0;
};

struct SpecConfig {
    int draft_depth = 4;     // tree depth
    int branch_factor = 2;  // top-k per node
    float draft_temp = 0.8f;
    bool enable = true;
};

class SpeculativeEngine {
public:
    void init(const SpecConfig& cfg);
    /** Build draft tree from a draft_fn that returns top-k (token, logprob) pairs */
    SpecTree draft_tree(
        int32_t root_token,
        std::function<std::vector<std::pair<int32_t,float>>(int32_t, int)> draft_topk);

    /**
     * Verify tree against target scores.
     * target_logprob(token_id, position_in_path) → logprob under target model.
     * Returns accepted token sequence (excluding root).
     */
    std::vector<int32_t> verify_tree(
        const SpecTree& tree,
        std::function<float(int32_t token, int depth)> target_logprob);

    const SpecConfig& config() const { return cfg_; }
    int last_accepted() const { return last_accepted_; }
    float accept_rate() const;

private:
    SpecConfig cfg_;
    int total_drafted_ = 0;
    int total_accepted_ = 0;
    int last_accepted_ = 0;
};

} // namespace winner
