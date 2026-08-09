#include "speculative.h"
#include <algorithm>
#include <cmath>
#include <queue>

namespace winner {

void SpeculativeEngine::init(const SpecConfig& cfg) {
    cfg_ = cfg;
    cfg_.draft_depth = std::max(0, std::min(cfg_.draft_depth, 32));
    cfg_.branch_factor = std::max(0, std::min(cfg_.branch_factor, 64));
    total_drafted_ = total_accepted_ = last_accepted_ = 0;
}

SpecTree SpeculativeEngine::draft_tree(
    int32_t root_token,
    std::function<std::vector<std::pair<int32_t,float>>(int32_t, int)> draft_topk) {

    SpecTree tree;
    tree.max_depth = cfg_.draft_depth;
    tree.max_branch = cfg_.branch_factor;
    tree.nodes.push_back({root_token, 0.f, -1, 0, {}});
    if (!draft_topk || cfg_.draft_depth == 0 || cfg_.branch_factor == 0) return tree;

    // BFS expand
    std::queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int idx = q.front(); q.pop();
        const int node_depth = tree.nodes[idx].depth;
        const int32_t node_token = tree.nodes[idx].token;
        if (node_depth >= cfg_.draft_depth) continue;

        auto topk = draft_topk(node_token, cfg_.branch_factor);
        if (topk.size() > static_cast<size_t>(cfg_.branch_factor)) {
            topk.resize(static_cast<size_t>(cfg_.branch_factor));
        }
        for (auto& [tok, lp] : topk) {
            SpecNode child;
            child.token = tok;
            child.logprob = lp;
            child.parent = idx;
            child.depth = node_depth + 1;
            int child_idx = int(tree.nodes.size());
            tree.nodes[idx].children.push_back(child_idx);
            tree.nodes.push_back(child);
            q.push(child_idx);
            total_drafted_++;
        }
    }
    return tree;
}

std::vector<int32_t> SpeculativeEngine::verify_tree(
    const SpecTree& tree,
    std::function<float(int32_t token, int depth)> target_logprob) {

    if (tree.nodes.empty() || !target_logprob) return {};

    // Find longest path from root where every node would be sampled
    // under a simple acceptance rule: target_logprob >= draft_logprob - margin
    // For scaffolding we accept nodes with target_lp > -10 (always-ish) and
    // prefer highest cumulative target score path.

    struct PathScore {
        std::vector<int> node_indices;
        float score = 0.f;
    };

    PathScore best;
    std::function<void(int, PathScore)> dfs = [&](int idx, PathScore cur) {
        if (idx != 0) {
            const SpecNode& n = tree.nodes[idx];
            float tlp = target_logprob(n.token, n.depth);
            // accept if target doesn't strongly disagree
            if (tlp < n.logprob - 5.f) return; // reject branch
            cur.node_indices.push_back(idx);
            cur.score += tlp;
            if (cur.node_indices.size() > best.node_indices.size() ||
                (cur.node_indices.size() == best.node_indices.size() && cur.score > best.score)) {
                best = cur;
            }
        }
        for (int c : tree.nodes[idx].children) {
            if (c >= 0 && c < static_cast<int>(tree.nodes.size())) dfs(c, cur);
        }
    };
    dfs(0, {});

    std::vector<int32_t> accepted;
    for (int ni : best.node_indices)
        accepted.push_back(tree.nodes[ni].token);
    last_accepted_ = int(accepted.size());
    total_accepted_ += last_accepted_;
    return accepted;
}

float SpeculativeEngine::accept_rate() const {
    if (total_drafted_ <= 0) return 0.f;
    return float(total_accepted_) / float(total_drafted_);
}

} // namespace winner
