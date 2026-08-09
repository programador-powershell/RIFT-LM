#pragma once
#include "winner.h"
#include <vector>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace winner {

enum class ReqState : uint8_t {
    WAITING = 0, PREFILL, DECODE, FINISHED, ABORTED
};

struct Request {
    uint64_t id = 0;
    std::vector<int32_t> prompt;
    std::vector<int32_t> output;
    int max_new_tokens = 128;
    float temperature = 0.7f;
    ReqState state = ReqState::WAITING;
    int n_computed = 0;
    std::vector<int> block_table;
};

class ContinuousBatcher {
public:
    void init(int max_batch_tokens = 2048, int max_seqs = 32);
    uint64_t submit(const std::vector<int32_t>& prompt, int max_new, float temperature = 0.7f);

    struct StepBatch {
        std::vector<Request*> prefills;
        std::vector<Request*> decodes;
        int total_tokens = 0;
    };
    StepBatch schedule();
    void on_token(Request& req, int32_t token);
    std::vector<Request*> finished();
    size_t pending() const;

private:
    int max_batch_tokens_ = 2048;
    int max_seqs_ = 32;
    uint64_t next_id_ = 1;
    std::deque<Request> waiting_;
    std::deque<Request> running_;
    std::deque<Request> done_;
    mutable std::mutex mu_;
};

} // namespace winner
