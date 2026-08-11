#include "sched/batching.h"
#include <algorithm>

namespace winner {

void ContinuousBatcher::init(int max_batch_tokens, int max_seqs) {
    max_batch_tokens_ = max_batch_tokens;
    max_seqs_ = max_seqs;
}

uint64_t ContinuousBatcher::submit(const std::vector<int32_t>& prompt, int max_new, float temperature) {
    std::lock_guard<std::mutex> lk(mu_);
    Request r;
    r.id = next_id_++;
    r.prompt = prompt;
    r.max_new_tokens = max_new;
    r.temperature = temperature;
    r.state = ReqState::WAITING;
    waiting_.push_back(std::move(r));
    return waiting_.back().id;
}

ContinuousBatcher::StepBatch ContinuousBatcher::schedule() {
    std::lock_guard<std::mutex> lk(mu_);
    StepBatch batch;
    int budget = max_batch_tokens_;

    // Prefer ongoing decodes (1 token each)
    for (auto& r : running_) {
        if (r.state == ReqState::DECODE && budget > 0) {
            batch.decodes.push_back(&r);
            batch.total_tokens += 1;
            budget -= 1;
        }
    }

    // Admit waiting prefills if room
    while (!waiting_.empty() && (int)running_.size() < max_seqs_ && budget > 0) {
        Request r = std::move(waiting_.front());
        waiting_.pop_front();
        int need = (int)r.prompt.size();
        if (need > budget && !batch.prefills.empty()) break;
        r.state = ReqState::PREFILL;
        running_.push_back(std::move(r));
        batch.prefills.push_back(&running_.back());
        batch.total_tokens += need;
        budget -= need;
    }
    return batch;
}

void ContinuousBatcher::on_token(Request& req, int32_t token) {
    req.output.push_back(token);
    req.n_computed++;
    if (req.state == ReqState::PREFILL) req.state = ReqState::DECODE;
    if ((int)req.output.size() >= req.max_new_tokens) {
        req.state = ReqState::FINISHED;
    }
}

std::vector<Request*> ContinuousBatcher::finished() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<Request*> out;
    for (auto it = running_.begin(); it != running_.end(); ) {
        if (it->state == ReqState::FINISHED) {
            done_.push_back(std::move(*it));
            it = running_.erase(it);
            out.push_back(&done_.back());
        } else ++it;
    }
    return out;
}

size_t ContinuousBatcher::pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return waiting_.size() + running_.size();
}

} // namespace winner
