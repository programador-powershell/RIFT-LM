#pragma once
#include "winner.h"
#include "bundle.h"
#include "backend/cpu_detect.h"
#include "backend/device.h"
#include "backend/kernels.h"
#include "kernels/residual_ls.h"
#include <vector>
#include <memory>

namespace winner {

/**
 * Quality profiles mapped from latency table (dim=512 proxy):
 *   MINMEM   → F0 only          (~6.7x vs Q4, cos~0.65, lowest RSS)
 *   FAST     → F0 + LS rank 16  (~5.8x, cos~0.69)
 *   BALANCED → F0 + LS rank 64  (~3.5x, cos~0.78)
 *   SAFE     → F0 + LS rank 128 (~2.4x, cos~0.86)
 * Gate can still skip residual per-token when confidence is high.
 */
struct RuntimeConfig {
    QualityProfile profile = QualityProfile::BALANCED;
    int   n_threads = 0;
    bool  enable_speculative = true;
    int   draft_length = 4;
    float drift_budget = 0.12f;
    bool  enable_prefetch = true;
    size_t ring_buffer_mb = 320;
    int   n_gpu_layers = -1;   // -1 = auto hybrid
    int   residual_rank = -1;  // -1 = from profile; 0 = F0 only
    float gate_threshold = -1.f; // -1 = from profile
    bool  force_residual = false; // ignore gate, always apply LS
};

struct TokenStats {
    int   stages_executed = 0;
    float drift_delta = 0.f;
    bool  residual_used = false;
    int   residual_rank = 0;
    double us_compute = 0;
};

class GateEngine {
public:
    bool evaluate(const float* activation, int dim, float threshold = 0.35f);
    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }
private:
    float threshold_ = 0.35f;
};

class PrefetchEngine {
public:
    void init(size_t ring_mb);
    void predict_and_prefetch(int layer, const std::vector<int>& stages);
    bool is_ready(uint32_t stage_id) const;
    void tick();
private:
    size_t ring_bytes_ = 0;
};

/** One transformer-layer weight slice in progressive form */
struct LayerWeights {
    kernels::TernaryMatrix f0;
    kernels::LowRankResidual residual;
    int dim = 0;
    bool has_residual() const { return residual.rank > 0; }
};

/** F0 INT4 groupwise matrix (CSCD real bundle) — data copied from the bundle */
struct Int4GroupMatrix {
    int rows = 0, cols = 0;            // out_features, in_features
    int group_size = 32;
    int n_groups = 0;                  // scale groups per row
    int packed_cols = 0;               // bytes per row (2 weights/byte)
    std::vector<uint8_t> codes;        // rows × packed_cols
    std::vector<float>   scales;       // rows × n_groups
    bool valid() const { return rows > 0 && cols > 0 && !codes.empty() && !scales.empty(); }
};

/** ONE real layer loaded from a CSCD bundle: F0 INT4 + low-rank residual F1 */
struct RealBundleLayer {
    Int4GroupMatrix f0;
    kernels::LowRankResidual residual; // U already scaled by diag(S)
    int in_dim = 0, out_dim = 0;
    bool has_residual() const { return residual.rank > 0; }
};

class Runtime {
public:
    Runtime(std::shared_ptr<Bundle> bundle, const RuntimeConfig& cfg);
    ~Runtime();

    bool init();
    /** Synthetic init for kernel bench without Bundle (uses random W fitted with LS) */
    bool init_synthetic(int n_layers, int dim, uint32_t seed = 42);
    /** Init from real CSCD F0/F1 stage tensors (called by init() when available) */
    bool init_from_bundle();

    /** true when decode runs the real CSCD layer (workload=real_bundle) */
    bool real_bundle_active() const { return real_mode_; }
    /** true if the hash-simulated residual fallback drove residual decisions */
    bool simulated_residual_fired() const { return simulated_residual_fired_; }
    const char* workload_label() const { return real_mode_ ? "real_bundle" : "synthetic"; }

    bool prefill(const std::vector<int32_t>& tokens);
    int32_t decode_one();
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new);

    const TokenStats& last_token_stats() const { return last_stats_; }
    double cumulative_drift() const { return cumulative_drift_; }
    size_t peak_rss_bytes() const;

    const RuntimeConfig& config() const { return cfg_; }
    OptimizerId optimizer() const;

    const backend::CpuFeatures& cpu_features() const { return cpu_feat_; }
    const backend::HybridPlan& hybrid_plan() const { return plan_; }
    const std::vector<backend::DeviceInfo>& devices() const { return devices_; }
    backend::KernelIsa active_isa() const { return kernels_.isa; }

    int residual_rank() const { return residual_rank_; }
    int n_layers() const { return n_layers_; }
    int dim() const { return n_embd_; }
    const std::vector<LayerWeights>& layers() const { return layers_; }

    /** Micro-bench one fused GEMV path (for --bench-kernels) */
    double bench_gemv_us(int iterations = 64) const;

private:
    std::shared_ptr<Bundle> bundle_;
    RuntimeConfig cfg_;
    GateEngine gate_;
    PrefetchEngine prefetch_;

    backend::CpuFeatures cpu_feat_;
    std::vector<backend::DeviceInfo> devices_;
    backend::HybridPlan plan_;
    backend::KernelTable kernels_;

    std::vector<LayerWeights> layers_;
    std::vector<float> activation_;
    std::vector<float> logits_;
    std::vector<float> stage_input_;
    int n_layers_ = 24;
    int n_embd_ = 512;
    int n_vocab_ = 32000;
    int residual_rank_ = 64;

    RealBundleLayer real_layer_;
    std::vector<float> real_out_;
    bool real_mode_ = false;
    bool simulated_residual_fired_ = false;

    TokenStats last_stats_;
    double cumulative_drift_ = 0.0;

    void apply_profile_defaults();
    bool run_layer(int layer_idx, float* x);
    bool run_fused_stage(int layer_idx, float* x, bool force_residual);
    bool run_real_layer(float* x);
};

int residual_rank_for_profile(QualityProfile p);

} // namespace winner
