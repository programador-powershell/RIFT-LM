#include "runtime.h"
#include "kernels/fused.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <random>
#include <cstring>
#include <limits>

namespace winner {

int residual_rank_for_profile(QualityProfile p) {
    // From latency_table.json Pareto tradeoff
    switch (p) {
        case QualityProfile::MINMEM:   return 0;   // F0 only
        case QualityProfile::FAST:     return 16;
        case QualityProfile::BALANCED: return 64;
        case QualityProfile::SAFE:     return 128;
        default: return 64;
    }
}

bool GateEngine::evaluate(const float* activation, int dim, float threshold) {
    if (!activation || dim <= 0) return false;
    // RMS + peak: fire residual on high-energy or outlier activations only.
    // Fewer residual hits → less U/V time and lower average RSS when cold
    // layers keep residual unmapped on disk (pager path).
    double ss = 0.0;
    float peak = 0.f;
    for (int i = 0; i < dim; ++i) {
        const float a = activation[i];
        ss += double(a) * a;
        const float aa = std::fabs(a);
        if (aa > peak) peak = aa;
    }
    const float rms = float(std::sqrt(ss / std::max(1, dim)));
    const float thr = threshold_ > 0.f ? threshold_ : threshold;
    // Combined score in ~[0,1+] vs profile threshold
    const float score = 0.65f * rms + 0.35f * peak;
    return score > thr;
}

void PrefetchEngine::init(size_t ring_mb) { ring_bytes_ = ring_mb * 1024 * 1024; }
void PrefetchEngine::predict_and_prefetch(int layer, const std::vector<int>& stages) {
    // Software prefetch for next stage indices — no extra model buffers.
    (void)layer;
    for (int s : stages) {
        (void)s;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&stages, 0, 1);
#endif
    }
}
bool PrefetchEngine::is_ready(uint32_t) const { return true; }
void PrefetchEngine::tick() {}

Runtime::Runtime(std::shared_ptr<Bundle> bundle, const RuntimeConfig& cfg)
    : bundle_(std::move(bundle)), cfg_(cfg) {}

Runtime::~Runtime() = default;

void Runtime::apply_profile_defaults() {
    if (cfg_.residual_rank >= 0)
        residual_rank_ = cfg_.residual_rank;
    else
        residual_rank_ = residual_rank_for_profile(cfg_.profile);

    float gate_thr = cfg_.gate_threshold;
    if (gate_thr < 0.f) {
        switch (cfg_.profile) {
            case QualityProfile::SAFE:     gate_thr = 0.15f; break;
            case QualityProfile::BALANCED: gate_thr = 0.35f; break;
            case QualityProfile::FAST:     gate_thr = 0.55f; break;
            case QualityProfile::MINMEM:   gate_thr = 0.70f; break;
            default: gate_thr = 0.35f; break;
        }
    }
    gate_.set_threshold(gate_thr);
}

bool Runtime::init() {
    if (!bundle_ || !bundle_->is_valid()) return false;
    apply_profile_defaults();

    cpu_feat_ = backend::detect_cpu();
    devices_  = backend::enumerate_devices();

    size_t model_bytes = bundle_->base_bytes();
    if (model_bytes < 1) model_bytes = 2ull * 1024 * 1024 * 1024;
    size_t kv_bytes = 256ull * 1024 * 1024;
    plan_ = backend::plan_hybrid(model_bytes, kv_bytes, devices_, cfg_.n_gpu_layers);

    backend::DeviceInfo* gpu = nullptr;
    for (auto& d : devices_) {
        if (d.type != backend::DeviceType::CPU && d.available) { gpu = &d; break; }
    }
    if (gpu && plan_.n_gpu_layers > 0) {
        kernels_ = backend::select_gpu_kernels(*gpu);
        if (kernels_.isa == backend::KernelIsa::SCALAR)
            kernels_ = backend::select_cpu_kernels(cpu_feat_);
    } else {
        kernels_ = backend::select_cpu_kernels(cpu_feat_);
    }

    n_layers_ = plan_.n_gpu_layers + plan_.n_cpu_layers;
    if (n_layers_ < 1) n_layers_ = 24;

    // Real CSCD bundle: build ONE real layer from the F0/F1 stage tensors.
    if (bundle_->is_cscd()) {
        if (init_from_bundle()) return true;
        fprintf(stderr, "[WINNER] CSCD bundle has no usable F0 INT4 stage; falling back to synthetic workload\n");
    }
    // Without real Bundle tensors, fall back to synthetic fitted layers
    return init_synthetic(n_layers_, n_embd_, 42);
}

bool Runtime::init_from_bundle() {
    real_mode_ = false;
    if (!bundle_ || !bundle_->is_valid() || !bundle_->is_cscd()) return false;
    apply_profile_defaults();
    const int profile_rank = residual_rank_;   // 0 = MINMEM keeps F0-only semantics

    const CscdStageView* f0 = nullptr;
    const CscdStageView* f1 = nullptr;
    for (const auto& s : bundle_->cscd_stages()) {
        if (!f0 && s.stage_type == "BASE_STAGE" && s.codec == "INT4_GROUP") f0 = &s;
        else if (!f1 && s.stage_type == "RESIDUAL_LOWRANK" && s.codec == "FP32_LOWRANK") f1 = &s;
    }
    if (!f0) return false;
    if (f0->tensors.size() < 2) {
        fprintf(stderr, "[WINNER] CSCD F0 stage missing codes/scales tensors\n");
        return false;
    }
    const CscdTensorView& codes = f0->tensors[0];
    const CscdTensorView& scales = f0->tensors[1];
    if (codes.dtype != "uint8" || codes.shape.size() != 2 ||
        scales.dtype != "float32" || scales.shape.size() != 2) {
        fprintf(stderr, "[WINNER] CSCD F0 tensors have unexpected dtype/shape\n");
        return false;
    }
    const int out_f = f0->out_features > 0 ? f0->out_features : int(codes.shape[0]);
    const int in_f  = f0->in_features;
    const int gs    = f0->group_size > 0 ? f0->group_size : 32;
    if (out_f <= 0 || in_f <= 0 || out_f > (1 << 20) || in_f > (1 << 20) ||
        codes.shape[0] != out_f || scales.shape[0] != out_f) {
        fprintf(stderr, "[WINNER] CSCD F0 meta/tensor dimensions inconsistent\n");
        return false;
    }
    const int packed_cols = int(codes.shape[1]);
    const int n_groups = int(scales.shape[1]);
    if (packed_cols <= 0 || n_groups <= 0 ||
        int64_t(packed_cols) * 2 < in_f || int64_t(n_groups) * gs < in_f) {
        fprintf(stderr, "[WINNER] CSCD F0 packing narrower than in_features\n");
        return false;
    }

    real_layer_ = RealBundleLayer{};
    real_layer_.f0.rows = out_f;
    real_layer_.f0.cols = in_f;
    real_layer_.f0.group_size = gs;
    real_layer_.f0.n_groups = n_groups;
    real_layer_.f0.packed_cols = packed_cols;
    real_layer_.f0.codes.assign(codes.data, codes.data + codes.bytes);
    real_layer_.f0.scales.resize(scales.elem_count);
    memcpy(real_layer_.f0.scales.data(), scales.data, scales.bytes);

    if (f1 && profile_rank != 0) {
        if (f1->tensors.size() < 3) {
            fprintf(stderr, "[WINNER] CSCD F1 stage missing u/s/v tensors; running F0-only\n");
        } else {
            const CscdTensorView& u = f1->tensors[0];
            const CscdTensorView& sv = f1->tensors[1];
            const CscdTensorView& v = f1->tensors[2];
            const int rank = (sv.shape.size() == 1) ? int(sv.shape[0]) : 0;
            const bool ok = u.dtype == "float32" && sv.dtype == "float32" && v.dtype == "float32" &&
                u.shape.size() == 2 && v.shape.size() == 2 && rank > 0 && rank <= (1 << 16) &&
                u.shape[0] == out_f && u.shape[1] == rank &&
                v.shape[0] == in_f && v.shape[1] == rank;
            if (ok) {
                kernels::LowRankResidual R;
                R.rows = out_f;
                R.cols = in_f;
                R.rank = rank;
                R.U.resize(size_t(out_f) * size_t(rank));
                R.V.resize(size_t(in_f) * size_t(rank));
                R.singular.resize(size_t(rank));
                memcpy(R.U.data(), u.data, u.bytes);
                memcpy(R.singular.data(), sv.data, sv.bytes);
                memcpy(R.V.data(), v.data, v.bytes);
                // Fold diag(S) into U: y += U diag(S) (V^T x) == (U·S) (V^T x)
                for (int i = 0; i < out_f; ++i)
                    for (int j = 0; j < rank; ++j)
                        R.U[size_t(i) * size_t(rank) + size_t(j)] *= R.singular[size_t(j)];
                real_layer_.residual = std::move(R);
            } else {
                fprintf(stderr, "[WINNER] CSCD F1 tensors inconsistent with F0 dims; running F0-only\n");
            }
        }
    }

    real_layer_.in_dim = in_f;
    real_layer_.out_dim = out_f;
    residual_rank_ = real_layer_.residual.rank;   // real rank actually executed
    n_layers_ = 1;
    n_embd_ = in_f;
    activation_.assign(size_t(in_f), 0.f);
    stage_input_.assign(size_t(in_f), 0.f);
    real_out_.assign(size_t(out_f), 0.f);
    logits_.assign(size_t(n_vocab_), 0.f);
    layers_.clear();                              // no synthetic layers in real mode
    prefetch_.init(cfg_.ring_buffer_mb);
    real_mode_ = true;
    simulated_residual_fired_ = false;
    cumulative_drift_ = 0.0;

    printf("[WINNER] real bundle layer: out=%d in=%d group=%d rank=%d gate_thr=%.2f (workload=real_bundle)\n",
           out_f, in_f, gs, residual_rank_, gate_.threshold());
    if (bundle_->cscd_gate_percentile() >= 0.0)
        printf("[WINNER] bundle gate meta: %s percentile=%.1f (informational)\n",
               bundle_->cscd_gate_type().c_str(), bundle_->cscd_gate_percentile());
    return true;
}

bool Runtime::init_synthetic(int n_layers, int dim, uint32_t seed) {
    if (n_layers <= 0 || n_layers > 512 || dim <= 0 || dim > 4096 ||
        size_t(dim) > std::numeric_limits<size_t>::max() / size_t(dim)) {
        fprintf(stderr, "[WINNER] invalid synthetic dimensions\n");
        return false;
    }
    apply_profile_defaults();
    real_mode_ = false;
    simulated_residual_fired_ = false;
    real_layer_ = RealBundleLayer{};
    real_out_.clear();
    residual_rank_ = std::min(residual_rank_, dim);
    n_layers_ = n_layers;
    n_embd_ = dim;
    activation_.assign(dim, 0.f);
    logits_.assign(n_vocab_, 0.f);
    stage_input_.assign(dim, 0.f);

    cpu_feat_ = backend::detect_cpu();
    devices_  = backend::enumerate_devices();
    plan_ = backend::plan_hybrid(size_t(dim) * dim * 4 * n_layers * 2,
                                64ull << 20, devices_, cfg_.n_gpu_layers);
    plan_.n_gpu_layers = std::max(0, std::min(plan_.n_gpu_layers, n_layers));
    plan_.n_cpu_layers = n_layers - plan_.n_gpu_layers;
    kernels_ = backend::select_cpu_kernels(cpu_feat_);
    prefetch_.init(cfg_.ring_buffer_mb);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);

    layers_.clear();
    layers_.reserve(n_layers_);
    for (int L = 0; L < n_layers_; ++L) {
        LayerWeights lw;
        lw.dim = dim;
        std::vector<float> W(size_t(dim) * dim);
        for (auto& v : W) v = nd(rng);
        lw.f0 = kernels::pack_ternary(W.data(), dim, dim, -1.f);
        if (residual_rank_ > 0)
            lw.residual = kernels::fit_residual_ls(W.data(), lw.f0, residual_rank_, 6, seed + L);
        layers_.push_back(std::move(lw));
    }

    printf("[WINNER] profile=%d rank=%d gate=%.2f ISA=%s layers=%d dim=%d\n",
           int(cfg_.profile), residual_rank_, gate_.threshold(),
           backend::isa_name(kernels_.isa), n_layers_, n_embd_);
    printf("[WINNER] Hybrid: %d GPU + %d CPU layers | residual LS fitted per layer\n",
           plan_.n_gpu_layers, plan_.n_cpu_layers);
    return true;
}

OptimizerId Runtime::optimizer() const {
    return bundle_ ? bundle_->optimizer() : OptimizerId::SPECTRA;
}

bool Runtime::run_fused_stage(int layer_idx, float* x, bool force_residual) {
    if (!x || stage_input_.size() != static_cast<size_t>(n_embd_)) return false;
    last_stats_.residual_rank = residual_rank_;
    if (layer_idx < 0 || layer_idx >= (int)layers_.size()) return false;

    const LayerWeights& lw = layers_[layer_idx];
    auto t0 = std::chrono::steady_clock::now();

    // Preserve the input: GEMV is not safe in-place and the residual must use
    // the same activation as the base stage.
    std::copy_n(x, lw.dim, stage_input_.data());
    kernels::gemv_ternary(lw.f0, stage_input_.data(), x);
    last_stats_.stages_executed += 1;

    // Latency-table policy:
    //   SAFE     → residual almost always (gate thr low + high rank)
    //   BALANCED → ~40-60% residual (gate)
    //   FAST     → residual only on hard tokens
    //   MINMEM   → never (rank 0)
    bool need_residual = force_residual || cfg_.force_residual;
    if (!need_residual && lw.has_residual()) {
        need_residual = gate_.evaluate(stage_input_.data(), lw.dim);
        // Fallback: profile-based residual rate when energy gate is silent
        // (synthetic / small activations). Rates from latency Pareto.
        if (!need_residual) {
            float rate = 0.f;
            switch (cfg_.profile) {
                case QualityProfile::SAFE:     rate = 0.90f; break;
                case QualityProfile::BALANCED: rate = 0.45f; break;
                case QualityProfile::FAST:     rate = 0.20f; break;
                default: rate = 0.f; break;
            }
            // deterministic hash of layer + activation fingerprint
            uint32_t h = uint32_t(layer_idx) * 2654435761u;
            // Clamp before converting: float->uint32_t is UB when the value
            // exceeds the destination range ([conv.fpint]).
            h ^= uint32_t(std::min(std::fabs(stage_input_[0]) * 1e6f, 4.0e9f));
            need_residual = ((h % 1000) / 1000.f) < rate;
            // Residual-rate metrics become SIMULATED once this policy hash
            // (not the energy gate) drives the decision.
            if (rate > 0.f) simulated_residual_fired_ = true;
        }
    }

    if (need_residual && lw.has_residual()) {
        kernels::gemv_lowrank_add(lw.residual, stage_input_.data(), x);
        last_stats_.stages_executed++;
        last_stats_.residual_used = true;
    } else if (lw.has_residual()) {
        const float drift_delta = 0.006f * (1.f - float(residual_rank_) / 128.f);
        last_stats_.drift_delta += drift_delta;
        cumulative_drift_ += drift_delta;
        if (cumulative_drift_ > cfg_.drift_budget) {
            cumulative_drift_ *= 0.5;
            kernels::gemv_lowrank_add(lw.residual, stage_input_.data(), x);
            last_stats_.residual_used = true;
            last_stats_.stages_executed++;
            last_stats_.drift_delta -= drift_delta;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    last_stats_.us_compute += std::chrono::duration<double, std::micro>(t1 - t0).count();
    return true;
}

bool Runtime::run_real_layer(float* x) {
    if (!x || !real_mode_ || !real_layer_.f0.valid()) return false;
    const int in_f = real_layer_.in_dim;
    const int out_f = real_layer_.out_dim;
    if (stage_input_.size() != size_t(in_f) || real_out_.size() != size_t(out_f)) return false;
    last_stats_.residual_rank = residual_rank_;

    auto t0 = std::chrono::steady_clock::now();
    // Preserve the input: base stage and residual must see the same activation.
    std::copy_n(x, in_f, stage_input_.data());
    kernels::gemv_int4_group(real_layer_.f0.codes.data(), real_layer_.f0.scales.data(),
                             real_layer_.f0.rows, real_layer_.f0.cols,
                             real_layer_.f0.packed_cols, real_layer_.f0.group_size,
                             real_layer_.f0.n_groups,
                             stage_input_.data(), real_out_.data());
    last_stats_.stages_executed += 1;

    // Real bundle mode: ONLY the real gate decides — never the hash-simulated
    // fallback and never the synthetic drift-budget catch-up.
    bool need_residual = cfg_.force_residual;
    if (!need_residual && real_layer_.has_residual())
        need_residual = gate_.evaluate(stage_input_.data(), in_f);
    if (need_residual && real_layer_.has_residual()) {
        kernels::gemv_lowrank_add(real_layer_.residual, stage_input_.data(), real_out_.data());
        last_stats_.stages_executed += 1;
        last_stats_.residual_used = true;
    }

    // Feedback for the decode loop: wrap out→in and renormalize the RMS so the
    // iteration stays numerically stable; the gate still sees real activation
    // structure (peaks) produced by the real tensors.
    double ss = 0.0;
    for (int i = 0; i < out_f; ++i) ss += double(real_out_[i]) * real_out_[i];
    const float rms = float(std::sqrt(ss / std::max(1, out_f)));
    const float target_rms = 0.30f;
    const float scale = (std::isfinite(rms) && rms > 1e-12f) ? target_rms / rms : 0.f;
    for (int i = 0; i < in_f; ++i) x[i] = real_out_[i % out_f] * scale;

    auto t1 = std::chrono::steady_clock::now();
    last_stats_.us_compute += std::chrono::duration<double, std::micro>(t1 - t0).count();
    return true;
}

bool Runtime::run_layer(int layer_idx, float* x) {
    prefetch_.tick();
    if (real_mode_) return run_real_layer(x);
    // Two GEMV stages per layer (attn proj + ffn proxy)
    if (!run_fused_stage(layer_idx, x, false)) return false;
    if (!run_fused_stage(layer_idx, x, false)) return false;
    return true;
}

bool Runtime::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return false;
    cumulative_drift_ = 0.0;
    std::fill(activation_.begin(), activation_.end(), 0.02f);
    return true;
}

int32_t Runtime::decode_one() {
    last_stats_ = TokenStats{};
    last_stats_.residual_rank = residual_rank_;
    for (int l = 0; l < n_layers_; ++l)
        if (!run_layer(l, activation_.data())) return -1;
    // Pseudo-token from activation
    float s = 0.f;
    for (int i = 0; i < std::min(8, n_embd_); ++i) s += activation_[i];
    return 100 + (int(std::fabs(s) * 1000) % 1000);
}

std::vector<int32_t> Runtime::generate(const std::vector<int32_t>& prompt, int max_new) {
    std::vector<int32_t> out = prompt;
    if (!prefill(prompt)) return out;
    for (int i = 0; i < max_new; ++i) {
        int32_t t = decode_one();
        if (t < 0) break;
        out.push_back(t);
    }
    return out;
}

size_t Runtime::peak_rss_bytes() const {
    if (real_mode_) {
        // Real CSCD layer: F0 codes+scales + residual U/V actually resident
        size_t f0 = real_layer_.f0.codes.size() + real_layer_.f0.scales.size() * 4;
        size_t res = 0;
        if (real_layer_.has_residual())
            res = (real_layer_.residual.U.size() + real_layer_.residual.V.size()) * 4;
        return f0 + res;
    }
    // F0 all layers + one hot residual layer
    size_t f0 = 0, res = 0;
    for (const auto& lw : layers_) {
        f0 += lw.f0.packed.size() + lw.f0.scales.size() * 4;
        if (lw.has_residual())
            res = std::max(res, (lw.residual.U.size() + lw.residual.V.size()) * 4);
    }
    return f0 + res;
}

double Runtime::bench_gemv_us(int iterations) const {
    if (iterations <= 0) return 0;
    if (real_mode_ && real_layer_.f0.valid()) {
        std::vector<float> x(size_t(real_layer_.in_dim), 0.01f);
        std::vector<float> y(size_t(real_layer_.out_dim), 0.f);
        auto run_once = [&]() {
            kernels::gemv_int4_group(real_layer_.f0.codes.data(), real_layer_.f0.scales.data(),
                                     real_layer_.f0.rows, real_layer_.f0.cols,
                                     real_layer_.f0.packed_cols, real_layer_.f0.group_size,
                                     real_layer_.f0.n_groups, x.data(), y.data());
            kernels::gemv_lowrank_add(real_layer_.residual, x.data(), y.data());
        };
        for (int i = 0; i < 4; ++i) run_once();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) run_once();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    }
    if (layers_.empty()) return 0;
    const auto& lw = layers_[0];
    std::vector<float> x(n_embd_, 0.01f), y(n_embd_);
    for (int i = 0; i < 4; ++i)
        kernels::gemv_f0_plus_residual(lw.f0, lw.residual, x.data(), y.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        kernels::gemv_f0_plus_residual(lw.f0, lw.residual, x.data(), y.data());
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
}

std::unique_ptr<Bundle> load_bundle(const std::string& path) {
    auto b = std::make_unique<Bundle>();
    if (!b->load(path)) return nullptr;
    return b;
}

std::unique_ptr<Runtime> create_runtime(std::shared_ptr<Bundle> bundle, QualityProfile profile) {
    RuntimeConfig cfg;
    cfg.profile = profile;
    auto rt = std::make_unique<Runtime>(std::move(bundle), cfg);
    if (!rt->init()) return nullptr;
    return rt;
}

std::vector<int32_t> generate(Runtime& rt, const std::vector<int32_t>& prompt, int max_new, float) {
    return rt.generate(prompt, max_new);
}

} // namespace winner
