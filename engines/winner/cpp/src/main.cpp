/**
 * winner.cpp v0.4 — Progressive inference with F0 + least-squares residual
 * Profiles from latency table: MINMEM/FAST/BALANCED/SAFE → rank 0/16/64/128
 */
#include "winner.h"
#include "bundle.h"
#include "runtime.h"
#include "backend/cpu_detect.h"
#include "backend/device.h"
#include "backend/vcpu.h"
#include "backend/multi_gpu.h"
#include "quant/kquant.h"
#include "backend/kernels.h"
#include "kernels/residual_ls.h"
#include "prefetch_uring.h"
#include "attention/page_table.h"
#include "sched/batching.h"
#include "speculative.h"
#if defined(WINNER_HTTP_SERVER)
#  include "server/http_server.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <fstream>
#include <cerrno>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace winner;

static void print_banner() {
    printf("WINNER.cpp v%u.%u.%u — F0-2bit + LS residual AVX2 (Gemma4-12B + VCpu + MoE LFU)\n",
           WINNER_VERSION_MAJOR, WINNER_VERSION_MINOR, WINNER_VERSION_PATCH);
    printf("Profiles: MINMEM=r0 FAST=r16 BALANCED=r64 SAFE=r128\n\n");
}

static const char* profile_name(QualityProfile p) {
    switch (p) {
        case QualityProfile::SAFE: return "SAFE";
        case QualityProfile::BALANCED: return "BALANCED";
        case QualityProfile::FAST: return "FAST";
        case QualityProfile::MINMEM: return "MINMEM";
        default: return "?";
    }
}

static void print_devices() {
    auto feat = backend::detect_cpu();
    auto devs = backend::enumerate_devices();
    printf("=== Hardware ===\n");
    printf("CPU: %s | ISA=%s score=%d cores=%d\n",
           feat.brand.c_str(), backend::best_cpu_isa_name(feat),
           backend::cpu_score(feat), feat.n_cores);
    for (const auto& d : devs) {
        printf("  [%s] %s  %.1f GB\n",
               d.type == backend::DeviceType::CPU ? "CPU" : "GPU",
               d.name.c_str(), d.total_memory_bytes/(1024.*1024*1024));
    }
    printf("\n");
}

static bool parse_int_arg(const char* text, int minimum, int maximum, int& out) {
    if (!text || !*text) return false;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < minimum || value > maximum) return false;
    out = static_cast<int>(value);
    return true;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

static int run_self_test(const std::string& bundle_path) {
    try {
        constexpr int dim = 8;
        std::vector<float> weights(dim * dim), x(dim), dense(dim), candidate(dim);
        for (int i = 0; i < dim * dim; ++i) weights[i] = std::sin(float(i + 1)) * 0.05f;
        for (int i = 0; i < dim; ++i) x[i] = std::cos(float(i + 1)) * 0.1f;
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) dense[row] += weights[row * dim + col] * x[col];
        }
        const auto f0 = kernels::pack_ternary(weights.data(), dim, dim, -1.f);
        const auto residual = kernels::fit_residual_ls(weights.data(), f0, 4, 4, 7);
        kernels::gemv_f0_plus_residual(f0, residual, x.data(), candidate.data());
        const float cosine = kernels::cosine_similarity(dense.data(), candidate.data(), dim);
        if (!std::isfinite(cosine) || cosine < 0.5f) throw std::runtime_error("residual quality check failed");

        const auto q4 = quant::pack_kquant(weights.data(), dim, dim, quant::KQuantType::Q4_0);
        const size_t blocks_per_row = (dim + 31u) / 32u;
        if (q4.data.size() != size_t(dim) * blocks_per_row * 20u) {
            throw std::runtime_error("Q4 payload size check failed");
        }

        SpeculativeEngine speculative;
        speculative.init({2, 2});
        const auto tree = speculative.draft_tree(1, [](int32_t token, int) {
            return std::vector<std::pair<int32_t, float>>{{token + 1, -0.1f}, {token + 2, -0.2f}};
        });
        if (tree.nodes.size() != 7) throw std::runtime_error("speculative tree check failed");

        attention::PageTable pages;
        pages.init(1, 1, dim, 1);
        if (!pages.alloc_block(0) || pages.alloc_block(0)) throw std::runtime_error("page limit check failed");

        if (!bundle_path.empty()) {
            auto b = load_bundle(bundle_path);
            if (!b) throw std::runtime_error("bundle load/validation failed");
            RuntimeConfig cfg;
            Runtime rt(std::shared_ptr<Bundle>(std::move(b)), cfg);
            if (!rt.init()) throw std::runtime_error("runtime init from bundle failed");
            if (!rt.prefill({1, 2, 3})) throw std::runtime_error("prefill over bundle failed");
            for (int i = 0; i < 4; ++i)
                if (rt.decode_one() < 0) throw std::runtime_error("decode over bundle failed");
            printf("[SELF-TEST] bundle %s → workload=%s rank=%d residual_used=%s\n",
                   bundle_path.c_str(), rt.workload_label(), rt.residual_rank(),
                   rt.last_token_stats().residual_used ? "yes" : "no");
        }

        printf("[SELF-TEST] PASS — residual, Q4 bounds, speculative tree and page limits\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "[SELF-TEST] FAIL — %s\n", error.what());
        return 1;
    }
}

static int bench_profiles(int dim, int layers, int n_tokens, const std::string& output_path,
                          const std::string& bundle_path) {
    std::shared_ptr<Bundle> bundle;
    if (!bundle_path.empty()) {
        auto b = load_bundle(bundle_path);
        if (!b) {
            fprintf(stderr, "[WINNER] cannot load bundle %s\n", bundle_path.c_str());
            return 1;
        }
        bundle = std::shared_ptr<Bundle>(std::move(b));
        printf("Bundle: %s (%s)\n", bundle_path.c_str(),
               bundle->is_cscd() ? "CSCD v0x0003, real stage tensors" : "legacy container");
    }

    printf("=== Profile latency table (dim=%d layers=%d tokens=%d) ===\n\n", dim, layers, n_tokens);
    printf("| %-10s | %6s | %10s | %10s | %10s | %8s | %8s | %-11s |\n",
           "Profile", "rank", "us/GEMV", "ms/tok", "tok/s", "RSS_MB", "res%", "workload");
    printf("|------------|-------:|-----------:|-----------:|-----------:|---------:|---------:|-------------|\n");

    QualityProfile profiles[] = {
        QualityProfile::MINMEM, QualityProfile::FAST,
        QualityProfile::BALANCED, QualityProfile::SAFE
    };

    std::ofstream jf(output_path, std::ios::binary | std::ios::trunc);
    if (!jf) {
        fprintf(stderr, "[WINNER] cannot write benchmark JSON: %s\n", output_path.c_str());
        return 1;
    }
    jf << "{\n  \"dim\": " << dim << ", \"layers\": " << layers << ", \"tokens\": " << n_tokens << ",\n  \"profiles\": [\n";

    std::string overall_workload = "synthetic";
    bool wrote_profile = false;
    for (size_t pi = 0; pi < 4; ++pi) {
        auto p = profiles[pi];
        RuntimeConfig cfg;
        cfg.profile = p;
        Runtime rt(bundle, cfg);
        const bool ok = bundle ? rt.init() : rt.init_synthetic(layers, dim, 42);
        if (!ok) {
            printf("init failed for %s\n", profile_name(p));
            continue;
        }
        overall_workload = rt.workload_label();
        // Base GEMV stages per token: 1 for the single real bundle layer,
        // 2 per synthetic layer (attn proj + ffn proxy).
        const int base_stages_per_tok = rt.real_bundle_active() ? rt.n_layers()
                                                                : rt.n_layers() * 2;

        double us = rt.bench_gemv_us(32);
        // timed generate
        std::vector<int32_t> prompt = {1, 2, 3};
        auto t0 = std::chrono::steady_clock::now();
        int residual_hits = 0, stages = 0;
        rt.prefill(prompt);
        for (int i = 0; i < n_tokens; ++i) {
            rt.decode_one();
            residual_hits += std::max(0, rt.last_token_stats().stages_executed - base_stages_per_tok);
            stages += rt.last_token_stats().stages_executed;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_tok = ms_total / n_tokens;
        double tok_s = 1000.0 / ms_tok;
        double rss_mb = rt.peak_rss_bytes() / (1024.0 * 1024.0);
        double res_pct = 100.0 * residual_hits / std::max(1, n_tokens * base_stages_per_tok);

        printf("| %-10s | %6d | %10.2f | %10.2f | %10.1f | %8.2f | %7.1f%% | %-11s |\n",
               profile_name(p), rt.residual_rank(), us, ms_tok, tok_s, rss_mb, res_pct,
               rt.workload_label());

        if (wrote_profile) jf << ",\n";
        wrote_profile = true;
        jf << "    {\"profile\": \"" << profile_name(p) << "\", \"rank\": " << rt.residual_rank()
           << ", \"us_gemv\": " << us << ", \"ms_tok\": " << ms_tok << ", \"tok_s\": " << tok_s
           << ", \"rss_mb\": " << rss_mb << ", \"residual_pct\": " << res_pct
           << ", \"workload\": \"" << rt.workload_label() << "\""
           << ", \"residual_simulated\": " << (rt.simulated_residual_fired() ? "true" : "false")
           << "}";
    }
    jf << "\n  ],\n";
    jf << "  \"workload\": \"" << overall_workload << "\",\n";
    if (bundle_path.empty())
        jf << "  \"bundle\": null,\n";
    else
        jf << "  \"bundle\": \"" << json_escape(bundle_path) << "\",\n";
    jf << "  \"note\": \"tok/s from synthetic workload measures WINNER kernels only; it is NOT model throughput\"\n}\n";
    jf.close();
    printf("\nJSON → %s (workload=%s)\n", output_path.c_str(), overall_workload.c_str());
    if (overall_workload != "real_bundle")
        printf("Note: synthetic tok/s measures WINNER kernels only — it is NOT model throughput.\n");
    return 0;
}

int main(int argc, char** argv) {
    print_banner();
    bool do_dev = false, do_cmp = false, do_serve = false, do_bench = false, do_vcpus = false, do_quants = false, do_self_test = false;
    std::string model, host = "127.0.0.1";
    std::string bundle_path;
    std::string output_path = "winner_profile_bench.json";
    int port = 8080, tokens = 16, ngl = -1, dim = 256, layers = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--devices")) do_dev = true;
        else if (!strcmp(argv[i], "--vcpus")) do_vcpus = true;
        else if (!strcmp(argv[i], "--quants")) do_quants = true;
        else if (!strcmp(argv[i], "--compare")) do_cmp = true;
        else if (!strcmp(argv[i], "--bench-kernels")) do_bench = true;
        else if (!strcmp(argv[i], "--self-test")) do_self_test = true;
        else if (!strcmp(argv[i], "--serve")) do_serve = true;
        else if (!strcmp(argv[i], "--model") && i+1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--bundle") && i+1 < argc) bundle_path = argv[++i];
        else if (!strcmp(argv[i], "--host") && i+1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i+1 < argc) { if (!parse_int_arg(argv[++i], 1, 65535, port)) return 2; }
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) { if (!parse_int_arg(argv[++i], 1, 100000, tokens)) return 2; }
        else if (!strcmp(argv[i], "--ngl") && i+1 < argc) { if (!parse_int_arg(argv[++i], -1, 100000, ngl)) return 2; }
        else if (!strcmp(argv[i], "--dim") && i+1 < argc) { if (!parse_int_arg(argv[++i], 1, 4096, dim)) return 2; }
        else if (!strcmp(argv[i], "--layers") && i+1 < argc) { if (!parse_int_arg(argv[++i], 1, 512, layers)) return 2; }
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output_path = argv[++i];
        else if (!strcmp(argv[i], "--help")) {
            printf("  --self-test [--bundle F] | --devices | --compare | --bench-kernels [--dim N] [--layers N]\n");
            printf("  --bundle F runs self-test/bench over a real CSCD bundle (workload=real_bundle)\n");
            printf("  --serve [--port N] | --model F [--ngl N]\n");
            printf("  --output FILE controls benchmark JSON output\n");
            return 0;
        } else { fprintf(stderr, "[WINNER] unknown or incomplete argument: %s\n", argv[i]); return 2; }
    }

    if (do_self_test) return run_self_test(bundle_path);

    if (do_dev || do_vcpus ||
        (!do_cmp && !do_serve && !do_bench && !do_vcpus && model.empty() && bundle_path.empty()))
        print_devices();

    
    if (do_vcpus || do_dev) {
        auto feat = backend::detect_cpu();
        backend::VCpuConfig vcfg;
        vcfg.reserved_cores_per_socket = 1;
        vcfg.pin_threads = true;
        vcfg.spread_sockets = true;
        backend::VCpuPool pool;
        pool.init(feat, vcfg);
        printf("%s\n", pool.summary().c_str());
        // smoke: parallel_for should not oversubscribe
        std::atomic<int> sum{0};
        pool.parallel_for(32, [&](int b, int e, int vid) {
            for (int i = b; i < e; ++i) sum.fetch_add(i);
            (void)vid;
        });
        printf("[VCPU] parallel_for smoke sum=%d (expected 496)\n", sum.load());
    }

    
    if (do_quants) {
        printf("=== K-quant catalog (Gemma 4 12B = 11.95B params) ===\n");
        printf("| %-12s | %12s | %s\n", "Type", "Est. size MB", "Role");
        printf("|--------------|-------------:|----\n");
        double pb = 11.95;
        using winner::quant::KQuantType;
        struct { KQuantType t; const char* role; } items[] = {
            {KQuantType::F0_TERNARY, "WINNER progressive base"},
            {KQuantType::Q4_0, "llama.cpp classic"},
            {KQuantType::Q4_K_S, "llama.cpp k-quant small"},
            {KQuantType::Q4_K_M, "llama.cpp k-quant medium (popular)"},
            {KQuantType::Q5_K_M, "higher fidelity"},
            {KQuantType::Q6_K, "near-Q8 quality"},
            {KQuantType::Q8_0, "reference quant"},
        };
        for (auto& it : items) {
            double mb = winner::quant::estimate_model_mb(pb, it.t);
            printf("| %-12s | %12.0f | %s\n", winner::quant::kquant_name(it.t), mb, it.role);
        }
        printf("\n=== Multi-GPU layer-split plan (simulated 2x GPU) ===\n");
        std::vector<backend::DeviceInfo> fake = backend::enumerate_devices();
        // inject 2 fake GPUs for planning demo if none
        bool has_gpu = false;
        for (auto& d : fake) if (d.type != backend::DeviceType::CPU) has_gpu = true;
        if (!has_gpu) {
            backend::DeviceInfo g0, g1;
            g0.type = backend::DeviceType::CUDA; g0.name = "SimGPU0"; g0.index = 0;
            g0.total_memory_bytes = g0.free_memory_bytes = 8ull<<30; g0.available = true;
            g1.type = backend::DeviceType::CUDA; g1.name = "SimGPU1"; g1.index = 1;
            g1.total_memory_bytes = g1.free_memory_bytes = 8ull<<30; g1.available = true;
            fake.push_back(g0); fake.push_back(g1);
        }
        size_t bpl = size_t(11.95e9 / 48 * 4.85 / 8); // ~Q4_K_M per layer
        auto plan = backend::plan_multi_gpu(48, bpl, fake, {}, -1);
        printf("%s\n", plan.summary.c_str());
        for (int L = 0; L < 48; L += 8)
            printf("  layer %2d → device %d\n", L, backend::device_for_layer(plan, L));
        printf("[QUANTS] K-quant + multi-GPU planner ready\n");
        return 0;
    }

    // --bundle without another mode runs the profile bench over the real bundle.
    if (do_bench || do_cmp || !bundle_path.empty()) {
        return bench_profiles(dim, layers, tokens, output_path, bundle_path);
    }

    if (do_serve) {
#if defined(WINNER_HTTP_SERVER)
        ContinuousBatcher batcher; batcher.init(2048, 32);
        HttpServer srv;
        srv.set_submit_handler([&](const ChatRequest& cr) {
            return batcher.submit({1,2,3}, cr.max_tokens, cr.temperature);
        });
        printf("[SERVE] %s:%d\n", host.c_str(), port);
        srv.start(host, port, false);
        return 0;
#else
        fprintf(stderr, "[WINNER] HTTP server is available only on Unix builds.\n");
        return 2;
#endif
    }

    if (!model.empty()) {
        auto b = load_bundle(model);
        if (!b) {
            fprintf(stderr, "[WINNER] model bundle could not be loaded; refusing a silent synthetic fallback\n");
            return 1;
        }
        RuntimeConfig cfg; cfg.n_gpu_layers = ngl;
        auto rt = std::make_unique<Runtime>(std::shared_ptr<Bundle>(std::move(b)), cfg);
        if (!rt->init()) return 1;
        printf("[RUN] rank=%d ISA=%s RSS=%.1f MB workload=%s\n",
               rt->residual_rank(), backend::isa_name(rt->active_isa()),
               rt->peak_rss_bytes()/(1024.*1024.), rt->workload_label());
        return 0;
    }
    return 0;
}

// note: kimi proxy is separate binary; profile table already uses residual ranks
