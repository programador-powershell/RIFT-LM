/**
 * WINNER.cpp — Universal Progressive Inference Runtime
 * =====================================================
 * Native C++ runtime for models optimized by SPECTRA / AETHER / CASCADE / RIFT.
 * The name is optimizer-agnostic: the best-performing optimizer on the target
 * hardware becomes the "winner" and is preferred for production runs.
 *
 * Design goals (inspired by llama.cpp + bitnet.cpp):
 *   - Zero Python dependency at inference time
 *   - mmap-friendly Bundle loading
 *   - F0 always-resident base + gated residual stages
 *   - Fused Stage Kernel path (CPU first: AVX2 / NEON)
 *   - Predictive prefetch (io_uring path planned)
 *   - Formal quality / drift budget tracking
 *   - Benchmark harness that compares optimizers and selects the winner
 *
 * License: MIT (planned)
 * Version: 0.1.0-dev  (2026-08-09)
 */

#ifndef WINNER_H
#define WINNER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace winner {

// ---------------------------------------------------------------------------
// Version & Constants
// ---------------------------------------------------------------------------
constexpr uint32_t WINNER_VERSION_MAJOR = 0;
constexpr uint32_t WINNER_VERSION_MINOR = 8;
constexpr uint32_t WINNER_VERSION_PATCH = 0;

constexpr char WINNER_MAGIC[4] = {'W', 'I', 'N', 'R'};
constexpr uint16_t WINNER_CONTAINER_VERSION = 0x0100;

// Stage types (must match IR / Bundle)
enum class StageType : uint8_t {
    BASE_STAGE        = 0,
    RESIDUAL_ADDITIVE = 1,
    RESIDUAL_LOWRANK  = 2,
    RESIDUAL_SPARSE   = 3,
    SPECTRAL_STAGE    = 4,
    FULL_STAGE        = 5
};

// Optimizer origin (which algorithm produced the Bundle)
enum class OptimizerId : uint8_t {
    SPECTRA  = 0,
    AETHER   = 1,
    CASCADE  = 2,
    RIFT     = 3,
    UNKNOWN  = 255
};

// Quality profiles
enum class QualityProfile : uint8_t {
    SAFE     = 0,
    BALANCED = 1,
    FAST     = 2,
    MINMEM   = 3
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
struct Tensor;
struct Operation;
struct StagePage;
struct BundleHeader;
class  Bundle;
class  Runtime;
class  GateEngine;
class  PrefetchEngine;

// ---------------------------------------------------------------------------
// Public API surface (Phase 1 + roadmap stubs)
// ---------------------------------------------------------------------------

/**
 * Load a Bundle from disk (mmap preferred).
 * Accepts magic: WINR / SPCT / AETH / CASC / RIFT
 */
std::unique_ptr<Bundle> load_bundle(const std::string& path);

/**
 * Create a Runtime from a loaded Bundle + quality profile.
 */
std::unique_ptr<Runtime> create_runtime(
    std::shared_ptr<Bundle> bundle,
    QualityProfile profile = QualityProfile::BALANCED
);

/**
 * Simple generation entry point (Phase 1 reference).
 */
std::vector<int32_t> generate(
    Runtime& rt,
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens = 128,
    float temperature = 0.7f
);

} // namespace winner

#endif // WINNER_H
