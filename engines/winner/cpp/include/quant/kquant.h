#pragma once
/**
 * K-quant family (llama.cpp-compatible block layouts) for WINNER
 *
 * Closes the quant variety gap vs llama.cpp:
 *   Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0
 *
 * Progressive path still preferred (F0+LS). K-quants are available when:
 *   - loading legacy GGUF without SPECTRA/RIFT conversion
 *   - SAFE profile wants higher fidelity than ternary F0
 */
#include <cstdint>
#include <vector>
#include <string>

namespace winner {
namespace quant {

enum class KQuantType : uint8_t {
    Q4_0 = 0,
    Q4_1,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    Q8_0,
    F0_TERNARY,  // native WINNER progressive base
    COUNT
};

const char* kquant_name(KQuantType t);
int kquant_block_size(KQuantType t);      // weights per block
int kquant_bytes_per_block(KQuantType t);

struct KQuantTensor {
    KQuantType type = KQuantType::Q4_K_M;
    int rows = 0, cols = 0;
    std::vector<uint8_t> data;   // packed blocks
    // optional per-row scales for simple Q4_0 path
    std::vector<float> scales;
};

/** Pack FP32 row-major → K-quant (Q4_0 / Q4_K_M simplified / Q8_0) */
KQuantTensor pack_kquant(const float* W, int rows, int cols, KQuantType type);

/** GEMV: y = W_q · x */
void gemv_kquant(const KQuantTensor& W, const float* x, float* y);

/** Bytes if full model were this quant (rough, from param count) */
double estimate_model_mb(double params_b, KQuantType type);

} // namespace quant
} // namespace winner
