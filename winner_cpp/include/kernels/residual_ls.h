#pragma once
/**
 * Least-squares residual for F0 ternary / HQR base.
 * R = W - W_F0  ≈  U_k V_k^T   (randomized truncated SVD)
 * Inference: y = F0(x) + U (V^T x)
 *
 * F0 stores ternary codes at 2 bits/weight (4 codes per byte) to cut GEMV
 * bandwidth ~4× vs int8-per-weight without changing residual rank or layer count.
 */
#include <cstdint>
#include <vector>

namespace winner {
namespace kernels {

struct TernaryMatrix {
    int rows = 0, cols = 0;
    int cols_stride = 0;              // padded to multiple of 4 for 2-bit packing
    std::vector<uint8_t> packed;      // 2 bits/weight: 00=0, 01=+1, 10=-1
    std::vector<float> scales;        // per-row abs-max scale
    float threshold = 0.5f;

    size_t packed_bytes() const { return packed.size(); }
    size_t logical_weights() const { return size_t(rows) * size_t(cols); }
};

struct LowRankResidual {
    int rows = 0, cols = 0, rank = 0;
    std::vector<float> U;  // rows × rank
    std::vector<float> V;  // cols × rank
    std::vector<float> singular;
};

TernaryMatrix pack_ternary(const float* W, int rows, int cols, float threshold = -1.f);
void reconstruct_ternary(const TernaryMatrix& T, float* W_out);
LowRankResidual fit_residual_ls(const float* W, const TernaryMatrix& F0,
                                int rank, int n_iter = 10, uint32_t seed = 42);

void gemv_ternary(const TernaryMatrix& T, const float* x, float* y);
void gemv_lowrank_add(const LowRankResidual& R, const float* x, float* y);
void gemv_f0_plus_residual(const TernaryMatrix& F0, const LowRankResidual& R,
                           const float* x, float* y);

void gemv_f0_plus_residual_fused(const TernaryMatrix& F0, const LowRankResidual& R,
                                 const float* x, float* y);
void gemv_f0_residual_rms(const TernaryMatrix& F0, const LowRankResidual& R,
                          const float* x, float* y, float eps = 1e-6f);

float cosine_similarity(const float* a, const float* b, int n);
float nrmse_metric(const float* ref, const float* pred, int n);

} // namespace kernels
} // namespace winner
