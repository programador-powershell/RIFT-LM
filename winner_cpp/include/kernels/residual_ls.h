#pragma once
/**
 * Least-squares residual for F0 ternary / HQR base.
 * R = W - W_F0  ≈  U_k V_k^T   (randomized truncated SVD)
 * Inference: y = F0(x) + U (V^T x)
 */
#include <cstdint>
#include <vector>

namespace winner {
namespace kernels {

struct TernaryMatrix {
    int rows = 0, cols = 0;
    std::vector<int8_t> weight;
    std::vector<float>  scales;
    float threshold = 0.5f;
};

struct LowRankResidual {
    int rows = 0, cols = 0, rank = 0;
    std::vector<float> U;
    std::vector<float> V;
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
