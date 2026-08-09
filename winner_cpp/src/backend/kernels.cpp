#include "backend/kernels.h"
#include <cmath>
#include <cstring>

namespace winner {
namespace backend {

// ---------- Scalar reference kernels ----------
static void f0_gemv_scalar(const void* W, const float* X, float* Y, int rows, int cols) {
    // W treated as float for stub; real ternary packs differently
    const float* w = static_cast<const float*>(W);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) sum += w[r * cols + c] * X[c];
        Y[r] = sum;
    }
}

static void fused_lowrank_scalar(const float* X, const float* U, const float* V,
                                 float* Y, int dim, int rank) {
    // Y += (X·U) · Vᵀ
    for (int r = 0; r < rank; ++r) {
        float xu = 0.f;
        for (int d = 0; d < dim; ++d) xu += X[d] * U[d * rank + r];
        for (int d = 0; d < dim; ++d) Y[d] += xu * V[d * rank + r];
    }
}

static void fused_stage_scalar(const float* X, float* Y, const void* f0,
                               const float* U, const float* V,
                               int dim, int rank, bool apply_residual) {
    // F0
    f0_gemv_scalar(f0, X, Y, dim, dim);
    if (apply_residual && U && V && rank > 0)
        fused_lowrank_scalar(X, U, V, Y, dim, rank);
}

#if defined(__AVX2__)
#include <immintrin.h>
static void f0_gemv_avx2(const void* W, const float* X, float* Y, int rows, int cols) {
    const float* w = static_cast<const float*>(W);
    for (int r = 0; r < rows; ++r) {
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 wv = _mm256_loadu_ps(w + r * cols + c);
            __m256 xv = _mm256_loadu_ps(X + c);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; c < cols; ++c) sum += w[r * cols + c] * X[c];
        Y[r] = sum;
    }
}
#endif

KernelTable select_cpu_kernels(const CpuFeatures& feat) {
    KernelTable t;
    t.f0_gemv = f0_gemv_scalar;
    t.fused_lowrank = fused_lowrank_scalar;
    t.fused_stage = fused_stage_scalar;
    t.isa = KernelIsa::SCALAR;

#if defined(__AVX2__)
    if (feat.avx2) {
        t.isa = KernelIsa::AVX2;
        t.f0_gemv = f0_gemv_avx2;
    }
#else
    (void)feat;
#endif
    return t;
}

KernelTable select_gpu_kernels(const DeviceInfo& gpu) {
    KernelTable t;
    t.isa = KernelIsa::SCALAR;
    if (!gpu.available) return t;
#if defined(WINNER_HAS_CUDA)
    if (gpu.type == DeviceType::CUDA) {
        t.isa = KernelIsa::CUDA;
    }
#endif
    return t;
}

const char* isa_name(KernelIsa isa) {
    switch (isa) {
        case KernelIsa::SCALAR: return "SCALAR";
        case KernelIsa::AVX2: return "AVX2";
        case KernelIsa::AVX_VNNI: return "AVX-VNNI";
        case KernelIsa::AVX512_VNNI: return "AVX512-VNNI";
        case KernelIsa::AMX: return "AMX";
        case KernelIsa::NEON: return "NEON";
        case KernelIsa::CUDA: return "CUDA";
        case KernelIsa::HIP: return "HIP";
        default: return "?";
    }
}

} // namespace backend
} // namespace winner
