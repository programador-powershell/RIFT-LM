#include "backend/kernels.h"
#include "kernels/residual_ls.h"
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define WINNER_HAS_AVX2 1
#endif

namespace winner {
namespace backend {

static void f0_gemv_scalar(const void* W, const float* X, float* Y, int rows, int cols) {
    // W points to TernaryMatrix when used via runtime; fallback treats as float*
    const float* w = static_cast<const float*>(W);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) sum += w[r * cols + c] * X[c];
        Y[r] = sum;
    }
}

static void fused_lowrank_scalar(const float* X, const float* U, const float* V,
                                 float* Y, int dim, int rank) {
    // Y += (X·V cols) via V[c*rank+j], then U
    if (rank <= 0 || !U || !V) return;
    for (int j = 0; j < rank; ++j) {
        float xu = 0.f;
        for (int d = 0; d < dim; ++d) xu += X[d] * V[d * rank + j];
        for (int d = 0; d < dim; ++d) Y[d] += xu * U[d * rank + j];
    }
}

static void fused_stage_scalar(const float* X, float* Y, const void* f0,
                               const float* U, const float* V,
                               int dim, int rank, bool apply_residual) {
    f0_gemv_scalar(f0, X, Y, dim, dim);
    if (apply_residual && U && V && rank > 0)
        fused_lowrank_scalar(X, U, V, Y, dim, rank);
}

#if defined(WINNER_HAS_AVX2)
static void f0_gemv_avx2(const void* W, const float* X, float* Y, int rows, int cols) {
    const float* w = static_cast<const float*>(W);
    for (int r = 0; r < rows; ++r) {
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 wv = _mm256_loadu_ps(w + r * cols + c);
            const __m256 xv = _mm256_loadu_ps(X + c);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; c < cols; ++c) sum += w[r * cols + c] * X[c];
        Y[r] = sum;
    }
}

static void fused_lowrank_avx2(const float* X, const float* U, const float* V,
                               float* Y, int dim, int rank) {
    if (rank <= 0 || !U || !V) return;
    // Match residual_ls layout: V[c * rank + j], U[r * rank + j]
    for (int j = 0; j < rank; ++j) {
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= dim; d += 8) {
            alignas(32) float vbuf[8];
            for (int i = 0; i < 8; ++i) vbuf[i] = V[(d + i) * rank + j];
            const __m256 vv = _mm256_load_ps(vbuf);
            const __m256 xv = _mm256_loadu_ps(X + d);
            acc = _mm256_fmadd_ps(vv, xv, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float xu = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; d < dim; ++d) xu += X[d] * V[d * rank + j];

        const __m256 xuv = _mm256_set1_ps(xu);
        d = 0;
        for (; d + 8 <= dim; d += 8) {
            alignas(32) float ubuf[8];
            for (int i = 0; i < 8; ++i) ubuf[i] = U[(d + i) * rank + j];
            __m256 yv = _mm256_loadu_ps(Y + d);
            const __m256 uv = _mm256_load_ps(ubuf);
            yv = _mm256_fmadd_ps(uv, xuv, yv);
            _mm256_storeu_ps(Y + d, yv);
        }
        for (; d < dim; ++d) Y[d] += xu * U[d * rank + j];
    }
}

static void fused_stage_avx2(const float* X, float* Y, const void* f0,
                             const float* U, const float* V,
                             int dim, int rank, bool apply_residual) {
    // Prefer real ternary GEMV when pointer is a TernaryMatrix* (runtime path
    // uses kernels::gemv_* directly). This stage path keeps float F0 for legacy.
    f0_gemv_avx2(f0, X, Y, dim, dim);
    if (apply_residual && U && V && rank > 0)
        fused_lowrank_avx2(X, U, V, Y, dim, rank);
}
#endif

KernelTable select_cpu_kernels(const CpuFeatures& feat) {
    KernelTable t;
    t.f0_gemv = f0_gemv_scalar;
    t.fused_lowrank = fused_lowrank_scalar;
    t.fused_stage = fused_stage_scalar;
    t.isa = KernelIsa::SCALAR;

#if defined(WINNER_HAS_AVX2)
    if (feat.avx2) {
        t.isa = KernelIsa::AVX2;
        t.f0_gemv = f0_gemv_avx2;
        t.fused_lowrank = fused_lowrank_avx2;
        t.fused_stage = fused_stage_avx2;
    }
#else
    (void)feat;
#endif
    return t;
}

KernelTable select_gpu_kernels(const DeviceInfo& gpu) {
    KernelTable t;
    t.isa = KernelIsa::SCALAR;
    t.f0_gemv = f0_gemv_scalar;
    t.fused_lowrank = fused_lowrank_scalar;
    t.fused_stage = fused_stage_scalar;
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
