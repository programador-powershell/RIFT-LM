#include "kernels/residual_ls.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define WINNER_HAS_AVX2 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define WINNER_PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#else
#  define WINNER_PREFETCH(addr) ((void)0)
#endif

namespace winner {
namespace kernels {

namespace {

inline int cols_stride_for(int cols) {
    return (cols + 3) & ~3;  // pad to multiple of 4
}

inline uint8_t encode_code(int8_t v) {
    if (v > 0) return 1u;   // +1
    if (v < 0) return 2u;   // -1
    return 0u;              // 0
}

inline float decode_code(uint8_t bits) {
    // 0→0, 1→+1, 2→-1, 3→0 (unused)
    static const float lut[4] = {0.f, 1.f, -1.f, 0.f};
    return lut[bits & 3u];
}

inline void pack_codes_row(const int8_t* codes, int cols, int stride, uint8_t* out) {
    const int nbytes = stride / 4;
    for (int b = 0; b < nbytes; ++b) {
        const int c0 = b * 4;
        uint8_t byte = 0;
        for (int k = 0; k < 4; ++k) {
            const int c = c0 + k;
            const int8_t v = (c < cols) ? codes[c] : int8_t(0);
            byte |= static_cast<uint8_t>(encode_code(v) << (k * 2));
        }
        out[b] = byte;
    }
}

void pack_thr(const float* W, int rows, int cols, float thr,
              std::vector<uint8_t>& packed, std::vector<float>& scales, int& stride) {
    stride = cols_stride_for(cols);
    scales.assign(static_cast<size_t>(rows), 1.f);
    packed.assign(static_cast<size_t>(rows) * static_cast<size_t>(stride / 4), 0);
    std::vector<int8_t> row(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        float amax = 0.f;
        for (int c = 0; c < cols; ++c)
            amax = std::max(amax, std::fabs(W[r * cols + c]));
        scales[static_cast<size_t>(r)] = amax > 1e-12f ? amax : 1.f;
        const float inv = 1.f / scales[static_cast<size_t>(r)];
        for (int c = 0; c < cols; ++c) {
            const float v = W[r * cols + c] * inv;
            if (v > thr) row[static_cast<size_t>(c)] = 1;
            else if (v < -thr) row[static_cast<size_t>(c)] = -1;
            else row[static_cast<size_t>(c)] = 0;
        }
        pack_codes_row(row.data(), cols, stride,
                       packed.data() + static_cast<size_t>(r) * static_cast<size_t>(stride / 4));
    }
}

float optimize_threshold(const float* W, int rows, int cols) {
    float best_thr = 0.5f;
    double best_err = 1e300;
    std::vector<uint8_t> packed;
    std::vector<float> scales, Wf(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    int stride = 0;
    for (int t = 1; t <= 18; ++t) {
        const float thr = t * 0.05f;
        pack_thr(W, rows, cols, thr, packed, scales, stride);
        for (int r = 0; r < rows; ++r) {
            const uint8_t* prow = packed.data() + static_cast<size_t>(r) * static_cast<size_t>(stride / 4);
            const float s = scales[static_cast<size_t>(r)];
            for (int c = 0; c < cols; ++c) {
                const uint8_t byte = prow[c >> 2];
                const uint8_t bits = (byte >> ((c & 3) * 2)) & 3u;
                Wf[static_cast<size_t>(r) * cols + c] = decode_code(bits) * s;
            }
        }
        double e = 0;
        for (size_t i = 0; i < Wf.size(); ++i) {
            const double d = W[i] - Wf[i];
            e += d * d;
        }
        e = std::sqrt(e);
        if (e < best_err) {
            best_err = e;
            best_thr = thr;
        }
    }
    return best_thr;
}

void matvec(const float* A, int rows, int cols, const float* x, float* y, bool transpose) {
    if (!transpose) {
        for (int r = 0; r < rows; ++r) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) s += A[r * cols + c] * x[c];
            y[r] = s;
        }
    } else {
        std::fill(y, y + cols, 0.f);
        for (int r = 0; r < rows; ++r) {
            const float xr = x[r];
            for (int c = 0; c < cols; ++c) y[c] += A[r * cols + c] * xr;
        }
    }
}

float dotv(const float* a, const float* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

void normalize(float* v, int n) {
    const float nrm = std::sqrt(std::max(1e-20f, dotv(v, v, n)));
    const float inv = 1.f / nrm;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

} // namespace

TernaryMatrix pack_ternary(const float* W, int rows, int cols, float threshold) {
    if (!W || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("pack_ternary requires a non-null matrix with positive dimensions");
    }
    TernaryMatrix T;
    T.rows = rows;
    T.cols = cols;
    if (threshold < 0.f) threshold = optimize_threshold(W, rows, cols);
    T.threshold = threshold;
    pack_thr(W, rows, cols, threshold, T.packed, T.scales, T.cols_stride);
    return T;
}

void reconstruct_ternary(const TernaryMatrix& T, float* W_out) {
    if (!W_out) return;
    const int stride_bytes = T.cols_stride / 4;
    for (int r = 0; r < T.rows; ++r) {
        const uint8_t* prow = T.packed.data() + static_cast<size_t>(r) * static_cast<size_t>(stride_bytes);
        const float s = T.scales[static_cast<size_t>(r)];
        for (int c = 0; c < T.cols; ++c) {
            const uint8_t byte = prow[c >> 2];
            const uint8_t bits = (byte >> ((c & 3) * 2)) & 3u;
            W_out[r * T.cols + c] = decode_code(bits) * s;
        }
    }
}

LowRankResidual fit_residual_ls(const float* W, const TernaryMatrix& F0,
                                int rank, int n_iter, uint32_t seed) {
    const int rows = F0.rows, cols = F0.cols;
    const size_t expected_packed = size_t(rows) * size_t(std::max(1, F0.cols_stride / 4));
    if (!W || rows <= 0 || cols <= 0 ||
        F0.packed.size() < expected_packed ||
        F0.scales.size() != static_cast<size_t>(rows)) {
        throw std::invalid_argument("fit_residual_ls received an invalid ternary matrix");
    }
    rank = std::max(0, std::min(rank, std::min(rows, cols)));
    n_iter = std::max(1, n_iter);

    LowRankResidual out;
    out.rows = rows;
    out.cols = cols;
    out.rank = rank;
    if (rank == 0) return out;

    std::vector<float> Wf0(size_t(rows) * cols), R(size_t(rows) * cols);
    reconstruct_ternary(F0, Wf0.data());
    for (size_t i = 0; i < R.size(); ++i) R[i] = W[i] - Wf0[i];

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> Omega(size_t(cols) * rank), Y(size_t(rows) * rank);
    for (auto& v : Omega) v = nd(rng);

    for (int it = 0; it < n_iter; ++it) {
        for (int j = 0; j < rank; ++j)
            matvec(R.data(), rows, cols, Omega.data() + j * cols, Y.data() + j * rows, false);
        for (int j = 0; j < rank; ++j) {
            float* col = Y.data() + j * rows;
            for (int p = 0; p < j; ++p) {
                float* prev = Y.data() + p * rows;
                float d = dotv(prev, col, rows);
                for (int i = 0; i < rows; ++i) col[i] -= d * prev[i];
            }
            normalize(col, rows);
        }
        for (int j = 0; j < rank; ++j)
            matvec(R.data(), rows, cols, Y.data() + j * rows, Omega.data() + j * cols, true);
        for (int j = 0; j < rank; ++j) {
            float* col = Omega.data() + j * cols;
            for (int p = 0; p < j; ++p) {
                float* prev = Omega.data() + p * cols;
                float d = dotv(prev, col, cols);
                for (int i = 0; i < cols; ++i) col[i] -= d * prev[i];
            }
            normalize(col, cols);
        }
    }
    for (int j = 0; j < rank; ++j)
        matvec(R.data(), rows, cols, Omega.data() + j * cols, Y.data() + j * rows, false);

    out.U.assign(size_t(rows) * rank, 0.f);
    out.V.assign(size_t(cols) * rank, 0.f);
    out.singular.resize(rank);
    for (int j = 0; j < rank; ++j) {
        float* ycol = Y.data() + j * rows;
        double ss = 0;
        for (int i = 0; i < rows; ++i) ss += double(ycol[i]) * ycol[i];
        const float sigma = float(std::sqrt(ss));
        out.singular[static_cast<size_t>(j)] = sigma;
        const float inv = sigma > 1e-12f ? 1.f / sigma : 0.f;
        for (int i = 0; i < rows; ++i)
            out.U[static_cast<size_t>(i) * rank + j] = ycol[i] * inv * sigma; // U * sigma absorbed
        for (int i = 0; i < cols; ++i)
            out.V[static_cast<size_t>(i) * rank + j] = Omega[static_cast<size_t>(j) * cols + i];
        // Fold sigma into U: ycol already has magnitude; store unit V and scaled U
        for (int i = 0; i < rows; ++i)
            out.U[static_cast<size_t>(i) * rank + j] = ycol[i];
    }
    return out;
}

// ---- GEMV ternary (2-bit packed) ----

static void gemv_ternary_scalar(const TernaryMatrix& T, const float* x, float* y) {
    const int stride_bytes = T.cols_stride / 4;
    for (int r = 0; r < T.rows; ++r) {
        const uint8_t* prow = T.packed.data() + static_cast<size_t>(r) * static_cast<size_t>(stride_bytes);
        WINNER_PREFETCH(prow + 16);
        float sum = 0.f;
        int c = 0;
        for (; c + 4 <= T.cols; c += 4) {
            const uint8_t byte = prow[c >> 2];
            sum += decode_code( byte        & 3u) * x[c];
            sum += decode_code((byte >> 2)  & 3u) * x[c + 1];
            sum += decode_code((byte >> 4)  & 3u) * x[c + 2];
            sum += decode_code((byte >> 6)  & 3u) * x[c + 3];
        }
        for (; c < T.cols; ++c) {
            const uint8_t byte = prow[c >> 2];
            const uint8_t bits = (byte >> ((c & 3) * 2)) & 3u;
            sum += decode_code(bits) * x[c];
        }
        y[r] = sum * T.scales[static_cast<size_t>(r)];
    }
}

#if defined(WINNER_HAS_AVX2)
static void gemv_ternary_avx2(const TernaryMatrix& T, const float* x, float* y) {
    const int stride_bytes = T.cols_stride / 4;
    // Process 8 outputs conceptually; inner loop still unpacks 2-bit groups.
    for (int r = 0; r < T.rows; ++r) {
        const uint8_t* prow = T.packed.data() + static_cast<size_t>(r) * static_cast<size_t>(stride_bytes);
        WINNER_PREFETCH(prow + 32);
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        // 8 columns at a time = 2 packed bytes
        for (; c + 8 <= T.cols; c += 8) {
            const uint8_t b0 = prow[c >> 2];
            const uint8_t b1 = prow[(c >> 2) + 1];
            alignas(32) float codes[8] = {
                decode_code( b0        & 3u),
                decode_code((b0 >> 2)  & 3u),
                decode_code((b0 >> 4)  & 3u),
                decode_code((b0 >> 6)  & 3u),
                decode_code( b1        & 3u),
                decode_code((b1 >> 2)  & 3u),
                decode_code((b1 >> 4)  & 3u),
                decode_code((b1 >> 6)  & 3u),
            };
            const __m256 cv = _mm256_load_ps(codes);
            const __m256 xv = _mm256_loadu_ps(x + c);
            acc = _mm256_fmadd_ps(cv, xv, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; c < T.cols; ++c) {
            const uint8_t byte = prow[c >> 2];
            const uint8_t bits = (byte >> ((c & 3) * 2)) & 3u;
            sum += decode_code(bits) * x[c];
        }
        y[r] = sum * T.scales[static_cast<size_t>(r)];
    }
}
#endif

void gemv_ternary(const TernaryMatrix& T, const float* x, float* y) {
    if (!x || !y || T.rows <= 0 || T.cols <= 0) return;
#if defined(WINNER_HAS_AVX2)
    if (T.cols >= 8) {
        gemv_ternary_avx2(T, x, y);
        return;
    }
#endif
    gemv_ternary_scalar(T, x, y);
}

// ---- Low-rank residual SIMD ----

static void gemv_lowrank_add_scalar(const LowRankResidual& R, const float* x, float* y) {
    if (R.rank <= 0) return;
    const int k = R.rank;
    std::vector<float> t(static_cast<size_t>(k));
    for (int j = 0; j < k; ++j) {
        float s = 0.f;
        for (int c = 0; c < R.cols; ++c) s += R.V[static_cast<size_t>(c) * k + j] * x[c];
        t[static_cast<size_t>(j)] = s;
    }
    for (int r = 0; r < R.rows; ++r) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += R.U[static_cast<size_t>(r) * k + j] * t[static_cast<size_t>(j)];
        y[r] += s;
    }
}

#if defined(WINNER_HAS_AVX2)
static void gemv_lowrank_add_avx2(const LowRankResidual& R, const float* x, float* y) {
    if (R.rank <= 0) return;
    const int k = R.rank;
    std::vector<float> t(static_cast<size_t>(k), 0.f);

    // t = V^T x   (V is cols × rank, stored row-major as V[c * k + j])
    for (int j = 0; j < k; ++j) {
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= R.cols; c += 8) {
            // Gather 8 V entries for fixed j is irregular; use scalar load of V slice
            // Layout: consecutive ranks at each col → not contiguous for fixed j.
            // Use broadcast mul of x chunks against contiguous if we transpose conceptually.
            alignas(32) float vbuf[8];
            for (int i = 0; i < 8; ++i)
                vbuf[i] = R.V[static_cast<size_t>(c + i) * k + j];
            const __m256 vv = _mm256_load_ps(vbuf);
            const __m256 xv = _mm256_loadu_ps(x + c);
            acc = _mm256_fmadd_ps(vv, xv, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; c < R.cols; ++c) s += R.V[static_cast<size_t>(c) * k + j] * x[c];
        t[static_cast<size_t>(j)] = s;
    }

    // y += U t
    for (int r = 0; r < R.rows; ++r) {
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= k; j += 8) {
            const __m256 uv = _mm256_loadu_ps(&R.U[static_cast<size_t>(r) * k + j]);
            const __m256 tv = _mm256_loadu_ps(&t[static_cast<size_t>(j)]);
            acc = _mm256_fmadd_ps(uv, tv, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; j < k; ++j) s += R.U[static_cast<size_t>(r) * k + j] * t[static_cast<size_t>(j)];
        y[r] += s;
    }
}
#endif

void gemv_lowrank_add(const LowRankResidual& R, const float* x, float* y) {
    if (!x || !y || R.rank <= 0) return;
#if defined(WINNER_HAS_AVX2)
    if (R.cols >= 8 || R.rank >= 8) {
        gemv_lowrank_add_avx2(R, x, y);
        return;
    }
#endif
    gemv_lowrank_add_scalar(R, x, y);
}

void gemv_f0_plus_residual(const TernaryMatrix& F0, const LowRankResidual& R,
                           const float* x, float* y) {
    gemv_ternary(F0, x, y);
    gemv_lowrank_add(R, x, y);
}

void gemv_f0_plus_residual_fused(const TernaryMatrix& F0, const LowRankResidual& R,
                                 const float* x, float* y) {
    gemv_f0_plus_residual(F0, R, x, y);
}

void gemv_f0_residual_rms(const TernaryMatrix& F0, const LowRankResidual& R,
                          const float* x, float* y, float eps) {
    gemv_f0_plus_residual_fused(F0, R, x, y);
    double ss = 0;
    for (int i = 0; i < F0.rows; ++i) ss += double(y[i]) * y[i];
    const float inv = float(1.0 / std::sqrt(ss / std::max(1, F0.rows) + double(eps)));
    for (int i = 0; i < F0.rows; ++i) y[i] *= inv;
}

float cosine_similarity(const float* a, const float* b, int n) {
    if (!a || !b || n <= 0) return 0.f;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += double(a[i]) * b[i];
        na += double(a[i]) * a[i];
        nb += double(b[i]) * b[i];
    }
    const double den = std::sqrt(na * nb);
    return den > 1e-20 ? float(dot / den) : 0.f;
}

float nrmse_metric(const float* ref, const float* pred, int n) {
    if (!ref || !pred || n <= 0) return 0.f;
    double err = 0, nrm = 0;
    for (int i = 0; i < n; ++i) {
        const double d = double(ref[i]) - pred[i];
        err += d * d;
        nrm += double(ref[i]) * ref[i];
    }
    return float(std::sqrt(err) / std::max(1e-12, std::sqrt(nrm)));
}

} // namespace kernels
} // namespace winner
