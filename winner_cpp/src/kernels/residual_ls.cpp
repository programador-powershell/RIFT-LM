#include "kernels/residual_ls.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define WINNER_AVX2 1
#endif

namespace winner {
namespace kernels {

static double frobenius(const float* A, int n) {
    double s = 0; for (int i = 0; i < n; ++i) s += double(A[i]) * A[i];
    return std::sqrt(s);
}

static void pack_thr(const float* W, int rows, int cols, float thr,
                     std::vector<int8_t>& out, std::vector<float>& scales) {
    scales.resize(rows);
    out.resize(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        float amax = 0.f;
        for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(W[r * cols + c]));
        scales[r] = amax > 1e-12f ? amax : 1.f;
        for (int c = 0; c < cols; ++c) {
            float v = W[r * cols + c] / scales[r];
            if (v > thr) out[r * cols + c] = 1;
            else if (v < -thr) out[r * cols + c] = -1;
            else out[r * cols + c] = 0;
        }
    }
}

static float optimize_threshold(const float* W, int rows, int cols) {
    float best_thr = 0.5f;
    double best_err = 1e300;
    std::vector<int8_t> W8;
    std::vector<float> scales, Wf(size_t(rows) * cols);
    for (int t = 1; t <= 18; ++t) {
        float thr = t * 0.05f;
        pack_thr(W, rows, cols, thr, W8, scales);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                Wf[r * cols + c] = float(W8[r * cols + c]) * scales[r];
        double e = 0;
        for (size_t i = 0; i < Wf.size(); ++i) {
            double d = W[i] - Wf[i]; e += d * d;
        }
        e = std::sqrt(e);
        if (e < best_err) { best_err = e; best_thr = thr; }
    }
    return best_thr;
}

TernaryMatrix pack_ternary(const float* W, int rows, int cols, float threshold) {
    if (!W || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("pack_ternary requires a non-null matrix with positive dimensions");
    }
    TernaryMatrix T;
    T.rows = rows; T.cols = cols;
    if (threshold < 0.f) threshold = optimize_threshold(W, rows, cols);
    T.threshold = threshold;
    pack_thr(W, rows, cols, threshold, T.weight, T.scales);
    return T;
}

void reconstruct_ternary(const TernaryMatrix& T, float* W_out) {
    for (int r = 0; r < T.rows; ++r)
        for (int c = 0; c < T.cols; ++c)
            W_out[r * T.cols + c] = float(T.weight[r * T.cols + c]) * T.scales[r];
}

static void matvec(const float* R, int rows, int cols, const float* x, float* y, bool transpose) {
    if (!transpose) {
        for (int r = 0; r < rows; ++r) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) s += R[r * cols + c] * x[c];
            y[r] = s;
        }
    } else {
        for (int c = 0; c < cols; ++c) {
            float s = 0.f;
            for (int r = 0; r < rows; ++r) s += R[r * cols + c] * x[r];
            y[c] = s;
        }
    }
}

static void normalize(float* v, int n) {
    double ss = 0; for (int i = 0; i < n; ++i) ss += v[i] * v[i];
    float inv = ss > 1e-18 ? float(1.0 / std::sqrt(ss)) : 0.f;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

static float dotv(const float* a, const float* b, int n) {
    double s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return float(s);
}

LowRankResidual fit_residual_ls(const float* W, const TernaryMatrix& F0,
                                int rank, int n_iter, uint32_t seed) {
    const int rows = F0.rows, cols = F0.cols;
    if (!W || rows <= 0 || cols <= 0 ||
        F0.weight.size() != size_t(rows) * size_t(cols) ||
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
        double ss = 0; for (int i = 0; i < rows; ++i) ss += ycol[i] * ycol[i];
        out.singular[j] = float(std::sqrt(ss));
        float inv = out.singular[j] > 1e-12f ? 1.f / out.singular[j] : 0.f;
        float s = std::sqrt(std::max(out.singular[j], 0.f));
        for (int i = 0; i < rows; ++i) out.U[i * rank + j] = ycol[i] * inv * s;
        for (int i = 0; i < cols; ++i) out.V[i * rank + j] = Omega[j * cols + i] * s;
    }
    return out;
}

void gemv_ternary(const TernaryMatrix& T, const float* x, float* y) {
    if (!x || !y || T.rows <= 0 || T.cols <= 0 ||
        T.weight.size() != size_t(T.rows) * size_t(T.cols) ||
        T.scales.size() != static_cast<size_t>(T.rows)) {
        throw std::invalid_argument("gemv_ternary received invalid input");
    }
    for (int r = 0; r < T.rows; ++r) {
        const int8_t* wrow = T.weight.data() + r * T.cols;
        float s = 0.f;
        int c = 0;
#if defined(WINNER_AVX2)
        __m256 acc = _mm256_setzero_ps();
        for (; c + 8 <= T.cols; c += 8) {
            __m128i b = _mm_loadl_epi64((const __m128i*)(wrow + c));
            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + c), acc);
        }
        float tmp[8]; _mm256_storeu_ps(tmp, acc);
        s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#endif
        for (; c < T.cols; ++c) s += float(wrow[c]) * x[c];
        y[r] = s * T.scales[r];
    }
}

void gemv_lowrank_add(const LowRankResidual& R, const float* x, float* y) {
    const int k = R.rank;
    if (k == 0) return;
    if (!x || !y || R.rows <= 0 || R.cols <= 0 || k < 0 ||
        R.U.size() != size_t(R.rows) * size_t(k) ||
        R.V.size() != size_t(R.cols) * size_t(k)) {
        throw std::invalid_argument("gemv_lowrank_add received invalid input");
    }
    std::vector<float> t(k, 0.f);
    for (int j = 0; j < k; ++j) {
        float s = 0.f;
        for (int c = 0; c < R.cols; ++c) s += R.V[c * k + j] * x[c];
        t[j] = s;
    }
    for (int r = 0; r < R.rows; ++r) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += R.U[r * k + j] * t[j];
        y[r] += s;
    }
}

void gemv_f0_plus_residual(const TernaryMatrix& F0, const LowRankResidual& R,
                           const float* x, float* y) {
    gemv_ternary(F0, x, y);
    if (R.rank > 0) gemv_lowrank_add(R, x, y);
}

float cosine_similarity(const float* a, const float* b, int n) {
    if (!a || !b || n <= 0) return 0.f;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    if (na < 1e-18 || nb < 1e-18) return 0.f;
    return float(dot / (std::sqrt(na) * std::sqrt(nb)));
}

float nrmse_metric(const float* ref, const float* pred, int n) {
    if (!ref || !pred || n <= 0) return 0.f;
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        double d = ref[i] - pred[i];
        num += d * d; den += ref[i] * ref[i];
    }
    if (den < 1e-18) return 0.f;
    return float(std::sqrt(num / den));
}


void gemv_f0_plus_residual_fused(const TernaryMatrix& F0, const LowRankResidual& R,
                                 const float* x, float* y) {
    gemv_ternary(F0, x, y);
    if (R.rank <= 0) return;
    const int k = R.rank;
    std::vector<float> t(static_cast<size_t>(k));
    for (int j = 0; j < k; ++j) {
        float s = 0.f;
        for (int c = 0; c < R.cols; ++c) s += R.V[c * k + j] * x[c];
        t[static_cast<size_t>(j)] = s;
    }
    for (int r = 0; r < R.rows; ++r) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += R.U[r * k + j] * t[static_cast<size_t>(j)];
        y[r] += s;
    }
}

void gemv_f0_residual_rms(const TernaryMatrix& F0, const LowRankResidual& R,
                          const float* x, float* y, float eps) {
    gemv_f0_plus_residual_fused(F0, R, x, y);
    double ss = 0;
    for (int i = 0; i < F0.rows; ++i) ss += double(y[i]) * y[i];
    float inv = float(1.0 / std::sqrt(ss / std::max(1, F0.rows) + eps));
    for (int i = 0; i < F0.rows; ++i) y[i] *= inv;
}

} // namespace kernels
} // namespace winner
