#include "kernels/fused.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace winner {
namespace kernels {

static void rmsnorm(const float* x, const float* w, float* y, int dim) {
    if (!x || !y || dim <= 0) return;
    float ss = 0.f;
    for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
    float inv = 1.f / std::sqrt(ss / dim + 1e-6f);
    for (int i = 0; i < dim; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.f);
}

void fused_norm_qkv_rope(
    const float* x, const float* rms_weight,
    const float* wq, const float* wk, const float* wv,
    float* q, float* k, float* v,
    int seq, int dim, int n_heads, int head_dim,
    float rope_theta, int pos_offset)
{
    if (!x || !q || !k || !v || seq <= 0 || dim <= 0 || n_heads <= 0 ||
        head_dim <= 0 || head_dim % 2 != 0 || n_heads * head_dim != dim ||
        !std::isfinite(rope_theta) || rope_theta <= 0.f) return;
    std::vector<float> tmp(dim);
    for (int t = 0; t < seq; ++t) {
        rmsnorm(x + t * dim, rms_weight, tmp.data(), dim);
        // QKV projections (dense reference)
        for (int i = 0; i < dim; ++i) {
            float qq = 0, kk = 0, vv = 0;
            for (int j = 0; j < dim; ++j) {
                float xj = tmp[j];
                if (wq) qq += xj * wq[i * dim + j];
                if (wk) kk += xj * wk[i * dim + j];
                if (wv) vv += xj * wv[i * dim + j];
            }
            q[t * dim + i] = qq; k[t * dim + i] = kk; v[t * dim + i] = vv;
        }
        // RoPE on Q,K
        int pos = pos_offset + t;
        for (int h = 0; h < n_heads; ++h) {
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.f / std::pow(rope_theta, float(i) / head_dim);
                float ang = pos * freq;
                float c = std::cos(ang), s = std::sin(ang);
                int idx = h * head_dim + i;
                float q0 = q[t * dim + idx], q1 = q[t * dim + idx + 1];
                q[t * dim + idx]     = q0 * c - q1 * s;
                q[t * dim + idx + 1] = q0 * s + q1 * c;
                float k0 = k[t * dim + idx], k1 = k[t * dim + idx + 1];
                k[t * dim + idx]     = k0 * c - k1 * s;
                k[t * dim + idx + 1] = k0 * s + k1 * c;
            }
        }
    }
}

void flash_attention(
    const float* q, const float* k, const float* v, float* out,
    int seq_q, int seq_kv, int n_heads, int head_dim,
    float scale, int block_m, int block_n)
{
    // Tiled attention reference (not IO-aware FlashAttention, but same API)
    (void)block_m; (void)block_n;
    if (!q || !k || !v || !out || seq_q <= 0 || seq_kv <= 0 ||
        n_heads <= 0 || head_dim <= 0 || !std::isfinite(scale)) return;
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < seq_q; ++i) {
            std::vector<float> scores(seq_kv);
            float maxs = -1e30f;
            for (int j = 0; j < seq_kv; ++j) {
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    dot += q[(i * n_heads + h) * head_dim + d] *
                           k[(j * n_heads + h) * head_dim + d];
                scores[j] = dot * scale;
                if (scores[j] > maxs) maxs = scores[j];
            }
            float sum = 0.f;
            for (int j = 0; j < seq_kv; ++j) {
                scores[j] = std::exp(scores[j] - maxs);
                sum += scores[j];
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.f;
                for (int j = 0; j < seq_kv; ++j)
                    acc += (scores[j] / sum) * v[(j * n_heads + h) * head_dim + d];
                out[(i * n_heads + h) * head_dim + d] = acc;
            }
        }
    }
}

void flash_decoding(const float* q, const float* k_cache, const float* v_cache,
                    float* out, int n_heads, int head_dim, int kv_len) {
    if (head_dim <= 0) return;
    flash_attention(q, k_cache, v_cache, out, 1, kv_len, n_heads, head_dim,
                    1.f / std::sqrt(float(head_dim)));
}

void quant_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K, const float*, const float*) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int k = 0; k < K; ++k) s += int32_t(A[m*K+k]) * int32_t(B[n*K+k]);
            C[m*N+n] = s;
        }
}

void quant_gemv_int4(const uint8_t* W, const float* X, float* Y,
                     int rows, int cols, const float* scales) {
    if (!W || !X || !Y || rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        float sum = 0.f;
        for (int c = 0; c < cols; c += 2) {
            uint8_t p = W[r * ((cols+1)/2) + c/2];
            int a = (p & 0xF) - 8, b = ((p >> 4) & 0xF) - 8;
            sum += a * X[c];
            if (c + 1 < cols) sum += b * X[c+1];
        }
        Y[r] = sum * (scales ? scales[r] : 1.f);
    }
}

// Adapted from quant_gemv_int4 for the CSCD F0 packing: two's-complement
// nibbles (not offset-binary) and per-(row, group) FP32 scales.
void gemv_int4_group(const uint8_t* codes, const float* scales,
                     int rows, int cols, int packed_cols,
                     int group_size, int n_groups,
                     const float* X, float* Y) {
    if (!codes || !scales || !X || !Y || rows <= 0 || cols <= 0 ||
        group_size <= 0 || n_groups <= 0 || packed_cols <= 0 ||
        int64_t(packed_cols) * 2 < cols ||
        int64_t(n_groups) * group_size < cols) return;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* wrow = codes + size_t(r) * size_t(packed_cols);
        const float* srow = scales + size_t(r) * size_t(n_groups);
        float sum = 0.f;
        for (int g = 0; g < n_groups; ++g) {
            const int c0 = g * group_size;
            if (c0 >= cols) break;                 // padded tail groups multiply x that does not exist
            int c1 = c0 + group_size;
            if (c1 > cols) c1 = cols;
            float gsum = 0.f;
            for (int c = c0; c < c1; ++c) {
                const uint8_t byte = wrow[c >> 1];
                int q = (c & 1) ? int((byte >> 4) & 0xF) : int(byte & 0xF);
                if (q >= 8) q -= 16;               // sign-extend two's-complement nibble
                gsum += float(q) * X[c];
            }
            sum += gsum * srow[g];
        }
        Y[r] = sum;
    }
}

} // namespace kernels
} // namespace winner
