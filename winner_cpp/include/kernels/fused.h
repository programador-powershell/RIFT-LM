#pragma once
/**
 * Fused kernels + FlashAttention/FlashDecoding interface
 * Single pass: Input -> RMSNorm -> QKV -> RoPE
 * Attention computed in tiles over fast memory.
 */
#include "backend/cpu_detect.h"
#include <cstddef>
#include <cstdint>

namespace winner {
namespace kernels {

// RMSNorm + Linear QKV + RoPE fused (CPU reference)
void fused_norm_qkv_rope(
    const float* x,          // [seq, dim]
    const float* rms_weight, // [dim]
    const float* wq, const float* wk, const float* wv, // projections
    float* q, float* k, float* v,
    int seq, int dim, int n_heads, int head_dim,
    float rope_theta, int pos_offset);

// FlashAttention-style tiled attention (CPU reference)
// Softmax done per block to keep working set in L1/L2
void flash_attention(
    const float* q, const float* k, const float* v,
    float* out,
    int seq_q, int seq_kv, int n_heads, int head_dim,
    float scale, int block_m = 64, int block_n = 64);

// FlashDecoding for single-token decode with long KV
void flash_decoding(
    const float* q,          // [1, n_heads, head_dim]
    const float* k_cache, const float* v_cache, // paged or dense
    float* out,
    int n_heads, int head_dim, int kv_len);

// INT4/INT8 GEMM entry points (dispatch to AVX512-VNNI / AMX / NEON)
void quant_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K, const float* scale_a, const float* scale_b);

void quant_gemv_int4(const uint8_t* W_packed, const float* X, float* Y,
                     int rows, int cols, const float* scales);

} // namespace kernels
} // namespace winner
