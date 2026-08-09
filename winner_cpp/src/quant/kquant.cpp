#include "quant/kquant.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
constexpr int kQ4BlockSize = 32;
constexpr int kQ4PayloadBytes = 4 + kQ4BlockSize / 2;
}

namespace winner {
namespace quant {

const char* kquant_name(KQuantType t) {
    switch (t) {
        case KQuantType::Q4_0: return "Q4_0";
        case KQuantType::Q4_1: return "Q4_1";
        case KQuantType::Q4_K_S: return "Q4_K_S";
        case KQuantType::Q4_K_M: return "Q4_K_M";
        case KQuantType::Q5_K_S: return "Q5_K_S";
        case KQuantType::Q5_K_M: return "Q5_K_M";
        case KQuantType::Q6_K: return "Q6_K";
        case KQuantType::Q8_0: return "Q8_0";
        case KQuantType::F0_TERNARY: return "F0_TERNARY";
        default: return "?";
    }
}

int kquant_block_size(KQuantType t) {
    switch (t) {
        case KQuantType::Q4_0:
        case KQuantType::Q4_1:
        case KQuantType::Q8_0: return 32;
        case KQuantType::Q4_K_S:
        case KQuantType::Q4_K_M:
        case KQuantType::Q5_K_S:
        case KQuantType::Q5_K_M:
        case KQuantType::Q6_K: return 256; // super-block
        case KQuantType::F0_TERNARY: return 32;
        default: return 32;
    }
}

int kquant_bytes_per_block(KQuantType t) {
    // Approximate storage per block (llama.cpp layouts simplified)
    switch (t) {
        case KQuantType::Q4_0: return kQ4PayloadBytes; // float scale + 16 packed bytes
        case KQuantType::Q4_1: return kQ4PayloadBytes;
        case KQuantType::Q4_K_S: return 84; // ~4.5 bpw effective over 256
        case KQuantType::Q4_K_M: return 144;
        case KQuantType::Q5_K_S: return 104;
        case KQuantType::Q5_K_M: return 176;
        case KQuantType::Q6_K: return 210;
        case KQuantType::Q8_0: return 34;   // 32 int8 + 2 scale
        case KQuantType::F0_TERNARY: return 10; // ~1.58 bpw
        default: return 18;
    }
}

double estimate_model_mb(double params_b, KQuantType type) {
    double bpw = 4.0;
    switch (type) {
        case KQuantType::Q4_0: case KQuantType::Q4_1: bpw = 4.5; break;
        case KQuantType::Q4_K_S: bpw = 4.6; break;
        case KQuantType::Q4_K_M: bpw = 4.85; break;
        case KQuantType::Q5_K_S: bpw = 5.4; break;
        case KQuantType::Q5_K_M: bpw = 5.7; break;
        case KQuantType::Q6_K: bpw = 6.6; break;
        case KQuantType::Q8_0: bpw = 8.5; break;
        case KQuantType::F0_TERNARY: bpw = 1.58; break;
        default: break;
    }
    return params_b * 1e9 * bpw / 8.0 / (1024.0 * 1024.0);
}

KQuantTensor pack_kquant(const float* W, int rows, int cols, KQuantType type) {
    if (!W || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("pack_kquant requires positive dimensions and a non-null matrix");
    }
    KQuantTensor T;
    T.type = type;
    T.rows = rows;
    T.cols = cols;

    if (type == KQuantType::F0_TERNARY) {
        // deferred to residual_ls pack_ternary
        return T;
    }

    if (type == KQuantType::Q8_0) {
        const int bs = 32;
        T.scales.resize(rows * ((cols + bs - 1) / bs));
        T.data.resize(size_t(rows) * cols);
        int si = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c0 = 0; c0 < cols; c0 += bs) {
                int n = std::min(bs, cols - c0);
                float amax = 0.f;
                for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(W[r*cols+c0+i]));
                float scale = amax / 127.f + 1e-8f;
                T.scales[si++] = scale;
                for (int i = 0; i < n; ++i) {
                    int q = (int)std::round(W[r*cols+c0+i] / scale);
                    q = std::max(-127, std::min(127, q));
                    T.data[r*cols+c0+i] = (uint8_t)(q + 128);
                }
            }
        }
        return T;
    }

    // Default path: Q4_0 / Q4_K_* treated as Q4_0 blocks of 32 for runtime GEMV
    // (full K-superblock decode can be added; layout metadata preserved in type)
    const int bs = kQ4BlockSize;
    int bpr = (cols + bs - 1) / bs;
    T.data.resize(size_t(rows) * size_t(bpr) * kQ4PayloadBytes);
    T.scales.clear();
    for (int r = 0; r < rows; ++r) {
        for (int bi = 0; bi < bpr; ++bi) {
            int c0 = bi * bs;
            int n = std::min(bs, cols - c0);
            float amax = 0.f;
            for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(W[r*cols+c0+i]));
            float scale = amax / 7.f + 1e-8f;
            uint8_t* blk = T.data.data() + (size_t(r) * bpr + bi) * kQ4PayloadBytes;
            std::memcpy(blk, &scale, 4); // store float scale first 4 bytes (simple)
            // remaining 14 bytes: pack nibbles (16 values need 8 bytes; pad)
            for (int i = 0; i < n; i += 2) {
                int q0 = std::max(-8, std::min(7, (int)std::round(W[r*cols+c0+i]/scale)));
                int q1 = (i+1<n) ? std::max(-8, std::min(7, (int)std::round(W[r*cols+c0+i+1]/scale))) : 0;
                blk[4 + i/2] = (uint8_t)((q0+8) | ((q1+8)<<4));
            }
        }
    }
    return T;
}

void gemv_kquant(const KQuantTensor& W, const float* x, float* y) {
    if (!x || !y || W.rows <= 0 || W.cols <= 0) {
        throw std::invalid_argument("gemv_kquant received invalid input");
    }
    if (W.type == KQuantType::Q8_0) {
        const int bs = 32;
        int bpr = (W.cols + bs - 1) / bs;
        if (W.data.size() != size_t(W.rows) * size_t(W.cols) ||
            W.scales.size() != size_t(W.rows) * size_t(bpr)) {
            throw std::invalid_argument("invalid Q8 payload size");
        }
        for (int r = 0; r < W.rows; ++r) {
            float s = 0.f;
            for (int bi = 0; bi < bpr; ++bi) {
                float scale = W.scales[r * bpr + bi];
                int c0 = bi * bs;
                int n = std::min(bs, W.cols - c0);
                for (int i = 0; i < n; ++i) {
                    int q = int(W.data[r * W.cols + c0 + i]) - 128;
                    s += q * scale * x[c0 + i];
                }
            }
            y[r] = s;
        }
        return;
    }
    // Q4 family
    const int bs = kQ4BlockSize;
    int bpr = (W.cols + bs - 1) / bs;
    const size_t expected = size_t(W.rows) * size_t(bpr) * kQ4PayloadBytes;
    if (W.data.size() != expected) throw std::invalid_argument("invalid Q4 payload size");
    for (int r = 0; r < W.rows; ++r) {
        float s = 0.f;
        for (int bi = 0; bi < bpr; ++bi) {
            const uint8_t* blk = W.data.data() + (size_t(r) * bpr + bi) * kQ4PayloadBytes;
            float scale;
            std::memcpy(&scale, blk, 4);
            int c0 = bi * bs;
            int n = std::min(bs, W.cols - c0);
            for (int i = 0; i < n; i += 2) {
                int q0 = (blk[4 + i/2] & 0xF) - 8;
                int q1 = ((blk[4 + i/2] >> 4) & 0xF) - 8;
                s += q0 * scale * x[c0 + i];
                if (i + 1 < n) s += q1 * scale * x[c0 + i + 1];
            }
        }
        y[r] = s;
    }
}

} // namespace quant
} // namespace winner
