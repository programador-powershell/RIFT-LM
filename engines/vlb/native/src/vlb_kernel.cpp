#include "vlb_kernel.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(VLB_COMPILED_AVX2)
#include <immintrin.h>
#endif

namespace vlb {

float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16u;
    std::uint32_t exp = (h >> 10u) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1u;
                ++shift;
            }
            mant &= 0x03ffu;
            const std::uint32_t fp32_exp = static_cast<std::uint32_t>(127 - 15 - shift);
            bits = sign | (fp32_exp << 23u) | (mant << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        const std::uint32_t fp32_exp = exp + (127u - 15u);
        bits = sign | (fp32_exp << 23u) | (mant << 13u);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void validate(const Q8G64Matrix& m, const float* x, float* y) {
    if (!m.qweight || !m.fp16_scales || !x || !y) {
        throw std::invalid_argument("VLB Q8_G64 kernel received null pointer");
    }
    if (m.rows == 0 || m.cols == 0 || m.group_size == 0) {
        throw std::invalid_argument("VLB Q8_G64 kernel received invalid shape");
    }
}

void q8_g64_gemv_scalar(const Q8G64Matrix& m, const float* x, float* y) {
    validate(m, x, y);
    const std::size_t total = m.rows * m.cols;
    const std::size_t groups = (total + m.group_size - 1) / m.group_size;
    (void)groups;

    for (std::size_t row = 0; row < m.rows; ++row) {
        float sum = 0.0f;
        const std::size_t base = row * m.cols;
        for (std::size_t col = 0; col < m.cols; ++col) {
            const std::size_t flat = base + col;
            const std::size_t group = flat / m.group_size;
            const float scale = fp16_to_fp32(m.fp16_scales[group]);
            sum += static_cast<float>(m.qweight[flat]) * scale * x[col];
        }
        y[row] = sum;
    }
}

#if defined(VLB_COMPILED_AVX2)
static float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void q8_g64_gemv_avx2(const Q8G64Matrix& m, const float* x, float* y) {
    // Fast path requires group boundaries to align with every row. Gemma-family
    // Linear dimensions satisfy this for Q8_G64; other shapes fall back to the
    // exact scalar implementation instead of making an alignment assumption.
    if (m.group_size != 64 || (m.cols % 64) != 0) {
        q8_g64_gemv_scalar(m, x, y);
        return;
    }

    for (std::size_t row = 0; row < m.rows; ++row) {
        const std::size_t row_base = row * m.cols;
        __m256 acc = _mm256_setzero_ps();

        for (std::size_t col = 0; col < m.cols; col += 64) {
            const std::size_t flat = row_base + col;
            const std::size_t group = flat / 64;
            const __m256 scale = _mm256_set1_ps(fp16_to_fp32(m.fp16_scales[group]));

            for (std::size_t k = 0; k < 64; k += 8) {
                const auto* qptr = reinterpret_cast<const __m128i*>(m.qweight + flat + k);
                const __m128i q8 = _mm_loadl_epi64(qptr);
                const __m256i q32 = _mm256_cvtepi8_epi32(q8);
                const __m256 qf = _mm256_cvtepi32_ps(q32);
                const __m256 xv = _mm256_loadu_ps(x + col + k);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(qf, scale), xv, acc);
            }
        }
        y[row] = hsum256(acc);
    }
}
#endif

void q8_g64_gemv(const Q8G64Matrix& m, const float* x, float* y) {
#if defined(VLB_COMPILED_AVX2)
    q8_g64_gemv_avx2(m, x, y);
#else
    q8_g64_gemv_scalar(m, x, y);
#endif
}

const char* kernel_backend() noexcept {
#if defined(VLB_COMPILED_AVX2)
    return "VLB_Q8_G64_AVX2_FMA";
#else
    return "VLB_Q8_G64_SCALAR";
#endif
}

} // namespace vlb
