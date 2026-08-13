// CASCADE F0 q4k — GEMV fundido AVX2 (dequant dentro do produto, sem W denso)
//
// Layout por super-bloco de 256 colunas (144 bytes = 4.5 bpw):
//   d fp16 | dmin fp16 | 12B: 8 sub-escalas u6 + 8 sub-mins i6 | 128B: q u4
//   grupo g=32; byte k do grupo: val[k] = nibble baixo, val[k+16] = nibble alto
//   w = d*sc6[j]*q + dmin*sm6[j]      (sm6 com sinal, -31..31)
//
// GEMV assimetrico sem dequant denso:
//   y[row] = sum_j ( s_j * dot(q_j, x_j) + m_j * sumx_j )
//   sumx_j (soma de x por grupo) e computado 1x por chamada — O(cols).
//
// Compilar: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC q4k_gemv.c -o libq4k.so
// Bench:    gcc -O3 -mavx2 -mfma -mf16c -fopenmp -DQ4K_BENCH q4k_gemv.c -o bench_q4k
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRP 32
#define SUP 256
#define GPS (SUP / GRP)
#define SUP_BYTES 144

static inline float f16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)h)));
}

static inline void decode_scales(const uint8_t *sc, float d, float dmin,
                                 float *s8, float *m8) {
    uint64_t su = 0, mu = 0;
    memcpy(&su, sc, 6);
    memcpy(&mu, sc + 6, 6);
    for (int j = 0; j < GPS; ++j) {
        s8[j] = d * (float)((su >> (6 * j)) & 63u);
        int mv = (int)((mu >> (6 * j)) & 63u) - 31;
        m8[j] = dmin * (float)mv;
    }
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// dot de um grupo de 32 valores u4 (16B packed) com x[32] fp32
static inline __m256 group_dot(const uint8_t *qp, const float *x, __m256 acc) {
    __m128i raw = _mm_loadu_si128((const __m128i *)qp);
    __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
    __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
    __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
    __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
    __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
    acc = _mm256_fmadd_ps(q0, _mm256_loadu_ps(x + 0), acc);
    acc = _mm256_fmadd_ps(q1, _mm256_loadu_ps(x + 8), acc);
    acc = _mm256_fmadd_ps(q2, _mm256_loadu_ps(x + 16), acc);
    acc = _mm256_fmadd_ps(q3, _mm256_loadu_ps(x + 24), acc);
    return acc;
}

// y[rows] = W(q4k packed) @ x[cols]; sumx: workspace cols/GRP floats
void q4k_gemv(int rows, int cols, const uint8_t *packed,
              const float *x, float *sumx, float *y) {
    const int nsup = cols / SUP;
    const int ngrp = cols / GRP;
    for (int gj = 0; gj < ngrp; ++gj) {
        __m256 a = _mm256_add_ps(_mm256_loadu_ps(x + gj * GRP),
                                 _mm256_loadu_ps(x + gj * GRP + 8));
        __m256 b = _mm256_add_ps(_mm256_loadu_ps(x + gj * GRP + 16),
                                 _mm256_loadu_ps(x + gj * GRP + 24));
        sumx[gj] = hsum256(_mm256_add_ps(a, b));
    }
    const size_t row_bytes = (size_t)nsup * SUP_BYTES;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t *rp = packed + (size_t)r * row_bytes;
        __m256 accv = _mm256_setzero_ps();
        __m256 maccv = _mm256_setzero_ps();
        for (int s = 0; s < nsup; ++s) {
            const uint8_t *sp = rp + (size_t)s * SUP_BYTES;
            float d = f16_to_f32(*(const uint16_t *)sp);
            float dmin = f16_to_f32(*(const uint16_t *)(sp + 2));
            float s8[GPS] __attribute__((aligned(32)));
            float m8[GPS] __attribute__((aligned(32)));
            decode_scales(sp + 4, d, dmin, s8, m8);
            const uint8_t *qs = sp + 16;
            const float *xs = x + s * SUP;
            for (int j = 0; j < GPS; ++j) {
                __m256 dv = group_dot(qs + j * (GRP / 2), xs + j * GRP,
                                      _mm256_setzero_ps());
                accv = _mm256_fmadd_ps(_mm256_set1_ps(s8[j]), dv, accv);
            }
            maccv = _mm256_fmadd_ps(_mm256_load_ps(m8),
                                    _mm256_loadu_ps(sumx + s * GPS), maccv);
        }
        y[r] = hsum256(accv) + hsum256(maccv);
    }
}

static inline int hsum256_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// Variante int8: ativacoes quantizadas por super-bloco (1 escala/256, estilo
// Q8_K) + maddubs; sub-escalas u6 dobradas no acumulador inteiro via madd.
// xq_re: layout rearranjado por par de grupos (ver q4k_prepare_x_i8).
void q4k_prepare_x_i8(int cols, const float *x, int8_t *xq_re,
                      float *qsx, float *sumx) {
    const int nsup = cols / SUP;
    for (int gj = 0; gj < cols / GRP; ++gj) {
        float s = 0.f;
        for (int i = 0; i < GRP; ++i) s += x[gj * GRP + i];
        sumx[gj] = s;
    }
    for (int s = 0; s < nsup; ++s) {
        const float *xs = x + s * SUP;
        float mx = 0.f;
        for (int i = 0; i < SUP; ++i) {
            float a = xs[i] < 0 ? -xs[i] : xs[i];
            if (a > mx) mx = a;
        }
        float q = mx > 0 ? mx / 127.f : 1.f;
        qsx[s] = q;
        int8_t *dst = xq_re + (size_t)s * SUP;
        for (int p = 0; p < GPS / 2; ++p) {
            const float *g0 = xs + (2 * p) * GRP;
            const float *g1 = xs + (2 * p + 1) * GRP;
            for (int k = 0; k < 16; ++k) {
                dst[p * 64 + k] = (int8_t)lrintf(g0[k] / q);
                dst[p * 64 + 16 + k] = (int8_t)lrintf(g1[k] / q);
                dst[p * 64 + 32 + k] = (int8_t)lrintf(g0[16 + k] / q);
                dst[p * 64 + 48 + k] = (int8_t)lrintf(g1[16 + k] / q);
            }
        }
    }
}

#define BCAST_PAIR(p) _mm256_setr_epi8( \
    4*(p), 4*(p)+1, 4*(p), 4*(p)+1, 4*(p), 4*(p)+1, 4*(p), 4*(p)+1, \
    4*(p), 4*(p)+1, 4*(p), 4*(p)+1, 4*(p), 4*(p)+1, 4*(p), 4*(p)+1, \
    4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3, \
    4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3, 4*(p)+2, 4*(p)+3)

void q4k_gemv_i8(int rows, int cols, const uint8_t *packed,
                 const int8_t *xq_re, const float *qsx, const float *sumx,
                 float *y) {
    const int nsup = cols / SUP;
    const size_t row_bytes = (size_t)nsup * SUP_BYTES;
    const __m256i nib = _mm256_set1_epi8(0x0F);
    const __m256i bc0 = BCAST_PAIR(0), bc1 = BCAST_PAIR(1);
    const __m256i bc2 = BCAST_PAIR(2), bc3 = BCAST_PAIR(3);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t *rp = packed + (size_t)r * row_bytes;
        float acc = 0.f;
        __m256 maccv = _mm256_setzero_ps();
        for (int s = 0; s < nsup; ++s) {
            const uint8_t *sp = rp + (size_t)s * SUP_BYTES;
            float d = f16_to_f32(*(const uint16_t *)sp);
            float dmin = f16_to_f32(*(const uint16_t *)(sp + 2));
            uint64_t su = 0, mu = 0;
            memcpy(&su, sp + 4, 6);
            memcpy(&mu, sp + 10, 6);
            int16_t sc16[GPS] __attribute__((aligned(16)));
            int16_t mi16[GPS] __attribute__((aligned(16)));
            for (int j = 0; j < GPS; ++j) {
                sc16[j] = (int16_t)((su >> (6 * j)) & 63u);
                mi16[j] = (int16_t)((int)((mu >> (6 * j)) & 63u) - 31);
            }
            __m256i scall = _mm256_broadcastsi128_si256(
                _mm_load_si128((const __m128i *)sc16));
            __m256 m8v = _mm256_mul_ps(
                _mm256_set1_ps(dmin),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_load_si128((const __m128i *)mi16))));
            const uint8_t *qs = sp + 16;
            const int8_t *xs = xq_re + (size_t)s * SUP;
            __m256i acci = _mm256_setzero_si256();
            __m256i raw, qlo, qhi, xlo, xhi, p16;
#define PAIR_STEP(p, bc) \
            raw = _mm256_loadu_si256((const __m256i *)(qs + (p) * 32)); \
            qlo = _mm256_and_si256(raw, nib); \
            qhi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), nib); \
            xlo = _mm256_loadu_si256((const __m256i *)(xs + (p) * 64)); \
            xhi = _mm256_loadu_si256((const __m256i *)(xs + (p) * 64 + 32)); \
            p16 = _mm256_add_epi16(_mm256_maddubs_epi16(qlo, xlo), \
                                   _mm256_maddubs_epi16(qhi, xhi)); \
            acci = _mm256_add_epi32(acci, _mm256_madd_epi16( \
                p16, _mm256_shuffle_epi8(scall, bc)));
            PAIR_STEP(0, bc0)
            PAIR_STEP(1, bc1)
            PAIR_STEP(2, bc2)
            PAIR_STEP(3, bc3)
#undef PAIR_STEP
            acc += d * qsx[s] * (float)hsum256_i32(acci);
            maccv = _mm256_fmadd_ps(m8v, _mm256_loadu_ps(sumx + s * GPS), maccv);
        }
        y[r] = acc + hsum256(maccv);
    }
}

#ifdef Q4K_BENCH
#include <stdio.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    // simula um modelo classe Qwen-3B: 36 camadas x 6 matrizes
    const int HID = 2048, FFN = 11008;
    const int shapes[6][2] = {{HID, HID}, {HID, HID}, {HID, HID},
                              {FFN, HID}, {FFN, HID}, {HID, FFN}};
    const int LAYERS = 36;
    size_t total = 0;
    uint8_t *bufs[36][6];
    for (int l = 0; l < LAYERS; ++l)
        for (int m = 0; m < 6; ++m) {
            int rows = shapes[m][0], cols = shapes[m][1];
            size_t b = (size_t)rows * (cols / SUP) * SUP_BYTES;
            bufs[l][m] = (uint8_t *)malloc(b);
            for (size_t i = 0; i < b; i += 4096) bufs[l][m][i] = (uint8_t)i;
            uint16_t one = 0x3C00, tiny = 0x1400;
            for (size_t s = 0; s < b; s += SUP_BYTES) {
                memcpy(bufs[l][m] + s, &tiny, 2);
                memcpy(bufs[l][m] + s + 2, &tiny, 2);
            }
            (void)one;
            total += b;
        }
    float *x_h = (float *)aligned_alloc(64, FFN * sizeof(float));
    for (int i = 0; i < FFN; ++i) x_h[i] = (float)(i % 13) * 0.01f - 0.06f;
    float *sumx = (float *)aligned_alloc(64, (FFN / GRP) * sizeof(float));
    float *y = (float *)aligned_alloc(64, FFN * sizeof(float));
    int8_t *xq = (int8_t *)aligned_alloc(64, FFN);
    float *qsx = (float *)aligned_alloc(64, (FFN / SUP) * sizeof(float));

    for (int m = 0; m < 6; ++m)
        q4k_gemv(shapes[m][0], shapes[m][1], bufs[0][m], x_h, sumx, y);

    const int REPS = 5;
    double t0 = now_s();
    for (int rep = 0; rep < REPS; ++rep)
        for (int l = 0; l < LAYERS; ++l)
            for (int m = 0; m < 6; ++m)
                q4k_gemv(shapes[m][0], shapes[m][1], bufs[l][m], x_h, sumx, y);
    double dt = (now_s() - t0) / REPS;

    printf("{\"kernel\": \"fp32\", \"modelo_simulado_gb\": %.2f, "
           "\"latencia_token_s\": %.4f, \"tok_s\": %.3f, "
           "\"banda_efetiva_gbs\": %.2f}\n",
           total / 1e9, dt, 1.0 / dt, total / dt / 1e9);

    for (int m = 0; m < 6; ++m) {
        q4k_prepare_x_i8(shapes[m][1], x_h, xq, qsx, sumx);
        q4k_gemv_i8(shapes[m][0], shapes[m][1], bufs[0][m], xq, qsx, sumx, y);
    }
    t0 = now_s();
    for (int rep = 0; rep < REPS; ++rep)
        for (int l = 0; l < LAYERS; ++l)
            for (int m = 0; m < 6; ++m) {
                q4k_prepare_x_i8(shapes[m][1], x_h, xq, qsx, sumx);
                q4k_gemv_i8(shapes[m][0], shapes[m][1], bufs[l][m],
                            xq, qsx, sumx, y);
            }
    double dt8 = (now_s() - t0) / REPS;

    printf("{\"kernel\": \"int8-maddubs\", \"modelo_simulado_gb\": %.2f, "
           "\"latencia_token_s\": %.4f, \"tok_s\": %.3f, "
           "\"banda_efetiva_gbs\": %.2f}\n",
           total / 1e9, dt8, 1.0 / dt8, total / dt8 / 1e9);
    return 0;
}
#endif

// ---------------------------------------------------------------------------
// F1 low-rank GEMV (corrigido): y[out] += U @ (S * (Vt @ x))
//   Vt: (rank, in)  ROW-MAJOR  — transposto na carga, leitura contigua
//   U : (out, rank) ROW-MAJOR  — leitura contigua por linha
// Corrige o avx2_lowrank.cpp original: remove o loop morto e o gather por
// stride (_mm256_set_ps elemento a elemento), que desperdicava o AVX2.
// ---------------------------------------------------------------------------
void lowrank_gemv_f32(const float *x, int in_f,
                      const float *U, const float *S, const float *Vt,
                      int out_f, int rank, float *y) {
    float tmp[512];
    if (rank > 512) return;
    for (int r = 0; r < rank; ++r) {
        const float *vr = Vt + (size_t)r * in_f;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in_f; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),
                                  _mm256_loadu_ps(vr + i), acc);
        float a = hsum256(acc);
        for (; i < in_f; ++i) a += x[i] * vr[i];
        tmp[r] = a * S[r];
    }
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out_f; ++o) {
        const float *uo = U + (size_t)o * rank;
        __m256 acc = _mm256_setzero_ps();
        int r = 0;
        for (; r + 8 <= rank; r += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(uo + r),
                                  _mm256_loadu_ps(tmp + r), acc);
        float a = hsum256(acc);
        for (; r < rank; ++r) a += uo[r] * tmp[r];
        y[o] += a;
    }
}
