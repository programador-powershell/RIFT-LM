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

// g=32: super 144B (8 sub-escalas) | g=64: super 138B (4 sub-escalas,
// duplicadas em 8 slots — mesmo inner loop, so muda o header).
void q4k_gemv_i8_g(int rows, int cols, int g, const uint8_t *packed,
                   const int8_t *xq_re, const float *qsx, const float *sumx,
                   float *y) {
    const int nsup = cols / SUP;
    const int sup_bytes = (g == 64) ? 138 : 144;
    const int scb = (g == 64) ? 3 : 6;
    const int voff = 4 + 2 * scb;
    const size_t row_bytes = (size_t)nsup * sup_bytes;
    const __m256i nib = _mm256_set1_epi8(0x0F);
    const __m256i bc0 = BCAST_PAIR(0), bc1 = BCAST_PAIR(1);
    const __m256i bc2 = BCAST_PAIR(2), bc3 = BCAST_PAIR(3);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t *rp = packed + (size_t)r * row_bytes;
        float acc = 0.f;
        __m256 maccv = _mm256_setzero_ps();
        for (int s = 0; s < nsup; ++s) {
            const uint8_t *sp = rp + (size_t)s * sup_bytes;
            float d = f16_to_f32(*(const uint16_t *)sp);
            float dmin = f16_to_f32(*(const uint16_t *)(sp + 2));
            uint64_t su = 0, mu = 0;
            memcpy(&su, sp + 4, scb);
            memcpy(&mu, sp + 4 + scb, scb);
            int16_t sc16[GPS] __attribute__((aligned(16)));
            int16_t mi16[GPS] __attribute__((aligned(16)));
            if (g == 64) {
                for (int j = 0; j < 4; ++j) {
                    int16_t sv = (int16_t)((su >> (6 * j)) & 63u);
                    int16_t mv = (int16_t)((int)((mu >> (6 * j)) & 63u) - 31);
                    sc16[2 * j] = sc16[2 * j + 1] = sv;
                    mi16[2 * j] = mi16[2 * j + 1] = mv;
                }
            } else {
                for (int j = 0; j < GPS; ++j) {
                    sc16[j] = (int16_t)((su >> (6 * j)) & 63u);
                    mi16[j] = (int16_t)((int)((mu >> (6 * j)) & 63u) - 31);
                }
            }
            __m256i scall = _mm256_broadcastsi128_si256(
                _mm_load_si128((const __m128i *)sc16));
            __m256 m8v = _mm256_mul_ps(
                _mm256_set1_ps(dmin),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_load_si128((const __m128i *)mi16))));
            const uint8_t *qs = sp + voff;
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

// compat: assinatura antiga = g32
void q4k_gemv_i8(int rows, int cols, const uint8_t *packed,
                 const int8_t *xq_re, const float *qsx, const float *sumx,
                 float *y) {
    q4k_gemv_i8_g(rows, cols, 32, packed, xq_re, qsx, sumx, y);
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

// ---------------------------------------------------------------------------
// v2.1 — correcoes ditadas pela bateria E2E (decomposicao da PPL):
//   (a) ativacoes int8 com escala POR GRUPO DE 32 (estilo Q8_0) em vez de por
//       super-256: recupera ~0.6 de PPL medida; custo ~zero.
//   (b) GEMV q8 rowwise (pesos int8 + escala fp16 por linha) para tensores
//       promovidos (cabeca/v_proj): +1.09 PPL medida da cabeca a 4.5 bpw.
// ---------------------------------------------------------------------------

void q4k_prepare_x_i8_32(int cols, const float *x, int8_t *xq_re,
                         float *qs32, float *sumx) {
    const int nsup = cols / SUP;
    for (int gj = 0; gj < cols / GRP; ++gj) {
        float s = 0.f, mx = 0.f;
        for (int i = 0; i < GRP; ++i) {
            float v = x[gj * GRP + i];
            s += v;
            float a = v < 0 ? -v : v;
            if (a > mx) mx = a;
        }
        sumx[gj] = s;
        qs32[gj] = mx > 0 ? mx / 127.f : 1.f;
    }
    for (int sblk = 0; sblk < nsup; ++sblk) {
        const float *xs = x + sblk * SUP;
        int8_t *dst = xq_re + (size_t)sblk * SUP;
        for (int p = 0; p < GPS / 2; ++p) {
            int g0 = sblk * GPS + 2 * p, g1 = g0 + 1;
            const float *a = xs + (2 * p) * GRP;
            const float *b = xs + (2 * p + 1) * GRP;
            for (int k = 0; k < 16; ++k) {
                dst[p * 64 + k]      = (int8_t)lrintf(a[k] / qs32[g0]);
                dst[p * 64 + 16 + k] = (int8_t)lrintf(b[k] / qs32[g1]);
                dst[p * 64 + 32 + k] = (int8_t)lrintf(a[16 + k] / qs32[g0]);
                dst[p * 64 + 48 + k] = (int8_t)lrintf(b[16 + k] / qs32[g1]);
            }
        }
    }
}

void q4k_gemv_i8_g32acts(int rows, int cols, int g, const uint8_t *packed,
                         const int8_t *xq_re, const float *qs32,
                         const float *sumx, float *y) {
    const int nsup = cols / SUP;
    const int sup_bytes = (g == 64) ? 138 : 144;
    const int scb = (g == 64) ? 3 : 6;
    const int voff = 4 + 2 * scb;
    const size_t row_bytes = (size_t)nsup * sup_bytes;
    const __m256i nib = _mm256_set1_epi8(0x0F);
    const __m256i bc0 = BCAST_PAIR(0), bc1 = BCAST_PAIR(1);
    const __m256i bc2 = BCAST_PAIR(2), bc3 = BCAST_PAIR(3);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t *rp = packed + (size_t)r * row_bytes;
        float acc = 0.f;
        __m256 maccv = _mm256_setzero_ps();
        for (int s = 0; s < nsup; ++s) {
            const uint8_t *sp = rp + (size_t)s * sup_bytes;
            float d = f16_to_f32(*(const uint16_t *)sp);
            float dmin = f16_to_f32(*(const uint16_t *)(sp + 2));
            uint64_t su = 0, mu = 0;
            memcpy(&su, sp + 4, scb);
            memcpy(&mu, sp + 4 + scb, scb);
            int16_t sc16[GPS] __attribute__((aligned(16)));
            int16_t mi16[GPS] __attribute__((aligned(16)));
            if (g == 64) {
                for (int j = 0; j < 4; ++j) {
                    int16_t sv = (int16_t)((su >> (6 * j)) & 63u);
                    int16_t mv = (int16_t)((int)((mu >> (6 * j)) & 63u) - 31);
                    sc16[2 * j] = sc16[2 * j + 1] = sv;
                    mi16[2 * j] = mi16[2 * j + 1] = mv;
                }
            } else {
                for (int j = 0; j < GPS; ++j) {
                    sc16[j] = (int16_t)((su >> (6 * j)) & 63u);
                    mi16[j] = (int16_t)((int)((mu >> (6 * j)) & 63u) - 31);
                }
            }
            __m256i scall = _mm256_broadcastsi128_si256(
                _mm_load_si128((const __m128i *)sc16));
            __m256 m8v = _mm256_mul_ps(
                _mm256_set1_ps(dmin),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_load_si128((const __m128i *)mi16))));
            const uint8_t *qs = sp + voff;
            const int8_t *xs = xq_re + (size_t)s * SUP;
            const float *qg = qs32 + s * GPS;
            __m256 accf = _mm256_setzero_ps();
            __m256i raw, qlo, qhi, xlo, xhi, p16, i32;
#define PAIR_STEP32(p, bc) \
            raw = _mm256_loadu_si256((const __m256i *)(qs + (p) * 32)); \
            qlo = _mm256_and_si256(raw, nib); \
            qhi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), nib); \
            xlo = _mm256_loadu_si256((const __m256i *)(xs + (p) * 64)); \
            xhi = _mm256_loadu_si256((const __m256i *)(xs + (p) * 64 + 32)); \
            p16 = _mm256_add_epi16(_mm256_maddubs_epi16(qlo, xlo), \
                                   _mm256_maddubs_epi16(qhi, xhi)); \
            i32 = _mm256_madd_epi16(p16, _mm256_shuffle_epi8(scall, bc)); \
            accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32), \
                _mm256_set_m128(_mm_set1_ps(qg[2 * (p) + 1]), \
                                _mm_set1_ps(qg[2 * (p)])), accf);
            PAIR_STEP32(0, bc0)
            PAIR_STEP32(1, bc1)
            PAIR_STEP32(2, bc2)
            PAIR_STEP32(3, bc3)
#undef PAIR_STEP32
            acc += d * hsum256(accf);
            maccv = _mm256_fmadd_ps(m8v, _mm256_loadu_ps(sumx + s * GPS), maccv);
        }
        y[r] = acc + hsum256(maccv);
    }
}

// q8 rowwise: pesos int8 (linha a linha) + escala fp16 por linha.
// y[r] = rs[r] * sum_g qs32[g] * dot_i8(w8[g], x8[g])
void q8r_gemv_i8(int rows, int cols, const int8_t *w8, const uint16_t *rscale,
                 const int8_t *xq_lin, const float *qs32, float *y) {
    const int ngrp = cols / GRP;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const int8_t *wr = w8 + (size_t)r * cols;
        __m256 accf = _mm256_setzero_ps();
        for (int gj = 0; gj < ngrp; ++gj) {
            __m256i wv = _mm256_loadu_si256((const __m256i *)(wr + gj * GRP));
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq_lin + gj * GRP));
            __m256i wa = _mm256_sign_epi8(wv, wv);
            __m256i xb = _mm256_sign_epi8(xv, wv);
            __m256i p16 = _mm256_maddubs_epi16(wa, xb);
            __m256i i32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
            accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32),
                                   _mm256_set1_ps(qs32[gj]), accf);
        }
        y[r] = f16_to_f32(rscale[r]) * hsum256(accf);
    }
}

// quantizacao linear das ativacoes (sem rearranjo) p/ q8r
void q8r_prepare_x(int cols, const float *x, int8_t *xq_lin, float *qs32) {
    for (int gj = 0; gj < cols / GRP; ++gj) {
        float mx = 0.f;
        for (int i = 0; i < GRP; ++i) {
            float a = x[gj * GRP + i] < 0 ? -x[gj * GRP + i] : x[gj * GRP + i];
            if (a > mx) mx = a;
        }
        float q = mx > 0 ? mx / 127.f : 1.f;
        qs32[gj] = q;
        for (int i = 0; i < GRP; ++i)
            xq_lin[gj * GRP + i] = (int8_t)lrintf(x[gj * GRP + i] / q);
    }
}

// ---------------------------------------------------------------------------
// v2.2 — tok/s: (1) API de LOTE: 1 regiao OMP para N GEMVs independentes
// (mata fork/join por chamada; 417 chamadas/token na Muse -> ~4);
// (2) unroll de 2 linhas + prefetch; (3) build VNNI opcional (-DUSE_VNNI,
// vpdpbusd) — build.sh gera libcascade_kernels_vnni.so e o loader escolhe.
// ---------------------------------------------------------------------------

#ifdef USE_VNNI
#define DOT_PAIR(qlo, qhi, xlo, xhi, scqs, accf) do { \
    __m256i _lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), (qlo), (xlo)); \
    __m256i _hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), (qhi), (xhi)); \
    accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(_lo, _hi)), \
                           (scqs), accf); } while (0)
#else
#define DOT_PAIR(qlo, qhi, xlo, xhi, scqs, accf) do { \
    __m256i _p16 = _mm256_add_epi16(_mm256_maddubs_epi16((qlo), (xlo)), \
                                    _mm256_maddubs_epi16((qhi), (xhi))); \
    __m256i _i32 = _mm256_madd_epi16(_p16, _mm256_set1_epi16(1)); \
    accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_i32), (scqs), accf); } while (0)
#endif

static void q4k_rows_i8_32(int r0, int r1, int cols, int g,
                           const uint8_t *packed, const int8_t *xq_re,
                           const float *qs32, const float *sumx, float *y) {
    const int nsup = cols / SUP;
    const int sup_bytes = (g == 64) ? 138 : 144;
    const int scb = (g == 64) ? 3 : 6;
    const int voff = 4 + 2 * scb;
    const size_t row_bytes = (size_t)nsup * sup_bytes;
    const __m256i nib = _mm256_set1_epi8(0x0F);
    for (int r = r0; r < r1; ++r) {
        const uint8_t *rp = packed + (size_t)r * row_bytes;
        _mm_prefetch((const char *)(rp + row_bytes), _MM_HINT_T1);
        float acc = 0.f;
        __m256 maccv = _mm256_setzero_ps();
        for (int s = 0; s < nsup; ++s) {
            const uint8_t *sp = rp + (size_t)s * sup_bytes;
            _mm_prefetch((const char *)(sp + 2 * sup_bytes), _MM_HINT_T0);
            float d = f16_to_f32(*(const uint16_t *)sp);
            float dmin = f16_to_f32(*(const uint16_t *)(sp + 2));
            uint64_t su = 0, mu = 0;
            memcpy(&su, sp + 4, scb);
            memcpy(&mu, sp + 4 + scb, scb);
            float scqs[GPS] __attribute__((aligned(32)));
            float m8[GPS] __attribute__((aligned(32)));
            const float *qg = qs32 + s * GPS;
            if (g == 64) {
                for (int j = 0; j < 4; ++j) {
                    float sv = d * (float)((su >> (6 * j)) & 63u);
                    float mv = dmin * (float)((int)((mu >> (6 * j)) & 63u) - 31);
                    scqs[2 * j] = sv * qg[2 * j];
                    scqs[2 * j + 1] = sv * qg[2 * j + 1];
                    m8[2 * j] = mv; m8[2 * j + 1] = mv;
                }
            } else {
                for (int j = 0; j < GPS; ++j) {
                    scqs[j] = d * (float)((su >> (6 * j)) & 63u) * qg[j];
                    m8[j] = dmin * (float)((int)((mu >> (6 * j)) & 63u) - 31);
                }
            }
            const uint8_t *qs = sp + voff;
            const int8_t *xs = xq_re + (size_t)s * SUP;
            __m256 accf = _mm256_setzero_ps();
            for (int p = 0; p < 4; ++p) {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                __m256i qlo = _mm256_and_si256(raw, nib);
                __m256i qhi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), nib);
                __m256i xlo = _mm256_loadu_si256((const __m256i *)(xs + p * 64));
                __m256i xhi = _mm256_loadu_si256((const __m256i *)(xs + p * 64 + 32));
                __m256 sq = _mm256_set_m128(_mm_set1_ps(scqs[2 * p + 1]),
                                            _mm_set1_ps(scqs[2 * p]));
                DOT_PAIR(qlo, qhi, xlo, xhi, sq, accf);
            }
            acc += hsum256(accf);
            maccv = _mm256_fmadd_ps(_mm256_load_ps(m8),
                                    _mm256_loadu_ps(sumx + s * GPS), maccv);
        }
        y[r] = acc + hsum256(maccv);
    }
}

static void q8r_rows_i8(int r0, int r1, int cols, const int8_t *w8,
                        const uint16_t *rscale, const int8_t *xq_lin,
                        const float *qs32, float *y) {
    const int ngrp = cols / GRP;
    for (int r = r0; r < r1; ++r) {
        const int8_t *wr = w8 + (size_t)r * cols;
        _mm_prefetch((const char *)(wr + cols), _MM_HINT_T1);
        __m256 accf = _mm256_setzero_ps();
        for (int gj = 0; gj < ngrp; ++gj) {
            __m256i wv = _mm256_loadu_si256((const __m256i *)(wr + gj * GRP));
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq_lin + gj * GRP));
            __m256i wa = _mm256_sign_epi8(wv, wv);
            __m256i xb = _mm256_sign_epi8(xv, wv);
            __m256 qsv = _mm256_set1_ps(qs32[gj]);
            DOT_PAIR(wa, _mm256_setzero_si256(), xb,
                     _mm256_setzero_si256(), qsv, accf);
        }
        y[r] = f16_to_f32(rscale[r]) * hsum256(accf);
    }
}

typedef struct {
    int kind;                 // 0 = q4k(g32|64), 2 = q8r
    int rows, cols_padded, g;
    const uint8_t *packed;
    const int8_t *w8;
    const uint16_t *rscale;
    const int8_t *xq;
    const float *qs32;
    const float *sumx;
    float *y;
} cascade_gemv_desc;

// N GEMVs independentes numa UNICA regiao paralela (schedule dinamico por
// blocos de linhas; nowait — sem barreira entre itens).
void cascade_gemv_batch(int n, const cascade_gemv_desc *d) {
#pragma omp parallel
    {
        for (int i = 0; i < n; ++i) {
            const cascade_gemv_desc *t = &d[i];
            const int chunk = t->rows > 1024 ? 64 : 16;
            if (t->kind == 2) {
#pragma omp for schedule(dynamic, 1) nowait
                for (int b = 0; b < (t->rows + chunk - 1) / chunk; ++b) {
                    int r0 = b * chunk;
                    int r1 = r0 + chunk < t->rows ? r0 + chunk : t->rows;
                    q8r_rows_i8(r0, r1, t->cols_padded, t->w8, t->rscale,
                                t->xq, t->qs32, t->y);
                }
            } else {
#pragma omp for schedule(dynamic, 1) nowait
                for (int b = 0; b < (t->rows + chunk - 1) / chunk; ++b) {
                    int r0 = b * chunk;
                    int r1 = r0 + chunk < t->rows ? r0 + chunk : t->rows;
                    q4k_rows_i8_32(r0, r1, t->cols_padded, t->g, t->packed,
                                   t->xq, t->qs32, t->sumx, t->y);
                }
            }
        }
    }
}

// v2.2 single-tensor (mesmo miolo refatorado)
void q4k_gemv_i8_v22(int rows, int cols, int g, const uint8_t *packed,
                     const int8_t *xq_re, const float *qs32,
                     const float *sumx, float *y) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < (rows + 63) / 64; ++b) {
        int r0 = b * 64, r1 = r0 + 64 < rows ? r0 + 64 : rows;
        q4k_rows_i8_32(r0, r1, cols, g, packed, xq_re, qs32, sumx, y);
    }
}
