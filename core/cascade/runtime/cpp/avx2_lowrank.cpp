// CASCADE low-rank residual GEMV — AVX2 when available
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cascade {

// y[out] += ((x · V) * S) · U^T
// V: in×r, U: out×r, S: r, x: in, y: out
void lowrank_gemv_f32(
    const float* x, int in_f,
    const float* U, const float* S, const float* V,
    int out_f, int rank,
    float* y)
{
  // SUPERSEDIDO por core/cascade/runtime_v2/kernels/kernels.c
  // (lowrank_gemv_f32), que transpõe V para (rank, in) e lê contíguo. Este
  // arquivo fica como referência do caminho C1; dois defeitos foram removidos:
  //  1. um loop `for (r)` cujo corpo calculava `acc` e `Vr` e não usava nenhum
  //     dos dois — código morto que nunca contribuiu para o resultado;
  //  2. a "vetorização" abaixo monta o vetor com _mm256_set_ps a partir de 8
  //     cargas ESCALARES em stride `rank` (V é (in, rank) row-major, então
  //     V[i,r] e V[i+1,r] estão rank*4 bytes distantes). Não é uma carga
  //     vetorial: é um gather emulado, e pode ser mais lento que o laço
  //     escalar. O v2 resolve transpondo V na carga do módulo.
  std::vector<float> tmp(static_cast<size_t>(rank), 0.f);
  // tmp[r] = dot(x, V[:,r]) * S[r]; V é (in, rank) row-major: V[i,r] em i*rank+r
  for (int r = 0; r < rank; ++r) {
    float acc = 0.f;
    int i = 0;
#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= in_f; i += 8) {
      __m256 vx = _mm256_loadu_ps(x + i);
      __m256 vv = _mm256_set_ps(
        V[(i+7)*rank + r], V[(i+6)*rank + r], V[(i+5)*rank + r], V[(i+4)*rank + r],
        V[(i+3)*rank + r], V[(i+2)*rank + r], V[(i+1)*rank + r], V[(i+0)*rank + r]);
      vacc = _mm256_fmadd_ps(vx, vv, vacc);
    }
    alignas(32) float tmp8[8];
    _mm256_store_ps(tmp8, vacc);
    acc = tmp8[0]+tmp8[1]+tmp8[2]+tmp8[3]+tmp8[4]+tmp8[5]+tmp8[6]+tmp8[7];
#endif
    for (; i < in_f; ++i) acc += x[i] * V[i * rank + r];
    tmp[static_cast<size_t>(r)] = acc * S[r];
  }
  // y[o] += sum_r U[o,r] * tmp[r]
  for (int o = 0; o < out_f; ++o) {
    float acc = y[o];
    for (int r = 0; r < rank; ++r) {
      acc += U[o * rank + r] * tmp[static_cast<size_t>(r)];
    }
    y[o] = acc;
  }
}

} // namespace cascade
