#pragma once
// CASCADE CUDA low-rank residual — skeleton (compile with nvcc when GPU available)
#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace cascade {
namespace cuda {

// y += (x @ V) * S @ U.T   — simple reference kernel (not production-tuned)
__global__ void lowrank_residual_kernel(
    const float* x, int in_f,
    const float* U, const float* S, const float* V,
    int out_f, int rank,
    float* y)
{
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_f) return;
  float acc = 0.f;
  for (int r = 0; r < rank; ++r) {
    float dot = 0.f;
    for (int i = 0; i < in_f; ++i) dot += x[i] * V[i * rank + r];
    acc += (dot * S[r]) * U[o * rank + r];
  }
  y[o] += acc;
}

inline void lowrank_residual_launch(
    const float* x, int in_f,
    const float* U, const float* S, const float* V,
    int out_f, int rank,
    float* y, cudaStream_t stream = 0)
{
  int threads = 256;
  int blocks = (out_f + threads - 1) / threads;
  lowrank_residual_kernel<<<blocks, threads, 0, stream>>>(x, in_f, U, S, V, out_f, rank, y);
}

} // namespace cuda
} // namespace cascade
#endif
