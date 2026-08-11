/**
 * Kernel dispatch — select best implementation at runtime
 */
#pragma once
#include "backend/cpu_detect.h"
#include "backend/device.h"
#include <cstddef>

namespace winner {
namespace backend {

enum class KernelIsa : uint8_t {
    SCALAR = 0,
    AVX2,
    AVX_VNNI,
    AVX512_VNNI,
    AMX,
    NEON,
    CUDA,
    HIP
};

struct KernelTable {
    KernelIsa isa = KernelIsa::SCALAR;
    // F0 ternary / HQR GEMV
    void (*f0_gemv)(const void* W, const float* X, float* Y, int rows, int cols) = nullptr;
    // Fused low-rank residual: Y += (X·U)·Vᵀ
    void (*fused_lowrank)(const float* X, const float* U, const float* V,
                          float* Y, int dim, int rank) = nullptr;
    // Full fused stage
    void (*fused_stage)(const float* X, float* Y, const void* f0,
                        const float* U, const float* V,
                        int dim, int rank, bool apply_residual) = nullptr;
};

// Select best CPU kernel table for detected features
KernelTable select_cpu_kernels(const CpuFeatures& feat);

// Select GPU kernels if device available (returns empty/scalar if not)
KernelTable select_gpu_kernels(const DeviceInfo& gpu);

const char* isa_name(KernelIsa isa);

} // namespace backend
} // namespace winner
