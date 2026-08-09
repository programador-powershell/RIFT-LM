/**
 * CPU feature detection (x86 + ARM)
 * Runtime dispatch like llama.cpp / ggml
 */
#pragma once
#include <cstdint>
#include <string>

namespace winner {
namespace backend {

struct CpuFeatures {
    // x86
    bool sse3 = false;
    bool ssse3 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool avx_vnni = false;      // 256-bit VNNI (Alder Lake+)
    bool avx512f = false;
    bool avx512_vnni = false;
    bool avx512_bf16 = false;
    bool amx_tile = false;
    bool amx_int8 = false;
    bool amx_bf16 = false;
    // ARM
    bool neon = false;
    bool dotprod = false;
    bool i8mm = false;
    bool sve = false;
    // meta
    int  n_cores = 1;
    std::string brand;
};

CpuFeatures detect_cpu();
int        cpu_score(const CpuFeatures& f);  // higher = better for our kernels
const char* best_cpu_isa_name(const CpuFeatures& f);

} // namespace backend
} // namespace winner
