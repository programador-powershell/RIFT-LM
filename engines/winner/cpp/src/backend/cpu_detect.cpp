#include "backend/cpu_detect.h"
#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  define WINNER_X86 1
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

#if defined(__aarch64__) || defined(__arm__)
#  define WINNER_ARM 1
#endif

namespace winner {
namespace backend {

#if defined(WINNER_X86)
static void cpuid(int leaf, int sub, int* regs) {
#  if defined(_MSC_VER)
    __cpuidex(regs, leaf, sub);
#  else
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#  endif
}

static int max_basic_leaf() {
    int regs[4] = {};
    cpuid(0, 0, regs);
    return regs[0];
}

static uint32_t max_extended_leaf() {
    int regs[4] = {};
    cpuid(static_cast<int>(0x80000000u), 0, regs);
    return static_cast<uint32_t>(regs[0]);
}

static uint64_t xgetbv0() {
#  if defined(_MSC_VER)
    return _xgetbv(0);
#  else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (uint64_t(edx) << 32) | eax;
#  endif
}
#endif

CpuFeatures detect_cpu() {
    CpuFeatures f;
    f.n_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (f.n_cores < 1) f.n_cores = 1;

#if defined(WINNER_X86)
    int r[4] = {};
    const int max_basic = max_basic_leaf();
    const uint32_t max_extended = max_extended_leaf();
    cpuid(0, 0, r);
    char vendor[13] = {};
    memcpy(vendor + 0, &r[1], 4);
    memcpy(vendor + 4, &r[3], 4);
    memcpy(vendor + 8, &r[2], 4);
    f.brand = vendor;

    if (max_basic < 1) return f;
    cpuid(1, 0, r);
    f.sse3  = (r[2] & (1 << 0)) != 0;
    f.ssse3 = (r[2] & (1 << 9)) != 0;
    bool osxsave = (r[2] & (1 << 27)) != 0;
    f.avx   = (r[2] & (1 << 28)) != 0;
    f.fma   = (r[2] & (1 << 12)) != 0;
    f.f16c  = (r[2] & (1 << 29)) != 0;

    uint64_t xcr0 = 0;
    if (osxsave) xcr0 = xgetbv0();
    bool ymm_ok = osxsave && ((xcr0 & 0x6) == 0x6);
    bool zmm_ok = ymm_ok && ((xcr0 & 0xE0) == 0xE0);
    bool amx_ok = zmm_ok && ((xcr0 & (3ull << 17)) == (3ull << 17));

    if (!ymm_ok) {
        f.avx = false;
        f.fma = false;
        f.f16c = false;
    }

    if (max_basic >= 7) {
        cpuid(7, 0, r);
        const uint32_t max_subleaf = static_cast<uint32_t>(r[0]);
        f.avx2 = ymm_ok && ((r[1] & (1 << 5)) != 0);
        f.avx512f = zmm_ok && ((r[1] & (1 << 16)) != 0);
        f.avx512_vnni = zmm_ok && ((r[2] & (1 << 11)) != 0);
        f.amx_tile = amx_ok && ((r[3] & (1 << 24)) != 0);
        f.amx_int8 = amx_ok && ((r[3] & (1 << 25)) != 0);
        f.amx_bf16 = amx_ok && ((r[3] & (1 << 22)) != 0);
        if (max_subleaf >= 1) {
            cpuid(7, 1, r);
            f.avx_vnni = ymm_ok && ((r[0] & (1 << 4)) != 0);
            f.avx512_bf16 = zmm_ok && ((r[0] & (1 << 5)) != 0);
        }
    }

    // Brand string
    if (max_extended >= 0x80000004u) {
        char brand[49] = {};
        cpuid(static_cast<int>(0x80000002u), 0, r); memcpy(brand + 0,  r, 16);
        cpuid(static_cast<int>(0x80000003u), 0, r); memcpy(brand + 16, r, 16);
        cpuid(static_cast<int>(0x80000004u), 0, r); memcpy(brand + 32, r, 16);
        f.brand = brand;
        while (!f.brand.empty() && f.brand.front() == ' ') f.brand.erase(f.brand.begin());
        while (!f.brand.empty() && f.brand.back() == ' ') f.brand.pop_back();
    }

#elif defined(WINNER_ARM)
    f.neon = true;
    f.brand = "ARM";
#  if defined(__ARM_FEATURE_DOTPROD)
    f.dotprod = true;
#  endif
#  if defined(__ARM_FEATURE_MATMUL_INT8)
    f.i8mm = true;
#  endif
#  if defined(__ARM_FEATURE_SVE)
    f.sve = true;
#  endif
#else
    f.brand = "unknown";
#endif
    return f;
}

int cpu_score(const CpuFeatures& f) {
    int s = 1;
    if (f.avx2) s += 10;
    if (f.fma) s += 2;
    if (f.avx_vnni) s += 15;
    if (f.avx512f) s += 20;
    if (f.avx512_vnni) s += 25;
    if (f.amx_int8) s += 40;
    if (f.neon) s += 10;
    if (f.i8mm) s += 20;
    if (f.sve) s += 15;
    s += f.n_cores;
    return s;
}

const char* best_cpu_isa_name(const CpuFeatures& f) {
    if (f.amx_int8) return "AMX-INT8";
    if (f.avx512_vnni) return "AVX512-VNNI";
    if (f.avx512f) return "AVX512";
    if (f.avx_vnni) return "AVX-VNNI";
    if (f.avx2) return "AVX2";
    if (f.i8mm) return "ARM-I8MM";
    if (f.neon) return "NEON";
    return "SCALAR";
}

} // namespace backend
} // namespace winner
