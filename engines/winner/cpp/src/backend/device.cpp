#include "backend/device.h"
#include <limits>
#include <cstdio>
#include <fstream>
#include <algorithm>
#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#elif defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#endif

// Optional CUDA
#if defined(WINNER_HAS_CUDA)
#  include <cuda_runtime.h>
#endif

namespace winner {
namespace backend {

static size_t system_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return static_cast<size_t>(status.ullTotalPhys);
#elif defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t kb = 0;
    while (f >> key) {
        if (key == "MemTotal:") {
            f >> kb;
            return kb * 1024ull;
        }
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
#endif
    return 16ull * 1024 * 1024 * 1024; // fallback 16 GB
}

std::vector<DeviceInfo> enumerate_devices() {
    std::vector<DeviceInfo> out;

    // CPU always present
    DeviceInfo cpu;
    cpu.type = DeviceType::CPU;
    cpu.index = 0;
    cpu.name = "CPU";
    cpu.total_memory_bytes = system_ram_bytes();
    cpu.free_memory_bytes  = cpu.total_memory_bytes / 2; // rough
    cpu.available = true;
    out.push_back(cpu);

#if defined(WINNER_HAS_CUDA)
    int n = 0;
    if (cudaGetDeviceCount(&n) == cudaSuccess) {
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
            size_t free_b = 0, total_b = 0;
            cudaSetDevice(i);
            cudaMemGetInfo(&free_b, &total_b);

            DeviceInfo g;
            g.type = DeviceType::CUDA;
            g.index = i;
            g.name = prop.name;
            g.total_memory_bytes = total_b;
            g.free_memory_bytes  = free_b;
            g.available = true;
            out.push_back(g);
        }
    }
#else
    // Stub: probe nvidia-smi style presence without linking CUDA
    // Real build should define WINNER_HAS_CUDA and link cudart
#endif
    return out;
}

HybridPlan plan_hybrid(size_t model_bytes, size_t kv_bytes,
                       const std::vector<DeviceInfo>& devices,
                       int force_gpu_layers)
{
    HybridPlan plan;
    plan.n_gpu_layers = 0;
    plan.n_cpu_layers = 32; // default guess until IR gives real count
    plan.use_mmap_cpu = true;

    const DeviceInfo* gpu = nullptr;
    const DeviceInfo* cpu = nullptr;
    for (const auto& d : devices) {
        if (d.type == DeviceType::CPU) cpu = &d;
        if ((d.type == DeviceType::CUDA || d.type == DeviceType::HIP || d.type == DeviceType::METAL)
            && d.available && d.free_memory_bytes > 256ull * 1024 * 1024) {
            if (!gpu || d.free_memory_bytes > gpu->free_memory_bytes) gpu = &d;
        }
    }

    if (!gpu) {
        // Pure CPU
        plan.cpu_budget_bytes = cpu ? cpu->free_memory_bytes : model_bytes;
        plan.gpu_budget_bytes = 0;
        plan.n_gpu_layers = 0;
        return plan;
    }

    // Reserve ~20% VRAM for KV + activations + fragmentation
    size_t vram_usable = static_cast<size_t>(gpu->free_memory_bytes * 0.80);
    plan.gpu_budget_bytes = vram_usable;
    plan.cpu_budget_bytes = cpu ? cpu->free_memory_bytes : 0;

    if (force_gpu_layers >= 0) {
        plan.n_gpu_layers = force_gpu_layers;
        plan.n_cpu_layers = std::max(0, plan.n_cpu_layers - force_gpu_layers);
        return plan;
    }

    // Auto: put as many layers as fit in VRAM
    // Rough: assume model_bytes proportional to layers
    int total_layers = plan.n_gpu_layers + plan.n_cpu_layers;
    if (total_layers < 1) total_layers = 32;

    if (model_bytes + kv_bytes <= vram_usable) {
        // Full GPU
        plan.n_gpu_layers = total_layers;
        plan.n_cpu_layers = 0;
    } else {
        // Hybrid: fraction of model that fits
        double frac = double(vram_usable) / double(model_bytes + kv_bytes);
        if (frac > 1.0) frac = 1.0;
        if (frac < 0.05) frac = 0.0;
        plan.n_gpu_layers = static_cast<int>(total_layers * frac);
        plan.n_cpu_layers = total_layers - plan.n_gpu_layers;
    }
    return plan;
}

} // namespace backend
} // namespace winner


#include <thread>

namespace winner {
namespace backend {

bool pin_thread_to_core(int core_id) {
    if (core_id < 0) return false;
#if defined(_WIN32)
    if (core_id >= static_cast<int>(sizeof(DWORD_PTR) * 8)) return false;
    const DWORD_PTR mask = DWORD_PTR{1} << core_id;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)core_id;
    return false;
#endif
}

void pin_worker_pool(std::vector<std::thread>& workers, int start_core) {
    for (size_t i = 0; i < workers.size(); ++i) {
        int core = start_core + int(i);
        // affinity must be set from within the thread — use a lambda at spawn site
        (void)core;
    }
    (void)workers;
    (void)start_core;
}

int suggested_workers(const CpuFeatures& feat) {
    int n = feat.n_cores > 0 ? feat.n_cores : int(std::thread::hardware_concurrency());
    if (n <= 1) return 1;
    return n - 1; // leave one core for OS / IO
}

} // namespace backend
} // namespace winner
