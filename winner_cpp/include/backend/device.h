#pragma once
#include "backend/cpu_detect.h"
#include <cstdint>
#include <string>
#include <vector>
#include <thread>

namespace winner {
namespace backend {

enum class DeviceType : uint8_t { CPU = 0, CUDA, HIP, METAL };

struct DeviceInfo {
    DeviceType type = DeviceType::CPU;
    std::string name;
    size_t total_memory_bytes = 0;
    size_t free_memory_bytes = 0;
    int index = 0;
    bool available = true;
};

struct HybridPlan {
    int n_gpu_layers = 0;
    int n_cpu_layers = 0;
    size_t cpu_budget_bytes = 0;
    size_t gpu_budget_bytes = 0;
    bool use_mmap_cpu = true;
};

std::vector<DeviceInfo> enumerate_devices();
HybridPlan plan_hybrid(size_t model_bytes, size_t kv_bytes,
                       const std::vector<DeviceInfo>& devices, int n_gpu_layers_req);

bool pin_thread_to_core(int core_id);
void pin_worker_pool(std::vector<std::thread>& workers, int start_core = 0);
int suggested_workers(const CpuFeatures& feat);

} // namespace backend
} // namespace winner
