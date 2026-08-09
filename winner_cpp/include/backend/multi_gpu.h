#pragma once
/**
 * Multi-GPU layer split (closes gap vs llama.cpp --tensor-split / --n-gpu-layers)
 *
 * Strategies:
 *   LAYER_SPLIT  — consecutive layers on each device (llama.cpp style)
 *   TENSOR_SPLIT — shard rows of large weights across devices (future)
 *
 * WINNER adds progressive twist: only F0 base is split; residuals stay on
 * the device that owns the layer (hot path).
 */
#include "backend/device.h"
#include <cstdint>
#include <string>
#include <vector>

namespace winner {
namespace backend {

enum class SplitMode : uint8_t {
    NONE = 0,
    LAYER_SPLIT,   // layers [0,a) GPU0, [a,b) GPU1, rest CPU
    TENSOR_SPLIT,  // row-shard (stub)
};

struct GpuSlice {
    int device_index = 0;
    DeviceType type = DeviceType::CUDA;
    int layer_begin = 0;
    int layer_end = 0;     // exclusive
    float split_fraction = 0.f;
    size_t budget_bytes = 0;
};

struct MultiGpuPlan {
    SplitMode mode = SplitMode::NONE;
    std::vector<GpuSlice> slices;
    int n_cpu_layers = 0;
    int n_gpu_layers = 0;
    std::string summary;
};

/**
 * Plan layer split across available GPUs + CPU.
 * @param split_weights  e.g. {0.5, 0.5} for 2 GPUs equal, or empty = auto by free VRAM
 * @param n_gpu_layers   -1 = fill VRAM, 0 = CPU only, >0 = force
 */
MultiGpuPlan plan_multi_gpu(
    int n_layers,
    size_t bytes_per_layer,
    const std::vector<DeviceInfo>& devices,
    const std::vector<float>& split_weights = {},
    int n_gpu_layers = -1);

/** Which device owns layer i (−1 = CPU) */
int device_for_layer(const MultiGpuPlan& plan, int layer);

} // namespace backend
} // namespace winner
