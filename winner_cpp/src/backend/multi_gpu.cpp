#include "backend/multi_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace winner {
namespace backend {

MultiGpuPlan plan_multi_gpu(
    int n_layers,
    size_t bytes_per_layer,
    const std::vector<DeviceInfo>& devices,
    const std::vector<float>& split_weights,
    int n_gpu_layers) {

    MultiGpuPlan plan;
    if (n_layers < 1) n_layers = 1;

    std::vector<const DeviceInfo*> gpus;
    for (auto& d : devices) {
        if (d.type != DeviceType::CPU && d.available &&
            d.free_memory_bytes > 256ull * 1024 * 1024)
            gpus.push_back(&d);
    }

    if (gpus.empty() || n_gpu_layers == 0) {
        plan.mode = SplitMode::NONE;
        plan.n_cpu_layers = n_layers;
        plan.n_gpu_layers = 0;
        plan.summary = "CPU only (no GPU or --ngl 0)";
        return plan;
    }

    plan.mode = SplitMode::LAYER_SPLIT;

    // How many layers fit on GPUs
    int gpu_layers = n_gpu_layers;
    if (gpu_layers < 0) {
        gpu_layers = 0;
        for (auto* g : gpus) {
            size_t usable = size_t(g->free_memory_bytes * 0.80);
            gpu_layers += bytes_per_layer > 0 ? int(usable / bytes_per_layer) : 0;
        }
        gpu_layers = std::min(gpu_layers, n_layers);
    }
    gpu_layers = std::min(gpu_layers, n_layers);
    plan.n_gpu_layers = gpu_layers;
    plan.n_cpu_layers = n_layers - gpu_layers;

    // Distribute gpu_layers across devices
    std::vector<float> fracs = split_weights;
    if (fracs.size() != gpus.size()) {
        fracs.assign(gpus.size(), 0.f);
        double total_free = 0;
        for (auto* g : gpus) total_free += double(g->free_memory_bytes);
        for (size_t i = 0; i < gpus.size(); ++i)
            fracs[i] = total_free > 0 ? float(gpus[i]->free_memory_bytes / total_free) : 1.f / gpus.size();
    }
    // normalize
    float sum = 0.f;
    for (float f : fracs) sum += f;
    if (sum <= 0.f) {
        for (auto& f : fracs) f = 1.f / fracs.size();
        sum = 1.f;
    }

    int cursor = 0;
    std::ostringstream oss;
    oss << "LAYER_SPLIT " << gpu_layers << " GPU + " << plan.n_cpu_layers << " CPU | ";
    for (size_t i = 0; i < gpus.size(); ++i) {
        int n = (i + 1 == gpus.size())
            ? (gpu_layers - cursor)
            : int(std::round(fracs[i] / sum * gpu_layers));
        n = std::max(0, std::min(n, gpu_layers - cursor));
        GpuSlice s;
        s.device_index = gpus[i]->index;
        s.type = gpus[i]->type;
        s.layer_begin = cursor;
        s.layer_end = cursor + n;
        s.split_fraction = fracs[i] / sum;
        s.budget_bytes = size_t(gpus[i]->free_memory_bytes * 0.80);
        plan.slices.push_back(s);
        oss << "GPU" << s.device_index << " L[" << s.layer_begin << "," << s.layer_end << ") ";
        cursor += n;
    }
    plan.summary = oss.str();
    return plan;
}

int device_for_layer(const MultiGpuPlan& plan, int layer) {
    for (auto& s : plan.slices) {
        if (layer >= s.layer_begin && layer < s.layer_end)
            return s.device_index;
    }
    return -1; // CPU
}

} // namespace backend
} // namespace winner
