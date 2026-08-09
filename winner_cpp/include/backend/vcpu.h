#pragma once
/**
 * Virtual CPU pool — evita sobrecarregar um socket NUMA
 *
 * Em vez de spawnar N = hardware_concurrency threads no mesmo socket
 * (o que satura L3, memory controller e interconnect), o VCpuPool:
 *   1. Detecta sockets/NUMA nodes e cores por socket
 *   2. Cria um "vCPU" lógico por core físico reservável
 *   3. Limita workers ativos por socket (default: cores_per_socket - 1)
 *   4. Round-robin entre sockets quando há multi-socket
 *   5. Pin de cada worker ao core mapeado (sem migração)
 */
#include "backend/cpu_detect.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

namespace winner {
namespace backend {

struct SocketInfo {
    int socket_id = 0;
    std::vector<int> core_ids;   // physical core OS indices on this socket
    int reserved_for_os = 1;     // leave at least 1 core free per socket
};

struct VCpu {
    int vcpu_id = 0;
    int socket_id = 0;
    int core_id = -1;            // OS CPU index to pin to
    bool busy = false;
};

struct VCpuConfig {
    int max_workers_total = 0;       // 0 = auto
    int max_workers_per_socket = 0;  // 0 = cores_on_socket - reserved
    int reserved_cores_per_socket = 1;
    bool pin_threads = true;
    bool spread_sockets = true;      // distribute across sockets before filling one
};

class VCpuPool {
public:
    void init(const CpuFeatures& feat, const VCpuConfig& cfg = {});
    int n_sockets() const { return int(sockets_.size()); }
    int n_vcpus() const { return int(vcpus_.size()); }
    int n_workers() const { return int(workers_.size()); }
    const std::vector<SocketInfo>& sockets() const { return sockets_; }

    /** Spawn worker threads, each pinned to its vCPU core */
    void start(std::function<void(int vcpu_id)> worker_fn);
    void stop();

    /** Acquire a free vCPU (round-robin sockets). Returns -1 if none. */
    int acquire();
    void release(int vcpu_id);

    /** Parallel for over [0, n) using the pool — never oversubscribes a socket */
    void parallel_for(int n, std::function<void(int begin, int end, int vcpu_id)> fn);

    std::string summary() const;

private:
    VCpuConfig cfg_;
    std::vector<SocketInfo> sockets_;
    std::vector<VCpu> vcpus_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::atomic<int> rr_{0};
    std::mutex busy_mu_;

    void detect_topology(const CpuFeatures& feat);
};

} // namespace backend
} // namespace winner
