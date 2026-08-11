#include "backend/vcpu.h"
#include "backend/device.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#endif

namespace winner {
namespace backend {

static std::vector<int> read_cpu_list(const std::string& path) {
    std::vector<int> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string s;
    if (!(f >> s)) return out;
    // format: 0-3,8-11 or 0,1,2
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto dash = part.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::atoi(part.c_str()));
        } else {
            int a = std::atoi(part.substr(0, dash).c_str());
            int b = std::atoi(part.substr(dash + 1).c_str());
            for (int i = a; i <= b; ++i) out.push_back(i);
        }
    }
    return out;
}

void VCpuPool::detect_topology(const CpuFeatures& feat) {
    sockets_.clear();
#if defined(__linux__)
    // Try sysfs NUMA nodes
    for (int n = 0; n < 64; ++n) {
        std::string base = "/sys/devices/system/node/node" + std::to_string(n);
        std::ifstream test(base + "/cpulist");
        if (!test) break;
        SocketInfo sock;
        sock.socket_id = n;
        sock.core_ids = read_cpu_list(base + "/cpulist");
        sock.reserved_for_os = cfg_.reserved_cores_per_socket;
        if (!sock.core_ids.empty()) sockets_.push_back(std::move(sock));
    }
#endif
    if (sockets_.empty()) {
        // Fallback: single socket with all reported cores
        SocketInfo sock;
        sock.socket_id = 0;
        int n = feat.n_cores > 0 ? feat.n_cores : int(std::thread::hardware_concurrency());
        if (n < 1) n = 1;
        for (int i = 0; i < n; ++i) sock.core_ids.push_back(i);
        sock.reserved_for_os = cfg_.reserved_cores_per_socket;
        sockets_.push_back(std::move(sock));
    }
}

void VCpuPool::init(const CpuFeatures& feat, const VCpuConfig& cfg) {
    cfg_ = cfg;
    detect_topology(feat);
    vcpus_.clear();

    int vcpu_id = 0;
    for (auto& sock : sockets_) {
        int usable = int(sock.core_ids.size()) - sock.reserved_for_os;
        if (usable < 1) usable = 1; // at least one if only 1 core
        if (cfg_.max_workers_per_socket > 0)
            usable = std::min(usable, cfg_.max_workers_per_socket);

        for (int i = 0; i < usable && i < (int)sock.core_ids.size(); ++i) {
            VCpu v;
            v.vcpu_id = vcpu_id++;
            v.socket_id = sock.socket_id;
            v.core_id = sock.core_ids[i];
            v.busy = false;
            vcpus_.push_back(v);
        }
    }

    if (cfg_.max_workers_total > 0 && (int)vcpus_.size() > cfg_.max_workers_total) {
        vcpus_.resize(cfg_.max_workers_total);
    }

    // Optional: interleave sockets for spread
    if (cfg_.spread_sockets && sockets_.size() > 1) {
        std::vector<VCpu> interleaved;
        size_t max_n = 0;
        for (auto& s : sockets_) max_n = std::max(max_n, s.core_ids.size());
        for (size_t i = 0; i < max_n; ++i) {
            for (auto& v : vcpus_) {
                // keep order by taking i-th vcpu of each socket
            }
        }
        // simpler: sort by (index within socket, socket) already sequential per socket
        // re-stripe: gather by position
        std::vector<std::vector<VCpu>> by_sock(sockets_.size());
        for (auto& v : vcpus_) {
            const auto socket_it = std::find_if(sockets_.begin(), sockets_.end(),
                [&](const SocketInfo& socket) { return socket.socket_id == v.socket_id; });
            if (socket_it != sockets_.end()) {
                by_sock[static_cast<size_t>(std::distance(sockets_.begin(), socket_it))].push_back(v);
            }
        }
        interleaved.clear();
        size_t idx = 0;
        bool any = true;
        while (any) {
            any = false;
            for (auto& bucket : by_sock) {
                if (idx < bucket.size()) {
                    interleaved.push_back(bucket[idx]);
                    any = true;
                }
            }
            idx++;
        }
        for (size_t i = 0; i < interleaved.size(); ++i)
            interleaved[i].vcpu_id = int(i);
        vcpus_.swap(interleaved);
    }
}

int VCpuPool::acquire() {
    std::lock_guard<std::mutex> lock(busy_mu_);
    if (vcpus_.empty()) return -1;
    int n = int(vcpus_.size());
    int start = rr_.fetch_add(1) % n;
    for (int k = 0; k < n; ++k) {
        int i = (start + k) % n;
        if (!vcpus_[i].busy) {
            vcpus_[i].busy = true;
            return vcpus_[i].vcpu_id;
        }
    }
    return -1; // all busy — caller should not oversubscribe
}

void VCpuPool::release(int vcpu_id) {
    std::lock_guard<std::mutex> lock(busy_mu_);
    if (vcpu_id < 0 || vcpu_id >= (int)vcpus_.size()) return;
    vcpus_[vcpu_id].busy = false;
}

void VCpuPool::start(std::function<void(int vcpu_id)> worker_fn) {
    stop();
    running_ = true;
    workers_.clear();
    for (auto& v : vcpus_) {
        int vid = v.vcpu_id;
        int core = v.core_id;
        bool do_pin = cfg_.pin_threads;
        workers_.emplace_back([this, worker_fn, vid, core, do_pin]() {
            if (do_pin && core >= 0) {
                pin_thread_to_core(core);
            }
            worker_fn(vid);
        });
    }
}

void VCpuPool::stop() {
    running_ = false;
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void VCpuPool::parallel_for(int n, std::function<void(int begin, int end, int vcpu_id)> fn) {
    if (n <= 0) return;
    int w = std::max(1, int(vcpus_.size()));
    // Cap chunk workers to available vCPUs — never oversubscribe
    w = std::min(w, n);
    std::vector<std::thread> tmp;
    tmp.reserve(w);
    for (int i = 0; i < w; ++i) {
        int begin = (i * n) / w;
        int end = ((i + 1) * n) / w;
        if (begin >= end) continue;
        int vid = i < (int)vcpus_.size() ? vcpus_[i].vcpu_id : 0;
        int core = i < (int)vcpus_.size() ? vcpus_[i].core_id : -1;
        bool do_pin = cfg_.pin_threads;
        tmp.emplace_back([fn, begin, end, vid, core, do_pin]() {
            if (do_pin && core >= 0) pin_thread_to_core(core);
            fn(begin, end, vid);
        });
    }
    for (auto& t : tmp) t.join();
}

std::string VCpuPool::summary() const {
    std::ostringstream os;
    os << "VCpuPool: " << vcpus_.size() << " vCPUs across " << sockets_.size() << " socket(s)\n";
    for (auto& s : sockets_) {
        int used = 0;
        for (auto& v : vcpus_) if (v.socket_id == s.socket_id) used++;
        os << "  socket " << s.socket_id << ": " << s.core_ids.size()
           << " cores, using " << used
           << " (reserved " << s.reserved_for_os << " for OS/IO)\n";
    }
    for (auto& v : vcpus_) {
        os << "  vCPU" << v.vcpu_id << " → core " << v.core_id
           << " (socket " << v.socket_id << ")\n";
    }
    return os.str();
}

} // namespace backend
} // namespace winner
