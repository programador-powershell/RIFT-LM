#include "vlb_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Header {
    char magic[8]; // VLBK001\0
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t group_size;
};

template <class T>
void read_exact(std::ifstream& in, T* ptr, std::size_t count) {
    in.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) throw std::runtime_error("unexpected EOF in VLB kernel proof vector");
}

int run_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open proof vector: " + path);

    Header h{};
    read_exact(in, &h, 1);
    if (std::string(h.magic, h.magic + 7) != "VLBK001") {
        throw std::runtime_error("invalid VLB kernel proof magic");
    }
    if (h.rows == 0 || h.cols == 0 || h.group_size == 0) {
        throw std::runtime_error("invalid proof shape");
    }

    const std::size_t rows = static_cast<std::size_t>(h.rows);
    const std::size_t cols = static_cast<std::size_t>(h.cols);
    const std::size_t total = rows * cols;
    const std::size_t groups = (total + static_cast<std::size_t>(h.group_size) - 1) / static_cast<std::size_t>(h.group_size);

    std::vector<std::int8_t> q(total);
    std::vector<std::uint16_t> scales(groups);
    std::vector<float> x(cols);
    std::vector<float> reference(rows);
    std::vector<float> candidate(rows);

    read_exact(in, q.data(), q.size());
    read_exact(in, scales.data(), scales.size());
    read_exact(in, x.data(), x.size());
    read_exact(in, reference.data(), reference.size());

    vlb::Q8G64Matrix matrix{q.data(), scales.data(), rows, cols, static_cast<std::size_t>(h.group_size)};
    vlb::q8_g64_gemv(matrix, x.data(), candidate.data());

    double sq = 0.0;
    double sq_ref = 0.0;
    double dot = 0.0;
    double norm_c = 0.0;
    double max_abs = 0.0;

    for (std::size_t i = 0; i < rows; ++i) {
        const double a = static_cast<double>(reference[i]);
        const double b = static_cast<double>(candidate[i]);
        const double d = b - a;
        sq += d * d;
        sq_ref += a * a;
        dot += a * b;
        norm_c += b * b;
        max_abs = std::max(max_abs, std::abs(d));
    }

    const double rmse = std::sqrt(sq / static_cast<double>(rows));
    const double rms_ref = std::sqrt(sq_ref / static_cast<double>(rows));
    const double nrmse = rmse / std::max(rms_ref, 1e-30);
    const double cosine = dot / std::max(std::sqrt(sq_ref * norm_c), 1e-30);

    std::cout << std::setprecision(12)
              << "{\"backend\":\"" << vlb::kernel_backend() << "\"," 
              << "\"rows\":" << rows << ","
              << "\"cols\":" << cols << ","
              << "\"group_size\":" << h.group_size << ","
              << "\"max_abs_error\":" << max_abs << ","
              << "\"rmse\":" << rmse << ","
              << "\"nrmse\":" << nrmse << ","
              << "\"cosine\":" << cosine << "}\n";
    return 0;
}

int selftest() {
    constexpr std::size_t rows = 17;
    constexpr std::size_t cols = 128;
    constexpr std::size_t group = 64;
    const std::size_t total = rows * cols;
    const std::size_t groups = (total + group - 1) / group;

    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> qdist(-127, 127);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<std::int8_t> q(total);
    std::vector<std::uint16_t> scales(groups, 0x2c00u); // exactly representable FP16 value
    std::vector<float> x(cols);
    std::vector<float> scalar(rows);
    std::vector<float> dispatched(rows);

    for (auto& v : q) v = static_cast<std::int8_t>(qdist(rng));
    for (auto& v : x) v = xdist(rng);

    vlb::Q8G64Matrix matrix{q.data(), scales.data(), rows, cols, group};
    vlb::q8_g64_gemv_scalar(matrix, x.data(), scalar.data());
    vlb::q8_g64_gemv(matrix, x.data(), dispatched.data());

    double max_abs = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(scalar[i]) - dispatched[i]));
    }

    std::cout << "VLB kernel selftest backend=" << vlb::kernel_backend()
              << " max_abs=" << std::setprecision(12) << max_abs << "\n";

    // This is only an implementation regression test, not a model quality gate.
    return max_abs <= 1e-3 ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--selftest") return selftest();
        if (argc == 3 && std::string(argv[1]) == "--vector") return run_file(argv[2]);
        std::cerr << "usage: vlb-kernel-proof --selftest | --vector FILE\n";
        return 64;
    } catch (const std::exception& exc) {
        std::cerr << "vlb-kernel-proof error: " << exc.what() << "\n";
        return 1;
    }
}
