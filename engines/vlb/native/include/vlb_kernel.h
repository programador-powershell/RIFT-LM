#pragma once

#include <cstddef>
#include <cstdint>

namespace vlb {

struct Q8G64Matrix {
    const std::int8_t* qweight = nullptr;
    const std::uint16_t* fp16_scales = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t group_size = 64;
};

float fp16_to_fp32(std::uint16_t value) noexcept;

void q8_g64_gemv_scalar(
    const Q8G64Matrix& matrix,
    const float* x,
    float* y
);

void q8_g64_gemv(
    const Q8G64Matrix& matrix,
    const float* x,
    float* y
);

const char* kernel_backend() noexcept;

} // namespace vlb
