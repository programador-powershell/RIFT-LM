cmake_minimum_required(VERSION 3.20)
project(vlb_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VLB_ENABLE_AVX2 "Enable AVX2 kernel specialization when compiler supports it" ON)

add_library(vlb_kernel STATIC
    src/vlb_kernel.cpp
)

target_include_directories(vlb_kernel PUBLIC include)

target_compile_options(vlb_kernel PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/O2;/W4>
)

if(VLB_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(vlb_kernel PRIVATE /arch:AVX2)
        target_compile_definitions(vlb_kernel PRIVATE VLB_COMPILED_AVX2=1)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(vlb_kernel PRIVATE -mavx2 -mfma)
        target_compile_definitions(vlb_kernel PRIVATE VLB_COMPILED_AVX2=1)
    endif()
endif()

add_executable(vlb-kernel-proof src/vlb_kernel_proof.cpp)
target_link_libraries(vlb-kernel-proof PRIVATE vlb_kernel)

add_executable(vlb-server src/vlb_server.cpp)
target_link_libraries(vlb-server PRIVATE vlb_kernel)

if(WIN32)
    target_link_libraries(vlb-server PRIVATE ws2_32)
endif()

enable_testing()
add_test(NAME vlb_kernel_selftest COMMAND vlb-kernel-proof --selftest)
