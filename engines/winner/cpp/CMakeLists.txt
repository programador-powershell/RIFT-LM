cmake_minimum_required(VERSION 3.16)
project(winner_cpp VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(CheckCXXCompilerFlag)
find_package(Threads REQUIRED)

option(WINNER_NATIVE "Tune generated code for the build machine" OFF)
option(WINNER_FAST_MATH "Enable non-IEEE fast-math optimizations" OFF)
option(WINNER_AVX512 "Compile AVX-512 paths (requires a compatible runtime CPU)" OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /EHsc)
  if(WINNER_NATIVE)
    add_compile_options(/O2 /arch:AVX2)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
  endif()
  if(WINNER_NATIVE)
    check_cxx_compiler_flag("-march=native" WINNER_HAS_MARCH_NATIVE)
    if(WINNER_HAS_MARCH_NATIVE)
      add_compile_options(-march=native -mtune=native)
    endif()
  endif()
  if(WINNER_FAST_MATH)
    add_compile_options(-ffast-math -fno-math-errno)
  endif()
  if(WINNER_AVX512)
    add_compile_options(-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni)
  endif()
endif()

set(WINNER_CORE_SRC
  src/bundle.cpp
  src/runtime.cpp
  src/backend/cpu_detect.cpp
  src/backend/device.cpp
  src/backend/kernels.cpp
  src/backend/vcpu.cpp
  src/backend/multi_gpu.cpp
  src/kernels/residual_ls.cpp
  src/moe/expert_page.cpp
  src/quant/kquant.cpp
  src/attention/page_table.cpp
  src/attention/fused_kernels.cpp
  src/sched/batching.cpp
  src/speculative/speculative.cpp
)

if(UNIX)
  list(APPEND WINNER_CORE_SRC src/prefetch/prefetch_uring.cpp src/server/http_server.cpp)
endif()

add_library(winner_core STATIC ${WINNER_CORE_SRC})
target_include_directories(winner_core PUBLIC include)
target_link_libraries(winner_core PUBLIC Threads::Threads)

add_executable(winner src/main.cpp)
target_link_libraries(winner PRIVATE winner_core)
if(UNIX)
  target_compile_definitions(winner PRIVATE WINNER_HTTP_SERVER=1)
endif()

add_executable(winner_residual_test bench/test_residual_module.cpp)
target_link_libraries(winner_residual_test PRIVATE winner_core)

if(BUILD_TESTING)
  add_test(NAME winner_self_test COMMAND winner --self-test)
  add_test(NAME winner_residual_test COMMAND winner_residual_test)
  if(UNIX)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME winner_http_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bench/test_http_server.py $<TARGET_FILE:winner>)
      set_tests_properties(winner_http_test PROPERTIES TIMEOUT 15)
    endif()
  endif()
endif()

install(TARGETS winner RUNTIME DESTINATION bin)
message(STATUS "WINNER.cpp ${PROJECT_VERSION} (native=${WINNER_NATIVE}, fast-math=${WINNER_FAST_MATH})")
