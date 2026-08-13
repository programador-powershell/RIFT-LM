#!/bin/bash
set -e
cd "$(dirname "$0")"
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC kernels.c \
    -o libcascade_kernels.so -lm
echo "ok: libcascade_kernels.so (AVX2)"
if gcc -mavx512vnni -mavx512vl -E -x c /dev/null >/dev/null 2>&1; then
  gcc -O3 -mavx2 -mfma -mf16c -mavx512vnni -mavx512vl -DUSE_VNNI -fopenmp \
      -shared -fPIC kernels.c -o libcascade_kernels_vnni.so -lm
  echo "ok: libcascade_kernels_vnni.so (AVX512-VNNI)"
fi
