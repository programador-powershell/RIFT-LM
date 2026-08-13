#!/bin/bash
# Compila libcascade_kernels.so (AVX2 portavel; funciona em qualquer x86-64
# com AVX2+FMA+F16C — PCs convencionais de 2014 em diante).
set -e
cd "$(dirname "$0")"
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC kernels.c \
    -o libcascade_kernels.so -lm
echo "ok: $(pwd)/libcascade_kernels.so"
