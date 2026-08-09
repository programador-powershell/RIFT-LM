#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. In the Codex setup field use:
#   bash winner_cpp/setup_test.sh
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${WINNER_BUILD_DIR:-$root/build}"

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$build" --parallel "${BUILD_JOBS:-$(nproc)}"
ctest --test-dir "$build" --output-on-failure
"$build/winner" --bench-kernels --dim 256 --layers 8 --tokens 16 \
  --output "$build/winner_profile_bench.json"

cat <<EOF

WINNER concluído: $build/winner
Servidor: $build/winner --serve --host 127.0.0.1 --port 8080

Observação: WINNER aceita Bundle .winr; llama.cpp aceita GGUF. Não compare o
microbenchmark sintético acima com tokens/s end-to-end do llama.cpp.
EOF

if [[ -n "${LLAMA_CPP_DIR:-}" ]]; then
  llama_build="${LLAMA_BUILD_DIR:-$LLAMA_CPP_DIR/build}"
  cmake -S "$LLAMA_CPP_DIR" -B "$llama_build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$llama_build" --parallel "${BUILD_JOBS:-$(nproc)}" --target llama-cli llama-bench
  [[ -z "${GGUF_MODEL:-}" ]] || "$llama_build/bin/llama-bench" -m "$GGUF_MODEL"
fi
