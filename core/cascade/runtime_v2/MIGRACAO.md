# cascade_runtime_v2 — guia de migração

Correções da auditoria do `runtime.zip`, todas implementadas e testadas
(`tests/test_all.py` — 5/5 PASS nesta máquina).

## O que mudou (arquivo antigo → novo)

| Antes | Depois | Correção |
|---|---|---|
| `block_runtime.py` → `CascadeLinearModule._w0_cache` (dequant FP32 residente, 32 bpw) | `q4k_linear.py` → `Q4KLinearModule` | F0 roda no kernel C fundido (`q4k_gemv_i8`): dequant dentro do produto, **nunca materializa W**. Residente = 4,5 bpw + F1. `stats()["w0_cache_bytes"]` agora é 0 por construção |
| `CASCADE_LOW_MEM=1` (dequant por chamada, ~0,3 GB/s) | idem | mesmo kernel; medido 42× mais rápido que o caminho antigo no teste de regressão |
| `confidence_gate.py` `decide_gate` (percentil do próprio batch → **F1 em 100% dos tokens de decode**) | `confidence_gate.py` v1 | threshold **calibrado offline** (`GateCalibrator.observe/freeze`) e congelado em `GateConfig.fixed_threshold` (gravar no bundle). Sem calibração + batch pequeno → **F0_ONLY fail-safe** com aviso na telemetria. Percentil por batch continua disponível só para prefill (`min_batch_for_batch_percentile`) |
| `cpp/avx2_lowrank.cpp` (loop morto; gather de V com `_mm256_set_ps` por stride) | `kernels/kernels.c` → `lowrank_gemv_f32` | V transposto para `(rank, in)` row-major (o módulo transpõe na carga), leitura contígua com FMA, OpenMP nas linhas de saída. Loop morto removido |
| `cpp/CMakeLists.txt` (só lowrank) | `kernels/CMakeLists.txt` + `build.sh` | um único `libcascade_kernels.so` com F0 (q4k fp32 + int8) e F1 |
| — | `q4k_pack.py` | pack/unpack **vetorizado** do layout de 144 B/super-bloco (o mesmo do `mmap_bundle.hpp`): pronto para o conversor gravar direto no bundle |

## Como usar

```python
from cascade_runtime_v2 import Q4KLinearModule, GateCalibrator, patch_block_linears

# 1. converter uma Linear (ou o bloco inteiro)
mod = Q4KLinearModule.from_linear(linear, rank=16)      # F1 opcional (SVD do resíduo)
replaced = patch_block_linears(block, rank=0)           # troca in-place

# 2. calibrar o gate UMA vez (ativações de calibração) e gravar no bundle
thr = mod.calibrate_gate(xs_calibracao)                 # congela em gate_cfg.fixed_threshold

# 3. decode normal — batch-1 usa o kernel C; sem threshold o F1 fica off (fail-safe)
y = mod(x)
print(mod.stats())   # resident_bytes reais, w0_cache_bytes == 0, taxa do gate
```

Compilar kernels: `kernels/build.sh` (gcc, `-mavx2 -mfma -mf16c`, AVX2 portável
— qualquer x86-64 desde ~2014). Sem a lib o módulo **recusa carregar** em vez
de cair num fallback 60× mais lento silenciosamente.

## Números desta máquina (4 threads)

- Kernel F0 isolado: 16,55 GB/s (llama.cpp Q4_K: 18,0 — 92%)
- Regressão (stack sintético 2 GB fp32 / 0,28 GB q4k):
  v2 **38,4 tok/s @ 0,28 GB** · low_mem antigo 0,92 tok/s · fp32-cache antigo
  22,3 tok/s @ 2,0 GB (7,1× a RAM)
- Gate batch-1: sem calibração 0% F1 (antes: 100%); calibrado p70 → 29%

## Pendências conscientes (não bloqueiam)

- Prefill usa loop de linhas sobre o GEMV; variante GEMM em lote é o próximo
  ganho quando prefill importar.
- Os 8% restantes até o llama.cpp: unpack vetorizado das escalas u6/i6,
  unroll de 2 linhas, variante AVX512-VNNI para CPUs novas.
- `mmap_bundle.hpp` de vocês permanece válido — o layout de 144 B é
  mmap-friendly; falta só o conversor gravar os planos com `pack_q4k`.
- Validação end-to-end (perplexity) continua obrigatória antes de publicar
  qualquer bundle — gate local ≠ qualidade fim-a-fim.
