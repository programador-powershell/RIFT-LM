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

## Conversor otimizado (`convert.py`) — adicionado nesta versão

```bash
python -m cascade_runtime_v2.convert --input ./modelo-hf-bf16 --output ./bundle --model-id org/modelo
```

Otimizações sobre o conversor v1 (todas medidas na Muse-Glimmer BF16 real):
1. **Grava no formato do executor** (138/144 B por super-bloco) — zero repack na carga.
2. **Escada q4k/g64+clip → q4k/g32+clip**, sem fallback raw para 2D: na amostra,
   34/36 tensores em 4,3125 bpw, 2 em 4,5, nenhum raw (v1: 3 raws de 16 bpw).
3. **Embeddings/lm_head em q4k/g32 com cosseno MEDIDO** (0,9969/0,9969 na Muse)
   em vez de raw 16 bpw (v1) — economiza 4,0 GB no 30B. Degrau q6k para
   embeddings = pendência de kernel 6-bit.
4. Exclusão padrão cobre também `token_embd`/`output.weight` (gap do v1).
5. Encode em chunks de linhas: RSS pico 3,14 GiB na amostra (ajustável via
   `CHUNK_ROWS`).
6. Resultado: **2,90 GB → 0,78 GB (73,0%) em 99,7 s** — v1: 69,35% em 198,7 s
   (2× mais rápido, 9,6% menor). Residência por classe de máquina no manifest.

## Correções pós-review (§28/§29.5)

- **Residência**: corrigida a subtração dupla (24 GB total → orçamento 16 GiB,
  nunca 8). `MACHINE_TOTAL_GIB = (16, 24, 32, 48)` lista o TOTAL da máquina;
  o `- 8 GiB` acontece só em `residency_report`, que agora grava
  `folga_gib` (negativa quando não cabe) e `regra_orcamento` no manifest.
  Anti-regressão: `tests/test_residency.py` (22 asserções, incl. limiar exato
  14,50/14,51 GiB).
- **Docstring** do conversor: `LADDER` é a fonte da verdade (2 degraus de
  4 bits hoje; q5k/q6k = pendência de kernel).
- **Gate por construção**: o resumo do manifest ganha
  `all_tensors_passed_gate`, `below_gate_tensor_count`, `below_gate_tensors`,
  e o console imprime ATENÇÃO quando qualquer tensor grava via
  RESCUE_LAST_RUNG — o bundle deixa de ser aprovado por construção e isso
  agora é visível no topo, não escondido em flag por tensor.
- **Faixa da projeção 24 GB** (no `pct_final.json`): central 15,08 GB
  (folga +0,46 GiB), worst realista 15,19 (+0,36), worst estrutural 15,67
  (−0,09). Ponto de ruptura: +3,24% sobre o central (~4,47 bpw médio ⇒ ~96%
  dos tensores em g32; medido: 5,6%). Rótulo PROJETADO até a conversão
  integral (~32 min em Colab), que é o que fecha o veredito binário.

## v2.2 — velocidade (tok/s)

Três ataques aos ladrões de banda medidos (fork/join OMP por chamada,
falta de ILP, maddubs onde há VNNI):
1. **`cascade_gemv_batch`**: N GEMVs independentes numa única região OMP
   (schedule dinâmico por blocos de linhas, `nowait` entre itens). No motor,
   fundir por camada os grupos de mesmo input: {q,k,v,attn_gate}, {gate,up}.
2. **Prefetch** (linha seguinte + super seguinte) e miolo refatorado por
   blocos de linhas (`q4k_rows_i8_32`/`q8r_rows_i8`).
3. **Build AVX512-VNNI** (`vpdpbusd`): `build.sh` gera
   `libcascade_kernels_vnni.so`; o loader escolhe pelo `/proc/cpuinfo`.
   AVX2 puro continua o baseline portável.

Medido na Muse-30B real (bundle 16,30 GB, caminho de pesos por token):
v2.1 **0,911 tok/s / 14,16 GB/s** → v2.2 **1,078 tok/s / 16,74 GB/s**
(+18,3%; fase B a 18,05 GB/s = paridade com o kernel do llama.cpp).
Contra o Q4_K_XL derivado (1,134 tok/s): 95%, com 627/627 no gate.
