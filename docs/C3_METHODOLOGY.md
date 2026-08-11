# C3_METHODOLOGY_V1 — Metodologia de 16 passos da série C3

Protocol ID: `C3_METHODOLOGY_V1` (campo `benchmark_protocol` do schema v2).

Este documento descreve a metodologia da terceira bateria (série C3) executada por
`batteries/c3_methodology_auto_batteries.py --technology rift|aether|cascade|spectra`
(no Colab o script é baixado para o diretório de execução com o mesmo nome local). Os contratos
formais (ids, campos obrigatórios, política do WINNER) vivem em
[`docs/C3_CONTRACTS_V1.md`](C3_CONTRACTS_V1.md); as regras de honestidade de medição herdam
[`docs/REAL_BENCHMARK_PROTOCOL_V3.md`](REAL_BENCHMARK_PROTOCOL_V3.md). Em caso de conflito,
os contratos prevalecem.

## Visão geral

A série C3 aplica a MESMA metodologia de 16 passos às quatro tecnologias elegíveis
(`RIFT`, `AETHER`, `CASCADE`, `SPECTRA`), variando apenas o codec F0
(flag `--codec`, default por tecnologia):

| Tecnologia | Codec F0 default | Observação |
| --- | --- | --- |
| CASCADE | `int4` | groupwise signed [-8,7], escala FP16, grupo 32 |
| RIFT | `int2` | mesmo esquema groupwise, 2 bits/peso, 4 níveis simétricos |
| AETHER | `ternary` | {-1,0,+1} com escala por linha + limiar de esparsidade |
| SPECTRA | `ternary` | idem AETHER + métrica de contrato de drift |

F1 (residual low-rank) e o Confidence Gate v0 são comuns a todas as tecnologias.
O script é auto-contido no estilo do repositório e reutiliza o pacote python
compartilhado `core/cascade/` (importável como `cascade`; no Colab é baixado para
`cascade/` local — INT4 + low-rank, IR, bundle, gate e runtimes).

## Os 16 passos mapeados para battery_ids

`<TECH>` ∈ {`RIFT`, `AETHER`, `CASCADE`, `SPECTRA`}. Espelha a tabela do contrato §2:

| Passos | battery_id | Conteúdo |
| --- | --- | --- |
| 1, 5 | `C3_<TECH>_BUNDLE_M0_FREEZE` | Bundle M0 congelado (CSCD v0x0003, header 128B) + golden tests válidos/inválidos |
| 2 | `C3_<TECH>_STAGE_PAGE_M0_FREEZE` | Stage Table/Page ABI congelada (entrada de 24 bytes) + golden tests |
| 4 | `C3_<TECH>_IR_WRITER` | CASCADE-IR v3 write→validate→reload roundtrip |
| 6 | `C3_<TECH>_CPP_BUNDLE_READER` | Compila e executa o leitor C++ mmap sobre o bundle real (POSIX; `SKIPPED` no Windows) |
| 3, 7–12 | `C3_<TECH>_LINEAR_ORIGINAL`, `C3_<TECH>_LINEAR_F0_ONLY`, `C3_<TECH>_LINEAR_F0_PLUS_F1_ALWAYS`, `C3_<TECH>_LINEAR_F0_GATE_F1` | Linear real, 4 caminhos, cada um como REGISTRO PRÓPRIO |
| 11–12 | `C3_<TECH>_BLOCK_ORIGINAL`, `C3_<TECH>_BLOCK_F0_ONLY`, `C3_<TECH>_BLOCK_F0_PLUS_F1_ALWAYS`, `C3_<TECH>_BLOCK_F0_GATE_F1` | Bloco Transformer real, 4 caminhos |
| 14 | `C3_<TECH>_C1_DECISION` | Aprova/reprova C1 (critérios abaixo) |
| 15 | `C3_<TECH>_BLOCKS4_GATED` | Expansão para 4 blocos reais patchados (gated vs original) |
| 16 | `C3_<TECH>_FULLMODEL_E2E_TOKS` | Modelo pequeno completo: TODOS os blocos patchados; tok/s reais via `model.generate` |

O passo 13 (alimentar o dashboard) não gera battery_id próprio: é o **publish incremental**
de cada registro via `POST /api/results` conforme cada passo termina.

### Papéis de comparação

`comparison_role = "primary"` apenas em:
`C3_<TECH>_LINEAR_F0_GATE_F1`, `C3_<TECH>_BLOCK_F0_GATE_F1`,
`C3_<TECH>_BLOCKS4_GATED`, `C3_<TECH>_FULLMODEL_E2E_TOKS`.
Todos os demais são diagnósticos (`comparison_role = null`). Os quatro caminhos de
Linear/Bloco (original, F0 only, F0+F1 always, F0 gate F1) são publicados como registros
separados para permitir auditoria da contribuição de cada estágio.

## Critérios de aprovação/reprovação C1 (`C3_<TECH>_C1_DECISION`)

O passo 14 emite `status = PASS` **se e somente se** TODAS as condições valem, avaliadas
sobre o caminho gated (`F0_GATE_F1`) tanto do Linear quanto do Bloco:

1. `cosine ≥ 0.995` (Linear gated E Bloco gated);
2. `NRMSE ≤ 0.05` (Linear gated E Bloco gated);
3. golden tests dos passos de freeze (bundle e Stage Table/Page) todos `PASS`;
4. `F1_skip_rate > 0` — o Confidence Gate precisa efetivamente pular o residual em parte
   das ativações; um gate que nunca pula não demonstrou funcionar.

Qualquer condição violada → `FAIL`. A decisão C1 reprova a expansão: os passos 15 e 16 só
executam para tecnologias com `C3_<TECH>_C1_DECISION = PASS`.

## Expansão full-model (task-8)

A expansão acontece em dois estágios após a decisão C1:

- **Passo 15 (`C3_<TECH>_BLOCKS4_GATED`)**: 4 blocos Transformer reais do modelo são
  patchados com o caminho gated e comparados contra o original, verificando que a
  qualidade não colapsa quando os erros dos blocos se compõem.
- **Passo 16 (`C3_<TECH>_FULLMODEL_E2E_TOKS`)**: um modelo pequeno completo tem TODOS os
  blocos patchados, gerando um candidato end-to-end executável — o pré-requisito que o
  protocolo V3 sempre exigiu para habilitar tok/s.

## Como tok/s finalmente se torna mensurável end-to-end

O `REAL_BENCHMARK_PROTOCOL_V3` publicava `baseline_tok_s = null` e `candidate_tok_s = null`
porque nenhuma tecnologia expunha um runtime candidato de modelo completo. O passo 16
remove exatamente essa limitação:

- **Baseline**: o modelo original, sem patch, executa `model.generate` com prompt,
  tokenização e parâmetros de geração fixos; tok/s = tokens gerados / tempo medido
  (`time.perf_counter_ns`, warmup, `torch.cuda.synchronize()` quando CUDA).
- **Candidato**: o MESMO modelo com todos os blocos patchados executa `model.generate`
  sob o MESMO protocolo (mesmo prompt, mesma tokenização, mesmos parâmetros, mesmo device).
- Ambos os valores entram nos campos de nível superior `baseline_tok_s` e
  `candidate_tok_s` SOMENTE neste registro (`C3_<TECH>_FULLMODEL_E2E_TOKS`).

Proxies de operação Linear/bloco continuam proibidos nos campos de comparação: ficam
apenas em `metrics.operation.*` com sufixo `_proxy`.

RAM segue a mesma disciplina: `*_ram_bytes` de nível superior é SOMENTE RSS medido —
thread de amostragem de `/proc/self/status` `VmRSS` a ~1 ms, pico por fase, método
registrado em `metrics.memory.method`. Estimativas aritméticas vivem apenas em
`metrics.memory.estimated_*`; sem medição → `null`.
`candidate_disk_bytes` = soma de `os.stat().st_size` de artefatos binários reais.

## Requisitos de schema (v2)

Todo registro da série C3 declara: `schema_version: 2`,
`benchmark_protocol: "C3_METHODOLOGY_V1"`, `comparison_group_id`
(`cmp-<sha256[:24]>` de `protocol|model_id|device|torch`), `comparison_context` e
`implementation {kind, native, simulated}`. Fallback sintético rebaixa o registro
(`comparison_role = null`, `activation_source = synthetic_fallback`); baterias simuladas
usam `status = "SIMULATED"` e `eligible_for_primary_ranking = false`.

Os registros primários da série C3 alimentam a política do WINNER dinâmico
(`selectWinnerArchitecture` / `select_winner_architecture` — contrato §1).

## Continuação: fases finais C4–C6 (`FINAL_PHASE_V1`)

A escada continua além da série C3 nas fases finais C4/C5/C6
(`batteries/final_phase_auto_batteries.py --technology rift|aether|cascade|spectra`,
protocolo `FINAL_PHASE_V1`), até o marco "compilador e executor LLM":
`C4_<TECH>_SECOND_FAMILY` (mesmo core em 2 famílias), `C5_<TECH>_REPR_BLOCKS`
(blocos representativos de modelo maior) e `C6_<TECH>_COMPILE_EXECUTE` (bundles
reais em disco, `generate` via mmap sem os pesos originais — única primária).
Contrato formal em [`docs/C3_CONTRACTS_V1.md`](C3_CONTRACTS_V1.md) §16.
