# RIFT-LM Real Benchmark Protocol V3

Protocol ID: `LINEAR_REAL_MEASURED_V3`

This protocol exists to prevent estimated, simulated, or synthetic values from being presented as measured benchmark gains.

## Primary rule

A number may enter a primary comparison only when the quantity was observed directly under a reproducible measurement scope.

The protocol deliberately reports `null` when the current runtime cannot isolate or measure a quantity truthfully.

## Latency

The existing technology battery is executed with its `benchmark_ms()` instrumented by the real benchmark runner.

Minimum protocol:

- 10 warmup executions;
- 50 measured executions by default;
- `time.perf_counter_ns()` timing;
- CUDA synchronization immediately before and after the measured function when CUDA is used;
- median;
- mean;
- p95;
- p99;
- minimum;
- maximum;
- population standard deviation.

Method identifier:

`perf_counter_ns_with_cuda_sync_v1`

These values are **single-operation Linear latency**, not model-level token generation throughput.

## CPU memory

During each measured operation the runner samples `/proc/self/status` `VmRSS` every 1 ms and records:

- RSS before;
- peak RSS;
- RSS after;
- observed peak delta.

The complete technology subprocess tree is also sampled externally every 20 ms.

Method identifiers:

- `proc_rss_1ms_and_torch_cuda_peak_v1`
- `external_proc_tree_sampling_v1`

### Important RAM rule

The current technology batteries usually keep original and candidate structures alive in the same Python process. Therefore the protocol **does not publish formula-derived `baseline_ram_bytes` or `candidate_ram_bytes` as primary RAM values**.

Observed process and operation memory remains available under `metrics.memory_real`.

Top-level RAM comparison stays `null` until baseline and candidate resident inference states can be executed and measured in isolated processes.

## GPU memory

When CUDA is available the operation probe records actual PyTorch allocator measurements using:

- `torch.cuda.memory_allocated()`;
- `torch.cuda.memory_reserved()`;
- `torch.cuda.reset_peak_memory_stats()`;
- `torch.cuda.max_memory_allocated()`;
- `torch.cuda.max_memory_reserved()`.

The outer runner additionally samples `nvidia-smi --query-compute-apps=pid,used_memory` for the technology process tree.

These values are observed measurements, not estimates.

## Disk/storage

### Baseline tensor

When the model is a cached Safetensors checkpoint, source tensor bytes are read directly from the tensor's `data_offsets` in the Safetensors header.

Method identifier:

`safetensors_data_offsets_v1`

### Candidate

The candidate disk value is accepted only if its reported byte count exactly matches a generated **binary artifact** found with `os.stat()`.

JSON, CSV, TXT, LOG, Python, Markdown, YAML and YML files cannot satisfy this proof.

Binary artifacts are inventoried and small/medium artifacts are SHA-256 hashed.

Method identifier:

`binary_os_stat_and_sha256_manifest_v2`

If no binary artifact proves the reported candidate size, `candidate_disk_bytes` becomes `null` for the sanitized result.

## Tok/s

No technology in the current Phase 1 comparison exposes a complete candidate model runtime capable of end-to-end generation under the same execution protocol.

Therefore:

- `baseline_tok_s = null`;
- `candidate_tok_s = null`;
- all technology-specific Tok/s aliases are `null`.

Rows/s from a Linear operation, a theoretical speedup, or a synthetic native microbenchmark must never be converted into model Tok/s.

Tok/s will only be enabled after baseline and candidate full-model generation can be executed with the same prompt/tokenization/generation settings.

## Activation provenance

A primary real-model battery requires an activation captured from an actual forward pass of the selected Hugging Face model.

Accepted source:

`real_model_activation`

If the technology falls back to a deterministic random/synthetic activation, its primary record is changed to:

- `comparison_role = diagnostic`;
- `status = INVALID_REAL_INPUT`;
- `implementation.eligible_for_primary_ranking = false`.

## Simulation

Batteries such as predictive prefetch policy simulations remain useful diagnostics, but they cannot enter primary ranking.

They are marked:

```json
{
  "implementation": {
    "kind": "SIMULATED",
    "native": false,
    "simulated": true,
    "eligible_for_primary_ranking": false
  }
}
```

## Reference vs native

The current RIFT/CASCADE/AETHER/SPECTRA model paths are PyTorch/reference implementations. Measuring them accurately does not make them native production kernels.

Valid real primary records are marked `REFERENCE_MEASURED` until a native model kernel executes the same real model operation.

A separate native self-test is not allowed to substitute for model-level native latency.

## Comparison group

The comparison fingerprint includes:

- protocol ID;
- model ID;
- local Hugging Face snapshot revision when available;
- resolved target layer;
- actual device type;
- GPU descriptor;
- platform/machine;
- Python version;
- PyTorch version;
- Transformers version;
- measured iterations;
- warmup count.

Only records with compatible `comparison_group_id` may be ranked against each other.

## Current limitations

This V3 protocol intentionally does **not** claim:

- model-level Tok/s;
- isolated baseline-vs-candidate required RAM;
- real asynchronous NVMe prefetch performance unless a native implementation executes the I/O;
- end-to-end perplexity or generation equivalence unless a full-model candidate runtime exists.

Those fields must remain absent/null rather than be inferred.

## Adendo C3_METHODOLOGY_V1

This addendum does not alter any rule above; it records the sanctioned path that satisfies them.

C3 full-model records (`benchmark_protocol = "C3_METHODOLOGY_V1"`, battery
`C3_<TECH>_FULLMODEL_E2E_TOKS`) are the **sanctioned source of candidate tok/s**: baseline and
candidate are both measured from full-model `model.generate` under the same prompt, tokenization
and generation settings, which is exactly the condition the Tok/s section requires before
`baseline_tok_s`/`candidate_tok_s` may be non-null. Linear/block proxies remain confined to
`metrics.operation.*` with the `_proxy` suffix and still must never enter comparison fields.

The same records are the sanctioned source of **measured-RSS RAM**: top-level `*_ram_bytes` come
only from per-phase `/proc/self/status` `VmRSS` sampling (~1 ms sampling thread, per-phase peak,
method recorded in `metrics.memory.method`). Arithmetic estimates stay under
`metrics.memory.estimated_*`; when nothing was measured the top-level value stays `null`.

All other batteries remain bound by the null rules of this protocol unchanged. See
`docs/C3_METHODOLOGY.md` and `docs/C3_CONTRACTS_V1.md` for the full C3 specification.

## Adendo E2E_TOKS_V1

Este adendo é APPEND-ONLY: nenhuma regra anterior deste protocolo é alterada. Ele registra o
caminho sancionado pelo qual registros end-to-end passam a preencher `baseline_tok_s` /
`candidate_tok_s` de nível superior (docs/C3_CONTRACTS_V1.md §12) — exatamente a condição que a
seção "Tok/s" exige: baseline e candidato gerados pelo modelo COMPLETO sob o mesmo
prompt/tokenização/configuração de geração.

Baterias e2e sancionadas (as ÚNICAS autorizadas a preencher tok/s de topo):

| battery_id | Escopo da medição |
| --- | --- |
| `P1_RIFT_E2E_TOKS`, `P1_AETHER_E2E_TOKS`, `P1_SPECTRA_E2E_TOKS`, `P1_WINNER_E2E_TOKS` | `model.generate` do modelo completo (greedy, ≥2 warmup, ≥3 medições, mediana); candidato = MESMO `generate` com todas as Linear dos blocos no runtime de referência Python do codec da tecnologia (W denso fora do caminho quente) |
| `P1_CASCADE_C2_E2E_TOKS` | mesma técnica dos M0 acima: baseline E candidato medidos; candidato = todas as nn.Linear dos blocos patchadas com `CascadeLinearModule` (F0 INT4 + Gate·F1, low_mem off), patch transacional com restauração garantida |
| `C3_<TECH>_FULLMODEL_E2E_TOKS` | passo 16 da metodologia C3 (já sancionado pelo Adendo C3_METHODOLOGY_V1 acima) |
| `G3_GEYSER_BURST` | escopo Python: promove `vanilla_tok_s_py` → `baseline_tok_s` e `burst_tok_s_py` → `candidate_tok_s`, ambos wall-clock REAIS sob o mesmo protocolo greedy, com equivalência greedy verificada na própria bateria; "tok/s medido em Python — não representa kernel nativo" |
| `P1_GGUF_E2E_TOKS` | escopo llama.cpp (decode real no T4, prompt fixo, ≥3 medições, mediana): `candidate_tok_s` medido; `baseline_tok_s = null` porque o baseline BF16 não é executável no T4 (nenhuma comparação é inventada) |

Regras do adendo:

- Todo registro sancionado DEVE gravar `metrics.e2e.measured = true`, com `metrics.e2e.scope`
  declarando o escopo (ex.: `python_reference_wall_clock`, `python_reference_model_generate`).
  Sem essa marca o registro permanece sujeito à regra de anulação da seção "Tok/s".
- `scripts/real_benchmark_runner.py` deixa de anular `baseline_tok_s`/`candidate_tok_s` SOMENTE
  quando `battery_id` termina em `_E2E_TOKS` E `metrics.e2e.measured === true` (baseline e
  candidato passam adiante). Proxies, aliases de tok/s por tecnologia e todos os demais campos
  continuam sendo anulados exatamente como antes. `G3_GEYSER_BURST` e `P1_GGUF_E2E_TOKS` não
  transitam pelo runner (publicam direto pelos próprios launchers) — a exceção do runner cobre
  apenas os ids `*_E2E_TOKS`.
- O `measurement_scope` de cada registro sancionado declara o escopo honesto (runtime de
  referência Python / Python wall-clock / llama.cpp), deixando explícito que NÃO representa
  kernel nativo.
- Proxies de Linear/bloco continuam confinados a `metrics.operation.*` com sufixo `_proxy` e
  seguem PROIBIDOS nos campos de comparação de nível superior.
