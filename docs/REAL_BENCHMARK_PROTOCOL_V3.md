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
