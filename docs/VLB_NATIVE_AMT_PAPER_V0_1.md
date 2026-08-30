# VLB Native + AMT: A Proof-First Runtime for Loss-Bounded LLM Re-Quantization

**Preprint v0.1 — 2026-08-30**  
**Project:** RIFT-LM / VLB Native  
**Status:** experimental; no full-model KR100 claim yet

## Abstract

VLB Native is an experimental model-conversion and inference stack designed to test whether an existing large language model can be re-quantized and executed through an independent runtime while preserving a frozen capability contract. The project deliberately separates storage compression, numerical kernel equivalence, full-model execution parity, capability retention and adaptive post-conversion learning. A compressed artifact is never considered verified merely because tensor reconstruction metrics are high or because it occupies fewer bytes.

The stack consists of a strict streaming converter, the `VLB1` binary container, VLB-owned low-bit kernels, a VLB-owned HTTP inference server, a proof ladder for full-model retention, and an Autodidactic Mastery Training (AMT) layer that is teacher-free and limited to empirically justified interventions. The first target architecture is `google/gemma-4-E4B-it`.

This paper specifies the current architecture and experimental contract. Numbers not produced by a real run remain unreported. At the time of this version, the native Q8_G64 kernel and server infrastructure compile and pass their self-tests, while complete Gemma 4 forward execution and full-model KR100 certification remain open milestones.

## 1. Motivation

Most quantization pipelines answer a storage question: how many bytes can be removed while maintaining an acceptable average metric. VLB asks a stricter question:

> What is the smallest executable representation that can preserve every unit of a frozen capability contract under the target runtime?

The distinction matters because mean perplexity, mean cosine similarity or aggregate accuracy can hide regressions in individual capabilities. VLB therefore uses a proof-first hierarchy in which compression and speed are secondary to retention.

VLB is not intended to be tied to one checkpoint. The goal is a model-agnostic conversion technology: a supported source LLM is converted to a VLB artifact and then executed by the VLB runtime rather than by the model's conventional inference engine.

## 2. Non-goals and claim discipline

VLB does **not** claim that:

- quantization is lossless by definition;
- tensor cosine similarity proves retained intelligence;
- a kernel microbenchmark proves end-to-end model speed;
- lower disk size proves lower peak RAM;
- a PyTorch or Transformers replay is a native VLB runtime;
- passing a small prompt set proves all possible intelligence was preserved.

A claim is published only at the level directly supported by an executed test.

## 3. System overview

```text
Hugging Face safetensors
        |
        v
strict HTTP byte-range streaming
        |
        v
VLB conversion / profile search
        |
        v
VLB1 binary model container
        |
        v
VLB native kernels
        |
        v
VLB model executor
        |
        +--> VLB KV cache
        +--> VLB tokenizer
        +--> VLB sampler
        |
        v
VLB server
        |
        v
frozen capability replay / KR100
        |
        v
AMT + adaptive compute (only after retention gates)
```

The final inference path must not depend on llama.cpp, bitnet.cpp, FreeToken, GGML/GGUF execution, PyTorch, Transformers, vLLM, ONNX Runtime or another external LLM runtime.

Reference frameworks may still be used on the **reference side** of an experiment to produce the canonical behavior that VLB must reproduce.

## 4. Streaming conversion

Large upstream checkpoints are read tensor-by-tensor. The converter first reads the Safetensors header, then requests only the byte range for the current tensor. The full source checkpoint is not intentionally materialized as a second local model copy.

The streaming reader requires a real HTTP partial-content response. If a server ignores the `Range` header and attempts to return the entire shard, the conversion must abort rather than consume the response as if it were a valid tensor range.

For each tensor the converter records provenance including source file, source byte range, tensor shape, source dtype, output format and cryptographic hashes.

## 5. VLB1 binary container

`VLB1` is the native on-disk contract for the final runtime. Its purpose is to remove the dependency on Python pickle or `torch.save` artifacts in the inference path.

A VLB1 model must provide sufficient metadata for deterministic tensor loading, including at minimum:

- format/version magic;
- tensor names and dimensions;
- tensor dtype/codec;
- group size where applicable;
- payload offsets and lengths;
- scale offsets and lengths;
- model/config metadata required by the executor;
- artifact integrity hashes at the packaging layer.

The format is intentionally independent of GGUF. Conversion from a source model to VLB1 is a new deployment artifact, not a rename of an existing runtime format.

## 6. Initial compression profile: Q8_G64

The first proof profile uses symmetric group-wise 8-bit weights with group size 64 for eligible matrices. For each group `g`:

```text
s_g = max(abs(w_g)) / 127
q_g = clamp(round(w_g / s_g), -127, 127)
```

and reconstruction is:

```text
w_hat_g = q_g * s_g
```

Q8_G64 is only the first rung because it is conservative enough to develop the native runtime without conflating runtime bugs with very aggressive compression. It is **not** assumed to preserve the model. Lower-bit profiles may be explored only after the proof machinery is working.

## 7. Native kernel

The first native operator is VLB Q8_G64 matrix-vector multiplication. Two CPU paths exist in the proof stack:

- scalar reference kernel;
- AVX2/FMA optimized kernel when supported by the CPU.

Kernel certification is numerical, not rhetorical. Real converted matrices are selected from the target model artifact and evaluated with real activation vectors. The proof records measured output disagreement such as maximum absolute error, RMSE, NRMSE and cosine similarity.

Passing this test certifies a kernel operator against its reference for the tested cases. It does not certify the intelligence of the complete model.

## 8. VLB server

`vlb-server` is a VLB-owned HTTP server. It must eventually own the full inference lifecycle:

- tokenizer;
- embeddings;
- normalization;
- attention;
- RoPE/position handling;
- feed-forward / expert layers;
- KV cache;
- VLB low-bit matrix kernels;
- logits;
- sampling / deterministic decode;
- AMT/adaptive-compute execution where certified.

The server must not silently fall back to an external LLM engine. If a required VLB operator or model architecture component is unavailable, the request fails explicitly.

## 9. Gemma 4 first target

The initial end-to-end target is:

`google/gemma-4-E4B-it`

Gemma 4 is intentionally useful as a proof target because it is materially larger than toy checkpoints and requires support for a modern architecture rather than a narrowly hard-coded legacy causal LM path.

The first certification scope is text execution. Multimodal retention is a separate proof contract and must not be inferred from text-only success.

## 10. KR100 definition

VLB uses **KR100** as a frozen-contract retention gate.

Let the frozen validation contract contain `N` units. For each unit `i`, the canonical reference produces metrics and/or expected outputs defined by the contract. The candidate VLB runtime is evaluated on exactly the same unit without using that unit for training or calibration.

A candidate satisfies KR100 only when:

```text
passed_units == total_frozen_units
```

under the per-unit rules of that certification contract.

KR100 therefore means **100% retention of the frozen certified contract**. It does not mean that all possible behavior of the original neural network has been mathematically proven identical.

Aggregate improvements cannot compensate for a failed unit.

## 11. Proof ladder

The current proof ladder is:

### P0 — Container integrity

The VLB1 artifact is structurally valid, complete, hashable and independently readable by VLB tooling.

### P1 — Native kernel equivalence

Native VLB kernels execute real matrices from the converted target artifact and pass their numerical operator contract.

### P2 — Complete native model replay

The VLB server performs a complete model forward/decode without loading the source weights through an external inference runtime.

### P3 — Frozen KR100 certification

The complete native runtime is replayed against the frozen capability/quality contract. Every unit must pass.

### P4 — Native AMT and adaptive compute

Only after the native base route is measurable can AMT/adaptive compute be evaluated without confusing framework behavior with VLB behavior.

A deployment may be called `VLB_NATIVE_DEPLOYMENT_CERTIFIED` only after the required proof stages for that profile pass.

## 12. AMT: Autodidactic Mastery Training

AMT is teacher-free. It does not optimize against teacher logits, KL divergence or hidden targets from another model.

The core principle is **minimum sufficient intervention**. A capability is not automatically retrained when a shallow route fails. Validation should distinguish at least:

- already mastered;
- compute-limited;
- likely learning gap;
- regressed;
- stalled.

AMT performs short targeted intervention bursts followed by fresh validation. A candidate state is committed only when it improves the defined target without violating the preservation contract. Otherwise the intervention is rolled back.

Training-token budget is a ceiling, not an obligation.

## 13. Adaptive compute and marginal-gain gating

The adaptive-compute gate must not rely on textual self-confidence. Its empirical target is marginal utility:

> Is another compute/recurrent step expected to improve the current answer under the validation contract?

Candidate gate features can include the current latent state, latent-state delta, output entropy, top-1/top-2 margin and recurrence identity. Labels are generated from measured validation deltas, not from model statements such as “I need to think more.”

## 14. Measurement contract

The following values are reported only when physically measured by the relevant process:

- source artifact bytes;
- VLB artifact bytes;
- peak RSS;
- peak accelerator memory where applicable;
- prefill time;
- decode time;
- token throughput;
- per-unit loss / NLL where the contract uses it;
- top-1 correctness where the contract uses it;
- generated tokens;
- kernel numerical error;
- KR passed/total.

If a value has not been measured, it remains absent/null rather than being replaced by an estimate in the primary result.

## 15. Current verified status

As of preprint v0.1:

| Component | Status |
|---|---|
| Strict Safetensors byte-range converter | implemented |
| VLB1 native binary container | implemented proof format |
| Native Q8_G64 scalar kernel | implemented / self-test |
| Native Q8_G64 AVX2/FMA kernel | implemented / CI build |
| Native HTTP server infrastructure | implemented / health self-test |
| Real Gemma matrix kernel-proof path | integrated into battery |
| Complete Gemma 4 forward in VLB server | **not yet certified** |
| VLB-owned tokenizer | **not yet certified** |
| VLB-owned KV cache | **not yet certified** |
| Full-model Gemma KR100 | **not yet tested/certified** |
| Native AMT after full-model replay | **not yet certified** |

No compression ratio, tok/s advantage, RAM advantage or Gemma intelligence-retention number is claimed in this paper until the corresponding real experiment has completed.

## 16. Falsification criteria

The VLB hypothesis is falsified for a tested profile when, after implementation defects are excluded, one or more of the following remains true:

- the compressed native artifact cannot satisfy the frozen retention contract;
- preserving the contract requires effectively reverting the candidate to the canonical representation;
- the native runtime introduces irreducible capability regressions;
- AMT cannot consolidate required capabilities without violating retention;
- measured efficiency does not improve relative to the reference despite passing retention.

Such a result is recorded rather than hidden.

## 17. Reproducibility

Each certified experiment should retain:

- source model ID and immutable revision/hash when available;
- conversion profile and parameters;
- VLB artifact hash;
- runtime build/compiler/CPU feature information;
- frozen validation fingerprint;
- per-unit validation ledger;
- aggregate metrics derived from that ledger;
- command/config used to reproduce the run.

## 18. Roadmap

The next engineering milestone is complete Gemma 4 text forward execution inside `vlb-server`, including VLB-owned tokenization/decode plumbing and KV cache. Once native replay is exact enough to execute the frozen evaluation, the project can run the first real Gemma VLB KR100 experiment.

Only after that result should the compression ladder be pushed below Q8 or the runtime be advertised as a verified replacement for the source model's conventional engine.

## 19. Citation

Until an archival identifier is assigned, cite this work as:

> RIFT-LM Project. **VLB Native + AMT: A Proof-First Runtime for Loss-Bounded LLM Re-Quantization.** Preprint v0.1, 2026.

Repository: `programador-powershell/RIFT-LM`
