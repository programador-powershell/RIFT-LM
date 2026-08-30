# VLB/AMT Engine v1 — generic re-quantization contract

## Goal

VLB is not a checkpoint-specific optimization. It is a **conversion + runtime contract** intended to be applied to arbitrary Hugging Face LLM/VLM checkpoints.

The target flow is:

```text
upstream HF checkpoint
        |
        v
streaming tensor reader
        |
        v
VLB re-quantization
        |
        +--> reconstruction proof
        |
        v
VLB-DIR artifact
        |
        v
VLB runtime
        |
        +--> frozen base route
        +--> adaptive latent/residual compute
        +--> empirical marginal-gain gate
        |
        v
AMT minimum-sufficient-exposure adaptation
        |
        v
fresh validation
        |
        v
VERIFIED VLB/AMT deployment
```

A model does **not** become a verified VLB model merely because its weights were quantized.

## First proof target

`google/gemma-4-E4B-it`

Gemma 4 is deliberately the first target because it is small enough for Colab experimentation while still exercising a modern multimodal Transformers architecture.

The launcher is exposed at:

```text
/vlb/google/gemma-4-E4B-it
```

The Colab battery is:

```text
engines/vlb/vlb_amt_streaming_battery.py
```

## Streaming requirement

The source checkpoint must not be downloaded as a complete local weight file before conversion.

VLB v1 reads the Safetensors header first, then requests each tensor's byte range from Hugging Face independently:

```text
HTTP Range(source tensor)
 -> RAM
 -> quantize
 -> verify reconstruction
 -> write VLB tensor artifact
 -> release source tensor
 -> next tensor
```

The manifest records:

- `source_checkpoint_materialized=false`
- transport (`HTTP_RANGE`)
- source byte range for each tensor
- source Safetensors header SHA-256
- artifact SHA-256 per converted tensor
- source bytes / artifact bytes
- cosine, NRMSE and maximum absolute reconstruction error

## VLB-DIR v1

The initial format uses a directory rather than a monolithic file so conversion and resume can remain tensor-granular.

```text
<model>-q8_g64/
  config.json
  generation_config.json
  vlb_manifest.json
  result.json
  tensors/
    <hash>_<tensor>.pt
    ...
```

### Initial precision

The first proof uses `Q8_G64` for eligible 2-D floating-point weights.

Non-eligible tensors remain FP16 or RAW passthrough.

This is intentional. VLB first proves the engine/runtime contract at a conservative precision. Q4/mixed-bit are later Deployment Profiles and must independently pass the same runtime/capability gates.

## Three independent proof gates

### Gate A — conversion

`CONVERSION_VERIFIED`

A tensor quantization pass is accepted only if all quantized tensors satisfy the declared reconstruction limits.

This gate proves only numerical reconstruction of weights. It does not prove model behavior.

### Gate B — runtime

`VLB_RUNTIME_VERIFIED`

The model must execute **from VLB-DIR**, through VLB runtime modules. Loading the upstream checkpoint with standard Transformers/bitsandbytes and merely attaching a VLB label is forbidden.

The runtime proof must compare deterministic baseline and VLB execution under the same prompts/protocol.

### Gate C — AMT

`AMT_VERIFIED`

AMT must:

- keep the VLB base frozen;
- use GOLD examples, not a teacher model or teacher logits;
- train only the minimum residual/gate state needed;
- use separate learn and fresh-validation examples;
- commit only interventions that preserve or improve validation;
- rollback regressions automatically;
- learn STOP/CONTINUE from empirical marginal gain, never textual self-confidence.

Only `A && B && C` yields:

```text
VLB_AMT_ENGINE_PROOF_PASS
```

## Current implementation status

The first repository change intentionally implements **Gate A first** and returns an explicit blocked runtime status for Gates B/C.

This is not a limitation hidden by the dashboard. A conversion-only record publishes:

```text
quality_gate_pass = false
metrics.proof.engine_status = VLB_AMT_ENGINE_NOT_YET_PROVEN
```

Therefore VLB cannot win the normal technology ranking until generation actually runs from VLB-DIR and the AMT gate is validated.

## Next runtime milestone

Implement a generic lazy Transformers loader:

1. instantiate the architecture from `config.json` under `accelerate.init_empty_weights()`;
2. replace quantized `nn.Linear` modules with `VLBQuantLinear`;
3. keep Q8 weights/scales compressed in CPU RAM;
4. dequantize only the active layer/block on demand;
5. load FP16/RAW passthrough tensors directly from VLB-DIR;
6. execute deterministic text-only Gemma 4 prompts;
7. compare upstream baseline vs VLB output;
8. add RC-LR residual compute + marginal-gain gate;
9. run teacher-free AMT on a separate development split;
10. publish only after fresh-validation replay.

## Generalization contract

Architecture-specific code is permitted only inside adapters/loaders. The converter format, manifest, verification and AMT scheduler must remain model-agnostic.

Future targets should require no change to the high-level flow:

```text
model adapter -> VLB-DIR -> VLB runtime -> AMT -> proof
```

If a model family needs an architecture adapter, the dashboard should report that explicitly rather than silently falling back to the standard model engine.
