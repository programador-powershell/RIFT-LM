# VLB/AMT Engine v1 — generic re-quantization contract

## Goal

VLB is not a checkpoint-specific optimization. It is a **conversion + runtime contract** intended to be applied to arbitrary Hugging Face LLM/VLM checkpoints.

The implemented proof flow is:

```text
upstream HF safetensors
        |
        v
HTTP Range tensor streaming
        |
        v
VLB re-quantization
        |
        +--> per-tensor reconstruction proof
        |
        v
VLB-DIR artifact
        |
        v
lazy VLB runtime
        |
        +--> VLBQuantLinear Q8 weights
        +--> FP16/RAW compatibility tensors
        +--> no upstream weight fallback
        |
        v
teacher-free AMT residual
        |
        +--> empirical marginal-gain gate
        |
        v
fresh validation
        |
        v
VLB_AMT_ENGINE_PROOF_PASS only if every gate passes
```

A model does **not** become a verified VLB model merely because its weights were quantized.

## First proof target

`google/gemma-4-E4B-it`

Gemma 4 is the first target because it is small enough for a Colab/T4 proof while still exercising a current multimodal Transformers architecture. The proof path is text-only initially; image capability is not claimed until a separate multimodal replay is added.

The launcher is exposed at:

```text
/vlb/google/gemma-4-E4B-it
```

The dedicated site page is:

```text
/vlb-lab
```

The implementation lives in:

```text
engines/vlb/vlb_amt_streaming_battery.py
engines/vlb/vlb_runtime.py
```

## Streaming requirement

The source checkpoint must not be downloaded as a complete local weight file before conversion.

VLB reads the Safetensors header first, then requests each tensor byte range independently:

```text
HTTP Range(source tensor)
 -> RAM
 -> quantize
 -> verify reconstruction
 -> write VLB tensor artifact
 -> release source tensor
 -> next tensor
```

A full source response where a byte range was requested is rejected. The converter therefore does not silently fall back to downloading the full checkpoint.

The manifest records:

- `source_checkpoint_materialized=false`
- transport (`HTTP_RANGE`)
- source byte range for each tensor
- source Safetensors header SHA-256
- artifact SHA-256 per converted tensor
- source bytes / artifact bytes
- cosine, NRMSE and maximum absolute reconstruction error

## VLB-DIR v1

The first format is directory-based so conversion, audit and future resume can remain tensor-granular:

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

The first proof uses `Q8_G64` for eligible 2-D floating-point weights. Embedding/tied-head tensors are preserved in FP16 in the initial Gemma 4 proof so weight tying is not confused with the quantized Linear runtime problem. Other non-eligible tensors remain FP16 or RAW passthrough.

This is intentional. Q4, mixed-bit and more aggressive profiles are later Deployment Profiles and must independently pass the runtime/capability gates.

## Gate A — conversion

`CONVERSION_VERIFIED`

Every quantized tensor must satisfy the declared numerical reconstruction limits.

This gate proves **weight reconstruction only**. It is not KR100 and it does not prove model behavior.

## Gate B — VLB runtime

`VLB_RUNTIME_VERIFIED`

The runtime is implemented in `engines/vlb/vlb_runtime.py`.

It:

1. instantiates the model architecture from config under `accelerate.init_empty_weights()`;
2. replaces eligible `nn.Linear` modules with `VLBQuantLinear`;
3. loads Q8 weights/scales from VLB-DIR;
4. loads FP16/RAW compatibility tensors from VLB-DIR;
5. resolves tied weights;
6. refuses unresolved meta tensors;
7. never calls `from_pretrained()` for upstream model weights;
8. executes a deterministic text-forward smoke from the VLB artifact.

The processor/tokenizer may still be obtained from Hugging Face; model weights may not.

If the architecture cannot be bound completely from VLB-DIR, runtime verification fails. There is no standard-weight fallback hidden behind the VLB label.

## Gate C — AMT

`AMT_VERIFIED`

The first generic AMT proof is intentionally small. It is meant to prove the mechanism, not claim broad post-training mastery.

The VLB base is frozen. AMT trains only:

- a low-rank residual logits head;
- a small marginal-gain gate.

The examples are partitioned into three non-overlapping sets:

```text
AMT_LEARN
AMT_GATE_DEV
AMT_FRESH_VALID
```

There is no teacher model, teacher logits or KL objective.

Gate labels are empirical:

```text
residual improves exact target correctness
        -> CONTINUE

same correctness + lower CE
        -> CONTINUE

otherwise
        -> STOP
```

AMT passes only if, on the fresh validation split, adaptive execution preserves/improves exact correctness **and** does not worsen mean CE.

## Combined status

Only all three proof gates produce:

```text
conversion_verified = true
runtime_verified    = true
amt_verified        = true
--------------------------------
VLB_AMT_ENGINE_PROOF_PASS
quality_gate_pass = true
```

Any failed gate produces:

```text
VLB_AMT_ENGINE_NOT_YET_PROVEN
quality_gate_pass = false
```

A failed experiment may still be published to the Observatório as experimental evidence, but it cannot be ranked as a verified VLB result.

## Integration with RIFT-LM

VLB is model-specific at execution time but technology-generic at the platform level.

The ordinary queue accepts an explicit:

```text
TECHS = ["vlb"]
```

and `TECHS=["all"]` appends the VLB proof after the historical technology list for every non-GGUF Hugging Face model.

VLB results are written through:

```text
/api/vlb-results
```

into the same append-only history used by the Observatório. The history validator recognizes `technology=VLB`, but a record is eligible only when its own proof gates pass.

## What this first proof does not establish

Even if Gemma 4 passes, the result proves the VLB/AMT mechanism only for the tested execution contract. It does not automatically prove:

- KR100 across all Gemma capabilities;
- multimodal/image retention;
- Q4 or mixed-bit safety;
- identical quality on every future architecture;
- that every model benefits from AMT.

Those require additional frozen capability batteries.

## Generalization contract

Architecture-specific code is permitted only inside adapters/loaders. The following layers remain model-agnostic:

- streaming Safetensors reader;
- VLB-DIR format;
- quantization manifest;
- reconstruction proof;
- VLB proof statuses;
- AMT learn/gate/fresh-validation discipline;
- Observatório publication contract.

Future families should retain the same high-level flow:

```text
model adapter -> VLB-DIR -> VLB runtime -> AMT -> capability proof
```

If a model family requires an adapter, the dashboard must report that explicitly rather than silently falling back to the standard model engine.
