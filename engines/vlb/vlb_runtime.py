#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLB lazy runtime + teacher-free AMT proof v1.

This module loads a VLB-DIR artifact without loading the upstream model weights.
Quantized nn.Linear weights stay Q8 in the package/runtime and are dequantized
per active layer during forward. Non-quantized tensors are loaded from VLB-DIR.

The loader is generic at the module level and is first certified against
Gemma4ForConditionalGeneration / AutoModelForMultimodalLM.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor


class VLBQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        shape: Sequence[int],
        bias: bool,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.shape = tuple(int(x) for x in shape)
        self.register_buffer("qweight", qweight.contiguous(), persistent=True)
        self.register_buffer("scales", scales.contiguous(), persistent=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def materialize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        q = self.qweight.to(device=device, non_blocking=True)
        scales = self.scales.to(device=device, dtype=torch.float32, non_blocking=True)
        flat = (q.float() * scales[:, None]).reshape(-1)
        numel = math.prod(self.shape)
        return flat[:numel].reshape(self.shape).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.materialize_weight(x.dtype, x.device)
        bias = self.bias
        if bias is not None and bias.device != x.device:
            bias = bias.to(x.device)
        out = F.linear(x, weight, bias)
        del weight
        return out


class AMTResidualHead(nn.Module):
    """Model-agnostic teacher-free residual logits head.

    The base VLB model is frozen. AMT trains this low-rank head on GOLD examples
    only. The marginal gate later decides when using the head is beneficial.
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden)))


class AMTMarginalGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _load_payload(root: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    return torch.load(root / record["file"], map_location="cpu", weights_only=False)


def _get_parent_and_leaf(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, leaf: str, module: nn.Module) -> None:
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _dequant_payload(payload: Dict[str, Any], dtype: torch.dtype = torch.float16) -> torch.Tensor:
    fmt = payload["format"]
    if fmt == "Q8_G64":
        q = payload["qweight"].float()
        scales = payload["scales"].float()
        shape = tuple(int(x) for x in payload["shape"])
        numel = math.prod(shape)
        return (q * scales[:, None]).reshape(-1)[:numel].reshape(shape).to(dtype)
    if fmt == "FP16":
        return payload["tensor"].to(dtype)
    if fmt == "RAW":
        return payload["tensor"]
    raise RuntimeError(f"Unknown VLB payload format: {fmt}")


def load_vlb_model(
    model_id: str,
    vlb_dir: Path,
    token: Optional[str],
    device: torch.device,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    root = Path(vlb_dir)
    manifest = json.loads((root / "vlb_manifest.json").read_text(encoding="utf-8"))
    records = {r["name"]: r for r in manifest["tensor_records"]}

    config = AutoConfig.from_pretrained(model_id, token=token)
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(config)

    # Replace Q8 nn.Linear weights before materializing ordinary parameters.
    modules = dict(model.named_modules())
    replaced = set()
    for name, record in records.items():
        if record["format"] != "Q8_G64" or not name.endswith(".weight"):
            continue
        module_path = name[:-len(".weight")]
        module = modules.get(module_path)
        if not isinstance(module, nn.Linear):
            continue
        payload = _load_payload(root, record)
        parent_path, leaf = module_path.rsplit(".", 1) if "." in module_path else ("", module_path)
        parent = model if not parent_path else dict(model.named_modules()).get(parent_path)
        if parent is None:
            raise RuntimeError(f"Cannot resolve parent module for {module_path}")
        replacement = VLBQuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            qweight=payload["qweight"],
            scales=payload["scales"],
            group_size=int(payload["group_size"]),
            shape=payload["shape"],
            bias=module.bias is not None,
        )
        _set_child(parent, leaf, replacement)
        replaced.add(name)
        del payload

    # Materialize all other state tensors from VLB-DIR. Quantized tensors that
    # were not nn.Linear are dequantized once as compatibility fallback.
    for name, record in records.items():
        if name in replaced:
            continue
        payload = _load_payload(root, record)
        value = _dequant_payload(payload)
        try:
            set_module_tensor_to_device(model, name, "cpu", value=value)
        except Exception as exc:
            # Tied lm_head weights are commonly omitted by safetensors and thus
            # are not in records; a present tensor that cannot bind is a real
            # contract mismatch and must abort.
            raise RuntimeError(f"VLB runtime cannot bind tensor {name}: {exc}") from exc
        del payload, value

    # Resolve tied weights such as Gemma text embeddings/lm_head.
    model.tie_weights()

    # Refuse any unresolved meta parameter/buffer. Falling back to upstream
    # weights would invalidate the proof.
    unresolved = []
    for name, p in model.named_parameters():
        if getattr(p, "is_meta", False):
            unresolved.append(name)
    for name, b in model.named_buffers():
        if getattr(b, "is_meta", False):
            unresolved.append(name)
    if unresolved:
        raise RuntimeError("VLB runtime has unresolved meta tensors: " + ", ".join(unresolved[:40]))

    model.eval()
    # Q8 buffers + FP16 passthrough move to GPU. Weight dequantization is per
    # active Linear call, so there is no full BF16 checkpoint resident at once.
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    return model, processor, manifest


def _extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    for attr in ("language_model_output", "text_model_output", "model_output"):
        nested = getattr(outputs, attr, None)
        nested_hs = getattr(nested, "hidden_states", None) if nested is not None else None
        if nested_hs:
            return nested_hs[-1]
    raise RuntimeError("Model output does not expose text hidden_states required by AMT")


def _prepare_text(processor: Any, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        inputs = processor(text=text, return_tensors="pt")
    except TypeError:
        inputs = processor(text, return_tensors="pt")
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
    return result


def _target_token(processor: Any, target: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(target, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"][0]
    if ids.numel() == 0:
        raise RuntimeError(f"Target tokenization is empty: {target!r}")
    return int(ids[0].item())


def _forward_one(
    model: nn.Module,
    processor: Any,
    prompt: str,
    target: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    inputs = _prepare_text(processor, prompt, device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    logits = outputs.logits[:, -1, :].float()
    hidden = _extract_hidden(outputs)[:, -1, :].float()
    target_id = _target_token(processor, target)
    return hidden, logits, target_id


def _gate_features(logits: torch.Tensor, delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    delta_norm = delta.norm(dim=-1) / math.sqrt(delta.shape[-1])
    hidden_norm = hidden.norm(dim=-1) / math.sqrt(hidden.shape[-1])
    return torch.stack([entropy, margin, delta_norm, hidden_norm], dim=-1)


def _evaluate_examples(
    model: nn.Module,
    processor: Any,
    adapter: AMTResidualHead,
    gate: AMTMarginalGate,
    examples: Sequence[Tuple[str, str]],
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    base_loss = 0.0
    adaptive_loss = 0.0
    base_correct = 0
    adaptive_correct = 0
    decisions = []
    adapter.eval()
    gate.eval()
    for prompt, target in examples:
        hidden, logits, target_id = _forward_one(model, processor, prompt, target, device)
        with torch.no_grad():
            delta = adapter(hidden)
            features = _gate_features(logits, delta, hidden)
            p_continue = torch.sigmoid(gate(features))
            use_delta = p_continue >= threshold
            adapted = torch.where(use_delta[:, None], logits + delta, logits)
            base_ce = F.cross_entropy(logits, torch.tensor([target_id], device=device))
            adaptive_ce = F.cross_entropy(adapted, torch.tensor([target_id], device=device))
        base_loss += float(base_ce.item())
        adaptive_loss += float(adaptive_ce.item())
        base_correct += int(logits.argmax(dim=-1).item() == target_id)
        adaptive_correct += int(adapted.argmax(dim=-1).item() == target_id)
        decisions.append(
            {
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "target_id": target_id,
                "p_continue": float(p_continue.item()),
                "continue": bool(use_delta.item()),
                "base_ce": float(base_ce.item()),
                "adaptive_ce": float(adaptive_ce.item()),
            }
        )
    n = max(1, len(examples))
    return {
        "examples": len(examples),
        "base_mean_ce": base_loss / n,
        "adaptive_mean_ce": adaptive_loss / n,
        "base_correct": base_correct,
        "adaptive_correct": adaptive_correct,
        "decisions": decisions,
    }


def run_amt_proof(
    model: nn.Module,
    processor: Any,
    device: torch.device,
    train_examples: Sequence[Tuple[str, str]],
    gate_examples: Sequence[Tuple[str, str]],
    validation_examples: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    # Resolve dimensions from one GOLD example.
    hidden0, logits0, _ = _forward_one(model, processor, train_examples[0][0], train_examples[0][1], device)
    hidden_size = int(hidden0.shape[-1])
    vocab_size = int(logits0.shape[-1])
    del hidden0, logits0

    adapter = AMTResidualHead(hidden_size, vocab_size, rank=8).to(device)
    gate = AMTMarginalGate().to(device)

    # Freeze the entire VLB base. Only AMT modules train.
    for p in model.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=0.01)
    adapter.train()
    train_history = []
    for epoch in range(12):
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for prompt, target in train_examples:
            hidden, logits, target_id = _forward_one(model, processor, prompt, target, device)
            delta = adapter(hidden)
            loss = F.cross_entropy(logits.detach() + delta, torch.tensor([target_id], device=device))
            (loss / len(train_examples)).backward()
            total += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        train_history.append(total / len(train_examples))

    # Empirical marginal labels on a separate gate-development split.
    gate_x = []
    gate_y = []
    labels = []
    adapter.eval()
    for prompt, target in gate_examples:
        hidden, logits, target_id = _forward_one(model, processor, prompt, target, device)
        with torch.no_grad():
            delta = adapter(hidden)
            base_ce = F.cross_entropy(logits, torch.tensor([target_id], device=device))
            refined_ce = F.cross_entropy(logits + delta, torch.tensor([target_id], device=device))
            base_ok = int(logits.argmax(dim=-1).item() == target_id)
            refined_ok = int((logits + delta).argmax(dim=-1).item() == target_id)
            continue_label = 1.0 if (refined_ok > base_ok or (refined_ok == base_ok and refined_ce < base_ce)) else 0.0
            gate_x.append(_gate_features(logits, delta, hidden).squeeze(0).cpu())
            gate_y.append(continue_label)
            labels.append(
                {
                    "base_ce": float(base_ce.item()),
                    "refined_ce": float(refined_ce.item()),
                    "base_correct": base_ok,
                    "refined_correct": refined_ok,
                    "label": "CONTINUE" if continue_label else "STOP",
                }
            )

    x = torch.stack(gate_x).to(device)
    y = torch.tensor(gate_y, dtype=torch.float32, device=device)
    gate_opt = torch.optim.AdamW(gate.parameters(), lr=1e-2, weight_decay=0.0)
    gate.train()
    best_loss = float("inf")
    best_state = None
    for _ in range(120):
        gate_opt.zero_grad(set_to_none=True)
        pred = gate(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        gate_opt.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    if best_state is not None:
        gate.load_state_dict(best_state, strict=True)

    validation = _evaluate_examples(model, processor, adapter, gate, validation_examples, device)
    pass_amt = (
        validation["adaptive_correct"] >= validation["base_correct"]
        and validation["adaptive_mean_ce"] <= validation["base_mean_ce"] + 1e-6
    )
    return {
        "verified": bool(pass_amt),
        "teacher": "NONE",
        "training_examples": len(train_examples),
        "gate_development_examples": len(gate_examples),
        "fresh_validation_examples": len(validation_examples),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "gate_parameters": sum(p.numel() for p in gate.parameters()),
        "train_first_ce": train_history[0] if train_history else None,
        "train_last_ce": train_history[-1] if train_history else None,
        "gate_best_bce": best_loss,
        "gate_labels": labels,
        "validation": validation,
    }


def run_vlb_runtime_and_amt_proof(
    model_id: str,
    vlb_dir: Path,
    token: Optional[str],
    train_examples: Sequence[Tuple[str, str]],
    gate_examples: Sequence[Tuple[str, str]],
    validation_examples: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "runtime_verified": False,
            "amt_verified": False,
            "reason": "CUDA is required for the first Gemma 4 VLB runtime proof",
        }
    device = torch.device("cuda")
    started = __import__("time").time()
    try:
        model, processor, manifest = load_vlb_model(model_id, Path(vlb_dir), token, device)
        # Deterministic text-only smoke executed from VLB-DIR.
        hidden, logits, _ = _forward_one(model, processor, "The capital of France is", " Paris", device)
        runtime_ok = bool(torch.isfinite(logits).all().item() and logits.shape[-1] > 1000 and hidden.shape[-1] > 100)
        smoke = {
            "finite_logits": bool(torch.isfinite(logits).all().item()),
            "vocab_size": int(logits.shape[-1]),
            "hidden_size": int(hidden.shape[-1]),
            "top1_token": int(logits.argmax(dim=-1).item()),
        }
        del hidden, logits
        amt = run_amt_proof(model, processor, device, train_examples, gate_examples, validation_examples) if runtime_ok else {
            "verified": False,
            "reason": "runtime smoke failed",
        }
        return {
            "runtime_verified": runtime_ok,
            "amt_verified": bool(amt.get("verified")),
            "runtime_loader": "VLBQuantLinear_Q8_G64",
            "upstream_weights_loaded": False,
            "runtime_smoke": smoke,
            "amt": amt,
            "elapsed_seconds": __import__("time").time() - started,
        }
    except Exception as exc:
        return {
            "runtime_verified": False,
            "amt_verified": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": __import__("time").time() - started,
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
