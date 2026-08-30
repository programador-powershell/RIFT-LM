#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLB lazy runtime + teacher-free AMT proof v2.

This module loads a VLB-DIR artifact without loading upstream model weights.
Q8 nn.Linear weights stay compressed and are materialized only for the active
linear call. The VLB base is immutable during AMT.

AMT v2 follows the VBL RC-LR/SCM contract more closely than v1:
- no vocabulary-sized trainable head;
- low-rank residual adaptation happens in latent/hidden state;
- two residual refinement depths (R3/R4) share one adapter;
- a marginal-gain gate is trained from empirical validation gain;
- teacher/KL/distillation/textual self-confidence are forbidden.

The first proof target is google/gemma-4-E4B-it via
AutoModelForMultimodalLM/Gemma4ForConditionalGeneration.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor


RESIDUAL_RANK = 16
RESIDUAL_GATE_BIAS = 4.0
MARGINAL_GATE_HIDDEN = 24


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
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        out = F.linear(x, weight, bias)
        del weight
        return out


class AMTResidualSCM(nn.Module):
    """State-conditioned low-rank residual refinement shared by R3/R4."""

    def __init__(self, hidden_size: int, rank: int = RESIDUAL_RANK, residual_steps: int = 2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.step_embeddings = nn.Parameter(torch.zeros(residual_steps, rank))
        self.up = nn.Linear(rank, hidden_size * 2, bias=False)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.step_embeddings, mean=0.0, std=0.01)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01 / math.sqrt(rank))

    def forward(self, state: torch.Tensor, step_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.silu(self.down(self.norm(state)) + self.step_embeddings[step_index])
        injection, gate_delta = self.up(z).chunk(2, dim=-1)
        candidate = state + injection
        copy_gate = torch.sigmoid(gate_delta + RESIDUAL_GATE_BIAS)
        refined = copy_gate * state + (1.0 - copy_gate) * candidate
        return refined, refined - state


class AMTMarginalGate(nn.Module):
    """Hidden-state marginal gain gate: 2H + entropy + margin + normalized depth."""

    def __init__(self, hidden_size: int, hidden: int = MARGINAL_GATE_HIDDEN):
        super().__init__()
        self.input_dim = hidden_size * 2 + 3
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _load_payload(root: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    return torch.load(root / record["file"], map_location="cpu", weights_only=False)


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
    architecture = (getattr(config, "architectures", None) or [""])[0]
    if model_id == "google/gemma-4-E4B-it" and architecture != "Gemma4ForConditionalGeneration":
        raise RuntimeError(f"Unexpected Gemma 4 architecture: {architecture!r}")

    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(config)

    # Replace Q8 Linear weights before ordinary tensor materialization.
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

    # Materialize every remaining checkpoint tensor from VLB-DIR only.
    for name, record in records.items():
        if name in replaced:
            continue
        payload = _load_payload(root, record)
        value = _dequant_payload(payload)
        try:
            set_module_tensor_to_device(model, name, "cpu", value=value)
        except Exception as exc:
            raise RuntimeError(f"VLB runtime cannot bind tensor {name}: {exc}") from exc
        del payload, value

    model.tie_weights()

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
    # Gemma E4B is ~16 GB upstream. Q8 Linear buffers + conservative FP16
    # passthrough fit the first T4 proof while each Linear is dequantized only
    # for its active call.
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
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        inputs = tokenizer(text, return_tensors="pt")
    else:
        try:
            inputs = processor(text=text, return_tensors="pt")
        except TypeError:
            inputs = processor(text, return_tensors="pt")
    result = {}
    # Text-only proof: never create/move image/audio tensors implicitly.
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = inputs.get(key) if hasattr(inputs, "get") else None
        if torch.is_tensor(value):
            result[key] = value.to(device)
    if "input_ids" not in result:
        raise RuntimeError("Processor/tokenizer did not produce input_ids for text-only Gemma proof")
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


def _output_weight(model: nn.Module, device: torch.device) -> torch.Tensor:
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("Model does not expose a tied/output embedding weight for latent AMT projection")
    weight = head.weight
    if getattr(weight, "is_meta", False):
        raise RuntimeError("Output embedding weight is still meta")
    return weight.to(device)


def _delta_logits(model: nn.Module, delta_hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    weight = _output_weight(model, device)
    return F.linear(delta_hidden.to(weight.dtype), weight).float()


def _refine_states(adapter: AMTResidualSCM, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r2 = hidden
    r3, _ = adapter(r2, 0)
    r4, _ = adapter(r3, 1)
    return r2, r3, r4


def _logits_for_state(model: nn.Module, base_logits: torch.Tensor, r2: torch.Tensor, state: torch.Tensor, device: torch.device) -> torch.Tensor:
    if state.data_ptr() == r2.data_ptr():
        return base_logits
    return base_logits + _delta_logits(model, state - r2, device)


def _entropy_margin(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return entropy, top2[:, 0] - top2[:, 1]


def _gate_features(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    current_logits: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    entropy, margin = _entropy_margin(current_logits)
    current_pool = current_state.float()
    delta_pool = (current_state - previous_state).float()
    depth_value = torch.full(
        (current_pool.shape[0], 1),
        float(depth) / 4.0,
        dtype=current_pool.dtype,
        device=current_pool.device,
    )
    return torch.cat(
        [current_pool, delta_pool, entropy[:, None], margin[:, None], depth_value],
        dim=-1,
    )


def _evaluate_examples(
    model: nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    gate: AMTMarginalGate,
    examples: Sequence[Tuple[str, str]],
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    base_loss = adaptive_loss = 0.0
    base_correct = adaptive_correct = 0
    depth_sum = 0
    decisions = []
    adapter.eval()
    gate.eval()

    for prompt, target in examples:
        hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
        with torch.no_grad():
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits, r2, r3, device)
            logits4 = _logits_for_state(model, base_logits, r2, r4, device)

            f2 = _gate_features(r2, r2, base_logits, 2)
            p3 = torch.sigmoid(gate(f2))
            go3 = bool((p3 >= threshold).item())
            chosen = base_logits
            depth = 2
            p4_value = None
            if go3:
                chosen = logits3
                depth = 3
                f3 = _gate_features(r3, r2, logits3, 3)
                p4 = torch.sigmoid(gate(f3))
                p4_value = float(p4.item())
                if bool((p4 >= threshold).item()):
                    chosen = logits4
                    depth = 4

            target_tensor = torch.tensor([target_id], device=device)
            base_ce = F.cross_entropy(base_logits, target_tensor)
            adaptive_ce = F.cross_entropy(chosen, target_tensor)

        base_loss += float(base_ce.item())
        adaptive_loss += float(adaptive_ce.item())
        base_correct += int(base_logits.argmax(dim=-1).item() == target_id)
        adaptive_correct += int(chosen.argmax(dim=-1).item() == target_id)
        depth_sum += depth
        decisions.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "target_id": target_id,
                "p_continue_r3": float(p3.item()),
                "p_continue_r4": p4_value,
                "depth": depth,
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
        "mean_depth": depth_sum / n,
        "decisions": decisions,
    }


def _empirical_gate_dataset(
    model: nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    examples: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    xs, ys, ledger = [], [], []
    adapter.eval()
    for prompt, target in examples:
        hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
        target_tensor = torch.tensor([target_id], device=device)
        with torch.no_grad():
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits, r2, r3, device)
            logits4 = _logits_for_state(model, base_logits, r2, r4, device)
            states = [(2, r2, r2, base_logits, logits3), (3, r3, r2, logits3, logits4)]
            for depth, current, previous, current_logits, next_logits in states:
                current_ce = F.cross_entropy(current_logits, target_tensor)
                next_ce = F.cross_entropy(next_logits, target_tensor)
                current_ok = int(current_logits.argmax(dim=-1).item() == target_id)
                next_ok = int(next_logits.argmax(dim=-1).item() == target_id)
                label = 1.0 if (next_ok > current_ok or (next_ok == current_ok and next_ce < current_ce)) else 0.0
                xs.append(_gate_features(current, previous, current_logits, depth).squeeze(0).cpu())
                ys.append(label)
                ledger.append(
                    {
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "depth": depth,
                        "current_ce": float(current_ce.item()),
                        "next_ce": float(next_ce.item()),
                        "current_correct": current_ok,
                        "next_correct": next_ok,
                        "label": "CONTINUE" if label else "STOP",
                    }
                )
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), ledger


def _select_threshold(
    model: nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    gate: AMTMarginalGate,
    examples: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    best = None
    for threshold in [x / 100.0 for x in range(50, 100, 5)]:
        result = _evaluate_examples(model, processor, adapter, gate, examples, device, threshold)
        safe = (
            result["adaptive_correct"] >= result["base_correct"]
            and result["adaptive_mean_ce"] <= result["base_mean_ce"] + 0.001
        )
        candidate = (safe, result["adaptive_correct"], -result["adaptive_mean_ce"], -result["mean_depth"], threshold, result)
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    assert best is not None
    return float(best[4]), {"safe": bool(best[0]), **best[5]}


def run_amt_proof(
    model: nn.Module,
    processor: Any,
    device: torch.device,
    train_examples: Sequence[Tuple[str, str]],
    gate_examples: Sequence[Tuple[str, str]],
    validation_examples: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    hidden0, _, _ = _forward_one(model, processor, train_examples[0][0], train_examples[0][1], device)
    hidden_size = int(hidden0.shape[-1])
    del hidden0

    adapter = AMTResidualSCM(hidden_size, rank=RESIDUAL_RANK).to(device)
    gate = AMTMarginalGate(hidden_size, hidden=MARGINAL_GATE_HIDDEN).to(device)

    for p in model.parameters():
        p.requires_grad = False

    # AMT = minimum sufficient exposure. Short bursts + rollback to best state.
    opt = torch.optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=0.02, betas=(0.9, 0.95))
    best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    best_metric = None
    history = []
    consecutive_rejects = 0

    for round_index in range(1, 7):
        adapter.train()
        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for prompt, target in train_examples:
            hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits.detach(), r2, r3, device)
            logits4 = _logits_for_state(model, base_logits.detach(), r2, r4, device)
            target_tensor = torch.tensor([target_id], device=device)
            loss3 = F.cross_entropy(logits3, target_tensor)
            loss4 = F.cross_entropy(logits4, target_tensor)
            loss = 0.65 * loss3 + 0.35 * loss4
            (loss / len(train_examples)).backward()
            train_loss += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()

        # Gate-development split is used only to accept/rollback AMT here;
        # fresh validation remains untouched until the end.
        gate_tmp = AMTMarginalGate(hidden_size).to(device)
        gate_tmp.load_state_dict(gate.state_dict())
        metric = 0.0
        adapter.eval()
        for prompt, target in gate_examples:
            hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
            with torch.no_grad():
                r2, r3, r4 = _refine_states(adapter, hidden)
                logits3 = _logits_for_state(model, base_logits, r2, r3, device)
                logits4 = _logits_for_state(model, base_logits, r2, r4, device)
                target_tensor = torch.tensor([target_id], device=device)
                best_ce = min(
                    float(F.cross_entropy(base_logits, target_tensor).item()),
                    float(F.cross_entropy(logits3, target_tensor).item()),
                    float(F.cross_entropy(logits4, target_tensor).item()),
                )
                metric += best_ce
        metric /= max(1, len(gate_examples))
        accepted = best_metric is None or metric <= best_metric + 1e-6
        history.append({"round": round_index, "train_ce": train_loss / len(train_examples), "gate_oracle_ce": metric, "accepted": accepted})
        if accepted:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            consecutive_rejects = 0
        else:
            adapter.load_state_dict(best_state, strict=True)
            consecutive_rejects += 1
            if consecutive_rejects >= 2:
                break

    adapter.load_state_dict(best_state, strict=True)

    x_cpu, y_cpu, labels = _empirical_gate_dataset(model, processor, adapter, gate_examples, device)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    gate_opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=0.0)
    best_bce = float("inf")
    best_gate_state = None
    for _ in range(180):
        gate.train()
        gate_opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(gate(x), y)
        loss.backward()
        gate_opt.step()
        value = float(loss.detach().item())
        if value < best_bce:
            best_bce = value
            best_gate_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    if best_gate_state is not None:
        gate.load_state_dict(best_gate_state, strict=True)

    threshold, calibration = _select_threshold(model, processor, adapter, gate, gate_examples, device)
    validation = _evaluate_examples(model, processor, adapter, gate, validation_examples, device, threshold)
    pass_amt = (
        validation["adaptive_correct"] >= validation["base_correct"]
        and validation["adaptive_mean_ce"] <= validation["base_mean_ce"] + 1e-6
    )

    return {
        "verified": bool(pass_amt),
        "teacher": "NONE",
        "kl": False,
        "distillation": False,
        "textual_self_confidence": False,
        "training_examples": len(train_examples),
        "gate_development_examples": len(gate_examples),
        "fresh_validation_examples": len(validation_examples),
        "residual_parameters": sum(p.numel() for p in adapter.parameters()),
        "gate_parameters": sum(p.numel() for p in gate.parameters()),
        "physical_parameter_fraction_of_4b_pct": 100.0 * (sum(p.numel() for p in adapter.parameters()) + sum(p.numel() for p in gate.parameters())) / 4_000_000_000,
        "amt_history": history,
        "gate_best_bce": best_bce,
        "gate_labels": labels,
        "threshold": threshold,
        "calibration": calibration,
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
    started = time.time()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        model, processor, manifest = load_vlb_model(model_id, Path(vlb_dir), token, device)
        hidden, logits, _ = _forward_one(model, processor, "The capital of France is", " Paris", device)
        runtime_ok = bool(
            torch.isfinite(logits).all().item()
            and logits.shape[-1] == 262144 if model_id == "google/gemma-4-E4B-it" else logits.shape[-1] > 1000
        )
        smoke = {
            "finite_logits": bool(torch.isfinite(logits).all().item()),
            "vocab_size": int(logits.shape[-1]),
            "hidden_size": int(hidden.shape[-1]),
            "top1_token": int(logits.argmax(dim=-1).item()),
            "model_architecture": (getattr(model.config, "architectures", None) or [None])[0],
            "source_checkpoint_materialized": bool(manifest.get("streaming", {}).get("source_checkpoint_materialized", True)),
        }
        del hidden, logits

        amt = (
            run_amt_proof(model, processor, device, train_examples, gate_examples, validation_examples)
            if runtime_ok
            else {"verified": False, "reason": "runtime smoke failed"}
        )
        return {
            "runtime_verified": runtime_ok,
            "amt_verified": bool(amt.get("verified")),
            "runtime_loader": "VLBQuantLinear_Q8_G64_LATENT_SCM_AMT_V2",
            "upstream_weights_loaded": False,
            "runtime_smoke": smoke,
            "amt": amt,
            "elapsed_seconds": time.time() - started,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "runtime_verified": False,
            "amt_verified": False,
            "error": f"CUDAOutOfMemory: {exc}",
            "diagnostic": "VLB artifact conversion passed but T4 residency failed; do not fall back to upstream runtime.",
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "runtime_verified": False,
            "amt_verified": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - started,
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
