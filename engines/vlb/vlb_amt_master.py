#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLB AMT Master v3 — persistent, teacher-free mastery scheduler.

This module is deliberately a REFERENCE-SIDE AMT implementation until the same
operators execute inside the VLB native model runtime. It advances the VLB AMT
contract substantially beyond the earlier smoke proof:

- immutable VLB base model;
- capability registry;
- empirical R2/R3/R4 diagnosis;
- MASTERED / COMPUTE_LIMITED / LEARNING / STALLED / REGRESSED states;
- minimum-sufficient exposure;
- targeted AMT bursts;
- protected-capability regression checks;
- COMMIT / ROLLBACK / early stop;
- marginal-gain STOP/CONTINUE gate;
- separate LEARN / AMT_VALIDATE / GATE_TRAIN / GATE_CALIBRATE / FINAL_VALIDATE roles;
- persistent adapter/gate checkpoint and append-only ledgers;
- no teacher, no KL, no distillation, no textual self-confidence.

The probe corpus below is NOT a KR100 certification corpus. It is only the
teacher-free AMT development harness used to exercise the learning machinery.
Final model retention must still be established by a separate frozen KR100
contract after the complete VLB-native forward exists.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from vlb_runtime import (
    AMTMarginalGate,
    AMTResidualSCM,
    _forward_one,
    _gate_features,
    _logits_for_state,
    _refine_states,
    load_vlb_model,
)

AMT_MASTER_VERSION = "VLB_AMT_MASTER_V3"
SEED = 2026083019
RESIDUAL_LR = 2e-4
GATE_LR = 1e-3
MAX_ROUNDS = 12
UPDATES_PER_ROUND = 4
MAX_CONSECUTIVE_REJECTS = 3
CE_IMPROVEMENT_EPS = 1e-4
PROTECTED_CE_SLACK = 1e-4
GATE_EPOCHS = 240
GATE_PATIENCE = 35
THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(20)]

# --------------------------------------------------------------------------------------
# Teacher-free GOLD probe corpus.
# Each item is intentionally short because the current bridge evaluates the first
# target token. This is a mastery-development harness, not a claim about broad model
# knowledge. Capabilities have independent examples per role to avoid using the same
# item for learning and acceptance.
# --------------------------------------------------------------------------------------

CAPABILITY_CORPUS: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "arithmetic": {
        "learn": [
            ("2 + 2 =", " 4"),
            ("8 - 3 =", " 5"),
            ("6 + 7 =", " 13"),
            ("3 * 4 =", " 12"),
        ],
        "amt_validate": [
            ("5 + 6 =", " 11"),
            ("9 - 2 =", " 7"),
        ],
        "gate_train": [
            ("7 + 5 =", " 12"),
            ("10 - 4 =", " 6"),
        ],
        "gate_calibrate": [
            ("4 + 9 =", " 13"),
            ("12 - 5 =", " 7"),
        ],
        "final_validate": [
            ("3 + 8 =", " 11"),
            ("14 - 6 =", " 8"),
        ],
    },
    "binary": {
        "learn": [
            ("Binary 10 equals decimal", " 2"),
            ("Binary 11 equals decimal", " 3"),
            ("Binary 100 equals decimal", " 4"),
            ("Binary 101 equals decimal", " 5"),
        ],
        "amt_validate": [
            ("Binary 110 equals decimal", " 6"),
            ("Binary 111 equals decimal", " 7"),
        ],
        "gate_train": [
            ("Binary 1000 equals decimal", " 8"),
            ("Binary 1001 equals decimal", " 9"),
        ],
        "gate_calibrate": [
            ("Binary 1010 equals decimal", " 10"),
            ("Binary 1011 equals decimal", " 11"),
        ],
        "final_validate": [
            ("Binary 1100 equals decimal", " 12"),
            ("Binary 1101 equals decimal", " 13"),
        ],
    },
    "geography": {
        "learn": [
            ("The capital of France is", " Paris"),
            ("The capital of Brazil is", " Brasilia"),
            ("The capital of Italy is", " Rome"),
            ("The capital of Japan is", " Tokyo"),
        ],
        "amt_validate": [
            ("The capital of Germany is", " Berlin"),
            ("The capital of Spain is", " Madrid"),
        ],
        "gate_train": [
            ("The capital of Portugal is", " Lisbon"),
            ("The capital of Canada is", " Ottawa"),
        ],
        "gate_calibrate": [
            ("The capital of Greece is", " Athens"),
            ("The capital of Norway is", " Oslo"),
        ],
        "final_validate": [
            ("The capital of Sweden is", " Stockholm"),
            ("The capital of Austria is", " Vienna"),
        ],
    },
    "http": {
        "learn": [
            ("HTTP status for OK is", " 200"),
            ("HTTP status for Not Found is", " 404"),
            ("HTTP status for Created is", " 201"),
            ("HTTP status for Unauthorized is", " 401"),
        ],
        "amt_validate": [
            ("HTTP status for Forbidden is", " 403"),
            ("HTTP status for No Content is", " 204"),
        ],
        "gate_train": [
            ("HTTP status for Bad Request is", " 400"),
            ("HTTP status for Conflict is", " 409"),
        ],
        "gate_calibrate": [
            ("HTTP status for Too Many Requests is", " 429"),
            ("HTTP status for Internal Server Error is", " 500"),
        ],
        "final_validate": [
            ("HTTP status for Bad Gateway is", " 502"),
            ("HTTP status for Service Unavailable is", " 503"),
        ],
    },
    "python_semantics": {
        "learn": [
            ("In Python, a list is", " mutable"),
            ("In Python, a tuple is", " immutable"),
            ("In Python, a set stores", " unique"),
            ("In Python, None represents", " absence"),
        ],
        "amt_validate": [
            ("In Python, dictionary keys must be", " hashable"),
            ("In Python, len returns an", " integer"),
        ],
        "gate_train": [
            ("In Python, range is commonly used for", " iteration"),
            ("In Python, def introduces a", " function"),
        ],
        "gate_calibrate": [
            ("In Python, class introduces a", " class"),
            ("In Python, True is a", " boolean"),
        ],
        "final_validate": [
            ("In Python, False is a", " boolean"),
            ("In Python, import loads a", " module"),
        ],
    },
    "logic": {
        "learn": [
            ("The opposite of true is", " false"),
            ("The opposite of false is", " true"),
            ("If A is true and B is true, A AND B is", " true"),
            ("If A is true and B is false, A AND B is", " false"),
        ],
        "amt_validate": [
            ("If A is false and B is true, A OR B is", " true"),
            ("If A is false and B is false, A OR B is", " false"),
        ],
        "gate_train": [
            ("NOT true is", " false"),
            ("NOT false is", " true"),
        ],
        "gate_calibrate": [
            ("True XOR true is", " false"),
            ("True XOR false is", " true"),
        ],
        "final_validate": [
            ("False XOR true is", " true"),
            ("False XOR false is", " false"),
        ],
    },
    "units": {
        "learn": [
            ("1000 meters equals", " 1"),
            ("60 seconds equals", " 1"),
            ("24 hours equals", " 1"),
            ("100 centimeters equals", " 1"),
        ],
        "amt_validate": [
            ("1000 grams equals", " 1"),
            ("60 minutes equals", " 1"),
        ],
        "gate_train": [
            ("7 days equals", " 1"),
            ("12 months equals", " 1"),
        ],
        "gate_calibrate": [
            ("10 millimeters equals", " 1"),
            ("1000 milliliters equals", " 1"),
        ],
        "final_validate": [
            ("1000 kilograms equals", " 1"),
            ("1000 milliseconds equals", " 1"),
        ],
    },
    "code_tokens": {
        "learn": [
            ("Python equality operator is", " =="),
            ("Python assignment operator is", " ="),
            ("Python floor division operator is", " //"),
            ("Python exponent operator is", " **"),
        ],
        "amt_validate": [
            ("Python modulo operator is", " %"),
            ("Python inequality operator is", " !="),
        ],
        "gate_train": [
            ("Python less-than operator is", " <"),
            ("Python greater-than operator is", " >"),
        ],
        "gate_calibrate": [
            ("Python less-than-or-equal operator is", " <="),
            ("Python greater-than-or-equal operator is", " >="),
        ],
        "final_validate": [
            ("Python bitwise AND operator is", " &"),
            ("Python bitwise OR operator is", " |"),
        ],
    },
}


@dataclass
class ExampleMetric:
    capability: str
    prompt: str
    target: str
    target_id: int
    r2_ce: float
    r3_ce: float
    r4_ce: float
    r2_correct: int
    r3_correct: int
    r4_correct: int
    best_depth: int


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _clone_module_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _model_parameter_count(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _collect_examples(role: str, capabilities: Optional[Sequence[str]] = None) -> List[Tuple[str, str, str]]:
    allowed = set(capabilities or CAPABILITY_CORPUS.keys())
    rows: List[Tuple[str, str, str]] = []
    for capability, roles in CAPABILITY_CORPUS.items():
        if capability not in allowed:
            continue
        for prompt, target in roles[role]:
            rows.append((capability, prompt, target))
    return rows


def _metric_for_example(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    capability: str,
    prompt: str,
    target: str,
    device: torch.device,
) -> ExampleMetric:
    hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
    with torch.no_grad():
        r2, r3, r4 = _refine_states(adapter, hidden)
        logits3 = _logits_for_state(model, base_logits, r2, r3, device)
        logits4 = _logits_for_state(model, base_logits, r2, r4, device)
        target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
        ce2 = float(F.cross_entropy(base_logits, target_tensor).item())
        ce3 = float(F.cross_entropy(logits3, target_tensor).item())
        ce4 = float(F.cross_entropy(logits4, target_tensor).item())
        c2 = int(base_logits.argmax(dim=-1).item() == target_id)
        c3 = int(logits3.argmax(dim=-1).item() == target_id)
        c4 = int(logits4.argmax(dim=-1).item() == target_id)
    candidates = [(2, c2, ce2), (3, c3, ce3), (4, c4, ce4)]
    best_depth = min(candidates, key=lambda x: (-x[1], x[2], x[0]))[0]
    return ExampleMetric(
        capability=capability,
        prompt=prompt,
        target=target,
        target_id=target_id,
        r2_ce=ce2,
        r3_ce=ce3,
        r4_ce=ce4,
        r2_correct=c2,
        r3_correct=c3,
        r4_correct=c4,
        best_depth=best_depth,
    )


def _evaluate_role(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    role: str,
    device: torch.device,
    capabilities: Optional[Sequence[str]] = None,
) -> List[ExampleMetric]:
    adapter.eval()
    rows = []
    for capability, prompt, target in _collect_examples(role, capabilities):
        rows.append(_metric_for_example(model, processor, adapter, capability, prompt, target, device))
    return rows


def _aggregate_metrics(rows: Sequence[ExampleMetric]) -> Dict[str, Any]:
    if not rows:
        return {"examples": 0, "r2_correct": 0, "oracle_correct": 0, "r2_mean_ce": None, "oracle_mean_ce": None, "mean_best_depth": None}
    r2_correct = sum(row.r2_correct for row in rows)
    oracle_correct = sum(max(row.r2_correct, row.r3_correct, row.r4_correct) for row in rows)
    r2_ce = statistics.mean(row.r2_ce for row in rows)
    oracle_ce = statistics.mean(min(row.r2_ce, row.r3_ce, row.r4_ce) for row in rows)
    mean_depth = statistics.mean(row.best_depth for row in rows)
    return {
        "examples": len(rows),
        "r2_correct": r2_correct,
        "oracle_correct": oracle_correct,
        "r2_mean_ce": r2_ce,
        "oracle_mean_ce": oracle_ce,
        "mean_best_depth": mean_depth,
        "best_depth_distribution": dict(Counter(str(row.best_depth) for row in rows)),
    }


def _capability_registry(rows: Sequence[ExampleMetric]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[ExampleMetric]] = defaultdict(list)
    for row in rows:
        grouped[row.capability].append(row)
    registry: Dict[str, Dict[str, Any]] = {}
    for capability, items in grouped.items():
        total = len(items)
        r2_correct = sum(x.r2_correct for x in items)
        oracle_correct = sum(max(x.r2_correct, x.r3_correct, x.r4_correct) for x in items)
        r2_ce = statistics.mean(x.r2_ce for x in items)
        oracle_ce = statistics.mean(min(x.r2_ce, x.r3_ce, x.r4_ce) for x in items)
        deeper = sum(x.best_depth > 2 for x in items)
        compute_gain = oracle_correct > r2_correct or oracle_ce < r2_ce - CE_IMPROVEMENT_EPS
        if r2_correct == total:
            state = "MASTERED"
        elif compute_gain:
            state = "COMPUTE_LIMITED"
        else:
            state = "LEARNING"
        registry[capability] = {
            "capability": capability,
            "state": state,
            "examples": total,
            "r2_correct": r2_correct,
            "oracle_correct": oracle_correct,
            "r2_mean_ce": r2_ce,
            "oracle_mean_ce": oracle_ce,
            "deeper_fraction": deeper / max(total, 1),
            "minimum_sufficient_depth_mean": statistics.mean(x.best_depth for x in items),
            "exposures": 0,
            "commits": 0,
            "rollbacks": 0,
        }
    return registry


def _target_capabilities(registry: Dict[str, Dict[str, Any]]) -> List[str]:
    learning = [row for row in registry.values() if row["state"] == "LEARNING"]
    learning.sort(key=lambda row: (-float(row["r2_mean_ce"]), row["capability"]))
    return [row["capability"] for row in learning[:4]]


def _train_residual_burst(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    capabilities: Sequence[str],
    device: torch.device,
    round_index: int,
) -> Dict[str, Any]:
    examples = _collect_examples("learn", capabilities)
    if not examples:
        return {"updates": 0, "examples": 0, "target_tokens": 0, "mean_loss": None}
    for p in model.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = True
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=RESIDUAL_LR, weight_decay=0.02, betas=(0.9, 0.95))
    rng = random.Random(SEED + round_index * 997)
    losses: List[float] = []
    target_tokens = 0
    for _ in range(UPDATES_PER_ROUND):
        optimizer.zero_grad(set_to_none=True)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        numerator = 0.0
        for capability, prompt, target in shuffled:
            hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
            # Base forward is immutable. Gradients start at residual hidden state only.
            hidden = hidden.detach()
            base_logits = base_logits.detach()
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits, r2, r3, device)
            logits4 = _logits_for_state(model, base_logits, r2, r4, device)
            target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
            loss3 = F.cross_entropy(logits3, target_tensor)
            loss4 = F.cross_entropy(logits4, target_tensor)
            loss = 0.65 * loss3 + 0.35 * loss4
            (loss / len(shuffled)).backward()
            numerator += float(loss.detach().item())
            target_tokens += 1
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(numerator / len(shuffled))
    return {
        "updates": UPDATES_PER_ROUND,
        "examples": len(examples),
        "target_tokens": target_tokens,
        "mean_loss": statistics.mean(losses),
        "losses": losses,
    }


def _registry_from_eval(rows: Sequence[ExampleMetric], previous: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    current = _capability_registry(rows)
    if previous:
        for capability, row in current.items():
            old = previous.get(capability, {})
            row["exposures"] = int(old.get("exposures", 0))
            row["commits"] = int(old.get("commits", 0))
            row["rollbacks"] = int(old.get("rollbacks", 0))
    return current


def _protected_regressions(
    previous: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
    targets: Sequence[str],
) -> List[Dict[str, Any]]:
    regressions = []
    target_set = set(targets)
    for capability, before in previous.items():
        if capability in target_set:
            continue
        after = candidate.get(capability)
        if after is None:
            continue
        if before["state"] in {"MASTERED", "COMPUTE_LIMITED"}:
            if int(after["r2_correct"]) < int(before["r2_correct"]):
                regressions.append({"capability": capability, "reason": "R2_CORRECT_REGRESSION"})
                continue
            if float(after["oracle_mean_ce"]) > float(before["oracle_mean_ce"]) + PROTECTED_CE_SLACK:
                regressions.append({"capability": capability, "reason": "ORACLE_CE_REGRESSION"})
    return regressions


def _target_improved(
    previous: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
    targets: Sequence[str],
) -> bool:
    improved = False
    for capability in targets:
        before = previous.get(capability)
        after = candidate.get(capability)
        if before is None or after is None:
            continue
        if int(after["oracle_correct"]) > int(before["oracle_correct"]):
            improved = True
        elif int(after["oracle_correct"]) == int(before["oracle_correct"]) and float(after["oracle_mean_ce"]) < float(before["oracle_mean_ce"]) - CE_IMPROVEMENT_EPS:
            improved = True
    return improved


def _gate_dataset(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    examples: Sequence[Tuple[str, str, str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    features: List[torch.Tensor] = []
    labels: List[float] = []
    ledger: List[Dict[str, Any]] = []
    adapter.eval()
    for capability, prompt, target in examples:
        hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
        with torch.no_grad():
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits, r2, r3, device)
            logits4 = _logits_for_state(model, base_logits, r2, r4, device)
            target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
            ce2 = float(F.cross_entropy(base_logits, target_tensor).item())
            ce3 = float(F.cross_entropy(logits3, target_tensor).item())
            ce4 = float(F.cross_entropy(logits4, target_tensor).item())
            c2 = int(base_logits.argmax(dim=-1).item() == target_id)
            c3 = int(logits3.argmax(dim=-1).item() == target_id)
            c4 = int(logits4.argmax(dim=-1).item() == target_id)
            f2 = _gate_features(r2, r2, base_logits, 2).squeeze(0).detach().float().cpu()
            f3 = _gate_features(r3, r2, logits3, 3).squeeze(0).detach().float().cpu()
        continue23 = float(c3 > c2 or (c3 == c2 and ce3 < ce2 - CE_IMPROVEMENT_EPS))
        continue34 = float(c4 > c3 or (c4 == c3 and ce4 < ce3 - CE_IMPROVEMENT_EPS))
        for depth, feature, label, delta_correct, delta_ce in (
            (2, f2, continue23, c3 - c2, ce3 - ce2),
            (3, f3, continue34, c4 - c3, ce4 - ce3),
        ):
            features.append(feature)
            labels.append(label)
            ledger.append({
                "capability": capability,
                "prompt": prompt,
                "current_depth": depth,
                "next_depth": depth + 1,
                "label": "CONTINUE" if label else "STOP",
                "delta_correct": delta_correct,
                "delta_ce": delta_ce,
            })
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), ledger


def _train_gate(
    gate: AMTMarginalGate,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=0.0)
    positives = float(y.sum().item())
    negatives = float(y.numel() - positives)
    pos_weight = negatives / max(positives, 1.0)
    best = float("inf")
    best_state = _clone_module_state(gate)
    stale = 0
    history = []
    for epoch in range(1, GATE_EPOCHS + 1):
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        logits = gate(x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
        )
        loss.backward()
        optimizer.step()
        value = float(loss.detach().item())
        history.append({"epoch": epoch, "bce": value})
        if value < best - 1e-7:
            best = value
            best_state = _clone_module_state(gate)
            stale = 0
        else:
            stale += 1
        if stale >= GATE_PATIENCE:
            break
    gate.load_state_dict(best_state, strict=True)
    return {
        "epochs": len(history),
        "best_bce": best,
        "positives": int(positives),
        "negatives": int(negatives),
        "pos_weight": pos_weight,
        "history": history,
    }


def _adaptive_eval(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    gate: AMTMarginalGate,
    examples: Sequence[Tuple[str, str, str]],
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    base_correct = adaptive_correct = 0
    base_ce_sum = adaptive_ce_sum = 0.0
    depths = Counter()
    decisions = []
    adapter.eval()
    gate.eval()
    for capability, prompt, target in examples:
        hidden, base_logits, target_id = _forward_one(model, processor, prompt, target, device)
        with torch.no_grad():
            r2, r3, r4 = _refine_states(adapter, hidden)
            logits3 = _logits_for_state(model, base_logits, r2, r3, device)
            logits4 = _logits_for_state(model, base_logits, r2, r4, device)
            p23 = float(torch.sigmoid(gate(_gate_features(r2, r2, base_logits, 2))).item())
            chosen = base_logits
            depth = 2
            p34 = None
            if p23 >= threshold:
                chosen = logits3
                depth = 3
                p34 = float(torch.sigmoid(gate(_gate_features(r3, r2, logits3, 3))).item())
                if p34 >= threshold:
                    chosen = logits4
                    depth = 4
            target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
            base_ce = float(F.cross_entropy(base_logits, target_tensor).item())
            adaptive_ce = float(F.cross_entropy(chosen, target_tensor).item())
            base_hit = int(base_logits.argmax(dim=-1).item() == target_id)
            adaptive_hit = int(chosen.argmax(dim=-1).item() == target_id)
        base_correct += base_hit
        adaptive_correct += adaptive_hit
        base_ce_sum += base_ce
        adaptive_ce_sum += adaptive_ce
        depths[depth] += 1
        decisions.append({
            "capability": capability,
            "prompt": prompt,
            "target": target,
            "selected_depth": depth,
            "p_continue_r2": p23,
            "p_continue_r3": p34,
            "base_correct": base_hit,
            "adaptive_correct": adaptive_hit,
            "base_ce": base_ce,
            "adaptive_ce": adaptive_ce,
        })
    count = max(1, len(examples))
    return {
        "examples": len(examples),
        "threshold": threshold,
        "base_correct": base_correct,
        "adaptive_correct": adaptive_correct,
        "base_mean_ce": base_ce_sum / count,
        "adaptive_mean_ce": adaptive_ce_sum / count,
        "mean_depth": sum(depth * n for depth, n in depths.items()) / count,
        "depth_distribution": dict(Counter(str(k) for k, v in depths.items() for _ in range(v))),
        "decisions": decisions,
    }


def _calibrate_threshold(
    model: torch.nn.Module,
    processor: Any,
    adapter: AMTResidualSCM,
    gate: AMTMarginalGate,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    examples = _collect_examples("gate_calibrate")
    trials = []
    safe = []
    for threshold in THRESHOLDS:
        row = _adaptive_eval(model, processor, adapter, gate, examples, device, threshold)
        row["delta_correct"] = row["adaptive_correct"] - row["base_correct"]
        row["delta_ce"] = row["adaptive_mean_ce"] - row["base_mean_ce"]
        trials.append(row)
        if row["delta_correct"] >= 0 and row["delta_ce"] <= PROTECTED_CE_SLACK:
            safe.append(row)
    if safe:
        best = min(safe, key=lambda row: (-row["adaptive_correct"], row["adaptive_mean_ce"], row["mean_depth"]))
        mode = "SAFE_GAIN"
    else:
        best = min(trials, key=lambda row: (max(0, -row["delta_correct"]), max(0.0, row["delta_ce"]), row["mean_depth"]))
        mode = "SAFEST_FALLBACK"
    return float(best["threshold"]), {"mode": mode, "best": {k: v for k, v in best.items() if k != "decisions"}, "trials": [{k: v for k, v in row.items() if k != "decisions"} for row in trials]}


def _base_artifact_fingerprint(vlb_dir: Path) -> Dict[str, Optional[str]]:
    return {
        "vlb_manifest_sha256": _sha256_file(vlb_dir / "vlb_manifest.json"),
        "vlb1_sha256": _sha256_file(vlb_dir / "model.vlb"),
    }


def run_amt_master_reference(
    model_id: str,
    vlb_dir: Path,
    token: Optional[str],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the complete AMT Master v3 against an immutable VLB artifact.

    Returns a reference-side result. Native certification remains false until
    this same path runs inside the VLB native executor/server.
    """
    if not torch.cuda.is_available():
        return {
            "runtime_verified": False,
            "amt_verified": False,
            "classification": "REFERENCE_ONLY_NOT_NATIVE_CERTIFICATION",
            "reason": "CUDA required for current VLB AMT bridge",
        }

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    root = Path(vlb_dir)
    out = Path(output_dir or (root / "amt_master_v3"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_fingerprint_before = _base_artifact_fingerprint(root)

    model, processor, manifest = load_vlb_model(model_id, root, token, device)
    hidden, logits, _ = _forward_one(model, processor, "The capital of France is", " Paris", device)
    runtime_ok = bool(torch.isfinite(logits).all().item() and logits.shape[-1] > 1000)
    hidden_size = int(hidden.shape[-1])
    del hidden, logits

    adapter = AMTResidualSCM(hidden_size).to(device)
    gate = AMTMarginalGate(hidden_size).to(device)
    base_model_params = _model_parameter_count(model)
    residual_params = sum(int(p.numel()) for p in adapter.parameters())
    gate_params = sum(int(p.numel()) for p in gate.parameters())

    initial_eval = _evaluate_role(model, processor, adapter, "amt_validate", device)
    registry = _registry_from_eval(initial_eval)
    initial_summary = _aggregate_metrics(initial_eval)
    best_adapter = _clone_module_state(adapter)
    ledger: List[Dict[str, Any]] = []
    consecutive_rejects = 0
    total_target_tokens = 0

    for round_index in range(1, MAX_ROUNDS + 1):
        targets = _target_capabilities(registry)
        if not targets:
            ledger.append({"round": round_index, "decision": "EARLY_STOP", "reason": "NO_LEARNING_CAPABILITIES"})
            break

        before_registry = copy.deepcopy(registry)
        before_adapter = _clone_module_state(adapter)
        train = _train_residual_burst(model, processor, adapter, targets, device, round_index)
        total_target_tokens += int(train.get("target_tokens") or 0)
        candidate_eval = _evaluate_role(model, processor, adapter, "amt_validate", device)
        candidate_registry = _registry_from_eval(candidate_eval, before_registry)
        regressions = _protected_regressions(before_registry, candidate_registry, targets)
        improved = _target_improved(before_registry, candidate_registry, targets)
        accepted = bool(improved and not regressions)

        for capability in targets:
            if capability in candidate_registry:
                candidate_registry[capability]["exposures"] = int(before_registry.get(capability, {}).get("exposures", 0)) + int(train.get("target_tokens") or 0)

        if accepted:
            for capability in targets:
                if capability in candidate_registry:
                    candidate_registry[capability]["commits"] = int(before_registry.get(capability, {}).get("commits", 0)) + 1
            registry = candidate_registry
            best_adapter = _clone_module_state(adapter)
            consecutive_rejects = 0
            decision = "COMMIT"
            reason = "TARGET_GAIN_WITHOUT_PROTECTED_REGRESSION"
        else:
            adapter.load_state_dict(before_adapter, strict=True)
            for capability in targets:
                if capability in before_registry:
                    before_registry[capability]["rollbacks"] = int(before_registry[capability].get("rollbacks", 0)) + 1
                    if int(before_registry[capability]["rollbacks"]) >= 2 and before_registry[capability]["state"] == "LEARNING":
                        before_registry[capability]["state"] = "STALLED"
            registry = before_registry
            consecutive_rejects += 1
            decision = "ROLLBACK"
            reason = "PROTECTED_REGRESSION" if regressions else "NO_TARGET_GAIN"

        ledger.append({
            "round": round_index,
            "targets": targets,
            "training": train,
            "decision": decision,
            "reason": reason,
            "protected_regressions": regressions,
            "registry": copy.deepcopy(registry),
        })
        _write_json(out / "mastery_registry.json", registry)
        _write_jsonl(out / "amt_ledger.jsonl", ledger)

        if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
            ledger.append({"round": round_index, "decision": "EARLY_STOP", "reason": "MAX_CONSECUTIVE_REJECTS"})
            break

    adapter.load_state_dict(best_adapter, strict=True)
    final_amt_eval = _evaluate_role(model, processor, adapter, "amt_validate", device)
    final_registry = _registry_from_eval(final_amt_eval, registry)

    gate_x, gate_y, gate_labels = _gate_dataset(
        model,
        processor,
        adapter,
        _collect_examples("gate_train"),
        device,
    )
    gate_training = _train_gate(gate, gate_x, gate_y, device)
    threshold, calibration = _calibrate_threshold(model, processor, adapter, gate, device)
    final_validation = _adaptive_eval(
        model,
        processor,
        adapter,
        gate,
        _collect_examples("final_validate"),
        device,
        threshold,
    )

    amt_pass = bool(
        final_validation["adaptive_correct"] >= final_validation["base_correct"]
        and final_validation["adaptive_mean_ce"] <= final_validation["base_mean_ce"] + PROTECTED_CE_SLACK
    )

    checkpoint = {
        "schema": "VLB_AMT_MASTER_STATE_V3",
        "model_id": model_id,
        "classification": "REFERENCE_ONLY_NOT_NATIVE_CERTIFICATION",
        "teacher": "NONE",
        "kl": False,
        "distillation": False,
        "textual_self_confidence": False,
        "base_artifact_fingerprint": base_fingerprint_before,
        "adapter_state_dict": _clone_module_state(adapter),
        "gate_state_dict": _clone_module_state(gate),
        "gate_threshold": threshold,
        "mastery_registry": final_registry,
    }
    torch.save(checkpoint, out / "amt_master_state.pt")
    _write_json(out / "mastery_registry.json", final_registry)
    _write_jsonl(out / "gate_labels.jsonl", gate_labels)
    _write_jsonl(out / "final_validation.jsonl", final_validation["decisions"])
    _write_jsonl(out / "amt_ledger.jsonl", ledger)

    base_fingerprint_after = _base_artifact_fingerprint(root)
    base_immutable = base_fingerprint_after == base_fingerprint_before
    state_counts = dict(Counter(row["state"] for row in final_registry.values()))

    report = {
        "schema": "VLB_AMT_MASTER_REPORT_V3",
        "version": AMT_MASTER_VERSION,
        "model_id": model_id,
        "classification": "REFERENCE_ONLY_NOT_NATIVE_CERTIFICATION",
        "runtime_verified": runtime_ok,
        "amt_verified": bool(amt_pass and base_immutable),
        "teacher": "NONE",
        "kl": False,
        "distillation": False,
        "textual_self_confidence": False,
        "base_model_parameters": base_model_params,
        "residual_parameters": residual_params,
        "gate_parameters": gate_params,
        "added_parameter_fraction_pct": 100.0 * (residual_params + gate_params) / max(base_model_params, 1),
        "base_artifact_fingerprint_before": base_fingerprint_before,
        "base_artifact_fingerprint_after": base_fingerprint_after,
        "base_immutable": base_immutable,
        "initial_amt_validate": initial_summary,
        "final_amt_validate": _aggregate_metrics(final_amt_eval),
        "mastery_state_counts": state_counts,
        "mastery_registry": final_registry,
        "rounds": len([row for row in ledger if row.get("decision") in {"COMMIT", "ROLLBACK"}]),
        "commits": sum(1 for row in ledger if row.get("decision") == "COMMIT"),
        "rollbacks": sum(1 for row in ledger if row.get("decision") == "ROLLBACK"),
        "training_target_tokens": total_target_tokens,
        "gate_training": gate_training,
        "gate_label_counts": dict(Counter(row["label"] for row in gate_labels)),
        "gate_threshold": threshold,
        "gate_calibration": calibration,
        "final_validation": {k: v for k, v in final_validation.items() if k != "decisions"},
        "native_amt_certified": False,
        "kr100_certified": False,
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            "state": str(out / "amt_master_state.pt"),
            "registry": str(out / "mastery_registry.json"),
            "ledger": str(out / "amt_ledger.jsonl"),
            "gate_labels": str(out / "gate_labels.jsonl"),
            "final_validation": str(out / "final_validation.jsonl"),
        },
    }
    _write_json(out / "amt_master_report.json", report)

    del model, processor, adapter, gate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


__all__ = [
    "AMT_MASTER_VERSION",
    "CAPABILITY_CORPUS",
    "run_amt_master_reference",
]
