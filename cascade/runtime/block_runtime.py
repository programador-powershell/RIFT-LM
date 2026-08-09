"""CASCADE-C1 — executa um Transformer block com Linears substituídas por F0+Gate·F1."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.compiler.block_decompose import BlockCascadePlan
from cascade.kernels.int4 import dequantize_int4
from cascade.kernels.lowrank import lowrank_linear
from cascade.runtime.confidence_gate import GateConfig, decide_gate


class CascadeLinearModule(nn.Module):
    """Substitui nn.Linear por path CASCADE (referência Python)."""

    def __init__(self, stages, *, gate_percentile: float = 70.0, path: str = "F0_GATE_F1"):
        super().__init__()
        self.path = path
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        self.register_buffer("codes", stages.codes)
        self.register_buffer("scales", stages.scales)
        self.register_buffer("u", stages.u)
        self.register_buffer("s", stages.s)
        self.register_buffer("v", stages.v)
        self.group_size = stages.group_size
        self.out_features = stages.out_features
        self.in_features = stages.in_features
        self.bias = None
        self.last_gate_rate = 1.0
        self.f1_calls = 0
        self.f0_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).float()
        w0 = dequantize_int4(
            self.codes, self.scales,
            group_size=self.group_size,
            out_features=self.out_features,
            in_features=self.in_features,
        ).to(device=x2.device, dtype=torch.float32)
        y0 = F.linear(x2, w0)
        self.f0_calls += int(x2.shape[0])
        if self.path == "F0_ONLY":
            self.last_gate_rate = 0.0
            return y0.reshape(*orig_shape[:-1], self.out_features)
        y1 = lowrank_linear(x2, self.u.to(x2.device), self.s.to(x2.device), self.v.to(x2.device))
        if self.path == "F0_PLUS_F1_ALWAYS":
            self.f1_calls += int(x2.shape[0])
            self.last_gate_rate = 1.0
            return (y0 + y1).reshape(*orig_shape[:-1], self.out_features)
        mask, meta = decide_gate(x2, self.gate_cfg)
        self.last_gate_rate = float(meta["activation_rate"])
        self.f1_calls += int(mask.sum().item())
        y = y0 + mask.to(y1.dtype).unsqueeze(1) * y1
        return y.reshape(*orig_shape[:-1], self.out_features)


def patch_block_linears(
    block: nn.Module,
    plan: BlockCascadePlan,
    *,
    gate_percentile: float = 70.0,
    path: str = "F0_GATE_F1",
) -> Dict[str, CascadeLinearModule]:
    """Troca in-place as nn.Linear do block pelos módulos CASCADE.

    Retorna mapa name→module para métricas.
    """
    replaced = {}
    linear_map = {name: mod for name, mod in block.named_modules() if isinstance(mod, nn.Linear)}
    # match plan keys which include block_name prefix
    for plan_name, stages in plan.linears.items():
        short = plan_name
        if plan_name.startswith(plan.block_name + "."):
            short = plan_name[len(plan.block_name) + 1 :]
        # find module by short name
        parent = block
        parts = short.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        if not hasattr(parent, leaf):
            continue
        casc = CascadeLinearModule(stages, gate_percentile=gate_percentile, path=path)
        setattr(parent, leaf, casc)
        replaced[plan_name] = casc
    return replaced


def unpatch_noop():
    pass
