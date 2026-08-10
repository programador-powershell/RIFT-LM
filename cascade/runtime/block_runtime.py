"""CASCADE-C1 — executa Transformer block com Linears F0+Gate·F1 (sem peso original)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.compiler.block_decompose import BlockCascadePlan
from cascade.kernels.int4 import dequantize_int4
from cascade.kernels.lowrank import lowrank_linear
from cascade.runtime.confidence_gate import GateConfig, decide_gate


class CascadeLinearModule(nn.Module):
    """Substitui nn.Linear pelo path CASCADE.

    Caminho quente:
      Y = F0(X) + Gate(X) · F1(X)
    onde F0 vem de INT4 (codes+scales) e F1 de fatores low-rank.
    O peso denso original NÃO é armazenado nem reconstruído aqui.
    """

    def __init__(self, stages, *, gate_percentile: float = 70.0, path: str = "F0_GATE_F1"):
        super().__init__()
        self.path = path
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        # F0 + F1 only — nunca o W original
        self.register_buffer("codes", stages.codes)
        self.register_buffer("scales", stages.scales)
        self.register_buffer("u", stages.u)
        self.register_buffer("s", stages.s)
        self.register_buffer("v", stages.v)
        self.group_size = int(stages.group_size)
        self.out_features = int(stages.out_features)
        self.in_features = int(stages.in_features)
        self.bias = None
        self.last_gate_rate = 1.0
        self.f1_calls = 0
        self.f0_calls = 0
        self.f1_skip_calls = 0
        # cache F0 dequant no device sob demanda (ainda é F0, não W original)
        self._w0_cache: Optional[torch.Tensor] = None

    def _w0(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._w0_cache is None or self._w0_cache.device != device:
            w0 = dequantize_int4(
                self.codes, self.scales,
                group_size=self.group_size,
                out_features=self.out_features,
                in_features=self.in_features,
            )
            self._w0_cache = w0.to(device=device, dtype=torch.float32)
        return self._w0_cache.to(dtype=dtype) if dtype != torch.float32 else self._w0_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        dtype = torch.float32
        x2f = x2.to(dtype=dtype)
        w0 = self._w0(x2f.device, dtype)
        y0 = F.linear(x2f, w0)
        n = int(x2f.shape[0])
        self.f0_calls += n

        if self.path == "F0_ONLY":
            self.last_gate_rate = 0.0
            self.f1_skip_calls += n
            out = y0
        else:
            y1 = lowrank_linear(
                x2f,
                self.u.to(device=x2f.device, dtype=dtype),
                self.s.to(device=x2f.device, dtype=dtype),
                self.v.to(device=x2f.device, dtype=dtype),
            )
            if self.path == "F0_PLUS_F1_ALWAYS":
                self.f1_calls += n
                self.last_gate_rate = 1.0
                out = y0 + y1
            else:
                mask, meta = decide_gate(x2f, self.gate_cfg)
                self.last_gate_rate = float(meta["activation_rate"])
                applied = int(mask.sum().item())
                self.f1_calls += applied
                self.f1_skip_calls += n - applied
                out = y0 + mask.to(dtype=y1.dtype).unsqueeze(1) * y1

        # volta ao dtype de entrada quando possível
        if x.dtype != out.dtype:
            out = out.to(dtype=x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def stats(self) -> Dict[str, Any]:
        total = max(self.f0_calls, 1)
        return {
            "f0_calls": self.f0_calls,
            "f1_calls": self.f1_calls,
            "f1_skip_rate": 1.0 - (self.f1_calls / total),
            "last_gate_rate": self.last_gate_rate,
            "resident_bytes": int(
                self.codes.numel()
                + self.scales.numel() * 4
                + self.u.numel() * 4
                + self.s.numel() * 4
                + self.v.numel() * 4
            ),
        }


def _set_module(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parts = dotted.split(".") if dotted else []
    if not parts:
        raise ValueError("empty module path")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_mod
    else:
        setattr(parent, leaf, new_mod)


def collect_block_linears(block: nn.Module, block_name: str) -> Dict[str, nn.Linear]:
    out: Dict[str, nn.Linear] = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            full = f"{block_name}.{name}" if name else block_name
            out[full] = mod
    return out


def patch_block_linears(
    block: nn.Module,
    plan: BlockCascadePlan,
    *,
    gate_percentile: float = 70.0,
    path: str = "F0_GATE_F1",
    device: Optional[torch.device] = None,
) -> Dict[str, CascadeLinearModule]:
    """Troca in-place as nn.Linear do block por CascadeLinearModule (sem W original)."""
    replaced: Dict[str, CascadeLinearModule] = {}
    for plan_name, stages in plan.linears.items():
        short = plan_name
        if plan_name.startswith(plan.block_name + "."):
            short = plan_name[len(plan.block_name) + 1 :]
        try:
            casc = CascadeLinearModule(stages, gate_percentile=gate_percentile, path=path)
            if device is not None:
                casc = casc.to(device)
            _set_module(block, short, casc)
            replaced[plan_name] = casc
        except Exception as exc:
            print(f"[patch] AVISO {plan_name}: {exc}")
    return replaced


def restore_block_linears(block: nn.Module, originals: Dict[str, nn.Linear], block_name: str) -> None:
    for full, linear in originals.items():
        short = full[len(block_name) + 1 :] if full.startswith(block_name + ".") else full
        try:
            _set_module(block, short, linear)
        except Exception:
            pass
