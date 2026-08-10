"""CASCADE-C1 — decompõe todas as Linears de um Transformer block."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cascade.compiler.decompose import CascadeLinearStages, decompose_linear_int4_lowrank


@dataclass
class BlockCascadePlan:
    block_name: str
    linears: Dict[str, CascadeLinearStages] = field(default_factory=dict)
    total_baseline_bytes: int = 0
    total_f0_bytes: int = 0
    total_f1_bytes: int = 0

    def to_meta(self) -> Dict[str, Any]:
        return {
            "block_name": self.block_name,
            "n_linears": len(self.linears),
            "total_baseline_bytes": self.total_baseline_bytes,
            "total_f0_bytes": self.total_f0_bytes,
            "total_f1_bytes": self.total_f1_bytes,
            "disk_reduction_pct": 100.0 * (
                1.0 - (self.total_f0_bytes + self.total_f1_bytes) / max(self.total_baseline_bytes, 1)
            ),
            "layers": {k: v.to_meta() for k, v in self.linears.items()},
        }


def find_transformer_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Localiza lista de blocks (layers / h / decoder layers).

    Cobre Qwen, Llama, Phi, Gemma, GPT-NeoX e fallback genérico.
    """
    candidates_attrs = (
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "transformer.layers",
        "model.decoder.layers",
        "gpt_neox.layers",
        "language_model.model.layers",
        "model.language_model.layers",
    )
    for attr in candidates_attrs:
        mod = model
        ok = True
        for part in attr.split("."):
            if not hasattr(mod, part):
                ok = False
                break
            mod = getattr(mod, part)
        if ok and isinstance(mod, (nn.ModuleList, list)) and len(mod) > 0:
            return [(f"{attr}.{i}", mod[i]) for i in range(len(mod))]
    # fallback: ModuleList cujo nome termina com layers / h e tem Linear dentro
    found = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            continue
        leaf = name.split(".")[-1]
        if leaf not in ("layers", "h", "blocks", "layer"):
            continue
        # confirma que o primeiro item tem Linear
        has_linear = any(isinstance(m, nn.Linear) for m in mod[0].modules())
        if has_linear:
            return [(f"{name}.{i}", mod[i]) for i in range(len(mod))]
        found.append(name)
    return []


def collect_linears(block: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    out: Dict[str, nn.Linear] = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            full = f"{prefix}.{name}" if prefix and name else (prefix or name)
            out[full if full else name] = mod
    return out


def decompose_block(
    block: nn.Module,
    *,
    block_name: str,
    rank: int = 16,
    group_size: int = 32,
) -> BlockCascadePlan:
    plan = BlockCascadePlan(block_name=block_name)
    for name, linear in collect_linears(block, prefix=block_name).items():
        w = linear.weight.detach().float().cpu()
        stages = decompose_linear_int4_lowrank(w, rank=rank, group_size=group_size)
        plan.linears[name] = stages
        plan.total_baseline_bytes += stages.baseline_bytes
        plan.total_f0_bytes += stages.f0_bytes
        plan.total_f1_bytes += stages.f1_bytes
    if not plan.linears:
        raise RuntimeError(f"Nenhuma nn.Linear em {block_name}")
    return plan
