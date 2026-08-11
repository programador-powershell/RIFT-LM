"""Fused stage reference: F0 + Gate features + conditional F1."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from cascade.kernels.int4 import int4_linear
from cascade.kernels.lowrank import lowrank_linear


def fused_stage_linear(
    x: torch.Tensor,
    *,
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out_features: int,
    in_features: int,
    u: Optional[torch.Tensor],
    s: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    gate_mask: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Executa F0 e, se gate_mask, acumula F1 (por linha do batch).

    gate_mask: bool tensor (batch,) — True => aplica residual naquela linha.
    Se None, aplica F1 always.
    """
    y0 = int4_linear(
        x, codes, scales,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
    )
    if u is None or s is None or v is None:
        return {"y": y0, "f0": y0, "f1_applied": False, "f1_calls": 0}
    y1 = lowrank_linear(x, u, s, v)
    if gate_mask is None:
        return {"y": y0 + y1, "f0": y0, "f1_applied": True, "f1_calls": int(x.shape[0])}
    mask = gate_mask.to(dtype=y1.dtype).view(-1, 1)
    y = y0 + mask * y1
    f1_calls = int(gate_mask.to(torch.int64).sum().item())
    return {"y": y, "f0": y0, "f1_applied": True, "f1_calls": f1_calls}
