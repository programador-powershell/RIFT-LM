"""Decomposição real C0: W ≈ W0_INT4 + U·diag(S)·Vᵀ."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

from cascade.kernels.int4 import dequantize_int4, quantize_int4_group
from cascade.kernels.lowrank import fit_lowrank_residual


@dataclass
class CascadeLinearStages:
    out_features: int
    in_features: int
    group_size: int
    codes: torch.Tensor          # INT4 packed
    scales: torch.Tensor
    u: torch.Tensor              # out × r
    s: torch.Tensor              # r
    v: torch.Tensor              # in × r
    rank: int
    f0_bytes: int
    f1_bytes: int
    baseline_bytes: int

    def to_meta(self) -> Dict[str, Any]:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "group_size": self.group_size,
            "rank": self.rank,
            "f0_bytes": self.f0_bytes,
            "f1_bytes": self.f1_bytes,
            "baseline_bytes": self.baseline_bytes,
            "disk_reduction_pct": 100.0 * (1.0 - (self.f0_bytes + self.f1_bytes) / max(self.baseline_bytes, 1)),
        }


def decompose_linear_int4_lowrank(
    weight: torch.Tensor,
    *,
    rank: int = 16,
    group_size: int = 32,
) -> CascadeLinearStages:
    """Primeira decomposição real do plano C0.

    W0 = INT4 group-quantized
    R1 = W - dequant(W0) ≈ U S Vᵀ
    """
    if weight.ndim != 2:
        raise ValueError("weight 2D requerido")
    w = weight.detach().to(dtype=torch.float32).cpu().contiguous()
    out_f, in_f = w.shape
    codes, scales, gs = quantize_int4_group(w, group_size=group_size)
    w0 = dequantize_int4(codes, scales, group_size=gs, out_features=out_f, in_features=in_f)
    residual = w - w0
    u, s, v = fit_lowrank_residual(residual, rank=rank)
    f0_bytes = int(codes.numel() + scales.numel() * 4)
    f1_bytes = int((u.numel() + s.numel() + v.numel()) * 4)
    baseline = int(w.numel() * 4)
    return CascadeLinearStages(
        out_features=out_f,
        in_features=in_f,
        group_size=gs,
        codes=codes,
        scales=scales,
        u=u,
        s=s,
        v=v,
        rank=int(s.numel()),
        f0_bytes=f0_bytes,
        f1_bytes=f1_bytes,
        baseline_bytes=baseline,
    )
