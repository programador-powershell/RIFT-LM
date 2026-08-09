"""Low-rank residual F1: Y += (X @ V) * S @ U.T  sem materializar U@V.T."""
from __future__ import annotations

from typing import Tuple

import torch


def fit_lowrank_residual(
    residual: torch.Tensor,
    *,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompõe residual ≈ U * diag(S) * V^T  (U: out×r, V: in×r)."""
    out_f, in_f = residual.shape
    r = min(int(rank), out_f, in_f)
    if r < 1:
        raise ValueError("rank deve ser >= 1")
    # pca_lowrank: residual ≈ U @ diag(S) @ V.T
    u, s, v = torch.pca_lowrank(residual, q=r, center=False, niter=4)
    return u.contiguous(), s.contiguous(), v.contiguous()


def lowrank_linear(
    x: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """F1(X) = (X @ V) * S @ U.T   — proibido materializar U·Vᵀ denso."""
    return ((x @ v) * s) @ u.T
