"""INT4 group-quantized linear (F0 base)."""
from __future__ import annotations

from typing import Any, Tuple

import torch


def quantize_int4_group(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Quantiza W (out, in) em INT4 por grupos ao longo de `in`.

    Retorna:
        codes: int8 empacotado 2×INT4 por byte, shape (out, in//2) se in par
        scales: FP32 (out, n_groups)
        group_size
    """
    if weight.ndim != 2:
        raise ValueError("weight deve ser 2D (out, in)")
    out_f, in_f = weight.shape
    gs = int(group_size)
    if in_f % gs != 0:
        # pad in dimension for grouping
        pad = gs - (in_f % gs)
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_f = weight.shape[1]
    n_groups = in_f // gs
    w = weight.view(out_f, n_groups, gs)
    scales = w.abs().amax(dim=2).clamp_min(1e-8) / 7.0  # INT4 signed range ~[-7,7]
    q = torch.round(w / scales[:, :, None]).clamp(-8, 7).to(torch.int8)
    q = q.view(out_f, in_f)
    # pack two int4 into one uint8/int8
    if in_f % 2 != 0:
        q = torch.nn.functional.pad(q, (0, 1))
        in_f = q.shape[1]
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    codes = (lo | (hi << 4)).contiguous()
    return codes, scales.contiguous(), gs


def dequantize_int4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Reconstrói W aproximado a partir de INT4 group-quant (só para calibração/ref)."""
    gs = int(group_size)
    packed_in = codes.shape[1] * 2
    lo = (codes.to(torch.int16) & 0x0F).to(torch.int8)
    hi = ((codes.to(torch.int16) >> 4) & 0x0F).to(torch.int8)
    # sign-extend 4-bit two's complement stored in 0..15 for -8..7
    def sext(t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.int16)
        return torch.where(t >= 8, t - 16, t).to(torch.float32)

    q = torch.empty(codes.shape[0], packed_in, dtype=torch.float32, device=codes.device)
    q[:, 0::2] = sext(lo)
    q[:, 1::2] = sext(hi)
    q = q[:, :in_features]
    n_groups = (in_features + gs - 1) // gs
    # pad to group
    if q.shape[1] % gs != 0:
        q = torch.nn.functional.pad(q, (0, gs - (q.shape[1] % gs)))
    q = q.view(out_features, -1, gs)
    sc = scales
    if sc.shape[1] < q.shape[1]:
        sc = torch.nn.functional.pad(sc, (0, q.shape[1] - sc.shape[1]))
    w = (q * sc[:, : q.shape[1], None]).view(out_features, -1)[:, :in_features]
    return w


def int4_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """F0: Y = X @ W0^T  com W0 INT4 group-quant (path de referência)."""
    w0 = dequantize_int4(
        codes, scales, group_size=group_size, out_features=out_features, in_features=in_features
    )
    return torch.nn.functional.linear(x, w0)
