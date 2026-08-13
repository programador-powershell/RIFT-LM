"""Pack/unpack vetorizado do formato q4k v2 (144 bytes por super-bloco).

Layout por super-bloco de 256 colunas:
    d fp16 | dmin fp16 | 6B: 8 sub-escalas u6 | 6B: 8 sub-mins i6 (+31) | 128B q u4
    grupo g=32; byte k do grupo: val[k] = nibble baixo, val[k+16] = nibble alto

E o mesmo layout consumido por kernels/kernels.c (q4k_gemv*). O pack e todo
numpy vetorizado — um 30B empacota em segundos, sem loop Python por linha.
"""
from __future__ import annotations

import numpy as np
import torch

SUP = 256
GRP = 32
GPS = SUP // GRP
SUP_BYTES = 144


def _pack_u6(vals: np.ndarray) -> np.ndarray:
    shifts = (6 * np.arange(GPS, dtype=np.uint64)).reshape(1, 1, GPS)
    packed = (vals.astype(np.uint64) << shifts).sum(-1, dtype=np.uint64)
    return packed[..., None].view(np.uint8)[..., :6]


def _unpack_u6(raw6: np.ndarray) -> np.ndarray:
    buf = np.zeros(raw6.shape[:-1] + (8,), np.uint8)
    buf[..., :6] = raw6
    packed = buf.view(np.uint64)[..., 0]
    shifts = (6 * np.arange(GPS, dtype=np.uint64)).reshape(1, 1, GPS)
    return ((packed[..., None] >> shifts) & np.uint64(63)).astype(np.uint8)


def pack_q4k(planes: dict) -> np.ndarray:
    """planes (de cascade_codec_v2.encode_qk) -> (rows, nsup*144) uint8."""
    rows, cols = int(planes["rows"]), int(planes["cols"])
    if cols % SUP:
        raise ValueError(f"cols={cols} deve ser multiplo de {SUP}")
    nsup = cols // SUP
    q = planes["q"].numpy().astype(np.uint8).reshape(rows, nsup, GPS, GRP)
    sq = planes["sub_scales_u6"].numpy().astype(np.uint8)
    mq = planes["sub_mins_i6"].numpy().astype(np.int16)
    d = planes["sup_scale_f16"].numpy().astype(np.float16).reshape(rows, nsup)
    dm = planes["sup_min_f16"].numpy().astype(np.float16).reshape(rows, nsup)

    out = np.empty((rows, nsup, SUP_BYTES), np.uint8)
    out[:, :, 0:2] = d[..., None].view(np.uint8)
    out[:, :, 2:4] = dm[..., None].view(np.uint8)
    out[:, :, 4:10] = _pack_u6(sq)
    out[:, :, 10:16] = _pack_u6((mq + 31).astype(np.uint8))
    lo = q[..., :16]
    hi = q[..., 16:]
    out[:, :, 16:] = (lo | (hi << 4)).reshape(rows, nsup, 128)
    return np.ascontiguousarray(out.reshape(rows, nsup * SUP_BYTES))


def unpack_q4k(packed: np.ndarray, rows: int, cols: int) -> dict:
    """(rows, nsup*144) uint8 -> planes (tensores torch, como encode_qk)."""
    nsup = cols // SUP
    b = packed.reshape(rows, nsup, SUP_BYTES)
    d = b[:, :, 0:2].copy().view(np.float16).reshape(rows, nsup, 1)
    dm = b[:, :, 2:4].copy().view(np.float16).reshape(rows, nsup, 1)
    sq = _unpack_u6(b[:, :, 4:10])
    mq = _unpack_u6(b[:, :, 10:16]).astype(np.int8) - 31
    vals = b[:, :, 16:].reshape(rows, nsup, GPS, 16)
    q = np.empty((rows, nsup, GPS, GRP), np.uint8)
    q[..., :16] = vals & 0x0F
    q[..., 16:] = vals >> 4
    return {
        "q": torch.from_numpy(q.reshape(rows, nsup * GPS, GRP).copy()),
        "sub_scales_u6": torch.from_numpy(sq.copy()),
        "sub_mins_i6": torch.from_numpy(mq.copy()),
        "sup_scale_f16": torch.from_numpy(d.copy()).to(torch.float16),
        "sup_min_f16": torch.from_numpy(dm.copy()).to(torch.float16),
        "g": GRP, "bits": 4, "rows": rows, "cols": cols,
    }


def pad_cols_to_sup(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Padding de colunas ate multiplo de 256 (o kernel exige supers inteiros).
    Retorna (w_padded, cols_originais). x tambem deve ser padded com zeros —
    zeros nao alteram o produto."""
    cols = w.shape[1]
    pad = (-cols) % SUP
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w, cols
