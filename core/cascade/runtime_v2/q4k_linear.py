"""Q4KLinearModule — substituto do CascadeLinearModule com as correcoes:

1. Caminho quente F0 via kernel C fundido (q4k_gemv_i8): dequant DENTRO do
   produto. Nunca materializa W denso — nem em cache (corrige o _w0_cache
   de 32 bpw residentes), nem por chamada (corrige o low_mem de 0.3 GB/s).
2. F1 low-rank via lowrank_gemv_f32 corrigido (Vt contiguo), aplicado so
   nas linhas que o gate CALIBRADO aprovar.
3. Gate v1: threshold fixo calibrado (GateCalibrator); sem threshold, F1
   fica desligado em batch pequeno (fail-safe F0_ONLY) em vez de disparar
   em 100% dos tokens de decode.
4. Prefill (batch>1): loop de linhas sobre o mesmo kernel — decode batch-1
   e o caminho que define tok/s.

Residente = bytes empacotados (4.5 bpw) + fatores F1. Sem cache fp32.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .confidence_gate import GateCalibrator, GateConfig, decide_gate
from .q4k_pack import GRP, SUP, SUP_BYTES, pack_q4k

_LIB: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        cand = os.environ.get("CASCADE_KERNELS_SO") or str(
            Path(__file__).parent / "kernels" / "libcascade_kernels.so")
        if not Path(cand).exists():
            raise RuntimeError(
                f"libcascade_kernels.so nao encontrada em {cand}. "
                "Rode kernels/build.sh (gcc -mavx2 -mfma -mf16c). "
                "Sem o kernel nativo o caminho F0 seria ~60x mais lento — "
                "por design nao existe fallback silencioso.")
        _LIB = ctypes.CDLL(cand)
    return _LIB


def _ptr(a: np.ndarray) -> ctypes.c_void_p:
    return a.ctypes.data_as(ctypes.c_void_p)


class Q4KLinearModule(nn.Module):
    def __init__(
        self,
        packed: np.ndarray,
        out_features: int,
        in_features: int,
        *,
        u: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        gate_cfg: Optional[GateConfig] = None,
        path: str = "F0_GATE_F1",
    ):
        super().__init__()
        self.lib = _load_lib()
        self.path = path
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.nsup = packed.shape[1] // SUP_BYTES
        self.cols_padded = self.nsup * SUP
        if packed.shape[0] != out_features:
            raise ValueError("packed rows != out_features")
        self.packed = np.ascontiguousarray(packed)
        self.gate_cfg = gate_cfg or GateConfig()

        self.has_f1 = u is not None and s is not None and v is not None \
            and u.numel() > 0
        if self.has_f1:
            self.u_c = np.ascontiguousarray(
                u.detach().to(torch.float32).numpy())          # (out, r)
            self.s_c = np.ascontiguousarray(
                s.detach().to(torch.float32).numpy())          # (r,)
            vt = v.detach().to(torch.float32)
            if vt.shape[0] == self.in_features:                # (in, r) -> (r, in)
                vt = vt.T
            self.vt_c = np.ascontiguousarray(vt.numpy())       # (r, in)
            self.rank = int(self.s_c.shape[0])
        else:
            self.rank = 0

        self._xbuf = np.zeros(self.cols_padded, np.float32)
        self._xq = np.zeros(self.cols_padded, np.int8)
        self._qsx = np.zeros(self.nsup, np.float32)
        self._sumx = np.zeros(self.cols_padded // GRP, np.float32)
        self._y = np.zeros(self.out_features, np.float32)
        self.f0_calls = 0
        self.f1_calls = 0
        self.f1_skip_calls = 0
        self.last_gate_rate = 0.0

    @classmethod
    def from_dense(cls, w: torch.Tensor, *, rank: int = 0,
                   gate_cfg: Optional[GateConfig] = None,
                   path: str = "F0_GATE_F1",
                   clips: tuple = (1.0, 0.975, 0.95, 0.925, 0.9)) -> "Q4KLinearModule":
        """Codifica um peso denso: F0 = q4k(W); F1 = SVD low-rank do residuo."""
        from .codec import encode_qk
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        pad = (-in_f) % SUP
        wp = torch.nn.functional.pad(w.to(torch.float32), (0, pad))
        dq, planes = encode_qk(wp, g=GRP, bits=4, clips=clips)
        packed = pack_q4k(planes)
        u = s = v = None
        if rank > 0:
            resid = (wp - dq)
            uu, ss, vv = torch.svd_lowrank(resid, q=min(rank + 8, min(resid.shape)),
                                           niter=4)
            u, s, v = uu[:, :rank].contiguous(), ss[:rank].contiguous(), \
                vv[:, :rank].contiguous()
        return cls(packed, out_f, in_f, u=u, s=s, v=v,
                   gate_cfg=gate_cfg, path=path)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kw) -> "Q4KLinearModule":
        mod = cls.from_dense(linear.weight.detach(), **kw)
        if linear.bias is not None:
            mod.bias = linear.bias.detach().to(torch.float32).clone()
        return mod

    bias: Optional[torch.Tensor] = None

    def _gemv(self, x_row: np.ndarray, y_out: np.ndarray) -> None:
        self._xbuf[: self.in_features] = x_row
        self._xbuf[self.in_features:] = 0.0
        self.lib.q4k_prepare_x_i8(
            ctypes.c_int(self.cols_padded), _ptr(self._xbuf),
            _ptr(self._xq), _ptr(self._qsx), _ptr(self._sumx))
        self.lib.q4k_gemv_i8(
            ctypes.c_int(self.out_features), ctypes.c_int(self.cols_padded),
            _ptr(self.packed), _ptr(self._xq), _ptr(self._qsx),
            _ptr(self._sumx), _ptr(y_out))

    def _f1_add(self, x_row: np.ndarray, y_out: np.ndarray) -> None:
        self.lib.lowrank_gemv_f32(
            _ptr(np.ascontiguousarray(x_row)), ctypes.c_int(self.in_features),
            _ptr(self.u_c), _ptr(self.s_c), _ptr(self.vt_c),
            ctypes.c_int(self.out_features), ctypes.c_int(self.rank),
            _ptr(y_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).to(torch.float32)
        n = int(x2.shape[0])
        xs = np.ascontiguousarray(x2.detach().cpu().numpy())

        if self.path == "F0_ONLY" or not self.has_f1:
            mask = torch.zeros(n, dtype=torch.bool)
            self.last_gate_rate = 0.0
        elif self.path == "F0_PLUS_F1_ALWAYS":
            mask = torch.ones(n, dtype=torch.bool)
            self.last_gate_rate = 1.0
        else:
            mask, meta = decide_gate(x2, self.gate_cfg)
            self.last_gate_rate = float(meta["activation_rate"])

        ys = np.empty((n, self.out_features), np.float32)
        for b in range(n):
            self._gemv(xs[b], ys[b])
            if bool(mask[b]):
                self._f1_add(xs[b], ys[b])
        self.f0_calls += n
        applied = int(mask.sum().item())
        self.f1_calls += applied
        self.f1_skip_calls += n - applied

        out = torch.from_numpy(ys)
        if self.bias is not None:
            out = out + self.bias
        if x.dtype != out.dtype:
            out = out.to(dtype=x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def calibrate_gate(self, xs: torch.Tensor) -> float:
        cal = GateCalibrator(self.gate_cfg)
        cal.observe(xs)
        return cal.freeze()

    def stats(self) -> Dict[str, Any]:
        total = max(self.f0_calls, 1)
        resident = int(self.packed.nbytes)
        if self.has_f1:
            resident += self.u_c.nbytes + self.s_c.nbytes + self.vt_c.nbytes
        return {
            "f0_calls": self.f0_calls,
            "f1_calls": self.f1_calls,
            "f1_skip_rate": 1.0 - (self.f1_calls / total),
            "last_gate_rate": self.last_gate_rate,
            "resident_bytes": resident,
            "w0_cache_bytes": 0,
            "resident_bytes_with_cache": resident,
            "bpw_f0": round(self.packed.nbytes * 8.0
                            / (self.out_features * self.cols_padded), 4),
            "gate_threshold": self.gate_cfg.fixed_threshold,
        }


def patch_block_linears(
    block: nn.Module,
    *,
    rank: int = 0,
    gate_percentile: float = 70.0,
    path: str = "F0_GATE_F1",
) -> Dict[str, Q4KLinearModule]:
    """Troca in-place as nn.Linear do bloco por Q4KLinearModule."""
    replaced: Dict[str, Q4KLinearModule] = {}
    for name, mod in list(block.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        cfg = GateConfig(percentile=gate_percentile)
        casc = Q4KLinearModule.from_linear(mod, rank=rank, gate_cfg=cfg, path=path)
        parent = block
        parts = name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = casc
        else:
            setattr(parent, leaf, casc)
        replaced[name] = casc
    return replaced
