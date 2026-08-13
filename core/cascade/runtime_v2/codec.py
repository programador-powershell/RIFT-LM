#!/usr/bin/env python3
"""CASCADE F0 v2 — codec Q4K-CASCADE (referencia + prova de roundtrip).

Motivacao (medida em pesos BF16 reais da Muse-Glimmer-30B, camadas 0/25/51):

    codec atual  int4/g32 simetrico ......... 4.500 bpw  cos ~0.9951
    GGUF UD-Q4_K_XL (imatrix) ............... 4.500 bpw  cos ~0.9973
    q4k/g32 +clip (este arquivo, data-free) . 4.500 bpw  cos ~0.9971
    q4k/g64 +clip (este arquivo, data-free) . 4.3125 bpw cos ~0.9962

A vantagem do GGUF e ~97% ESTRUTURA, nao imatrix. Tres mudancas fecham o gap:
  1. quantizacao ASSIMETRICA por grupo (min + escala, nao simetrica);
  2. escalas em DOIS NIVEIS: escala/min de cada grupo quantizados a 6 bits
     contra um super-bloco de 256 colunas (overhead 12/g + 32/256 em vez de
     16/g em fp16) — granularidade fina quase de graca;
  3. CLIP SEARCH por grupo: testa fatores de encolhimento do intervalo e fica
     com o de menor MSE (mesma ideia do ZDC do GEYSER).

Formato binario por tensor (little-endian), por linha:
    [super0 | super1 | ...]  onde cada super-bloco de S=256 colunas =
      d_scale fp16, d_min fp16,
      S/g sub-escalas u6 + S/g sub-mins i6 (empacotados 4 valores/3 bytes),
      S valores u4 (2/byte)  [g=32 ou 64]
Degraus de resgate (mesma estrutura, mais bits): q5k (5.5 bpw) e q6k
(6.56 bpw) substituem o fallback raw de 16 bpw — na Muse, os 3 tensores que
o conversor atual mandou para raw custariam 5.5 bpw com cos >=0.999.

Integracao no cascade_converter.py:
  - F0_CODECS["q4k"] = F0Codec("q4k", 4, "Q4K_CASCADE_TWO_LEVEL_ASYM",
                               "f0.q4k", True)  (bpw via kq_bpw abaixo)
  - LADDERS["kquant"] = [("q4k", 64), ("q4k", 32), ("q5k", 32), ("q6k", 32)]
    (auto: fonte BF16/F16/F32 -> "kquant"; low-bit -> "safe" como hoje)
  - write_f0: usar encode_q4k por chunk de linhas (streaming identico ao atual)
"""
from __future__ import annotations

import math

import torch

SUP = 256


def kq_bpw(bits: int, g: int, sup: int = SUP) -> float:
    return bits + 12.0 / g + 32.0 / sup


def _quantize_one_clip(wg: torch.Tensor, bits: int, sup_groups: int,
                       pad: int, clip: float):
    n = 2 ** bits - 1
    rows, ngroups, _ = wg.shape
    lo = wg.amin(-1, keepdim=True)
    hi = wg.amax(-1, keepdim=True)
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 * clip
    s = ((2 * half) / n).clamp_min(1e-12)
    m = mid - half
    s2 = torch.nn.functional.pad(s.squeeze(-1), (0, pad)).reshape(rows, -1, sup_groups)
    m2 = torch.nn.functional.pad(m.squeeze(-1), (0, pad)).reshape(rows, -1, sup_groups)
    d_s = (s2.amax(-1, keepdim=True) / 63.0).clamp_min(1e-12).to(torch.float16)
    d_m = (m2.abs().amax(-1, keepdim=True) / 31.0).clamp_min(1e-12).to(torch.float16)
    sq = (s2 / d_s.to(torch.float32)).round().clamp(0, 63).to(torch.uint8)
    mq = (m2 / d_m.to(torch.float32)).round().clamp(-31, 31).to(torch.int8)
    s_eff = (sq.to(torch.float32) * d_s.to(torch.float32)).reshape(
        rows, -1)[:, :ngroups].unsqueeze(-1).clamp_min(1e-12)
    m_eff = (mq.to(torch.float32) * d_m.to(torch.float32)).reshape(
        rows, -1)[:, :ngroups].unsqueeze(-1)
    q = ((wg - m_eff) / s_eff).round().clamp(0, n).to(torch.uint8)
    dq = q.to(torch.float32) * s_eff + m_eff
    err_g = (dq - wg).pow(2).sum(-1)
    err_sup = torch.nn.functional.pad(err_g, (0, pad)).reshape(
        rows, -1, sup_groups).sum(-1, keepdim=True)
    return (q, sq, mq, d_s, d_m), err_sup


def _two_level_params(wg: torch.Tensor, bits: int, sup_groups: int,
                      clips: tuple[float, ...]):
    rows, ngroups, g = wg.shape
    pad = (-ngroups) % sup_groups
    best = None
    best_err = None
    for clip in clips:
        planes, err_sup = _quantize_one_clip(wg, bits, sup_groups, pad, clip)
        if best is None:
            best, best_err = planes, err_sup
            continue
        sel_sup = err_sup < best_err
        best_err = torch.minimum(err_sup, best_err)
        sel_g = sel_sup.expand(-1, -1, sup_groups).reshape(
            rows, -1)[:, :ngroups].unsqueeze(-1)
        best = (torch.where(sel_g, planes[0], best[0]),
                torch.where(sel_sup, planes[1], best[1]),
                torch.where(sel_sup, planes[2], best[2]),
                torch.where(sel_sup, planes[3], best[3]),
                torch.where(sel_sup, planes[4], best[4]))
    return best


def encode_qk(w: torch.Tensor, g: int = 64, bits: int = 4,
              clips: tuple[float, ...] = (1.0, 0.975, 0.95, 0.925, 0.9)):
    """Retorna (dequant p/ metricas, dict de planos p/ serializar)."""
    rows, cols = w.shape
    padc = (-cols) % g
    wp = torch.nn.functional.pad(w, (0, padc)) if padc else w
    wg = wp.reshape(rows, -1, g)
    q, sq, mq, d_s, d_m = _two_level_params(wg, bits, SUP // g, clips)
    planes = {"q": q, "sub_scales_u6": sq, "sub_mins_i6": mq,
              "sup_scale_f16": d_s, "sup_min_f16": d_m,
              "g": g, "bits": bits, "rows": rows, "cols": cols}
    return decode_qk(planes), planes


def decode_qk(planes: dict) -> torch.Tensor:
    g, rows, cols = planes["g"], planes["rows"], planes["cols"]
    q = planes["q"].to(torch.float32)
    sup_groups = SUP // g
    ngroups = q.shape[1]
    pad = (-ngroups) % sup_groups
    s_eff = (planes["sub_scales_u6"].to(torch.float32).reshape(rows, -1, sup_groups)
             * planes["sup_scale_f16"].to(torch.float32))
    m_eff = (planes["sub_mins_i6"].to(torch.float32).reshape(rows, -1, sup_groups)
             * planes["sup_min_f16"].to(torch.float32))
    s_eff = s_eff.reshape(rows, -1)[:, :ngroups].unsqueeze(-1)
    m_eff = m_eff.reshape(rows, -1)[:, :ngroups].unsqueeze(-1)
    w = q * s_eff + m_eff
    return w.reshape(rows, -1)[:, :cols]


def packed_bytes(rows: int, cols: int, g: int, bits: int) -> int:
    ngroups = math.ceil(cols / g)
    nsup = math.ceil(ngroups * g / SUP)
    vals = rows * math.ceil(cols * bits / 8)
    subs = rows * math.ceil(ngroups * 12 / 8)
    sups = rows * nsup * 4
    return vals + subs + sups


def _cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = (a * b).sum(dtype=torch.float64).item()
    na = a.pow(2).sum(dtype=torch.float64).item()
    nb = b.pow(2).sum(dtype=torch.float64).item()
    return dot / max(math.sqrt(na * nb), 1e-300)


def self_test() -> int:
    torch.manual_seed(0)
    w = torch.randn(512, 1024) * 0.02
    for g, bits, min_cos in ((64, 4, 0.9955), (32, 4, 0.9965), (32, 5, 0.999),
                             (32, 6, 0.9997)):
        dq, planes = encode_qk(w, g=g, bits=bits)
        rt = decode_qk(planes)
        assert torch.allclose(dq, rt, atol=1e-6), (g, bits)
        c = _cos64(w, dq)
        bpw_real = packed_bytes(512, 1024, g, bits) * 8 / w.numel()
        print(f"q{bits}k/g{g}: cos={c:.6f} (>= {min_cos}) "
              f"bpw={kq_bpw(bits, g):.4f} (packed real {bpw_real:.4f}) "
              f"roundtrip OK")
        assert c >= min_cos, c
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
