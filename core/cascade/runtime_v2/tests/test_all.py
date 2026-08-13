#!/usr/bin/env python3
"""Valida as 4 correcoes do cascade_runtime_v2 e mede a regressao de
desempenho contra os dois caminhos do runtime antigo."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cascade_runtime_v2 import (GateCalibrator, GateConfig, Q4KLinearModule,
                                decide_gate, pack_q4k, unpack_q4k)
from cascade_runtime_v2.codec import decode_qk, encode_qk

torch.manual_seed(7)
REPORT: dict = {}


def teste_1_pack_roundtrip():
    w = torch.randn(384, 1024) * 0.02
    dq, planes = encode_qk(w, g=32, bits=4)
    packed = pack_q4k(planes)
    planes2 = unpack_q4k(packed, 384, 1024)
    dq2 = decode_qk(planes2)
    assert torch.allclose(dq, dq2, atol=1e-6), "roundtrip pack/unpack"
    REPORT["1_pack_roundtrip"] = {"status": "PASS",
                                  "bytes": int(packed.nbytes),
                                  "bpw": round(packed.nbytes * 8 / w.numel(), 3)}
    print("[1] pack/unpack 144B: PASS")


def teste_2_f0_correcao():
    w = torch.randn(512, 1280) * 0.02  # 1280 nao e multiplo de 256 -> testa pad
    x = torch.randn(4, 1280)
    mod = Q4KLinearModule.from_dense(w, rank=0, path="F0_ONLY")
    y = mod(x)
    dq, planes = encode_qk(
        torch.nn.functional.pad(w, (0, (-1280) % 256)), g=32, bits=4)
    y_ref = x @ dq[:, :1280].T
    cos = torch.nn.functional.cosine_similarity(
        y.reshape(1, -1), y_ref.reshape(1, -1)).item()
    assert cos > 0.9995, cos
    st = mod.stats()
    assert st["w0_cache_bytes"] == 0
    assert st["bpw_f0"] < 4.6
    REPORT["2_f0_kernel"] = {"status": "PASS", "cos_vs_ref": round(cos, 6),
                             "bpw_residente": st["bpw_f0"],
                             "w0_cache_bytes": st["w0_cache_bytes"]}
    print(f"[2] F0 kernel (com padding): PASS cos={cos:.6f} "
          f"residente {st['bpw_f0']} bpw, cache fp32 = 0")


def teste_3_gate_batch1():
    cfg = GateConfig(percentile=70.0)
    xs_decode = [torch.randn(1, 256) for _ in range(200)]
    sem_cal = [decide_gate(x, cfg)[0].item() for x in xs_decode[:50]]
    taxa_sem_cal = sum(sem_cal) / len(sem_cal)
    assert taxa_sem_cal == 0.0, "sem calibracao deve ser F0_ONLY (fail-safe)"
    cal = GateCalibrator(cfg)
    for x in xs_decode[:100]:
        cal.observe(x)
    thr = cal.freeze()
    com_cal = [decide_gate(x, cfg)[0].item() for x in xs_decode[100:]]
    taxa = sum(com_cal) / len(com_cal)
    assert 0.10 <= taxa <= 0.50, f"esperado ~30% (p70), veio {taxa}"
    REPORT["3_gate_batch1"] = {
        "status": "PASS", "threshold_calibrado": round(thr, 4),
        "taxa_f1_sem_calibracao (bug antigo: 1.0)": taxa_sem_cal,
        "taxa_f1_batch1_calibrado (~30% esperado p/ p70)": round(taxa, 3)}
    print(f"[3] gate batch-1: PASS sem-cal={taxa_sem_cal:.0%} (antes: 100%) "
          f"| calibrado={taxa:.0%} (~30% alvo p70)")


def teste_4_f1_lowrank():
    w = torch.randn(512, 1024) * 0.02
    x = torch.randn(1, 1024)
    mod = Q4KLinearModule.from_dense(w, rank=16, path="F0_PLUS_F1_ALWAYS")
    y = mod(x).reshape(-1)
    dq, planes = encode_qk(w, g=32, bits=4)
    f0 = decode_qk(planes)
    resid_hat = (torch.from_numpy(mod.u_c) * torch.from_numpy(mod.s_c)) \
        @ torch.from_numpy(mod.vt_c)
    y_ref = (x @ (f0 + resid_hat).T).reshape(-1)
    rel = (y - y_ref).abs().max().item() / max(y_ref.abs().max().item(), 1e-9)
    assert rel < 0.02, rel
    cos_f0f1 = torch.nn.functional.cosine_similarity(
        y.reshape(1, -1), (x @ w.T).reshape(1, -1)).item()
    cos_f0 = torch.nn.functional.cosine_similarity(
        (x @ f0.T).reshape(1, -1), (x @ w.T).reshape(1, -1)).item()
    REPORT["4_f1_lowrank_c"] = {
        "status": "PASS", "rel_err_vs_ref": round(rel, 5),
        "cos_saida_f0_vs_w": round(cos_f0, 6),
        "cos_saida_f0+f1_vs_w": round(cos_f0f1, 6)}
    print(f"[4] F1 lowrank C: PASS rel={rel:.1e} | cos F0={cos_f0:.6f} "
          f"-> F0+F1={cos_f0f1:.6f}")


def teste_5_regressao_desempenho():
    HID, FFN, LAYERS = 2048, 8192, 8
    shapes = [(HID, HID)] * 3 + [(FFN, HID)] * 2 + [(HID, FFN)]
    mods, dense = [], []
    for _ in range(LAYERS):
        lyr_m, lyr_d = [], []
        for r, c in shapes:
            w = torch.randn(r, c) * 0.02
            lyr_m.append(Q4KLinearModule.from_dense(w, rank=0, path="F0_ONLY"))
            lyr_d.append(w)
            del w
        mods.append(lyr_m)
        dense.append(lyr_d)
    packed_gb = sum(m.packed.nbytes for lyr in mods for m in lyr) / 1e9
    fp32_gb = sum(w.numel() * 4 for lyr in dense for w in lyr) / 1e9

    x_h, x_f = torch.randn(1, HID), torch.randn(1, FFN)

    def token_v2():
        for lyr in mods:
            for m, (r, c) in zip(lyr, shapes):
                m(x_h if c == HID else x_f)

    token_v2()
    t0 = time.perf_counter()
    for _ in range(3):
        token_v2()
    dt_v2 = (time.perf_counter() - t0) / 3

    q_planes = [[encode_qk(w, g=32, bits=4)[1] for w in lyr] for lyr in dense[:2]]

    def token_lowmem():  # caminho antigo CASCADE_LOW_MEM=1: dequant por chamada
        for lyr_p, lyr_d in zip(q_planes, dense[:2]):
            for planes, w in zip(lyr_p, lyr_d):
                w0 = decode_qk(planes)
                (x_h if w.shape[1] == HID else x_f) @ w0.T

    token_lowmem()
    t0 = time.perf_counter()
    token_lowmem()
    dt_low = (time.perf_counter() - t0) * (LAYERS / 2)

    def token_fp32cache():  # caminho antigo default: W fp32 residente + BLAS
        for lyr in dense:
            for w in lyr:
                (x_h if w.shape[1] == HID else x_f) @ w.T

    token_fp32cache()
    t0 = time.perf_counter()
    for _ in range(3):
        token_fp32cache()
    dt_fp32 = (time.perf_counter() - t0) / 3

    REPORT["5_regressao"] = {
        "modelo_sintetico_gb_fp32": round(fp32_gb, 2),
        "v2_kernel": {"residente_gb": round(packed_gb, 2),
                      "latencia_token_s": round(dt_v2, 4),
                      "tok_s": round(1 / dt_v2, 2),
                      "banda_gbs": round(packed_gb / dt_v2, 2)},
        "antigo_low_mem (dequant/chamada)": {
            "residente_gb": round(packed_gb, 2),
            "latencia_token_s": round(dt_low, 3),
            "tok_s": round(1 / dt_low, 3),
            "speedup_v2": round(dt_low / dt_v2, 1)},
        "antigo_default (cache fp32 + BLAS)": {
            "residente_gb": round(fp32_gb, 2),
            "latencia_token_s": round(dt_fp32, 4),
            "tok_s": round(1 / dt_fp32, 2),
            "ram_extra_vs_v2": f"{round(fp32_gb / packed_gb, 1)}x"},
    }
    print(f"[5] regressao: v2 {1/dt_v2:.2f} tok/s ({packed_gb:.2f} GB res) | "
          f"low_mem {1/dt_low:.3f} tok/s (v2 = {dt_low/dt_v2:.0f}x) | "
          f"fp32-cache {1/dt_fp32:.2f} tok/s ({fp32_gb:.2f} GB res = "
          f"{fp32_gb/packed_gb:.1f}x RAM)")


def main() -> int:
    teste_1_pack_roundtrip()
    teste_2_f0_correcao()
    teste_3_gate_batch1()
    teste_4_f1_lowrank()
    teste_5_regressao_desempenho()
    out = Path(__file__).parent / "test_report.json"
    out.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n")
    print(f"\nTODOS OS TESTES PASS -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
