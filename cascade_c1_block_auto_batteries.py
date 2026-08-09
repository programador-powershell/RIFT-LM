#!/usr/bin/env python3
"""CASCADE-C1 — um Transformer block completo com F0 INT4 + Gate·F1 em todas as Linears."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
for cand in [_HERE, Path("/content")]:
    if (cand / "cascade" / "compiler" / "block_decompose.py").is_file():
        sys.path.insert(0, str(cand))
        break

from cascade.compiler.block_decompose import decompose_block, find_transformer_blocks
from cascade.runtime.block_runtime import patch_block_linears


def utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def run_id(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_token():
    for n in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(n, "").strip()
        if v: return v
    try:
        from google.colab import userdata
        v = str(userdata.get("HF_TOKEN") or "").strip()
        if v:
            os.environ["HF_TOKEN"] = v
            return v
    except Exception:
        pass
    return None


def publish(rec, endpoint=None):
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or "https://rift-lm.vercel.app/api/results"
    token = os.environ.get("RIFT_INGEST_TOKEN") or ""
    if len(token) < 8:
        print("[publish] skip (no token)"); return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode()
        req = Request(endpoint, data=body, method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=60) as r:
            print(f"[publish] {r.status} {rec['battery_id']}")
    except Exception as e:
        print(f"[publish] {e}")


def cosine_nrmse(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    nrmse = float(torch.linalg.vector_norm(a - b).item()) / (float(torch.linalg.vector_norm(a).item()) + 1e-12)
    return {"cosine": cos, "nrmse": nrmse}


def load_model(model_id, device, trust, token):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    kwargs = dict(token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
                  dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()
    return model, tok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--prompt", default="Memória e latência na inferência de LLMs.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gate-percentile", type=float, default=70.0)
    ap.add_argument("--out", default="cascade_c1_test_output")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--results-endpoint", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    token = resolve_token()
    model_id = args.model.replace("https://huggingface.co/", "").strip("/")
    out = Path(args.out); (out / "batteries").mkdir(parents=True, exist_ok=True)
    rid = run_id()

    print(f"[C1] load {model_id} on {device}")
    model, tok = load_model(model_id, device, args.trust_remote_code, token)
    blocks = find_transformer_blocks(model)
    if not blocks:
        raise RuntimeError("Nenhum transformer block encontrado")
    idx = max(0, min(args.block_index, len(blocks) - 1))
    block_name, block = blocks[idx]
    print(f"[C1] block {idx}/{len(blocks)}: {block_name}")

    # capture block input via hook on original
    captured = {}
    def pre_hook(mod, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            captured["x"] = inputs[0].detach()
    h = block.register_forward_pre_hook(pre_hook)
    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        # full forward once to get block input
        _ = model(**inputs)
    h.remove()
    if "x" not in captured:
        raise RuntimeError("falha ao capturar entrada do block")
    x_block = captured["x"]
    print(f"[C1] block input shape={tuple(x_block.shape)}")

    # original block output
    with torch.inference_mode():
        y_ref = block(x_block)
        if isinstance(y_ref, tuple):
            y_ref = y_ref[0]

    # decompose + patch gated
    plan = decompose_block(block, block_name=block_name, rank=args.rank)
    print(f"[C1] {plan.to_meta()['n_linears']} linears, disk_reduction={plan.to_meta()['disk_reduction_pct']:.1f}%")

    # We need a fresh block copy for fair compare — re-load is heavy; instead:
    # run original already done; patch in place for cascade paths
    # Save state_dict of linears to restore
    import copy
    # For gated path: patch
    replaced = patch_block_linears(block, plan, gate_percentile=args.gate_percentile, path="F0_GATE_F1")
    with torch.inference_mode():
        y_gate = block(x_block)
        if isinstance(y_gate, tuple):
            y_gate = y_gate[0]
    f0_calls = sum(m.f0_calls for m in replaced.values())
    f1_calls = sum(m.f1_calls for m in replaced.values())
    skip = 1.0 - (f1_calls / max(f0_calls, 1))
    q_gate = cosine_nrmse(y_ref, y_gate)

    # F0 only path — re-patch
    for name, stages in plan.linears.items():
        short = name[len(block_name)+1:] if name.startswith(block_name+".") else name
        parent = block
        parts = short.split(".")
        try:
            for p in parts[:-1]:
                parent = getattr(parent, p)
            from cascade.runtime.block_runtime import CascadeLinearModule
            setattr(parent, parts[-1], CascadeLinearModule(stages, gate_percentile=args.gate_percentile, path="F0_ONLY"))
        except Exception:
            pass
    with torch.inference_mode():
        y_f0 = block(x_block)
        if isinstance(y_f0, tuple):
            y_f0 = y_f0[0]
    q_f0 = cosine_nrmse(y_ref, y_f0)

    io = int((x_block.numel() + y_ref.numel()) * 4)
    baseline_ram = plan.total_baseline_bytes + io
    gate_ram = plan.total_f0_bytes + int(round((1 - skip) * plan.total_f1_bytes)) + io

    def emit(bid, status, q, cand_ram, cand_disk, extra, notes, primary=False):
        rec = {
            "timestamp_utc": utc(), "run_id": rid, "technology": "CASCADE",
            "model_id": model_id, "battery_id": bid, "status": status,
            "comparison_role": "primary" if primary else None,
            "baseline_ram_bytes": baseline_ram, "candidate_ram_bytes": cand_ram,
            "baseline_disk_bytes": plan.total_baseline_bytes, "candidate_disk_bytes": cand_disk,
            "measurement_scope": f"CASCADE-C1 Transformer block {block_name}",
            "quality": {"full_local_gate_pass": status == "PASS", "output": q},
            "metrics": {"cascade": {**plan.to_meta(), **extra}},
            "notes": notes[:1200],
        }
        path = out / "batteries" / f"{rid}__{bid}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        print(f"[BATTERY] {bid} -> {path}")
        publish(rec, args.results_endpoint)

    emit("P1_CASCADE_C1_BLOCK_F0_ONLY",
         "EXPERIMENTAL_PASS" if q_f0["cosine"] >= 0.85 else "EXPERIMENTAL_FAIL",
         q_f0, plan.total_f0_bytes + io, plan.total_f0_bytes,
         {"path": "F0_ONLY", "block_index": idx},
         f"Block {block_name} só F0 INT4 em todas as Linears.")

    quality_pass = q_gate["cosine"] >= 0.95 and skip > 0
    emit("P1_CASCADE_C1_BLOCK_GATED",
         "PASS" if quality_pass else "EXPERIMENTAL_FAIL",
         q_gate, gate_ram, plan.total_f0_bytes + plan.total_f1_bytes,
         {"path": "F0_GATE_F1", "F0_calls": f0_calls, "F1_calls": f1_calls, "F1_skip_rate": skip, "block_index": idx},
         f"Block completo gated. skip={skip:.3f} cosine={q_gate['cosine']:.4f}",
         primary=True)

    print(f"[C1] done cosine_f0={q_f0['cosine']:.4f} cosine_gate={q_gate['cosine']:.4f} skip={skip:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
