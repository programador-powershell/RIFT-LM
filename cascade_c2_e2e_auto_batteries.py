#!/usr/bin/env python3
"""CASCADE-C2 — modelo pequeno end-to-end + Tok/s real (baseline) + métricas CASCADE por Linear amostrada."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for cand in [_HERE, Path("/content")]:
    if (cand / "cascade" / "compiler" / "decompose.py").is_file():
        sys.path.insert(0, str(cand))
        break

from cascade.compiler.decompose import decompose_linear_int4_lowrank
from cascade.runtime.reference import CascadeLinearRuntime


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
        print("[publish] skip"); return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode()
        req = Request(endpoint, data=body, method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=60) as r:
            print(f"[publish] {r.status} {rec['battery_id']}")
    except Exception as e:
        print(f"[publish] {e}")


def measure_tok_s(model, tokenizer, prompt, device, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
    tok_s = n_new / max(elapsed, 1e-9)
    return {"tok_s": tok_s, "n_new": n_new, "elapsed_s": elapsed, "ttft_proxy_ms": (elapsed / max(n_new, 1)) * 1000}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompt", default="Liste três técnicas para reduzir RAM na inferência:")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gate-percentile", type=float, default=70.0)
    ap.add_argument("--out", default="cascade_c2_test_output")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--results-endpoint", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    token = resolve_token()
    model_id = args.model.replace("https://huggingface.co/", "").strip("/")
    out = Path(args.out); (out / "batteries").mkdir(parents=True, exist_ok=True)
    rid = run_id()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[C2] load {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=args.trust_remote_code)
    kwargs = dict(token=token, trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True,
                  dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()

    print("[C2] measuring baseline tok/s...")
    speed = measure_tok_s(model, tok, args.prompt, device, max_new_tokens=args.max_new_tokens)
    print(f"[C2] baseline {speed['tok_s']:.2f} tok/s ({speed['n_new']} tokens in {speed['elapsed_s']:.2f}s)")

    # Sample one linear for CASCADE quality/RAM metrics
    weight = None
    layer_name = None
    for n, t in model.state_dict().items():
        if t.ndim == 2 and min(t.shape) >= 64 and any(k in n for k in ("q_proj", "o_proj", "gate_proj", "down_proj")):
            layer_name, weight = n, t.detach().float().cpu()
            break
    if weight is None:
        for n, t in model.state_dict().items():
            if t.ndim == 2 and min(t.shape) >= 32:
                layer_name, weight = n, t.detach().float().cpu()
                break

    stages = decompose_linear_int4_lowrank(weight, rank=args.rank)
    rt = CascadeLinearRuntime(stages, gate_percentile=args.gate_percentile)
    x = torch.randn(32, weight.shape[1])
    y_ref = torch.nn.functional.linear(x, weight)
    r = rt.execute(x, path="F0_GATE_F1")
    cos = float(torch.nn.functional.cosine_similarity(
        y_ref.reshape(1, -1), r["y"].reshape(1, -1)).item())

    io = int((x.numel() + y_ref.numel()) * 4)
    baseline_ram = stages.baseline_bytes + io
    rate = 1.0 - r["metrics"].f1_skip_rate
    cand_ram = stages.f0_bytes + int(round(rate * stages.f1_bytes)) + io

    rec = {
        "timestamp_utc": utc(), "run_id": rid, "technology": "CASCADE",
        "model_id": model_id, "battery_id": "P1_CASCADE_C2_E2E_TOKS",
        "status": "PASS" if speed["tok_s"] > 0 and cos >= 0.95 else "EXPERIMENTAL_FAIL",
        "comparison_role": "primary",
        "baseline_tok_s": speed["tok_s"],
        "candidate_tok_s": None,  # full CASCADE generate still experimental
        "baseline_ram_bytes": baseline_ram,
        "candidate_ram_bytes": cand_ram,
        "baseline_disk_bytes": stages.baseline_bytes,
        "candidate_disk_bytes": stages.f0_bytes + stages.f1_bytes,
        "measurement_scope": "CASCADE-C2 e2e baseline tok/s + sample Linear CASCADE metrics",
        "quality": {"full_local_gate_pass": cos >= 0.95, "output": {"cosine": cos}},
        "metrics": {
            "operation": {"metric": "e2e_tok_s", "baseline_tok_s": speed["tok_s"],
                          "ttft_proxy_ms": speed["ttft_proxy_ms"], "n_new": speed["n_new"]},
            "cascade": {
                **stages.to_meta(),
                "sample_layer": layer_name,
                "F1_skip_rate": r["metrics"].f1_skip_rate,
                "path": "F0_GATE_F1",
                "e2e_note": "tok/s is ORIGINAL model; candidate full-CASCADE generate is next iteration",
            },
        },
        "notes": (
            f"C2 baseline {speed['tok_s']:.2f} tok/s on {device.type}. "
            f"Sample Linear {layer_name} cosine={cos:.4f} skip={r['metrics'].f1_skip_rate:.3f}."
        )[:1200],
    }
    path = out / "batteries" / f"{rid}__P1_CASCADE_C2_E2E_TOKS.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"[BATTERY] {path}")
    publish(rec, args.results_endpoint)
    print("[C2] done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
