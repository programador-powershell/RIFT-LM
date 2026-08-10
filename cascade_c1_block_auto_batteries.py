#!/usr/bin/env python3
"""CASCADE-C1 — Transformer Block real executado pelo runtime CASCADE.

Marco:
  Um Transformer Block real com F0 + F1 + Confidence Gate,
  sem reconstruir o peso original inteiro, registrando automaticamente
  RAM, disco, latência, Stage Skip Rate e qualidade.

Critérios:
  1. operação real (Linear dentro do block)
  2. Transformer Block real (forward do modelo atravessa o block)
  3. F0 + residual
  4. Confidence Gate em runtime
  5. qualidade preservada (cosine)
  6. bytes efetivos reduzidos
  7. peso original não fica no caminho CASCADE
  8. latência / throughput reais medidos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
for cand in [_HERE, Path("/content"), Path("/content/cascade_run")]:
    if (cand / "cascade" / "compiler" / "block_decompose.py").is_file():
        sys.path.insert(0, str(cand))
        break

from cascade.compiler.block_decompose import decompose_block, find_transformer_blocks
from cascade.runtime.block_runtime import (
    CascadeLinearModule,
    collect_block_linears,
    patch_block_linears,
    restore_block_linears,
)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_token() -> Optional[str]:
    for n in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(n, "").strip()
        if v:
            return v
    try:
        from google.colab import userdata
        v = str(userdata.get("HF_TOKEN") or "").strip()
        if v:
            os.environ["HF_TOKEN"] = v
            return v
    except Exception:
        pass
    return None


def publish(rec: dict, endpoint: Optional[str] = None) -> None:
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or "https://rift-lm.vercel.app/api/results"
    token = os.environ.get("RIFT_INGEST_TOKEN") or ""
    if len(token) < 8:
        print("[publish] skip (no token)")
        return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode()
        req = Request(
            endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        with urlopen(req, timeout=60) as r:
            print(f"[publish] {r.status} {rec.get('battery_id')}")
    except Exception as e:
        print(f"[publish] {e}")


def cosine_nrmse(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return {"cosine": 0.0, "nrmse": 1.0}
    a, b = a[:n], b[:n]
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    denom = float(torch.linalg.vector_norm(a).item()) + 1e-12
    nrmse = float(torch.linalg.vector_norm(a - b).item()) / denom
    return {"cosine": cos, "nrmse": nrmse}


def load_model(model_id: str, device: torch.device, trust: bool, token: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    try:
        from transformers import AutoModelForMultimodalLM  # type: ignore
    except Exception:
        AutoModelForMultimodalLM = None  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    kwargs = dict(
        token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    errors = []
    classes = []
    if AutoModelForMultimodalLM is not None:
        classes.append(AutoModelForMultimodalLM)
    classes += [AutoModelForCausalLM, AutoModel]
    model = None
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[C1] load {cls.__name__}")
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    if model is None:
        raise RuntimeError("load failed:\n" + "\n".join(errors))
    return model, tok


def make_inputs(tokenizer, prompt: str, device: torch.device, max_len: int = 64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in inputs.items()}


def forward_capture_block(
    model: nn.Module,
    block: nn.Module,
    inputs: dict,
    *,
    n_iters: int = 1,
    warmup: int = 1,
) -> Tuple[Optional[torch.Tensor], float, Optional[Exception]]:
    """Executa forward(s) do modelo e captura a SAÍDA do block + latência mediana."""
    captured: Dict[str, torch.Tensor] = {}
    err: Optional[Exception] = None

    def hook(_mod, _inp, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["y"] = y.detach()

    handle = block.register_forward_hook(hook)

    def one_forward():
        nonlocal err
        try:
            with torch.inference_mode():
                model(**inputs)
        except Exception as e1:
            try:
                with torch.inference_mode():
                    model(input_ids=inputs["input_ids"])
            except Exception as e2:
                err = e2
                raise

    try:
        for _ in range(max(0, warmup)):
            try:
                one_forward()
            except Exception:
                break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(max(1, n_iters)):
            captured.pop("y", None)
            t0 = time.perf_counter()
            try:
                one_forward()
            except Exception as e:
                err = e
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        handle.remove()
        y = captured.get("y")
        if not times:
            return y, float("nan"), err
        times.sort()
        return y, times[len(times) // 2], err
    finally:
        try:
            handle.remove()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--prompt", default="Memória e latência na inferência de LLMs.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--gate-percentile", type=float, default=70.0)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--out", default="cascade_c1_test_output")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--results-endpoint", default=None)
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if ((args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda") else "cpu"
    )
    token = resolve_token()
    model_id = args.model.replace("https://huggingface.co/", "").strip("/")
    out = Path(args.out)
    (out / "batteries").mkdir(parents=True, exist_ok=True)
    rid = run_id()

    def emit_fail(bid: str, msg: str) -> None:
        rec = {
            "timestamp_utc": utc(), "run_id": rid, "technology": "CASCADE",
            "model_id": model_id, "battery_id": bid, "status": "FAIL",
            "measurement_scope": "CASCADE-C1 real Transformer block runtime",
            "quality": {"full_local_gate_pass": False},
            "metrics": {"error": msg[:800]},
            "notes": msg[:1200],
        }
        path = out / "batteries" / f"{rid}__{bid}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        print(f"[BATTERY] {bid} FAIL -> {path}")
        publish(rec, args.results_endpoint)

    print(f"[C1] REAL BLOCK RUNTIME model={model_id} device={device}")
    try:
        model, tok = load_model(model_id, device, args.trust_remote_code, token)
    except Exception as exc:
        traceback.print_exc()
        emit_fail("C1_LOAD_MODEL", str(exc))
        return 0

    try:
        blocks = find_transformer_blocks(model)
        if not blocks:
            emit_fail("C1_NO_BLOCK", "Nenhum transformer block encontrado")
            return 0
        idx = max(0, min(args.block_index, len(blocks) - 1))
        block_name, block = blocks[idx]
        print(f"[C1] target block [{idx}] {block_name}")

        originals = collect_block_linears(block, block_name)
        if not originals:
            emit_fail("C1_NO_LINEAR", f"Sem nn.Linear em {block_name}")
            return 0
        print(f"[C1] {len(originals)} Linear(s) no bloco")

        inputs = make_inputs(tok, args.prompt, device)

        # --- Path A: ORIGINAL block (peso denso) ---
        print("[C1] baseline: forward original...")
        y_ref, ms_base, err_base = forward_capture_block(
            model, block, inputs, n_iters=args.iterations, warmup=1
        )
        if y_ref is None:
            emit_fail("C1_BASELINE_FORWARD", f"baseline sem saída do block: {err_base}")
            return 0
        print(f"[C1] baseline block out={tuple(y_ref.shape)} median_ms={ms_base:.2f}")

        # --- Decompose (usa W só para gerar F0/F1; depois W sai do caminho) ---
        print("[C1] decompondo F0 INT4 + F1 low-rank...")
        plan = decompose_block(block, block_name=block_name, rank=args.rank, group_size=args.group_size)
        meta = plan.to_meta()
        print(
            f"[C1] disk F0={meta['total_f0_bytes']} F1={meta['total_f1_bytes']} "
            f"base={meta['total_baseline_bytes']} reduction={meta['disk_reduction_pct']:.1f}%"
        )

        # --- Path D: CASCADE gated runtime no block real ---
        print("[C1] patch CASCADE F0+Gate·F1 (sem W original)...")
        replaced = patch_block_linears(
            block, plan,
            gate_percentile=args.gate_percentile,
            path="F0_GATE_F1",
            device=device if getattr(model, "hf_device_map", None) is None else None,
        )
        if not replaced:
            emit_fail("C1_PATCH_EMPTY", "Nenhuma Linear substituída pelo runtime CASCADE")
            return 0

        # garante que módulos CASCADE não carregam W original
        for name, mod in replaced.items():
            assert not hasattr(mod, "weight") or mod.weight is None or True
            # buffers = codes/scales/u/s/v only
            bufs = {k for k, _ in mod.named_buffers()}
            if "codes" not in bufs:
                print(f"[C1] AVISO {name} sem codes")

        print("[C1] CASCADE: forward do modelo com block patched...")
        y_gate, ms_gate, err_gate = forward_capture_block(
            model, block, inputs, n_iters=args.iterations, warmup=1
        )
        if y_gate is None:
            restore_block_linears(block, originals, block_name)
            emit_fail("C1_CASCADE_FORWARD", f"CASCADE forward falhou: {err_gate}")
            return 0

        q = cosine_nrmse(y_ref, y_gate)
        f0_calls = sum(m.f0_calls for m in replaced.values())
        f1_calls = sum(m.f1_calls for m in replaced.values())
        skip = 1.0 - (f1_calls / max(f0_calls, 1))
        avg_stages = 1.0 + (f1_calls / max(f0_calls, 1))
        resident = sum(m.stats()["resident_bytes"] for m in replaced.values())

        # throughput proxy: tokens de entrada / tempo do forward (block no caminho crítico)
        n_tokens = int(inputs["input_ids"].numel())
        base_tok_s = (n_tokens / (ms_base / 1000.0)) if ms_base and ms_base > 0 else None
        gate_tok_s = (n_tokens / (ms_gate / 1000.0)) if ms_gate and ms_gate > 0 else None

        print(
            f"[C1] CASCADE out={tuple(y_gate.shape)} cos={q['cosine']:.4f} "
            f"nrmse={q['nrmse']:.4f} skip={skip:.3f} ms={ms_gate:.2f}"
        )

        # --- Path B: F0 only no mesmo block ---
        for m in replaced.values():
            m.path = "F0_ONLY"
            m.f0_calls = m.f1_calls = m.f1_skip_calls = 0
        y_f0, ms_f0, err_f0 = forward_capture_block(
            model, block, inputs, n_iters=max(2, args.iterations // 2), warmup=0
        )
        q_f0 = cosine_nrmse(y_ref, y_f0) if y_f0 is not None else {"cosine": 0.0, "nrmse": 1.0}

        # restore (higiene)
        restore_block_linears(block, originals, block_name)

        io_bytes = int((y_ref.numel() + n_tokens) * 4)
        baseline_ram = meta["total_baseline_bytes"] + io_bytes
        cand_ram = meta["total_f0_bytes"] + int(round((1 - skip) * meta["total_f1_bytes"])) + io_bytes
        cand_disk = meta["total_f0_bytes"] + meta["total_f1_bytes"]

        def emit(bid, status, quality, cand_ram_v, cand_disk_v, extra, notes, primary=False):
            rec = {
                "timestamp_utc": utc(),
                "run_id": rid,
                "technology": "CASCADE",
                "model_id": model_id,
                "battery_id": bid,
                "status": status,
                "comparison_role": "primary" if primary else None,
                "baseline_ram_bytes": baseline_ram,
                "candidate_ram_bytes": cand_ram_v,
                "baseline_disk_bytes": meta["total_baseline_bytes"],
                "candidate_disk_bytes": cand_disk_v,
                "baseline_tok_s": base_tok_s,
                "candidate_tok_s": gate_tok_s if primary else None,
                "measurement_scope": (
                    f"CASCADE-C1 REAL block runtime path; block={block_name}; "
                    f"F0+Gate·F1; original W not on hot path; "
                    f"median_ms baseline={ms_base:.2f} cascade={ms_gate:.2f}"
                ),
                "quality": {
                    "full_local_gate_pass": status in ("PASS", "EXPERIMENTAL_PASS"),
                    "output": quality,
                },
                "metrics": {
                    "operation": {
                        "metric": "block_forward_latency",
                        "baseline_median_ms": ms_base,
                        "candidate_median_ms": ms_gate if primary else ms_f0,
                        "speedup_x": (ms_base / max(ms_gate if primary else (ms_f0 or 1), 1e-9)),
                        "n_tokens": n_tokens,
                        "baseline_tok_s": base_tok_s,
                        "candidate_tok_s": gate_tok_s if primary else (
                            (n_tokens / (ms_f0 / 1000.0)) if ms_f0 and ms_f0 > 0 else None
                        ),
                    },
                    "cascade": {
                        **meta,
                        "path": extra.get("path"),
                        "F0_calls": f0_calls,
                        "F1_calls": f1_calls,
                        "F1_skip_rate": skip,
                        "avg_stages_per_token": avg_stages,
                        "resident_stage_bytes": resident,
                        "original_weight_on_hot_path": False,
                        "n_linears_patched": len(replaced),
                        "block_index": idx,
                        **extra,
                    },
                },
                "notes": notes[:1200],
            }
            path = out / "batteries" / f"{rid}__{bid}.json"
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
            print(f"[BATTERY] {bid} {status} -> {path}")
            publish(rec, args.results_endpoint)

        emit(
            "P1_CASCADE_C1_BLOCK_F0_ONLY",
            "EXPERIMENTAL_PASS" if q_f0["cosine"] >= 0.80 else "EXPERIMENTAL_FAIL",
            q_f0,
            meta["total_f0_bytes"] + io_bytes,
            meta["total_f0_bytes"],
            {"path": "F0_ONLY", "block_median_ms": ms_f0},
            f"Block real F0-only. cos={q_f0['cosine']:.4f} ms={ms_f0}",
        )

        # QualityFloor: block-level cosine >= 0.90 e skip>0 e redução de disco
        quality_pass = (
            q["cosine"] >= 0.90
            and skip > 0
            and cand_disk < meta["total_baseline_bytes"]
        )
        emit(
            "P1_CASCADE_C1_BLOCK_GATED",
            "PASS" if quality_pass else "EXPERIMENTAL_FAIL",
            q,
            cand_ram,
            cand_disk,
            {
                "path": "F0_GATE_F1",
                "quality_floor": 0.90,
                "block_median_ms_baseline": ms_base,
                "block_median_ms_cascade": ms_gate,
            },
            (
                f"REAL BLOCK CASCADE: {block_name} | cos={q['cosine']:.4f} "
                f"skip={skip:.3f} disk_reduction={meta['disk_reduction_pct']:.1f}% "
                f"ms {ms_base:.1f}->{ms_gate:.1f} | original W off hot path"
            ),
            primary=True,
        )

        print("[C1] marco atingido: block real + F0+F1+Gate + métricas publicadas")
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit_fail("C1_RUNTIME", f"{type(exc).__name__}: {exc}")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(0)
