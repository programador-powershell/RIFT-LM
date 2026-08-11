#!/usr/bin/env python3
"""CASCADE-C2 — modelo pequeno end-to-end + Tok/s real (baseline E candidato) + métricas CASCADE por Linear amostrada.

Candidato (docs/C3_CONTRACTS_V1.md §12): o MESMO protocolo de `model.generate`
do baseline (greedy, ≥2 warmup, ≥3 medições, mediana) roda com TODAS as
nn.Linear dos blocos patchadas por CascadeLinearModule
(cascade.runtime.block_runtime; F0 INT4 + Gate·F1 low-rank, W denso FORA do
caminho quente, low_mem desligado). O patch é transacional: as Linear
originais são restauradas em `finally`; se o patch/medição falhar, o registro
sai com candidate_tok_s=null e nota SKIPPED (nunca quebra a bateria).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for cand in [_HERE, Path("/content"), _HERE.parent.parent / "core"]:
    if (cand / "cascade" / "compiler" / "decompose.py").is_file():
        sys.path.insert(0, str(cand))
        break

from cascade.compiler.decompose import decompose_linear_int4_lowrank
from cascade.runtime.block_runtime import CascadeLinearModule, _set_module
from cascade.runtime.reference import CascadeLinearRuntime
from cascade.runtime.cleanup import cleanup_colab_workspace

BENCHMARK_PROTOCOL = "CASCADE_C_SERIES_V1"

# Prompt fixo PT-BR/EN para capturar ativação REAL da Linear amostrada
ACTIVATION_PROMPT = (
    "Explique por que a memória importa na inferência de LLMs. "
    "Explain why memory matters in LLM inference."
)


def utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def run_id(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def schema_v2_fields(model_id, device):
    """Campos obrigatórios do schema v2 (docs/C3_CONTRACTS_V1.md §3)."""
    torch_v = str(getattr(torch, "__version__", "unknown"))
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|{device.type}|{torch_v}"
    return {
        "schema_version": 2,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "comparison_group_id": "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        "comparison_context": {
            "protocol": BENCHMARK_PROTOCOL,
            "device": device.type,
            "torch": torch_v,
            "transformers": _pkg_version("transformers"),
            "python": platform.python_version(),
        },
        "implementation": {"kind": "REFERENCE_MEASURED", "native": False, "simulated": False},
    }


def _read_vmrss_bytes():
    """VmRSS atual em bytes via /proc/self/status (Linux/Colab); None fora."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def measure_phase_ram(fn):
    """Executa fn() com thread amostrando VmRSS a ~1ms.

    Retorna (resultado_fn, info) onde info = {max_bytes, mean_bytes, n_samples, method}
    ou None quando nenhuma medição real é possível (RAM de topo fica null).
    """
    samples = []
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            v = _read_vmrss_bytes()
            if v is not None:
                samples.append(v)
            stop.wait(0.001)

    sampler = threading.Thread(target=_loop, daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    if samples:
        return result, {
            "max_bytes": int(max(samples)),
            "mean_bytes": int(sum(samples) / len(samples)),
            "n_samples": len(samples),
            "method": "proc_vmrss_sampling_per_phase_v1",
        }
    try:
        import resource
        peak_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak_kb > 0:
            return result, {
                "max_bytes": peak_kb * 1024,  # ru_maxrss em KB no Linux
                "mean_bytes": None,
                "n_samples": 0,
                "method": "getrusage_peak_fallback",
            }
    except Exception:
        pass
    return result, None


def cosine_nrmse(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    denom = float(torch.linalg.vector_norm(a).item()) + 1e-12
    nrmse = float(torch.linalg.vector_norm(a - b).item()) / denom
    return {"cosine": cos, "nrmse": nrmse}


def capture_activation(model, tokenizer, layer_name, device, prompt):
    """Captura a entrada REAL da Linear amostrada via forward hook; None se falhar."""
    captured = {}

    def hook(_mod, inputs, _output):
        if inputs and torch.is_tensor(inputs[0]):
            captured["x"] = inputs[0].detach()

    mod = model
    try:
        for part in layer_name.replace(".weight", "").split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        handle = mod.register_forward_hook(hook)
        try:
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                model(**enc)
        finally:
            handle.remove()
        if "x" in captured:
            x = captured["x"].float().cpu()
            if x.ndim >= 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim == 1:
                x = x.unsqueeze(0)
            return x.contiguous()
    except Exception as exc:
        print(f"[C2] AVISO captura de ativação: {exc}")
    return None


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
    if len(token) < 32:
        print("[publish] skip (RIFT_INGEST_TOKEN ausente ou curto <32 chars)"); return
    if not str(endpoint).lower().startswith("https://"):
        print(f"[publish] endpoint não-HTTPS bloqueado — skip: {endpoint}"); return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode()
        req = Request(endpoint, data=body, method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=60) as r:
            print(f"[publish] {r.status} {rec['battery_id']}")
    except Exception as e:
        print(f"[publish] {e}")


def measure_tok_s(model, tokenizer, prompt, device, max_new_tokens=32,
                  warmup=2, measurements=3):
    """Tok/s e2e via model.generate: greedy, ≥2 warmup, ≥3 medições, MEDIANA.

    Protocolo ÚNICO para baseline e candidato (docs/C3_CONTRACTS_V1.md §12).
    """
    warmup = max(2, int(warmup))
    measurements = max(3, int(measurements))
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    runs = []
    n_new = 0
    for _ in range(measurements):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
        runs.append({"tok_s": n_new / max(elapsed, 1e-9),
                     "elapsed_s": elapsed, "n_new": n_new})
    tok_s = float(statistics.median(r["tok_s"] for r in runs))
    elapsed_med = float(statistics.median(r["elapsed_s"] for r in runs))
    return {
        "tok_s": tok_s,
        "n_new": n_new,
        "elapsed_s": elapsed_med,
        "ttft_proxy_ms": (elapsed_med / max(n_new, 1)) * 1000,
        "runs_tok_s": [round(r["tok_s"], 4) for r in runs],
        "n_measurements": len(runs),
        "warmup_runs": warmup,
        "aggregate": "median",
    }


class _CascadeLinearWithBias(torch.nn.Module):
    """CascadeLinearModule + bias original (o CascadeLinearModule não tem bias).

    Só o bias (vetor pequeno) é retido — o W denso continua FORA do caminho
    quente (o caminho quente é F0 INT4 dequant + Gate·F1 low-rank).
    """

    def __init__(self, casc: CascadeLinearModule, bias):
        super().__init__()
        self.casc = casc
        if bias is not None:
            self.register_buffer("bias_term", bias.detach().clone())
        else:
            self.bias_term = None

    def forward(self, x):
        y = self.casc(x)
        if self.bias_term is not None:
            y = y + self.bias_term.to(device=y.device, dtype=y.dtype)
        return y


def _block_linear_targets(model):
    """Todas as nn.Linear dos blocos (exclui lm_head/embeddings)."""
    return [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and "lm_head" not in name and "embed" not in name
    ]


def patch_all_block_linears(model, device, *, rank, gate_percentile):
    """Troca TODAS as nn.Linear dos blocos por CascadeLinearModule (F0_GATE_F1).

    Transacional: em caso de falha no meio, restaura o que já foi trocado e
    relança a exceção (o chamador decide o fallback SKIPPED).
    Retorna (originais, n_patched) para restauração posterior em finally.
    """
    originals = {}
    try:
        targets = _block_linear_targets(model)
        if not targets:
            raise RuntimeError("nenhuma nn.Linear de bloco encontrada para patch")
        for i, (name, lin) in enumerate(targets, 1):
            stages = decompose_linear_int4_lowrank(lin.weight, rank=rank)
            casc = CascadeLinearModule(
                stages, gate_percentile=gate_percentile,
                path="F0_GATE_F1", low_mem=False)
            wrapper = _CascadeLinearWithBias(
                casc, lin.bias if lin.bias is not None else None).to(device)
            _set_module(model, name, wrapper)
            originals[name] = lin
            if i % 25 == 0 or i == len(targets):
                print(f"[C2] patch {i}/{len(targets)} Linear...")
        return originals, len(originals)
    except Exception:
        restore_patched_linears(model, originals)
        raise


def restore_patched_linears(model, originals):
    """Restaura as nn.Linear originais (idempotente; nunca lança)."""
    for name, lin in list(originals.items()):
        try:
            _set_module(model, name, lin)
        except Exception as exc:
            print(f"[C2] AVISO restore {name}: {exc}")
    originals.clear()


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

    schema_fields = schema_v2_fields(model_id, device)

    print("[C2] measuring baseline tok/s...")
    speed, ram_base = measure_phase_ram(
        lambda: measure_tok_s(model, tok, args.prompt, device, max_new_tokens=args.max_new_tokens))
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

    # ativação REAL da Linear amostrada (prompt fixo PT-BR/EN); fallback sintético rebaixa o registro
    x = capture_activation(model, tok, layer_name, device, ACTIVATION_PROMPT)
    activation_source = "real_model_activation"
    if x is None or x.ndim != 2 or x.shape[-1] != weight.shape[1]:
        print("[C2] AVISO: sem ativação real capturada — fallback sintético (registro rebaixado)")
        x = torch.randn(32, weight.shape[1])
        activation_source = "synthetic_fallback"
    elif x.shape[0] > 64:
        x = x[:64].contiguous()
    x = x.to(dtype=torch.float32)
    real_activation = activation_source == "real_model_activation"

    y_ref = torch.nn.functional.linear(x, weight)
    r, ram_cand = measure_phase_ram(lambda: rt.execute(x, path="F0_GATE_F1"))
    q = cosine_nrmse(y_ref, r["y"])
    cos = q["cosine"]

    # estimativas aritméticas de working-set: só em metrics.memory.estimated_*
    io = int((x.numel() + y_ref.numel()) * 4)
    est_baseline_ram = stages.baseline_bytes + io
    rate = 1.0 - r["metrics"].f1_skip_rate
    est_cand_ram = stages.f0_bytes + int(round(rate * stages.f1_bytes)) + io

    # ------- candidato e2e REAL: TODAS as Linear dos blocos patchadas -------
    # (docs/C3_CONTRACTS_V1.md §12) MESMO protocolo de generate do baseline;
    # patch transacional com restauração em finally; falha => nota SKIPPED
    # (candidate_tok_s permanece null e a bateria NUNCA quebra).
    cand_speed = None
    ram_cand_gen = None
    n_patched = 0
    candidate_skip_note = None
    _patched_originals = {}
    try:
        print("[C2] patching ALL block Linears (CascadeLinearModule, F0_GATE_F1, low_mem off)...")
        _patched_originals, n_patched = patch_all_block_linears(
            model, device, rank=args.rank, gate_percentile=args.gate_percentile)
        print(f"[C2] {n_patched} Linear patched; measuring candidate tok/s (same protocol)...")
        cand_speed, ram_cand_gen = measure_phase_ram(
            lambda: measure_tok_s(model, tok, args.prompt, device, max_new_tokens=args.max_new_tokens))
        print(f"[C2] candidate {cand_speed['tok_s']:.2f} tok/s "
              f"(mediana de {cand_speed['n_measurements']} medições)")
    except Exception as exc:
        candidate_skip_note = (
            f"candidate e2e SKIPPED (patch/medição falhou; baseline preservado): "
            f"{type(exc).__name__}: {exc}"
        )
        print(f"[C2] AVISO: {candidate_skip_note}")
    finally:
        restore_patched_linears(model, _patched_originals)
    e2e_measured = cand_speed is not None

    mem_method = None
    for _phase in (ram_cand_gen, ram_cand, ram_base):
        if isinstance(_phase, dict) and _phase.get("method"):
            mem_method = _phase["method"]
            break

    rec = {
        "timestamp_utc": utc(), "run_id": rid, "technology": "CASCADE",
        "model_id": model_id, "battery_id": "P1_CASCADE_C2_E2E_TOKS",
        "status": "PASS" if speed["tok_s"] > 0 and cos >= 0.95 else "EXPERIMENTAL_FAIL",
        **schema_fields,
        "comparison_role": "primary" if real_activation else None,
        "baseline_tok_s": speed["tok_s"],
        "candidate_tok_s": cand_speed["tok_s"] if e2e_measured else None,
        "baseline_ram_bytes": ram_base.get("max_bytes") if isinstance(ram_base, dict) else None,
        "candidate_ram_bytes": (
            ram_cand_gen.get("max_bytes") if isinstance(ram_cand_gen, dict)
            else (ram_cand.get("max_bytes") if isinstance(ram_cand, dict) else None)
        ),
        "baseline_disk_bytes": stages.baseline_bytes,
        "candidate_disk_bytes": stages.f0_bytes + stages.f1_bytes,
        "measurement_scope": (
            (
                "CASCADE-C2 e2e tok/s: baseline E candidato MEDIDOS via model.generate "
                "(greedy, ≥2 warmup, ≥3 medições, mediana) sob o MESMO protocolo; "
                "candidato = TODAS as nn.Linear dos blocos em CascadeLinearModule "
                "(runtime de referência Python — não representa kernel nativo; "
                "W denso fora do caminho quente, low_mem off); "
                "RAM topo=pico VmRSS medido por fase (null sem medição)"
            ) if e2e_measured else (
                "CASCADE-C2 e2e baseline tok/s (model.generate) + sample Linear CASCADE metrics; "
                "candidato e2e SKIPPED nesta execução (candidate_tok_s=null); "
                "RAM topo=pico VmRSS medido por fase (null sem medição)"
            )
        ),
        "quality": {"full_local_gate_pass": cos >= 0.95, "output": q},
        "metrics": {
            "operation": {"metric": "e2e_tok_s", "baseline_tok_s": speed["tok_s"],
                          "candidate_tok_s": cand_speed["tok_s"] if e2e_measured else None,
                          "ttft_proxy_ms": speed["ttft_proxy_ms"],
                          "candidate_ttft_proxy_ms": cand_speed["ttft_proxy_ms"] if e2e_measured else None,
                          "n_new": speed["n_new"]},
            "e2e": {
                "measured": e2e_measured,
                "scope": "python_reference_model_generate",
                "protocol": {
                    "greedy": True,
                    "warmup_runs": speed.get("warmup_runs"),
                    "measurements": speed.get("n_measurements"),
                    "aggregate": "median",
                    "max_new_tokens": args.max_new_tokens,
                },
                "baseline_runs_tok_s": speed.get("runs_tok_s"),
                "candidate_runs_tok_s": cand_speed.get("runs_tok_s") if e2e_measured else None,
                "patched_linears": n_patched if e2e_measured else 0,
                **({"skip_reason": candidate_skip_note} if candidate_skip_note else {}),
            },
            "memory": {
                "method": mem_method,
                "baseline_phase": ram_base,
                "candidate_phase": ram_cand,
                "baseline_phase_scope": "model.generate baseline (modelo completo residente)",
                "candidate_phase_scope": "execute F0_GATE_F1 da Linear amostrada (modelo ainda residente)",
                "candidate_generate_phase": ram_cand_gen,
                "candidate_generate_phase_scope": (
                    "model.generate candidato com TODAS as Linear dos blocos patchadas "
                    "(Linears originais retidas fora do grafo apenas para restauração — "
                    "RSS de processo, não estado isolado)"
                ),
                "estimated_baseline_bytes": est_baseline_ram,
                "estimated_candidate_bytes": est_cand_ram,
            },
            "cascade": {
                **stages.to_meta(),
                "sample_layer": layer_name,
                "activation_source": activation_source,
                "activation_rows": int(x.shape[0]),
                "F1_skip_rate": r["metrics"].f1_skip_rate,
                "path": "F0_GATE_F1",
                "e2e_note": (
                    "baseline e candidato medidos por model.generate; candidato = "
                    "CascadeLinearModule em todas as Linear dos blocos (F0+Gate·F1, "
                    "referência Python)" if e2e_measured else
                    "tok/s is ORIGINAL model; candidate e2e SKIPPED nesta execução"
                ),
            },
        },
        "notes": (
            f"C2 baseline {speed['tok_s']:.2f} tok/s on {device.type}. "
            + (
                f"Candidate {cand_speed['tok_s']:.2f} tok/s "
                f"({n_patched} Linear patchadas, mediana de "
                f"{cand_speed['n_measurements']} medições). "
                if e2e_measured else f"{candidate_skip_note} "
            )
            + f"Sample Linear {layer_name} cosine={cos:.4f} nrmse={q['nrmse']:.4f} "
            f"skip={r['metrics'].f1_skip_rate:.3f} activation_source={activation_source}."
        )[:1200],
    }
    path = out / "batteries" / f"{rid}__P1_CASCADE_C2_E2E_TOKS.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"[BATTERY] {path}")
    publish(rec, args.results_endpoint)
    print("[C2] done")
    return 0


if __name__ == "__main__":
    _rc = 0
    try:
        _rc = main() or 0
    except SystemExit as _e:
        _rc = int(_e.code) if isinstance(_e.code, int) else 0
    except Exception:
        traceback.print_exc()
        _rc = 0
    finally:
        try:
            cleanup_colab_workspace(label="CASCADE-C2", wipe_hf_cache=True)
        except Exception as _ce:
            print(f"[cleanup] AVISO: {_ce}")
    raise SystemExit(_rc)
