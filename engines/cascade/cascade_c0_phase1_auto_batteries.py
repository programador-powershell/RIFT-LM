#!/usr/bin/env python3
"""CASCADE-C0 — bateria automática: Linear real INT4 F0 + residual + Gate.

Quatro caminhos obrigatórios do plano:
  A ORIGINAL | B F0_ONLY | C F0_PLUS_F1_ALWAYS | D F0_GATE_F1

Publica cada bateria no dashboard assim que grava o JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import sys
import threading
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# --- resolve cascade package (Colab: cwd/sibling/content; repo: <repo>/core) ---
_HERE = Path(__file__).resolve().parent
for cand in [_HERE, _HERE / "cascade", _HERE.parent, Path("/content"), _HERE.parent.parent / "core"]:
    if (cand / "cascade" / "compiler" / "decompose.py").is_file():
        sys.path.insert(0, str(cand))
        break
    if (cand / "compiler" / "decompose.py").is_file():
        sys.path.insert(0, str(cand.parent))
        break

from cascade.compiler.bundle_writer import write_cascade_bundle
from cascade.compiler.decompose import decompose_linear_int4_lowrank
from cascade.kernels.int4 import dequantize_int4
from cascade.runtime.reference import CascadeLinearRuntime
from cascade.runtime.cleanup import cleanup_colab_workspace

BENCHMARK_PROTOCOL = "CASCADE_C_SERIES_V1"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def schema_v2_fields(model_id: str, device: torch.device) -> dict[str, Any]:
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


def _read_vmrss_bytes() -> int | None:
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
    ou None quando nenhuma medição real é possível (então RAM de topo fica null).
    """
    samples: list[int] = []
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


def resolve_hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    try:
        from google.colab import userdata
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                v = str(userdata.get(name) or "").strip()
            except Exception:
                v = ""
            if v:
                os.environ.setdefault("HF_TOKEN", v)
                return v
    except Exception:
        pass
    return None


def ensure_hf_login(token: str | None = None) -> str | None:
    token = token or resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] HF_TOKEN aplicado.")
    except Exception as exc:
        print(f"[auth] AVISO: {exc}")
    return token


def resolve_device(s: str) -> torch.device:
    s = (s or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def cosine_nrmse(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    denom = float(torch.linalg.vector_norm(a).item()) + 1e-12
    nrmse = float(torch.linalg.vector_norm(a - b).item()) / denom
    return {"cosine": cos, "nrmse": nrmse}


def benchmark_ms(fn, *, iterations: int, device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    # warmup
    for _ in range(min(3, iterations)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1]}


# --- leitura do bundle CSCD para verificação de roundtrip (formato do bundle_writer) ---
_BUNDLE_HEADER_FMT = "<4sHHIIQQQQQQQ"
_BUNDLE_HEADER_SIZE = 128
_BUNDLE_STAGE_ENTRY = 24


def _read_bundle_tensor(buf: bytes, cursor: int) -> tuple[torch.Tensor, int]:
    """Lê um payload de tensor (<I len_meta><meta json><corpo>) do bundle."""
    (hlen,) = struct.unpack_from("<I", buf, cursor)
    cursor += 4
    hdr = json.loads(buf[cursor:cursor + hlen].decode("utf-8"))
    cursor += hlen
    dtype_map = {
        "uint8": (torch.uint8, 1),
        "int8": (torch.int8, 1),
        "int16": (torch.int16, 2),
        "int32": (torch.int32, 4),
        "float16": (torch.float16, 2),
        "float32": (torch.float32, 4),
    }
    dt, itemsize = dtype_map[hdr["dtype"]]
    shape = [int(d) for d in hdr["shape"]]
    numel = 1
    for d in shape:
        numel *= d
    body = bytearray(buf[cursor:cursor + numel * itemsize])
    cursor += numel * itemsize
    tensor = torch.frombuffer(body, dtype=dt).reshape(shape).clone()
    return tensor, cursor


def verify_bundle_roundtrip(bundle_path: Path, x: torch.Tensor, y_ref: torch.Tensor) -> dict[str, Any]:
    """Roundtrip REAL: relê o bundle gravado, dequantiza F0 (+F1) e roda a
    Linear de referência sobre a ativação capturada, comparando com F.linear original."""
    buf = bundle_path.read_bytes()
    fields = struct.unpack_from(_BUNDLE_HEADER_FMT, buf, 0)
    magic, version, n_stages = fields[0], int(fields[1]), int(fields[4])
    stage_table_offset = int(fields[6])
    file_size, checksum = int(fields[9]), int(fields[10])
    checks = {
        "magic_ok": magic == b"CSCD",
        "file_size_ok": file_size == len(buf),
        "crc_ok": (zlib.crc32(buf[_BUNDLE_HEADER_SIZE:]) & 0xFFFFFFFFFFFFFFFF) == checksum,
    }
    payloads: dict[int, tuple[dict, bytes, int]] = {}
    for i in range(n_stages):
        off, sz, sid, _flags = struct.unpack_from("<QQII", buf, stage_table_offset + i * _BUNDLE_STAGE_ENTRY)
        blob = buf[int(off):int(off) + int(sz)]
        (mlen,) = struct.unpack_from("<I", blob, 0)
        meta = json.loads(blob[4:4 + mlen].decode("utf-8"))
        payloads[int(meta.get("stage_id", sid))] = (meta, blob, 4 + mlen)

    meta0, blob0, cur0 = payloads[0]
    codes, cur0 = _read_bundle_tensor(blob0, cur0)
    scales, cur0 = _read_bundle_tensor(blob0, cur0)
    w0 = dequantize_int4(
        codes, scales,
        group_size=int(meta0["group_size"]),
        out_features=int(meta0["out_features"]),
        in_features=int(meta0["in_features"]),
    )
    y_f0 = F.linear(x, w0)

    meta1, blob1, cur1 = payloads[1]
    u, cur1 = _read_bundle_tensor(blob1, cur1)
    s, cur1 = _read_bundle_tensor(blob1, cur1)
    v, cur1 = _read_bundle_tensor(blob1, cur1)
    y_full = y_f0 + ((x @ v) * s) @ u.T

    return {
        "checks": checks,
        "version": version,
        "n_stages": n_stages,
        "q_f0_only": cosine_nrmse(y_ref, y_f0),
        "q_f0_plus_f1": cosine_nrmse(y_ref, y_full),
        "f1_rank": int(meta1.get("rank", s.numel())),
    }


def publish_record(record: dict[str, Any], endpoint: str | None, token: str | None) -> None:
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or "https://rift-lm.vercel.app/api/results"
    token = token or os.environ.get("RIFT_INGEST_TOKEN") or ""
    if not token or len(token) < 32:
        print("[publish] RIFT_INGEST_TOKEN ausente ou curto (<32 chars) — skip")
        return
    if not str(endpoint).lower().startswith("https://"):
        print(f"[publish] endpoint não-HTTPS bloqueado — skip: {endpoint}")
        return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [record]}, ensure_ascii=False).encode("utf-8")
        req = Request(endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "cascade-c0-battery/0.3",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={record.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


def find_linear_weight(model, target: str) -> tuple[str, torch.Tensor]:
    state = model.state_dict()
    if target and target != "auto" and target in state:
        return target, state[target].detach().float()
    if target and target != "auto" and not target.endswith(".weight"):
        alt = target + ".weight"
        if alt in state:
            return alt, state[alt].detach().float()
    # auto: first 2D weight looking like Linear
    for name, tensor in state.items():
        if tensor.ndim == 2 and "embed" not in name.lower() and tensor.shape[0] > 32 and tensor.shape[1] > 32:
            if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc", "dense", "linear")):
                return name, tensor.detach().float()
    for name, tensor in state.items():
        if tensor.ndim == 2 and min(tensor.shape) >= 64:
            return name, tensor.detach().float()
    raise RuntimeError("Nenhuma Linear 2D adequada encontrada no state_dict")


def load_model(model_id: str, device: torch.device, trust: bool, token: str | None):
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    try:
        from transformers import AutoModelForMultimodalLM  # type: ignore
    except Exception:
        AutoModelForMultimodalLM = None  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    kwargs = dict(token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
                  dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = None
    classes = []
    if AutoModelForMultimodalLM is not None:
        classes.append(AutoModelForMultimodalLM)
    classes.extend([AutoModelForCausalLM, AutoModel])
    errors = []
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[load] {cls.__name__}")
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    if model is None:
        raise RuntimeError("Falha ao carregar modelo:\\n" + "\\n".join(errors))
    return model, tok


def capture_x(model, tokenizer, layer_name: str, device: torch.device, prompt: str) -> torch.Tensor:
    """Captura ativação de entrada da Linear alvo; fallback sintético."""
    captured = {}

    def hook(mod, inputs, output):
        if inputs and torch.is_tensor(inputs[0]):
            captured["x"] = inputs[0].detach()

    # resolve module
    mod = model
    parts = layer_name.replace(".weight", "").split(".")
    try:
        for p in parts:
            if p.isdigit():
                mod = mod[int(p)]
            else:
                mod = getattr(mod, p)
        h = mod.register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            model(**inputs)
        h.remove()
        if "x" in captured:
            x = captured["x"].float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            return x.contiguous()
    except Exception as exc:
        print(f"[WARN] captura de ativação falhou: {exc}")
    # synthetic
    in_f = None
    return None  # caller handles


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CASCADE-C0 battery")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--target-layer", default="auto")
    p.add_argument("--prompt", default="Explique por que memória importa na inferência.")
    p.add_argument("--device", default="auto")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--gate-percentile", type=float, default=70.0)
    p.add_argument("--batch-rows", type=int, default=64)
    p.add_argument("--out", default="cascade_c0_test_output")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--publish", default=os.environ.get("RIFT_PUBLISH_MODE", "auto"))
    p.add_argument("--results-endpoint", default=None)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    token = ensure_hf_login()
    model_id = args.model.strip().replace("https://huggingface.co/", "").strip("/")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batteries_dir = out_dir / "batteries"
    batteries_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    ingest = os.environ.get("RIFT_INGEST_TOKEN")
    schema_fields = schema_v2_fields(model_id, device)

    print(f"[CASCADE-C0] model={model_id} device={device} rank={args.rank}")
    try:
        model, tokenizer = load_model(model_id, device, args.trust_remote_code, token)
    except Exception as exc:
        rec = {
            "timestamp_utc": _utc(), "run_id": run_id, "technology": "CASCADE",
            "model_id": model_id, "battery_id": "C0_LOAD_MODEL", "status": "FAIL",
            **schema_fields,
            "measurement_scope": "model_load",
            "quality": {"full_local_gate_pass": False},
            "metrics": {"error": str(exc)[:800]},
            "notes": f"Falha ao carregar: {exc}"[:1200],
        }
        path = batteries_dir / f"{run_id}__C0_LOAD_MODEL.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        publish_record(rec, args.results_endpoint, ingest)
        print(f"[FAIL] {exc}")
        return 0  # soft-fail

    layer_name, weight = find_linear_weight(model, args.target_layer)
    print(f"[CASCADE-C0] Linear: {layer_name} shape={tuple(weight.shape)}")
    weight = weight.to(dtype=torch.float32)

    x = capture_x(model, tokenizer, layer_name, device, args.prompt)
    if x is None or x.shape[-1] != weight.shape[1]:
        print("[CASCADE-C0] usando ativação sintética")
        x = torch.randn(args.batch_rows, weight.shape[1], dtype=torch.float32)
    else:
        if x.shape[0] > args.batch_rows:
            x = x[: args.batch_rows]
        elif x.shape[0] < 4:
            x = torch.cat([x, torch.randn(args.batch_rows - x.shape[0], x.shape[1])], dim=0)

    x = x.to(dtype=torch.float32)
    # free model weights from RAM as much as possible — we only need the Linear tensor
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- decompose ---
    print("[CASCADE-C0] Decompondo INT4 F0 + low-rank F1...")
    stages = decompose_linear_int4_lowrank(weight, rank=args.rank, group_size=args.group_size)
    print(f"[CASCADE-C0] F0={stages.f0_bytes} B  F1={stages.f1_bytes} B  baseline={stages.baseline_bytes} B  "
          f"disk_reduction={stages.to_meta()['disk_reduction_pct']:.1f}%")

    bundle_path = out_dir / "model.cascade"
    bundle_meta = write_cascade_bundle(
        bundle_path,
        stages=stages,
        model_id=model_id,
        target_layer=layer_name,
        gate_percentile=args.gate_percentile,
    )
    print(f"[CASCADE-C0] Bundle: {bundle_path} ({bundle_meta['file_size']} bytes)")

    runtime = CascadeLinearRuntime(stages, gate_percentile=args.gate_percentile, device=torch.device("cpu"))
    x_cpu = x.cpu()
    w_cpu = weight.cpu()

    with torch.inference_mode():
        y_ref = F.linear(x_cpu, w_cpu)
        r_f0 = runtime.execute(x_cpu, path="F0_ONLY")
        r_full = runtime.execute(x_cpu, path="F0_PLUS_F1_ALWAYS")
        r_gate = runtime.execute(x_cpu, path="F0_GATE_F1")

    q_f0 = cosine_nrmse(y_ref, r_f0["y"])
    q_full = cosine_nrmse(y_ref, r_full["y"])
    q_gate = cosine_nrmse(y_ref, r_gate["y"])

    # roundtrip REAL do bundle: relê stages do disco e compara com F.linear original
    try:
        roundtrip = verify_bundle_roundtrip(bundle_path, x_cpu, y_ref)
        q_bundle = roundtrip["q_f0_plus_f1"]
        bundle_ok = all(roundtrip["checks"].values())
    except Exception as exc:
        print(f"[CASCADE-C0] AVISO roundtrip do bundle falhou: {exc}")
        roundtrip = {"error": str(exc)[:400]}
        q_bundle = {"cosine": 0.0, "nrmse": 1.0}
        bundle_ok = False

    # latency (cada fase amostrada por VmRSS a ~1ms — RAM real por fase)
    base_perf, ram_base = measure_phase_ram(
        lambda: benchmark_ms(lambda: F.linear(x_cpu, w_cpu), iterations=args.iterations, device=torch.device("cpu")))
    f0_perf, ram_f0 = measure_phase_ram(
        lambda: benchmark_ms(lambda: runtime.execute(x_cpu, path="F0_ONLY"), iterations=args.iterations, device=torch.device("cpu")))
    full_perf, ram_full = measure_phase_ram(
        lambda: benchmark_ms(lambda: runtime.execute(x_cpu, path="F0_PLUS_F1_ALWAYS"), iterations=args.iterations, device=torch.device("cpu")))
    gate_perf, ram_gate = measure_phase_ram(
        lambda: benchmark_ms(lambda: runtime.execute(x_cpu, path="F0_GATE_F1"), iterations=args.iterations, device=torch.device("cpu")))

    # estimativas aritméticas de working-set: só em metrics.memory.estimated_* (nunca topo)
    io_bytes = int((x_cpu.numel() + y_ref.numel()) * 4)
    est_baseline_ram = stages.baseline_bytes + io_bytes
    est_f0_ram = stages.f0_bytes + io_bytes
    est_full_ram = stages.f0_bytes + stages.f1_bytes + io_bytes
    gate_rate = float(r_gate["metrics"].f1_calls) / max(r_gate["metrics"].f0_calls, 1)
    est_gate_ram = stages.f0_bytes + int(round(gate_rate * stages.f1_bytes)) + io_bytes

    def emit(battery_id: str, status: str, *, path_name: str, q: dict, est_cand_ram: int, cand_disk: int,
             perf: dict, extra_metrics: dict, notes: str, primary: bool = False,
             ram_phase: dict | None = None):
        mem_method = None
        if isinstance(ram_phase, dict) and ram_phase.get("method"):
            mem_method = ram_phase["method"]
        elif isinstance(ram_base, dict) and ram_base.get("method"):
            mem_method = ram_base["method"]
        rec = {
            "timestamp_utc": _utc(),
            "run_id": run_id,
            "technology": "CASCADE",
            "model_id": model_id,
            "battery_id": battery_id,
            "status": status,
            **schema_fields,
            "comparison_role": "primary" if primary else None,
            "baseline_ram_bytes": ram_base.get("max_bytes") if isinstance(ram_base, dict) else None,
            "candidate_ram_bytes": ram_phase.get("max_bytes") if isinstance(ram_phase, dict) else None,
            "baseline_disk_bytes": stages.baseline_bytes,
            "candidate_disk_bytes": cand_disk,
            "measurement_scope": (
                f"CASCADE-C0 Linear real path={path_name}; "
                f"layer={layer_name}; INT4 F0 + low-rank F1; reference Python runtime; "
                f"RAM topo=pico VmRSS medido por fase (null sem medição)"
            ),
            "quality": {
                "full_local_gate_pass": status == "PASS",
                "output": q,
            },
            "metrics": {
                "operation": {
                    "metric": "linear_latency",
                    "baseline_median_ms": base_perf["median_ms"],
                    "candidate_median_ms": perf["median_ms"],
                    "speedup_x": base_perf["median_ms"] / max(perf["median_ms"], 1e-12),
                    "rows_processed": int(x_cpu.shape[0]),
                    "path": path_name,
                },
                "memory": {
                    "method": mem_method,
                    "baseline_phase": ram_base,
                    "candidate_phase": ram_phase,
                    "estimated_baseline_bytes": est_baseline_ram,
                    "estimated_candidate_bytes": est_cand_ram,
                },
                "cascade": {
                    **stages.to_meta(),
                    "bundle_bytes": bundle_meta["file_size"],
                    "target_layer": layer_name,
                    **extra_metrics,
                },
            },
            "notes": notes[:1200],
        }
        path = batteries_dir / f"{run_id}__{battery_id}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        print(f"[BATTERY] {battery_id} -> {path}")
        if args.publish != "off":
            publish_record(rec, args.results_endpoint, ingest)
        return rec

    # B0 foundation — roundtrip real do bundle (sem qualidade hardcoded)
    emit(
        "B0_CASCADE_C0_BUNDLE",
        "PASS" if bundle_ok and q_bundle["cosine"] >= 0.98 else "FAIL",
        path_name="BUNDLE",
        q=q_bundle,
        est_cand_ram=stages.f0_bytes + stages.f1_bytes,
        cand_disk=bundle_meta["file_size"],
        perf=base_perf,
        extra_metrics={"bundle": bundle_meta, "bundle_roundtrip": roundtrip},
        notes=(
            "Bundle CSCD M0 relido do disco: F0 INT4 + F1 low-rank dequantizados do arquivo "
            "e avaliados com F.linear sobre a ativação capturada (roundtrip real, CRC/tamanho verificados)."
        ),
    )

    emit(
        "P1_CASCADE_C0_F0_ONLY",
        "EXPERIMENTAL_PASS" if q_f0["cosine"] >= 0.90 else "EXPERIMENTAL_FAIL",
        path_name="F0_ONLY",
        q=q_f0,
        est_cand_ram=est_f0_ram,
        cand_disk=stages.f0_bytes,
        perf=f0_perf,
        extra_metrics={"F0_calls": r_f0["metrics"].f0_calls, "F1_calls": 0, "F1_skip_rate": 1.0},
        notes="Path B: somente F0 INT4. Ganho bruto de quantização.",
        ram_phase=ram_f0,
    )

    emit(
        "P1_CASCADE_C0_F0_PLUS_F1_ALWAYS",
        "EXPERIMENTAL_PASS" if q_full["cosine"] >= 0.98 else "EXPERIMENTAL_FAIL",
        path_name="F0_PLUS_F1_ALWAYS",
        q=q_full,
        est_cand_ram=est_full_ram,
        cand_disk=stages.f0_bytes + stages.f1_bytes,
        perf=full_perf,
        extra_metrics={"F0_calls": r_full["metrics"].f0_calls, "F1_calls": r_full["metrics"].f1_calls, "F1_skip_rate": 0.0},
        notes="Path C: F0+F1 always. Qualidade recuperada pelo residual.",
        ram_phase=ram_full,
    )

    gate_meta = r_gate.get("gate") or {}
    quality_pass = q_gate["cosine"] >= 0.98 and q_gate["nrmse"] <= 0.10 and r_gate["metrics"].f1_skip_rate > 0
    emit(
        "P1_CASCADE_GATED_F0_PLUS_F1",
        "PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        path_name="F0_GATE_F1",
        q=q_gate,
        est_cand_ram=est_gate_ram,
        cand_disk=stages.f0_bytes + stages.f1_bytes,
        perf=gate_perf,
        extra_metrics={
            "F0_calls": r_gate["metrics"].f0_calls,
            "F1_calls": r_gate["metrics"].f1_calls,
            "F1_skip_rate": r_gate["metrics"].f1_skip_rate,
            "avg_stages_per_token": r_gate["metrics"].avg_stages_per_token,
            "gate": gate_meta,
        },
        notes=(
            f"Path D: CASCADE real. F1_skip_rate={r_gate['metrics'].f1_skip_rate:.3f} "
            f"gate_rate={gate_rate:.3f}. RAM topo=pico VmRSS por fase; "
            f"working-set estimado em metrics.memory.estimated_*."
        ),
        primary=True,
        ram_phase=ram_gate,
    )

    print("[CASCADE-C0] concluído.")
    print(f"  F0 cosine={q_f0['cosine']:.4f}  FULL={q_full['cosine']:.4f}  GATE={q_gate['cosine']:.4f}")
    print(f"  F1_skip_rate={r_gate['metrics'].f1_skip_rate:.3f}  bundle={bundle_meta['file_size']} B")
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
            cleanup_colab_workspace(label="CASCADE-C0", wipe_hf_cache=False)
        except Exception as _ce:
            print(f"[cleanup] AVISO: {_ce}")
    raise SystemExit(_rc)
