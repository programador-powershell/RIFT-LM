#!/usr/bin/env python3
"""
RIFT-LM Real Benchmark Runner v1.

Runs an existing technology battery with publishing disabled, instruments its
benchmark_ms() at runtime, removes estimated metrics from comparison fields,
verifies physical storage where possible, and publishes sanitized records.

No end-to-end Tok/s is fabricated. Until a technology exposes a full-model
candidate runtime, Tok/s remains null.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "programador-powershell/RIFT-LM"
DEFAULT_RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results"
BENCHMARK_PROTOCOL = "LINEAR_REAL_MEASURED_V3"
SCHEMA_VERSION = 2

TECHNOLOGIES = {
    "rift": {
        "label": "RIFT",
        "script": "rift_m0_phase1_test_v035_auto_batteries.py",
        "args": ["--mode", "phase1"],
        "primary": "P1_Q4_LINEAR_BASE_PLUS_REF_4BIT",
        "probe_labels": ["baseline", "base_predecoded", "full_predecoded", "base_reference", "candidate"],
    },
    "cascade": {
        "label": "CASCADE",
        "script": "cascade_m0_phase1_test_v030_auto_batteries.py",
        "args": [],
        "primary": "P1_CASCADE_GATED_F0_PLUS_F1",
        "probe_labels": ["baseline", "f0", "candidate"],
    },
    "aether": {
        "label": "AETHER",
        "script": "aether_m0_phase1_test_v100_auto_batteries.py",
        "args": ["--mode", "phase1"],
        "primary": "P1_AETHER_HQR_PLUS_TADDS_DYNAMIC",
        "probe_labels": ["baseline", "f0", "candidate"],
    },
    "spectra": {
        "label": "SPECTRA",
        "script": "SPECTRA_Colab_Test_M0.py",
        "args": ["--mode", "phase1"],
        "primary": "P1_SPECTRA_HQR_PLUS_TADDS_DYNAMIC",
        "probe_labels": ["baseline", "f0", "candidate"],
    },
    "winner": {
        "label": "WINNER",
        "script": "winner_m0_phase1_test_v080_auto_batteries.py",
        "args": ["--mode", "phase1"],
        "primary": "P1_WINNER_F0_PLUS_LS",
        "probe_labels": ["baseline", "f0", "candidate", "full"],
    },
}

REAL_BENCHMARK_PATCH = r"""
# ===== injected by RIFT-LM Real Benchmark Runner =====
import json as _rb_json
import math as _rb_math
import os as _rb_os
import statistics as _rb_statistics
import threading as _rb_threading
import time as _rb_time

_rb_benchmark_call_index = 0

def _rb_rss_bytes():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) * 1024
    except Exception:
        pass
    return None

def _rb_percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(_rb_math.floor(pos))
    hi = int(_rb_math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

def _rb_append_probe(payload):
    path = _rb_os.environ.get("RIFT_REAL_PROBE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as _f:
            _f.write(_rb_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as _exc:
        print("[REAL-METRICS] aviso ao gravar probe:", _exc)

def benchmark_ms(fn, *, device, iterations=20, warmup=None, **_kwargs):
    global _rb_benchmark_call_index
    _rb_benchmark_call_index += 1
    call_index = _rb_benchmark_call_index

    requested_iterations = int(_rb_os.environ.get("RIFT_REAL_ITERATIONS", "50"))
    requested_warmup = int(_rb_os.environ.get("RIFT_REAL_WARMUP", "10"))
    iterations = max(int(iterations), requested_iterations)
    warmup = max(int(warmup or 0), requested_warmup)

    def _sync():
        try:
            if getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
        except Exception:
            pass

    for _ in range(warmup):
        fn()
    _sync()

    rss_before = _rb_rss_bytes()
    rss_peak = rss_before
    _stop = _rb_threading.Event()

    def _sample_rss():
        nonlocal rss_peak
        while not _stop.is_set():
            value = _rb_rss_bytes()
            if value is not None and (rss_peak is None or value > rss_peak):
                rss_peak = value
            _stop.wait(0.001)

    _sampler = _rb_threading.Thread(target=_sample_rss, daemon=True)

    gpu_before_allocated = None
    gpu_before_reserved = None
    try:
        if getattr(device, "type", "") == "cuda":
            _sync()
            torch.cuda.reset_peak_memory_stats(device)
            gpu_before_allocated = int(torch.cuda.memory_allocated(device))
            gpu_before_reserved = int(torch.cuda.memory_reserved(device))
    except Exception:
        pass

    values = []
    _sampler.start()
    try:
        for _ in range(iterations):
            _sync()
            t0 = _rb_time.perf_counter_ns()
            fn()
            _sync()
            values.append((_rb_time.perf_counter_ns() - t0) / 1e6)
    finally:
        _stop.set()
        _sampler.join(timeout=1.0)
        _sync()

    gpu_peak_allocated = None
    gpu_peak_reserved = None
    gpu_current_allocated = None
    gpu_current_reserved = None
    try:
        if getattr(device, "type", "") == "cuda":
            gpu_peak_allocated = int(torch.cuda.max_memory_allocated(device))
            gpu_peak_reserved = int(torch.cuda.max_memory_reserved(device))
            gpu_current_allocated = int(torch.cuda.memory_allocated(device))
            gpu_current_reserved = int(torch.cuda.memory_reserved(device))
    except Exception:
        pass

    rss_after = _rb_rss_bytes()
    if rss_after is not None and (rss_peak is None or rss_after > rss_peak):
        rss_peak = rss_after

    result = {
        "median_ms": float(_rb_statistics.median(values)),
        "mean_ms": float(_rb_statistics.mean(values)),
        "p95_ms": _rb_percentile(values, 0.95),
        "p99_ms": _rb_percentile(values, 0.99),
        "minimum_ms": float(min(values)),
        "min_ms": float(min(values)),
        "maximum_ms": float(max(values)),
        "max_ms": float(max(values)),
        "stddev_ms": float(_rb_statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "iterations": iterations,
        "warmup": warmup,
        "measurement_method": "perf_counter_ns_with_cuda_sync_v1",
        "memory": {
            "scope": "operation_transient_and_process_absolute",
            "measurement_method": "proc_rss_1ms_and_torch_cuda_peak_v1",
            "cpu_rss_before_bytes": rss_before,
            "cpu_rss_peak_bytes": rss_peak,
            "cpu_rss_after_bytes": rss_after,
            "cpu_rss_delta_peak_bytes": (
                max(0, int(rss_peak) - int(rss_before))
                if rss_peak is not None and rss_before is not None else None
            ),
            "gpu_allocated_before_bytes": gpu_before_allocated,
            "gpu_peak_allocated_bytes": gpu_peak_allocated,
            "gpu_current_allocated_bytes": gpu_current_allocated,
            "gpu_delta_peak_allocated_bytes": (
                max(0, gpu_peak_allocated - gpu_before_allocated)
                if gpu_peak_allocated is not None and gpu_before_allocated is not None else None
            ),
            "gpu_reserved_before_bytes": gpu_before_reserved,
            "gpu_peak_reserved_bytes": gpu_peak_reserved,
            "gpu_current_reserved_bytes": gpu_current_reserved,
            "gpu_delta_peak_reserved_bytes": (
                max(0, gpu_peak_reserved - gpu_before_reserved)
                if gpu_peak_reserved is not None and gpu_before_reserved is not None else None
            ),
        },
    }
    _rb_append_probe({"call_index": call_index, **result})
    return result
# ===== end injected real benchmark =====
"""

def normalize_model_id(value: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith("https://huggingface.co/"):
        parts = [p for p in raw.split("huggingface.co/", 1)[1].split("/") if p]
        raw = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        raise ValueError("model deve usar org/modelo ou URL huggingface.co")
    return raw

def resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "gpu":
        value = "auto"
    if value in {"cpu", "cuda"}:
        return value
    try:
        subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        return "cuda"
    except Exception:
        return "cpu"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def file_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append({
            "path": str(path.relative_to(root)),
            "bytes": int(size),
            "sha256": sha256_file(path) if size <= 128 * 1024 * 1024 else None,
        })
    return rows

def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

def gpu_descriptor() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        rows = [line.strip() for line in output.splitlines() if line.strip()]
        return rows[0] if rows else None
    except Exception:
        return None

def process_tree_pids(root_pid: int) -> set[int]:
    found = {root_pid}
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = [int(v) for v in path.read_text().split()]
        except Exception:
            children = []
        for child in children:
            if child not in found:
                found.add(child)
                queue.append(child)
    return found

def rss_for_pid(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0

def gpu_memory_for_pids(pids: set[int]) -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
    except Exception:
        return None
    total = 0
    seen = False
    for line in output.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid in pids:
            seen = True
            total += int(mib * 1024 * 1024)
    return total if seen else 0

def instrument_source(source: str) -> str:
    matches = re.findall(r"(?m)^def benchmark_ms\(", source)
    if len(matches) != 1:
        raise RuntimeError(f"battery precisa possuir exatamente um benchmark_ms; encontrados={len(matches)}")
    replaced = re.sub(r"(?m)^def benchmark_ms\(", "def _battery_original_benchmark_ms(", source, count=1)
    for marker in ('\nif __name__ == "__main__":\n    main()', "\nif __name__ == '__main__':\n    main()"):
        if marker in replaced:
            return replaced.replace(marker, "\n" + REAL_BENCHMARK_PATCH + marker, 1)
    raise RuntimeError("entrypoint __main__ não encontrado para injetar benchmark real")

def download_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "rift-real-benchmark/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")

def load_records(root: Path) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "").strip()
            battery_id = str(item.get("battery_id") or "").strip()
            if run_id and battery_id:
                records[(run_id, battery_id)] = item
    return list(records.values())

def load_probe_log(path: Path, technology: str) -> dict[str, dict[str, Any]]:
    labels = TECHNOLOGIES[technology]["probe_labels"]
    probes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                probes.append(value)
    return {labels[i]: p for i, p in enumerate(probes) if i < len(labels)}

def resolve_target_layer(root: Path, requested: str) -> str | None:
    for path in sorted(root.glob("*_gain_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(report, dict):
            value = report.get("target_layer") or report.get("tensor")
            if value:
                return str(value)
    return None if requested == "auto" else requested

def activation_is_real(root: Path, records: list[dict[str, Any]]) -> tuple[bool, str]:
    values: list[str] = []
    for path in sorted(root.glob("*_gain_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(report, dict):
            if report.get("activation_source"):
                values.append(str(report["activation_source"]))
            for block in ("cascade", "aether", "spectra", "winner"):
                data = report.get(block)
                if isinstance(data, dict) and data.get("activation_source"):
                    values.append(str(data["activation_source"]))
    for record in records:
        metrics = record.get("metrics") or {}
        if isinstance(metrics, dict):
            for block in ("cascade", "aether", "spectra", "winner"):
                data = metrics.get(block)
                if isinstance(data, dict) and data.get("activation_source"):
                    values.append(str(data["activation_source"]))
    if any("synthetic" in v.lower() for v in values):
        return False, "synthetic_fallback"
    if any("real_model_activation" in v.lower() for v in values):
        return True, "real_model_activation"
    return False, "activation_provenance_unknown"

def find_cached_snapshot(model_id: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        value = snapshot_download(repo_id=model_id, local_files_only=True, token=token)
        path = Path(value)
        if path.exists():
            return path
    except Exception:
        pass
    snapshots = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / ("models--" + model_id.replace("/", "--")) / "snapshots"
    )
    if snapshots.is_dir():
        dirs = [p for p in snapshots.iterdir() if p.is_dir()]
        if dirs:
            return max(dirs, key=lambda p: p.stat().st_mtime)
    return None

def safetensor_tensor_bytes(model_id: str, tensor_name: str | None) -> tuple[int | None, str | None]:
    if not tensor_name:
        return None, None
    root = find_cached_snapshot(model_id)
    if root is None:
        return None, None
    shard = None
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        try:
            shard_name = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {}).get(tensor_name)
            if shard_name:
                shard = root / shard_name
        except Exception:
            pass
    candidates = [shard] if shard else sorted(root.glob("*.safetensors"))
    for path in [p for p in candidates if p and p.exists()]:
        try:
            with path.open("rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    continue
                header_len = struct.unpack("<Q", raw)[0]
                header = json.loads(f.read(header_len))
            meta = header.get(tensor_name)
            if isinstance(meta, dict):
                start, end = meta["data_offsets"]
                return int(end - start), path.name
        except Exception:
            continue
    return None, None

def comparison_context(model_id: str, target_layer: str | None, device: str, iterations: int, warmup: int, source_ref: str):
    root = find_cached_snapshot(model_id)
    context = {
        "protocol": BENCHMARK_PROTOCOL,
        "model_id": model_id,
        "model_revision": root.name if root is not None else None,
        "target_layer_resolved": target_layer,
        "device": device,
        "gpu": gpu_descriptor(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "iterations": iterations,
        "warmup": warmup,
        "source_ref": source_ref,
    }
    fingerprint = {k: v for k, v in context.items() if k != "source_ref"}
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return context, "cmp-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

def verify_candidate_disk(record: dict[str, Any], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    value = record.get("candidate_disk_bytes")
    if value is None:
        for key in ("rift_disk_bytes", "cascade_disk_bytes", "aether_disk_bytes", "spectra_disk_bytes", "winner_disk_bytes"):
            if record.get(key) is not None:
                value = record[key]
                break
    try:
        candidate = int(value) if value is not None else None
    except Exception:
        candidate = None
    exact = [row["path"] for row in artifacts if candidate is not None and row["bytes"] == candidate]
    return {
        "candidate_reported_bytes": candidate,
        "candidate_exact_file_match": bool(exact),
        "matching_files": exact[:8],
        "artifact_count": len(artifacts),
        "artifact_total_bytes": int(sum(row["bytes"] for row in artifacts)),
        "measurement_method": "os_stat_and_sha256_manifest_v1",
    }

def sanitize_records(records, *, technology, model_id, target_layer, device, iterations, warmup,
                     source_ref, probes, run_metrics, artifacts, real_activation, activation_source):
    tech = TECHNOLOGIES[technology]
    context, group_id = comparison_context(model_id, target_layer, device, iterations, warmup, source_ref)
    source_bytes, source_shard = safetensor_tensor_bytes(model_id, target_layer)
    baseline_probe = probes.get("baseline")
    candidate_probe = probes.get("candidate")
    out = []

    for source in records:
        record = json.loads(json.dumps(source))
        battery_id = str(record.get("battery_id") or "")
        status = str(record.get("status") or "").upper()
        simulated = status == "SIMULATED" or "_SIM" in battery_id.upper()

        record["schema_version"] = SCHEMA_VERSION
        record["technology"] = tech["label"]
        record["benchmark_protocol"] = BENCHMARK_PROTOCOL
        record["comparison_context"] = context
        record["comparison_group_id"] = group_id

        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            record["metrics"] = metrics

        # Formula/estimated RAM never goes into top-level comparison fields.
        for key in ("baseline_ram_bytes", "candidate_ram_bytes", "rift_ram_bytes",
                    "cascade_ram_bytes", "aether_ram_bytes", "spectra_ram_bytes", "winner_ram_bytes"):
            if key in record:
                record[key] = None

        metrics["memory_real"] = {
            "scope": "measured_process_and_operation_transient",
            "IMPORTANT": (
                "Top-level RAM comparison is null until baseline and candidate resident "
                "inference states can be isolated. These values are observed measurements."
            ),
            "run_process": run_metrics,
            "baseline_operation": (baseline_probe or {}).get("memory"),
            "candidate_operation": (candidate_probe or {}).get("memory"),
        }

        if not battery_id.startswith("B0_"):
            record["baseline_disk_bytes"] = source_bytes
            if source_bytes is not None:
                metrics["source_tensor_storage"] = {
                    "bytes": source_bytes,
                    "shard": source_shard,
                    "measurement_method": "safetensors_data_offsets_v1",
                }

        storage_real = verify_candidate_disk(record, artifacts)
        metrics["storage_real"] = storage_real
        if not battery_id.startswith("B0_") and not storage_real["candidate_exact_file_match"]:
            record["candidate_disk_bytes"] = None
            for alias in ("rift_disk_bytes", "cascade_disk_bytes", "aether_disk_bytes", "spectra_disk_bytes", "winner_disk_bytes"):
                if alias in record:
                    record[alias] = None

        metrics["artifacts"] = {"files": artifacts[:100], "truncated": len(artifacts) > 100}
        metrics["benchmark_probe"] = {"baseline": baseline_probe, "candidate": candidate_probe}
        metrics["activation_provenance"] = {"real": real_activation, "source": activation_source}
        metrics["end_to_end_generation"] = {
            "measured": False,
            "baseline_tok_s": None,
            "candidate_tok_s": None,
            "reason": "candidate technology does not yet expose a full-model generation runtime",
        }

        for key in ("baseline_tok_s", "candidate_tok_s", "rift_tok_s", "cascade_tok_s",
                    "aether_tok_s", "spectra_tok_s", "winner_tok_s"):
            if key in record:
                record[key] = None

        if simulated:
            record["comparison_role"] = "diagnostic"
            record["implementation"] = {
                "kind": "SIMULATED", "native": False, "simulated": True,
                "eligible_for_primary_ranking": False,
            }
        else:
            primary = battery_id == tech["primary"] and real_activation
            record["comparison_role"] = "primary" if primary else "diagnostic"
            record["implementation"] = {
                "kind": "REFERENCE_MEASURED", "native": False, "simulated": False,
                "eligible_for_primary_ranking": bool(primary),
            }

        if battery_id == tech["primary"] and not real_activation:
            record["status"] = "INVALID_REAL_INPUT"
            record["notes"] = (
                str(record.get("notes") or "")
                + " Primary ranking disabled because a real model activation was not captured."
            ).strip()

        gains = record.get("gains")
        if not isinstance(gains, dict):
            gains = {}
            record["gains"] = gains
        gains["tok_s_gain_pct"] = None
        gains["ram_reduction_pct"] = None
        base_disk = record.get("baseline_disk_bytes")
        cand_disk = record.get("candidate_disk_bytes")
        try:
            if base_disk and cand_disk is not None:
                gains["disk_reduction_pct"] = (1.0 - float(cand_disk) / float(base_disk)) * 100.0
                gains["disk_compression_ratio_x"] = float(base_disk) / max(float(cand_disk), 1.0)
            else:
                gains["disk_reduction_pct"] = None
                gains["disk_compression_ratio_x"] = None
        except Exception:
            gains["disk_reduction_pct"] = None
            gains["disk_compression_ratio_x"] = None
        gains["overall_gain_pct"] = gains.get("disk_reduction_pct")

        record["measurement_scope"] = (
            "REAL_MEASUREMENT_V3: Linear latency uses synchronized repeated trials; "
            "CPU RSS and CUDA allocation peaks are observed; baseline disk bytes come "
            "from Safetensors data_offsets when available; end-to-end Tok/s is not measured; "
            "formula-based RAM is excluded."
        )
        out.append(record)
    return out

def post_results(records: list[dict[str, Any]], endpoint: str, token: str) -> dict[str, Any]:
    body = json.dumps({"records": records}, ensure_ascii=False).encode("utf-8")
    req = Request(endpoint, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "rift-real-benchmark-runner/1.0",
    })
    try:
        with urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"results endpoint HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"results endpoint unavailable: {exc.reason}") from exc

def run(args: argparse.Namespace) -> int:
    technology = args.technology.lower()
    tech = TECHNOLOGIES[technology]
    model_id = normalize_model_id(args.model)
    actual_device = resolve_device(args.device)
    source_ref = args.source_ref or os.environ.get("RIFT_SOURCE_REF", "main")
    script_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{source_ref}/{tech['script']}"

    workspace_root = Path("/content/rift_real_runs") if Path("/content").is_dir() else (Path.cwd() / ".rift_real_runs")
    workspace_root.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f"{technology}_", dir=workspace_root))
    try:
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        probe_log = tmp_path / "benchmark_probes.jsonl"
        script_path = tmp_path / tech["script"]

        print(f"[REAL-METRICS] tecnologia={tech['label']} modelo={model_id} device={actual_device}")
        print(f"[REAL-METRICS] protocolo={BENCHMARK_PROTOCOL}")
        original_source = download_text(script_url)
        script_path.write_text(instrument_source(original_source), encoding="utf-8")

        command = [
            sys.executable, str(script_path), *tech["args"],
            "--model", model_id,
            "--target-layer", args.target_layer,
            "--device", actual_device,
            "--iterations", str(args.iterations),
            "--out", str(out_dir),
            "--publish", "off",
        ]
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        env = os.environ.copy()
        env["RIFT_REAL_PROBE_LOG"] = str(probe_log)
        env["RIFT_REAL_ITERATIONS"] = str(args.iterations)
        env["RIFT_REAL_WARMUP"] = str(args.warmup)
        env["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
        env["RIFT_SOURCE_REF"] = source_ref

        started = time.perf_counter()
        process = subprocess.Popen(command, env=env)
        peak_rss, peak_gpu, samples = 0, None, 0
        while process.poll() is None:
            pids = process_tree_pids(process.pid)
            peak_rss = max(peak_rss, sum(rss_for_pid(pid) for pid in pids))
            gpu_now = gpu_memory_for_pids(pids)
            if gpu_now is not None:
                peak_gpu = max(peak_gpu or 0, gpu_now)
            samples += 1
            time.sleep(0.02)

        return_code = process.wait()
        run_metrics = {
            "measurement_method": "external_proc_tree_sampling_v1",
            "sample_interval_ms": 20,
            "samples": samples,
            "wall_time_seconds": time.perf_counter() - started,
            "peak_process_tree_rss_bytes": peak_rss or None,
            "peak_process_tree_gpu_bytes": peak_gpu,
            "exit_code": return_code,
        }

        records = load_records(out_dir)
        if not records:
            raise RuntimeError(f"battery exit={return_code}, mas nenhum histórico JSON foi encontrado")

        probes = load_probe_log(probe_log, technology)
        target_layer = resolve_target_layer(out_dir, args.target_layer)
        real_activation, activation_source = activation_is_real(out_dir, records)
        artifacts = file_manifest(out_dir)

        sanitized = sanitize_records(
            records,
            technology=technology,
            model_id=model_id,
            target_layer=target_layer,
            device=actual_device,
            iterations=args.iterations,
            warmup=args.warmup,
            source_ref=source_ref,
            probes=probes,
            run_metrics=run_metrics,
            artifacts=artifacts,
            real_activation=real_activation,
            activation_source=activation_source,
        )

        local_path = Path(args.output or f"real_{technology}_batteries.json")
        local_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[REAL-METRICS] resultado sanitizado: {local_path}")
        print(f"[REAL-METRICS] RSS pico processo+filhos: {run_metrics['peak_process_tree_rss_bytes']} bytes")
        print(f"[REAL-METRICS] GPU pico por PID: {run_metrics['peak_process_tree_gpu_bytes']} bytes")
        print(f"[REAL-METRICS] ativação={activation_source}; primary_eligible={real_activation}")

        if args.publish != "off":
            token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
            if not token:
                if args.publish == "required":
                    raise RuntimeError("RIFT_INGEST_TOKEN ausente")
                print("[REAL-METRICS] publicação ignorada: token ausente")
            else:
                endpoint = args.results_endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT", DEFAULT_RESULTS_ENDPOINT)
                print("[REAL-METRICS] publicação:", json.dumps(post_results(sanitized, endpoint, token), ensure_ascii=False))

        if return_code != 0:
            print(f"[REAL-METRICS] bateria original terminou com código {return_code}; registros de falha foram preservados.")
        return return_code
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(tmp_path, ignore_errors=True)

def build_parser():
    p = argparse.ArgumentParser(description="Real measurement wrapper for RIFT-LM batteries")
    p.add_argument("--technology", choices=sorted(TECHNOLOGIES), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--target-layer", default="auto")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="auto")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--publish", choices=["auto", "required", "off"], default="required")
    p.add_argument("--results-endpoint", default=None)
    p.add_argument("--source-ref", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--keep-artifacts", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    if args.iterations < 10:
        raise SystemExit("--iterations deve ser >= 10")
    if args.warmup < 1:
        raise SystemExit("--warmup deve ser >= 1")
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
