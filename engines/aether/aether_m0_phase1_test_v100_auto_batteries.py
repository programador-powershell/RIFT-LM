#!/usr/bin/env python3
"""AETHER-LM v1.0 — M0/Phase 1 reference battery for Colab.

This validates a ternary HQR-style base, token-adaptive low-rank refinement and
a simulated predictive paging policy. ANS 0.85-bit coding, asynchronous P-IO
and the native SRFA fused kernel are not implemented by this Python path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


# Faixas testadas (piso + teto) — nunca `-U` sem pino (ver docs/C3_CONTRACTS_V1.md §5).
PINNED_PIP_PACKAGES = {
    "xxhash": "xxhash>=3.0,<4",
    "sentencepiece": "sentencepiece>=0.1.99,<0.3",
    "tiktoken": "tiktoken>=0.7,<1",
    "transformers": "transformers>=4.44,<5",
    "accelerate": "accelerate>=0.33,<2",
    "huggingface_hub": "huggingface_hub>=0.24,<1",
}


def pip_auto_install_allowed() -> bool:
    """pip automático só no Colab ou com RIFT_AUTO_INSTALL=1; nunca silencioso em máquina local."""
    if os.environ.get("RIFT_AUTO_INSTALL", "").strip() == "1":
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_import(module: str, pip_name: str | None = None, *, required: bool = True):
    try:
        return __import__(module)
    except ImportError:
        package = PINNED_PIP_PACKAGES.get(pip_name or module, pip_name or module)
        if not pip_auto_install_allowed():
            message = (
                f"Dependência ausente: {module}. Instale manualmente (pip install \"{package}\") "
                "ou defina RIFT_AUTO_INSTALL=1 — a instalação automática só roda no Colab."
            )
            if required:
                raise SystemExit(message)
            print(f"[deps] AVISO: {message}")
            return None
        print(f"[deps] Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return __import__(module)


xxhash = ensure_import("xxhash")
torch = None
F = None
AutoModel = None
AutoModelForCausalLM = None
AutoTokenizer = None
AutoModel = None
AutoModelForMultimodalLM = None


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForMultimodalLM
    if torch is not None:
        return
    ensure_import("sentencepiece", required=False)
    ensure_import("tiktoken", required=False)
    if pip_auto_install_allowed():
        print("[deps] Garantindo transformers/accelerate/huggingface_hub nas faixas pinadas (Gemma 4 / multimodal)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             PINNED_PIP_PACKAGES["transformers"],
             PINNED_PIP_PACKAGES["accelerate"],
             PINNED_PIP_PACKAGES["huggingface_hub"]]
        )
    else:
        print("[deps] Instalação automática desativada (fora do Colab e sem RIFT_AUTO_INSTALL=1); usando versões locais.")
    try:
        import torch as _torch
        import torch.nn.functional as _F
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import AutoModel as _AutoModel
    except ImportError as exc:
        raise SystemExit(
            "PyTorch e Transformers são necessários. Instale com: "
            "pip install torch transformers accelerate sentencepiece tiktoken\n"
            f"Erro: {exc}"
        )
    _AutoMM = None
    try:
        from transformers import AutoModelForMultimodalLM as _AutoMM
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as _AutoMM
        except ImportError:
            _AutoMM = None
    torch = _torch
    F = _F
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    AutoModel = _AutoModel
    AutoModelForMultimodalLM = _AutoMM




AETHER_MAGIC = b"AETH"
AETHER_VERSION_M0 = 0x0100
AETHER_HEADER_SIZE = 128
AETHER_HEADER_FORMAT = "<4sHHIIQQQQQQQ56s"
AETHER_CHECKSUM_OFFSET = 64
AETHER_CHECKSUM_SEED = 42
STAGE_ENTRY_FORMAT = "<IIQQQ"
STAGE_ENTRY_SIZE = struct.calcsize(STAGE_ENTRY_FORMAT)
EXPECTED_GOLDEN_CHECKSUM = 0x694E219D50D72132
assert struct.calcsize(AETHER_HEADER_FORMAT) == AETHER_HEADER_SIZE
assert STAGE_ENTRY_SIZE == 32

BENCHMARK_PROTOCOL = "LINEAR_REFERENCE_V2"
SCHEMA_VERSION = 2


class AetherFormatError(ValueError):
    """Formato inválido no container AETH (header, offsets ou checksums)."""


def align_up(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def xxh3_64(data: bytes) -> int:
    return int(xxhash.xxh3_64_intdigest(data, seed=AETHER_CHECKSUM_SEED))


def create_aether_header(
    *,
    ir_offset: int,
    stage_table_offset: int,
    stage_count: int,
    prediction_table_offset: int,
    payload_offset: int,
    file_size: int,
    magic: bytes = AETHER_MAGIC,
    version: int = AETHER_VERSION_M0,
) -> bytes:
    raw = struct.pack(
        AETHER_HEADER_FORMAT,
        magic,
        version,
        0,
        0x07,
        0,
        ir_offset,
        stage_table_offset,
        stage_count,
        prediction_table_offset,
        payload_offset,
        file_size,
        0,
        bytes(56),
    )
    checksum = xxh3_64(raw)
    return raw[:AETHER_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def parse_aether_header(data: bytes, actual_file_size: int) -> dict[str, int]:
    if len(data) < AETHER_HEADER_SIZE:
        raise AetherFormatError("truncated_header")
    values = struct.unpack(AETHER_HEADER_FORMAT, data[:AETHER_HEADER_SIZE])
    (
        magic,
        version,
        _profile,
        _flags,
        _reserved,
        ir_offset,
        stage_table_offset,
        stage_count,
        prediction_table_offset,
        payload_offset,
        file_size,
        checksum,
        _reserved_tail,
    ) = values
    if magic != AETHER_MAGIC:
        raise AetherFormatError("bad_magic")
    if version != AETHER_VERSION_M0:
        raise AetherFormatError("bad_version")
    zeroed = bytearray(data[:AETHER_HEADER_SIZE])
    zeroed[AETHER_CHECKSUM_OFFSET:72] = bytes(8)
    if xxh3_64(bytes(zeroed)) != checksum:
        raise AetherFormatError("checksum_mismatch")
    if file_size != actual_file_size:
        raise AetherFormatError("file_size_mismatch")
    offsets = (ir_offset, stage_table_offset, prediction_table_offset, payload_offset)
    if any(offset < AETHER_HEADER_SIZE or offset > file_size for offset in offsets):
        raise AetherFormatError("offset_out_of_range")
    if offsets != tuple(sorted(offsets)):
        raise AetherFormatError("offset_order")
    if stage_count > 1_000_000:
        raise AetherFormatError("stage_count_overflow")
    if stage_table_offset + stage_count * STAGE_ENTRY_SIZE > prediction_table_offset:
        raise AetherFormatError("stage_table_overlap")
    return {
        "ir_offset": ir_offset,
        "stage_table_offset": stage_table_offset,
        "stage_count": stage_count,
        "prediction_table_offset": prediction_table_offset,
        "payload_offset": payload_offset,
        "file_size": file_size,
        "checksum": checksum,
    }


def _rewrite_header(header: bytes, **changes: int | bytes) -> bytes:
    names = [
        "magic", "version", "profile", "flags", "reserved", "ir_offset",
        "stage_table_offset", "stage_count", "prediction_table_offset",
        "payload_offset", "file_size", "checksum", "reserved_tail",
    ]
    current = dict(zip(names, struct.unpack(AETHER_HEADER_FORMAT, header)))
    current.update(changes)
    current["checksum"] = 0
    raw = struct.pack(AETHER_HEADER_FORMAT, *(current[name] for name in names))
    checksum = xxh3_64(raw)
    return raw[:AETHER_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def run_golden_header_tests() -> dict[str, Any]:
    golden = create_aether_header(
        ir_offset=128,
        stage_table_offset=128,
        stage_count=0,
        prediction_table_offset=128,
        payload_offset=128,
        file_size=128,
    )
    parsed = parse_aether_header(golden, 128)
    checksum = parsed["checksum"]
    if EXPECTED_GOLDEN_CHECKSUM and checksum != EXPECTED_GOLDEN_CHECKSUM:
        raise AssertionError(
            f"Golden checksum mudou: 0x{checksum:016x} != 0x{EXPECTED_GOLDEN_CHECKSUM:016x}"
        )
    cases: list[tuple[bytes, int]] = [
        (golden[:0], 0),
        (golden[:64], 64),
        (golden[:127], 127),
        (_rewrite_header(golden, magic=b"NOPE"), 128),
        (_rewrite_header(golden, version=99), 128),
        (golden[:64] + bytes([golden[64] ^ 1]) + golden[65:], 128),
        (_rewrite_header(golden, file_size=129), 128),
        (_rewrite_header(golden, ir_offset=127), 128),
        (_rewrite_header(golden, stage_table_offset=129), 128),
        (_rewrite_header(golden, payload_offset=127), 128),
        (_rewrite_header(golden, stage_count=1_000_001), 128),
    ]
    passed = 0
    for mutated, actual_size in cases:
        try:
            parse_aether_header(mutated, actual_size)
        except AetherFormatError:
            passed += 1
    if passed != len(cases):
        raise AssertionError(f"Negative golden tests: {passed}/{len(cases)}")
    return {"header": golden, "checksum": checksum, "negative_tests_passed": passed}


def normalize_huggingface_model_id(value: str) -> str:
    model_id = str(value or "").strip()
    parsed = urlparse(model_id)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("--model aceita somente model ID ou URL do huggingface.co")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("URL incompleta; use https://huggingface.co/ORG/MODELO")
        model_id = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("Model ID inválido; formato esperado: organizacao/modelo")
    return model_id


def resolve_linear_weight_name(model: Any, requested: str) -> str:
    state = model.state_dict()
    if requested and requested.lower() != "auto":
        if requested not in state:
            raise KeyError(f"Tensor não encontrado: {requested}")
        if getattr(state[requested], "ndim", 0) != 2:
            raise ValueError(f"Tensor precisa ser uma matriz 2D: {requested}")
        return requested
    preferred = (
        "self_attn.q_proj", "self_attn.qkv_proj", "self_attn.query_key_value",
        "attention.q_proj", "attention.wq", "attn.q_proj",
        "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
        "self_attn.o_proj",
    )
    linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and getattr(module, "weight", None) is not None
        and module.weight.ndim == 2
    ]
    for suffix in preferred:
        match = next((name for name in linear_names if name.lower().endswith(suffix)), None)
        if match:
            return f"{match}.weight"
    if linear_names:
        return f"{linear_names[0]}.weight"
    raise KeyError("O modelo não expõe camada torch.nn.Linear com peso 2D")


def capture_activation(model: Any, tokenizer: Any, module_name: str, device: Any, prompt: str):
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Módulo não encontrado para hook: {module_name}")
    captured: list[Any] = []

    def hook(_module, inputs, _output):
        if inputs:
            captured.append(inputs[0].detach())

    handle = modules[module_name].register_forward_hook(hook)
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Hook não capturou ativação")
    return captured[0]


def compute_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    cosine = float(F.cosine_similarity(ref, cand, dim=0, eps=1e-12).item())
    rmse = torch.sqrt(torch.mean((ref - cand) ** 2))
    scale = torch.max(ref) - torch.min(ref)
    nrmse = float((rmse / (scale + 1e-12)).item())
    return {"cosine": cosine, "nrmse": nrmse}


def benchmark_ms(fn, *, device: Any, iterations: int) -> dict[str, float]:
    for _ in range(5):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": float(statistics.median(values)),
        "minimum_ms": float(min(values)),
        "iterations": iterations,
    }


def ternary_quantize(weight: Any) -> tuple[Any, float, float]:
    threshold = float((0.7 * weight.abs().mean()).item())
    selected = weight.abs() >= threshold
    scale = float(weight.abs()[selected].mean().item()) if bool(selected.any()) else 0.0
    ternary = torch.where(selected, torch.sign(weight) * scale, torch.zeros_like(weight))
    sparsity = float((ternary == 0).float().mean().item())
    return ternary, scale, sparsity


def pack_ternary(weight: Any) -> bytes:
    values = weight.detach().cpu().reshape(-1).numpy()
    codes = np.where(values < 0, 0, np.where(values > 0, 2, 1)).astype(np.uint8)
    padding = (-len(codes)) % 4
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=1)
    packed = codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
    return packed.tobytes()


def entropy_and_ranks(x: Any, maximum_rank: int) -> tuple[Any, Any, list[float]]:
    probabilities = torch.softmax(x.abs(), dim=1)
    entropy = -(probabilities * torch.log2(probabilities + 1e-12)).sum(dim=1)
    thresholds = torch.quantile(entropy, torch.tensor([0.85, 0.90, 0.97], device=x.device))
    ranks = torch.zeros_like(entropy, dtype=torch.int64)
    ranks = torch.where(entropy >= thresholds[0], min(2, maximum_rank), ranks)
    ranks = torch.where(entropy >= thresholds[1], min(8, maximum_rank), ranks)
    ranks = torch.where(entropy >= thresholds[2], maximum_rank, ranks)
    return entropy, ranks, [float(value.item()) for value in thresholds]


def lowrank_correction(x: Any, u: Any, s: Any, v: Any, rank: int) -> Any:
    return ((x @ v[:, :rank]) * s[:rank]) @ u[:, :rank].T


def aether_dynamic_linear(x: Any, ternary: Any, u: Any, s: Any, v: Any, ranks: Any) -> Any:
    output = F.linear(x, ternary)
    for rank in sorted({int(value) for value in ranks.tolist()}):
        if rank <= 0:
            continue
        mask = ranks == rank
        output[mask] += lowrank_correction(x[mask], u, s, v, rank)
    return output


def write_bundle(
    path: Path,
    *,
    model_id: str,
    target_layer: str,
    ternary: Any,
    scale: float,
    u: Any,
    s: Any,
    v: Any,
    thresholds: list[float],
) -> dict[str, Any]:
    base_payload = struct.pack("<f", scale) + pack_ternary(ternary)
    residual_payload = b"".join(
        tensor.detach().cpu().to(dtype=torch.float16).contiguous().numpy().tobytes()
        for tensor in (u, s, v)
    )
    ir = json.dumps({
        "ir_version": 1,
        "engine": "AETHER_REFERENCE_PYTHON",
        "model_id": model_id,
        "target_layer": target_layer,
        "operations": ["HQR_TERNARY", "TADDS_DYNAMIC_LOW_RANK"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prediction = json.dumps({
        "policy": "ENTROPY_QUANTILE_REFERENCE",
        "ranks": [0, 2, 8, int(s.numel())],
        "thresholds": thresholds,
        "pio_async_native": False,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ir_offset = AETHER_HEADER_SIZE
    stage_table_offset = align_up(ir_offset + len(ir))
    prediction_table_offset = align_up(stage_table_offset + 2 * STAGE_ENTRY_SIZE)
    payload_offset = align_up(prediction_table_offset + len(prediction))
    base_offset = payload_offset
    residual_offset = align_up(base_offset + len(base_payload))
    file_size = residual_offset + len(residual_payload)
    stages = [
        (0, 0, base_offset, len(base_payload), xxh3_64(base_payload)),
        (1, int(s.numel()), residual_offset, len(residual_payload), xxh3_64(residual_payload)),
    ]
    stage_table = b"".join(struct.pack(STAGE_ENTRY_FORMAT, *stage) for stage in stages)
    header = create_aether_header(
        ir_offset=ir_offset,
        stage_table_offset=stage_table_offset,
        stage_count=len(stages),
        prediction_table_offset=prediction_table_offset,
        payload_offset=payload_offset,
        file_size=file_size,
    )
    blob = bytearray(file_size)
    blob[:AETHER_HEADER_SIZE] = header
    blob[ir_offset:ir_offset + len(ir)] = ir
    blob[stage_table_offset:stage_table_offset + len(stage_table)] = stage_table
    blob[prediction_table_offset:prediction_table_offset + len(prediction)] = prediction
    blob[base_offset:base_offset + len(base_payload)] = base_payload
    blob[residual_offset:residual_offset + len(residual_payload)] = residual_payload
    path.write_bytes(blob)
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            parsed = parse_aether_header(mapped[:AETHER_HEADER_SIZE], len(mapped))
            for stage_id, rank, offset, size, checksum in stages:
                if xxh3_64(mapped[offset:offset + size]) != checksum:
                    raise AetherFormatError(f"stage_checksum_{stage_id}")
    return {
        "file_size": file_size,
        "header_checksum": parsed["checksum"],
        "hqr_packed_bytes": len(base_payload),
        "tadds_residual_bytes": len(residual_payload),
        "stages": [
            {"stage_id": stage[0], "rank": stage[1], "offset": stage[2], "size": stage[3]}
            for stage in stages
        ],
    }


def pct_lower(base: float | None, candidate: float | None) -> float | None:
    return None if not base or candidate is None else (1.0 - candidate / base) * 100.0


def pct_higher(base: float | None, candidate: float | None) -> float | None:
    """Ganho % quando maior é melhor (tok/s)."""
    return None if not base or candidate is None else (candidate / base - 1.0) * 100.0


def read_vmrss_bytes() -> int | None:
    """VmRSS do processo em bytes via /proc/self/status (Linux/Colab); None fora do Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def getrusage_peak_bytes() -> int | None:
    """Fallback: pico de RSS do processo via resource.getrusage (ru_maxrss em KB no Linux)."""
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if peak <= 0:
        return None
    return peak if sys.platform == "darwin" else peak * 1024


class PhaseRamSampler:
    """Thread que amostra VmRSS a ~1ms durante uma fase de benchmark (máximo e média por fase)."""

    def __init__(self, interval_seconds: float = 0.001):
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = read_vmrss_bytes()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PhaseRamSampler":
        if read_vmrss_bytes() is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def summary(self) -> dict[str, Any] | None:
        if self.samples:
            return {
                "max_bytes": int(max(self.samples)),
                "mean_bytes": int(statistics.fmean(self.samples)),
                "samples": len(self.samples),
                "method": "proc_vmrss_sampling_per_phase_v1",
            }
        peak = getrusage_peak_bytes()
        if peak is not None:
            return {
                "max_bytes": peak,
                "mean_bytes": None,
                "samples": 0,
                "method": "getrusage_peak_fallback",
            }
        return None


def measured_phase_max(summary: dict[str, Any] | None) -> int | None:
    """Só VmRSS medido por fase alimenta *_ram_bytes de nível superior; o fallback fica em metrics.memory."""
    if summary and summary.get("method") == "proc_vmrss_sampling_per_phase_v1":
        return int(summary["max_bytes"])
    return None


def build_memory_metrics(
    baseline_measured: dict[str, Any] | None,
    candidate_measured: dict[str, Any] | None,
    *,
    estimated_baseline_bytes: int | None = None,
    estimated_candidate_bytes: int | None = None,
) -> dict[str, Any]:
    method = None
    for summary in (candidate_measured, baseline_measured):
        if summary and summary.get("method"):
            method = summary["method"]
            break
    return {
        "method": method,
        "baseline_phase": baseline_measured,
        "candidate_phase": candidate_measured,
        "estimated_baseline_bytes": estimated_baseline_bytes,
        "estimated_candidate_bytes": estimated_candidate_bytes,
    }


def package_version(name: str) -> str | None:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        module = sys.modules.get(name)
        return getattr(module, "__version__", None) if module is not None else None


def build_comparison_context(model_id: str, device: str) -> tuple[dict[str, Any], str]:
    """Contexto de comparação do schema v2 + comparison_group_id (cmp-<sha256[:24]>)."""
    torch_version = package_version("torch")
    context = {
        "protocol": BENCHMARK_PROTOCOL,
        "device": device,
        "torch": torch_version,
        "transformers": package_version("transformers"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    fingerprint = f"{BENCHMARK_PROTOCOL}|{model_id}|{device}|{torch_version}"
    group_id = "cmp-" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return context, group_id


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str, publish_mode: str = "off", results_endpoint: str | None = None, device: str | None = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "aether_test_batteries.json"
        self.csv_path = out_dir / "aether_test_batteries.csv"
        self.model_id = model_id
        self.device = str(device) if device else "unknown"
        self.comparison_context, self.comparison_group_id = build_comparison_context(model_id, self.device)
        self.publish_mode = publish_mode
        self.results_endpoint = results_endpoint
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        battery_id: str,
        status: str,
        measurement_scope: str,
        baseline_tok_s: float | None = None,
        candidate_tok_s: float | None = None,
        baseline_ram_bytes: int | None = None,
        candidate_ram_bytes: int | None = None,
        baseline_disk_bytes: int | None = None,
        candidate_disk_bytes: int | None = None,
        quality: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        comparison_role: str | None = None,
    ) -> dict[str, Any]:
        # tok/s de topo SOMENTE de model.generate (bateria E2E); demais baterias
        # continuam com None — nunca inventamos Tok/s (protocolo V3 / contracts §12).
        tok_gain = pct_higher(baseline_tok_s, candidate_tok_s)
        ram_gain = pct_lower(baseline_ram_bytes, candidate_ram_bytes)
        disk_gain = pct_lower(baseline_disk_bytes, candidate_disk_bytes)
        measured = [value for value in (tok_gain, ram_gain, disk_gain) if value is not None]
        simulated = str(status).upper() == "SIMULATED" or battery_id.endswith("_PIO_POLICY_SIM")
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "spec": "AETHER-LM v1.0 reference",
            "technology": "AETHER",
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            "comparison_role": comparison_role,
            "schema_version": SCHEMA_VERSION,
            "benchmark_protocol": BENCHMARK_PROTOCOL,
            "comparison_group_id": self.comparison_group_id,
            "comparison_context": dict(self.comparison_context),
            "implementation": {
                "kind": "SIMULATED" if simulated else "REFERENCE_MEASURED",
                "native": False,
                "simulated": simulated,
                "eligible_for_primary_ranking": (comparison_role == "primary") and not simulated,
            },
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "baseline_ram_bytes": baseline_ram_bytes,
            "candidate_ram_bytes": candidate_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes,
            "candidate_disk_bytes": candidate_disk_bytes,
            "gains": {
                "tok_s_gain_pct": tok_gain,
                "ram_reduction_pct": ram_gain,
                "disk_reduction_pct": disk_gain,
                "disk_compression_ratio_x": (
                    baseline_disk_bytes / candidate_disk_bytes
                    if baseline_disk_bytes and candidate_disk_bytes else None
                ),
                "overall_gain_pct": sum(measured) / len(measured) if measured else None,
            },
            "measurement_scope": measurement_scope,
            "quality": quality or {},
            "metrics": metrics or {},
            "notes": notes,
        }
        self.records = [item for item in self.records if item["battery_id"] != battery_id]
        self.records.append(record)
        self.records.sort(key=lambda item: item["battery_id"])
        self.json_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        single = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        single.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "timestamp_utc", "run_id", "technology", "model_id", "battery_id",
                "status", "baseline_ram_bytes", "candidate_ram_bytes",
                "baseline_disk_bytes", "candidate_disk_bytes", "measurement_scope",
            ])
            writer.writeheader()
            for item in self.records:
                writer.writerow({key: item.get(key) for key in writer.fieldnames})
        print(f"[BATTERY] {battery_id}: gravada automaticamente -> {single}")
        # Publica imediatamente no site (não espera o fim de todas as baterias)
        try:
            publish_to_vercel(
                path=self.json_path,
                mode=self.publish_mode,
                endpoint=self.results_endpoint,
                records=list(self.records),
                quiet=False,
            )
        except ResultsPublishError as exc:
            print(f"[PUBLISH] AVISO (incremental): {exc}")

        return record


class ResultsPublishError(RuntimeError):
    pass


def running_in_colab() -> bool:
    if "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG")):
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def read_setting(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata
        value = str(userdata.get(name) or "").strip()
        return value or None
    except Exception:
        return None


def publish_to_vercel(path: Path | None = None, *, mode: str, endpoint: str | None, records: list | None = None, quiet: bool = False) -> None:
    if mode == "off":
        if not quiet:
            print("[PUBLISH] Publicação remota desativada (--publish off).")
        return
    endpoint = (endpoint or read_setting("RIFT_RESULTS_ENDPOINT") or "").strip()
    token = read_setting("RIFT_INGEST_TOKEN")
    missing = []
    if not endpoint:
        missing.append("RIFT_RESULTS_ENDPOINT")
    if not token:
        missing.append("RIFT_INGEST_TOKEN")
    if missing:
        message = "Configure " + " e ".join(missing)
        if mode == "required" and not quiet:
            raise ResultsPublishError(message)
        if not quiet:
            print(f"[PUBLISH] AVISO: {message}; resultados preservados localmente.")
        return
    if not endpoint.startswith("https://") or len(token) < 32:
        raise ResultsPublishError("Endpoint precisa ser HTTPS e token deve ter ao menos 32 caracteres")
    if records is None:
        if path is None:
            raise ResultsPublishError("path ou records é obrigatório")
        records = json.loads(path.read_text(encoding="utf-8"))
    request = Request(
        endpoint, data=json.dumps({"records": records}).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": "aether-colab-publisher/1.0"},
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ResultsPublishError(f"Vercel respondeu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ResultsPublishError(f"Falha de rede ao publicar: {exc.reason}") from exc
    publication = result.get("publication", {})
    print(f"[PUBLISH] {len(records)} registro(s) aceito(s); snapshot publicado com {publication.get('records', '?')} registro(s).")
    if publication.get("commit_url"):
        print(f"[PUBLISH] Commit: {publication['commit_url']}")






def resolve_hf_token() -> str | None:
    """HF_TOKEN / HUGGING_FACE_HUB_TOKEN a partir do ambiente ou Secrets do Colab."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from google.colab import userdata
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            try:
                value = str(userdata.get(name) or "").strip()
            except Exception:
                value = ""
            if value:
                # Espelha no ambiente para transformers/huggingface_hub
                os.environ.setdefault("HF_TOKEN", value)
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
                return value
    except Exception:
        pass
    return None


def ensure_hf_login(token: str | None = None) -> str | None:
    """Autentica no Hugging Face Hub quando há token (modelos gated)."""
    token = token or resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] HF_TOKEN aplicado (valor não exibido).")
    except Exception as exc:
        print(f"[auth] AVISO: não foi possível fazer login no Hub: {exc}")
    return token


def is_gated_access_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "gated repo",
        "access to model",
        "restricted",
        "401 client error",
        "403 client error",
        "cannot access gated",
        "you must have access",
        "please log in",
        "authentication",
        "authorized",
    )
    return any(m in text for m in markers)


def resolve_torch_device(requested: str):
    """cuda se disponível; senão CPU (Colab sem GPU / TPU sem CUDA)."""
    requested = (requested or "auto").strip().lower()
    if requested in {"auto", "gpu"}:
        requested = "cuda"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"device inválido: {requested} (use auto, cuda ou cpu)")
    if requested == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
                print(f"[device] CUDA disponível → usando GPU ({name})")
                return torch.device("cuda")
        except Exception as exc:
            print(f"[device] CUDA indisponível ({exc}); caindo para CPU")
        print("[device] Sem GPU CUDA — executando em CPU (adequado a Colab CPU/TPU sem torch_xla)")
        return torch.device("cpu")
    print("[device] Forçado para CPU")
    return torch.device("cpu")



def cleanup_colab_workspace(*, label: str = "battery", wipe_hf_cache: bool = False) -> None:
    """Libera artefatos temporários no Colab.

    Por padrão NÃO apaga o cache Hugging Face entre tecnologias da mesma célula
    serial (evita re-download de dezenas de GB). Wipe completo do hub só com
    wipe_hf_cache=True (final da fila / célula).
    """
    import gc
    import shutil
    import glob as _glob

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    allow_local = os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP", "").strip() == "1"
    if not (running_in_colab() or allow_local):
        print(
            f"[cleanup] {label}: fora do Colab — limpeza destrutiva ignorada "
            "(defina RIFT_ALLOW_LOCAL_CLEANUP=1 para forçar)."
        )
        return

    removed = []
    if wipe_hf_cache:
        home = Path.home()
        targets = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers",
            home / ".cache" / "huggingface" / "modules",
            home / ".cache" / "torch",
            Path("/content") / ".cache",
            Path("/root") / ".cache" / "huggingface" / "hub",
            Path("/root") / ".cache" / "huggingface" / "transformers",
        ]
        for path in targets:
            try:
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path, ignore_errors=True)
                    removed.append(f"{path} (~{size / (1024**3):.2f} GiB)")
                elif path.is_file():
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
            except Exception as exc:
                print(f"[cleanup] AVISO ao remover {path}: {exc}")

    patterns = [
        "/tmp/winner_cpp_*",
        "/tmp/winner_phase1_*",
        "/tmp/phase1_load_fail*",
        "/tmp/cascade_load_fail*",
        "/tmp/rift_*",
        "/content/*_launcher.py",
            ]
    for pattern in patterns:
        for match in _glob.glob(pattern):
            p = Path(match)
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
                removed.append(str(p))
            except Exception as exc:
                print(f"[cleanup] AVISO ao remover {p}: {exc}")

    if removed:
        print(f"[cleanup] {label}: espaço liberado ({len(removed)} item(ns)):")
        for item in removed[:12]:
            print(f"  - {item}")
        if len(removed) > 12:
            print(f"  - … +{len(removed) - 12} outros")
    else:
        print(f"[cleanup] {label}: nada temporário para limpar (cache HF preservado)")
    gc.collect()




def load_tokenizer(model_id: str, *, trust_remote_code: bool = False, token: str | None = None):
    """Carrega tokenizer com fallbacks (subfolder, use_fast=False).

    Para de tentar subpastas se o Hub responder 401/gated — o problema é auth,
    não layout de arquivos.
    """
    token = token or resolve_hf_token()
    ensure_hf_login(token)
    common = {"trust_remote_code": trust_remote_code}
    if token:
        common["token"] = token
    attempts = [
        {},
        {"use_fast": False},
        {"subfolder": "tokenizer"},
        {"subfolder": "tokenizer", "use_fast": False},
        {"subfolder": "processor"},
        {"subfolder": "processor", "use_fast": False},
    ]
    errors: list[str] = []
    for extra in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, **common, **extra)
            print(f"[tokenizer] OK com {extra or {'root': True}}")
            return tok
        except Exception as exc:
            errors.append(f"{extra or 'root'}: {type(exc).__name__}: {exc}")
            if is_gated_access_error(exc):
                raise RuntimeError(
                    f"Modelo gated/restrito: {model_id}.\n"
                    "1) Aceite os termos em https://huggingface.co/" + model_id + "\n"
                    "2) Configure o Secret HF_TOKEN no Colab (token com acesso de leitura).\n"
                    "3) Rode de novo. Sem token válido a bateria grava FAIL e segue a fila."
                ) from exc
    raise RuntimeError(
        "Não foi possível carregar o tokenizer de "
        f"{model_id}.\nTentativas:\n- " + "\n- ".join(errors)
    )



def load_model(model_id: str, *, device: Any, trust_remote_code: bool):
    hf_token = ensure_hf_login(resolve_hf_token())
    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code, token=hf_token)
    load_dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs = {
        "token": hf_token,
        "dtype": load_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
    classes = []
    if AutoModelForMultimodalLM is not None:
        classes.append(AutoModelForMultimodalLM)
    classes.extend([AutoModelForCausalLM, AutoModel])
    errors = []
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[load] Modelo carregado via {cls.__name__}")
            return tokenizer, model
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(
        "Não foi possível carregar o modelo.\n"
        "Gemma 4 Unified exige transformers recente (AutoModelForMultimodalLM).\n"
        + "\n".join(errors)
    )


# ------------------------------------------------------------------------------
# E2E tok/s de modelo completo — P1_AETHER_E2E_TOKS
# (docs/C3_CONTRACTS_V1.md §12; crib da técnica C3InlineLinearModule /
#  CascadeLinearModule de c3_methodology_auto_batteries.py)
# ------------------------------------------------------------------------------

E2E_GENERATION_PROMPT = (
    "Liste três técnicas para reduzir o uso de memória na inferência de LLMs:"
)
E2E_MAX_NEW_TOKENS = 48
E2E_MEASUREMENT_SCOPE = (
    "E2E model.generate greedy do modelo completo (prompt PT-BR fixo, "
    f"max_new_tokens={E2E_MAX_NEW_TOKENS}, 2 warmup + 3 medições, mediana, "
    "cuda sync). Candidato executa TODAS as nn.Linear dos blocos no runtime "
    "de referência do codec AETHER (ternário HQR + TADDS low-rank, W0 "
    "ternário dequantizado fp32 cacheado, W denso original fora do caminho "
    "quente) — runtime de referência Python — velocidade não representa "
    "kernel nativo. RAM=pico VmRSS por fase; disco=payloads F0/F1 reais (os.stat)."
)


def e2e_max_params() -> float:
    """Limite de parâmetros do guard (mesmo guard do GEYSER G3, 3e9)."""
    raw = os.environ.get("RIFT_E2E_MAX_PARAMS", "").strip()
    try:
        return float(raw) if raw else 3e9
    except ValueError:
        return 3e9


def read_meminfo_available_bytes() -> int | None:
    """MemAvailable via /proc/meminfo (Linux/Colab); None fora do Linux."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def e2e_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "linear"


def e2e_find_transformer_blocks(model: Any) -> list[tuple[str, Any]]:
    """Localiza a lista de blocos transformer (crib de cascade/compiler/block_decompose)."""
    candidate_attrs = (
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "transformer.layers",
        "model.decoder.layers",
        "gpt_neox.layers",
        "language_model.model.layers",
        "model.language_model.layers",
    )
    for attr in candidate_attrs:
        mod = model
        ok = True
        for part in attr.split("."):
            if not hasattr(mod, part):
                ok = False
                break
            mod = getattr(mod, part)
        if ok and isinstance(mod, (torch.nn.ModuleList, list)) and len(mod) > 0:
            return [(f"{attr}.{i}", mod[i]) for i in range(len(mod))]
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
            continue
        if name.split(".")[-1] not in ("layers", "h", "blocks", "layer"):
            continue
        if any(isinstance(m, torch.nn.Linear) for m in mod[0].modules()):
            return [(f"{name}.{i}", mod[i]) for i in range(len(mod))]
    return []


def e2e_set_module_by_path(root: Any, dotted: str, new_module: Any) -> None:
    """Substitui submódulo por caminho pontilhado (suporta índices numéricos)."""
    parts = dotted.split(".") if dotted else []
    if not parts:
        raise ValueError("caminho de módulo vazio")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def e2e_collect_linears(block: Any, block_name: str) -> dict[str, Any]:
    """Todas as nn.Linear nomeadas dentro de um bloco (nome completo -> módulo)."""
    found: dict[str, Any] = {}
    for name, mod in block.named_modules():
        if name and isinstance(mod, torch.nn.Linear):
            found[f"{block_name}.{name}"] = mod
    return found


_AETHER_INLINE_LINEAR_CLS = None


def make_aether_inline_linear_cls():
    """Cria (uma única vez) a classe do módulo inline; torch só existe em runtime."""
    global _AETHER_INLINE_LINEAR_CLS
    if _AETHER_INLINE_LINEAR_CLS is not None:
        return _AETHER_INLINE_LINEAR_CLS

    class AetherInlineLinearModule(torch.nn.Module):
        """Substitui nn.Linear pelo runtime de referência do codec AETHER.

        Crib da técnica C3InlineLinearModule/CascadeLinearModule: quantiza W
        UMA vez (ternário HQR via ternary_quantize + resíduo TADDS low-rank
        via svd_lowrank, reutilizando os helpers deste script), cacheia W0
        (ternário dequantizado) em fp32 e mantém o W denso original FORA do
        caminho quente. A seleção dinâmica de rank por token reutiliza
        entropy_and_ranks/aether_dynamic_linear. Runtime de referência
        Python — velocidade não representa kernel nativo.
        """

        def __init__(self, linear: Any, maximum_rank: int):
            super().__init__()
            device = linear.weight.device
            weight = linear.weight.detach().to(dtype=torch.float32)
            ternary, scale, sparsity = ternary_quantize(weight)
            rank = min(max(1, int(maximum_rank)), int(weight.shape[0]), int(weight.shape[1]))
            u, s, v = torch.svd_lowrank(weight - ternary, q=rank, niter=2)
            self.out_features = int(weight.shape[0])
            self.in_features = int(weight.shape[1])
            self.maximum_rank = rank
            self.scale = float(scale)
            self.sparsity = float(sparsity)
            self._w0 = ternary.to(device=device, dtype=torch.float32)
            self._u = u.to(device=device, dtype=torch.float32)
            self._s = s.to(device=device, dtype=torch.float32)
            self._v = v.to(device=device, dtype=torch.float32)
            self._bias = (
                linear.bias.detach().to(device=device, dtype=torch.float32)
                if linear.bias is not None
                else None
            )

        def f0_payload(self) -> bytes:
            """Payload F0 real: escala + base ternária empacotada em 2 bits."""
            return struct.pack("<f", self.scale) + pack_ternary(self._w0)

        def f1_payload(self) -> bytes:
            """Payload F1 real: fatores TADDS U/S/V em fp16 (como write_bundle)."""
            return b"".join(
                tensor.detach().cpu().to(dtype=torch.float16).contiguous().numpy().tobytes()
                for tensor in (self._u, self._s, self._v)
            )

        def forward(self, x: Any) -> Any:
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
            if self._w0.device != x2.device:
                self._w0 = self._w0.to(x2.device)
                self._u = self._u.to(x2.device)
                self._s = self._s.to(x2.device)
                self._v = self._v.to(x2.device)
                if self._bias is not None:
                    self._bias = self._bias.to(x2.device)
            _entropy, ranks, _thresholds = entropy_and_ranks(x2, self.maximum_rank)
            y = aether_dynamic_linear(x2, self._w0, self._u, self._s, self._v, ranks)
            if self._bias is not None:
                y = y + self._bias
            if y.dtype != x.dtype:
                y = y.to(dtype=x.dtype)
            return y.reshape(*orig_shape[:-1], self.out_features)

    _AETHER_INLINE_LINEAR_CLS = AetherInlineLinearModule
    return _AETHER_INLINE_LINEAR_CLS


def e2e_restore_patched(patched: dict[str, tuple[Any, str, Any]]) -> None:
    """Devolve as nn.Linear originais aos blocos (rollback/limpeza final)."""
    for full, (block, short, original) in list(patched.items()):
        try:
            e2e_set_module_by_path(block, short, original)
        except Exception as exc:
            print(f"[E2E] AVISO: rollback de {full} falhou: {exc}")


def e2e_patch_all_blocks(
    model: Any,
    blocks: list[tuple[str, Any]],
    artifacts_dir: Path,
    maximum_rank: int,
) -> tuple[dict[str, tuple[Any, str, Any]], int, int]:
    """Troca TODAS as nn.Linear dos blocos pelo módulo inline (transacional).

    Grava payloads F0 (ternário 2-bit) e F1 (TADDS U/S/V fp16) reais por
    Linear em <out>/artifacts/e2e/ e devolve (patched, baseline_disk_bytes,
    candidate_disk_bytes). Os originais são guardados ANTES da troca; em
    exceção no meio do loop tudo o que já foi trocado é RESTAURADO e a
    exceção é relançada (o modelo nunca fica parcialmente patchado).
    """
    inline_cls = make_aether_inline_linear_cls()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    patched: dict[str, tuple[Any, str, Any]] = {}
    baseline_disk = 0
    candidate_disk = 0
    try:
        for block_name, block in blocks:
            for full, linear in e2e_collect_linears(block, block_name).items():
                module = inline_cls(linear, maximum_rank)
                slug = e2e_slug(full)
                f0_path = artifacts_dir / f"{slug}.f0.bin"
                f1_path = artifacts_dir / f"{slug}.f1.bin"
                f0_path.write_bytes(module.f0_payload())
                f1_path.write_bytes(module.f1_payload())
                candidate_disk += int(os.stat(f0_path).st_size)
                candidate_disk += int(os.stat(f1_path).st_size)
                baseline_disk += int(linear.weight.numel() * 4)
                short = full[len(block_name) + 1:]
                e2e_set_module_by_path(block, short, module)
                patched[full] = (block, short, linear)
    except Exception:
        e2e_restore_patched(patched)
        patched.clear()
        raise
    if not patched:
        raise RuntimeError("nenhuma nn.Linear encontrada nos blocos para patch")
    return patched, baseline_disk, candidate_disk


def measure_generate_tok_s(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    *,
    max_new_tokens: int,
    warmup: int = 2,
    timed: int = 3,
) -> dict[str, Any]:
    """Tok/s REAL de model.generate (greedy) — mesmo protocolo p/ baseline e candidato."""
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            model.generate(**encoded, **{**gen_kwargs, "max_new_tokens": min(8, int(max_new_tokens))})
    if device.type == "cuda":
        torch.cuda.synchronize()
    tok_s_runs: list[float] = []
    last_out = None
    n_new = 0
    for _ in range(max(1, int(timed))):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            out = model.generate(**encoded, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        n_new = int(out.shape[1] - encoded["input_ids"].shape[1])
        tok_s_runs.append(n_new / max(elapsed_s, 1e-9))
        last_out = out
    runs = sorted(tok_s_runs)
    new_ids: list[int] = []
    if last_out is not None:
        new_ids = [int(t) for t in last_out[0][encoded["input_ids"].shape[1]:].tolist()]
    return {
        "tok_s_median": runs[len(runs) // 2],
        "tok_s_runs": tok_s_runs,
        "n_new_tokens": n_new,
        "warmup_runs": int(warmup),
        "timed_runs": len(tok_s_runs),
        "greedy": True,
        "max_new_tokens": int(max_new_tokens),
        "new_token_ids": new_ids,
        "method": "model_generate_perf_counter_ns_median_v1",
    }


def e2e_forward_logits(model: Any, inputs: dict[str, Any]) -> Any:
    """Logits de 1 forward (para cosine baseline vs candidato); None em falha."""
    try:
        with torch.inference_mode():
            out = model(**inputs)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            logits = out[0]
        return logits.detach().float().cpu() if torch.is_tensor(logits) else None
    except Exception as exc:
        print(f"[E2E] AVISO forward de logits: {exc}")
        return None


def token_exact_match(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    if n == 0:
        return {
            "exact_match_rate": 0.0,
            "n_compared": 0,
            "len_baseline": len(a),
            "len_candidate": len(b),
        }
    same = sum(1 for i in range(n) if a[i] == b[i])
    return {
        "exact_match_rate": same / n,
        "n_compared": n,
        "len_baseline": len(a),
        "len_candidate": len(b),
        "length_equal": len(a) == len(b),
    }


def run_e2e_toks_battery(
    *,
    recorder: "BatteryRecorder",
    model: Any,
    tokenizer: Any,
    device: Any,
    out_dir: Path,
    maximum_rank: int,
) -> dict[str, Any]:
    """P1_AETHER_E2E_TOKS: baseline E candidato REAIS via model.generate.

    Nunca lança: guardas degradam para SKIPPED (modelo >3e9 params, RAM/VRAM
    insuficiente) e qualquer falha de fase vira registro FAIL — a fila segue.
    O modelo é sempre restaurado no finally (transacional).
    """
    battery_id = "P1_AETHER_E2E_TOKS"
    print("\n" + "=" * 78)
    print("P1_AETHER_E2E_TOKS — tok/s de modelo completo (contracts §12)")
    print("=" * 78)

    def _skip(reason: str, extra: dict[str, Any] | None = None, note: str = "") -> dict[str, Any]:
        recorder.record(
            battery_id=battery_id,
            status="SKIPPED",
            measurement_scope=E2E_MEASUREMENT_SCOPE,
            quality={"full_local_gate_pass": None},
            metrics={"e2e": {"measured": False, "skipped": True, "reason": reason, **(extra or {})}},
            notes=(note or f"SKIPPED: {reason}. tok/s de topo permanecem null.")[:1200],
        )
        print(f"[E2E] SKIPPED: {reason}")
        return {"battery_id": battery_id, "status": "SKIPPED", "reason": reason}

    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = 0
    max_params = e2e_max_params()
    if n_params > max_params:
        return _skip(
            "modelo grande demais para o runtime de referência neste ambiente",
            {"params_total": n_params, "limit": max_params, "override": "RIFT_E2E_MAX_PARAMS"},
            note=(
                f"SKIPPED: n_params={n_params} > {max_params:.0f} (mesmo guard do "
                "GEYSER G3; override via RIFT_E2E_MAX_PARAMS). tok/s de topo "
                "permanecem null — nenhum valor estimado é publicado."
            ),
        )

    supports_generate = callable(getattr(model, "generate", None))
    try:
        can_generate = getattr(model, "can_generate", None)
        if callable(can_generate):
            supports_generate = supports_generate and bool(can_generate())
    except Exception:
        pass
    if not supports_generate:
        return _skip(
            "modelo não expõe model.generate — bateria E2E não se aplica",
            {"params_total": n_params},
        )

    blocks = e2e_find_transformer_blocks(model)
    if not blocks:
        recorder.record(
            battery_id=battery_id,
            status="FAIL",
            measurement_scope=E2E_MEASUREMENT_SCOPE,
            quality={"full_local_gate_pass": False},
            metrics={"e2e": {"measured": False}, "error": "nenhum transformer block encontrado"},
            notes="Sem blocos transformer para patch do modelo completo.",
        )
        print("[E2E] FAIL: nenhum transformer block encontrado")
        return {"battery_id": battery_id, "status": "FAIL", "reason": "sem blocos"}

    block_linear_params = 0
    for block_name, block in blocks:
        for linear in e2e_collect_linears(block, block_name).values():
            block_linear_params += int(linear.weight.numel())
    w0_cache_bytes = block_linear_params * 4     # W0 ternário fp32 cacheado
    packed_bytes_est = block_linear_params // 4  # base ternária 2-bit nos artefatos

    if device.type == "cuda":
        try:
            free_vram = int(torch.cuda.mem_get_info()[0])
        except Exception:
            free_vram = None
        if free_vram is not None and w0_cache_bytes > 0.8 * free_vram:
            return _skip(
                "VRAM insuficiente para o cache fp32 do W0 ternário",
                {
                    "w0_cache_bytes": w0_cache_bytes,
                    "free_vram_bytes": free_vram,
                    "params_total": n_params,
                },
            )
    else:
        mem_available = read_meminfo_available_bytes()
        if mem_available is not None and (w0_cache_bytes + packed_bytes_est) > 0.8 * mem_available:
            return _skip(
                "RAM insuficiente para o cache fp32 do W0 ternário (CPU)",
                {
                    "w0_cache_bytes": w0_cache_bytes,
                    "packed_bytes_est": packed_bytes_est,
                    "mem_available_bytes": mem_available,
                    "params_total": n_params,
                },
            )

    artifacts_dir = out_dir / "artifacts" / "e2e"
    patched: dict[str, tuple[Any, str, Any]] = {}
    summary: dict[str, Any] = {"battery_id": battery_id, "status": "FAIL"}
    try:
        encoded = tokenizer(E2E_GENERATION_PROMPT, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        logits_base = e2e_forward_logits(model, encoded)
        with PhaseRamSampler() as baseline_sampler:
            base_gen = measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        baseline_phase = baseline_sampler.summary()
        print(
            f"[E2E] baseline {base_gen['tok_s_median']:.2f} tok/s "
            f"({base_gen['n_new_tokens']} tokens novos, greedy)"
        )

        print(
            f"[E2E] quantizando e patchando {len(blocks)} blocos "
            "(ternário HQR + TADDS low-rank)..."
        )
        with PhaseRamSampler() as patch_sampler:
            patched, baseline_disk, candidate_disk = e2e_patch_all_blocks(
                model, blocks, artifacts_dir, maximum_rank
            )
        patch_phase = patch_sampler.summary()
        n_linears = len(patched)
        print(f"[E2E] {n_linears} nn.Linear substituídas; artefatos F0/F1 em {artifacts_dir}")

        logits_cand = e2e_forward_logits(model, encoded)
        with PhaseRamSampler() as candidate_sampler:
            cand_gen = measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        candidate_phase = candidate_sampler.summary()
        print(f"[E2E] candidato {cand_gen['tok_s_median']:.2f} tok/s")

        em = token_exact_match(base_gen["new_token_ids"], cand_gen["new_token_ids"])
        logits_q = None
        if logits_base is not None and logits_cand is not None:
            logits_q = compute_metrics(logits_base, logits_cand)
        logits_cos = logits_q["cosine"] if logits_q else None

        baseline_tok_s = float(base_gen["tok_s_median"])
        candidate_tok_s = float(cand_gen["tok_s_median"])
        speedup_x = candidate_tok_s / max(baseline_tok_s, 1e-12)
        e2e_pass = (
            baseline_tok_s > 0
            and candidate_tok_s > 0
            and logits_cos is not None
            and logits_cos >= 0.95
        )

        memory = build_memory_metrics(baseline_phase, candidate_phase)
        memory["patch_phase"] = patch_phase

        recorder.record(
            battery_id=battery_id,
            status="PASS" if e2e_pass else "FAIL",
            baseline_tok_s=baseline_tok_s,
            candidate_tok_s=candidate_tok_s,
            baseline_ram_bytes=measured_phase_max(baseline_phase),
            candidate_ram_bytes=measured_phase_max(candidate_phase),
            baseline_disk_bytes=int(baseline_disk),
            candidate_disk_bytes=int(candidate_disk),
            measurement_scope=E2E_MEASUREMENT_SCOPE,
            quality={
                "full_local_gate_pass": e2e_pass,
                "output": logits_q,
                "token_exact_match": em,
            },
            metrics={
                "e2e": {
                    "measured": True,
                    "prompt_pt_br": E2E_GENERATION_PROMPT,
                    "max_new_tokens": E2E_MAX_NEW_TOKENS,
                    "baseline": {k: v for k, v in base_gen.items() if k != "new_token_ids"},
                    "candidate": {k: v for k, v in cand_gen.items() if k != "new_token_ids"},
                    "speedup_x": speedup_x,
                    "n_blocks_patched": len(blocks),
                    "n_linears_patched": n_linears,
                    "codec": "ternário HQR + TADDS low-rank dinâmico (rank máx clampado por Linear)",
                    "runtime": "python_reference_aether_codec",
                    "maximum_rank": int(maximum_rank),
                    "w0_cache_fp32": True,
                    "original_weight_on_hot_path": False,
                    "lm_head_note": "lm_head/embeddings fora dos blocos permanecem originais",
                    "n_params": n_params,
                },
                "operation": {
                    "metric": "e2e_generate_tok_s",
                    "speedup_x": speedup_x,
                },
                "memory": memory,
            },
            notes=(
                f"E2E REAL: baseline={baseline_tok_s:.2f} tok/s "
                f"candidato={candidate_tok_s:.2f} tok/s (ambos via model.generate, "
                f"mesmo protocolo greedy). "
                f"logits_cos={'null' if logits_cos is None else round(logits_cos, 4)} "
                f"exact_match={em['exact_match_rate']:.3f}. "
                f"Disco {baseline_disk}->{candidate_disk} B (payloads F0/F1 reais). "
                "Runtime de referência Python — velocidade não representa kernel nativo."
            ),
            comparison_role="primary",
        )
        summary = {
            "battery_id": battery_id,
            "status": "PASS" if e2e_pass else "FAIL",
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "speedup_x": speedup_x,
            "logits_cosine": logits_cos,
            "token_exact_match_rate": em["exact_match_rate"],
            "baseline_disk_bytes": int(baseline_disk),
            "candidate_disk_bytes": int(candidate_disk),
            "n_blocks_patched": len(blocks),
            "n_linears_patched": n_linears,
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, MemoryError) or "out of memory" in message.lower():
            recorder.record(
                battery_id=battery_id,
                status="SKIPPED",
                measurement_scope=E2E_MEASUREMENT_SCOPE,
                quality={"full_local_gate_pass": None},
                metrics={
                    "e2e": {
                        "measured": False,
                        "skipped": True,
                        "reason": "memória insuficiente durante a fase E2E",
                    },
                    "error": message[:800],
                },
                notes=(
                    "SKIPPED: memória insuficiente durante quantização/patch/"
                    "generate; modelo restaurado, fila segue. " + message
                )[:1200],
            )
            print(f"[E2E] SKIPPED (memória insuficiente): {message}")
            summary = {"battery_id": battery_id, "status": "SKIPPED", "reason": "oom"}
        else:
            recorder.record(
                battery_id=battery_id,
                status="FAIL",
                measurement_scope=E2E_MEASUREMENT_SCOPE,
                quality={"full_local_gate_pass": False},
                metrics={"e2e": {"measured": False}, "error": message[:800]},
                notes=(
                    f"Falha na bateria E2E ({message}); modelo restaurado, fila segue."
                )[:1200],
            )
            print(f"[E2E] FAIL: {message}")
            summary = {"battery_id": battery_id, "status": "FAIL", "error": message[:500]}
    finally:
        if patched:
            e2e_restore_patched(patched)
            print("[E2E] modelo restaurado (unpatch de todos os blocos)")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return summary


def run_phase1(args: argparse.Namespace) -> Path:
    ensure_ml_dependencies()
    device = resolve_torch_device(args.device)
    model_id = normalize_huggingface_model_id(args.model)
    print(f"[Phase1] Carregando {model_id} em {device}...")
    try:
        tokenizer, model = load_model(model_id, device=device, trust_remote_code=args.trust_remote_code)
    except Exception as load_exc:
        print(f"[Phase1] FALHA ao carregar modelo/tokenizer: {load_exc}")
        out_dir = Path(getattr(args, "out", None) or "/tmp/phase1_load_fail")
        out_dir.mkdir(parents=True, exist_ok=True)
        recorder = BatteryRecorder(
            out_dir,
            model_id=model_id,
            publish_mode=getattr(args, "publish", "auto"),
            results_endpoint=getattr(args, "results_endpoint", None),
            device=str(device),
        )
        recorder.record(
            battery_id="P1_LOAD_MODEL",
            status="FAIL",
            measurement_scope="model_load",
            quality={"full_local_gate_pass": False},
            metrics={"error": str(load_exc)[:800]},
            notes=(
                f"Falha ao carregar {model_id}. "
                + (
                    "Modelo gated/restrito: aceite os termos no Hugging Face e configure o Secret HF_TOKEN no Colab. "
                    if is_gated_access_error(load_exc) else
                    "Modelos diffusers/vídeo ou formato incompatível podem falhar nesta bateria CausalLM. "
                )
                + f"Detalhe: {load_exc}"
            )[:1200],
        )
        return recorder.json_path
    target_layer = resolve_linear_weight_name(model, args.target_layer)
    print(f"[Phase1] Camada Linear selecionada: {target_layer}")
    weight = model.state_dict()[target_layer].detach().to(device=device, dtype=torch.float32)
    out_features, in_features = weight.shape

    try:
        x = capture_activation(
            model, tokenizer, target_layer.removesuffix(".weight"), device, args.prompt,
        ).reshape(-1, in_features).to(dtype=torch.float32)
        activation_source = "real_model_activation"
    except Exception as exc:
        print(f"[WARN] Ativação real indisponível: {exc}; usando fallback determinístico.")
        torch.manual_seed(1234)
        x = torch.randn(16, in_features, device=device, dtype=torch.float32)
        activation_source = "synthetic_fallback"

    ternary, scale, sparsity = ternary_quantize(weight)
    maximum_rank = min(max(1, args.maximum_rank), out_features, in_features)
    residual = weight - ternary
    print(f"[AETHER] Decomposição TADDS low-rank máxima: r={maximum_rank}")
    u, s, v = torch.svd_lowrank(residual, q=maximum_rank, niter=2)
    entropy, ranks, thresholds = entropy_and_ranks(x, maximum_rank)

    with torch.no_grad():
        y_reference = F.linear(x, weight)
        y_base = F.linear(x, ternary)
        y_dynamic = aether_dynamic_linear(x, ternary, u, s, v, ranks)
        y_full = y_base + lowrank_correction(x, u, s, v, maximum_rank)
        weight_full = ternary + (u * s) @ v.T

    q_weight_base = compute_metrics(weight, ternary)
    q_weight_full = compute_metrics(weight, weight_full)
    q_base = compute_metrics(y_reference, y_base)
    q_dynamic = compute_metrics(y_reference, y_dynamic)
    q_full = compute_metrics(y_reference, y_full)
    quality_pass = q_dynamic["cosine"] >= 0.995 and q_dynamic["nrmse"] <= 0.05

    with PhaseRamSampler() as baseline_sampler:
        baseline_perf = benchmark_ms(
            lambda: F.linear(x, weight), device=device, iterations=args.iterations,
        )
    baseline_ram_measured = baseline_sampler.summary()
    with PhaseRamSampler() as base_sampler:
        base_perf = benchmark_ms(
            lambda: F.linear(x, ternary), device=device, iterations=args.iterations,
        )
    base_ram_measured = base_sampler.summary()
    with PhaseRamSampler() as dynamic_sampler:
        dynamic_perf = benchmark_ms(
            lambda: aether_dynamic_linear(x, ternary, u, s, v, ranks),
            device=device,
            iterations=args.iterations,
        )
    dynamic_ram_measured = dynamic_sampler.summary()
    speedup = baseline_perf["median_ms"] / dynamic_perf["median_ms"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(
        out_dir / "aether_bundle_m0.aeth",
        model_id=model_id,
        target_layer=target_layer,
        ternary=ternary,
        scale=scale,
        u=u,
        s=s,
        v=v,
        thresholds=thresholds,
    )
    golden = run_golden_header_tests()
    print(
        f"[B0] GOLDEN HEADER + AETHER-IR + MMAP PASS — "
        f"XXH3-64=0x{golden['checksum']:016x}, negativos={golden['negative_tests_passed']}"
    )

    baseline_disk = int(weight.numel() * 4)
    base_disk = int(bundle["hqr_packed_bytes"])
    candidate_disk = int(bundle["file_size"])
    input_bytes = int(x.numel() * 4)
    output_bytes = int(y_reference.numel() * 4)
    # Estimativas aritméticas de working set: só em metrics.memory.estimated_* (nunca nível superior).
    estimated_baseline_ram = baseline_disk + input_bytes + output_bytes
    estimated_base_ram = int(ternary.numel() * 4 + input_bytes + output_bytes)
    factor_ram = int((u.numel() + s.numel() + v.numel()) * 4)
    estimated_candidate_ram = estimated_base_ram + factor_ram
    # Nível superior: só RSS medido por fase (VmRSS); sem medição → None.
    baseline_ram = measured_phase_max(baseline_ram_measured)
    base_ram = measured_phase_max(base_ram_measured)
    candidate_ram = measured_phase_max(dynamic_ram_measured)
    active_rate = float((ranks > 0).float().mean().item())
    mean_rank = float(ranks.float().mean().item())

    recorder = BatteryRecorder(out_dir, model_id=model_id, publish_mode=args.publish, results_endpoint=args.results_endpoint, device=str(device))
    recorder.record(
        battery_id="B0_AETHER_BINARY_IR_FOUNDATION",
        status="PASS",
        measurement_scope="AETHER Header/IR/Stage Pages/MMAP correctness; desempenho não se aplica.",
        quality={"full_local_gate_pass": True},
        metrics={"bundle": bundle, "negative_tests_passed": golden["negative_tests_passed"]},
        notes="Container AETH v1.0 test-local; ABI de produção ainda não congelada.",
    )
    recorder.record(
        battery_id="P1_AETHER_HQR_TERNARY_2BIT",
        status="EXPERIMENTAL_PASS" if q_base["cosine"] >= 0.95 else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=base_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=base_disk,
        measurement_scope="Single Linear op; base ternária realmente empacotada em 2 bits; RAM de nível superior é RSS medido por fase (VmRSS) quando disponível; working set estimado fica em metrics.memory.estimated_*.",
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {
            "metric": "linear_latency",
            "baseline_median_ms": baseline_perf["median_ms"],
            "candidate_median_ms": base_perf["median_ms"],
            "speedup_x": baseline_perf["median_ms"] / base_perf["median_ms"],
        }, "memory": build_memory_metrics(
            baseline_ram_measured, base_ram_measured,
            estimated_baseline_bytes=estimated_baseline_ram,
            estimated_candidate_bytes=estimated_base_ram,
        ), "aether": {"ternary_sparsity": sparsity, "hqr_ans_native": False}},
        notes="HQR ANS 0.85-bit não implementado; o dashboard usa somente bytes físicos medidos.",
    )
    recorder.record(
        battery_id="P1_AETHER_HQR_PLUS_TADDS_DYNAMIC",
        status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=candidate_disk,
        measurement_scope="Single Linear op; bundle AETH real; TADDS por entropia; latência do path Python/PyTorch; RAM de nível superior é RSS medido por fase (VmRSS) quando disponível; Tok/s não medido.",
        quality={
            "full_local_gate_pass": quality_pass,
            "weight": q_weight_full,
            "output": q_dynamic,
            "output_base": q_base,
            "output_rank_max": q_full,
        },
        metrics={
            "operation": {
                "metric": "linear_latency",
                "baseline_median_ms": baseline_perf["median_ms"],
                "candidate_median_ms": dynamic_perf["median_ms"],
                "speedup_x": speedup,
                "rows_processed": int(x.shape[0]),
            },
            "memory": build_memory_metrics(
                baseline_ram_measured, dynamic_ram_measured,
                estimated_baseline_bytes=estimated_baseline_ram,
                estimated_candidate_bytes=estimated_candidate_ram,
            ),
            "aether": {
                "maximum_rank": maximum_rank,
                "mean_selected_rank": mean_rank,
                "stage_activation_rate": active_rate,
                "rank_histogram": {str(rank): int((ranks == rank).sum().item()) for rank in sorted(set(ranks.tolist()))},
                "entropy_mean": float(entropy.mean().item()),
                "entropy_thresholds": thresholds,
                "activation_source": activation_source,
                "hqr_ans_native": False,
                "pio_async_native": False,
                "srfa_fused_native": False,
                "ring_buffer_budget_bytes": int(args.ring_buffer_mb * 1024 * 1024),
            },
            "bundle": bundle,
        },
        notes="P-IO e SRFA são simulados. Nenhum ganho nativo de inferência deve ser reivindicado.",
        comparison_role="primary",
    )
    recorder.record(
        battery_id="P1_AETHER_PIO_POLICY_SIM",
        status="SIMULATED",
        measurement_scope="Política de prefetch derivada dos ranks do TADDS; nenhuma leitura NVMe assíncrona foi medida.",
        quality={"full_local_gate_pass": None},
        metrics={"aether": {
            "predicted_refinement_rows": int((ranks > 0).sum().item()),
            "stage_activation_rate": active_rate,
            "ring_buffer_budget_bytes": int(args.ring_buffer_mb * 1024 * 1024),
            "pio_async_native": False,
        }},
        notes="A bateria mede somente decisões da política, não I/O real.",
    )

    # ------------------------------------------------------------------
    # P1_AETHER_E2E_TOKS — tok/s de modelo completo (docs/C3_CONTRACTS_V1.md §12)
    # ------------------------------------------------------------------
    try:
        e2e_summary = run_e2e_toks_battery(
            recorder=recorder,
            model=model,
            tokenizer=tokenizer,
            device=device,
            out_dir=out_dir,
            maximum_rank=args.maximum_rank,
        )
    except Exception as e2e_exc:
        # Nunca derruba a fila: a bateria já registra FAIL/SKIPPED internamente;
        # este guard cobre apenas erro inesperado de gravação/publicação.
        print(f"[E2E] AVISO: bateria P1_AETHER_E2E_TOKS abortada: {e2e_exc}")
        e2e_summary = {
            "battery_id": "P1_AETHER_E2E_TOKS",
            "status": "ERROR",
            "error": str(e2e_exc)[:500],
        }

    report = {
        "spec": "AETHER-LM v1.0 reference",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "target_layer": target_layer,
        "shape": [out_features, in_features],
        "quality": {"base": q_base, "dynamic": q_dynamic, "rank_max": q_full, "gate_pass": quality_pass},
        "aether": {"active_rate": active_rate, "mean_rank": mean_rank, "thresholds": thresholds},
        "performance": {"baseline": baseline_perf, "base": base_perf, "dynamic": dynamic_perf, "speedup_x": speedup},
        "e2e_toks": e2e_summary,
        "bundle": bundle,
    }
    (out_dir / "aether_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("AETHER-LM PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer}")
    print(f"TADDS ativo             : {active_rate * 100:.1f}% | rank médio={mean_rank:.2f}")
    print(f"Qualidade dinâmica      : cosine={q_dynamic['cosine']:.6f} / NRMSE={q_dynamic['nrmse']:.6f}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Baseline Linear         : {baseline_perf['median_ms']:.4f} ms")
    print(f"AETHER ref dinâmico     : {dynamic_perf['median_ms']:.4f} ms | {speedup:.3f}x")
    if e2e_summary.get("baseline_tok_s") is not None:
        print(
            f"E2E tok/s (REAL)        : baseline={e2e_summary['baseline_tok_s']:.2f} "
            f"candidato={e2e_summary['candidate_tok_s']:.2f} "
            f"({e2e_summary.get('speedup_x', 0.0):.3f}x — runtime de referência Python)"
        )
    else:
        print(f"E2E tok/s               : {e2e_summary.get('status', 'SKIPPED')}")
    print("HQR ANS / P-IO / SRFA   : NÃO IMPLEMENTADOS — path Python de referência.")
    print(f"Baterias JSON           : {recorder.json_path}")
    print("=" * 78)
    return recorder.json_path


def without_ipykernel_connection_args(argv: Iterable[str]) -> list[str]:
    values = list(argv)
    filtered = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "-f" and index + 1 < len(values):
            name = Path(values[index + 1]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 2
                continue
        if value.startswith("-f="):
            name = Path(value[3:]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 1
                continue
        filtered.append(value)
        index += 1
    return filtered


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AETHER-LM v1.0 M0 + Phase 1 reference test")
    parser.add_argument("--mode", choices=["self-test", "phase1"], default="phase1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="org/modelo ou URL do Hugging Face")
    parser.add_argument("--target-layer", default="auto", help="Tensor .weight ou auto")
    parser.add_argument("--prompt", default="Explique por que memória importa na inferência de modelos.")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu — auto usa GPU se houver, senão CPU")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--maximum-rank", type=int, default=16)
    parser.add_argument("--ring-buffer-mb", type=int, default=320)
    parser.add_argument("--out", default="aether_m0_test_output")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--publish", choices=["auto", "required", "off"], default=os.environ.get("RIFT_PUBLISH_MODE", "auto"))
    parser.add_argument("--results-endpoint", default=None, help="URL HTTPS /api/results da Vercel")
    values = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(without_ipykernel_connection_args(values))
    if args.maximum_rank < 1:
        parser.error("--maximum-rank precisa ser positivo")
    if args.iterations < 1:
        parser.error("--iterations precisa ser positivo")
    if args.ring_buffer_mb < 0:
        parser.error("--ring-buffer-mb não pode ser negativo")
    try:
        if args.mode == "self-test":
            golden = run_golden_header_tests()
            print(
                f"AETHER SELF-TEST PASS — XXH3-64=0x{golden['checksum']:016x}, "
                f"negativos={golden['negative_tests_passed']}"
            )
            return
        batteries_path = run_phase1(args)
        publish_to_vercel(batteries_path, mode=args.publish, endpoint=args.results_endpoint)
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc
    finally:
        cleanup_colab_workspace(label="AETHER")


if __name__ == "__main__":
    main()
