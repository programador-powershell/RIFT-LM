#!/usr/bin/env python3
"""SPECTRA-LM v0.1 — M0/Phase 1 reference battery for Colab.

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
import traceback
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




SPECTRA_MAGIC = b"SPCT"
SPECTRA_VERSION_M0 = 0x0100
SPECTRA_HEADER_SIZE = 128
SPECTRA_HEADER_FORMAT = "<4sHHIIQQQQQQQ56s"
SPECTRA_CHECKSUM_OFFSET = 64
SPECTRA_CHECKSUM_SEED = 42
STAGE_ENTRY_FORMAT = "<IIQQQ"
STAGE_ENTRY_SIZE = struct.calcsize(STAGE_ENTRY_FORMAT)
EXPECTED_GOLDEN_CHECKSUM = 0xDDB2AADA219F6C40
assert struct.calcsize(SPECTRA_HEADER_FORMAT) == SPECTRA_HEADER_SIZE
assert STAGE_ENTRY_SIZE == 32

BENCHMARK_PROTOCOL = "LINEAR_REFERENCE_V2"
SCHEMA_VERSION = 2

# Bateria E2E de tok/s (docs/C3_CONTRACTS_V1.md §12): prompt PT-BR fixo + greedy.
E2E_GENERATION_PROMPT = "Liste três técnicas para reduzir o uso de memória na inferência de LLMs:"
E2E_MAX_NEW_TOKENS = 48
E2E_MAX_PARAMS = 3_000_000_000  # mesmo guard do G3_GEYSER_BURST: acima disso → SKIPPED


class SpectraFormatError(ValueError):
    """Formato inválido no container SPCT (header, offsets ou checksums)."""


def align_up(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def xxh3_64(data: bytes) -> int:
    return int(xxhash.xxh3_64_intdigest(data, seed=SPECTRA_CHECKSUM_SEED))


def create_spectra_header(
    *,
    ir_offset: int,
    stage_table_offset: int,
    stage_count: int,
    prediction_table_offset: int,
    payload_offset: int,
    file_size: int,
    magic: bytes = SPECTRA_MAGIC,
    version: int = SPECTRA_VERSION_M0,
) -> bytes:
    raw = struct.pack(
        SPECTRA_HEADER_FORMAT,
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
    return raw[:SPECTRA_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def parse_spectra_header(data: bytes, actual_file_size: int) -> dict[str, int]:
    if len(data) < SPECTRA_HEADER_SIZE:
        raise SpectraFormatError("truncated_header")
    values = struct.unpack(SPECTRA_HEADER_FORMAT, data[:SPECTRA_HEADER_SIZE])
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
    if magic != SPECTRA_MAGIC:
        raise SpectraFormatError("bad_magic")
    if version != SPECTRA_VERSION_M0:
        raise SpectraFormatError("bad_version")
    zeroed = bytearray(data[:SPECTRA_HEADER_SIZE])
    zeroed[SPECTRA_CHECKSUM_OFFSET:72] = bytes(8)
    if xxh3_64(bytes(zeroed)) != checksum:
        raise SpectraFormatError("checksum_mismatch")
    if file_size != actual_file_size:
        raise SpectraFormatError("file_size_mismatch")
    offsets = (ir_offset, stage_table_offset, prediction_table_offset, payload_offset)
    if any(offset < SPECTRA_HEADER_SIZE or offset > file_size for offset in offsets):
        raise SpectraFormatError("offset_out_of_range")
    if offsets != tuple(sorted(offsets)):
        raise SpectraFormatError("offset_order")
    if stage_count > 1_000_000:
        raise SpectraFormatError("stage_count_overflow")
    if stage_table_offset + stage_count * STAGE_ENTRY_SIZE > prediction_table_offset:
        raise SpectraFormatError("stage_table_overlap")
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
    current = dict(zip(names, struct.unpack(SPECTRA_HEADER_FORMAT, header)))
    current.update(changes)
    current["checksum"] = 0
    raw = struct.pack(SPECTRA_HEADER_FORMAT, *(current[name] for name in names))
    checksum = xxh3_64(raw)
    return raw[:SPECTRA_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def run_golden_header_tests() -> dict[str, Any]:
    golden = create_spectra_header(
        ir_offset=128,
        stage_table_offset=128,
        stage_count=0,
        prediction_table_offset=128,
        payload_offset=128,
        file_size=128,
    )
    parsed = parse_spectra_header(golden, 128)
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
            parse_spectra_header(mutated, actual_size)
        except SpectraFormatError:
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


def spectra_dynamic_linear(x: Any, ternary: Any, u: Any, s: Any, v: Any, ranks: Any) -> Any:
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
        "engine": "SPECTRA_REFERENCE_PYTHON",
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
    ir_offset = SPECTRA_HEADER_SIZE
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
    header = create_spectra_header(
        ir_offset=ir_offset,
        stage_table_offset=stage_table_offset,
        stage_count=len(stages),
        prediction_table_offset=prediction_table_offset,
        payload_offset=payload_offset,
        file_size=file_size,
    )
    blob = bytearray(file_size)
    blob[:SPECTRA_HEADER_SIZE] = header
    blob[ir_offset:ir_offset + len(ir)] = ir
    blob[stage_table_offset:stage_table_offset + len(stage_table)] = stage_table
    blob[prediction_table_offset:prediction_table_offset + len(prediction)] = prediction
    blob[base_offset:base_offset + len(base_payload)] = base_payload
    blob[residual_offset:residual_offset + len(residual_payload)] = residual_payload
    path.write_bytes(blob)
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            parsed = parse_spectra_header(mapped[:SPECTRA_HEADER_SIZE], len(mapped))
            for stage_id, rank, offset, size, checksum in stages:
                if xxh3_64(mapped[offset:offset + size]) != checksum:
                    raise SpectraFormatError(f"stage_checksum_{stage_id}")
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
    """Ganho percentual quando maior é melhor (tok/s)."""
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


# ---------------------------------------------------------------------------
# P1_SPECTRA_E2E_TOKS — tok/s de topo REAL baseline E candidato
# (docs/C3_CONTRACTS_V1.md §12; crib de C3InlineLinearModule/CascadeLinearModule
# de c3_methodology_auto_batteries.py — W denso original FORA do caminho quente)
# ---------------------------------------------------------------------------


def e2e_params_limit() -> float:
    """Limite de parâmetros do e2e (mesmo guard do G3_GEYSER_BURST); override por env."""
    raw = os.environ.get("RIFT_E2E_MAX_PARAMS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            print(f"[E2E] AVISO: RIFT_E2E_MAX_PARAMS inválido ({raw}); usando {E2E_MAX_PARAMS}")
    return float(E2E_MAX_PARAMS)


def find_transformer_blocks(model: Any) -> list[tuple[str, Any]]:
    """Localiza a lista de blocos transformer (crib de cascade/compiler/block_decompose.py).

    Cobre Qwen, Llama, Phi, Gemma, GPT-NeoX e fallback genérico por ModuleList.
    """
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
        node = model
        ok = True
        for part in attr.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok and isinstance(node, (torch.nn.ModuleList, list)) and len(node) > 0:
            return [(f"{attr}.{i}", node[i]) for i in range(len(node))]
    for name, node in model.named_modules():
        if not isinstance(node, torch.nn.ModuleList) or len(node) == 0:
            continue
        if name.split(".")[-1] not in ("layers", "h", "blocks", "layer"):
            continue
        if any(isinstance(m, torch.nn.Linear) for m in node[0].modules()):
            return [(f"{name}.{i}", node[i]) for i in range(len(node))]
    return []


def set_module_by_path(root: Any, dotted: str, new_module: Any) -> None:
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


def collect_block_linears(block: Any, block_name: str) -> dict[str, Any]:
    """Mapa nome-completo -> nn.Linear do bloco (originais guardados ANTES da troca)."""
    out: dict[str, Any] = {}
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear):
            out[f"{block_name}.{name}" if name else block_name] = module
    return out


def restore_block_linears(block: Any, originals: dict[str, Any], block_name: str) -> None:
    """Devolve as nn.Linear originais ao bloco (unpatch transacional)."""
    for full, linear in originals.items():
        short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
        try:
            set_module_by_path(block, short, linear)
        except Exception as exc:
            print(f"[E2E] AVISO ao restaurar {full}: {exc}")


def forward_logits(model: Any, inputs: dict[str, Any]) -> Any:
    """1 forward para logits (qualidade e2e); None quando o modelo não expõe logits."""
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


def measure_generate_tok_s(model: Any, tokenizer: Any, prompt: str, device: Any, *,
                           max_new_tokens: int, warmup: int = 2, timed: int = 3) -> dict[str, Any]:
    """Tok/s REAL de model.generate (greedy) — MESMO protocolo para baseline e candidato.

    Contrato §12: >=2 warmup + >=3 medições, mediana, perf_counter_ns e
    torch.cuda.synchronize quando CUDA.
    """
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {key: value.to(device) for key, value in enc.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            model.generate(**enc, **{**gen_kwargs, "max_new_tokens": min(8, int(max_new_tokens))})
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
            out = model.generate(**enc, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        n_new = int(out.shape[1] - enc["input_ids"].shape[1])
        tok_s_runs.append(n_new / max(elapsed_s, 1e-9))
        last_out = out
    ordered = sorted(tok_s_runs)
    new_token_ids: list[int] = []
    if last_out is not None:
        new_token_ids = [int(t) for t in last_out[0][enc["input_ids"].shape[1]:].tolist()]
    return {
        "tok_s_median": ordered[len(ordered) // 2],
        "tok_s_runs": tok_s_runs,
        "n_new_tokens": n_new,
        "warmup_runs": int(warmup),
        "timed_runs": len(tok_s_runs),
        "greedy": True,
        "max_new_tokens": int(max_new_tokens),
        "new_token_ids": new_token_ids,
        "method": "model_generate_perf_counter_ns_median_v1",
    }


def token_exact_match(a: list[int], b: list[int]) -> dict[str, Any]:
    """Exact-match posicional dos token ids gerados (baseline vs candidato)."""
    n = min(len(a), len(b))
    if n == 0:
        return {"exact_match_rate": 0.0, "n_compared": 0, "len_baseline": len(a), "len_candidate": len(b)}
    same = sum(1 for i in range(n) if a[i] == b[i])
    return {
        "exact_match_rate": same / n,
        "n_compared": n,
        "len_baseline": len(a),
        "len_candidate": len(b),
        "length_equal": len(a) == len(b),
    }


def write_e2e_artifacts(artifacts_dir: Path, prefix: str, *, f0_payload: bytes, f1_payload: bytes) -> dict[str, Any]:
    """Grava payloads F0/F1 REAIS em <out>/artifacts/e2e e retorna bytes via os.stat."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    f0_path = artifacts_dir / f"{prefix}_f0.bin"
    f1_path = artifacts_dir / f"{prefix}_f1.bin"
    f0_path.write_bytes(f0_payload)
    f1_path.write_bytes(f1_payload)
    f0_bytes = int(os.stat(f0_path).st_size)
    f1_bytes = int(os.stat(f1_path).st_size)
    return {
        "f0_path": str(f0_path),
        "f1_path": str(f1_path),
        "f0_bytes": f0_bytes,
        "f1_bytes": f1_bytes,
        "total_bytes": f0_bytes + f1_bytes,
        "method": "binary_os_stat_v1",
    }


_SPECTRA_INLINE_LINEAR_CLS = None


def spectra_inline_linear_class():
    """Classe criada sob demanda (torch é dependência tardia neste script)."""
    global _SPECTRA_INLINE_LINEAR_CLS
    if _SPECTRA_INLINE_LINEAR_CLS is not None:
        return _SPECTRA_INLINE_LINEAR_CLS

    class SpectraInlineLinear(torch.nn.Module):
        """Runtime de referência SPECTRA (ternário HQR + TADDS dinâmico) do generate e2e.

        Quantiza W UMA vez no __init__ e cacheia o F0 dequantizado em fp32; o W denso
        original fica FORA do caminho quente — apenas ternário + fatores residem aqui
        (crib de C3InlineLinearModule/CascadeLinearModule de
        c3_methodology_auto_batteries.py). Bias, quando existe, permanece fp32.
        """

        def __init__(self, linear: Any, maximum_rank: int):
            super().__init__()
            weight = linear.weight.detach().to(dtype=torch.float32)
            out_features, in_features = weight.shape
            ternary, scale, sparsity = ternary_quantize(weight)
            rank = min(max(1, int(maximum_rank)), out_features, in_features)
            u, s, v = torch.svd_lowrank(weight - ternary, q=rank, niter=2)
            # Fatores gravados em FP16 no artefato → o runtime usa o round-trip FP16
            # (o que está em RAM corresponde ao que está no disco).
            u = u.to(torch.float16).to(torch.float32).contiguous()
            s = s.to(torch.float16).to(torch.float32).contiguous()
            v = v.to(torch.float16).to(torch.float32).contiguous()
            self.register_buffer("w0", ternary.contiguous())
            self.register_buffer("u", u)
            self.register_buffer("s", s)
            self.register_buffer("v", v)
            if linear.bias is not None:
                self.register_buffer("bias_fp32", linear.bias.detach().to(dtype=torch.float32).contiguous())
            else:
                self.register_buffer("bias_fp32", None)
            self.out_features = int(out_features)
            self.in_features = int(in_features)
            self.maximum_rank = int(rank)
            self.ternary_scale = float(scale)
            self.ternary_sparsity = float(sparsity)
            self.rows_processed = 0
            self.rows_refined = 0
            self.rank_sum = 0
            # Payloads binários reais (gravados uma vez em <out>/artifacts/e2e e descartados)
            self.f0_payload: bytes | None = struct.pack("<f", scale) + pack_ternary(ternary)
            self.f1_payload: bytes | None = b"".join(
                tensor.detach().cpu().to(dtype=torch.float16).contiguous().numpy().tobytes()
                for tensor in (u, s, v)
            )

        def forward(self, x: Any) -> Any:
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
            _entropy, ranks, _thresholds = entropy_and_ranks(x2, self.maximum_rank)
            y = spectra_dynamic_linear(x2, self.w0, self.u, self.s, self.v, ranks)
            if self.bias_fp32 is not None:
                y = y + self.bias_fp32
            self.rows_processed += int(x2.shape[0])
            self.rows_refined += int((ranks > 0).sum().item())
            self.rank_sum += int(ranks.sum().item())
            if y.dtype != x.dtype:
                y = y.to(dtype=x.dtype)
            return y.reshape(*orig_shape[:-1], self.out_features)

    _SPECTRA_INLINE_LINEAR_CLS = SpectraInlineLinear
    return _SPECTRA_INLINE_LINEAR_CLS


def run_e2e_tok_s_battery(recorder: "BatteryRecorder", *, model: Any, tokenizer: Any,
                          device: Any, out_dir: Path, maximum_rank: int) -> dict[str, Any] | None:
    """P1_SPECTRA_E2E_TOKS (contrato §12): baseline E candidato REAIS via model.generate.

    Transacional: os módulos originais são guardados ANTES da troca e restaurados no
    finally; falha em qualquer fase → registro FAIL e a run segue.
    """
    battery_id = "P1_SPECTRA_E2E_TOKS"
    n_params = sum(int(p.numel()) for p in model.parameters())
    limit = e2e_params_limit()
    if n_params > limit:
        recorder.record(
            battery_id=battery_id,
            status="SKIPPED",
            measurement_scope=(
                "e2e tok/s não executado: modelo acima do limite de parâmetros para o "
                "runtime de referência Python — velocidade não representa kernel nativo."
            ),
            quality={"full_local_gate_pass": None},
            metrics={"e2e": {
                "measured": False,
                "skipped": True,
                "n_params": n_params,
                "limit": int(limit),
                "override_env": "RIFT_E2E_MAX_PARAMS",
            }},
            notes=(
                f"Modelo com {n_params / 1e9:.2f}B parâmetros excede o limite de "
                f"{limit / 1e9:.0f}e9 (mesmo guard do G3_GEYSER_BURST); o patch fp32 de "
                "referência de todas as Linears causaria OOM neste ambiente."
            ),
        )
        print(f"[E2E] {battery_id}: SKIPPED (n_params={n_params / 1e9:.2f}B > {limit / 1e9:.0f}B)")
        return None

    # Guardas pré-voo espelhadas de RIFT/AETHER: limitação de ambiente vira
    # SKIPPED (nunca FAIL) — a fila serial segue.
    def _skip_env(reason: str, extra: dict[str, Any] | None = None) -> None:
        recorder.record(
            battery_id=battery_id,
            status="SKIPPED",
            measurement_scope=(
                "e2e tok/s não executado: " + reason + " — runtime de referência "
                "Python; velocidade não representa kernel nativo."
            ),
            quality={"full_local_gate_pass": None},
            metrics={"e2e": {"measured": False, "skipped": True, "reason": reason, **(extra or {})}},
            notes=f"SKIPPED: {reason}. tok/s de topo permanecem null.",
        )
        print(f"[E2E] {battery_id}: SKIPPED ({reason})")

    supports_generate = callable(getattr(model, "generate", None))
    try:
        can_generate = getattr(model, "can_generate", None)
        if callable(can_generate):
            supports_generate = supports_generate and bool(can_generate())
    except Exception:
        pass
    if not supports_generate:
        _skip_env("modelo não expõe model.generate — bateria E2E não se aplica",
                  {"n_params": n_params})
        return None

    guard_params = 0
    for guard_block_name, guard_block in find_transformer_blocks(model):
        for guard_linear in collect_block_linears(guard_block, guard_block_name).values():
            guard_params += int(guard_linear.weight.numel())
    w0_cache_bytes = guard_params * 4       # W0 ternário fp32 cacheado
    packed_bytes_est = guard_params // 4    # base ternária 2-bit nos artefatos
    if getattr(device, "type", "") == "cuda":
        try:
            free_vram = int(torch.cuda.mem_get_info()[0])
        except Exception:
            free_vram = None
        if free_vram is not None and w0_cache_bytes > 0.8 * free_vram:
            _skip_env("VRAM insuficiente para o cache fp32 do W0 ternário",
                      {"w0_cache_bytes": w0_cache_bytes, "free_vram_bytes": free_vram,
                       "n_params": n_params})
            return None
    else:
        mem_available = read_meminfo_available_bytes()
        if mem_available is not None and (w0_cache_bytes + packed_bytes_est) > 0.8 * mem_available:
            _skip_env("RAM insuficiente para o cache fp32 do W0 ternário (CPU)",
                      {"w0_cache_bytes": w0_cache_bytes, "packed_bytes_est": packed_bytes_est,
                       "mem_available_bytes": mem_available, "n_params": n_params})
            return None

    patched: list[tuple[Any, str, dict[str, Any]]] = []
    replaced_modules: list[Any] = []
    artifacts_dir = out_dir / "artifacts" / "e2e"
    try:
        blocks = find_transformer_blocks(model)
        if not blocks:
            raise RuntimeError("find_transformer_blocks não localizou blocos transformer")
        print(f"[E2E] baseline model.generate (greedy, max_new_tokens={E2E_MAX_NEW_TOKENS})...")
        enc = tokenizer(E2E_GENERATION_PROMPT, return_tensors="pt")
        enc = {key: value.to(device) for key, value in enc.items()}
        logits_base = forward_logits(model, enc)
        with PhaseRamSampler() as base_sampler:
            base_gen = measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        base_ram_phase = base_sampler.summary()
        print(f"[E2E] baseline {base_gen['tok_s_median']:.2f} tok/s ({base_gen['n_new_tokens']} tokens novos)")

        inline_cls = spectra_inline_linear_class()
        artifact_bytes = 0
        baseline_disk = 0
        n_linears = 0

        def _patch_all() -> None:
            nonlocal artifact_bytes, baseline_disk, n_linears
            for block_name, block in blocks:
                originals = collect_block_linears(block, block_name)
                # Transacional: originais guardados ANTES de qualquer troca deste bloco.
                patched.append((block, block_name, originals))
                for full, linear in originals.items():
                    module = inline_cls(linear, maximum_rank)
                    art = write_e2e_artifacts(
                        artifacts_dir, full.replace(".", "_"),
                        f0_payload=module.f0_payload, f1_payload=module.f1_payload,
                    )
                    module.f0_payload = None
                    module.f1_payload = None
                    artifact_bytes += int(art["total_bytes"])
                    # Referência FP32 (numel*4): mesma base do baseline_disk_bytes
                    # dos E2E de RIFT/AETHER — os quatro *_E2E_TOKS dividem o mesmo
                    # card "Antes" no dashboard (§13.1).
                    baseline_disk += int(linear.weight.numel() * 4)
                    short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
                    set_module_by_path(block, short, module)
                    replaced_modules.append(module)
                    n_linears += 1

        print(f"[E2E] patch de todas as Linears de {len(blocks)} blocos (ternário + TADDS)...")
        with PhaseRamSampler() as patch_sampler:
            _patch_all()
        patch_ram_phase = patch_sampler.summary()
        logits_cand = forward_logits(model, enc)
        with PhaseRamSampler() as cand_sampler:
            cand_gen = measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        cand_ram_phase = cand_sampler.summary()
        print(f"[E2E] candidato {cand_gen['tok_s_median']:.2f} tok/s")

        em = token_exact_match(base_gen["new_token_ids"], cand_gen["new_token_ids"])
        logits_q = (
            compute_metrics(logits_base, logits_cand)
            if (logits_base is not None and logits_cand is not None) else None
        )
        logits_cos = logits_q["cosine"] if logits_q else None
        baseline_tok_s = float(base_gen["tok_s_median"])
        candidate_tok_s = float(cand_gen["tok_s_median"])
        e2e_pass = (
            baseline_tok_s > 0 and candidate_tok_s > 0
            and logits_cos is not None and logits_cos >= 0.95
        )
        rows_processed = sum(m.rows_processed for m in replaced_modules)
        rows_refined = sum(m.rows_refined for m in replaced_modules)
        rank_sum = sum(m.rank_sum for m in replaced_modules)
        memory = build_memory_metrics(base_ram_phase, cand_ram_phase)
        memory["patch_phase"] = patch_ram_phase
        memory["baseline_phase_scope"] = "model.generate baseline (modelo original completo)"
        memory["candidate_phase_scope"] = "model.generate candidato (todas as Linears dos blocos patchadas)"
        recorder.record(
            battery_id=battery_id,
            status="PASS" if e2e_pass else "FAIL",
            baseline_tok_s=baseline_tok_s,
            candidate_tok_s=candidate_tok_s,
            baseline_ram_bytes=measured_phase_max(base_ram_phase),
            candidate_ram_bytes=measured_phase_max(cand_ram_phase),
            baseline_disk_bytes=baseline_disk,
            candidate_disk_bytes=artifact_bytes,
            measurement_scope=(
                "Modelo completo: baseline E candidato via model.generate (prompt PT-BR fixo, "
                f"greedy, max_new_tokens={E2E_MAX_NEW_TOKENS}, 2 warmup + 3 medições, mediana, "
                f"cuda sync); candidato = TODAS as {n_linears} nn.Linear dos {len(blocks)} blocos "
                "no runtime de referência Python — velocidade não representa kernel nativo — do "
                "codec SPECTRA (ternário HQR + TADDS); RAM topo = pico VmRSS por fase; disco "
                "candidato = os.stat de artefatos F0/F1 reais em artifacts/e2e."
            ),
            quality={
                "full_local_gate_pass": e2e_pass,
                "output": logits_q,
                "token_exact_match": em,
            },
            metrics={
                "e2e": {
                    "measured": True,
                    "baseline": {k: v for k, v in base_gen.items() if k != "new_token_ids"},
                    "candidate": {k: v for k, v in cand_gen.items() if k != "new_token_ids"},
                    "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
                    "prompt_pt_br": E2E_GENERATION_PROMPT,
                    "max_new_tokens": E2E_MAX_NEW_TOKENS,
                    "n_params": n_params,
                    "n_blocks_patched": len(blocks),
                    "n_linears_patched": n_linears,
                    "runtime": "python_reference_ternary_tadds",
                    "original_weight_on_hot_path": False,
                    "lm_head_note": "lm_head/embeddings fora dos blocos permanecem originais",
                },
                "memory": memory,
                "spectra": {
                    "tadds_maximum_rank": int(maximum_rank),
                    "runtime_rows_processed": rows_processed,
                    "runtime_refined_row_rate": rows_refined / max(rows_processed, 1),
                    "runtime_mean_selected_rank": rank_sum / max(rows_processed, 1),
                    "hqr_ans_native": False,
                    "fused_kernel_native": False,
                },
                "artifacts": {
                    "dir": str(artifacts_dir),
                    "total_bytes": artifact_bytes,
                    "method": "binary_os_stat_v1",
                },
            },
            notes=(
                f"E2E §12: baseline={baseline_tok_s:.2f} tok/s candidato={candidate_tok_s:.2f} "
                "tok/s (ambos REAIS via model.generate). "
                f"logits_cos={logits_cos if logits_cos is None else round(logits_cos, 4)} "
                f"exact_match={em['exact_match_rate']:.3f}. Módulos originais restaurados no "
                "finally (transacional)."
            ),
            comparison_role="primary",
        )
        return {
            "pass": e2e_pass,
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
            "logits_cosine": logits_cos,
            "token_exact_match_rate": em["exact_match_rate"],
        }
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, MemoryError) or "out of memory" in message.lower():
            # Paridade RIFT/AETHER: OOM em runtime é limitação de ambiente → SKIPPED.
            recorder.record(
                battery_id=battery_id,
                status="SKIPPED",
                measurement_scope=(
                    "e2e tok/s não executado: memória insuficiente durante a fase E2E; "
                    "modelo restaurado no finally."
                ),
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
                    "SKIPPED: memória insuficiente durante quantização/patch/generate; "
                    "modelo restaurado, fila segue. " + message
                )[:1200],
            )
            print(f"[E2E] SKIPPED (memória insuficiente): {message}")
            return None
        recorder.record(
            battery_id=battery_id,
            status="FAIL",
            measurement_scope="e2e tok/s (erro em runtime); modelo restaurado no finally.",
            quality={"full_local_gate_pass": False},
            metrics={"e2e": {"measured": False}, "error": message[:800]},
            notes=f"Falha na bateria e2e: {exc}"[:800],
        )
        return None
    finally:
        for block, block_name, originals in patched:
            restore_block_linears(block, originals, block_name)
        if patched:
            print("[E2E] modelo restaurado (unpatch de todos os blocos)")


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
        self.json_path = out_dir / "spectra_test_batteries.json"
        self.csv_path = out_dir / "spectra_test_batteries.csv"
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
        # tok/s de topo: SOMENTE a bateria e2e (§12) preenche — demais permanecem null.
        tok_gain = pct_higher(baseline_tok_s, candidate_tok_s)
        ram_gain = pct_lower(baseline_ram_bytes, candidate_ram_bytes)
        disk_gain = pct_lower(baseline_disk_bytes, candidate_disk_bytes)
        measured = [value for value in (tok_gain, ram_gain, disk_gain) if value is not None]
        simulated = str(status).upper() == "SIMULATED" or battery_id.endswith("_PIO_POLICY_SIM")
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "spec": "SPECTRA-LM v0.1 reference",
            "technology": "SPECTRA",
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
                 "User-Agent": "spectra-colab-publisher/1.0"},
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
    print(f"[SPECTRA] Decomposição TADDS low-rank máxima: r={maximum_rank}")
    u, s, v = torch.svd_lowrank(residual, q=maximum_rank, niter=2)
    entropy, ranks, thresholds = entropy_and_ranks(x, maximum_rank)

    with torch.no_grad():
        y_reference = F.linear(x, weight)
        y_base = F.linear(x, ternary)
        y_dynamic = spectra_dynamic_linear(x, ternary, u, s, v, ranks)
        y_full = y_base + lowrank_correction(x, u, s, v, maximum_rank)
        weight_full = ternary + (u * s) @ v.T

    q_weight_base = compute_metrics(weight, ternary)
    q_weight_full = compute_metrics(weight, weight_full)
    q_base = compute_metrics(y_reference, y_base)
    q_dynamic = compute_metrics(y_reference, y_dynamic)
    q_full = compute_metrics(y_reference, y_full)
    row_reference_norm = torch.linalg.vector_norm(y_reference, dim=1).clamp_min(1e-12)
    row_drift = torch.linalg.vector_norm(y_reference - y_dynamic, dim=1) / row_reference_norm
    drift_mean = float(row_drift.mean().item())
    drift_max = float(row_drift.max().item())
    drift_budget = 0.08
    drift_pass = drift_max <= drift_budget
    quality_pass = (
        q_dynamic["cosine"] >= 0.995
        and q_dynamic["nrmse"] <= 0.05
        and drift_pass
    )

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
            lambda: spectra_dynamic_linear(x, ternary, u, s, v, ranks),
            device=device,
            iterations=args.iterations,
        )
    dynamic_ram_measured = dynamic_sampler.summary()
    speedup = baseline_perf["median_ms"] / dynamic_perf["median_ms"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(
        out_dir / "spectra_bundle_m0.spct",
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
        f"[B0] GOLDEN HEADER + SPECTRA-IR + MMAP PASS — "
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
        battery_id="B0_SPECTRA_BINARY_IR_FOUNDATION",
        status="PASS",
        measurement_scope="SPECTRA Header/IR/Stage Pages/MMAP correctness; desempenho não se aplica.",
        quality={"full_local_gate_pass": True},
        metrics={"bundle": bundle, "negative_tests_passed": golden["negative_tests_passed"]},
        notes="Container SPCT v0.1 test-local; ABI de produção ainda não congelada.",
    )
    recorder.record(
        battery_id="P1_SPECTRA_HQR_TERNARY_2BIT",
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
        ), "spectra": {"ternary_sparsity": sparsity, "hqr_ans_native": False}},
        notes="HQR ANS 0.85-bit não implementado; o dashboard usa somente bytes físicos medidos.",
    )
    recorder.record(
        battery_id="P1_SPECTRA_HQR_PLUS_TADDS_DYNAMIC",
        status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=candidate_disk,
        measurement_scope="Single Linear op; bundle SPCT real; Gate/TADDS por entropia; latência do path Python/PyTorch; RAM de nível superior é RSS medido por fase (VmRSS) quando disponível; Tok/s não medido.",
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
            "spectra": {
                "maximum_rank": maximum_rank,
                "mean_selected_rank": mean_rank,
                "stage_activation_rate": active_rate,
                "rank_histogram": {str(rank): int((ranks == rank).sum().item()) for rank in sorted(set(ranks.tolist()))},
                "entropy_mean": float(entropy.mean().item()),
                "entropy_thresholds": thresholds,
                "activation_source": activation_source,
                "single_layer_drift_proxy_mean": drift_mean,
                "single_layer_drift_proxy_max": drift_max,
                "drift_budget": drift_budget,
                "drift_budget_pass": drift_pass,
                "hqr_ans_native": False,
                "pio_async_native": False,
                "fused_kernel_native": False,
                "drift_compensation_native": False,
                "speculative_path_native": False,
                "ring_buffer_budget_bytes": int(args.ring_buffer_mb * 1024 * 1024),
            },
            "bundle": bundle,
        },
        notes="Prefetch, fused kernel, drift compensation e speculative path são simulados. Nenhum ganho nativo de inferência deve ser reivindicado.",
        comparison_role="primary",
    )
    recorder.record(
        battery_id="P1_SPECTRA_DRIFT_CONTRACT_REF",
        status="PASS" if drift_pass else "EXPERIMENTAL_FAIL",
        measurement_scope="Proxy de drift relativo por linha para uma única operação Linear; não é drift acumulado end-to-end.",
        quality={"full_local_gate_pass": drift_pass, "output": q_dynamic},
        metrics={"spectra": {
            "single_layer_drift_proxy_mean": drift_mean,
            "single_layer_drift_proxy_max": drift_max,
            "drift_budget": drift_budget,
            "drift_budget_pass": drift_pass,
            "drift_compensation_native": False,
        }},
        notes="A compensação de drift e a certificação end-to-end permanecem fora da Fase 1.",
    )
    recorder.record(
        battery_id="P1_SPECTRA_PIO_POLICY_SIM",
        status="SIMULATED",
        measurement_scope="Política de prefetch derivada dos ranks do TADDS; nenhuma leitura NVMe assíncrona foi medida.",
        quality={"full_local_gate_pass": None},
        metrics={"spectra": {
            "predicted_refinement_rows": int((ranks > 0).sum().item()),
            "stage_activation_rate": active_rate,
            "ring_buffer_budget_bytes": int(args.ring_buffer_mb * 1024 * 1024),
            "pio_async_native": False,
        }},
        notes="A bateria mede somente decisões da política, não I/O real.",
    )

    # P1_SPECTRA_E2E_TOKS — tok/s de topo REAL baseline E candidato (contrato §12);
    # roda por último: patch transacional de TODAS as Linears dos blocos.
    e2e_summary = run_e2e_tok_s_battery(
        recorder,
        model=model,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        maximum_rank=args.maximum_rank,
    )

    report = {
        "spec": "SPECTRA-LM v0.1 reference",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "target_layer": target_layer,
        "shape": [out_features, in_features],
        "quality": {
            "base": q_base,
            "dynamic": q_dynamic,
            "rank_max": q_full,
            "gate_pass": quality_pass,
            "single_layer_drift_proxy_mean": drift_mean,
            "single_layer_drift_proxy_max": drift_max,
            "drift_budget": drift_budget,
        },
        "spectra": {"active_rate": active_rate, "mean_rank": mean_rank, "thresholds": thresholds},
        "performance": {"baseline": baseline_perf, "base": base_perf, "dynamic": dynamic_perf, "speedup_x": speedup},
        "e2e": e2e_summary,
        "bundle": bundle,
    }
    (out_dir / "spectra_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("SPECTRA-LM PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer}")
    print(f"TADDS ativo             : {active_rate * 100:.1f}% | rank médio={mean_rank:.2f}")
    print(f"Qualidade dinâmica      : cosine={q_dynamic['cosine']:.6f} / NRMSE={q_dynamic['nrmse']:.6f}")
    print(f"Drift proxy (máximo)    : {drift_max:.6f} | orçamento={drift_budget:.3f} | {'PASS' if drift_pass else 'FAIL'}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Baseline Linear         : {baseline_perf['median_ms']:.4f} ms")
    print(f"SPECTRA ref dinâmico     : {dynamic_perf['median_ms']:.4f} ms | {speedup:.3f}x")
    if e2e_summary:
        print(
            f"Tok/s e2e (REAL)        : baseline={e2e_summary['baseline_tok_s']:.2f} "
            f"candidato={e2e_summary['candidate_tok_s']:.2f} ({e2e_summary['speedup_x']:.3f}x) | "
            f"exact_match={e2e_summary['token_exact_match_rate']:.3f}"
        )
    else:
        print("Tok/s e2e               : SKIPPED/FAIL (ver registro P1_SPECTRA_E2E_TOKS)")
    print("HQR ANS / Prefetch / Fused   : NÃO IMPLEMENTADOS — path Python de referência.")
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
    parser = argparse.ArgumentParser(description="SPECTRA-LM v0.1 M0 + Phase 1 reference test")
    parser.add_argument("--mode", choices=["self-test", "phase1"], default="phase1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="org/modelo ou URL do Hugging Face")
    parser.add_argument("--target-layer", default="auto", help="Tensor .weight ou auto")
    parser.add_argument("--prompt", default="Explique por que memória importa na inferência de modelos.")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu — auto usa GPU se houver, senão CPU")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--maximum-rank", type=int, default=16)
    parser.add_argument("--ring-buffer-mb", type=int, default=320)
    parser.add_argument("--out", default="spectra_m0_test_output")
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
                f"SPECTRA SELF-TEST PASS — XXH3-64=0x{golden['checksum']:016x}, "
                f"negativos={golden['negative_tests_passed']}"
            )
            return
        batteries_path = run_phase1(args)
        publish_to_vercel(batteries_path, mode=args.publish, endpoint=args.results_endpoint)
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc
    finally:
        cleanup_colab_workspace(label="SPECTRA")


if __name__ == "__main__":
    main()
