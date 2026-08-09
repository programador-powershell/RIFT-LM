#!/usr/bin/env python3
"""SPECTRA-LM v0.1 — M0/Phase 1 reference battery for Colab.

This validates a ternary HQR-style base, token-adaptive low-rank refinement and
a simulated predictive paging policy. ANS 0.85-bit coding, asynchronous P-IO
and the native SRFA fused kernel are not implemented by this Python path.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import re
import statistics
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


def ensure_import(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError:
        package = pip_name or module
        print(f"[deps] Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return __import__(module)


xxhash = ensure_import("xxhash")
torch = None
F = None
AutoModel = None
AutoModelForCausalLM = None
AutoTokenizer = None


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModel, AutoModelForCausalLM, AutoTokenizer
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn.functional as _F
        from transformers import AutoModel as _AutoModel
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "PyTorch e Transformers são necessários. Instale com: "
            f"pip install torch transformers\nErro: {exc}"
        )
    torch = _torch
    F = _F
    AutoModel = _AutoModel
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer


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


class SpectraFormatError(RuntimeError):
    pass


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


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "spectra_test_batteries.json"
        self.csv_path = out_dir / "spectra_test_batteries.csv"
        self.model_id = model_id
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        battery_id: str,
        status: str,
        measurement_scope: str,
        baseline_ram_bytes: int | None = None,
        candidate_ram_bytes: int | None = None,
        baseline_disk_bytes: int | None = None,
        candidate_disk_bytes: int | None = None,
        quality: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        comparison_role: str | None = None,
    ) -> dict[str, Any]:
        ram_gain = pct_lower(baseline_ram_bytes, candidate_ram_bytes)
        disk_gain = pct_lower(baseline_disk_bytes, candidate_disk_bytes)
        measured = [value for value in (ram_gain, disk_gain) if value is not None]
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "spec": "SPECTRA-LM v0.1 reference",
            "technology": "SPECTRA",
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            "comparison_role": comparison_role,
            "baseline_tok_s": None,
            "candidate_tok_s": None,
            "baseline_ram_bytes": baseline_ram_bytes,
            "candidate_ram_bytes": candidate_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes,
            "candidate_disk_bytes": candidate_disk_bytes,
            "gains": {
                "tok_s_gain_pct": None,
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
        return record


class ResultsPublishError(RuntimeError):
    pass


def running_in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


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


def publish_to_vercel(path: Path, *, mode: str, endpoint: str | None) -> None:
    if mode == "off":
        print("[PUBLISH] Publicação remota desativada (--publish off).")
        return
    endpoint = (endpoint or read_setting("RIFT_RESULTS_ENDPOINT") or "").strip()
    token = read_setting("RIFT_INGEST_TOKEN")
    if not endpoint or not token:
        missing = []
        if not endpoint:
            missing.append("RIFT_RESULTS_ENDPOINT")
        if not token:
            missing.append("RIFT_INGEST_TOKEN")
        message = "Configure " + " e ".join(missing)
        if mode == "required" or running_in_colab():
            raise ResultsPublishError(message)
        print(f"[PUBLISH] AVISO: {message}; resultado mantido localmente.")
        return
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ResultsPublishError("RIFT_RESULTS_ENDPOINT precisa ser uma URL HTTPS")
    records = json.loads(path.read_text(encoding="utf-8"))
    request = Request(
        endpoint,
        data=json.dumps({"records": records}, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "spectra-colab-publisher/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ResultsPublishError(f"Vercel retornou HTTP {exc.code}") from exc
    except URLError as exc:
        raise ResultsPublishError(f"Falha ao conectar com a Vercel: {exc.reason}") from exc
    if not result.get("ok") or not isinstance(result.get("publication"), dict):
        raise ResultsPublishError(str(result.get("error") or "Resposta inválida da Vercel"))
    publication = result["publication"]
    print(
        f"[PUBLISH] {publication['records']} registro(s) aceito(s) pela Vercel e "
        f"publicado(s) em {publication['repository']}:{publication['branch']}/{publication['path']}"
    )
    if publication.get("commit_url"):
        print(f"[PUBLISH] Commit: {publication['commit_url']}")


def load_model(model_id: str, *, device: Any, trust_remote_code: bool):
    hf_token = read_setting("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    common = {"token": hf_token, "dtype": dtype, "trust_remote_code": trust_remote_code}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
    except (ValueError, KeyError) as exc:
        if not trust_remote_code:
            raise
        print(f"[WARN] AutoModelForCausalLM indisponível ({exc}); tentando AutoModel.")
        model = AutoModel.from_pretrained(model_id, **common)
    return tokenizer, model.to(device).eval()


def run_phase1(args: argparse.Namespace) -> Path:
    ensure_ml_dependencies()
    device = torch.device(args.device)
    model_id = normalize_huggingface_model_id(args.model)
    print(f"[Phase1] Carregando {model_id} em {device}...")
    tokenizer, model = load_model(model_id, device=device, trust_remote_code=args.trust_remote_code)
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

    baseline_perf = benchmark_ms(
        lambda: F.linear(x, weight), device=device, iterations=args.iterations,
    )
    base_perf = benchmark_ms(
        lambda: F.linear(x, ternary), device=device, iterations=args.iterations,
    )
    dynamic_perf = benchmark_ms(
        lambda: spectra_dynamic_linear(x, ternary, u, s, v, ranks),
        device=device,
        iterations=args.iterations,
    )
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
    baseline_ram = baseline_disk + input_bytes + output_bytes
    base_ram = int(ternary.numel() * 4 + input_bytes + output_bytes)
    factor_ram = int((u.numel() + s.numel() + v.numel()) * 4)
    candidate_ram = base_ram + factor_ram
    active_rate = float((ranks > 0).float().mean().item())
    mean_rank = float(ranks.float().mean().item())

    recorder = BatteryRecorder(out_dir, model_id=model_id)
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
        measurement_scope="Single Linear op; base ternária realmente empacotada em 2 bits; RAM é o working set FP32 do path Python.",
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {
            "metric": "linear_latency",
            "baseline_median_ms": baseline_perf["median_ms"],
            "candidate_median_ms": base_perf["median_ms"],
            "speedup_x": baseline_perf["median_ms"] / base_perf["median_ms"],
        }, "spectra": {"ternary_sparsity": sparsity, "hqr_ans_native": False}},
        notes="HQR ANS 0.85-bit não implementado; o dashboard usa somente bytes físicos medidos.",
    )
    recorder.record(
        battery_id="P1_SPECTRA_HQR_PLUS_TADDS_DYNAMIC",
        status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=candidate_disk,
        measurement_scope="Single Linear op; bundle SPCT real; Gate/TADDS por entropia; latência do path Python/PyTorch; Tok/s não medido.",
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
    parser.add_argument("--device", default="cuda", help="cpu ou cuda")
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


if __name__ == "__main__":
    main()
