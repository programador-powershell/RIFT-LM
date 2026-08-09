#!/usr/bin/env python3
"""AETHER-LM v1.0 — M0/Phase 1 reference battery for Colab.

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
AutoModel = None
AutoModelForMultimodalLM = None


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForMultimodalLM
    if torch is not None:
        return
    ensure_import("sentencepiece")
    ensure_import("tiktoken")
    print("[deps] Garantindo transformers e accelerate atualizados (Gemma 4 / multimodal)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "huggingface_hub"]
    )
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


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str, publish_mode: str = "off", results_endpoint: str | None = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "aether_test_batteries.json"
        self.csv_path = out_dir / "aether_test_batteries.csv"
        self.model_id = model_id
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
            "spec": "AETHER-LM v1.0 reference",
            "technology": "AETHER",
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


def cleanup_colab_workspace(*, label: str = "battery") -> None:
    """Libera disco no Colab após publicar resultados, para o próximo modelo."""
    import gc
    import shutil

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    removed = []
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

    # Artefatos temporários conhecidos no /tmp e /content
    patterns = [
        "/tmp/winner_cpp_*",
        "/tmp/winner_phase1_*",
        "/tmp/phase1_load_fail*",
        "/tmp/cascade_load_fail*",
        "/tmp/rift_*",
        "/content/*_launcher.py",
        "/content/rift_serial_queue",
    ]
    import glob as _glob
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
        print(f"[cleanup] {label}: nada relevante para limpar")
    gc.collect()



def load_tokenizer(model_id: str, *, trust_remote_code: bool = False, token: str | None = None):
    """Carrega tokenizer com fallbacks (subfolder, use_fast=False).

    Modelos multimodais/diffusers (ex.: MiniMax-H3) guardam o tokenizer em
    subpastas; AutoTokenizer na raiz falha com mensagem enganosa de sentencepiece/tiktoken.
    """
    common = {"trust_remote_code": trust_remote_code, "token": token}
    attempts = [
        {},
        {"use_fast": False},
        {"subfolder": "tokenizer"},
        {"subfolder": "tokenizer", "use_fast": False},
        {"subfolder": "processor"},
        {"subfolder": "processor", "use_fast": False},
        {"subfolder": "text_encoder"},
        {"subfolder": "text_encoder", "use_fast": False},
    ]
    errors: list[str] = []
    for extra in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, **common, **extra)
            print(f"[tokenizer] OK com {extra or {'root': True}}")
            return tok
        except Exception as exc:
            errors.append(f"{extra or 'root'}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Não foi possível carregar o tokenizer de "
        f"{model_id}.\n"
        "Possíveis causas: (1) arquivos só em subpasta; (2) modelo diffusers/vídeo "
        "sem checkpoint Transformers CausalLM; (3) tokenizers desatualizado.\n"
        "Tentativas:\n- " + "\n- ".join(errors)
    )


def load_model(model_id: str, *, device: Any, trust_remote_code: bool):
    hf_token = read_setting("HF_TOKEN")
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
        )
        recorder.record(
            battery_id="P1_LOAD_MODEL",
            status="FAIL",
            measurement_scope="model_load",
            quality={"full_local_gate_pass": False},
            metrics={"error": str(load_exc)[:800]},
            notes=(
                f"Falha ao carregar {model_id}. Modelos diffusers/vídeo "
                "(ex. MiniMax-H3) não são suportados por esta bateria CausalLM. "
                f"Detalhe: {load_exc}"
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

    baseline_perf = benchmark_ms(
        lambda: F.linear(x, weight), device=device, iterations=args.iterations,
    )
    base_perf = benchmark_ms(
        lambda: F.linear(x, ternary), device=device, iterations=args.iterations,
    )
    dynamic_perf = benchmark_ms(
        lambda: aether_dynamic_linear(x, ternary, u, s, v, ranks),
        device=device,
        iterations=args.iterations,
    )
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
    baseline_ram = baseline_disk + input_bytes + output_bytes
    base_ram = int(ternary.numel() * 4 + input_bytes + output_bytes)
    factor_ram = int((u.numel() + s.numel() + v.numel()) * 4)
    candidate_ram = base_ram + factor_ram
    active_rate = float((ranks > 0).float().mean().item())
    mean_rank = float(ranks.float().mean().item())

    recorder = BatteryRecorder(out_dir, model_id=model_id, publish_mode=args.publish, results_endpoint=args.results_endpoint)
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
        measurement_scope="Single Linear op; base ternária realmente empacotada em 2 bits; RAM é o working set FP32 do path Python.",
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {
            "metric": "linear_latency",
            "baseline_median_ms": baseline_perf["median_ms"],
            "candidate_median_ms": base_perf["median_ms"],
            "speedup_x": baseline_perf["median_ms"] / base_perf["median_ms"],
        }, "aether": {"ternary_sparsity": sparsity, "hqr_ans_native": False}},
        notes="HQR ANS 0.85-bit não implementado; o dashboard usa somente bytes físicos medidos.",
    )
    recorder.record(
        battery_id="P1_AETHER_HQR_PLUS_TADDS_DYNAMIC",
        status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=candidate_disk,
        measurement_scope="Single Linear op; bundle AETH real; TADDS por entropia; latência do path Python/PyTorch; Tok/s não medido.",
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

    report = {
        "spec": "AETHER-LM v1.0 reference",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "target_layer": target_layer,
        "shape": [out_features, in_features],
        "quality": {"base": q_base, "dynamic": q_dynamic, "rank_max": q_full, "gate_pass": quality_pass},
        "aether": {"active_rate": active_rate, "mean_rank": mean_rank, "thresholds": thresholds},
        "performance": {"baseline": baseline_perf, "base": base_perf, "dynamic": dynamic_perf, "speedup_x": speedup},
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
    parser.add_argument("--device", default="cuda", help="cpu ou cuda")
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
