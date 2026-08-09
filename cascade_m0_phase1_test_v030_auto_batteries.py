#!/usr/bin/env python3
"""CASCADE v0.3 — M0/Phase 1 reference battery for Colab.

This is a correctness and measurement path. Predictive prefetch is simulated
and no native fused kernel or model-level Tok/s gain is claimed.
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




CASCADE_MAGIC = b"CSCD"
CASCADE_VERSION_M0 = 0x0003
CASCADE_HEADER_SIZE = 128
CASCADE_HEADER_FORMAT = "<4sHHIIQQQQQQQ56s"
CASCADE_CHECKSUM_OFFSET = 64
EXPECTED_GOLDEN_CHECKSUM = 0xF5050B62BBDAC01B
STAGE_ENTRY_FORMAT = "<IIIIQQQ"
STAGE_ENTRY_SIZE = struct.calcsize(STAGE_ENTRY_FORMAT)
assert struct.calcsize(CASCADE_HEADER_FORMAT) == CASCADE_HEADER_SIZE


def align_up(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def xxh3_64(data: bytes) -> int:
    return int(xxhash.xxh3_64_intdigest(data, seed=0))


def create_cascade_header(
    *,
    ir_offset: int,
    stage_table_offset: int,
    stage_count: int,
    gate_table_offset: int,
    payload_offset: int,
    file_size: int,
    magic: bytes = CASCADE_MAGIC,
    version: int = CASCADE_VERSION_M0,
) -> bytes:
    raw = struct.pack(
        CASCADE_HEADER_FORMAT,
        magic,
        version,
        0,
        0x01,
        0,
        ir_offset,
        stage_table_offset,
        stage_count,
        gate_table_offset,
        payload_offset,
        file_size,
        0,
        bytes(56),
    )
    checksum = xxh3_64(raw)
    return raw[:CASCADE_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def parse_cascade_header(data: bytes, actual_file_size: int) -> dict[str, int]:
    if len(data) < CASCADE_HEADER_SIZE:
        raise CascadeFormatError("truncated_header")
    values = struct.unpack(CASCADE_HEADER_FORMAT, data[:CASCADE_HEADER_SIZE])
    (
        magic,
        version,
        _profile,
        _flags,
        _reserved,
        ir_offset,
        stage_table_offset,
        stage_count,
        gate_table_offset,
        payload_offset,
        file_size,
        checksum,
        _reserved_tail,
    ) = values
    if magic != CASCADE_MAGIC:
        raise CascadeFormatError("bad_magic")
    if version != CASCADE_VERSION_M0:
        raise CascadeFormatError("bad_version")
    zeroed = bytearray(data[:CASCADE_HEADER_SIZE])
    zeroed[CASCADE_CHECKSUM_OFFSET:72] = bytes(8)
    if xxh3_64(bytes(zeroed)) != checksum:
        raise CascadeFormatError("checksum_mismatch")
    if file_size != actual_file_size:
        raise CascadeFormatError("file_size_mismatch")
    offsets = (ir_offset, stage_table_offset, gate_table_offset, payload_offset)
    if any(offset < CASCADE_HEADER_SIZE or offset > file_size for offset in offsets):
        raise CascadeFormatError("offset_out_of_range")
    if offsets != tuple(sorted(offsets)):
        raise CascadeFormatError("offset_order")
    if stage_count > 1_000_000:
        raise CascadeFormatError("stage_count_overflow")
    if stage_table_offset + stage_count * STAGE_ENTRY_SIZE > gate_table_offset:
        raise CascadeFormatError("stage_table_overlap")
    return {
        "ir_offset": ir_offset,
        "stage_table_offset": stage_table_offset,
        "stage_count": stage_count,
        "gate_table_offset": gate_table_offset,
        "payload_offset": payload_offset,
        "file_size": file_size,
        "checksum": checksum,
    }


def _rewrite_header(header: bytes, **changes: int | bytes) -> bytes:
    fields = list(struct.unpack(CASCADE_HEADER_FORMAT, header))
    names = [
        "magic", "version", "profile", "flags", "reserved", "ir_offset",
        "stage_table_offset", "stage_count", "gate_table_offset",
        "payload_offset", "file_size", "checksum", "reserved_tail",
    ]
    current = dict(zip(names, fields))
    current.update(changes)
    current["checksum"] = 0
    raw = struct.pack(CASCADE_HEADER_FORMAT, *(current[name] for name in names))
    checksum = xxh3_64(raw)
    return raw[:CASCADE_CHECKSUM_OFFSET] + struct.pack("<Q", checksum) + raw[72:]


def run_golden_header_tests() -> dict[str, Any]:
    golden = create_cascade_header(
        ir_offset=128,
        stage_table_offset=128,
        stage_count=0,
        gate_table_offset=128,
        payload_offset=128,
        file_size=128,
    )
    parsed = parse_cascade_header(golden, 128)
    checksum = parsed["checksum"]
    if checksum != EXPECTED_GOLDEN_CHECKSUM:
        raise AssertionError(
            f"Golden checksum mudou: 0x{checksum:016x} != "
            f"0x{EXPECTED_GOLDEN_CHECKSUM:016x}"
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
            parse_cascade_header(mutated, actual_size)
        except CascadeFormatError:
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
        if requested not in state or state[requested].ndim != 2:
            raise KeyError(f"Tensor Linear 2D não encontrado: {requested}")
        return requested
    suffixes = (
        "self_attn.q_proj", "self_attn.qkv_proj", "self_attn.query_key_value",
        "attention.q_proj", "attention.wq", "attn.q_proj",
        "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
        "self_attn.o_proj",
    )
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and getattr(module, "weight", None) is not None
        and module.weight.ndim == 2
    ]
    for suffix in suffixes:
        match = next((name for name in names if name.lower().endswith(suffix)), None)
        if match:
            return f"{match}.weight"
    if names:
        return f"{names[0]}.weight"
    raise KeyError("Nenhuma camada torch.nn.Linear 2D encontrada")


def capture_activation(model: Any, tokenizer: Any, module_name: str, device: Any, prompt: str):
    module = model.get_submodule(module_name)
    captured: dict[str, Any] = {}

    def hook(_module, args):
        captured["x"] = args[0].detach()

    handle = module.register_forward_pre_hook(hook)
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()
    if "x" not in captured:
        raise RuntimeError("Hook não capturou a ativação")
    return captured["x"]


def compute_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    a = reference.detach().float().reshape(-1)
    b = candidate.detach().float().reshape(-1)
    cosine = float(F.cosine_similarity(a, b, dim=0).item())
    rmse = torch.sqrt(torch.mean((a - b) ** 2))
    dynamic_range = torch.max(a) - torch.min(a)
    nrmse = float((rmse / (dynamic_range + 1e-12)).item())
    return {"cosine": cosine, "nrmse": nrmse}


def validate_cascade_ir(ir: dict[str, Any]) -> None:
    if ir.get("ir_version") != 3 or not str(ir.get("model_id", "")).strip():
        raise CascadeFormatError("invalid_ir_header")
    operations = ir.get("operations")
    if not isinstance(operations, list) or not operations:
        raise CascadeFormatError("invalid_ir_operations")
    allowed = {
        "CASCADE_OP_EMBEDDING", "CASCADE_OP_LINEAR", "CASCADE_OP_RMSNORM",
        "CASCADE_OP_ROPE", "CASCADE_OP_ATTENTION", "CASCADE_OP_ACTIVATION",
        "CASCADE_OP_ADD", "CASCADE_OP_OUTPUT", "CASCADE_OP_CUSTOM",
    }
    ids = [operation.get("id") for operation in operations]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise CascadeFormatError("ir_not_canonical_topological_order")
    if any(operation.get("opcode") not in allowed for operation in operations):
        raise CascadeFormatError("ir_unknown_opcode")
    if any(not isinstance(operation.get("cascade_ref"), int) for operation in operations):
        raise CascadeFormatError("ir_missing_cascade_ref")


def sync_device(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_ms(fn, *, device: Any, iterations: int) -> dict[str, float]:
    for _ in range(3):
        fn()
    sync_device(device)
    samples = []
    for _ in range(max(3, iterations)):
        start = time.perf_counter()
        fn()
        sync_device(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def lowrank_linear(x: Any, U: Any, S: Any, V: Any) -> Any:
    return ((x @ V) * S) @ U.transpose(0, 1)


def stage_payload(U: Any, S: Any, V: Any) -> bytes:
    out_features, rank = U.shape
    in_features = V.shape[0]
    return (
        struct.pack("<III", out_features, in_features, rank)
        + U.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        + S.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        + V.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
    )


def write_bundle(
    path: Path,
    *,
    model_id: str,
    target_layer: str,
    architecture: str,
    stages: list[tuple[Any, Any, Any]],
    gate_percentile: float,
) -> dict[str, Any]:
    ir = {
        "ir_version": 3,
        "model_id": model_id,
        "architecture_hint": architecture,
        "operations": [{
            "id": 0,
            "opcode": "CASCADE_OP_LINEAR",
            "inputs": [0],
            "outputs": [1],
            "weights": [target_layer],
            "cascade_ref": 0,
        }],
    }
    validate_cascade_ir(ir)
    ir_bytes = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    gate_bytes = json.dumps({
        "type": "ACTIVATION_L2_PERCENTILE_V1",
        "percentile": gate_percentile,
        "prefetch": "SIMULATED_LAG_ONE",
    }, separators=(",", ":")).encode("utf-8")
    payloads = [stage_payload(*stage) for stage in stages]

    ir_offset = CASCADE_HEADER_SIZE
    stage_table_offset = align_up(ir_offset + 4 + len(ir_bytes))
    gate_table_offset = stage_table_offset + len(payloads) * STAGE_ENTRY_SIZE
    payload_offset = align_up(gate_table_offset + 4 + len(gate_bytes))
    offsets = []
    cursor = payload_offset
    for payload in payloads:
        offsets.append(cursor)
        cursor = align_up(cursor + len(payload))
    file_size = cursor

    header = create_cascade_header(
        ir_offset=ir_offset,
        stage_table_offset=stage_table_offset,
        stage_count=len(payloads),
        gate_table_offset=gate_table_offset,
        payload_offset=payload_offset,
        file_size=file_size,
    )
    blob = bytearray(file_size)
    blob[:CASCADE_HEADER_SIZE] = header
    blob[ir_offset:ir_offset + 4] = struct.pack("<I", len(ir_bytes))
    blob[ir_offset + 4:ir_offset + 4 + len(ir_bytes)] = ir_bytes
    for index, payload in enumerate(payloads):
        U, _S, _V = stages[index]
        entry = struct.pack(
            STAGE_ENTRY_FORMAT,
            index,
            0,
            index,
            int(U.shape[1]),
            offsets[index],
            len(payload),
            xxh3_64(payload),
        )
        start = stage_table_offset + index * STAGE_ENTRY_SIZE
        blob[start:start + STAGE_ENTRY_SIZE] = entry
        blob[offsets[index]:offsets[index] + len(payload)] = payload
    blob[gate_table_offset:gate_table_offset + 4] = struct.pack("<I", len(gate_bytes))
    blob[gate_table_offset + 4:gate_table_offset + 4 + len(gate_bytes)] = gate_bytes
    path.write_bytes(blob)

    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        parsed = parse_cascade_header(mapped[:CASCADE_HEADER_SIZE], len(mapped))
        for index, payload in enumerate(payloads):
            start = offsets[index]
            if xxh3_64(mapped[start:start + len(payload)]) != xxh3_64(payload):
                raise CascadeFormatError(f"stage_checksum_{index}")
    return {
        "file_size": file_size,
        "header_checksum": parsed["checksum"],
        "ir_offset": ir_offset,
        "stage_table_offset": stage_table_offset,
        "gate_table_offset": gate_table_offset,
        "payload_offset": payload_offset,
        "stages": [
            {"stage_id": i, "offset": offsets[i], "size": len(payload)}
            for i, payload in enumerate(payloads)
        ],
    }


def pct_higher(base: float | None, candidate: float | None) -> float | None:
    return None if not base or candidate is None else (candidate / base - 1.0) * 100.0


def pct_lower(base: float | None, candidate: float | None) -> float | None:
    return None if not base or candidate is None else (1.0 - candidate / base) * 100.0


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str, publish_mode: str = "off", results_endpoint: str | None = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "cascade_test_batteries.json"
        self.csv_path = out_dir / "cascade_test_batteries.csv"
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
        baseline_ram_bytes: int | None = None,
        candidate_ram_bytes: int | None = None,
        baseline_disk_bytes: int | None = None,
        candidate_disk_bytes: int | None = None,
        measurement_scope: str,
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
            "spec": "CASCADE v0.3",
            "technology": "CASCADE",
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
                 "User-Agent": "cascade-colab-publisher/0.3"},
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
        "/content/rift_serial_queue",
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


def run_phase1(args: argparse.Namespace) -> Path:
    ensure_ml_dependencies()
    device = resolve_torch_device(args.device)
    model_id = normalize_huggingface_model_id(args.model)
    print(f"[Phase1] Carregando {model_id} em {device}...")
    hf_token = read_setting("HF_TOKEN")
    try:
        tokenizer = load_tokenizer(model_id, trust_remote_code=args.trust_remote_code, token=hf_token)
        load_dtype = torch.float16 if device.type == "cuda" else torch.float32
        load_kwargs = {
            "token": hf_token,
            "dtype": load_dtype,
            "trust_remote_code": args.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        classes = []
        if AutoModelForMultimodalLM is not None:
            classes.append(AutoModelForMultimodalLM)
        classes.extend([AutoModelForCausalLM, AutoModel])
        last_err = None
        model = None
        for cls in classes:
            try:
                model = cls.from_pretrained(model_id, **load_kwargs)
                if getattr(model, "hf_device_map", None) is None:
                    model = model.to(device)
                model.eval()
                print(f"[load] Modelo carregado via {cls.__name__}")
                break
            except Exception as exc:
                last_err = exc
                print(f"[load] {cls.__name__} falhou: {exc}")
        if model is None:
            raise RuntimeError(f"Falha ao carregar modelo: {last_err}")
    except Exception as load_exc:
        print(f"[Phase1] FALHA ao carregar modelo/tokenizer: {load_exc}")
        out_dir = Path(getattr(args, "out", None) or "/tmp/cascade_load_fail")
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
    W = model.state_dict()[target_layer].detach().to(device=device, dtype=torch.float32)
    out_features, in_features = W.shape

    try:
        x = capture_activation(
            model,
            tokenizer,
            target_layer.removesuffix(".weight"),
            device,
            args.prompt,
        ).reshape(-1, in_features).to(dtype=torch.float32)
        activation_source = "real_model_activation"
    except Exception as exc:
        print(f"[WARN] Ativação real indisponível: {exc}; usando fallback determinístico.")
        torch.manual_seed(1234)
        x = torch.randn(16, in_features, device=device, dtype=torch.float32)
        activation_source = "synthetic_fallback"

    max_rank = min(out_features, in_features)
    base_rank = min(max(1, args.base_rank), max_rank)
    refine_rank = min(max(1, args.refinement_rank), max_rank - base_rank)
    total_rank = base_rank + refine_rank
    if refine_rank <= 0:
        raise ValueError("A matriz não comporta um estágio de refinamento adicional")
    print(f"[CASCADE] Decomposição low-rank aleatorizada: F0={base_rank}, F1={refine_rank}")
    U, S, V = torch.svd_lowrank(W, q=total_rank, niter=2)
    U0, S0, V0 = U[:, :base_rank], S[:base_rank], V[:, :base_rank]
    U1, S1, V1 = U[:, base_rank:total_rank], S[base_rank:total_rank], V[:, base_rank:total_rank]

    with torch.no_grad():
        y_ref = F.linear(x, W)
        y_f0 = lowrank_linear(x, U0, S0, V0)
        y_f1 = lowrank_linear(x, U1, S1, V1)
        gate_features = torch.linalg.vector_norm(x, dim=1) / max(in_features ** 0.5, 1.0)
        threshold = torch.quantile(gate_features, args.gate_percentile / 100.0)
        gate_mask = gate_features >= threshold
        y_gated = y_f0 + gate_mask[:, None].to(y_f1.dtype) * y_f1
        y_full = y_f0 + y_f1

    stage_rate = float(gate_mask.float().mean().item())
    relative_drift = torch.linalg.vector_norm(y_ref - y_gated, dim=1) / (
        torch.linalg.vector_norm(y_ref, dim=1) + 1e-12
    )
    drift_mean = float(relative_drift.mean().item())
    drift_max = float(relative_drift.max().item())
    q_f0 = compute_metrics(y_ref, y_f0)
    q_gated = compute_metrics(y_ref, y_gated)
    q_full = compute_metrics(y_ref, y_full)

    predicted = torch.zeros_like(gate_mask)
    if gate_mask.numel() > 1:
        predicted[1:] = gate_mask[:-1]
    tp = int((predicted & gate_mask).sum().item())
    predicted_count = int(predicted.sum().item())
    required_count = int(gate_mask.sum().item())
    prefetch_precision = tp / predicted_count if predicted_count else None
    prefetch_recall = tp / required_count if required_count else None

    def cascade_reference():
        base = lowrank_linear(x, U0, S0, V0)
        features = torch.linalg.vector_norm(x, dim=1) / max(in_features ** 0.5, 1.0)
        mask = features >= threshold
        residual = lowrank_linear(x, U1, S1, V1)
        return base + mask[:, None].to(residual.dtype) * residual

    baseline_perf = benchmark_ms(lambda: F.linear(x, W), device=device, iterations=args.iterations)
    f0_perf = benchmark_ms(lambda: lowrank_linear(x, U0, S0, V0), device=device, iterations=args.iterations)
    gated_perf = benchmark_ms(cascade_reference, device=device, iterations=args.iterations)
    speedup = baseline_perf["median_ms"] / gated_perf["median_ms"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(
        out_dir / "cascade_bundle_m0.cscd",
        model_id=model_id,
        target_layer=target_layer,
        architecture=str(getattr(model.config, "model_type", "unknown")),
        stages=[(U0, S0, V0), (U1, S1, V1)],
        gate_percentile=args.gate_percentile,
    )
    golden = run_golden_header_tests()
    print(
        f"[B0] GOLDEN HEADER + CASCADE-IR + MMAP PASS — "
        f"XXH3-64=0x{golden['checksum']:016x}, negativos={golden['negative_tests_passed']}"
    )

    baseline_disk = int(W.numel() * 4)
    candidate_disk = int(bundle["file_size"])
    stage0_bytes = int(bundle["stages"][0]["size"])
    stage1_bytes = int(bundle["stages"][1]["size"])
    input_bytes = int(x.numel() * 4)
    output_bytes = int(y_ref.numel() * 4)
    baseline_ram = baseline_disk + input_bytes + output_bytes
    factor_bytes = sum(stage["size"] for stage in bundle["stages"])
    candidate_ram = factor_bytes + input_bytes + output_bytes
    quality_pass = q_gated["cosine"] >= 0.995 and q_gated["nrmse"] <= 0.05 and drift_mean <= 0.05

    recorder = BatteryRecorder(out_dir, model_id=model_id, publish_mode=args.publish, results_endpoint=args.results_endpoint)
    recorder.record(
        battery_id="B0_CASCADE_BINARY_IR_FOUNDATION",
        status="PASS",
        measurement_scope="CASCADE Header/IR/Stage Pages/MMAP correctness; Tok/s, RAM e compressão não se aplicam.",
        quality={"full_local_gate_pass": True},
        metrics={"bundle": bundle, "negative_tests_passed": golden["negative_tests_passed"]},
        notes="CSCD v0.3 test-local container; ABI de produção ainda não congelada.",
    )
    recorder.record(
        battery_id="P1_CASCADE_BASE_STAGE_F0",
        status="EXPERIMENTAL_PASS" if q_f0["cosine"] >= 0.95 else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=int(bundle["stages"][0]["size"] + input_bytes + output_bytes),
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=int(bundle["stages"][0]["size"]),
        measurement_scope="Single Linear op; disk=FP32 raw vs F0 FP16 low-rank payload; RAM=working-set estimate; model Tok/s not measured.",
        quality={"full_local_gate_pass": None, "output": q_f0},
        metrics={"operation": {"baseline_median_ms": baseline_perf["median_ms"], "candidate_median_ms": f0_perf["median_ms"], "speedup_x": baseline_perf["median_ms"] / f0_perf["median_ms"]}},
        notes="F0 sempre residente; resultado aproximado isolado.",
    )
    recorder.record(
        battery_id="P1_CASCADE_GATED_F0_PLUS_F1",
        status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram,
        candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk,
        candidate_disk_bytes=candidate_disk,
        measurement_scope="Single Linear op; disk=arquivo CSCD real; RAM=working-set estimate; latency=Python/PyTorch reference path; model Tok/s not measured.",
        quality={
            "full_local_gate_pass": quality_pass,
            "output": q_gated,
            "output_f0": q_f0,
            "output_f0_plus_f1_always": q_full,
            "cumulative_drift_mean": drift_mean,
            "cumulative_drift_max": drift_max,
        },
        metrics={
            "operation": {
                "metric": "linear_latency",
                "baseline_median_ms": baseline_perf["median_ms"],
                "candidate_median_ms": gated_perf["median_ms"],
                "speedup_x": speedup,
                "rows_processed": int(x.shape[0]),
            },
            "cascade": {
                "base_rank": base_rank,
                "refinement_rank": refine_rank,
                "stage_activation_rate": stage_rate,
                "gate": "ACTIVATION_L2_PERCENTILE_V1",
                "gate_percentile": args.gate_percentile,
                "activation_source": activation_source,
                "fused_semantics_reference": True,
                "fused_kernel_native": False,
                "logical_bytes_addressed_per_vector": {
                    "baseline_dense_weight": baseline_disk,
                    "cascade_expected_stage_payload": float(
                        stage0_bytes + stage_rate * stage1_bytes
                    ),
                    "IMPORTANT": "logical payload, not measured memory-bus traffic",
                },
            },
            "bundle": bundle,
        },
        notes="Gate heurístico por ativação. Caminho de referência não é kernel fused; nenhum speedup nativo é reivindicado.",
        comparison_role="primary",
    )
    recorder.record(
        battery_id="P1_CASCADE_PREDICTIVE_PREFETCH_SIM",
        status="SIMULATED",
        measurement_scope="Lag-one simulation over gate decisions; no real asynchronous I/O latency measured.",
        quality={"full_local_gate_pass": None},
        metrics={"prefetch": {
            "precision": prefetch_precision,
            "recall": prefetch_recall,
            "required_pages": required_count,
            "predicted_pages": predicted_count,
            "true_positive_pages": tp,
            "prefetched_bytes": predicted_count * stage1_bytes,
            "wasted_prefetch_bytes": max(0, predicted_count - tp) * stage1_bytes,
            "missed_prefetch_bytes": max(0, required_count - tp) * stage1_bytes,
            "IMPORTANT": "simulated page requests, not measured asynchronous I/O",
        }},
        notes="Erro de prefetch afeta apenas custo estimado; esta bateria não implementa I/O assíncrono.",
    )

    report = {
        "spec": "CASCADE v0.3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "target_layer": target_layer,
        "shape": [out_features, in_features],
        "quality": {"f0": q_f0, "gated": q_gated, "full": q_full, "gate_pass": quality_pass},
        "cascade": {"stage_activation_rate": stage_rate, "drift_mean": drift_mean, "drift_max": drift_max},
        "performance": {"baseline": baseline_perf, "f0": f0_perf, "gated": gated_perf, "speedup_x": speedup},
        "bundle": bundle,
    }
    (out_dir / "cascade_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("CASCADE PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer}")
    print(f"Gate F1                 : {stage_rate * 100:.1f}% dos vetores")
    print(f"Qualidade gated         : cosine={q_gated['cosine']:.6f} / NRMSE={q_gated['nrmse']:.6f}")
    print(f"Cumulative drift        : mean={drift_mean:.6f} / max={drift_max:.6f}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Baseline Linear         : {baseline_perf['median_ms']:.4f} ms")
    print(f"CASCADE ref gated       : {gated_perf['median_ms']:.4f} ms | {speedup:.3f}x")
    print("Native fused kernel     : NÃO IMPLEMENTADO — nenhum ganho nativo deve ser reivindicado.")
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
    parser = argparse.ArgumentParser(description="CASCADE v0.3 M0 + Phase 1 reference test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="org/modelo ou URL do Hugging Face")
    parser.add_argument("--target-layer", default="auto", help="Tensor .weight ou auto")
    parser.add_argument("--prompt", default="Explique por que memória importa na inferência de modelos.")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--base-rank", type=int, default=32)
    parser.add_argument("--refinement-rank", type=int, default=32)
    parser.add_argument("--gate-percentile", type=float, default=50.0)
    parser.add_argument("--out", default="cascade_m0_test_output")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--publish", choices=["auto", "required", "off"], default=os.environ.get("RIFT_PUBLISH_MODE", "auto"))
    parser.add_argument("--results-endpoint", default=None, help="URL HTTPS /api/results da Vercel")
    values = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(without_ipykernel_connection_args(values))
    if not 0 <= args.gate_percentile <= 100:
        parser.error("--gate-percentile precisa estar entre 0 e 100")
    try:
        batteries_path = run_phase1(args)
        publish_to_vercel(batteries_path, mode=args.publish, endpoint=args.results_endpoint)
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc
    finally:
        cleanup_colab_workspace(label="CASCADE")


if __name__ == "__main__":
    main()
