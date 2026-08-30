#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLB/AMT streaming re-quantization + runtime proof battery v1.

The upstream checkpoint is never materialized as a complete local weight file.
Safetensors headers are read first, then each tensor is fetched by HTTP Range,
re-quantized, verified and immediately written into VLB-DIR.

Proof gates are independent:
  A. CONVERSION_VERIFIED  — numerical reconstruction gate.
  B. VLB_RUNTIME_VERIFIED — generation forward executes from VLB-DIR, not HF weights.
  C. AMT_VERIFIED         — teacher-free residual/gate improves or preserves fresh validation.

Only A+B+C => VLB_AMT_ENGINE_PROOF_PASS.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_deps() -> None:
    # Gemma 4 config declares the Transformers 5.5 generation.
    required = {
        "torch": "torch",
        "requests": "requests>=2.32",
        "huggingface_hub": "huggingface_hub>=0.27,<2",
        "safetensors": "safetensors>=0.5,<1",
        "transformers": "transformers>=5.5.0",
        "accelerate": "accelerate>=1.2,<2",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        print("[VLB] installing:", " ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])


ensure_deps()

import requests
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata

from vlb_runtime import run_vlb_runtime_and_amt_proof


BATTERY_ID = "VLB_AMT_STREAMING_V1"
TECHNOLOGY = "VLB"
ENGINE_VERSION = "vlb-engine/0.1"
DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_GROUP_SIZE = 64
DEFAULT_QUANT = "Q8_G64"
RESULTS_ENDPOINT = os.environ.get("RIFT_RESULTS_ENDPOINT", "https://rift-lm.vercel.app/api/results")
SOURCE_REF = os.environ.get("RIFT_SOURCE_REF", "unknown")
SUPPORTED_QUANTS = {"Q8_G64"}

# Weight reconstruction gate only. This is intentionally not called KR100.
MIN_TENSOR_COSINE = 0.9990
MAX_TENSOR_NRMSE = 0.0200

# First runtime keeps tied/embedding matrices FP16. That removes tied-weight
# ambiguity from the first engine proof while still quantizing the expensive
# Transformer Linear path. Later profiles may independently quantize them.
FP16_NAME_PATTERNS = (
    "embed_tokens.weight",
    "embedding.weight",
    "embeddings.weight",
    "lm_head.weight",
    "shared.weight",
)

# Teacher-free GOLD examples. The three partitions never overlap.
AMT_LEARN = [
    ("2 + 2 =", " 4"),
    ("5 + 5 =", " 10"),
    ("Binary 10 equals decimal", " 2"),
    ("The capital of France is", " Paris"),
    ("The capital of Brazil is", " Brasilia"),
    ("In Python, a list is", " mutable"),
    ("HTTP status for Not Found is", " 404"),
    ("The opposite of true is", " false"),
]
AMT_GATE_DEV = [
    ("7 + 1 =", " 8"),
    ("The capital of Italy is", " Rome"),
    ("Binary 11 equals decimal", " 3"),
    ("A Python tuple is usually", " immutable"),
]
AMT_FRESH_VALID = [
    ("3 + 3 =", " 6"),
    ("9 - 4 =", " 5"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Germany is", " Berlin"),
    ("HTTP status for OK is", " 200"),
    ("The boolean opposite of false is", " true"),
]


DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-96:]
    return f"{digest}_{clean}.pt"


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"User-Agent": "rift-vlb-stream/0.1", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class RangeReader:
    """Strict byte-range reader; full source responses are rejected."""

    def __init__(self, model_id: str, filename: str, token: Optional[str]):
        url = hf_hub_url(model_id, filename=filename)
        meta = get_hf_file_metadata(url, token=token)
        self.location = meta.location
        self.size = int(meta.size or 0)
        self.etag = str(meta.etag or "")
        self.filename = filename
        self.token = token
        if self.size <= 0:
            raise RuntimeError(f"Cannot resolve size for {model_id}/{filename}")

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        headers = auth_headers(self.token)
        headers["Range"] = f"bytes={start}-{start + length - 1}"
        with requests.get(self.location, headers=headers, stream=True, timeout=180, allow_redirects=True) as response:
            if response.status_code not in (200, 206):
                raise RuntimeError(f"Range request HTTP {response.status_code}: {self.filename}")
            data = response.content
        if len(data) != length:
            raise RuntimeError(
                f"Range contract failed for {self.filename}: requested {length}, received {len(data)}. "
                "Refusing a possible full-checkpoint response."
            )
        return data


@dataclass
class SafeTensorEntry:
    name: str
    dtype: str
    shape: List[int]
    absolute_start: int
    absolute_end: int

    @property
    def nbytes(self) -> int:
        return self.absolute_end - self.absolute_start


class SafeTensorRemoteFile:
    def __init__(self, model_id: str, filename: str, token: Optional[str]):
        self.reader = RangeReader(model_id, filename, token)
        self.filename = filename
        header_len = struct.unpack("<Q", self.reader.read(0, 8))[0]
        if header_len <= 2 or header_len > 256 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length: {header_len}")
        header_raw = self.reader.read(8, header_len)
        self.header_sha256 = sha256_bytes(header_raw)
        header = json.loads(header_raw.decode("utf-8"))
        data_start = 8 + header_len
        entries = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise RuntimeError(f"Invalid offsets for {name}")
            entries.append(
                SafeTensorEntry(
                    name=name,
                    dtype=str(meta["dtype"]),
                    shape=[int(x) for x in meta["shape"]],
                    absolute_start=data_start + int(offsets[0]),
                    absolute_end=data_start + int(offsets[1]),
                )
            )
        self.entries = sorted(entries, key=lambda x: x.absolute_start)

    def tensor(self, entry: SafeTensorEntry) -> torch.Tensor:
        dtype = DTYPE_MAP.get(entry.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported safetensors dtype {entry.dtype} for {entry.name}")
        raw = self.reader.read(entry.absolute_start, entry.nbytes)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
        expected = math.prod(entry.shape) if entry.shape else 1
        if tensor.numel() != expected:
            raise RuntimeError(f"Element count mismatch for {entry.name}: {tensor.numel()} != {expected}")
        return tensor.reshape(entry.shape).clone()


class VLBQuantizer:
    def __init__(self, output_dir: Path, group_size: int = DEFAULT_GROUP_SIZE):
        self.output_dir = output_dir
        self.tensor_dir = output_dir / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = int(group_size)

    @staticmethod
    def force_fp16(name: str) -> bool:
        lower = name.lower()
        return any(pattern in lower for pattern in FP16_NAME_PATTERNS)

    def quantize_q8(self, tensor: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
        x = tensor.detach().float().contiguous().reshape(-1)
        original_numel = x.numel()
        pad = (-original_numel) % self.group_size
        x_work = F.pad(x, (0, pad)) if pad else x
        groups = x_work.reshape(-1, self.group_size)
        max_abs = groups.abs().amax(dim=1)
        scales = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
        q = torch.round(groups / scales[:, None]).clamp(-127, 127).to(torch.int8)
        recon = (q.float() * scales[:, None]).reshape(-1)[:original_numel]
        return (
            {
                "format": "Q8_G64",
                "group_size": self.group_size,
                "shape": list(tensor.shape),
                "numel": original_numel,
                "qweight": q.cpu(),
                "scales": scales.to(torch.float16).cpu(),
            },
            recon.reshape(tensor.shape),
        )

    @staticmethod
    def metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        a = original.detach().float().reshape(-1)
        b = reconstructed.detach().float().reshape(-1)
        if a.numel() == 0:
            return {"cosine": 1.0, "nrmse": 0.0, "max_abs_error": 0.0}
        norm_a = float(torch.linalg.vector_norm(a).item())
        norm_b = float(torch.linalg.vector_norm(b).item())
        cosine = 1.0 if norm_a == 0.0 and norm_b == 0.0 else (
            0.0 if norm_a == 0.0 else float(F.cosine_similarity(a.double(), b.double(), dim=0).item())
        )
        rmse = float(torch.sqrt(torch.mean((a - b) ** 2)).item())
        rms_ref = float(torch.sqrt(torch.mean(a ** 2)).item())
        return {
            "cosine": cosine,
            "nrmse": rmse / max(rms_ref, 1e-12),
            "max_abs_error": float((a - b).abs().max().item()),
        }

    def write_tensor(self, name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        quantizable = (
            tensor.is_floating_point()
            and tensor.ndim == 2
            and tensor.numel() >= self.group_size
            and not self.force_fp16(name)
        )
        if quantizable:
            payload, recon = self.quantize_q8(tensor)
        elif tensor.is_floating_point():
            fp = tensor.detach().to(torch.float16).cpu().contiguous()
            payload = {"format": "FP16", "shape": list(tensor.shape), "tensor": fp}
            recon = fp.float()
        else:
            raw = tensor.detach().cpu().contiguous()
            payload = {"format": "RAW", "shape": list(tensor.shape), "tensor": raw}
            recon = raw.float()

        metric = self.metrics(tensor, recon)
        file_name = safe_name(name)
        destination = self.tensor_dir / file_name
        torch.save(payload, destination)
        return {
            "name": name,
            "file": f"tensors/{file_name}",
            "format": payload["format"],
            "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype).replace("torch.", ""),
            "source_bytes": tensor.numel() * tensor.element_size(),
            "artifact_bytes": destination.stat().st_size,
            "artifact_sha256": sha256_file(destination),
            "metrics": metric,
        }


def resolve_safetensor_files(model_id: str, token: Optional[str]) -> List[str]:
    info = HfApi(token=token).model_info(model_id, files_metadata=True)
    names = [str(s.rfilename) for s in info.siblings if str(s.rfilename).endswith(".safetensors")]
    canonical = [n for n in names if Path(n).name.startswith("model")]
    return sorted(canonical or names)


def fetch_small_json(model_id: str, filename: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        meta = get_hf_file_metadata(hf_hub_url(model_id, filename=filename), token=token)
        response = requests.get(meta.location, headers=auth_headers(token), timeout=60)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def stream_convert(model_id: str, output_dir: Path, token: Optional[str], quant: str) -> Dict[str, Any]:
    if quant not in SUPPORTED_QUANTS:
        raise RuntimeError(f"Unsupported VLB quant {quant}")
    files = resolve_safetensor_files(model_id, token)
    if not files:
        raise RuntimeError("VLB streaming v1 requires safetensors weights")

    output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = VLBQuantizer(output_dir)
    records: List[Dict[str, Any]] = []
    source_bytes = 0
    started = time.time()

    print(f"[VLB] model={model_id}")
    print(f"[VLB] safetensors files={len(files)}")
    print("[VLB] FULL SOURCE CHECKPOINT ON DISK = FALSE")

    for shard_idx, filename in enumerate(files, start=1):
        remote = SafeTensorRemoteFile(model_id, filename, token)
        print(
            f"[VLB] shard {shard_idx}/{len(files)} {filename} "
            f"remote_size={remote.reader.size / 1024**3:.2f}GiB tensors={len(remote.entries)}",
            flush=True,
        )
        for tensor_idx, entry in enumerate(remote.entries, start=1):
            tensor = remote.tensor(entry)
            source_bytes += entry.nbytes
            record = quantizer.write_tensor(entry.name, tensor)
            record.update(
                {
                    "source_file": filename,
                    "source_range": [entry.absolute_start, entry.absolute_end],
                    "source_range_bytes": entry.nbytes,
                    "source_header_sha256": remote.header_sha256,
                }
            )
            records.append(record)
            if tensor_idx % 25 == 0 or tensor_idx == len(remote.entries):
                print(
                    f"[VLB] {tensor_idx}/{len(remote.entries)} {entry.name} "
                    f"{record['format']} cos={record['metrics']['cosine']:.6f} "
                    f"nrmse={record['metrics']['nrmse']:.6f}",
                    flush=True,
                )
            del tensor
            gc.collect()

    for filename in ("config.json", "generation_config.json"):
        obj = fetch_small_json(model_id, filename, token)
        if obj is not None:
            (output_dir / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    qrecords = [r for r in records if r["format"] == "Q8_G64"]
    failed = [
        r["name"]
        for r in qrecords
        if r["metrics"]["cosine"] < MIN_TENSOR_COSINE or r["metrics"]["nrmse"] > MAX_TENSOR_NRMSE
    ]
    artifact_bytes = sum(int(r["artifact_bytes"]) for r in records)
    manifest = {
        "schema": "VLB-DIR-v1",
        "engine": ENGINE_VERSION,
        "technology": TECHNOLOGY,
        "model_id": model_id,
        "quant": quant,
        "streaming": {
            "enabled": True,
            "source_checkpoint_materialized": False,
            "transport": "HTTP_RANGE",
            "peak_source_tensor_count": 1,
        },
        "source_weight_bytes": source_bytes,
        "artifact_weight_bytes": artifact_bytes,
        "compression_ratio": source_bytes / max(artifact_bytes, 1),
        "tensor_count": len(records),
        "quantized_tensor_count": len(qrecords),
        "passthrough_tensor_count": len(records) - len(qrecords),
        "conversion_gate": {
            "min_tensor_cosine": MIN_TENSOR_COSINE,
            "max_tensor_nrmse": MAX_TENSOR_NRMSE,
            "observed_worst_cosine": min((r["metrics"]["cosine"] for r in qrecords), default=1.0),
            "observed_worst_nrmse": max((r["metrics"]["nrmse"] for r in qrecords), default=0.0),
            "failed_tensors": failed,
            "pass": not failed,
        },
        "tensor_records": records,
        "elapsed_seconds": time.time() - started,
        "verification": {
            "conversion_verified": not failed,
            "runtime_verified": False,
            "amt_verified": False,
            "engine_status": "CONVERSION_VERIFIED_RUNTIME_PENDING" if not failed else "CONVERSION_GATE_FAIL",
        },
    }
    manifest_path = output_dir / "vlb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def publish_record(record: Dict[str, Any]) -> bool:
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if len(token) < 32:
        print("[VLB] no valid RIFT_INGEST_TOKEN; result kept locally")
        return False
    response = requests.post(
        RESULTS_ENDPOINT,
        data=json.dumps({"records": [record]}, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "rift-vlb-battery/0.2",
        },
        timeout=60,
    )
    if not response.ok:
        print("[VLB] dashboard publish failed:", response.status_code, response.text[:500])
        return False
    print("[VLB] dashboard result published")
    return True


def build_record(model_id: str, manifest: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    conversion_pass = bool(manifest["verification"]["conversion_verified"])
    runtime_pass = bool(runtime.get("runtime_verified"))
    amt_pass = bool(runtime.get("amt_verified"))
    engine_pass = conversion_pass and runtime_pass and amt_pass
    source_bytes = int(manifest["source_weight_bytes"])
    artifact_bytes = int(manifest["artifact_weight_bytes"])
    return {
        "schema_version": 1,
        "technology": TECHNOLOGY,
        "model": model_id,
        "battery_id": BATTERY_ID,
        "benchmark_protocol": "VLB_AMT_PROOF_FIRST_V1",
        "status": "MEASURED" if conversion_pass else "FAILED",
        "source_ref": SOURCE_REF,
        "implementation": {
            "kind": "EXPERIMENTAL_NATIVE_RUNTIME",
            "scope": "streaming_requantization_vlb_runtime_amt",
            "native": runtime_pass,
            "simulated": False,
        },
        "baseline_disk_bytes": source_bytes,
        "candidate_disk_bytes": artifact_bytes,
        "disk_reduction_pct": (1.0 - artifact_bytes / max(source_bytes, 1)) * 100.0,
        "quality_gate_pass": engine_pass,
        "metrics": {
            "vlb": {
                "engine_version": ENGINE_VERSION,
                "quant": manifest["quant"],
                "streaming": manifest["streaming"],
                "conversion_gate": manifest["conversion_gate"],
                "compression_ratio": manifest["compression_ratio"],
                "manifest_sha256": manifest.get("manifest_sha256"),
            },
            "runtime_amt": runtime,
            "proof": {
                "conversion_verified": conversion_pass,
                "runtime_verified": runtime_pass,
                "amt_verified": amt_pass,
                "engine_status": "VLB_AMT_ENGINE_PROOF_PASS" if engine_pass else "VLB_AMT_ENGINE_NOT_YET_PROVEN",
            },
        },
        "notes": "VLB can rank only after conversion + native VLB runtime + teacher-free AMT all pass.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", default=DEFAULT_QUANT, choices=sorted(SUPPORTED_QUANTS))
    parser.add_argument("--output", default=None)
    parser.add_argument("--publish", choices=["on", "off"], default="on")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    model_name = args.model.split("/")[-1]
    root = Path(args.output or f"/content/vlb_run/{model_name}-{args.quant.lower()}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("VLB / AMT STREAMING ENGINE PROOF v1")
    print("=" * 112)
    print("model:", args.model)
    print("quant:", args.quant)
    print("output:", root)
    print("full source checkpoint materialized:", False)

    manifest = stream_convert(args.model, root, token, args.quant)
    if manifest["verification"]["conversion_verified"]:
        print("[VLB] conversion gate PASS; starting native VLB runtime proof")
        runtime = run_vlb_runtime_and_amt_proof(
            args.model,
            root,
            token,
            AMT_LEARN,
            AMT_GATE_DEV,
            AMT_FRESH_VALID,
        )
    else:
        runtime = {
            "runtime_verified": False,
            "amt_verified": False,
            "reason": "conversion gate failed; runtime not attempted",
        }

    manifest["verification"].update(
        {
            "runtime_verified": bool(runtime.get("runtime_verified")),
            "amt_verified": bool(runtime.get("amt_verified")),
            "engine_status": (
                "VLB_AMT_ENGINE_PROOF_PASS"
                if manifest["verification"]["conversion_verified"]
                and runtime.get("runtime_verified")
                and runtime.get("amt_verified")
                else "VLB_AMT_ENGINE_NOT_YET_PROVEN"
            ),
        }
    )
    (root / "vlb_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record = build_record(args.model, manifest, runtime)
    (root / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.publish == "on":
        publish_record(record)

    print()
    print("[VLB] source bytes   :", manifest["source_weight_bytes"])
    print("[VLB] artifact bytes :", manifest["artifact_weight_bytes"])
    print("[VLB] compression    : %.3fx" % manifest["compression_ratio"])
    print("[VLB] conversion     :", manifest["verification"]["conversion_verified"])
    print("[VLB] runtime        :", runtime.get("runtime_verified"))
    print("[VLB] AMT            :", runtime.get("amt_verified"))
    print("[VLB] engine status  :", manifest["verification"]["engine_status"])
    print("[VLB] manifest       :", root / "vlb_manifest.json")

    # Failed proof should be visible to the queue as failure; a measured record
    # was already published when credentials were available.
    return 0 if manifest["verification"]["engine_status"] == "VLB_AMT_ENGINE_PROOF_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
