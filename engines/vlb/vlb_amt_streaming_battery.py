#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLB/AMT Streaming Conversion Battery v1
=======================================

Proof-first generic conversion path for Hugging Face safetensors checkpoints.
The source checkpoint is NEVER materialized as a complete local file. Tensor
byte ranges are read directly from the Hub, quantized one tensor at a time and
written to a VLB-DIR artifact.

This first battery intentionally separates three claims:

  CONVERSION_VERIFIED
      Every source tensor was streamed, hashed/audited and reconstructed within
      the declared numerical gate.

  VLB_RUNTIME_VERIFIED
      The VLB package was loaded through the VLB runtime rather than the normal
      Transformers checkpoint path and produced a deterministic text-forward
      smoke result.

  AMT_VERIFIED
      A tiny teacher-free adapter/gate was trained only on embedded GOLD
      examples and improved or preserved a held-out validation slice.

Only when all three pass is engine_status=VLB_AMT_ENGINE_PROOF_PASS.
No KR100/general-intelligence claim is made by this battery.

Initial proof target:
    google/gemma-4-E4B-it

The implementation is architecture-agnostic at the package layer: any HF model
whose weights are safetensors can be streamed into VLB-DIR. Runtime proof is
currently enabled for models that Transformers can instantiate from config and
whose quantized 2-D weights belong to nn.Linear modules; unsupported modules
remain FP16 passthrough tensors in the package.
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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def ensure_deps() -> None:
    required = {
        "torch": "torch",
        "requests": "requests>=2.32",
        "huggingface_hub": "huggingface_hub>=0.27,<2",
        "safetensors": "safetensors>=0.5,<1",
        "transformers": "transformers>=4.56",
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
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata


BATTERY_ID = "VLB_AMT_STREAMING_V1"
TECHNOLOGY = "VLB"
ENGINE_VERSION = "vlb-engine/0.1"
DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_GROUP_SIZE = 64
DEFAULT_QUANT = "Q8_G64"
RESULTS_ENDPOINT = os.environ.get("RIFT_RESULTS_ENDPOINT", "https://rift-lm.vercel.app/api/results")
SOURCE_REF = os.environ.get("RIFT_SOURCE_REF", "unknown")

# Proof-first: Q8 is intentionally the first generic format. Q4 comes only
# after the end-to-end runtime path is proven on Gemma 4.
SUPPORTED_QUANTS = {"Q8_G64"}

# Numerical conversion gate. This is NOT KR100. It only certifies weight
# reconstruction quality for this deployment artifact.
MIN_TENSOR_COSINE = 0.9990
MAX_TENSOR_NRMSE = 0.0200

# Tiny AMT set. GOLD is embedded and teacher-free. Train/validation are fixed.
AMT_TRAIN = [
    ("2 + 2 =", " 4"),
    ("The capital of France is", " Paris"),
    ("In Python, a list is", " mutable"),
    ("Binary 10 equals decimal", " 2"),
]
AMT_VALID = [
    ("3 + 3 =", " 6"),
    ("The capital of Japan is", " Tokyo"),
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
    """HTTP byte-range reader. It never writes the source checkpoint to disk."""

    def __init__(self, model_id: str, filename: str, token: Optional[str]):
        self.model_id = model_id
        self.filename = filename
        self.token = token
        url = hf_hub_url(model_id, filename=filename)
        meta = get_hf_file_metadata(url, token=token)
        self.location = meta.location
        self.size = int(meta.size or 0)
        self.etag = str(meta.etag or "")
        if self.size <= 0:
            raise RuntimeError(f"Cannot resolve size for {model_id}/{filename}")

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = start + length - 1
        headers = auth_headers(self.token)
        headers["Range"] = f"bytes={start}-{end}"
        with requests.get(self.location, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Range request failed HTTP {r.status_code} for {self.filename}")
            data = r.content
        # A server returning 200 ignored Range. Refuse rather than accidentally
        # retaining a full 16 GB source response in memory.
        if len(data) != length:
            raise RuntimeError(
                f"Range contract failed for {self.filename}: requested {length} bytes, received {len(data)}"
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
        self.model_id = model_id
        self.filename = filename
        first = self.reader.read(0, 8)
        header_len = struct.unpack("<Q", first)[0]
        if header_len <= 2 or header_len > 256 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length: {header_len}")
        header_raw = self.reader.read(8, header_len)
        self.header_sha256 = sha256_bytes(header_raw)
        self.header = json.loads(header_raw.decode("utf-8"))
        self.data_start = 8 + header_len
        self.entries: List[SafeTensorEntry] = []
        for name, meta in self.header.items():
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise RuntimeError(f"Invalid offsets for tensor {name}")
            self.entries.append(
                SafeTensorEntry(
                    name=name,
                    dtype=str(meta["dtype"]),
                    shape=[int(x) for x in meta["shape"]],
                    absolute_start=self.data_start + int(offsets[0]),
                    absolute_end=self.data_start + int(offsets[1]),
                )
            )
        self.entries.sort(key=lambda x: x.absolute_start)

    def tensor(self, entry: SafeTensorEntry) -> torch.Tensor:
        if entry.dtype not in DTYPE_MAP:
            raise RuntimeError(f"Unsupported safetensors dtype {entry.dtype} for {entry.name}")
        raw = self.reader.read(entry.absolute_start, entry.nbytes)
        # bytearray gives writable backing storage and avoids torch warnings.
        tensor = torch.frombuffer(bytearray(raw), dtype=DTYPE_MAP[entry.dtype])
        expected = math.prod(entry.shape) if entry.shape else 1
        if tensor.numel() != expected:
            raise RuntimeError(
                f"Tensor element count mismatch for {entry.name}: {tensor.numel()} != {expected}"
            )
        return tensor.reshape(entry.shape).clone()


class VLBQuantizer:
    def __init__(self, output_dir: Path, group_size: int = DEFAULT_GROUP_SIZE):
        self.output_dir = output_dir
        self.tensor_dir = output_dir / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = int(group_size)

    def quantize_q8(self, tensor: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
        x = tensor.detach().float().contiguous().reshape(-1)
        original_numel = x.numel()
        pad = (-original_numel) % self.group_size
        if pad:
            x_work = F.pad(x, (0, pad))
        else:
            x_work = x
        groups = x_work.reshape(-1, self.group_size)
        max_abs = groups.abs().amax(dim=1)
        scales = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
        q = torch.round(groups / scales[:, None]).clamp(-127, 127).to(torch.int8)
        recon = (q.float() * scales[:, None]).reshape(-1)[:original_numel]
        payload = {
            "format": "Q8_G64",
            "group_size": self.group_size,
            "shape": list(tensor.shape),
            "numel": original_numel,
            "qweight": q.cpu(),
            "scales": scales.to(torch.float16).cpu(),
        }
        return payload, recon.reshape(tensor.shape)

    def fp16_passthrough(self, tensor: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor]:
        fp = tensor.detach().to(torch.float16).cpu().contiguous()
        payload = {
            "format": "FP16",
            "shape": list(tensor.shape),
            "tensor": fp,
        }
        return payload, fp.float()

    @staticmethod
    def metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        a = original.detach().float().reshape(-1)
        b = reconstructed.detach().float().reshape(-1)
        if a.numel() == 0:
            return {"cosine": 1.0, "nrmse": 0.0, "max_abs_error": 0.0}
        denom = float(torch.linalg.vector_norm(a).item())
        if denom == 0.0:
            cosine = 1.0 if float(torch.linalg.vector_norm(b).item()) == 0.0 else 0.0
        else:
            cosine = float(F.cosine_similarity(a.double(), b.double(), dim=0).item())
        rmse = float(torch.sqrt(torch.mean((a - b) ** 2)).item())
        rms_ref = float(torch.sqrt(torch.mean(a ** 2)).item())
        nrmse = rmse / max(rms_ref, 1e-12)
        max_abs = float((a - b).abs().max().item())
        return {"cosine": cosine, "nrmse": nrmse, "max_abs_error": max_abs}

    def write_tensor(self, name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        quantizable = tensor.is_floating_point() and tensor.ndim == 2 and tensor.numel() >= self.group_size
        if quantizable:
            payload, recon = self.quantize_q8(tensor)
        else:
            payload, recon = self.fp16_passthrough(tensor) if tensor.is_floating_point() else (
                {"format": "RAW", "shape": list(tensor.shape), "tensor": tensor.cpu().contiguous()},
                tensor.detach().float().cpu(),
            )
        metrics = self.metrics(tensor, recon)
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
            "metrics": metrics,
        }


def resolve_safetensor_files(model_id: str, token: Optional[str]) -> List[str]:
    info = HfApi(token=token).model_info(model_id, files_metadata=True)
    names = [str(s.rfilename) for s in info.siblings if str(s.rfilename).endswith(".safetensors")]
    # Exclude adapter/optimizer artifacts if a canonical model file exists.
    model_files = [n for n in names if Path(n).name.startswith("model")]
    return sorted(model_files or names)


def fetch_small_json(model_id: str, filename: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    url = hf_hub_url(model_id, filename=filename)
    try:
        meta = get_hf_file_metadata(url, token=token)
        r = requests.get(meta.location, headers=auth_headers(token), timeout=60)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def stream_convert(model_id: str, output_dir: Path, token: Optional[str], quant: str) -> Dict[str, Any]:
    if quant not in SUPPORTED_QUANTS:
        raise RuntimeError(f"Unsupported VLB quant {quant}; supported={sorted(SUPPORTED_QUANTS)}")
    safetensor_files = resolve_safetensor_files(model_id, token)
    if not safetensor_files:
        raise RuntimeError("No safetensors weights found; VLB streaming v1 requires safetensors")

    output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = VLBQuantizer(output_dir)
    tensor_records: List[Dict[str, Any]] = []
    source_bytes = 0
    start = time.time()

    print(f"[VLB] model={model_id}")
    print(f"[VLB] source safetensors={len(safetensor_files)}")
    print("[VLB] source checkpoint will NOT be materialized on disk")

    for file_index, filename in enumerate(safetensor_files, start=1):
        remote = SafeTensorRemoteFile(model_id, filename, token)
        print(
            f"[VLB] shard {file_index}/{len(safetensor_files)} {filename} "
            f"size={remote.reader.size / 1024**3:.2f} GiB tensors={len(remote.entries)}",
            flush=True,
        )
        for tensor_index, entry in enumerate(remote.entries, start=1):
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
            tensor_records.append(record)
            if tensor_index % 25 == 0 or tensor_index == len(remote.entries):
                print(
                    f"[VLB] {filename}: {tensor_index}/{len(remote.entries)} "
                    f"last={entry.name} {record['format']} cos={record['metrics']['cosine']:.6f} "
                    f"nrmse={record['metrics']['nrmse']:.6f}",
                    flush=True,
                )
            del tensor
            gc.collect()

    config = fetch_small_json(model_id, "config.json", token)
    generation_config = fetch_small_json(model_id, "generation_config.json", token)
    if config is not None:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if generation_config is not None:
        (output_dir / "generation_config.json").write_text(
            json.dumps(generation_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    worst_cos = min((r["metrics"]["cosine"] for r in tensor_records if r["format"] == "Q8_G64"), default=1.0)
    worst_nrmse = max((r["metrics"]["nrmse"] for r in tensor_records if r["format"] == "Q8_G64"), default=0.0)
    failed = [
        r["name"]
        for r in tensor_records
        if r["format"] == "Q8_G64"
        and (r["metrics"]["cosine"] < MIN_TENSOR_COSINE or r["metrics"]["nrmse"] > MAX_TENSOR_NRMSE)
    ]
    artifact_bytes = sum(int(r["artifact_bytes"]) for r in tensor_records)
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
        "tensor_count": len(tensor_records),
        "quantized_tensor_count": sum(r["format"] == "Q8_G64" for r in tensor_records),
        "passthrough_tensor_count": sum(r["format"] != "Q8_G64" for r in tensor_records),
        "conversion_gate": {
            "min_tensor_cosine": MIN_TENSOR_COSINE,
            "max_tensor_nrmse": MAX_TENSOR_NRMSE,
            "observed_worst_cosine": worst_cos,
            "observed_worst_nrmse": worst_nrmse,
            "failed_tensors": failed,
            "pass": not failed,
        },
        "tensor_records": tensor_records,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "verification": {
            "conversion_verified": not failed,
            "runtime_verified": False,
            "amt_verified": False,
            "engine_status": "CONVERSION_ONLY_UNVERIFIED_RUNTIME",
        },
    }
    manifest_path = output_dir / "vlb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


class AMTLogitAdapter(nn.Module):
    """Tiny model-agnostic logit-space residual used only by the runtime proof.

    It learns a low-rank correction over the top-K vocabulary slice selected by
    the base model. This keeps the proof adapter tiny and avoids architecture-
    specific transformer surgery. A gate decides whether the correction is
    applied. The base model is always frozen.
    """

    def __init__(self, k: int = 64, rank: int = 8):
        super().__init__()
        self.k = k
        self.down = nn.Linear(k, rank, bias=False)
        self.up = nn.Linear(rank, k, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 1))
        nn.init.zeros_(self.up.weight)

    def forward(self, top_logits: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self.up(F.silu(self.down(top_logits)))
        p_continue = torch.sigmoid(self.gate(features)).squeeze(-1)
        refined = top_logits + p_continue.unsqueeze(-1) * delta
        return refined, p_continue


def runtime_and_amt_proof(*args, **kwargs) -> Dict[str, Any]:
    """Runtime v1 status gate.

    The VLB-DIR converter is real and generic. A fully lazy Transformers loader
    for Gemma 4 is the next runtime milestone and is intentionally NOT faked in
    this first commit. Returning an explicit blocked status prevents the
    dashboard from treating conversion metrics as proof that generation uses
    VLB weights.
    """
    return {
        "runtime_verified": False,
        "amt_verified": False,
        "reason": (
            "VLB-DIR streaming conversion is implemented, but the generic lazy "
            "VLBQuantLinear Transformers loader is not certified yet. AMT cannot "
            "be claimed until generation is executed from VLB-DIR rather than the upstream checkpoint."
        ),
        "next_gate": "IMPLEMENT_AND_REPLAY_VLB_LAZY_RUNTIME",
    }


def publish_record(record: Dict[str, Any]) -> bool:
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if len(token) < 32:
        print("[VLB] RIFT_INGEST_TOKEN missing/short; result kept locally only")
        return False
    payload = json.dumps({"records": [record]}, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "rift-vlb-battery/0.1",
    }
    r = requests.post(RESULTS_ENDPOINT, data=payload, headers=headers, timeout=60)
    if not r.ok:
        print("[VLB] dashboard publish failed:", r.status_code, r.text[:500])
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
    reduction = 1.0 - artifact_bytes / max(source_bytes, 1)
    return {
        "schema_version": 1,
        "technology": TECHNOLOGY,
        "model": model_id,
        "battery_id": BATTERY_ID,
        "benchmark_protocol": "VLB_AMT_PROOF_FIRST_V1",
        "status": "MEASURED" if conversion_pass else "FAILED",
        "source_ref": SOURCE_REF,
        "implementation": {
            "kind": "EXPERIMENTAL_NATIVE_FORMAT",
            "scope": "streaming_requantization_and_engine_proof",
            "native": False,
            "simulated": False,
        },
        "baseline_disk_bytes": source_bytes,
        "candidate_disk_bytes": artifact_bytes,
        "disk_reduction_pct": reduction * 100.0,
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
            "amt": runtime,
            "proof": {
                "conversion_verified": conversion_pass,
                "runtime_verified": runtime_pass,
                "amt_verified": amt_pass,
                "engine_status": "VLB_AMT_ENGINE_PROOF_PASS" if engine_pass else "VLB_AMT_ENGINE_NOT_YET_PROVEN",
            },
        },
        "notes": (
            "Streaming VLB-DIR conversion is measured. quality_gate_pass remains false "
            "until generation is replayed through the VLB runtime and AMT is validated."
        ),
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

    print("=" * 110)
    print("VLB / AMT STREAMING ENGINE PROOF v1")
    print("=" * 110)
    print("model:", args.model)
    print("quant:", args.quant)
    print("output:", root)
    print("source full checkpoint on disk: NEVER")

    manifest = stream_convert(args.model, root, token, args.quant)
    runtime = runtime_and_amt_proof(args.model, root, token)
    manifest["verification"].update(runtime)
    if manifest["verification"]["conversion_verified"] and runtime.get("runtime_verified") and runtime.get("amt_verified"):
        manifest["verification"]["engine_status"] = "VLB_AMT_ENGINE_PROOF_PASS"
    else:
        manifest["verification"]["engine_status"] = "VLB_AMT_ENGINE_NOT_YET_PROVEN"
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

    # Conversion may pass while engine proof remains intentionally incomplete.
    # Return 0 so the queue publishes the measured experimental result; the
    # quality gate remains false and therefore cannot win the technology ranking.
    return 0 if manifest["verification"]["conversion_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
