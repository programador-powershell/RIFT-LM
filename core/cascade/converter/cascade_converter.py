#!/usr/bin/env python3
# ==============================================================================
# CASCADE Model Converter v0.1
# Development format: CASCADE-DIR/0.1
#
# Converts local open-weight checkpoints into a smaller CASCADE representation:
#
#     W ~= F0_INT4 + F1_LOWRANK
#
# IMPORTANT:
# - This is a WEIGHT CONVERTER, not yet the production CASCADE runtime.
# - It does not claim end-to-end model quality from weight-local metrics.
# - Dynamic Confidence Gates require activation calibration and are emitted as
#   CALIBRATION_REQUIRED by default.
# - The output format is a development directory format because the binary
#   CASCADE Bundle M0 ABI is not frozen in the source specification.
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


FORMAT_NAME = "CASCADE-DIR"
FORMAT_VERSION = "0.1"
CONVERTER_VERSION = "0.1"
DEFAULT_COSINE_MIN = 0.995
DEFAULT_NRMSE_MAX = 0.05

# Schema v2 (docs/C3_CONTRACTS_V1.md §3) para dashboard_battery.json.
SCHEMA_VERSION = 2
BENCHMARK_PROTOCOL = "CONVERTER_STATIC_V1"
DEFAULT_RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results"

# Guarda de disco: margem mínima de espaço livre além da projeção do tensor.
DEFAULT_DISK_BUDGET_GB = 75.0
DISK_FREE_MARGIN_BYTES = 1024 ** 3  # 1 GiB


# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug(name: str) -> str:
    v = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    v = v.strip("._-")
    return v or hashlib.sha256(name.encode()).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def json_dump_atomic(path: Path, obj: Any) -> None:
    """Escrita atômica (tmp + os.replace) para estado retomável por tensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def package_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(name)
    except Exception:
        return None


def colab_available() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def destructive_delete_allowed(path: Path) -> bool:
    """
    Guarda de segurança para operações destrutivas (shutil.rmtree / unlink de
    shards de origem): permitido somente no Colab, sob /content ou /tmp, ou
    quando o usuário opta explicitamente via RIFT_ALLOW_LOCAL_CLEANUP=1.
    """
    if os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP") == "1":
        return True
    if colab_available():
        return True
    posix = path.resolve().as_posix()
    return (
        posix.startswith("/content/")
        or posix.startswith("/tmp/")
        or posix in ("/content", "/tmp")
    )


def parse_ranks(value: str) -> List[int]:
    ranks = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not ranks or any(r <= 0 for r in ranks):
        raise argparse.ArgumentTypeError("ranks deve conter inteiros positivos, ex: 8,16,32")
    return ranks


def dtype_itemsize(dtype: str) -> int:
    table = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
    if dtype not in table:
        raise ValueError(f"dtype Safetensors não suportado para tamanho: {dtype}")
    return table[dtype]


# ------------------------------------------------------------------------------
# Streaming metrics
# ------------------------------------------------------------------------------

class Metrics:
    def __init__(self):
        self.dot = 0.0
        self.a2 = 0.0
        self.b2 = 0.0
        self.sse = 0.0
        self.n = 0
        self.amin = math.inf
        self.amax = -math.inf

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        af = np.asarray(a, dtype=np.float32).reshape(-1)
        bf = np.asarray(b, dtype=np.float32).reshape(-1)
        self.dot += float(np.dot(af, bf))
        self.a2 += float(np.dot(af, af))
        self.b2 += float(np.dot(bf, bf))
        diff = af - bf
        self.sse += float(np.dot(diff, diff))
        self.n += af.size
        if af.size:
            self.amin = min(self.amin, float(af.min()))
            self.amax = max(self.amax, float(af.max()))

    def result(self) -> Dict[str, float]:
        cosine = self.dot / max(math.sqrt(self.a2) * math.sqrt(self.b2), 1e-30)
        rmse = math.sqrt(self.sse / max(self.n, 1))
        rng = max(self.amax - self.amin, 1e-12)
        return {
            "cosine": float(cosine),
            "rmse": float(rmse),
            "nrmse": float(rmse / rng),
        }


def quality_pass(m: Dict[str, float], cosine_min: float, nrmse_max: float) -> bool:
    return m["cosine"] >= cosine_min and m["nrmse"] <= nrmse_max


# ------------------------------------------------------------------------------
# Source descriptors
# ------------------------------------------------------------------------------

@dataclass
class TensorDesc:
    name: str
    shape: List[int]
    dtype: str
    nbytes: int
    source_file: str
    data_start: int = 0
    data_end: int = 0
    source_kind: str = "safetensors"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


class Source:
    model_id: str

    def tensors(self) -> List[TensorDesc]:
        raise NotImplementedError

    def iter_rows(self, desc: TensorDesc, chunk_rows: int) -> Iterator[Tuple[int, np.ndarray]]:
        raise NotImplementedError

    def copy_raw_tensor(self, desc: TensorDesc, dst: Path) -> None:
        raise NotImplementedError

    def copy_sidecars(self, output_dir: Path) -> List[str]:
        return []

    def source_weights_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors())


# ------------------------------------------------------------------------------
# NPZ source (also used by self-test)
# ------------------------------------------------------------------------------

class NPZSource(Source):
    def __init__(self, path: Path):
        self.path = path
        self.model_id = path.stem
        with np.load(path, mmap_mode="r") as z:
            self._descs = []
            for k in z.files:
                a = z[k]
                self._descs.append(
                    TensorDesc(
                        name=k,
                        shape=list(a.shape),
                        dtype=str(a.dtype),
                        nbytes=int(a.nbytes),
                        source_file=str(path),
                        source_kind="npz",
                    )
                )

    def tensors(self) -> List[TensorDesc]:
        return self._descs

    def iter_rows(self, desc: TensorDesc, chunk_rows: int):
        with np.load(self.path, mmap_mode="r") as z:
            a = z[desc.name]
            if a.ndim != 2:
                raise ValueError("iter_rows suporta somente tensor 2D")
            for start in range(0, a.shape[0], chunk_rows):
                yield start, np.asarray(a[start:start + chunk_rows], dtype=np.float32)

    def copy_raw_tensor(self, desc: TensorDesc, dst: Path) -> None:
        with np.load(self.path, mmap_mode="r") as z:
            a = np.asarray(z[desc.name])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(a.tobytes(order="C"))


# ------------------------------------------------------------------------------
# Safetensors source
# ------------------------------------------------------------------------------

def parse_safetensors_header(path: Path) -> Tuple[int, Dict[str, Any]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Safetensors inválido: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"Header truncado: {path}")
        header = json.loads(header_raw)
    return 8 + header_len, header


class SafeTensorSource(Source):
    SIDECAR_PATTERNS = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "*.model",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    )

    def __init__(self, input_path: Path, model_id: Optional[str] = None):
        self.input_path = input_path
        self.root = input_path if input_path.is_dir() else input_path.parent
        self.model_id = model_id or self.root.name

        files: List[Path]
        if input_path.is_file():
            files = [input_path]
        else:
            files = sorted(input_path.glob("*.safetensors"))
        if not files:
            raise ValueError(f"Nenhum .safetensors encontrado em {input_path}")

        descs: List[TensorDesc] = []
        seen = set()
        self._tensor_file: Dict[str, Path] = {}
        for file in files:
            data_base, header = parse_safetensors_header(file)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if name in seen:
                    raise ValueError(f"Tensor duplicado entre shards: {name}")
                seen.add(name)
                start, end = meta["data_offsets"]
                dtype = meta["dtype"]
                shape = [int(x) for x in meta["shape"]]
                nbytes = int(end - start)
                expected = math.prod(shape) * dtype_itemsize(dtype)
                if expected != nbytes:
                    raise ValueError(
                        f"{name}: bytes do header ({nbytes}) != shape*dtype ({expected})"
                    )
                descs.append(
                    TensorDesc(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        nbytes=nbytes,
                        source_file=str(file),
                        data_start=int(data_base + start),
                        data_end=int(data_base + end),
                        source_kind="safetensors",
                    )
                )
                self._tensor_file[name] = file
        self._descs = sorted(descs, key=lambda x: x.name)

    def tensors(self) -> List[TensorDesc]:
        return self._descs

    @staticmethod
    def _require_torch_safetensors():
        try:
            import torch
            from safetensors import safe_open
            return torch, safe_open
        except ImportError as exc:
            raise SystemExit(
                "Para converter Safetensors instale: pip install torch safetensors\n"
                f"Erro: {exc}"
            )

    def iter_rows(self, desc: TensorDesc, chunk_rows: int):
        torch, safe_open = self._require_torch_safetensors()
        file = self._tensor_file[desc.name]
        with safe_open(str(file), framework="pt", device="cpu") as f:
            s = f.get_slice(desc.name)
            rows = desc.shape[0]
            for start in range(0, rows, chunk_rows):
                t = s[start:start + chunk_rows, :]
                yield start, t.float().contiguous().numpy()

    def copy_raw_tensor(self, desc: TensorDesc, dst: Path) -> None:
        src = Path(desc.source_file)
        dst.parent.mkdir(parents=True, exist_ok=True)
        remaining = desc.data_end - desc.data_start
        with src.open("rb") as fi, dst.open("wb") as fo:
            fi.seek(desc.data_start)
            while remaining:
                block = fi.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise IOError(f"EOF inesperado ao copiar {desc.name}")
                fo.write(block)
                remaining -= len(block)

    def copy_sidecars(self, output_dir: Path) -> List[str]:
        copied = []
        config_dir = output_dir / "source_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for pattern in self.SIDECAR_PATTERNS:
            for p in self.root.glob(pattern):
                if p.is_file() and p.name not in seen:
                    seen.add(p.name)
                    shutil.copy2(p, config_dir / p.name)
                    copied.append(str(Path("source_config") / p.name))
        return sorted(copied)


# ------------------------------------------------------------------------------
# INT4 F0
# ------------------------------------------------------------------------------

def quantize_int4_rows(x: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Signed two's-complement INT4, groupwise on last dimension.
    Returns packed uint8 rows, FP16 scales, reconstructed FP32.
    """
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size

    if padded_cols != cols:
        xp = np.pad(x, ((0, 0), (0, padded_cols - cols)))
    else:
        xp = x

    g = xp.reshape(rows, groups, group_size)
    maxabs = np.max(np.abs(g), axis=2)
    scales32 = maxabs / 7.0
    scales32[scales32 == 0] = 1.0

    # Store FP16 scales, and quantize against exactly the stored value.
    scales16 = scales32.astype(np.float16)
    scales = scales16.astype(np.float32)

    q = np.rint(g / scales[:, :, None]).clip(-8, 7).astype(np.int8)
    qflat = q.reshape(rows, padded_cols)[:, :cols]

    # two's complement nibble
    nib = (qflat.astype(np.int16) & 0x0F).astype(np.uint8)
    packed_cols = math.ceil(cols / 2)
    packed = np.zeros((rows, packed_cols), dtype=np.uint8)
    packed[:, :] = nib[:, 0::2]
    if cols > 1:
        high = nib[:, 1::2]
        packed[:, :high.shape[1]] |= high << 4

    recon_groups = q.astype(np.float32) * scales[:, :, None]
    recon = recon_groups.reshape(rows, padded_cols)[:, :cols]
    return packed, scales16, recon


def dequant_int4_rows(
    packed: np.ndarray,
    scales16: np.ndarray,
    cols: int,
    group_size: int,
) -> np.ndarray:
    rows = packed.shape[0]
    nib = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    nib = nib[:, :cols]

    qi = nib.astype(np.int8)
    qi[qi >= 8] -= 16

    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols != cols:
        qp = np.pad(qi, ((0, 0), (0, padded_cols - cols)))
    else:
        qp = qi
    qg = qp.reshape(rows, groups, group_size).astype(np.float32)
    scales = scales16.astype(np.float32)
    recon = qg * scales[:, :, None]
    return recon.reshape(rows, padded_cols)[:, :cols]


def write_f0(
    source: Source,
    desc: TensorDesc,
    out_dir: Path,
    group_size: int,
    chunk_rows: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    rows, cols = desc.shape
    groups = math.ceil(cols / group_size)
    packed_cols = math.ceil(cols / 2)

    packed_path = out_dir / "f0.int4"
    scales_path = out_dir / "f0.scales.f16"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = Metrics()
    with packed_path.open("wb") as fp, scales_path.open("wb") as fs:
        for _, w in source.iter_rows(desc, chunk_rows):
            packed, scales16, recon = quantize_int4_rows(w, group_size)
            fp.write(packed.tobytes(order="C"))
            fs.write(scales16.tobytes(order="C"))
            metrics.update(w, recon)

    meta = {
        "stage_index": 0,
        "stage_type": "BASE_STAGE",
        "representation": "INT4_GROUP_SYMMETRIC_TWOS_COMPLEMENT",
        "group_size": group_size,
        "shape": [rows, cols],
        "packed_row_bytes": packed_cols,
        "scale_groups_per_row": groups,
        "scale_dtype": "FP16",
        "resident_hint": "HOT",
        "files": {
            "packed": str(packed_path.name),
            "scales": str(scales_path.name),
        },
        "bytes": packed_path.stat().st_size + scales_path.stat().st_size,
        "sha256": {
            "packed": sha256_file(packed_path),
            "scales": sha256_file(scales_path),
        },
    }
    return meta, metrics.result()


def open_f0_memmaps(out_dir: Path, rows: int, cols: int, group_size: int):
    packed_cols = math.ceil(cols / 2)
    groups = math.ceil(cols / group_size)
    packed = np.memmap(out_dir / "f0.int4", dtype=np.uint8, mode="r",
                       shape=(rows, packed_cols))
    scales = np.memmap(out_dir / "f0.scales.f16", dtype=np.float16, mode="r",
                       shape=(rows, groups))
    return packed, scales


# ------------------------------------------------------------------------------
# Streaming randomized low-rank residual
# ------------------------------------------------------------------------------

def residual_chunk(
    source_chunk: np.ndarray,
    packed_chunk: np.ndarray,
    scales_chunk: np.ndarray,
    cols: int,
    group_size: int,
) -> np.ndarray:
    f0 = dequant_int4_rows(packed_chunk, scales_chunk, cols, group_size)
    return source_chunk - f0


def randomized_residual_factors(
    source: Source,
    desc: TensorDesc,
    out_dir: Path,
    group_size: int,
    chunk_rows: int,
    max_rank: int,
    oversample: int,
    power_iters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = desc.shape
    l = min(max_rank + oversample, rows, cols)
    if l <= 0:
        raise ValueError("rank inválido")

    packed, scales = open_f0_memmaps(out_dir, rows, cols, group_size)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, l), dtype=np.float32) / math.sqrt(max(cols, 1))

    y_path = out_dir / ".tmp_random_projection.f32"
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(rows, l))

    # Y = R @ Omega
    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        r = residual_chunk(w, packed[start:end], scales[start:end], cols, group_size)
        y[start:end] = r @ omega
    y.flush()

    q, _ = np.linalg.qr(np.asarray(y), mode="reduced")
    q = np.asarray(q, dtype=np.float32)

    # Optional power iterations: Q <- orth(R @ (R.T @ Q))
    for _ in range(power_iters):
        z = np.zeros((cols, q.shape[1]), dtype=np.float32)
        for start, w in source.iter_rows(desc, chunk_rows):
            end = start + w.shape[0]
            r = residual_chunk(w, packed[start:end], scales[start:end], cols, group_size)
            z += r.T @ q[start:end]

        for start, w in source.iter_rows(desc, chunk_rows):
            end = start + w.shape[0]
            r = residual_chunk(w, packed[start:end], scales[start:end], cols, group_size)
            y[start:end] = r @ z
        y.flush()
        q, _ = np.linalg.qr(np.asarray(y), mode="reduced")
        q = np.asarray(q, dtype=np.float32)

    # B = Q.T @ R
    b = np.zeros((q.shape[1], cols), dtype=np.float32)
    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        r = residual_chunk(w, packed[start:end], scales[start:end], cols, group_size)
        b += q[start:end].T @ r

    uhat, s, vh = np.linalg.svd(b, full_matrices=False)
    rmax = min(max_rank, len(s))
    # absorb singular values into U
    u = (q @ uhat[:, :rmax]) * s[:rmax][None, :]
    v = vh[:rmax, :].T

    try:
        del y
        y_path.unlink(missing_ok=True)
    except Exception:
        pass

    return np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)


def evaluate_rank_candidates(
    source: Source,
    desc: TensorDesc,
    out_dir: Path,
    group_size: int,
    chunk_rows: int,
    u: np.ndarray,
    v: np.ndarray,
    ranks: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    rows, cols = desc.shape
    packed, scales = open_f0_memmaps(out_dir, rows, cols, group_size)

    # Evaluate what will actually be stored: FP16 factors.
    uq = u.astype(np.float16).astype(np.float32)
    vq = v.astype(np.float16).astype(np.float32)

    acc = {r: Metrics() for r in ranks if r <= uq.shape[1]}
    ranks2 = sorted(acc)
    if not ranks2:
        return {}

    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        pred = dequant_int4_rows(packed[start:end], scales[start:end], cols, group_size)
        prev = 0
        for r in ranks2:
            pred = pred + uq[start:end, prev:r] @ vq[:, prev:r].T
            acc[r].update(w, pred)
            prev = r

    return {r: acc[r].result() for r in ranks2}


def write_f1(
    out_dir: Path,
    u: np.ndarray,
    v: np.ndarray,
    rank: int,
) -> Dict[str, Any]:
    up = out_dir / "f1.u.f16"
    vp = out_dir / "f1.v.f16"
    np.asarray(u[:, :rank], dtype=np.float16).tofile(up)
    np.asarray(v[:, :rank], dtype=np.float16).tofile(vp)

    return {
        "stage_index": 1,
        "stage_type": "RESIDUAL_LOWRANK",
        "representation": "LOWRANK_FP16",
        "rank": int(rank),
        "u_shape": [int(u.shape[0]), int(rank)],
        "v_shape": [int(v.shape[0]), int(rank)],
        "resident_hint": "WARM",
        "files": {"u": up.name, "v": vp.name},
        "bytes": up.stat().st_size + vp.stat().st_size,
        "sha256": {"u": sha256_file(up), "v": sha256_file(vp)},
    }


# ------------------------------------------------------------------------------
# Raw passthrough
# ------------------------------------------------------------------------------

def write_raw_stage(source: Source, desc: TensorDesc, out_dir: Path, reason: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "f0.raw"
    source.copy_raw_tensor(desc, p)
    return {
        "stage_index": 0,
        "stage_type": "FULL_STAGE",
        "representation": "SOURCE_RAW",
        "source_dtype": desc.dtype,
        "shape": desc.shape,
        "resident_hint": "HOT",
        "files": {"raw": p.name},
        "bytes": p.stat().st_size,
        "sha256": {"raw": sha256_file(p)},
        "reason": reason,
    }


# ------------------------------------------------------------------------------
# Conversion policy
# ------------------------------------------------------------------------------

DEFAULT_EXCLUDE = re.compile(
    r"(?:^|\.)(?:embed_tokens|embeddings?|lm_head)(?:\.|$)|"
    r"(?:^|\.)(?:experts?|moe)(?:\.|$)",
    re.IGNORECASE,
)


def eligible_matrix(
    desc: TensorDesc,
    min_elements: int,
    include_embeddings: bool,
    include_moe: bool,
    include_regex: Optional[re.Pattern],
    exclude_regex: Optional[re.Pattern],
) -> Tuple[bool, str]:
    if desc.ndim != 2:
        return False, "non_2d_passthrough"
    if desc.elements < min_elements:
        return False, "small_tensor_passthrough"
    if include_regex and not include_regex.search(desc.name):
        return False, "not_selected_by_include_regex"
    if exclude_regex and exclude_regex.search(desc.name):
        return False, "excluded_by_user_regex"

    lname = desc.name.lower()
    if not include_embeddings and (
        "embed_tokens" in lname or "embedding" in lname or "lm_head" in lname
    ):
        return False, "embedding_or_lm_head_passthrough"
    if not include_moe and ("expert" in lname or ".moe" in lname):
        return False, "moe_passthrough_phase1"

    return True, "eligible_linear_matrix"


# ------------------------------------------------------------------------------
# Converter
# ------------------------------------------------------------------------------

def make_source(input_path: Path, model_id: Optional[str]) -> Source:
    if input_path.suffix.lower() == ".npz":
        return NPZSource(input_path)
    return SafeTensorSource(input_path, model_id=model_id)


def cleanup_cascade_stages(tensor_out: Path):
    for name in (
        "f0.int4", "f0.scales.f16", "f1.u.f16", "f1.v.f16",
        ".tmp_random_projection.f32"
    ):
        (tensor_out / name).unlink(missing_ok=True)


def purge_tensor_stage_files(tensor_out: Path) -> None:
    """Remove artefatos parciais de um tensor antes de reconvertê-lo (--resume)."""
    cleanup_cascade_stages(tensor_out)
    (tensor_out / "f0.raw").unlink(missing_ok=True)


# ------------------------------------------------------------------------------
# Disk budget / resume / package-by-package helpers
# ------------------------------------------------------------------------------

def estimate_tensor_output_peak(
    desc: TensorDesc,
    eligible: bool,
    group_size: int,
    ranks: Sequence[int],
    oversample: int,
) -> int:
    """
    Pior caso de bytes gravados no disco durante a conversão de um tensor:
    passthrough raw (nbytes) OU estágios CASCADE (F0 + projeção temporária + F1).
    """
    raw_bytes = int(desc.nbytes)
    if not eligible or desc.ndim != 2:
        return raw_bytes
    rows, cols = int(desc.shape[0]), int(desc.shape[1])
    groups = math.ceil(cols / group_size)
    f0_bytes = rows * math.ceil(cols / 2) + rows * groups * 2
    effective = [r for r in ranks if r <= min(rows, cols)]
    max_rank = max(effective) if effective else 0
    l = min(max_rank + oversample, rows, cols) if max_rank else 0
    tmp_bytes = rows * l * 4
    f1_bytes = (rows + cols) * max_rank * 2
    return max(raw_bytes, f0_bytes + tmp_bytes + f1_bytes)


def check_disk_budget(
    out: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
    budget_bytes: int,
    written_bytes: int,
    projected_bytes: int,
    tensor_name: str,
) -> None:
    """
    Aborta de forma limpa (estado retomável) se a conversão do próximo tensor
    derrubar o espaço livre abaixo da margem mínima ou estourar o orçamento
    --disk-budget-gb de saída.
    """
    free = shutil.disk_usage(str(out)).free
    over_free = projected_bytes + DISK_FREE_MARGIN_BYTES > free
    over_budget = budget_bytes > 0 and (written_bytes + projected_bytes) > budget_bytes
    if not (over_free or over_budget):
        return
    json_dump_atomic(manifest_path, manifest)
    print("-" * 78)
    print(f"[disk-guard] ABORTANDO antes do tensor '{tensor_name}'.")
    print(f"[disk-guard] pico projetado do tensor      : {projected_bytes:,} bytes")
    print(f"[disk-guard] disco livre atual             : {free:,} bytes")
    print(f"[disk-guard] margem mínima de disco livre  : {DISK_FREE_MARGIN_BYTES:,} bytes")
    print(f"[disk-guard] bytes já gravados na saída    : {written_bytes:,} bytes")
    if over_budget:
        print(f"[disk-guard] orçamento (--disk-budget-gb)  : {budget_bytes:,} bytes")
    print("[disk-guard] Estado retomável preservado em cascade_manifest.json.")
    print("[disk-guard] Libere espaço (use --delete-source-shards no Colab) ou ajuste")
    print("[disk-guard] --disk-budget-gb e re-execute o mesmo comando com --resume.")
    raise SystemExit(2)


def verify_tensor_outputs(tensor_dir: Path, record: Dict[str, Any]) -> bool:
    """
    Verifica se todos os arquivos de estágio registrados existem no disco e se a
    soma dos tamanhos bate com o campo 'bytes' de cada estágio.
    """
    stages = record.get("stages") or []
    if not stages:
        return False
    for stage in stages:
        files = stage.get("files") or {}
        if not files:
            return False
        total = 0
        for fname in files.values():
            fp = tensor_dir / str(fname)
            if not fp.is_file():
                return False
            total += fp.stat().st_size
        try:
            if int(stage.get("bytes", -1)) != total:
                return False
        except (TypeError, ValueError):
            return False
    return True


def build_resume_state(
    out: Path,
    previous_manifest: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Índice nome->{record, dir, verified} do manifesto anterior e o próximo
    tensor_id livre. Somente registros com artefatos íntegros são retomáveis.
    """
    index: Dict[str, Dict[str, Any]] = {}
    next_tensor_id = 0
    if not previous_manifest:
        return index, next_tensor_id
    for rec in previous_manifest.get("tensors", []):
        if not isinstance(rec, dict):
            continue
        name = rec.get("name")
        try:
            tid = int(rec.get("tensor_id"))
        except (TypeError, ValueError):
            continue
        next_tensor_id = max(next_tensor_id, tid + 1)
        if not isinstance(name, str) or not name:
            continue
        tdir = out / "tensors" / f"{tid:06d}_{slug(name)}"
        verified = (
            bool(rec.get("stages"))
            and isinstance(rec.get("output_bytes"), (int, float))
            and verify_tensor_outputs(tdir, rec)
        )
        index[name] = {"record": rec, "dir": tdir, "verified": verified}
    return index, next_tensor_id


def maybe_delete_source_shard(
    shard_path: Path,
    entries: List[Tuple[Path, Dict[str, Any]]],
) -> Optional[int]:
    """
    Apaga um shard .safetensors de origem depois que TODOS os seus tensores
    foram convertidos e verificados. Retorna os bytes liberados ou None.
    """
    if shard_path.suffix.lower() != ".safetensors":
        print(f"[delete-source-shards] fonte não é .safetensors — preservada: {shard_path.name}")
        return None
    if not shard_path.is_file():
        print(f"[delete-source-shards] shard já ausente: {shard_path.name}")
        return None
    for tensor_dir, record in entries:
        if not verify_tensor_outputs(tensor_dir, record):
            print(
                f"[delete-source-shards] verificação falhou para "
                f"'{record.get('name')}' — shard preservado: {shard_path.name}"
            )
            return None
    if not destructive_delete_allowed(shard_path):
        print(
            "[delete-source-shards] deleção bloqueada fora do Colab "
            "(defina RIFT_ALLOW_LOCAL_CLEANUP=1 para permitir localmente): "
            f"{shard_path.name}"
        )
        return None
    freed = shard_path.stat().st_size
    try:
        shard_path.unlink()
    except OSError as exc:
        print(f"[delete-source-shards] falha ao remover {shard_path.name}: {exc}")
        return None
    print(
        f"[delete-source-shards] shard removido: {shard_path.name} | "
        f"liberado: {freed:,} bytes"
    )
    return freed


# ------------------------------------------------------------------------------
# Schema v2 (dashboard) + publish
# ------------------------------------------------------------------------------

def build_comparison_identity(model_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    comparison_group_id = cmp-<sha256[:24]> de 'protocol|model_id|device|torch'
    (docs/C3_CONTRACTS_V1.md §3) + comparison_context.
    """
    device = "cpu"
    torch_version = package_version("torch")
    context = {
        "protocol": BENCHMARK_PROTOCOL,
        "model_id": model_id,
        "device": device,
        "torch": torch_version,
        "transformers": package_version("transformers"),
        "python": platform.python_version(),
        "numpy": getattr(np, "__version__", None),
    }
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|{device}|{torch_version or 'none'}"
    group_id = "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return group_id, context


def publish_battery(record: Dict[str, Any]) -> bool:
    """
    POST do registro em RIFT_RESULTS_ENDPOINT. Endurecido: HTTPS obrigatório e
    Bearer RIFT_INGEST_TOKEN com >=32 caracteres (fallback Colab userdata).
    """
    endpoint = (os.environ.get("RIFT_RESULTS_ENDPOINT") or DEFAULT_RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print(f"[publish] recusado: endpoint deve usar HTTPS: {endpoint}")
        return False
    token = (os.environ.get("RIFT_INGEST_TOKEN") or "").strip()
    if not token:
        try:
            from google.colab import userdata  # type: ignore
            token = str(userdata.get("RIFT_INGEST_TOKEN") or "").strip()
        except Exception:
            token = ""
    if len(token) < 32:
        print("[publish] recusado: RIFT_INGEST_TOKEN ausente ou com menos de 32 caracteres.")
        return False
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [record]}, ensure_ascii=False).encode("utf-8")
        req = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] {resp.status} {record.get('battery_id')}")
        return True
    except Exception as exc:
        print(f"[publish] falha: {exc}")
        return False


def convert_tensor(
    source: Source,
    desc: TensorDesc,
    tensor_out: Path,
    tensor_id: int,
    eligible: bool,
    reason: str,
    args,
    ranks: List[int],
    idx: int,
) -> Dict[str, Any]:
    """Converte um tensor e retorna o registro completo para o manifesto."""
    record: Dict[str, Any] = {
        "tensor_id": tensor_id,
        "name": desc.name,
        "shape": desc.shape,
        "source_dtype": desc.dtype,
        "source_bytes": desc.nbytes,
        "source_file": desc.source_file,
        "selection": reason,
        "stages": [],
        "local_quality": {},
        "gate": {
            "status": "NOT_APPLICABLE",
            "safe_policy": "F0_ONLY",
        },
    }

    if not eligible:
        stage = write_raw_stage(source, desc, tensor_out, reason)
        record["stages"] = [stage]
        record["output_bytes"] = stage["bytes"]
        return record

    # F0
    f0, f0_metrics = write_f0(
        source, desc, tensor_out, args.group_size, args.chunk_rows
    )
    record["local_quality"]["f0"] = f0_metrics

    if quality_pass(f0_metrics, args.cosine_min, args.nrmse_max):
        record["stages"] = [f0]
        record["output_bytes"] = f0["bytes"]
        record["gate"] = {
            "status": "F0_LOCAL_GATE_PASS",
            "safe_policy": "F0_ONLY",
        }
        record["local_quality"]["selected"] = f0_metrics
        record["local_quality"]["selected_local_pass"] = True
        return record

    # F1 low-rank
    effective_ranks = [r for r in ranks if r <= min(desc.shape)]
    if not effective_ranks:
        cleanup_cascade_stages(tensor_out)
        stage = write_raw_stage(source, desc, tensor_out, "rank_not_applicable_fallback")
        record["stages"] = [stage]
        record["output_bytes"] = stage["bytes"]
        record["local_quality"]["selected_local_pass"] = True
        record["local_quality"]["fallback_exact_raw"] = True
        return record

    u, v = randomized_residual_factors(
        source=source,
        desc=desc,
        out_dir=tensor_out,
        group_size=args.group_size,
        chunk_rows=args.chunk_rows,
        max_rank=max(effective_ranks),
        oversample=args.oversample,
        power_iters=args.power_iters,
        seed=args.seed + idx,
    )

    rank_metrics = evaluate_rank_candidates(
        source=source,
        desc=desc,
        out_dir=tensor_out,
        group_size=args.group_size,
        chunk_rows=args.chunk_rows,
        u=u,
        v=v,
        ranks=effective_ranks,
    )
    record["local_quality"]["rank_candidates"] = {
        str(k): val for k, val in rank_metrics.items()
    }

    chosen = None
    for r in effective_ranks:
        m = rank_metrics.get(r)
        if m and quality_pass(m, args.cosine_min, args.nrmse_max):
            chosen = r
            break

    if chosen is None:
        cleanup_cascade_stages(tensor_out)
        stage = write_raw_stage(
            source, desc, tensor_out,
            "cascade_local_quality_failed_fallback_exact_raw"
        )
        record["stages"] = [stage]
        record["output_bytes"] = stage["bytes"]
        record["local_quality"]["selected_local_pass"] = True
        record["local_quality"]["fallback_exact_raw"] = True
        record["gate"] = {
            "status": "NOT_APPLICABLE",
            "safe_policy": "F0_ONLY_RAW",
        }
        return record

    f1 = write_f1(tensor_out, u, v, chosen)
    record["stages"] = [f0, f1]
    record["output_bytes"] = f0["bytes"] + f1["bytes"]
    record["local_quality"]["selected"] = rank_metrics[chosen]
    record["local_quality"]["selected_local_pass"] = True
    record["gate"] = {
        "status": "CALIBRATION_REQUIRED",
        "safe_policy": "F1_ALWAYS",
        "stage_1_rank": chosen,
        "candidate_features": ["RMS(X)", "max_abs(X)", "variance(X)"],
    }
    return record


def convert(args) -> Dict[str, Any]:
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    resume_run = bool(getattr(args, "resume", False))
    previous_manifest: Optional[Dict[str, Any]] = None

    if out.exists():
        if resume_run:
            prev_path = out / "cascade_manifest.json"
            if prev_path.exists():
                try:
                    previous_manifest = json.loads(prev_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[resume] manifesto anterior ilegível ({exc}); reconvertendo do zero.")
        elif args.force:
            if not destructive_delete_allowed(out):
                raise SystemExit(
                    f"Saída já existe: {out}. --force removeria o diretório, mas a "
                    "limpeza destrutiva local é bloqueada fora do Colab. Remova "
                    "manualmente, use --resume, ou defina RIFT_ALLOW_LOCAL_CLEANUP=1."
                )
            shutil.rmtree(out)
        else:
            raise SystemExit(f"Saída já existe: {out}. Use --force ou --resume.")
    out.mkdir(parents=True, exist_ok=True)

    source = make_source(inp, args.model_id)

    delete_source_shards = bool(getattr(args, "delete_source_shards", False))
    if delete_source_shards:
        src_root = (inp if inp.is_dir() else inp.parent).resolve()
        if src_root == out:
            raise SystemExit(
                "--delete-source-shards recusado: o diretório de entrada é o mesmo da saída."
            )

    ranks = args.ranks
    budget_bytes = int(float(getattr(args, "disk_budget_gb", 0.0) or 0.0) * (1024 ** 3))

    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    copied_sidecars = source.copy_sidecars(out)
    descs = source.tensors()
    if delete_source_shards:
        # Modo pacote-por-pacote: agrupa os tensores por shard de origem para
        # liberar cada shard assim que todos os seus tensores terminarem.
        descs = sorted(descs, key=lambda d: (d.source_file, d.name))

    resume_index, next_tensor_id = build_resume_state(out, previous_manifest)

    manifest: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "created_utc": now_utc(),
        "model_id": source.model_id,
        "source": {
            "path": str(inp),
            "weight_tensor_bytes": int(source.source_weights_bytes()),
            "tensor_count": len(descs),
            "sidecars_copied": copied_sidecars,
        },
        "policy": {
            "f0": "INT4_GROUP_SYMMETRIC_TWOS_COMPLEMENT",
            "group_size": args.group_size,
            "f1": "LOWRANK_FP16",
            "rank_candidates": ranks,
            "oversample": args.oversample,
            "power_iters": args.power_iters,
            "quality_local_only": True,
            "cosine_min": args.cosine_min,
            "nrmse_max": args.nrmse_max,
            "fallback": "SOURCE_RAW",
            "disk_budget_gb": float(getattr(args, "disk_budget_gb", 0.0) or 0.0),
            "delete_source_shards": delete_source_shards,
            "resume": resume_run,
            "dynamic_gate": {
                "status": "CALIBRATION_REQUIRED",
                "default_safe_policy": "F1_ALWAYS_WHEN_PRESENT",
                "candidate_features": ["RMS(X)", "max_abs(X)", "variance(X)"],
            },
        },
        "tensors": [],
        "warnings": [
            "Local weight reconstruction metrics are not an end-to-end Quality Certificate.",
            "Dynamic Confidence Gate is not calibrated by this weight-only converter.",
            "CASCADE-DIR/0.1 is a development format, not the frozen production Bundle ABI.",
        ],
    }

    # Preserva o histórico de shards já apagados em execuções anteriores.
    prev_source = previous_manifest.get("source") if isinstance(previous_manifest, dict) else None
    if isinstance(prev_source, dict) and prev_source.get("deleted_shards"):
        manifest["source"]["deleted_shards"] = list(prev_source["deleted_shards"])

    start_time = time.time()
    total_original = 0
    total_stage_bytes = 0
    converted = 0
    passthrough = 0
    locally_accepted = 0
    resumed = 0

    manifest_path = out / "cascade_manifest.json"

    def account_record(rec: Dict[str, Any]) -> None:
        nonlocal total_stage_bytes, converted, passthrough, locally_accepted
        total_stage_bytes += int(rec.get("output_bytes", 0) or 0)
        stages = rec.get("stages") or []
        if stages and stages[0].get("stage_type") == "FULL_STAGE":
            passthrough += 1
        else:
            converted += 1
            locally_accepted += 1

    # Registros retomados cujo shard de origem já foi removido em execução
    # anterior (--delete-source-shards): são preservados no novo manifesto.
    current_names = {d.name for d in descs}
    carried_bytes = 0
    carried_count = 0
    for name, entry in resume_index.items():
        if name in current_names:
            continue
        if not entry["verified"]:
            print(f"[resume] '{name}': fonte ausente e saída incompleta — registro descartado.")
            continue
        rec = entry["record"]
        manifest["tensors"].append(rec)
        account_record(rec)
        carried_bytes += int(rec.get("source_bytes", 0) or 0)
        resumed += 1
        carried_count += 1
    total_original += carried_bytes
    if carried_count:
        print(f"[resume] {carried_count} tensores retomados de shards de origem já removidos.")
    manifest["source"]["tensor_count"] = len(descs) + carried_count
    manifest["source"]["weight_tensor_bytes"] = int(
        source.source_weights_bytes() + carried_bytes
    )

    # Rastreio por shard de origem (progresso + deleção pacote-por-pacote).
    shard_order: List[str] = []
    shard_total: Dict[str, int] = {}
    for d in descs:
        if d.source_file not in shard_total:
            shard_order.append(d.source_file)
            shard_total[d.source_file] = 0
        shard_total[d.source_file] += 1
    shard_pending = dict(shard_total)
    shard_entries: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {
        sf: [] for sf in shard_order
    }
    shards_done = 0

    print(f"CASCADE Model Converter {CONVERTER_VERSION}")
    print(f"Modelo: {source.model_id}")
    print(f"Tensores: {len(descs)}")
    print(f"Shards de origem: {len(shard_order)}")
    free0 = shutil.disk_usage(str(out)).free
    budget_txt = (
        f"{budget_bytes / (1024 ** 3):.1f} GB" if budget_bytes > 0 else "desativado"
    )
    print(f"Disco livre: {free0 / (1024 ** 3):.2f} GiB | --disk-budget-gb: {budget_txt}")
    print("-" * 78)

    for idx, desc in enumerate(descs, 1):
        total_original += desc.nbytes

        entry = resume_index.get(desc.name)
        reuse = (
            entry is not None
            and entry["verified"]
            and entry["record"].get("shape") == list(desc.shape)
            and entry["record"].get("source_dtype") == desc.dtype
        )

        if reuse:
            record = entry["record"]
            resumed += 1
            print(
                f"[{idx:>5}/{len(descs)}] {desc.name} {tuple(desc.shape)} "
                "[resume] completo — pulando",
                flush=True,
            )
        else:
            eligible, reason = eligible_matrix(
                desc,
                min_elements=args.min_elements,
                include_embeddings=args.include_embeddings,
                include_moe=args.include_moe,
                include_regex=include_re,
                exclude_regex=exclude_re,
            )

            projected = estimate_tensor_output_peak(
                desc, eligible, args.group_size, ranks, args.oversample
            )
            check_disk_budget(
                out, manifest, manifest_path, budget_bytes,
                total_stage_bytes, projected, desc.name,
            )

            tensor_id = next_tensor_id
            next_tensor_id += 1
            tensor_out = out / "tensors" / f"{tensor_id:06d}_{slug(desc.name)}"

            if entry is not None:
                purge_tensor_stage_files(entry["dir"])
            if resume_run:
                purge_tensor_stage_files(tensor_out)

            print(f"[{idx:>5}/{len(descs)}] {desc.name} {tuple(desc.shape)}", flush=True)
            record = convert_tensor(
                source, desc, tensor_out, tensor_id, eligible, reason,
                args, ranks, idx,
            )

        manifest["tensors"].append(record)
        account_record(record)

        # Manifesto incremental (tmp + rename atômico): estado retomável por tensor.
        json_dump_atomic(manifest_path, manifest)

        record_dir = out / "tensors" / f"{int(record['tensor_id']):06d}_{slug(desc.name)}"
        sf = desc.source_file
        shard_entries[sf].append((record_dir, record))
        shard_pending[sf] -= 1
        if shard_pending[sf] == 0:
            shards_done += 1
            free_now = shutil.disk_usage(str(out)).free
            print(
                f"[shard {shards_done}/{len(shard_order)}] concluído: {Path(sf).name} | "
                f"tensores: {shard_total[sf]} | "
                f"disco livre: {free_now / (1024 ** 3):.2f} GiB",
                flush=True,
            )
            if delete_source_shards:
                freed = maybe_delete_source_shard(Path(sf), shard_entries[sf])
                if freed is not None:
                    manifest["source"].setdefault("deleted_shards", []).append(
                        {"file": Path(sf).name, "freed_bytes": int(freed)}
                    )
                    json_dump_atomic(manifest_path, manifest)

    # Write preliminary manifest, then compute actual bundle directory size.
    elapsed = time.time() - start_time
    summary = {
        "source_weight_tensor_bytes": int(total_original),
        "cascade_stage_bytes": int(total_stage_bytes),
        "logical_compression_ratio_x": float(total_original / max(total_stage_bytes, 1)),
        "logical_disk_reduction_pct": float(
            (1.0 - total_stage_bytes / max(total_original, 1)) * 100.0
        ),
        "converted_tensor_count": converted,
        "passthrough_tensor_count": passthrough,
        "local_gate_accepted_tensor_count": locally_accepted,
        "resumed_tensor_count": resumed,
        "carried_over_tensor_count": carried_count,
        "conversion_seconds": elapsed,
    }
    manifest["summary"] = summary

    # Development CASCADE-IR inventory. Full operation DAG is intentionally not
    # fabricated from tensor names; architecture adapters must build it.
    ir = {
        "ir_version": "M0-development",
        "model_id": source.model_id,
        "architecture_hint": None,
        "graph_status": "ADAPTER_REQUIRED",
        "operations": [],
        "tensor_inventory": [
            {
                "tensor_id": t["tensor_id"],
                "name": t["name"],
                "shape": t["shape"],
                "source_dtype": t["source_dtype"],
                "cascade_stage_count": len(t["stages"]),
            }
            for t in manifest["tensors"]
        ],
        "warning": (
            "The converter does not infer a topological execution DAG from tensor "
            "names. A family adapter must populate operations[]."
        ),
    }

    gate_cfg = {
        "version": "0.1",
        "status": "CALIBRATION_REQUIRED",
        "safe_runtime_default": "F1_ALWAYS_WHEN_PRESENT",
        "features": ["rms_x", "max_abs_x", "variance_x"],
        "calibration_required_for_dynamic_skip": True,
    }

    json_dump(out / "cascade_ir.json", ir)
    json_dump(out / "gate_config.json", gate_cfg)
    json_dump_atomic(manifest_path, manifest)

    actual_bundle = dir_size(out)
    manifest["summary"]["cascade_bundle_directory_bytes"] = actual_bundle
    manifest["summary"]["bundle_vs_source_ratio_x"] = float(
        total_original / max(actual_bundle, 1)
    )
    manifest["summary"]["bundle_disk_reduction_pct"] = float(
        (1.0 - actual_bundle / max(total_original, 1)) * 100.0
    )
    json_dump_atomic(manifest_path, manifest)

    # Dashboard-compatible battery (schema v2 — docs/C3_CONTRACTS_V1.md §3).
    # Campos legados rift_* mantidos como aliases; *_ram_bytes de topo são null
    # porque este conversor não mede RSS: estimativas aritméticas ficam apenas
    # em metrics.memory.estimated_*.
    group_id, comparison_context = build_comparison_identity(source.model_id)
    battery = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": now_utc(),
        "run_id": "cascade-convert-" + uuid.uuid4().hex[:8],
        "spec": "CASCADE v0.3 / Converter v0.1",
        "technology": "CASCADE",
        "model_id": source.model_id,
        "battery_id": "CASCADE_MODEL_CONVERSION",
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "comparison_role": None,
        "comparison_group_id": group_id,
        "comparison_context": comparison_context,
        "implementation": {
            "kind": "REFERENCE_MEASURED",
            "native": False,
            "simulated": False,
            "eligible_for_primary_ranking": False,
        },
        "status": "LOCAL_WEIGHT_GATE_PASS",
        "baseline_tok_s": None,
        "candidate_tok_s": None,
        "rift_tok_s": None,
        "baseline_ram_bytes": None,
        "candidate_ram_bytes": None,
        "rift_ram_bytes": None,
        "baseline_disk_bytes": int(total_original),
        "candidate_disk_bytes": int(actual_bundle),
        "rift_disk_bytes": int(actual_bundle),
        "gains": {
            "tok_s_gain_pct": None,
            "ram_reduction_pct": float(
                (1.0 - total_stage_bytes / max(total_original, 1)) * 100.0
            ),
            "disk_reduction_pct": float(
                (1.0 - actual_bundle / max(total_original, 1)) * 100.0
            ),
            "disk_compression_ratio_x": float(
                total_original / max(actual_bundle, 1)
            ),
            "overall_gain_pct": None,
        },
        "metrics": {
            "memory": {
                "estimated_baseline_bytes": int(total_original),
                "estimated_candidate_bytes": int(total_stage_bytes),
                "method": None,
                "note": (
                    "Estimativas aritméticas de tamanho de representação; "
                    "nenhuma medição de RSS neste conversor estático."
                ),
            },
        },
        "measurement_scope": (
            "Weight representation conversion only. RAM is static representation "
            "size estimate, not measured runtime peak (top-level *_ram_bytes are "
            "null; estimates live in metrics.memory.estimated_*). Tok/s not "
            "measured."
        ),
        "quality": {
            "local_weight_metrics_only": True,
            "end_to_end_certified": False,
        },
        "notes": (
            "Dynamic Confidence Gate requires activation calibration. "
            "Safe runtime policy is F1_ALWAYS when F1 exists."
        ),
    }
    json_dump(out / "dashboard_battery.json", battery)

    if getattr(args, "publish", False):
        publish_battery(battery)

    print("-" * 78)
    print("CONVERSÃO CONCLUÍDA")
    print(f"Source tensor bytes : {total_original:,}")
    print(f"CASCADE stage bytes : {total_stage_bytes:,}")
    print(f"Bundle dir bytes    : {actual_bundle:,}")
    print(f"Disk reduction      : {manifest['summary']['bundle_disk_reduction_pct']:.2f}%")
    print(f"Tensores retomados  : {resumed}")
    print(f"Output              : {out}")
    print("Gate                : CALIBRATION_REQUIRED")
    print("End-to-end quality  : NOT YET CERTIFIED")

    return manifest


# ------------------------------------------------------------------------------
# Inspect / self-test
# ------------------------------------------------------------------------------

def inspect_bundle(path: Path):
    manifest = json.loads((path / "cascade_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "model_id": manifest["model_id"],
        "summary": manifest["summary"],
        "warnings": manifest["warnings"],
    }, indent=2, ensure_ascii=False))


def self_test(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    src = root / "synthetic_model.npz"
    out = root / "synthetic_model.cascade"

    rng = np.random.default_rng(12345)
    # Create a matrix that has a low-rank component + noise so F1 has useful work.
    u = rng.normal(size=(256, 12)).astype(np.float32)
    v = rng.normal(size=(384, 12)).astype(np.float32)
    w = (u @ v.T + 0.08 * rng.normal(size=(256, 384))).astype(np.float32)
    norm = rng.normal(size=(384,)).astype(np.float32)
    np.savez(src, **{
        "model.layers.0.self_attn.q_proj.weight": w,
        "model.layers.0.input_layernorm.weight": norm,
    })

    ns = argparse.Namespace(
        input=str(src),
        output=str(out),
        model_id="synthetic/cascade-selftest",
        group_size=64,
        ranks=[8, 16, 32],
        oversample=8,
        power_iters=1,
        seed=1234,
        chunk_rows=64,
        min_elements=4096,
        cosine_min=0.995,
        nrmse_max=0.05,
        include_embeddings=False,
        include_moe=False,
        include_regex=None,
        exclude_regex=None,
        force=True,
        disk_budget_gb=DEFAULT_DISK_BUDGET_GB,
        delete_source_shards=False,
        resume=False,
        publish=False,
    )

    # O self-test opta explicitamente pela limpeza local do seu próprio
    # diretório sintético (a guarda destrutiva bloqueia fora do Colab).
    prev_allow = os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP")
    os.environ["RIFT_ALLOW_LOCAL_CLEANUP"] = "1"
    try:
        manifest = convert(ns)
        if not (out / "cascade_manifest.json").exists():
            raise AssertionError("manifest não foi criado")
        if manifest["summary"]["cascade_stage_bytes"] >= manifest["summary"]["source_weight_tensor_bytes"]:
            raise AssertionError("self-test esperava redução de representação")

        # --- resume: simula execução interrompida removendo o último registro ---
        mpath = out / "cascade_manifest.json"
        data = json.loads(mpath.read_text(encoding="utf-8"))
        total_records = len(data["tensors"])
        if total_records < 2:
            raise AssertionError("self-test esperava ao menos 2 tensores")
        data["tensors"] = data["tensors"][:-1]
        data.pop("summary", None)
        json_dump_atomic(mpath, data)

        ns.resume = True
        ns.force = False
        manifest2 = convert(ns)
        if len(manifest2["tensors"]) != total_records:
            raise AssertionError("resume não reconstruiu o manifesto completo")
        if manifest2["summary"].get("resumed_tensor_count") != total_records - 1:
            raise AssertionError("resume deveria pular exatamente os tensores completos")
        names1 = sorted(t["name"] for t in manifest["tensors"])
        names2 = sorted(t["name"] for t in manifest2["tensors"])
        if names1 != names2:
            raise AssertionError("resume alterou o conjunto de tensores")
    finally:
        if prev_allow is None:
            os.environ.pop("RIFT_ALLOW_LOCAL_CLEANUP", None)
        else:
            os.environ["RIFT_ALLOW_LOCAL_CLEANUP"] = prev_allow
    print("[SELF-TEST] PASS (conversão + resume)")


def build_parser():
    p = argparse.ArgumentParser(
        description="CASCADE Model Converter v0.1 — Safetensors/NPZ -> CASCADE-DIR"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Converter um checkpoint local")
    c.add_argument("--input", required=True, help="Pasta com .safetensors, arquivo .safetensors ou .npz")
    c.add_argument("--output", required=True, help="Diretório .cascade de saída")
    c.add_argument("--model-id", default=None)
    c.add_argument("--group-size", type=int, default=64)
    c.add_argument("--ranks", type=parse_ranks, default=[8, 16, 32])
    c.add_argument("--oversample", type=int, default=8)
    c.add_argument("--power-iters", type=int, default=1)
    c.add_argument("--seed", type=int, default=1234)
    c.add_argument("--chunk-rows", type=int, default=128)
    c.add_argument("--min-elements", type=int, default=4096)
    c.add_argument("--cosine-min", type=float, default=DEFAULT_COSINE_MIN)
    c.add_argument("--nrmse-max", type=float, default=DEFAULT_NRMSE_MAX)
    c.add_argument("--include-embeddings", action="store_true")
    c.add_argument("--include-moe", action="store_true")
    c.add_argument("--include-regex", default=None)
    c.add_argument("--exclude-regex", default=None)
    c.add_argument("--force", action="store_true")
    c.add_argument(
        "--disk-budget-gb", type=float, default=DEFAULT_DISK_BUDGET_GB,
        help="Orçamento máximo de disco (GB) para a saída; aborta com estado "
             "retomável antes de exceder (<=0 desativa o orçamento; a margem "
             "mínima de disco livre continua valendo)",
    )
    c.add_argument(
        "--delete-source-shards", action="store_true",
        help="Modo pacote-por-pacote: apaga cada shard .safetensors de origem "
             "após TODOS os seus tensores serem convertidos e verificados "
             "(recusado se entrada == saída; fora do Colab exige "
             "RIFT_ALLOW_LOCAL_CLEANUP=1)",
    )
    c.add_argument(
        "--resume", action="store_true",
        help="Retoma uma conversão interrompida pulando tensores já completos "
             "e verificados no cascade_manifest.json",
    )
    c.add_argument(
        "--publish", action="store_true",
        help="Publica dashboard_battery.json em RIFT_RESULTS_ENDPOINT "
             "(HTTPS obrigatório + Bearer RIFT_INGEST_TOKEN >=32 chars)",
    )

    i = sub.add_parser("inspect", help="Inspecionar bundle CASCADE-DIR")
    i.add_argument("bundle")

    s = sub.add_parser("self-test", help="Executar teste sintético")
    s.add_argument("--out", default="cascade_converter_selftest")

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "convert":
        convert(args)
    elif args.cmd == "inspect":
        inspect_bundle(Path(args.bundle))
    elif args.cmd == "self-test":
        self_test(Path(args.out))


if __name__ == "__main__":
    main()
