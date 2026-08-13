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
import gc
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


# Margem do early-abort do F1: só desiste quando a energia capturada fica
# claramente abaixo do necessário (1.00 = exatamente no limite teórico). O
# vínculo cosseno usa 1-cos ~ nrmse^2/2, que é aproximação de 2ª ordem — a
# margem existe para essa aproximação nunca descartar um tensor resgatável.
F1_ENERGY_SAFETY = 0.75


def required_capture_fraction(
    m: Dict[str, float], cosine_min: float, nrmse_max: float
) -> float:
    """
    Fração MÍNIMA da energia do resíduo que o F1 precisa capturar para o tensor
    ter chance de passar o gate. Derivada do próprio gate, sem constante mágica:

      nrmse  : ||E'|| <= (nrmse_max/nrmse_f0)·||E||  ->  1 - (razão)^2   [exato]
      cosseno: (1-cos) escala com ||E||^2            ->  1 - (1-cos_min)/(1-cos_f0)

    O gate exige as duas condições, então vale o maior dos dois pisos. Zero
    significa "o F0 já satisfaz esta métrica": nesse caso o F1 não é obrigado a
    capturar nada por ela.
    """
    need = 0.0
    nrmse = float(m.get("nrmse") or 0.0)
    if nrmse_max > 0 and nrmse > nrmse_max:
        need = max(need, 1.0 - (nrmse_max / nrmse) ** 2)
    gap_f0 = 1.0 - float(m.get("cosine") or 0.0)
    gap_gate = 1.0 - float(cosine_min)
    if gap_f0 > 0 and gap_gate < gap_f0:
        need = max(need, 1.0 - max(gap_gate, 0.0) / gap_f0)
    return min(max(need, 0.0), 1.0)


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
# GGUF source (streaming por blocos de linhas)
# ------------------------------------------------------------------------------

def _require_gguf():
    try:
        import gguf  # type: ignore
    except ImportError:
        raise SystemExit(
            "Entrada .gguf exige o pacote 'gguf'. Instale com:\n"
            "    pip install 'gguf>=0.10,<1'"
        )
    return gguf


class GGUFSource(Source):
    """
    Lê checkpoints GGUF (llama.cpp) SEM materializar o tensor inteiro.

    A quantização do ggml é por blocos ao longo de ne0 (a dimensão contígua) e
    nenhum bloco cruza a fronteira de uma linha lógica, então é possível
    desquantizar apenas a faixa de linhas pedida por iter_rows(). O pico de RAM
    fica em chunk_rows x cols x 4 bytes, e não no tensor completo — é a
    diferença entre ~2 MB e vários GB numa embedding grande.

    Orientação: o ggml guarda shape como [ne0, ne1] com ne0 contíguo; a matriz
    lógica (linhas, colunas) é o shape invertido.
    """

    def __init__(self, path: Path, model_id: Optional[str] = None):
        self.gguf = _require_gguf()
        self.path = path
        self.model_id = model_id or path.stem
        self._reader = self.gguf.GGUFReader(str(path))
        self._by_name: Dict[str, Any] = {}
        self._descs: List[TensorDesc] = []
        for t in self._reader.tensors:
            shape = [int(x) for x in t.shape]
            if len(shape) == 2:
                logical = [shape[1], shape[0]]  # (rows, cols) = shape invertido
            else:
                logical = shape
            self._by_name[t.name] = t
            self._descs.append(
                TensorDesc(
                    name=t.name,
                    shape=logical,
                    dtype=f"GGUF_{t.tensor_type.name}",
                    nbytes=int(np.asarray(t.data).nbytes),
                    source_file=str(path),
                    source_kind="gguf",
                )
            )

    def tensors(self) -> List[TensorDesc]:
        return self._descs

    def metadata(self) -> Dict[str, Any]:
        """Campos escalares do KV do GGUF (arrays grandes, como o tokenizer,
        ficam de fora para o sidecar não explodir)."""
        out: Dict[str, Any] = {}
        for key, field in self._reader.fields.items():
            name = key if isinstance(key, str) else str(key)
            try:
                parts = field.parts[-1]
                if len(parts) == 1:
                    value = parts[0]
                    out[name] = int(value) if float(value).is_integer() else float(value)
            except Exception:
                continue
        return out

    def _raw_rows(self, desc: TensorDesc):
        """Vista (rows, bytes_por_linha) do payload bruto do tensor."""
        t = self._by_name[desc.name]
        rows = int(desc.shape[0])
        raw = np.asarray(t.data)
        flat = raw.reshape(-1)
        if flat.nbytes % rows:
            raise ValueError(
                f"payload de {desc.name} não divide por {rows} linhas "
                f"({flat.nbytes} bytes) — streaming por linha indisponível"
            )
        per_row = flat.size // rows
        return t, flat.reshape(rows, per_row)

    def iter_rows(self, desc: TensorDesc, chunk_rows: int):
        if desc.ndim != 2:
            raise ValueError("iter_rows suporta somente tensor 2D")
        t, view = self._raw_rows(desc)
        rows, cols = int(desc.shape[0]), int(desc.shape[1])
        qtype = t.tensor_type
        gg = self.gguf
        simple = {
            gg.GGMLQuantizationType.F32: np.float32,
            gg.GGMLQuantizationType.F16: np.float16,
        }
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            block = view[start:end]
            if qtype in simple:
                w = np.asarray(block, dtype=np.float32).reshape(end - start, cols)
            elif qtype == gg.GGMLQuantizationType.BF16:
                u = np.ascontiguousarray(block).view(np.uint16).astype(np.uint32)
                w = (u << 16).view(np.float32).reshape(end - start, cols)
            else:
                deq = gg.dequantize(np.ascontiguousarray(block), qtype)
                w = np.asarray(deq, dtype=np.float32).reshape(end - start, cols)
            yield start, w
            del block, w

    def copy_raw_tensor(self, desc: TensorDesc, dst: Path) -> None:
        """
        Passthrough exato: copia os bytes do bloco GGUF como estão. O
        `source_dtype` do estágio (GGUF_<QTYPE>) diz ao leitor como decodificar.

        A cópia é EM BLOCOS a partir do memmap do reader: materializar o tensor
        inteiro (`.tobytes()`) estourava a RAM em token_embd/output.weight de
        modelos grandes (~1 GB no arquivo, 2x em pico).
        """
        t = self._by_name[desc.name]
        dst.parent.mkdir(parents=True, exist_ok=True)
        flat = np.asarray(t.data).reshape(-1).view(np.uint8)
        step = max(1, (8 * 1024 * 1024) // max(flat.itemsize, 1))
        with dst.open("wb") as f:
            for start in range(0, flat.size, step):
                f.write(flat[start:start + step].tobytes())

    def copy_sidecars(self, output_dir: Path) -> List[str]:
        meta = self.metadata()
        if not meta:
            return []
        target = output_dir / "gguf_metadata.json"
        json_dump_atomic(target, {"source_gguf": str(self.path), "kv": meta})
        return [target.name]


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

def quantize_int2_rows(
    x: np.ndarray, group_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    INT2 groupwise ASSIMÉTRICO (min-max, 4 níveis) — mesmo esquema do ZDC do
    GEYSER: w ~= q*scale + wmin, q em {0,1,2,3}, escala e mínimo FP16 por grupo
    (2 + 32/group bits por peso). Assimétrico porque com só 4 níveis um
    quantizador simétrico desperdiça faixa em grupos não centrados em zero.

    Retorna packed uint8 (4 códigos/byte), scales FP16, mins FP16, recon FP32.
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
    wmin32 = g.min(axis=2)
    wmax32 = g.max(axis=2)
    scale32 = (wmax32 - wmin32) / 3.0
    scale32[scale32 <= 0] = 1.0

    # Quantiza contra os valores EXATOS que serão gravados (FP16).
    scales16 = scale32.astype(np.float16)
    mins16 = wmin32.astype(np.float16)
    scales = scales16.astype(np.float32)
    mins = mins16.astype(np.float32)
    scales[scales == 0] = 1.0

    q = np.rint((g - mins[:, :, None]) / scales[:, :, None]).clip(0, 3).astype(np.uint8)
    qflat = q.reshape(rows, padded_cols)

    packed_cols = math.ceil(cols / 4)
    pad4 = packed_cols * 4 - cols
    codes = qflat[:, :cols]
    if pad4:
        codes = np.pad(codes, ((0, 0), (0, pad4)))
    c = codes.reshape(rows, packed_cols, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)

    recon_groups = q.astype(np.float32) * scales[:, :, None] + mins[:, :, None]
    recon = recon_groups.reshape(rows, padded_cols)[:, :cols]
    return packed, scales16, mins16, recon


def dequant_int2_rows(
    packed: np.ndarray,
    scales16: np.ndarray,
    mins16: np.ndarray,
    cols: int,
    group_size: int,
) -> np.ndarray:
    rows = packed.shape[0]
    wide = packed.shape[1] * 4
    codes = np.empty((rows, wide), dtype=np.uint8)
    codes[:, 0::4] = packed & 0x03
    codes[:, 1::4] = (packed >> 2) & 0x03
    codes[:, 2::4] = (packed >> 4) & 0x03
    codes[:, 3::4] = (packed >> 6) & 0x03
    codes = codes[:, :cols]

    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols != cols:
        codes = np.pad(codes, ((0, 0), (0, padded_cols - cols)))
    qg = codes.reshape(rows, groups, group_size).astype(np.float32)
    scales = scales16.astype(np.float32)
    mins = mins16.astype(np.float32)
    recon = qg * scales[:, :, None] + mins[:, :, None]
    return recon.reshape(rows, padded_cols)[:, :cols]


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


@dataclass(frozen=True)
class F0Codec:
    """Codec do estágio base. `overhead_bits` são os bits de metadado por grupo
    (escala FP16, mais o mínimo FP16 quando o codec é assimétrico)."""
    name: str
    bits: int
    representation: str
    packed_file: str
    has_mins: bool

    @property
    def overhead_bits(self) -> int:
        return 32 if self.has_mins else 16

    def bpw(self, group_size: int) -> float:
        return self.bits + self.overhead_bits / float(group_size)

    def packed_row_bytes(self, cols: int) -> int:
        per_byte = 8 // self.bits
        return math.ceil(cols / per_byte)


F0_CODECS: Dict[str, F0Codec] = {
    "int4": F0Codec(
        "int4", 4, "INT4_GROUP_SYMMETRIC_TWOS_COMPLEMENT", "f0.int4", False
    ),
    "int2": F0Codec(
        "int2", 2, "INT2_GROUP_ASYMMETRIC_MINMAX", "f0.int2", True
    ),
}


def write_f0(
    source: Source,
    desc: TensorDesc,
    out_dir: Path,
    group_size: int,
    chunk_rows: int,
    codec: F0Codec = F0_CODECS["int4"],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    rows, cols = desc.shape
    groups = math.ceil(cols / group_size)

    packed_path = out_dir / codec.packed_file
    scales_path = out_dir / "f0.scales.f16"
    mins_path = out_dir / "f0.mins.f16"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = Metrics()
    handles = [packed_path.open("wb"), scales_path.open("wb")]
    if codec.has_mins:
        handles.append(mins_path.open("wb"))
    try:
        fp, fs = handles[0], handles[1]
        fm = handles[2] if codec.has_mins else None
        for _, w in source.iter_rows(desc, chunk_rows):
            if codec.name == "int2":
                packed, scales16, mins16, recon = quantize_int2_rows(w, group_size)
                fm.write(mins16.tobytes(order="C"))
            else:
                packed, scales16, recon = quantize_int4_rows(w, group_size)
            fp.write(packed.tobytes(order="C"))
            fs.write(scales16.tobytes(order="C"))
            metrics.update(w, recon)
            del packed, scales16, recon, w
    finally:
        for h in handles:
            h.close()

    files = {"packed": packed_path.name, "scales": scales_path.name}
    digests = {"packed": sha256_file(packed_path), "scales": sha256_file(scales_path)}
    total = packed_path.stat().st_size + scales_path.stat().st_size
    if codec.has_mins:
        files["mins"] = mins_path.name
        digests["mins"] = sha256_file(mins_path)
        total += mins_path.stat().st_size

    meta = {
        "stage_index": 0,
        "stage_type": "BASE_STAGE",
        "representation": codec.representation,
        "codec": codec.name,
        "bits": codec.bits,
        "group_size": group_size,
        "effective_bits_per_weight": round(codec.bpw(group_size), 4),
        "shape": [rows, cols],
        "packed_row_bytes": codec.packed_row_bytes(cols),
        "scale_groups_per_row": groups,
        "scale_dtype": "FP16",
        "resident_hint": "HOT",
        "files": files,
        "bytes": total,
        "sha256": digests,
    }
    return meta, metrics.result()


def open_f0_memmaps(
    out_dir: Path,
    rows: int,
    cols: int,
    group_size: int,
    codec: F0Codec = F0_CODECS["int4"],
) -> Dict[str, np.ndarray]:
    groups = math.ceil(cols / group_size)
    maps = {
        "packed": np.memmap(
            out_dir / codec.packed_file, dtype=np.uint8, mode="r",
            shape=(rows, codec.packed_row_bytes(cols)),
        ),
        "scales": np.memmap(
            out_dir / "f0.scales.f16", dtype=np.float16, mode="r",
            shape=(rows, groups),
        ),
    }
    if codec.has_mins:
        maps["mins"] = np.memmap(
            out_dir / "f0.mins.f16", dtype=np.float16, mode="r",
            shape=(rows, groups),
        )
    return maps


def dequant_f0_chunk(
    maps: Dict[str, np.ndarray],
    start: int,
    end: int,
    cols: int,
    group_size: int,
    codec: F0Codec,
) -> np.ndarray:
    if codec.name == "int2":
        return dequant_int2_rows(
            maps["packed"][start:end], maps["scales"][start:end],
            maps["mins"][start:end], cols, group_size,
        )
    return dequant_int4_rows(
        maps["packed"][start:end], maps["scales"][start:end], cols, group_size
    )


# ------------------------------------------------------------------------------
# Streaming randomized low-rank residual
# ------------------------------------------------------------------------------

def residual_chunk(
    source_chunk: np.ndarray,
    maps: Dict[str, np.ndarray],
    start: int,
    end: int,
    cols: int,
    group_size: int,
    codec: F0Codec,
) -> np.ndarray:
    f0 = dequant_f0_chunk(maps, start, end, cols, group_size, codec)
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
    codec: F0Codec = F0_CODECS["int4"],
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = desc.shape
    l = min(max_rank + oversample, rows, cols)
    if l <= 0:
        raise ValueError("rank inválido")

    maps = open_f0_memmaps(out_dir, rows, cols, group_size, codec)
    residual_energy = 0.0  # ||R||_F^2 acumulado no 1º passe (custo zero)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, l), dtype=np.float32) / math.sqrt(max(cols, 1))

    y_path = out_dir / ".tmp_random_projection.f32"
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(rows, l))

    # Y = R @ Omega (e a energia total do resíduo, de graça no mesmo passe)
    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        r = residual_chunk(w, maps, start, end, cols, group_size, codec)
        residual_energy += float(np.dot(r.reshape(-1), r.reshape(-1)))
        y[start:end] = r @ omega
    y.flush()

    q, _ = np.linalg.qr(np.asarray(y), mode="reduced")
    q = np.asarray(q, dtype=np.float32)

    # Optional power iterations: Q <- orth(R @ (R.T @ Q))
    for _ in range(power_iters):
        z = np.zeros((cols, q.shape[1]), dtype=np.float32)
        for start, w in source.iter_rows(desc, chunk_rows):
            end = start + w.shape[0]
            r = residual_chunk(w, maps, start, end, cols, group_size, codec)
            z += r.T @ q[start:end]

        for start, w in source.iter_rows(desc, chunk_rows):
            end = start + w.shape[0]
            r = residual_chunk(w, maps, start, end, cols, group_size, codec)
            y[start:end] = r @ z
        y.flush()
        q, _ = np.linalg.qr(np.asarray(y), mode="reduced")
        q = np.asarray(q, dtype=np.float32)

    # B = Q.T @ R
    b = np.zeros((q.shape[1], cols), dtype=np.float32)
    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        r = residual_chunk(w, maps, start, end, cols, group_size, codec)
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

    # Fração da energia do resíduo capturada pelos rmax primeiros valores
    # singulares. Perto de zero => o resíduo NÃO é low-rank e nenhum rank
    # disponível vai fechar o gate (medido no Glimmer: rank 8->32 não moveu o
    # cosine de attn_output). Serve de early-abort do F1.
    captured = float(np.dot(s[:rmax], s[:rmax])) if rmax else 0.0
    info = {
        "residual_energy": residual_energy,
        "captured_energy": captured,
        "captured_fraction": (
            float(captured / residual_energy) if residual_energy > 0 else 0.0
        ),
        "max_rank": int(rmax),
        "top_singular_values": [float(x) for x in s[: min(4, len(s))]],
    }
    return np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32), info


def evaluate_rank_candidates(
    source: Source,
    desc: TensorDesc,
    out_dir: Path,
    group_size: int,
    chunk_rows: int,
    u: np.ndarray,
    v: np.ndarray,
    ranks: Sequence[int],
    codec: F0Codec = F0_CODECS["int4"],
) -> Dict[int, Dict[str, float]]:
    rows, cols = desc.shape
    maps = open_f0_memmaps(out_dir, rows, cols, group_size, codec)

    # Evaluate what will actually be stored: FP16 factors.
    uq = u.astype(np.float16).astype(np.float32)
    vq = v.astype(np.float16).astype(np.float32)

    acc = {r: Metrics() for r in ranks if r <= uq.shape[1]}
    ranks2 = sorted(acc)
    if not ranks2:
        return {}

    for start, w in source.iter_rows(desc, chunk_rows):
        end = start + w.shape[0]
        pred = dequant_f0_chunk(maps, start, end, cols, group_size, codec)
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

def write_external_stage(desc: TensorDesc, reason: str) -> Dict[str, Any]:
    """
    Estágio SEM cópia: o tensor continua vivo APENAS no checkpoint de origem.
    Economiza disco e RAM de cópia (token_embd/output.weight de modelo grande
    estouravam a memória), mas o bundle passa a DEPENDER do arquivo de origem —
    registrado em `requires_source_file` e no resumo do manifesto.

    Atenção: economiza DISCO, não RAM de execução. `external_bytes` entra no
    residente (HOT) do relatório de residência, porque o peso continua sendo
    necessário para rodar o modelo.
    """
    return {
        "stage_index": 0,
        "stage_type": "FULL_STAGE",
        "representation": "SOURCE_EXTERNAL",
        "source_dtype": desc.dtype,
        "shape": desc.shape,
        "resident_hint": "HOT",
        "files": {},
        "bytes": 0,
        "external_bytes": int(desc.nbytes),
        "requires_source_file": True,
        "source_file": desc.source_file,
        "reason": reason,
    }


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

# Escadas de codec (cheapest-first). Cada degrau é (codec, group_size); o
# primeiro que passa o gate de qualidade é o escolhido, e só se TODOS falharem
# o tensor cai em passthrough exato (16 bpw).
LADDERS: Dict[str, List[Tuple[str, int]]] = {
    # Sem risco de regressão: mesmo primeiro degrau de sempre (int4/g64) e um
    # degrau de resgate mais fino antes do raw — troca 16 bpw por 4.5 bpw nos
    # tensores que hoje caem em raw.
    "safe": [("int4", 64), ("int4", 32)],
    # Máxima compressão para o alvo de 8 GB: tenta 2 bits primeiro (2.5 bpw) e
    # só sobe quando o gate reprova.
    "compact": [("int2", 64), ("int4", 64), ("int4", 32)],
    # Compatibilidade estrita com o comportamento anterior a esta versão.
    "int4": [("int4", 64)],
}


# Fonte já comprimida em baixa precisão: qualquer quant do ggml que não seja
# F32/F16/BF16. Nesses checkpoints o INT2 medido fica em cosine ~0.91-0.92 e
# NUNCA passa o gate — tentar é tempo morto.
LOW_BIT_SOURCE_RE = re.compile(r"^GGUF_(?!F32$|F16$|BF16$)", re.IGNORECASE)


def source_is_low_bit(desc: Optional[TensorDesc]) -> bool:
    return bool(desc and LOW_BIT_SOURCE_RE.match(str(desc.dtype or "")))


def resolve_ladder_mode(args, desc: Optional[TensorDesc] = None) -> str:
    """
    'auto' escolhe a escada pelo tipo da FONTE:
      fonte já low-bit (IQ2/Q4_K/...) -> 'safe'    (não tenta int2)
      fonte BF16/F16/F32              -> 'compact' (int2 vale a tentativa,
                                                    porque o raw custa 16 bpw)
    """
    mode = getattr(args, "codec_ladder", "auto")
    if mode != "auto":
        return mode
    return "safe" if source_is_low_bit(desc) else "compact"


def ladder_rungs(args, desc: Optional[TensorDesc] = None) -> List[Tuple[str, int]]:
    """Degraus da escada, com o --group-size do usuário respeitado no 1º degrau."""
    mode = resolve_ladder_mode(args, desc)
    rungs = [tuple(r) for r in LADDERS[mode]]
    user_group = int(getattr(args, "group_size", 64))
    first_codec = rungs[0][0]
    rungs[0] = (first_codec, user_group)
    out: List[Tuple[str, int]] = []
    for rung in rungs:
        if rung not in out:
            out.append(rung)  # type: ignore[arg-type]
    return out


def projected_f0_bytes(desc: TensorDesc, codec: "F0Codec", group_size: int) -> int:
    """Bytes do estágio base antes de escrevê-lo (packed + escalas + mínimos)."""
    rows, cols = int(desc.shape[0]), int(desc.shape[1])
    groups = math.ceil(cols / group_size)
    meta_per_group = 4 if codec.has_mins else 2
    return rows * codec.packed_row_bytes(cols) + rows * groups * meta_per_group


def adaptive_chunk_rows(desc: TensorDesc, chunk_rows: int, ram_budget_mb: float) -> int:
    """
    Limita a fatia de linhas para o pico de RAM por chunk ficar dentro do
    orçamento. O pico por fatia é ~ linhas x colunas x 4 bytes (float32) e o
    pipeline mantém ~3 fatias vivas (fonte, reconstrução, resíduo).
    """
    if desc.ndim != 2 or ram_budget_mb <= 0:
        return chunk_rows
    cols = max(int(desc.shape[1]), 1)
    per_row = cols * 4 * 3
    budget = int(ram_budget_mb * 1024 * 1024)
    allowed = max(budget // max(per_row, 1), 8)
    return int(min(chunk_rows, allowed))


# Reserva para KV-cache, ativações e runtime no veredito de residência.
RUNTIME_RESERVE_BYTES = int(1.5 * 1024 ** 3)

# Reserva para SO e aplicativos do usuário ao derivar o orçamento da máquina.
OS_RESERVE_BYTES = 8 * 1024 ** 3

# Larguras de bits do relatório de RAM (estilo cartão de modelo do Hugging
# Face). fp16 entra como referência do peso original.
REPORT_BIT_WIDTHS = (1, 2, 3, 4, 5, 6, 8, 16)


def detect_total_ram_bytes() -> Optional[int]:
    """RAM física total, sem dependências novas. None se indetectável."""
    try:  # Linux / Colab
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:  # POSIX genérico (macOS incluído)
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page > 0:
            return int(pages) * int(page)
    except Exception:
        pass
    if sys.platform == "win32":  # Windows: GlobalMemoryStatusEx
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        except Exception:
            pass
    return None


def auto_target_ram_bytes(total_ram_bytes: Optional[int]) -> Tuple[int, str]:
    """
    Orçamento do MODELO derivado da máquina: total menos 8 GiB para SO e
    aplicativos, com piso de 50% do total para não zerar em máquinas pequenas.

        16 GiB -> 8 GiB    24 GiB -> 16 GiB    32 GiB -> 24 GiB
         8 GiB -> 4 GiB (piso de 50%)

    Sem detecção, cai no alvo canônico do projeto (8 GiB livres).
    """
    if not total_ram_bytes or total_ram_bytes <= 0:
        return 8 * 1024 ** 3, "default_conventional_8gb"
    budget = max(total_ram_bytes - OS_RESERVE_BYTES, total_ram_bytes // 2)
    return int(budget), "auto_total_minus_os_reserve"


def memory_by_bits(
    tensors: Sequence[Dict[str, Any]], target_bytes: int
) -> List[Dict[str, Any]]:
    """
    Tabela "RAM por largura de bits" no estilo dos cartões de modelo do Hugging
    Face. PROJEÇÃO: recalcula só o estágio base dos tensores convertidos com
    cada largura hipotética (mantendo o overhead de escala por grupo) e soma o
    passthrough REAL, que não é quantizado.
    """
    quantized_elements = 0
    quantized_groups = 0
    passthrough_bytes = 0
    for rec in tensors:
        for stage in rec.get("stages") or []:
            if stage.get("representation") == "SOURCE_RAW":
                passthrough_bytes += int(stage.get("bytes") or 0)
                continue
            if int(stage.get("stage_index", 0)) != 0:
                continue
            shape = stage.get("shape") or []
            if len(shape) != 2:
                continue
            rows, cols = int(shape[0]), int(shape[1])
            quantized_elements += rows * cols
            groups = int(stage.get("scale_groups_per_row") or 0)
            quantized_groups += rows * groups

    out: List[Dict[str, Any]] = []
    for bits in REPORT_BIT_WIDTHS:
        # fp16 é a referência densa: sem grupos, sem overhead de escala.
        overhead = 0 if bits >= 16 else quantized_groups * 2
        total = quantized_elements * bits // 8 + overhead + passthrough_bytes
        out.append({
            "bits": bits,
            "label": "fp16" if bits == 16 else f"{bits}-bit",
            "resident_bytes": int(total),
            "resident_gib": round(total / 1024 ** 3, 3),
            "fits_in_target": bool(
                target_bytes and total <= max(target_bytes - RUNTIME_RESERVE_BYTES, 0)
            ),
        })
    return out


def residency_report(
    tensors: Sequence[Dict[str, Any]],
    target_ram_gb: float,
    total_ram_bytes: Optional[int] = None,
    target_source: str = "explicit_flag",
) -> Dict[str, Any]:
    """
    Onde os bytes vão parar em execução e se o modelo cabe na máquina alvo.
    F0/raw são residentes (HOT); F1 é refinamento paginável (WARM).
    """
    hot = warm = raw = external = 0
    f0_bits = 0
    f0_elements = 0
    rungs: Dict[str, int] = {}
    for rec in tensors:
        rung = str((rec.get("ladder") or {}).get("selected_rung") or "n/a")
        rungs[rung] = rungs.get(rung, 0) + 1
        for stage in rec.get("stages") or []:
            nbytes = int(stage.get("bytes") or 0)
            if stage.get("representation") == "SOURCE_EXTERNAL":
                # Não ocupa disco no bundle, mas o peso continua tendo de estar
                # em RAM para rodar: conta como residente (HOT).
                ext = int(stage.get("external_bytes") or 0)
                external += ext
                raw += ext
                hot += ext
            elif stage.get("representation") == "SOURCE_RAW":
                raw += nbytes
                hot += nbytes
            elif int(stage.get("stage_index", 0)) == 0:
                hot += nbytes
                shape = stage.get("shape") or []
                if len(shape) == 2:
                    elems = int(shape[0]) * int(shape[1])
                    f0_elements += elems
                    f0_bits += nbytes * 8
            else:
                warm += nbytes

    target_bytes = int(max(target_ram_gb, 0.0) * 1024 ** 3)
    usable = max(target_bytes - RUNTIME_RESERVE_BYTES, 0)
    return {
        "target_ram_gb": round(float(target_ram_gb), 3),
        "target_source": target_source,
        "machine_total_ram_bytes": int(total_ram_bytes) if total_ram_bytes else None,
        "os_reserve_bytes": OS_RESERVE_BYTES if total_ram_bytes else None,
        "runtime_reserve_bytes": RUNTIME_RESERVE_BYTES,
        "memory_by_bits": memory_by_bits(tensors, target_bytes),
        "resident_hot_bytes": int(hot),
        "pageable_warm_bytes": int(warm),
        "raw_passthrough_bytes": int(raw),
        "external_source_bytes": int(external),
        "bundle_requires_source": bool(external > 0),
        "all_in_ram_bytes": int(hot + warm),
        "f0_effective_bits_per_weight": (
            round(f0_bits / f0_elements, 4) if f0_elements else None
        ),
        "fits_resident_in_target": bool(target_bytes and hot <= usable),
        "fits_all_in_ram_in_target": bool(target_bytes and (hot + warm) <= usable),
        "selected_rungs": rungs,
        "note": (
            "HOT = F0 + passthrough exato (precisa estar residente); WARM = F1 "
            "(residual paginável). Veredito desconta "
            f"{RUNTIME_RESERVE_BYTES / 1024 ** 3:.1f} GiB de reserva para "
            "KV-cache, ativações e runtime."
        ),
    }


def read_vmrss_bytes() -> Optional[int]:
    """Pico/RSS atual do processo (Linux/Colab). None onde /proc não existe."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def make_source(input_path: Path, model_id: Optional[str]) -> Source:
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        return NPZSource(input_path)
    if suffix == ".gguf":
        return GGUFSource(input_path, model_id=model_id)
    if input_path.is_dir():
        ggufs = sorted(input_path.glob("*.gguf"))
        safes = sorted(input_path.glob("*.safetensors"))
        if ggufs and not safes:
            if len(ggufs) > 1:
                raise SystemExit(
                    f"{len(ggufs)} arquivos .gguf em {input_path}; aponte --input "
                    "para o arquivo desejado (GGUF multi-parte não é suportado)"
                )
            return GGUFSource(ggufs[0], model_id=model_id)
    return SafeTensorSource(input_path, model_id=model_id)


def cleanup_cascade_stages(tensor_out: Path):
    for name in (
        "f0.int4", "f0.int2", "f0.scales.f16", "f0.mins.f16",
        "f1.u.f16", "f1.v.f16",
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
    keep_source: bool = False,
) -> int:
    """
    Pior caso de bytes gravados no disco durante a conversão de um tensor:
    passthrough raw (nbytes) OU estágios CASCADE (F0 + projeção temporária + F1).
    """
    raw_bytes = int(desc.nbytes)
    if not eligible or desc.ndim != 2:
        # Passthrough sem cópia não consome disco na saída.
        return 0 if keep_source else raw_bytes
    rows, cols = int(desc.shape[0]), int(desc.shape[1])
    effective = [r for r in ranks if r <= min(rows, cols)]
    max_rank = max(effective) if effective else 0
    l = min(max_rank + oversample, rows, cols) if max_rank else 0
    tmp_bytes = rows * l * 4
    f1_bytes = (rows + cols) * max_rank * 2
    # Pior caso entre os degraus da escada (um degrau por vez em disco: o
    # anterior é limpo antes do próximo).
    worst_f0 = 0
    for codec_name, gsize in ladder_rungs(args_like(group_size)):
        codec = F0_CODECS[codec_name]
        groups = math.ceil(cols / gsize)
        meta_per_group = 4 if codec.has_mins else 2
        worst_f0 = max(
            worst_f0,
            rows * codec.packed_row_bytes(cols) + rows * groups * meta_per_group,
        )
    return max(raw_bytes, worst_f0 + tmp_bytes + f1_bytes)


class args_like:
    """Adaptador mínimo para reaproveitar ladder_rungs na projeção de disco."""

    def __init__(self, group_size: int, mode: str = "compact"):
        self.group_size = group_size
        self.codec_ladder = mode


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
        # SOURCE_EXTERNAL não tem arquivo no bundle por design: verifica-se que
        # o checkpoint de origem ainda existe.
        if stage.get("representation") == "SOURCE_EXTERNAL":
            src = stage.get("source_file")
            if not src or not Path(str(src)).is_file():
                return False
            continue
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
        if getattr(args, "keep_source_passthrough", False):
            stage = write_external_stage(desc, reason)
        else:
            stage = write_raw_stage(source, desc, tensor_out, reason)
        record["stages"] = [stage]
        record["output_bytes"] = stage["bytes"]
        return record

    chunk_rows = adaptive_chunk_rows(
        desc, int(args.chunk_rows), float(getattr(args, "ram_budget_mb", 0.0) or 0.0)
    )
    if chunk_rows != int(args.chunk_rows):
        record["chunk_rows_effective"] = chunk_rows

    rungs = ladder_rungs(args, desc)
    source_bytes = int(desc.nbytes)
    guard_expansion = not getattr(args, "allow_byte_expansion", False)
    record["ladder"] = {
        "mode": resolve_ladder_mode(args, desc),
        "requested_mode": getattr(args, "codec_ladder", "auto"),
        "source_low_bit": source_is_low_bit(desc),
        "rungs": [f"{c}/g{g}" for c, g in rungs],
        "source_bytes": source_bytes,
        "attempts": [],
    }
    effective_ranks = [r for r in ranks if r <= min(desc.shape)]

    for rung_idx, (codec_name, group_size) in enumerate(rungs):
        codec = F0_CODECS[codec_name]
        last_rung = rung_idx == len(rungs) - 1
        attempt: Dict[str, Any] = {
            "rung": f"{codec_name}/g{group_size}",
            "bpw_f0": round(codec.bpw(group_size), 4),
        }

        # Guarda de expansão: se este degrau já ficaria >= a fonte, ele não
        # interessa nem passando o gate — o passthrough exato é menor E sem
        # perda. Como a escada é crescente em bpw, os degraus seguintes também
        # expandem: vai direto para o raw. (Caso real: fonte GGUF IQ2 ~2.66 bpw
        # contra INT4/g64 4.25 bpw.)
        projected = projected_f0_bytes(desc, codec, group_size)
        if guard_expansion and source_bytes > 0 and projected >= source_bytes:
            attempt["skipped"] = "projected_byte_expansion"
            attempt["projected_f0_bytes"] = int(projected)
            record["ladder"]["attempts"].append(attempt)
            record["ladder"]["stopped_by"] = "byte_expansion_guard"
            break

        f0, f0_metrics = write_f0(
            source, desc, tensor_out, group_size, chunk_rows, codec
        )
        attempt["f0"] = f0_metrics
        if rung_idx == 0:
            record["local_quality"]["f0"] = f0_metrics

        if quality_pass(f0_metrics, args.cosine_min, args.nrmse_max):
            attempt["selected"] = "F0_ONLY"
            record["ladder"]["attempts"].append(attempt)
            record["stages"] = [f0]
            record["output_bytes"] = f0["bytes"]
            record["gate"] = {"status": "F0_LOCAL_GATE_PASS", "safe_policy": "F0_ONLY"}
            record["local_quality"]["selected"] = f0_metrics
            record["local_quality"]["selected_local_pass"] = True
            record["ladder"]["selected_rung"] = attempt["rung"]
            return record

        # F1 low-rank sobre este F0. Numa escada, um F0 muito distante do gate
        # dificilmente é resgatado por rank <= 32: pular o SVD nesse caso evita
        # trabalho inútil (heurística; --ladder-f0-min-cosine 0 desliga).
        skip_floor = float(getattr(args, "ladder_f0_min_cosine", 0.0) or 0.0)
        hopeless = (
            not last_rung
            and skip_floor > 0.0
            and f0_metrics["cosine"] < skip_floor
        )
        if not effective_ranks or hopeless:
            attempt["skipped_f1"] = (
                "rank_not_applicable" if not effective_ranks else "f0_below_ladder_floor"
            )
            record["ladder"]["attempts"].append(attempt)
            if last_rung:
                break
            cleanup_cascade_stages(tensor_out)
            continue

        u, v, f1_info = randomized_residual_factors(
            source=source,
            desc=desc,
            out_dir=tensor_out,
            group_size=group_size,
            chunk_rows=chunk_rows,
            max_rank=max(effective_ranks),
            oversample=args.oversample,
            power_iters=args.power_iters,
            seed=args.seed + idx,
            codec=codec,
        )
        # Early-abort: resíduo sem estrutura low-rank não é resgatável por rank
        # <= max(ranks). Evita o passe de avaliação e a gravação do F1. O piso
        # sai do PRÓPRIO gate (quanta energia o F1 teria de capturar para o
        # tensor passar); --f1-min-energy é um piso absoluto adicional.
        captured = float(f1_info["captured_fraction"])
        needed = required_capture_fraction(f0_metrics, args.cosine_min, args.nrmse_max)
        energy_floor = float(getattr(args, "f1_min_energy", 0.0) or 0.0)
        attempt["f1_spectrum"] = {
            "captured_fraction": round(captured, 6),
            "required_fraction": round(needed, 6),
            "safety_margin": F1_ENERGY_SAFETY,
            "max_rank": f1_info["max_rank"],
        }
        hopeless_energy = needed > 0 and captured < needed * F1_ENERGY_SAFETY
        below_floor = energy_floor > 0 and captured < energy_floor
        if hopeless_energy or below_floor:
            attempt["skipped_f1"] = "residual_not_low_rank"
            attempt["f1_spectrum"]["trigger"] = (
                "below_gate_requirement" if hopeless_energy else "below_explicit_floor"
            )
            record["ladder"]["attempts"].append(attempt)
            del u, v
            if last_rung:
                break
            cleanup_cascade_stages(tensor_out)
            continue

        rank_metrics = evaluate_rank_candidates(
            source=source,
            desc=desc,
            out_dir=tensor_out,
            group_size=group_size,
            chunk_rows=chunk_rows,
            u=u,
            v=v,
            ranks=effective_ranks,
            codec=codec,
        )
        attempt["rank_candidates"] = {str(k): val for k, val in rank_metrics.items()}
        if rung_idx == 0:
            record["local_quality"]["rank_candidates"] = attempt["rank_candidates"]

        chosen = None
        for r in effective_ranks:
            m = rank_metrics.get(r)
            if m and quality_pass(m, args.cosine_min, args.nrmse_max):
                chosen = r
                break

        if chosen is None:
            attempt["selected"] = None
            record["ladder"]["attempts"].append(attempt)
            del u, v
            if last_rung:
                break
            cleanup_cascade_stages(tensor_out)
            continue

        f1 = write_f1(tensor_out, u, v, chosen)
        # Guarda de expansão também no total F0+F1: o passthrough exato é menor
        # e sem perda, então vence.
        if guard_expansion and source_bytes > 0 and (f0["bytes"] + f1["bytes"]) >= source_bytes:
            attempt["skipped"] = "byte_expansion_with_f1"
            attempt["f0_plus_f1_bytes"] = int(f0["bytes"] + f1["bytes"])
            record["ladder"]["attempts"].append(attempt)
            record["ladder"]["stopped_by"] = "byte_expansion_guard"
            del u, v
            break

        attempt["selected"] = f"F0_PLUS_F1_RANK_{chosen}"
        record["ladder"]["attempts"].append(attempt)
        record["ladder"]["selected_rung"] = attempt["rung"]
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

    # Nenhum degrau da escada passou o gate: passthrough exato (nunca degrada
    # a qualidade em silêncio — é o contrato do conversor).
    cleanup_cascade_stages(tensor_out)
    if record["ladder"].get("stopped_by") == "byte_expansion_guard":
        reason_raw = "cascade_would_expand_bytes_source_is_smaller_and_exact"
    elif not effective_ranks:
        reason_raw = "rank_not_applicable_fallback"
    else:
        reason_raw = "cascade_local_quality_failed_fallback_exact_raw"
    if getattr(args, "keep_source_passthrough", False):
        stage = write_external_stage(desc, reason_raw)
    else:
        stage = write_raw_stage(source, desc, tensor_out, reason_raw)
    record["stages"] = [stage]
    record["output_bytes"] = stage["bytes"]
    record["local_quality"]["selected_local_pass"] = True
    record["local_quality"]["fallback_exact_raw"] = True
    record["ladder"]["selected_rung"] = "raw"
    record["gate"] = {"status": "NOT_APPLICABLE", "safe_policy": "F0_ONLY_RAW"}
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

    # Orçamento de RAM derivado da máquina (§28): --target-ram-gb 0 = auto.
    machine_ram = detect_total_ram_bytes()
    requested_target = float(getattr(args, "target_ram_gb", 0.0) or 0.0)
    if requested_target > 0:
        target_ram_gb = requested_target
        target_source = "explicit_flag"
    else:
        auto_bytes, target_source = auto_target_ram_bytes(machine_ram)
        target_ram_gb = auto_bytes / 1024 ** 3
    if machine_ram:
        print(
            f"RAM da máquina: {machine_ram / 1024 ** 3:.1f} GiB | "
            f"orçamento do modelo: {target_ram_gb:.1f} GiB ({target_source})"
        )
    else:
        print(f"RAM da máquina: indetectável | orçamento do modelo: {target_ram_gb:.1f} GiB")

    # Fatia de conversão também acompanha a máquina (16 MB em 8 GiB, teto 128 MB).
    if float(getattr(args, "ram_budget_mb", 0.0) or 0.0) <= 0 and machine_ram:
        slice_mb = min(max(machine_ram / (512 * 1024 * 1024), 16.0), 128.0)
        args.ram_budget_mb = float(round(slice_mb))
        print(f"Fatia de conversão: {args.ram_budget_mb:.0f} MB (auto)")

    start_time = time.time()

    peak_rss = read_vmrss_bytes() or 0
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
                desc, eligible, args.group_size, ranks, args.oversample,
                keep_source=bool(getattr(args, "keep_source_passthrough", False)),
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

        # Amostra de RSS por tensor + coleta: mantém o pico visível e o
        # working-set baixo em máquinas de 8 GB.
        rss_now = read_vmrss_bytes()
        if rss_now:
            peak_rss = max(peak_rss, rss_now)
        gc.collect()

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
    resolved_ladder_modes: Dict[str, int] = {}
    for rec in manifest["tensors"]:
        mode_used = (rec.get("ladder") or {}).get("mode")
        if mode_used:
            resolved_ladder_modes[mode_used] = resolved_ladder_modes.get(mode_used, 0) + 1
    dominant_ladder_mode = (
        max(resolved_ladder_modes.items(), key=lambda kv: kv[1])[0]
        if resolved_ladder_modes
        else resolve_ladder_mode(args, None)
    )
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
        "codec_ladder": getattr(args, "codec_ladder", "auto"),
        # Com --codec-ladder auto a escada é decidida POR TENSOR (pela fonte):
        # reportar só o pedido mentiria. `codec_ladder_resolved` conta os modos
        # realmente usados e `ladder_rungs` segue o modo dominante.
        "codec_ladder_resolved": resolved_ladder_modes,
        "ladder_rungs": [
            f"{c}/g{g}"
            for c, g in ladder_rungs(
                args_like(int(args.group_size), dominant_ladder_mode)
            )
        ],
    }
    summary["residency"] = residency_report(
        manifest["tensors"], target_ram_gb, machine_ram, target_source
    )
    if peak_rss:
        summary["conversion_peak_rss_bytes"] = int(peak_rss)
        summary["conversion_peak_rss_method"] = "proc_vmrss_per_tensor_sampling_v1"
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
            # Card "Modelos convertidos" do painel (§28): residência medida +
            # tabela de RAM por largura de bits (projeção rotulada).
            "converter": {
                "codec_ladder": manifest["summary"]["codec_ladder"],
                "codec_ladder_resolved": manifest["summary"]["codec_ladder_resolved"],
                "ladder_rungs": manifest["summary"]["ladder_rungs"],
                "selected_rungs": manifest["summary"]["residency"]["selected_rungs"],
                "bundle_requires_source": (
                    manifest["summary"]["residency"]["bundle_requires_source"]
                ),
                "external_source_bytes": (
                    manifest["summary"]["residency"]["external_source_bytes"]
                ),
                "f0_effective_bits_per_weight": (
                    manifest["summary"]["residency"]["f0_effective_bits_per_weight"]
                ),
                "resident_hot_bytes": (
                    manifest["summary"]["residency"]["resident_hot_bytes"]
                ),
                "pageable_warm_bytes": (
                    manifest["summary"]["residency"]["pageable_warm_bytes"]
                ),
                "raw_passthrough_bytes": (
                    manifest["summary"]["residency"]["raw_passthrough_bytes"]
                ),
                "target_ram_gb": manifest["summary"]["residency"]["target_ram_gb"],
                "target_source": manifest["summary"]["residency"]["target_source"],
                "machine_total_ram_bytes": (
                    manifest["summary"]["residency"]["machine_total_ram_bytes"]
                ),
                "fits_resident_in_target": (
                    manifest["summary"]["residency"]["fits_resident_in_target"]
                ),
                "memory_by_bits": manifest["summary"]["residency"]["memory_by_bits"],
                "memory_by_bits_label": "PROJETADO",
                "converted_tensor_count": int(converted),
                "passthrough_tensor_count": int(passthrough),
                "source_format": descs[0].source_kind if descs else None,
                "conversion_peak_rss_bytes": int(peak_rss) if peak_rss else None,
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

    res = manifest["summary"]["residency"]
    gib = 1024 ** 3
    print("-" * 78)
    resolved_desc = ", ".join(
        f"{mode}×{count}"
        for mode, count in sorted(
            manifest["summary"]["codec_ladder_resolved"].items(),
            key=lambda kv: -kv[1],
        )
    )
    print(f"Escada de codecs    : {manifest['summary']['codec_ladder']} "
          f"({' -> '.join(manifest['summary']['ladder_rungs'])} -> raw)"
          + (f" | resolvida: {resolved_desc}" if resolved_desc else ""))
    print(f"Degraus escolhidos  : {res['selected_rungs']}")
    if res["f0_effective_bits_per_weight"] is not None:
        print(f"F0 médio (bpw)      : {res['f0_effective_bits_per_weight']:.3f}")
    print(f"Residente (HOT)     : {res['resident_hot_bytes'] / gib:.3f} GiB "
          f"(raw exato: {res['raw_passthrough_bytes'] / gib:.3f} GiB)")
    if res.get("bundle_requires_source"):
        print(f"Fora do bundle      : {res['external_source_bytes'] / gib:.3f} GiB "
              f"em SOURCE_EXTERNAL — o bundle DEPENDE do checkpoint de origem")
    guard_hits = sum(
        1 for t in manifest["tensors"]
        if (t.get("ladder") or {}).get("stopped_by") == "byte_expansion_guard"
    )
    if guard_hits:
        print(f"Guarda de expansão  : {guard_hits} tensor(es) mantidos na fonte "
              f"(CASCADE ficaria maior que o original, sem ganho de qualidade)")
    print(f"Paginável (WARM F1) : {res['pageable_warm_bytes'] / gib:.3f} GiB")
    verdict = "CABE" if res["fits_resident_in_target"] else "NAO CABE"
    print(f"Alvo {res['target_ram_gb']:.1f} GB livres  : {verdict} residente "
          f"(reserva de {res['runtime_reserve_bytes'] / gib:.1f} GiB p/ KV+runtime)")
    if peak_rss:
        print(f"Pico de RSS medido  : {peak_rss / gib:.3f} GiB (conversão)")

    print("-" * 78)
    print("RAM NECESSÁRIA POR LARGURA DE BITS (projeção)")
    achieved = res["f0_effective_bits_per_weight"]
    for row in res["memory_by_bits"]:
        mark = "cabe" if row["fits_in_target"] else "não cabe"
        star = ""
        if achieved is not None and row["bits"] <= achieved < row["bits"] + 1:
            star = "  <- largura média desta conversão"
        print(f"  {row['label']:>6} : {row['resident_gib']:8.2f} GiB   [{mark}]{star}")
    if achieved is not None:
        print(f"  MEDIDO : {res['resident_hot_bytes'] / gib:8.2f} GiB   "
              f"[{'cabe' if res['fits_resident_in_target'] else 'não cabe'}]"
              f"  (escada real, {achieved:.3f} bpw)")
    print("  Projeção recalcula só o estágio base; passthrough exato entra como está.")
    print("  1-2 bits uniformes NÃO preservam qualidade neste projeto (medido: PPL")
    print("  41,5M em ternário e 22k em INT2 vs 49,1 do original) — a escada só")
    print("  desce de bits onde o gate de qualidade aprova.")
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
        codec_ladder="compact",
        ladder_f0_min_cosine=0.98,
        ram_budget_mb=384.0,
        target_ram_gb=8.0,
        allow_byte_expansion=False,
        keep_source_passthrough=False,
        f1_min_energy=0.0,
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
    c.add_argument(
        "--allow-byte-expansion", action="store_true",
        help="Desliga a guarda de expansão: aceita estágios CASCADE mesmo quando "
             "ficam MAIORES que o tensor de origem. Por padrão a guarda prefere "
             "o passthrough exato nesses casos (menor e sem perda) — relevante "
             "em fonte GGUF já quantizada, onde INT4 expande os bytes",
    )
    c.add_argument(
        "--keep-source-passthrough", action="store_true",
        help="Não COPIA tensores que ficam em passthrough (embeddings, lm_head, "
             "MoE, fora do --include-regex): registra-os como SOURCE_EXTERNAL e "
             "o bundle passa a depender do checkpoint de origem. Economiza disco "
             "e evita o pico de RAM da cópia em tensores gigantes; a RAM de "
             "execução NÃO muda (os bytes continuam contando como residentes)",
    )
    c.add_argument(
        "--f1-min-energy", type=float, default=0.0,
        help="Piso ABSOLUTO extra para a fração de energia do resíduo capturada "
             "pelos ranks disponíveis (0 = padrão, usa só o piso derivado do "
             "gate). O conversor já aborta o F1 quando a energia capturada fica "
             # %% obrigatório: argparse interpola o help com %-formatting.
             f"abaixo de {F1_ENERGY_SAFETY * 100:.0f}%% do que o gate exigiria "
             "— medido: "
             "resíduo INT4 entrega ~0,25 com rank<=32, e attn_output precisaria "
             "de ~0,38 para sair de cosine 0,992 e fechar 0,995",
    )
    c.add_argument(
        "--codec-ladder", choices=sorted(set(LADDERS) | {"auto"}), default="auto",
        help="Escada de codecs por tensor, do mais barato ao mais caro; o "
             "primeiro degrau que passa o gate de qualidade vence. "
             "auto (padrão)=escolhe pela fonte: já low-bit (IQ2/Q4_K/...) usa "
             "safe, BF16/F16/F32 usa compact. "
             "safe=int4/g64 -> int4/g32 (resgata tensores que cairiam em raw); "
             "compact=int2/g64 -> int4/g64 -> int4/g32 (menor RAM, alvo 8 GB); "
             "int4=somente int4/g64 (comportamento das versões anteriores). "
             "Em TODOS os modos, tensor que reprova em todos os degraus cai em "
             "passthrough exato — a qualidade nunca degrada em silêncio",
    )
    c.add_argument(
        "--ladder-f0-min-cosine", type=float, default=0.98,
        help="Heurística de custo: num degrau intermediário cujo F0 fique com "
             "cosine abaixo deste valor, pula o SVD do residual e vai direto "
             "ao próximo degrau (0 desliga a heurística e sempre tenta F1)",
    )
    c.add_argument(
        "--ram-budget-mb", type=float, default=0.0,
        help="Orçamento de RAM por fatia de linhas. 0 = automático pela RAM da "
             "máquina (16 MB em 8 GiB, teto 128 MB). O pipeline já é streaming, "
             "então em larguras típicas isso não muda nada; em tensores muito "
             "largos (ex.: FFN de 70B com 28k colunas) reduz --chunk-rows "
             "automaticamente e mantém o pico baixo",
    )
    c.add_argument(
        "--target-ram-gb", type=float, default=0.0,
        help="Orçamento de RAM do MODELO na máquina alvo. 0 = automático: RAM "
             "total menos 8 GiB de reserva para SO/apps, com piso de 50%% "
             "(16 GiB->8, 24->16, 32->24, 8->4). Sem detecção, usa 8 GiB",
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
