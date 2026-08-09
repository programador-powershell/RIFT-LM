#!/usr/bin/env python3
# ==============================================================================
# RIFT-LM v0.3.5 — M0 / B0 + Phase 1 Gain Tracker + Auto Battery Recorder
#
# Target default: Qwen/Qwen2.5-0.5B
#
# IMPORTANT:
#   1) B0 MUST pass before codecs are exercised.
#   2) Q4_LINEAR_TEST below is NOT MXFP4. It only validates Progressive
#      Precision / BASE + BITPLANE semantics with physically packed 2-bit streams.
#   3) The Python/Numpy path is a REFERENCE path, not the production low-bit
#      kernel. A speedup < 1.0x is valid and MUST be reported honestly.
#   4) The M0 spec freezes Header M0 but does not yet freeze numeric codec IDs,
#      PageEntry byte size, nor an explicit Shape Blob section. This script uses
#      TEST-LOCAL conventions for those items and reports them as experimental.
# ==============================================================================

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import ctypes.util
import csv
import json
import mmap
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import uuid
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np


# ------------------------------------------------------------------------------
# Dependency helper
# ------------------------------------------------------------------------------

def ensure_import(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError:
        pkg = pip_name or module
        print(f"[deps] Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return __import__(module)


# XXH3-64 backend:
# 1) módulo Python xxhash, quando disponível;
# 2) libxxhash nativa via ctypes;
# 3) instalação automática apenas como último recurso.
_xxhash_py = None
_xxhash_native = None

try:
    import xxhash as _xxhash_py
except ImportError:
    lib_name = ctypes.util.find_library("xxhash")
    if lib_name:
        _xxhash_native = ctypes.CDLL(lib_name)
        _xxhash_native.XXH3_64bits_withSeed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        _xxhash_native.XXH3_64bits_withSeed.restype = ctypes.c_uint64
    else:
        try:
            _xxhash_py = ensure_import("xxhash")
        except Exception as exc:
            raise SystemExit(
                "XXH3-64 indisponível. Instale 'xxhash' ou 'libxxhash'.\n"
                f"Erro: {exc}"
            )

# PyTorch/Transformers são carregados apenas em --mode phase1.
# Isso permite que --mode self-test valide B0/Container/codec sintético sem
# baixar modelo nem depender de Transformers.
torch = None
F = None
AutoModelForCausalLM = None
AutoTokenizer = None
AutoModel = None
AutoModelForMultimodalLM = None


def ensure_phase1_dependencies():
    global torch, F, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForMultimodalLM
    if torch is not None:
        return

    # Transformers delega alguns tokenizadores a bibliotecas opcionais. O
    # Colab nem sempre inclui ambas, mesmo quando torch/transformers já estão
    # disponíveis (Llama/SentencePiece e Qwen/tiktoken).
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
            "PyTorch/Transformers são necessários para --mode phase1.\n"
            "Instale com: pip install torch transformers accelerate sentencepiece tiktoken\n"
            f"Erro original: {exc}"
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


def normalize_huggingface_model_id(value: str) -> str:
    """Aceita ``org/modelo`` ou uma URL pública do Hugging Face."""
    model_id = str(value or "").strip()
    parsed = urlparse(model_id)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("--model aceita somente um model ID ou URL do huggingface.co")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("URL do Hugging Face incompleta; use https://huggingface.co/ORG/MODELO")
        model_id = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("Model ID inválido; formato esperado: organizacao/modelo")
    return model_id


def resolve_linear_weight_name(model: Any, requested: str) -> str:
    """Resolve automaticamente uma camada Linear 2D comparável entre famílias."""
    state = model.state_dict()
    if requested and requested.lower() != "auto":
        if requested not in state:
            raise KeyError(f"Tensor não encontrado: {requested}")
        if getattr(state[requested], "ndim", 0) != 2:
            raise ValueError(f"Tensor precisa ser uma matriz 2D: {requested}")
        return requested

    preferred_suffixes = (
        "self_attn.q_proj",
        "self_attn.qkv_proj",
        "self_attn.query_key_value",
        "attention.q_proj",
        "attention.wq",
        "attn.q_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.o_proj",
    )
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and getattr(module, "weight", None) is not None
        and module.weight.ndim == 2
    ]
    for suffix in preferred_suffixes:
        match = next((name for name in linear_names if name.lower().endswith(suffix)), None)
        if match:
            return f"{match}.weight"
    if linear_names:
        return f"{linear_names[0]}.weight"
    raise KeyError("O modelo não expõe nenhuma camada torch.nn.Linear com peso 2D")


# ------------------------------------------------------------------------------
# 1. Constantes congeladas — RIFT v0.3.5 / Seção 51
# ------------------------------------------------------------------------------

RIFT_MAGIC = b"RIFT"
RIFT_CONTAINER_VERSION_M0 = 0x0001
RIFT_PROFILE_M0 = 0x0000
RIFT_HEADER_M0_SIZE = 128
RIFT_HEADER_CHECKSUM_OFFSET = 0x40
RIFT_HEADER_CHECKSUM_SIZE = 8
RIFT_CHECKSUM_SEED = 0
EXPECTED_GOLDEN_CHECKSUM = 0x8DB3A70FC08A38F3

# Header M0 exato: 128 bytes
# 4s, H, H, I, I, 7xQ, 56s
RIFT_HEADER_FORMAT = "<4sHHIIQQQQQQQ56s"
assert struct.calcsize(RIFT_HEADER_FORMAT) == RIFT_HEADER_M0_SIZE

# --------------------------------------------------------------------------
# TEST-LOCAL M0 layout conventions.
# TensorEntry matches the listed fields and totals 32 bytes.
# PageEntry byte size/numeric codec IDs are not frozen by v0.3.5; do NOT
# promote these constants to production ABI without a RIFT_M0_CHANGE_REQUEST.
# --------------------------------------------------------------------------
TENSOR_ENTRY_FORMAT = "<IHBBQQII"
TENSOR_ENTRY_SIZE = struct.calcsize(TENSOR_ENTRY_FORMAT)  # 32
assert TENSOR_ENTRY_SIZE == 32

PAGE_ENTRY_FORMAT = "<IIHBBQQQQ"
PAGE_ENTRY_SIZE = struct.calcsize(PAGE_ENTRY_FORMAT)  # 44

RIFT_DTYPE_FP32_TEST = 0x0001
RIFT_ROLE_ALWAYS_ACTIVE_TEST = 0x01

RIFT_CODEC_RAW_TEST = 0x0000
RIFT_CODEC_Q4_LINEAR_TEST = 0x7F01   # TEST-LOCAL, NOT production/frozen
RIFT_PAGE_FULL_TEST = 0x03
RIFT_PAGE_BASE_TEST = 0x01
RIFT_PAGE_REFINEMENT_TEST = 0x02

UINT64_MAX = (1 << 64) - 1


# ------------------------------------------------------------------------------
# Errors with stable names for golden negative tests
# ------------------------------------------------------------------------------

class RiftError(Exception):
    code = "RIFT_ERROR"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)


class RiftBadMagic(RiftError):
    code = "RIFT_BAD_MAGIC"


class RiftBadVersion(RiftError):
    code = "RIFT_BAD_VERSION"


class RiftTruncatedHeader(RiftError):
    code = "RIFT_TRUNCATED_HEADER"


class RiftHeaderChecksumMismatch(RiftError):
    code = "RIFT_HEADER_CHECKSUM_MISMATCH"


class RiftOffsetOverflow(RiftError):
    code = "RIFT_OFFSET_OVERFLOW"


class RiftRangeOutOfFile(RiftError):
    code = "RIFT_RANGE_OUT_OF_FILE"


class RiftIRInvalid(RiftError):
    code = "RIFT_IR_INVALID"


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment precisa ser potência de 2")
    return (value + alignment - 1) & ~(alignment - 1)


def checked_u64_range(offset: int, size: int, file_size: int) -> None:
    if offset < 0 or size < 0:
        raise RiftRangeOutOfFile("offset/size negativo")
    if offset > UINT64_MAX or size > UINT64_MAX:
        raise RiftOffsetOverflow("valor excede u64")
    if offset > UINT64_MAX - size:
        raise RiftOffsetOverflow("offset + size excede u64")
    end = offset + size
    if end > file_size:
        raise RiftRangeOutOfFile(
            f"range [{offset}, {end}) excede file_size={file_size}"
        )


def xxh3_64(data: bytes) -> int:
    raw = bytes(data)
    if _xxhash_py is not None:
        return int(_xxhash_py.xxh3_64_intdigest(raw, seed=RIFT_CHECKSUM_SEED))

    if _xxhash_native is not None:
        if len(raw) == 0:
            # XXH3 aceita ponteiro nulo com tamanho zero.
            ptr = ctypes.c_void_p()
            return int(
                _xxhash_native.XXH3_64bits_withSeed(
                    ptr, 0, ctypes.c_uint64(RIFT_CHECKSUM_SEED)
                )
            )

        buf = ctypes.create_string_buffer(raw)
        return int(
            _xxhash_native.XXH3_64bits_withSeed(
                ctypes.cast(buf, ctypes.c_void_p),
                len(raw),
                ctypes.c_uint64(RIFT_CHECKSUM_SEED),
            )
        )

    raise RuntimeError("Nenhum backend XXH3-64 disponível")


# ------------------------------------------------------------------------------
# 2. Header M0 — Writer/Reader + Golden Test
# ------------------------------------------------------------------------------

def create_rift_header_m0(
    *,
    tensor_table_offset: int,
    tensor_count: int,
    page_table_offset: int,
    page_count: int,
    payload_offset: int,
    file_size: int,
    flags: int = 0,
) -> Tuple[bytes, int]:
    """
    Serializa o Header M0 canônico de 128 bytes, little-endian.
    O checksum é calculado com 0x40..0x47 zerado.
    """
    reserved0 = 0
    reserved1 = bytes(56)

    header_zero_checksum = struct.pack(
        RIFT_HEADER_FORMAT,
        RIFT_MAGIC,
        RIFT_CONTAINER_VERSION_M0,
        RIFT_PROFILE_M0,
        flags,
        reserved0,
        tensor_table_offset,
        tensor_count,
        page_table_offset,
        page_count,
        payload_offset,
        file_size,
        0,  # header_checksum zerado durante o cálculo
        reserved1,
    )

    assert len(header_zero_checksum) == RIFT_HEADER_M0_SIZE

    checksum = xxh3_64(header_zero_checksum)

    header_final = bytearray(header_zero_checksum)
    header_final[
        RIFT_HEADER_CHECKSUM_OFFSET:
        RIFT_HEADER_CHECKSUM_OFFSET + RIFT_HEADER_CHECKSUM_SIZE
    ] = struct.pack("<Q", checksum)

    return bytes(header_final), checksum


def parse_rift_header_m0(header_bytes: bytes, actual_file_size: int) -> Dict[str, int]:
    if len(header_bytes) < RIFT_HEADER_M0_SIZE:
        raise RiftTruncatedHeader(
            f"esperado={RIFT_HEADER_M0_SIZE}, recebido={len(header_bytes)}"
        )

    values = struct.unpack(RIFT_HEADER_FORMAT, header_bytes[:RIFT_HEADER_M0_SIZE])

    (
        magic,
        container_version,
        profile,
        flags,
        reserved0,
        tensor_table_offset,
        tensor_count,
        page_table_offset,
        page_count,
        payload_offset,
        file_size,
        received_checksum,
        reserved1,
    ) = values

    if magic != RIFT_MAGIC:
        raise RiftBadMagic(repr(magic))
    if container_version != RIFT_CONTAINER_VERSION_M0 or profile != RIFT_PROFILE_M0:
        raise RiftBadVersion(
            f"version={container_version:#x}, profile={profile:#x}"
        )
    if reserved0 != 0 or reserved1 != bytes(56):
        raise RiftError("reserved fields M0 precisam estar zerados")

    tmp = bytearray(header_bytes[:RIFT_HEADER_M0_SIZE])
    tmp[0x40:0x48] = bytes(8)
    calculated = xxh3_64(bytes(tmp))
    if calculated != received_checksum:
        raise RiftHeaderChecksumMismatch(
            f"recebido={received_checksum:#018x}, calculado={calculated:#018x}"
        )

    if file_size != actual_file_size:
        raise RiftRangeOutOfFile(
            f"header file_size={file_size}, tamanho real={actual_file_size}"
        )

    # Validação estrutural/bounds das regiões congeladas.
    # O tamanho de TensorEntry é 32 bytes conforme seus campos.
    tensor_table_bytes = tensor_count * TENSOR_ENTRY_SIZE
    if tensor_count and tensor_table_bytes // tensor_count != TENSOR_ENTRY_SIZE:
        raise RiftOffsetOverflow("overflow tensor table")

    page_table_bytes = page_count * PAGE_ENTRY_SIZE
    if page_count and page_table_bytes // page_count != PAGE_ENTRY_SIZE:
        raise RiftOffsetOverflow("overflow page table")

    checked_u64_range(tensor_table_offset, tensor_table_bytes, file_size)
    checked_u64_range(page_table_offset, page_table_bytes, file_size)
    checked_u64_range(payload_offset, 0, file_size)

    return {
        "flags": flags,
        "tensor_table_offset": tensor_table_offset,
        "tensor_count": tensor_count,
        "page_table_offset": page_table_offset,
        "page_count": page_count,
        "payload_offset": payload_offset,
        "file_size": file_size,
        "header_checksum": received_checksum,
    }


def read_rift_header_m0(path: Path) -> Dict[str, int]:
    actual = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(RIFT_HEADER_M0_SIZE)
    return parse_rift_header_m0(raw, actual)


def test_golden_header() -> bytes:
    header, checksum = create_rift_header_m0(
        tensor_table_offset=0x80,
        tensor_count=0,
        page_table_offset=0x80,
        page_count=0,
        payload_offset=0x80,
        file_size=0x80,
    )

    assert len(header) == 128
    assert checksum == EXPECTED_GOLDEN_CHECKSUM, (
        "FALHA NO GOLDEN CHECKSUM: "
        f"esperado={EXPECTED_GOLDEN_CHECKSUM:#018x}, "
        f"obtido={checksum:#018x}"
    )

    expected_le = bytes.fromhex("F3 38 8A C0 0F A7 B3 8D")
    assert header[0x40:0x48] == expected_le

    print(
        "[B0.1] GOLDEN HEADER PASS — "
        f"XXH3-64={checksum:#018x}, size={len(header)}"
    )
    return header


# ------------------------------------------------------------------------------
# 3. RIFT-IR M0 — criação + DAG validator
# ------------------------------------------------------------------------------

KNOWN_M0_OPCODES = {
    "RIFT_OP_EMBEDDING",
    "RIFT_OP_LINEAR",
    "RIFT_OP_RMSNORM",
    "RIFT_OP_ROPE",
    "RIFT_OP_ATTENTION",
    "RIFT_OP_ACTIVATION",
    "RIFT_OP_ADD",
    "RIFT_OP_OUTPUT",
    "RIFT_OP_CUSTOM",
}


def build_linear_ir_m0(
    *,
    model_id: str,
    architecture: str,
    weight_name: str,
    weight_shape: Iterable[int],
) -> Dict[str, Any]:
    out_features, in_features = map(int, weight_shape)

    tensors = [
        {
            "id": 0,
            "name": "rift.test.activation.input",
            "dtype": "FP32",
            "rank": 2,
            "shape": [1, in_features],
            "semantic_role": "RIFT_ROLE_STATIC",
            "activation_class": "EXTERNAL_INPUT",
            "source_tensor_name": "",
            "flags": 0,
        },
        {
            "id": 1,
            "name": weight_name,
            "dtype": "FP32",
            "rank": 2,
            "shape": [out_features, in_features],
            "semantic_role": "RIFT_ROLE_ALWAYS_ACTIVE",
            "activation_class": "WEIGHT",
            "source_tensor_name": weight_name,
            "flags": 0,
        },
        {
            "id": 2,
            "name": "rift.test.activation.output",
            "dtype": "FP32",
            "rank": 2,
            "shape": [1, out_features],
            "semantic_role": "RIFT_ROLE_STATIC",
            "activation_class": "OUTPUT",
            "source_tensor_name": "",
            "flags": 0,
        },
    ]

    operations = [
        {
            "id": 0,
            "opcode": "RIFT_OP_LINEAR",
            "inputs": [0],
            "outputs": [2],
            "weights": [1],
            "attrs": {},
        }
    ]

    return {
        "ir_version": 1,
        "execution_model": "TOPOLOGICAL_ARRAY_V1",
        "model_id": model_id,
        "architecture": architecture,
        "tensors": tensors,
        "operations": operations,
        "input_ids_tensor": 0,
        "output_tensor": 2,
    }


def validate_rift_ir_m0(ir: Dict[str, Any]) -> None:
    if ir.get("ir_version") != 1:
        raise RiftIRInvalid("ir_version inválida")
    if ir.get("execution_model") != "TOPOLOGICAL_ARRAY_V1":
        raise RiftIRInvalid("execution_model inválido")

    tensors = ir.get("tensors")
    operations = ir.get("operations")
    if not isinstance(tensors, list) or not isinstance(operations, list):
        raise RiftIRInvalid("tensors/operations precisam ser arrays")

    tensor_ids = [t.get("id") for t in tensors]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise RiftIRInvalid("duplicate_tensor_id")

    tensor_id_set = set(tensor_ids)
    op_ids = [op.get("id") for op in operations]
    if len(op_ids) != len(set(op_ids)):
        raise RiftIRInvalid("duplicate_op_id")

    external_inputs = {ir.get("input_ids_tensor")}
    produced: set[int] = set()

    for op_index, op in enumerate(operations):
        opcode = op.get("opcode")
        if opcode not in KNOWN_M0_OPCODES:
            raise RiftIRInvalid(f"unknown_opcode:{opcode}")
        if opcode == "RIFT_OP_CUSTOM":
            raise RiftIRInvalid("RIFT_OP_CUSTOM exige backend custom registrado")

        inputs = op.get("inputs", [])
        outputs = op.get("outputs", [])
        weights = op.get("weights", [])

        for tid in list(inputs) + list(outputs) + list(weights):
            if tid not in tensor_id_set:
                raise RiftIRInvalid(f"tensor inexistente: {tid}")

        for tid in inputs:
            if tid not in external_inputs and tid not in produced:
                raise RiftIRInvalid(
                    f"forward_dependency op_index={op_index}, tensor={tid}"
                )

        for out in outputs:
            if out in produced or out in external_inputs:
                raise RiftIRInvalid(f"output com produtor inválido/duplicado: {out}")
            produced.add(out)

    final_output = ir.get("output_tensor")
    if final_output not in tensor_id_set:
        raise RiftIRInvalid("output_tensor inexistente")
    if final_output not in produced and final_output not in external_inputs:
        raise RiftIRInvalid("output_tensor sem produtor")


# ------------------------------------------------------------------------------
# 4. Golden files — positivos e negativos
# ------------------------------------------------------------------------------

def rewrite_header_with_mutation(base_header: bytes, **fields: int) -> bytes:
    vals = list(struct.unpack(RIFT_HEADER_FORMAT, base_header))
    names = [
        "magic", "container_version", "profile", "flags", "reserved0",
        "tensor_table_offset", "tensor_count", "page_table_offset", "page_count",
        "payload_offset", "file_size", "header_checksum", "reserved1",
    ]
    index = {name: i for i, name in enumerate(names)}
    for k, v in fields.items():
        vals[index[k]] = v
    vals[index["header_checksum"]] = 0
    raw = struct.pack(RIFT_HEADER_FORMAT, *vals)
    checksum = xxh3_64(raw)
    vals[index["header_checksum"]] = checksum
    return struct.pack(RIFT_HEADER_FORMAT, *vals)


def write_golden_files(out_dir: Path, valid_ir: Dict[str, Any]) -> Dict[str, str]:
    golden = out_dir / "tests" / "golden"
    invalid = golden / "invalid"
    invalid.mkdir(parents=True, exist_ok=True)

    expected: Dict[str, str] = {}

    # Positive empty header
    empty_header = test_golden_header()
    (golden / "header_m0_empty.rift").write_bytes(empty_header)

    # IR positive
    (golden / "ir_m0_linear.json").write_text(
        json.dumps(valid_ir, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    # Binary negatives
    b = bytearray(empty_header)
    b[0:4] = b"NOPE"
    (invalid / "bad_magic.rift").write_bytes(bytes(b))
    expected["bad_magic.rift"] = RiftBadMagic.code

    bad_version = rewrite_header_with_mutation(empty_header, container_version=2)
    (invalid / "bad_version.rift").write_bytes(bad_version)
    expected["bad_version.rift"] = RiftBadVersion.code

    (invalid / "truncated_header.rift").write_bytes(empty_header[:64])
    expected["truncated_header.rift"] = RiftTruncatedHeader.code

    bad_checksum = bytearray(empty_header)
    # Corrompe o próprio checksum sem tocar reserved fields.
    bad_checksum[RIFT_HEADER_CHECKSUM_OFFSET] ^= 0x01
    (invalid / "bad_checksum.rift").write_bytes(bytes(bad_checksum))
    expected["bad_checksum.rift"] = RiftHeaderChecksumMismatch.code

    # Deliberate u64 overflow: tensor offset near UINT64_MAX, count > 0.
    overflow = rewrite_header_with_mutation(
        empty_header,
        tensor_table_offset=UINT64_MAX - 8,
        tensor_count=1,
    )
    (invalid / "offset_overflow.rift").write_bytes(overflow)
    expected["offset_overflow.rift"] = RiftOffsetOverflow.code

    out_of_file = rewrite_header_with_mutation(
        empty_header,
        tensor_table_offset=0x1000,
        tensor_count=1,
    )
    (invalid / "range_out_of_file.rift").write_bytes(out_of_file)
    expected["range_out_of_file.rift"] = RiftRangeOutOfFile.code

    # IR negatives
    def dump_invalid(name: str, obj: Dict[str, Any], code: str = RiftIRInvalid.code):
        (invalid / name).write_text(
            json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8"
        )
        expected[name] = code

    d = copy.deepcopy(valid_ir)
    d["operations"].append(copy.deepcopy(d["operations"][0]))
    dump_invalid("duplicate_op_id.json", d)

    d = copy.deepcopy(valid_ir)
    d["tensors"].append(copy.deepcopy(d["tensors"][0]))
    dump_invalid("duplicate_tensor_id.json", d)

    d = copy.deepcopy(valid_ir)
    d["operations"][0]["inputs"] = [2]  # output é necessário antes de existir
    dump_invalid("forward_dependency.json", d)

    d = copy.deepcopy(valid_ir)
    # Ciclo simples: op0 depende de tensor 3, op1 depende de output 2.
    d["tensors"].append({
        "id": 3, "name": "cycle.tmp", "dtype": "FP32", "rank": 2,
        "shape": [1, 1], "semantic_role": "RIFT_ROLE_STATIC",
        "activation_class": "OUTPUT", "source_tensor_name": "", "flags": 0
    })
    d["operations"][0]["inputs"] = [3]
    d["operations"].append({
        "id": 1, "opcode": "RIFT_OP_ADD",
        "inputs": [2], "outputs": [3], "weights": [], "attrs": {}
    })
    d["output_tensor"] = 3
    dump_invalid("cycle.json", d)

    d = copy.deepcopy(valid_ir)
    d["operations"][0]["opcode"] = "RIFT_OP_DOES_NOT_EXIST"
    dump_invalid("unknown_opcode.json", d)

    manifest = {
        "spec": "RIFT-LM v0.3.5",
        "positive": [
            "header_m0_empty.rift",
            "ir_m0_linear.json",
        ],
        "invalid_expected_errors": expected,
    }
    (golden / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return expected


def run_negative_golden_tests(out_dir: Path, expected: Dict[str, str]) -> None:
    invalid = out_dir / "tests" / "golden" / "invalid"

    for name, expected_code in expected.items():
        path = invalid / name
        try:
            if path.suffix == ".rift":
                read_rift_header_m0(path)
            else:
                obj = json.loads(path.read_text(encoding="utf-8"))
                validate_rift_ir_m0(obj)
        except RiftError as exc:
            if exc.code != expected_code:
                raise AssertionError(
                    f"{name}: esperado={expected_code}, recebido={exc.code}"
                ) from exc
        else:
            raise AssertionError(f"{name}: deveria falhar com {expected_code}")

    print(f"[B0.6] GOLDEN NEGATIVE TESTS PASS — {len(expected)} casos")


# ------------------------------------------------------------------------------
# 5. Container M0 one-tensor — reference test writer/reader + mmap
# ------------------------------------------------------------------------------

def write_one_tensor_container_m0(
    path: Path,
    tensor: np.ndarray,
) -> Dict[str, Any]:
    """
    B0 reference container.

    TEST-LOCAL layout convention:
        HEADER
        TENSOR TABLE
        SHAPE BLOB        <- shape_offset aponta aqui
        PAGE TABLE
        padding/alignment
        PAYLOAD

    A v0.3.5 não congela explicitamente a região SHAPE BLOB nem o tamanho
    binário do PageEntry; portanto isto é implementação de teste, NÃO ABI final.
    """
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    payload = tensor.tobytes(order="C")

    tensor_table_offset = 128
    tensor_count = 1

    shape_blob_offset = tensor_table_offset + TENSOR_ENTRY_SIZE
    shape_blob = struct.pack("<" + ("Q" * tensor.ndim), *map(int, tensor.shape))

    page_table_offset = align_up(shape_blob_offset + len(shape_blob), 8)
    page_count = 1
    page_table_end = page_table_offset + PAGE_ENTRY_SIZE

    # Test-local. Whole-file mmap começa em offset 0; payload não precisa ser
    # page-aligned para este B0 reference reader.
    payload_offset = align_up(page_table_end, 64)
    file_size = payload_offset + len(payload)

    tensor_entry = struct.pack(
        TENSOR_ENTRY_FORMAT,
        1,                              # tensor_id
        RIFT_DTYPE_FP32_TEST,
        tensor.ndim,
        RIFT_ROLE_ALWAYS_ACTIVE_TEST,
        shape_blob_offset,
        0,                              # first_page index
        1,                              # page_count
        0,                              # flags
    )

    payload_checksum = xxh3_64(payload)  # TEST-LOCAL page checksum choice
    page_entry = struct.pack(
        PAGE_ENTRY_FORMAT,
        1,                              # tensor_id
        0,                              # tile_id
        RIFT_CODEC_RAW_TEST,
        RIFT_PAGE_FULL_TEST,
        0,                              # refinement_level
        payload_offset,
        len(payload),
        UINT64_MAX,                     # dependency_page = none (test-local)
        payload_checksum,
    )

    header, _ = create_rift_header_m0(
        tensor_table_offset=tensor_table_offset,
        tensor_count=tensor_count,
        page_table_offset=page_table_offset,
        page_count=page_count,
        payload_offset=payload_offset,
        file_size=file_size,
    )

    buf = bytearray(file_size)
    buf[0:128] = header
    buf[tensor_table_offset:tensor_table_offset + TENSOR_ENTRY_SIZE] = tensor_entry
    buf[shape_blob_offset:shape_blob_offset + len(shape_blob)] = shape_blob
    buf[page_table_offset:page_table_offset + PAGE_ENTRY_SIZE] = page_entry
    buf[payload_offset:payload_offset + len(payload)] = payload

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)

    return {
        "file_size": file_size,
        "payload_bytes": len(payload),
        "container_overhead_bytes": file_size - len(payload),
        "payload_offset": payload_offset,
        "shape_blob_offset": shape_blob_offset,
        "page_entry_size_test_local": PAGE_ENTRY_SIZE,
    }


def read_one_tensor_container_m0(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    header = read_rift_header_m0(path)
    actual_file_size = path.stat().st_size

    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            te_off = header["tensor_table_offset"]
            pe_off = header["page_table_offset"]

            checked_u64_range(te_off, TENSOR_ENTRY_SIZE, actual_file_size)
            checked_u64_range(pe_off, PAGE_ENTRY_SIZE, actual_file_size)

            te = struct.unpack(
                TENSOR_ENTRY_FORMAT,
                mm[te_off:te_off + TENSOR_ENTRY_SIZE],
            )
            (
                tensor_id,
                dtype_original,
                rank,
                semantic_role,
                shape_offset,
                first_page,
                tensor_page_count,
                tensor_flags,
            ) = te

            checked_u64_range(shape_offset, rank * 8, actual_file_size)
            shape = struct.unpack(
                "<" + ("Q" * rank),
                mm[shape_offset:shape_offset + rank * 8],
            )

            pe = struct.unpack(
                PAGE_ENTRY_FORMAT,
                mm[pe_off:pe_off + PAGE_ENTRY_SIZE],
            )
            (
                page_tensor_id,
                tile_id,
                codec_id,
                page_type,
                refinement_level,
                file_offset,
                payload_bytes,
                dependency_page,
                payload_checksum,
            ) = pe

            if page_tensor_id != tensor_id:
                raise RiftError("PageEntry tensor_id não corresponde ao TensorEntry")
            checked_u64_range(file_offset, payload_bytes, actual_file_size)

            # zero-copy addressability: memoryview aponta para o mmap.
            payload_view = memoryview(mm)[file_offset:file_offset + payload_bytes]
            try:
                if xxh3_64(payload_view.tobytes()) != payload_checksum:
                    raise RiftError("payload checksum mismatch")
                arr = np.frombuffer(payload_view, dtype="<f4").reshape(shape).copy()
            finally:
                payload_view.release()

            meta = {
                "tensor_id": tensor_id,
                "shape": list(shape),
                "payload_bytes": payload_bytes,
                "codec_id_test_local": codec_id,
                "payload_offset": file_offset,
                "mapped_file_bytes": actual_file_size,
            }
            return arr, meta
        finally:
            mm.close()


# ------------------------------------------------------------------------------
# 6. Progressive Precision reference codec
# ------------------------------------------------------------------------------

def pack_int2(values: np.ndarray) -> Tuple[bytes, int]:
    """
    Empacota quatro códigos 0..3 por byte.
    Retorna (packed_bytes, original_value_count).
    """
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(flat > 3):
        raise ValueError("pack_int2 aceita apenas valores 0..3")

    n = int(flat.size)
    pad = (-n) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)

    q = flat.reshape(-1, 4)
    packed = (
        q[:, 0]
        | (q[:, 1] << 2)
        | (q[:, 2] << 4)
        | (q[:, 3] << 6)
    ).astype(np.uint8)

    return packed.tobytes(), n


def unpack_int2(packed: bytes, value_count: int) -> np.ndarray:
    b = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(b.size * 4, dtype=np.uint8)
    out[0::4] = b & 0x03
    out[1::4] = (b >> 2) & 0x03
    out[2::4] = (b >> 4) & 0x03
    out[3::4] = (b >> 6) & 0x03
    return out[:value_count]


def quantize_q4_linear_test(W: np.ndarray) -> Dict[str, Any]:
    """
    TEST CODEC — NÃO É MXFP4.

    Quantização uniforme global em 16 níveis.
    Serve apenas para validar:
      Q4 code -> BASE 2-bit + REFINEMENT 2-bit -> Q4 code exato.
    """
    W = np.asarray(W, dtype=np.float32)
    w_min = float(W.min())
    w_max = float(W.max())
    span = max(w_max - w_min, 1e-12)

    normalized = (W - w_min) / span
    codes = np.clip(np.rint(normalized * 15.0), 0, 15).astype(np.uint8)

    base = (codes >> 2) & 0x03
    refinement = codes & 0x03

    base_packed, count = pack_int2(base)
    ref_packed, count2 = pack_int2(refinement)
    assert count == count2 == W.size

    # Prova de round-trip do CÓDIGO alvo.
    codes_roundtrip = (
        (unpack_int2(base_packed, count) << 2)
        | unpack_int2(ref_packed, count)
    ).reshape(W.shape)
    assert np.array_equal(codes, codes_roundtrip)

    return {
        "codes": codes,
        "base_packed": base_packed,
        "ref_packed": ref_packed,
        "value_count": count,
        "shape": W.shape,
        "w_min": w_min,
        "w_max": w_max,
    }


def decode_q4_linear_test(codec: Dict[str, Any], use_refinement: bool) -> np.ndarray:
    n = codec["value_count"]
    base = unpack_int2(codec["base_packed"], n)
    if use_refinement:
        refinement = unpack_int2(codec["ref_packed"], n)
        codes = (base << 2) | refinement
    else:
        # BASE-only escolhe o início do bucket de 4 códigos. Esta é uma política
        # simples de referência, não necessariamente a melhor representação INT2.
        codes = base << 2

    codes = codes.reshape(codec["shape"]).astype(np.float32)
    w_min = codec["w_min"]
    w_max = codec["w_max"]
    return (codes / 15.0) * (w_max - w_min) + w_min


def write_q4_linear_payloads(out_dir: Path, codec: Dict[str, Any]) -> Dict[str, int]:
    """
    Payload experimental autocontido para medir bytes reais no disco.

    Header TEST-LOCAL:
        magic[4] = Q4L0
        w_min f32
        w_max f32
        value_count u64
    """
    hdr = struct.pack(
        "<4sffQ",
        b"Q4L0",
        float(codec["w_min"]),
        float(codec["w_max"]),
        int(codec["value_count"]),
    )

    base_path = out_dir / "q4_linear_base_only.payload"
    full_path = out_dir / "q4_linear_base_plus_ref.payload"

    base_path.write_bytes(hdr + codec["base_packed"])
    full_path.write_bytes(hdr + codec["base_packed"] + codec["ref_packed"])

    return {
        "codec_header_bytes": len(hdr),
        "base_payload_disk_bytes": base_path.stat().st_size,
        "full_payload_disk_bytes": full_path.stat().st_size,
        "base_bitstream_bytes": len(codec["base_packed"]),
        "refinement_bitstream_bytes": len(codec["ref_packed"]),
    }


# ------------------------------------------------------------------------------
# 7. Quality metrics
# ------------------------------------------------------------------------------

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    a = np.asarray(original, dtype=np.float32).reshape(-1)
    b = np.asarray(reconstructed, dtype=np.float32).reshape(-1)

    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    cosine = dot / max(norm_a * norm_b, 1e-30)

    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = max(float(a.max() - a.min()), 1e-12)
    nrmse = rmse / denom

    return {
        "cosine": cosine,
        "rmse": rmse,
        "nrmse": nrmse,
    }


# ------------------------------------------------------------------------------
# 8. Benchmark / Gain Tracker
# ------------------------------------------------------------------------------

def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_ms(
    fn,
    *,
    device: torch.device,
    warmup: int = 8,
    iterations: int = 30,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync_device(device)

    samples = []
    for _ in range(iterations):
        sync_device(device)
        t0 = time.perf_counter_ns()
        fn()
        sync_device(device)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)

    samples.sort()
    p95_idx = min(len(samples) - 1, int(np.ceil(0.95 * len(samples))) - 1)

    return {
        "mean_ms": float(statistics.mean(samples)),
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(samples[p95_idx]),
        "min_ms": float(samples[0]),
        "max_ms": float(samples[-1]),
    }


def speedup_metrics(baseline_ms: float, candidate_ms: float) -> Dict[str, float]:
    speedup = baseline_ms / max(candidate_ms, 1e-30)
    return {
        "speedup_x": float(speedup),
        "speed_gain_pct": float((speedup - 1.0) * 100.0),
    }


def storage_gain(baseline_bytes: int, candidate_bytes: int) -> Dict[str, float]:
    ratio = baseline_bytes / max(candidate_bytes, 1)
    reduction = 100.0 * (1.0 - candidate_bytes / max(baseline_bytes, 1))
    return {
        "baseline_bytes": int(baseline_bytes),
        "candidate_bytes": int(candidate_bytes),
        "compression_ratio_x": float(ratio),
        "disk_reduction_pct": float(reduction),
    }


def capture_real_qproj_input(
    model,
    tokenizer,
    module_name: str,
    device: torch.device,
    prompt: str,
) -> Tuple[torch.Tensor, str]:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Módulo não encontrado: {module_name}")

    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(_module, args):
        if args:
            captured["x"] = args[0].detach()

    handle = modules[module_name].register_forward_pre_hook(pre_hook)
    try:
        toks = tokenizer(prompt, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            model(**toks)
    finally:
        handle.remove()

    if "x" not in captured:
        raise RuntimeError("Hook não capturou ativação real")

    return captured["x"].detach(), "real_model_activation"



def _json_safe(value: Any) -> Any:
    """Converte numpy/Path/tuplas para tipos JSON nativos."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("._-") or "record"


class ResultsPublishError(RuntimeError):
    """Falha segura e sem vazamento de credenciais durante a publicação."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _read_colab_secret(name: str) -> str | None:
    """Lê um Secret do Colab sem criar dependência obrigatória do Colab."""
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
    except Exception:
        return None
    value = str(value).strip() if value is not None else ""
    return value or None


def _read_setting(name: str) -> str | None:
    """Prioriza variável de ambiente e usa Colab Secrets como fallback."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    return _read_colab_secret(name)


def _running_in_colab() -> bool:
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_GPU")
        or "google.colab" in sys.modules
    )


def _normalize_github_repository(value: str) -> str:
    """Aceita owner/repo ou uma URL GitHub e devolve somente owner/repo."""
    candidate = value.strip().rstrip("/")
    patterns = (
        r"^(?:https?://)?github\.com/([^/]+/[^/]+)$",
        r"^git@github\.com:([^/]+/[^/]+)$",
        r"^ssh://git@github\.com/([^/]+/[^/]+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, candidate, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            break
    if candidate.lower().endswith(".git"):
        candidate = candidate[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate):
        raise ResultsPublishError(
            "Repositório GitHub inválido. Use o formato 'owner/repo'."
        )
    return candidate


def _infer_github_repository() -> str | None:
    configured = _read_setting("RIFT_GITHUB_REPOSITORY")
    if configured:
        return _normalize_github_repository(configured)

    # Funciona quando o script é executado dentro de um clone do dashboard.
    try:
        completed = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        remote = completed.stdout.strip()
        return _normalize_github_repository(remote) if remote else None
    except (OSError, subprocess.SubprocessError, ResultsPublishError):
        return None


def _github_api_request(
    *,
    method: str,
    url: str,
    token: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 45,
) -> Any:
    body = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "rift-lm-colab-publisher/0.3.5",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw).get("message", raw)
        except Exception:
            detail = raw
        raise ResultsPublishError(
            f"GitHub API retornou HTTP {exc.code}: {detail}",
            status=exc.code,
        ) from exc
    except URLError as exc:
        raise ResultsPublishError(
            f"Não foi possível conectar à API do GitHub: {exc.reason}"
        ) from exc


def _validate_battery_history(value: Any, *, source: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ResultsPublishError(
            f"O histórico de baterias em {source} precisa ser um array JSON."
        )
    invalid = [index for index, item in enumerate(value) if not isinstance(item, dict)]
    if invalid:
        raise ResultsPublishError(
            f"O histórico em {source} contém registro inválido no índice {invalid[0]}."
        )
    return value


def _merge_battery_histories(
    remote: List[Dict[str, Any]],
    local: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Faz upsert por run_id+battery_id, preservando o histórico remoto."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    anonymous: List[Dict[str, Any]] = []
    for item in [*remote, *local]:
        run_id = str(item.get("run_id", "")).strip()
        battery_id = str(item.get("battery_id", "")).strip()
        if run_id and battery_id:
            merged[(run_id, battery_id)] = item
        elif item not in anonymous:
            anonymous.append(item)

    result = [*anonymous, *merged.values()]
    result.sort(
        key=lambda item: (
            str(item.get("timestamp_utc", "")),
            str(item.get("run_id", "")),
            str(item.get("battery_id", "")),
        )
    )
    return result


def publish_battery_history_to_github(
    json_path: Path,
    *,
    repository: str,
    token: str,
    branch: str | None = None,
    target_path: str = "data/rift_test_batteries.json",
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Mescla o JSON local no repositório e cria um commit via GitHub Contents API.

    Quando o repositório está conectado ao Vercel, esse commit dispara o deploy
    automático sem colocar credenciais Git dentro do ambiente do Colab.
    """
    repository = _normalize_github_repository(repository)
    target_path = target_path.replace("\\", "/").strip("/")
    if not target_path or any(part in ("", ".", "..") for part in target_path.split("/")):
        raise ResultsPublishError("Caminho de destino do GitHub inválido.")

    try:
        local_value = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultsPublishError(f"Arquivo de baterias não encontrado: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultsPublishError(f"JSON local inválido: {exc}") from exc
    local = _validate_battery_history(local_value, source=str(json_path))

    api_root = f"https://api.github.com/repos/{repository}"
    repo_info = _github_api_request(
        method="GET",
        url=api_root,
        token=token,
    )
    target_branch = (branch or repo_info.get("default_branch") or "main").strip()
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", target_branch) or ".." in target_branch:
        raise ResultsPublishError("Branch GitHub inválida.")

    content_url = f"{api_root}/contents/{quote(target_path, safe='/')}"
    for attempt in range(1, retries + 1):
        sha = None
        remote: List[Dict[str, Any]] = []
        try:
            current = _github_api_request(
                method="GET",
                url=f"{content_url}?{urlencode({'ref': target_branch})}",
                token=token,
            )
            sha = current.get("sha")
            if current.get("encoding") != "base64" or not isinstance(current.get("content"), str):
                raise ResultsPublishError(
                    "O arquivo remoto não foi retornado como conteúdo base64."
                )
            decoded = base64.b64decode(current["content"], validate=False).decode("utf-8")
            remote = _validate_battery_history(
                json.loads(decoded),
                source=f"GitHub:{target_path}",
            )
        except ResultsPublishError as exc:
            if exc.status != 404:
                raise
            # Arquivo ainda inexistente: o PUT abaixo irá criá-lo.
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultsPublishError(
                f"O histórico remoto não é um JSON UTF-8 válido: {exc}"
            ) from exc

        merged = _merge_battery_histories(remote, local)
        run_ids = sorted({str(item.get("run_id")) for item in local if item.get("run_id")})
        run_label = ", ".join(run_ids[:2]) or "sem-run-id"
        if len(run_ids) > 2:
            run_label += f" (+{len(run_ids) - 2})"
        payload: Dict[str, Any] = {
            "message": f"data: publish RIFT-LM results {run_label}",
            "content": base64.b64encode(
                (json.dumps(merged, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            ).decode("ascii"),
            "branch": target_branch,
        }
        if sha:
            payload["sha"] = sha

        try:
            response = _github_api_request(
                method="PUT",
                url=content_url,
                token=token,
                payload=payload,
            )
            return {
                "repository": repository,
                "branch": target_branch,
                "path": target_path,
                "records": len(merged),
                "commit_sha": response.get("commit", {}).get("sha"),
                "commit_url": response.get("commit", {}).get("html_url"),
            }
        except ResultsPublishError as exc:
            if exc.status not in (409, 422) or attempt == retries:
                raise
            print(
                f"[PUBLISH] Conflito de atualização; nova tentativa "
                f"{attempt + 1}/{retries}..."
            )

    raise ResultsPublishError("Não foi possível publicar após as tentativas previstas.")


def trigger_vercel_deploy_hook(hook_url: str) -> Dict[str, Any]:
    """Dispara opcionalmente um Deploy Hook sem expor sua URL no código."""
    if not re.fullmatch(r"https://api\.vercel\.com/v1/integrations/deploy/[^\s]+", hook_url):
        raise ResultsPublishError("RIFT_VERCEL_DEPLOY_HOOK_URL não é uma URL Vercel válida.")
    request = Request(
        hook_url,
        data=b"{}",
        headers={"Content-Type": "application/json", "User-Agent": "rift-lm-colab-publisher/0.3.5"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ResultsPublishError(
            f"Deploy Hook da Vercel retornou HTTP {exc.code}: {detail}",
            status=exc.code,
        ) from exc
    except URLError as exc:
        raise ResultsPublishError(
            f"Não foi possível acionar o Deploy Hook da Vercel: {exc.reason}"
        ) from exc


def publish_battery_history_to_vercel(
    json_path: Path,
    *,
    endpoint: str,
    ingest_token: str,
) -> Dict[str, Any]:
    """Envia o histórico para a Function /api/results protegida por Bearer token."""
    parsed = urlparse(endpoint.strip())
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ResultsPublishError(
            "RIFT_RESULTS_ENDPOINT precisa ser uma URL HTTPS válida da Vercel."
        )
    if len(ingest_token) < 32:
        raise ResultsPublishError("RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")

    try:
        local_value = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultsPublishError(f"Arquivo de baterias não encontrado: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultsPublishError(f"JSON local inválido: {exc}") from exc
    local = _validate_battery_history(local_value, source=str(json_path))

    body = json.dumps({"records": local}, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint.strip(),
        data=body,
        headers={
            "Authorization": f"Bearer {ingest_token}",
            "Content-Type": "application/json",
            "User-Agent": "rift-lm-colab-publisher/0.3.5",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw).get("error", raw)
        except Exception:
            detail = raw
        raise ResultsPublishError(
            f"API de resultados da Vercel retornou HTTP {exc.code}: {detail}",
            status=exc.code,
        ) from exc
    except URLError as exc:
        raise ResultsPublishError(
            f"Não foi possível conectar à API de resultados da Vercel: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ResultsPublishError(
            f"A API de resultados da Vercel retornou JSON inválido: {exc}"
        ) from exc

    if not payload.get("ok") or not isinstance(payload.get("publication"), dict):
        raise ResultsPublishError(
            f"A API da Vercel recusou a publicação: {payload.get('error', 'resposta inválida')}"
        )
    return payload["publication"]


def publish_results_if_configured(
    json_path: Path,
    *,
    mode: str,
    results_endpoint: str | None,
    repository: str | None,
    branch: str | None,
    target_path: str,
) -> Dict[str, Any] | None:
    if mode == "off":
        print("[PUBLISH] Publicação remota desativada (--publish off).")
        return None

    endpoint = results_endpoint or _read_setting("RIFT_RESULTS_ENDPOINT")
    ingest_token = _read_setting("RIFT_INGEST_TOKEN")
    required = mode == "required" or (mode == "auto" and _running_in_colab())

    # Fluxo recomendado: o PAT do GitHub fica apenas no Vercel. O Colab conhece
    # somente o endpoint e uma chave de ingestão independente e revogável.
    if endpoint or ingest_token:
        missing_vercel = []
        if not endpoint:
            missing_vercel.append("RIFT_RESULTS_ENDPOINT ou --results-endpoint")
        if not ingest_token:
            missing_vercel.append("Secret RIFT_INGEST_TOKEN")
        if missing_vercel:
            message = "Configuração da API Vercel incompleta: " + "; ".join(missing_vercel)
            if required:
                raise ResultsPublishError(message)
            print(f"[PUBLISH] AVISO: {message}. Resultado mantido apenas localmente.")
            return None

        result = publish_battery_history_to_vercel(
            json_path,
            endpoint=endpoint,
            ingest_token=ingest_token,
        )
        print(
            f"[PUBLISH] {result['records']} registro(s) aceito(s) pela Vercel e "
            f"publicado(s) em {result['repository']}:{result['branch']}/{result['path']}"
        )
        if result.get("commit_url"):
            print(f"[PUBLISH] Commit: {result['commit_url']}")
        return result

    resolved_repository = (
        _normalize_github_repository(repository) if repository else _infer_github_repository()
    )
    token = _read_setting("RIFT_GITHUB_TOKEN") or _read_setting("GITHUB_TOKEN")
    missing = []
    if not resolved_repository:
        missing.append("RIFT_GITHUB_REPOSITORY ou --github-repo owner/repo")
    if not token:
        missing.append("Secret RIFT_GITHUB_TOKEN")

    # No Colab, concluir sem publicar seria um falso sucesso. Localmente, o modo
    # auto continua útil para desenvolvimento sem exigir credenciais.
    if missing:
        message = "Configuração de publicação ausente: " + "; ".join(missing)
        if required:
            raise ResultsPublishError(message)
        print(f"[PUBLISH] AVISO: {message}. Resultado mantido apenas localmente.")
        return None

    result = publish_battery_history_to_github(
        json_path,
        repository=resolved_repository,
        token=token,
        branch=branch or _read_setting("RIFT_GITHUB_BRANCH"),
        target_path=target_path,
    )
    print(
        f"[PUBLISH] {result['records']} registro(s) publicado(s) em "
        f"{result['repository']}:{result['branch']}/{result['path']}"
    )
    if result.get("commit_url"):
        print(f"[PUBLISH] Commit: {result['commit_url']}")

    deploy_hook = _read_setting("RIFT_VERCEL_DEPLOY_HOOK_URL")
    if deploy_hook:
        deploy = trigger_vercel_deploy_hook(deploy_hook)
        job = deploy.get("job", {}) if isinstance(deploy, dict) else {}
        print(f"[PUBLISH] Deploy Hook Vercel acionado: {job.get('state', 'solicitado')}")
    else:
        print(
            "[PUBLISH] O commit acionará o deploy automático se o repositório "
            "estiver conectado ao Vercel."
        )
    return result


def _pct_gain_higher_is_better(baseline: float | None, rift: float | None) -> float | None:
    if baseline is None or rift is None or baseline == 0:
        return None
    return float((rift / baseline - 1.0) * 100.0)


def _pct_reduction_lower_is_better(baseline: float | None, rift: float | None) -> float | None:
    if baseline is None or rift is None or baseline == 0:
        return None
    return float((1.0 - rift / baseline) * 100.0)


def _ratio_baseline_over_rift(baseline: float | None, rift: float | None) -> float | None:
    if baseline is None or rift is None or rift == 0:
        return None
    return float(baseline / rift)


class BatteryRecorder:
    """
    Persistência automática das baterias para o RIFT Test Observatory.

    Arquivos gerados:
      <out>/rift_test_batteries.json
      <out>/rift_test_batteries.csv
      <out>/batteries/<run_id>__<battery_id>.json

    Regras:
      - nunca inventa Tok/s;
      - RAM e espaço precisam declarar seu measurement_scope;
      - regressões são persistidas como ganho negativo;
      - cada bateria é autocontida e também entra no histórico consolidado.
    """

    CSV_FIELDS = [
        "timestamp_utc",
        "run_id",
        "spec",
        "model_id",
        "battery_id",
        "status",
        "baseline_tok_s",
        "rift_tok_s",
        "tok_s_gain_pct",
        "baseline_ram_bytes",
        "rift_ram_bytes",
        "ram_reduction_pct",
        "baseline_disk_bytes",
        "rift_disk_bytes",
        "disk_reduction_pct",
        "disk_compression_ratio_x",
        "overall_gain_pct",
        "measurement_scope",
        "quality_gate_pass",
        "baseline_operation_ms",
        "rift_operation_ms",
        "operation_speedup_x",
        "notes",
    ]

    def __init__(
        self,
        out_dir: Path,
        *,
        model_id: str,
        run_id: str | None = None,
        spec: str = "RIFT-LM v0.3.5",
        technology: str = "RIFT",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = self.out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)

        self.model_id = model_id
        self.spec = spec
        self.technology = technology
        self.run_id = run_id or (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + uuid.uuid4().hex[:8]
        )

        self.json_path = self.out_dir / "rift_test_batteries.json"
        self.csv_path = self.out_dir / "rift_test_batteries.csv"

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.json_path.exists():
            return []
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            # Não destrói histórico corrompido silenciosamente.
            backup = self.json_path.with_suffix(
                f".corrupt-{int(time.time())}.json"
            )
            self.json_path.replace(backup)
            return []

    def record(
        self,
        *,
        battery_id: str,
        status: str,
        baseline_tok_s: float | None = None,
        rift_tok_s: float | None = None,
        baseline_ram_bytes: int | None = None,
        rift_ram_bytes: int | None = None,
        baseline_disk_bytes: int | None = None,
        rift_disk_bytes: int | None = None,
        measurement_scope: str,
        quality: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        notes: str = "",
        comparison_role: str | None = None,
    ) -> Dict[str, Any]:

        tok_gain = _pct_gain_higher_is_better(baseline_tok_s, rift_tok_s)
        ram_gain = _pct_reduction_lower_is_better(
            baseline_ram_bytes, rift_ram_bytes
        )
        disk_gain = _pct_reduction_lower_is_better(
            baseline_disk_bytes, rift_disk_bytes
        )
        disk_ratio = _ratio_baseline_over_rift(
            baseline_disk_bytes, rift_disk_bytes
        )

        available = [x for x in (tok_gain, ram_gain, disk_gain) if x is not None]
        overall = float(sum(available) / len(available)) if available else None

        metrics = metrics or {}
        operation = metrics.get("operation", {}) if isinstance(metrics, dict) else {}

        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "spec": self.spec,
            "technology": self.technology,
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            "comparison_role": comparison_role,

            # Campos consumidos diretamente pelo dashboard:
            "baseline_tok_s": baseline_tok_s,
            "rift_tok_s": rift_tok_s,
            "baseline_ram_bytes": baseline_ram_bytes,
            "rift_ram_bytes": rift_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes,
            "rift_disk_bytes": rift_disk_bytes,
            # Aliases neutros permitem comparar tecnologias no mesmo histórico.
            "candidate_tok_s": rift_tok_s,
            "candidate_ram_bytes": rift_ram_bytes,
            "candidate_disk_bytes": rift_disk_bytes,

            "gains": {
                "tok_s_gain_pct": tok_gain,
                "ram_reduction_pct": ram_gain,
                "disk_reduction_pct": disk_gain,
                "disk_compression_ratio_x": disk_ratio,
                "overall_gain_pct": overall,
            },

            "measurement_scope": measurement_scope,
            "quality": quality or {},
            "metrics": metrics,
            "notes": notes,
        }

        record = _json_safe(record)

        # JSON individual da bateria
        single_name = (
            f"{_safe_slug(self.run_id)}__{_safe_slug(battery_id)}.json"
        )
        (self.batteries_dir / single_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Histórico JSON consolidado
        history = self._load_history()
        # upsert por run_id+battery_id para rerun dentro da mesma execução
        history = [
            r for r in history
            if not (
                r.get("run_id") == self.run_id
                and r.get("battery_id") == battery_id
            )
        ]
        history.append(record)
        history.sort(key=lambda r: (str(r.get("timestamp_utc", "")),
                                    str(r.get("run_id", "")),
                                    str(r.get("battery_id", ""))))
        self.json_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Histórico CSV consolidado
        csv_row = {
            "timestamp_utc": record["timestamp_utc"],
            "run_id": record["run_id"],
            "spec": record["spec"],
            "model_id": record["model_id"],
            "battery_id": record["battery_id"],
            "status": record["status"],
            "baseline_tok_s": record["baseline_tok_s"],
            "rift_tok_s": record["rift_tok_s"],
            "tok_s_gain_pct": tok_gain,
            "baseline_ram_bytes": record["baseline_ram_bytes"],
            "rift_ram_bytes": record["rift_ram_bytes"],
            "ram_reduction_pct": ram_gain,
            "baseline_disk_bytes": record["baseline_disk_bytes"],
            "rift_disk_bytes": record["rift_disk_bytes"],
            "disk_reduction_pct": disk_gain,
            "disk_compression_ratio_x": disk_ratio,
            "overall_gain_pct": overall,
            "measurement_scope": measurement_scope,
            "quality_gate_pass": (quality or {}).get("full_local_gate_pass"),
            "baseline_operation_ms": operation.get("baseline_median_ms"),
            "rift_operation_ms": operation.get("rift_median_ms"),
            "operation_speedup_x": operation.get("speedup_x"),
            "notes": notes,
        }

        # Reescreve CSV a partir do JSON para manter upsert determinístico.
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for item in history:
                gains = item.get("gains", {})
                op = item.get("metrics", {}).get("operation", {})
                q = item.get("quality", {})
                writer.writerow({
                    "timestamp_utc": item.get("timestamp_utc"),
                    "run_id": item.get("run_id"),
                    "spec": item.get("spec"),
                    "model_id": item.get("model_id"),
                    "battery_id": item.get("battery_id"),
                    "status": item.get("status"),
                    "baseline_tok_s": item.get("baseline_tok_s"),
                    "rift_tok_s": item.get("rift_tok_s"),
                    "tok_s_gain_pct": gains.get("tok_s_gain_pct"),
                    "baseline_ram_bytes": item.get("baseline_ram_bytes"),
                    "rift_ram_bytes": item.get("rift_ram_bytes"),
                    "ram_reduction_pct": gains.get("ram_reduction_pct"),
                    "baseline_disk_bytes": item.get("baseline_disk_bytes"),
                    "rift_disk_bytes": item.get("rift_disk_bytes"),
                    "disk_reduction_pct": gains.get("disk_reduction_pct"),
                    "disk_compression_ratio_x": gains.get("disk_compression_ratio_x"),
                    "overall_gain_pct": gains.get("overall_gain_pct"),
                    "measurement_scope": item.get("measurement_scope"),
                    "quality_gate_pass": q.get("full_local_gate_pass"),
                    "baseline_operation_ms": op.get("baseline_median_ms"),
                    "rift_operation_ms": op.get("rift_median_ms"),
                    "operation_speedup_x": op.get("speedup_x"),
                    "notes": item.get("notes", ""),
                })

        print(
            f"[BATTERY] {battery_id}: gravada automaticamente -> "
            f"{self.batteries_dir / single_name}"
        )
        # Publica imediatamente no site (não espera o fim de todas as baterias)
        try:
            endpoint = _read_setting("RIFT_RESULTS_ENDPOINT")
            token = _read_setting("RIFT_INGEST_TOKEN")
            if endpoint and token and len(token) >= 32:
                publish_battery_history_to_vercel(
                    self.json_path,
                    endpoint=endpoint,
                    ingest_token=token,
                )
        except ResultsPublishError as exc:
            print(f"[PUBLISH] AVISO (incremental): {exc}")
        except Exception as exc:
            print(f"[PUBLISH] AVISO (incremental): {exc}")
        return record


def estimated_linear_working_set_bytes(
    *,
    weight_bytes: int,
    input_bytes: int,
    output_bytes: int,
    packed_bytes: int = 0,
    reconstructed_weight_bytes: int = 0,
) -> int:
    """
    Estimativa explícita do working set do teste Linear.

    No reference path atual o RIFT ainda reconstrói FP32, então isso pode mostrar
    RAM igual ou PIOR que o baseline. Isso é intencional e honesto.
    """
    return int(
        weight_bytes
        + input_bytes
        + output_bytes
        + packed_bytes
        + reconstructed_weight_bytes
    )


def append_gain_history(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    flat = {
        "timestamp_utc": row["timestamp_utc"],
        "model_id": row["model_id"],
        "tensor": row["tensor"],
        "source_dtype": row["source_dtype"],
        "source_tensor_bytes": row["storage"]["source_tensor_bytes"],
        "rift_base_payload_bytes": row["storage"]["base"]["candidate_bytes"],
        "rift_full_payload_bytes": row["storage"]["full"]["candidate_bytes"],
        "base_disk_reduction_pct": row["storage"]["base"]["disk_reduction_pct"],
        "full_disk_reduction_pct": row["storage"]["full"]["disk_reduction_pct"],
        "baseline_median_ms": row["performance"]["baseline_fp32_matmul"]["median_ms"],
        "rift_full_predecoded_median_ms": row["performance"]["rift_full_predecoded"]["median_ms"],
        "rift_full_reference_path_median_ms": row["performance"]["rift_full_decode_plus_matmul"]["median_ms"],
        "predecoded_speedup_x": row["performance"]["rift_full_predecoded_vs_baseline"]["speedup_x"],
        "reference_path_speedup_x": row["performance"]["rift_full_reference_vs_baseline"]["speedup_x"],
        "weight_cosine_full": row["quality"]["weight_full"]["cosine"],
        "weight_nrmse_full": row["quality"]["weight_full"]["nrmse"],
        "output_cosine_full": row["quality"]["output_full"]["cosine"],
        "output_nrmse_full": row["quality"]["output_full"]["nrmse"],
    }

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


# ------------------------------------------------------------------------------
# 9. B0 runner
# ------------------------------------------------------------------------------

def run_b0(out_dir: Path, valid_ir: Dict[str, Any], one_tensor: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("CHECKPOINT B0 — Binary/IR Foundation")
    print("=" * 78)

    # B0.1 Header + golden checksum
    golden_header = test_golden_header()

    # B0.4/5 IR validator
    validate_rift_ir_m0(valid_ir)
    ir_path = out_dir / "model.riftir.json"
    ir_path.write_text(json.dumps(valid_ir, indent=2), encoding="utf-8")
    validate_rift_ir_m0(json.loads(ir_path.read_text(encoding="utf-8")))
    print("[B0.4/B0.5] RIFT-IR M0 + DAG validator PASS")

    # B0.6 Golden fixtures
    expected = write_golden_files(out_dir, valid_ir)
    run_negative_golden_tests(out_dir, expected)

    # B0.7 one-tensor + mmap
    container_path = out_dir / "container_m0_one_tensor.rift"
    write_meta = write_one_tensor_container_m0(container_path, one_tensor)
    loaded, read_meta = read_one_tensor_container_m0(container_path)

    if not np.array_equal(np.asarray(one_tensor, dtype=np.float32), loaded):
        raise AssertionError("Container one-tensor não fez round-trip bit-exact FP32")

    print(
        "[B0.7] ONE-TENSOR + MMAP PASS — "
        f"container={container_path.stat().st_size:,} bytes"
    )

    # Importante: Golden Files e Page/Tensor ABI final ainda dependem das lacunas
    # explicitamente marcadas como TEST-LOCAL neste script.
    result = {
        "b0_passed_reference": True,
        "golden_checksum": f"{EXPECTED_GOLDEN_CHECKSUM:#018x}",
        "container_path": str(container_path),
        "container": {**write_meta, **read_meta},
        "note": (
            "Reference B0 passou. PageEntry binary size, codec IDs e Shape Blob "
            "layout usados aqui são TEST-LOCAL até serem congelados na especificação."
        ),
    }
    return result


# ------------------------------------------------------------------------------
# 10. Phase 1 experimental runner
# ------------------------------------------------------------------------------



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



def run_phase1(
    *,
    out_dir: Path,
    model_id: str,
    target_layer_name: str,
    prompt: str,
    iterations: int,
    device_str: str,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    ensure_phase1_dependencies()
    device = resolve_torch_device(device_str)
    model_id = normalize_huggingface_model_id(model_id)

    print(f"\n[Phase1] Carregando {model_id} em {device}...")
    hf_token = _read_setting("HF_TOKEN")
    tokenizer = load_tokenizer(
        model_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )

    # Em CUDA carregamos o checkpoint em FP16 para permitir modelos maiores;
    # o tensor alvo e todas as métricas continuam convertidos a FP32.
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
    model = None
    errors = []
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[load] Modelo carregado via {cls.__name__}")
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
            print(f"[load] {cls.__name__} falhou: {exc}")
    if model is None:
        raise RuntimeError(
            "Não foi possível carregar o modelo.\n"
            "Gemma 4 Unified exige transformers recente (AutoModelForMultimodalLM).\n"
            + "\n".join(errors)
        )

    state = model.state_dict()
    target_layer_name = resolve_linear_weight_name(model, target_layer_name)
    print(f"[Phase1] Camada Linear selecionada: {target_layer_name}")

    weight_t = state[target_layer_name].detach().cpu().float().contiguous()
    W = weight_t.numpy()
    source_tensor_bytes = int(W.nbytes)

    valid_ir = build_linear_ir_m0(
        model_id=model_id,
        architecture=str(getattr(model.config, "model_type", "unknown")),
        weight_name=target_layer_name,
        weight_shape=W.shape,
    )

    recorder = BatteryRecorder(
        out_dir,
        model_id=model_id,
    )

    # B0 HARD GATE. Codecs só rodam após este retorno sem exceção.
    b0 = run_b0(out_dir, valid_ir, W)

    recorder.record(
        battery_id="B0_BINARY_IR_FOUNDATION",
        status="PASS" if b0["b0_passed_reference"] else "FAIL",
        baseline_tok_s=None,
        rift_tok_s=None,
        baseline_ram_bytes=None,
        rift_ram_bytes=None,
        baseline_disk_bytes=source_tensor_bytes,
        rift_disk_bytes=int(b0["container"]["file_size"]),
        measurement_scope=(
            "B0 binary/container validation. Tok/s e RAM não se aplicam; "
            "disk compara tensor FP32 raw com Container M0 raw one-tensor."
        ),
        quality={"full_local_gate_pass": bool(b0["b0_passed_reference"])},
        metrics={
            "container": b0["container"],
            "golden_checksum": b0["golden_checksum"],
        },
        notes=(
            "B0 mede correção/overhead do container, não compressão. "
            "Ganho de disco pode ser negativo por design."
        ),
    )

    if not b0["b0_passed_reference"]:
        raise RuntimeError("B0 não aprovado; abortando codecs por contrato v0.3.5")

    print("\n" + "=" * 78)
    print("PHASE 1 EXPERIMENTAL — Progressive Precision + Gain Tracker")
    print("=" * 78)
    print("ATENÇÃO: Q4_LINEAR_TEST != MXFP4; Python path != native low-bit kernel.")

    codec = quantize_q4_linear_test(W)
    W_base = decode_q4_linear_test(codec, use_refinement=False)
    W_full = decode_q4_linear_test(codec, use_refinement=True)

    payload_sizes = write_q4_linear_payloads(out_dir, codec)

    # Raw FP32 baseline file, para comparação de espaço em disco byte-a-byte.
    raw_path = out_dir / "baseline_fp32_tensor.raw"
    raw_path.write_bytes(W.tobytes(order="C"))
    assert raw_path.stat().st_size == source_tensor_bytes

    # Weight quality
    q_weight_base = compute_metrics(W, W_base)
    q_weight_full = compute_metrics(W, W_full)

    # Captura ativação real do módulo q_proj.
    module_name = target_layer_name.removesuffix(".weight")
    try:
        x_real, activation_source = capture_real_qproj_input(
            model, tokenizer, module_name, device, prompt
        )
        x = x_real.detach().to(device=device, dtype=torch.float32)
    except Exception as exc:
        print(f"[WARN] Falha ao capturar ativação real: {exc}")
        print("[WARN] Usando ativação sintética determinística; relatório marcará isso.")
        torch.manual_seed(1234)
        in_features = W.shape[1]
        x = torch.randn(8, in_features, device=device, dtype=torch.float32)
        activation_source = "synthetic_fallback"

    # Flatten batch/sequence para benchmark de linear.
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    W_ref_t = torch.from_numpy(W).to(device=device, dtype=torch.float32)
    W_base_t = torch.from_numpy(W_base).to(device=device, dtype=torch.float32)
    W_full_t = torch.from_numpy(W_full).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        Y_ref = F.linear(x2d, W_ref_t).detach().cpu().numpy()
        Y_base = F.linear(x2d, W_base_t).detach().cpu().numpy()
        Y_full = F.linear(x2d, W_full_t).detach().cpu().numpy()

    q_output_base = compute_metrics(Y_ref, Y_base)
    q_output_full = compute_metrics(Y_ref, Y_full)

    # -------------------------
    # Performance measurements
    # -------------------------
    baseline_perf = benchmark_ms(
        lambda: F.linear(x2d, W_ref_t),
        device=device,
        iterations=iterations,
    )

    base_predecoded_perf = benchmark_ms(
        lambda: F.linear(x2d, W_base_t),
        device=device,
        iterations=iterations,
    )
    full_predecoded_perf = benchmark_ms(
        lambda: F.linear(x2d, W_full_t),
        device=device,
        iterations=iterations,
    )

    # Reference path: inclui unpack + decode + cópia para tensor + GEMM.
    # É propositalmente honesto; NÃO representa kernel fused futuro.
    def base_reference_path():
        w_np = decode_q4_linear_test(codec, use_refinement=False)
        w_t = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
        return F.linear(x2d, w_t)

    def full_reference_path():
        w_np = decode_q4_linear_test(codec, use_refinement=True)
        w_t = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
        return F.linear(x2d, w_t)

    base_reference_perf = benchmark_ms(
        base_reference_path, device=device, iterations=max(5, iterations // 3)
    )
    full_reference_perf = benchmark_ms(
        full_reference_path, device=device, iterations=max(5, iterations // 3)
    )

    # -------------------------
    # Storage gains
    # -------------------------
    storage_base = storage_gain(
        source_tensor_bytes,
        payload_sizes["base_payload_disk_bytes"],
    )
    storage_full = storage_gain(
        source_tensor_bytes,
        payload_sizes["full_payload_disk_bytes"],
    )

    # Container M0 raw reference is not compressed; track overhead separately.
    container_disk_bytes = int(
        (out_dir / "container_m0_one_tensor.rift").stat().st_size
    )

    perf = {
        "baseline_fp32_matmul": baseline_perf,
        "rift_base_predecoded": base_predecoded_perf,
        "rift_full_predecoded": full_predecoded_perf,
        "rift_base_decode_plus_matmul": base_reference_perf,
        "rift_full_decode_plus_matmul": full_reference_perf,
        "rift_base_predecoded_vs_baseline": speedup_metrics(
            baseline_perf["median_ms"], base_predecoded_perf["median_ms"]
        ),
        "rift_full_predecoded_vs_baseline": speedup_metrics(
            baseline_perf["median_ms"], full_predecoded_perf["median_ms"]
        ),
        "rift_base_reference_vs_baseline": speedup_metrics(
            baseline_perf["median_ms"], base_reference_perf["median_ms"]
        ),
        "rift_full_reference_vs_baseline": speedup_metrics(
            baseline_perf["median_ms"], full_reference_perf["median_ms"]
        ),
        "native_lowbit_kernel_speedup": None,
        "native_lowbit_kernel_status": "NOT_IMPLEMENTED_PHASE1_REFERENCE",
    }

    # ------------------------------------------------------------------
    # RAM requirement — escopo desta bateria
    # ------------------------------------------------------------------
    # No protótipo de referência o peso RIFT é reconstruído para FP32 antes
    # da GEMM. Por isso medimos o working set REALÍSTICO DESTE PATH atual:
    #   baseline = W_FP32 + X + Y
    #   RIFT     = packed streams + W_reconstructed_FP32 + X + Y
    #
    # O futuro kernel low-bit fused deverá reduzir esse valor; enquanto não
    # existir, o dashboard deve mostrar ausência de ganho ou regressão.
    input_bytes = int(x2d.numel() * x2d.element_size())
    output_bytes = int(Y_ref.nbytes)

    baseline_ram_working_set = estimated_linear_working_set_bytes(
        weight_bytes=source_tensor_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )

    base_packed_bytes = int(
        payload_sizes["codec_header_bytes"]
        + payload_sizes["base_bitstream_bytes"]
    )
    full_packed_bytes = int(
        payload_sizes["codec_header_bytes"]
        + payload_sizes["base_bitstream_bytes"]
        + payload_sizes["refinement_bitstream_bytes"]
    )

    # weight_bytes=0 porque o candidato persistido é o stream packed;
    # reconstructed_weight_bytes representa a materialização FP32 transitória.
    base_ram_working_set = estimated_linear_working_set_bytes(
        weight_bytes=0,
        packed_bytes=base_packed_bytes,
        reconstructed_weight_bytes=int(W_base.nbytes),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )
    full_ram_working_set = estimated_linear_working_set_bytes(
        weight_bytes=0,
        packed_bytes=full_packed_bytes,
        reconstructed_weight_bytes=int(W_full.nbytes),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )

    # ------------------------------------------------------------------
    # Baterias automáticas
    # ------------------------------------------------------------------
    base_speedup = speedup_metrics(
        baseline_perf["median_ms"], base_reference_perf["median_ms"]
    )
    full_speedup = speedup_metrics(
        baseline_perf["median_ms"], full_reference_perf["median_ms"]
    )

    recorder.record(
        battery_id="P1_Q4_LINEAR_BASE_2BIT",
        status=(
            "PASS"
            if q_weight_base["cosine"] >= 0.995
            and q_weight_base["nrmse"] <= 0.05
            else "EXPERIMENTAL_FAIL"
        ),
        baseline_tok_s=None,
        rift_tok_s=None,
        baseline_ram_bytes=baseline_ram_working_set,
        rift_ram_bytes=base_ram_working_set,
        baseline_disk_bytes=source_tensor_bytes,
        rift_disk_bytes=int(payload_sizes["base_payload_disk_bytes"]),
        measurement_scope=(
            "Single Linear op. RAM=estimated current reference working set; "
            "disk=real bytes on disk; Tok/s=model-level NOT_MEASURED."
        ),
        quality={
            "full_local_gate_pass": bool(
                q_weight_base["cosine"] >= 0.995
                and q_weight_base["nrmse"] <= 0.05
            ),
            "weight": q_weight_base,
            "output": q_output_base,
        },
        metrics={
            "operation": {
                "metric": "linear_latency",
                "baseline_median_ms": baseline_perf["median_ms"],
                "rift_median_ms": base_reference_perf["median_ms"],
                "speedup_x": base_speedup["speedup_x"],
                "speed_gain_pct": base_speedup["speed_gain_pct"],
                "rows_processed": int(x2d.shape[0]),
                "baseline_rows_s": float(
                    x2d.shape[0] / (baseline_perf["median_ms"] / 1000.0)
                ),
                "rift_rows_s": float(
                    x2d.shape[0] / (base_reference_perf["median_ms"] / 1000.0)
                ),
                "IMPORTANT": "rows/s de uma única Linear NÃO é model Tok/s",
            },
            "storage": storage_base,
            "ram": {
                "baseline_working_set_bytes": baseline_ram_working_set,
                "rift_working_set_bytes": base_ram_working_set,
                "includes_reconstructed_fp32_weight": True,
            },
        },
        notes=(
            "BASE 2-bit packed. Q4_LINEAR_TEST não é MXFP4. "
            "Speedup nativo não pode ser reivindicado."
        ),
    )

    recorder.record(
        battery_id="P1_Q4_LINEAR_BASE_PLUS_REF_4BIT",
        status=(
            "PASS"
            if q_weight_full["cosine"] >= 0.995
            and q_weight_full["nrmse"] <= 0.05
            else "EXPERIMENTAL_FAIL"
        ),
        baseline_tok_s=None,
        rift_tok_s=None,
        baseline_ram_bytes=baseline_ram_working_set,
        rift_ram_bytes=full_ram_working_set,
        baseline_disk_bytes=source_tensor_bytes,
        rift_disk_bytes=int(payload_sizes["full_payload_disk_bytes"]),
        measurement_scope=(
            "Single Linear op. RAM=estimated current reference working set; "
            "disk=real bytes on disk; Tok/s=model-level NOT_MEASURED."
        ),
        quality={
            "full_local_gate_pass": bool(
                q_weight_full["cosine"] >= 0.995
                and q_weight_full["nrmse"] <= 0.05
            ),
            "weight": q_weight_full,
            "output": q_output_full,
        },
        metrics={
            "operation": {
                "metric": "linear_latency",
                "baseline_median_ms": baseline_perf["median_ms"],
                "rift_median_ms": full_reference_perf["median_ms"],
                "speedup_x": full_speedup["speedup_x"],
                "speed_gain_pct": full_speedup["speed_gain_pct"],
                "rows_processed": int(x2d.shape[0]),
                "baseline_rows_s": float(
                    x2d.shape[0] / (baseline_perf["median_ms"] / 1000.0)
                ),
                "rift_rows_s": float(
                    x2d.shape[0] / (full_reference_perf["median_ms"] / 1000.0)
                ),
                "IMPORTANT": "rows/s de uma única Linear NÃO é model Tok/s",
            },
            "storage": storage_full,
            "ram": {
                "baseline_working_set_bytes": baseline_ram_working_set,
                "rift_working_set_bytes": full_ram_working_set,
                "includes_reconstructed_fp32_weight": True,
            },
        },
        notes=(
            "BASE+REF 4-bit packed. Q4_LINEAR_TEST não é MXFP4. "
            "O path ainda materializa FP32 antes da GEMM."
        ),
        comparison_role="primary",
    )

    report = {
        "spec": "RIFT-LM v0.3.5",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "tensor": target_layer_name,
        "shape": list(W.shape),
        "source_dtype": "FP32_REFERENCE_LOAD",
        "activation_source": activation_source,
        "device": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "b0": b0,
        "quality": {
            "experimental_thresholds": {
                "weight_cosine_min": 0.995,
                "weight_nrmse_max": 0.05,
                "product_contract": False,
            },
            "weight_base": q_weight_base,
            "weight_full": q_weight_full,
            "output_base": q_output_base,
            "output_full": q_output_full,
            "full_local_gate_pass": bool(
                q_weight_full["cosine"] >= 0.995
                and q_weight_full["nrmse"] <= 0.05
            ),
        },
        "memory": {
            "measurement_scope": "single_linear_reference_working_set_estimate",
            "baseline_ram_bytes": baseline_ram_working_set,
            "rift_base_ram_bytes": base_ram_working_set,
            "rift_full_ram_bytes": full_ram_working_set,
            "includes_reconstructed_fp32_weight": True,
        },
        "storage": {
            "source_tensor_bytes": source_tensor_bytes,
            "baseline_raw_file_bytes": int(raw_path.stat().st_size),
            "base": storage_base,
            "full": storage_full,
            "progressive_streams": payload_sizes,
            "b0_raw_container_bytes": container_disk_bytes,
            "note": (
                "Disk gain usa bytes reais do payload experimental e baseline "
                "FP32 raw. Container M0 raw é reportado separadamente."
            ),
        },
        "performance": perf,
        "battery_recorder": {
            "run_id": recorder.run_id,
            "history_json": str(recorder.json_path),
            "history_csv": str(recorder.csv_path),
            "batteries_dir": str(recorder.batteries_dir),
        },
        "interpretation": {
            "reference_path": True,
            "speed_claim_allowed": False,
            "reason": (
                "Ainda não existe kernel native INT2/bitplane fused. "
                "Speedup do Python reference path mede o protótipo atual, "
                "não o potencial do RIFT."
            ),
        },
    }

    # Persist report
    report_path = out_dir / "rift_phase1_gain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    append_gain_history(out_dir / "rift_phase1_gain_history.csv", report)

    # Console dashboard
    print("\n" + "=" * 78)
    print("RIFT PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer_name}")
    print(f"Shape                   : {tuple(W.shape)}")
    print(f"Ativação                : {activation_source}")
    print("-" * 78)
    print("ESPAÇO EM DISCO / BYTES")
    print(
        f"Baseline FP32 raw        : {source_tensor_bytes:,} bytes"
    )
    print(
        f"BASE packed 2-bit        : {storage_base['candidate_bytes']:,} bytes "
        f"| redução {storage_base['disk_reduction_pct']:.2f}% "
        f"| {storage_base['compression_ratio_x']:.2f}x menor"
    )
    print(
        f"BASE+REF packed 4-bit    : {storage_full['candidate_bytes']:,} bytes "
        f"| redução {storage_full['disk_reduction_pct']:.2f}% "
        f"| {storage_full['compression_ratio_x']:.2f}x menor"
    )
    print("-" * 78)
    print("QUALIDADE — PESOS")
    print(
        f"BASE cosine/NRMSE        : "
        f"{q_weight_base['cosine']:.6f} / {q_weight_base['nrmse']:.6f}"
    )
    print(
        f"FULL cosine/NRMSE        : "
        f"{q_weight_full['cosine']:.6f} / {q_weight_full['nrmse']:.6f}"
    )
    print("QUALIDADE — SAÍDA DA LINEAR")
    print(
        f"BASE cosine/NRMSE        : "
        f"{q_output_base['cosine']:.6f} / {q_output_base['nrmse']:.6f}"
    )
    print(
        f"FULL cosine/NRMSE        : "
        f"{q_output_full['cosine']:.6f} / {q_output_full['nrmse']:.6f}"
    )
    print("-" * 78)
    print("VELOCIDADE — MEDIANA")
    print(
        f"Baseline FP32 GEMM       : {baseline_perf['median_ms']:.4f} ms"
    )
    print(
        f"RIFT FULL predecoded     : {full_predecoded_perf['median_ms']:.4f} ms "
        f"| {perf['rift_full_predecoded_vs_baseline']['speedup_x']:.3f}x"
    )
    print(
        f"RIFT FULL decode+GEMM    : {full_reference_perf['median_ms']:.4f} ms "
        f"| {perf['rift_full_reference_vs_baseline']['speedup_x']:.3f}x"
    )
    print(
        "Native low-bit kernel    : NÃO IMPLEMENTADO — nenhum ganho nativo "
        "de inferência deve ser reivindicado ainda."
    )
    print("-" * 78)
    print(f"Relatório JSON           : {report_path}")
    print(f"Histórico CSV legado     : {out_dir / 'rift_phase1_gain_history.csv'}")
    print(f"Baterias JSON            : {recorder.json_path}")
    print(f"Baterias CSV             : {recorder.csv_path}")
    print(f"JSON por bateria         : {recorder.batteries_dir}")
    print("=" * 78)

    return report


# ------------------------------------------------------------------------------
# 11. Self-test sintético sem Hugging Face
# ------------------------------------------------------------------------------

def run_synthetic_self_test(out_dir: Path) -> None:
    rng = np.random.default_rng(1234)
    W = rng.normal(size=(128, 256)).astype(np.float32)

    model_id = "synthetic/rift-b0"
    recorder = BatteryRecorder(out_dir, model_id=model_id)

    ir = build_linear_ir_m0(
        model_id=model_id,
        architecture="synthetic",
        weight_name="linear.weight",
        weight_shape=W.shape,
    )
    validate_rift_ir_m0(ir)

    b0 = run_b0(out_dir, ir, W)

    recorder.record(
        battery_id="B0_BINARY_IR_FOUNDATION",
        status="PASS",
        baseline_disk_bytes=int(W.nbytes),
        rift_disk_bytes=int(b0["container"]["file_size"]),
        baseline_tok_s=None,
        rift_tok_s=None,
        baseline_ram_bytes=None,
        rift_ram_bytes=None,
        measurement_scope="synthetic B0; Tok/s/RAM not applicable",
        quality={"full_local_gate_pass": True},
        metrics={"container": b0["container"]},
        notes="Self-test sintético automático.",
    )

    codec = quantize_q4_linear_test(W)
    full = decode_q4_linear_test(codec, True)
    q = compute_metrics(W, full)
    payload = write_q4_linear_payloads(out_dir, codec)

    recorder.record(
        battery_id="SELFTEST_Q4_BASE_PLUS_REF_4BIT",
        status="PASS" if q["cosine"] >= 0.995 and q["nrmse"] <= 0.05 else "EXPERIMENTAL_FAIL",
        baseline_disk_bytes=int(W.nbytes),
        rift_disk_bytes=int(payload["full_payload_disk_bytes"]),
        baseline_tok_s=None,
        rift_tok_s=None,
        baseline_ram_bytes=None,
        rift_ram_bytes=None,
        measurement_scope="synthetic codec/storage self-test",
        quality={"full_local_gate_pass": bool(q["cosine"] >= 0.995 and q["nrmse"] <= 0.05), "weight": q},
        notes="Sem benchmark de modelo; Tok/s e RAM ficam null.",
    )

    print(
        "[SELF-TEST] Q4_LINEAR_TEST full: "
        f"cosine={q['cosine']:.6f}, nrmse={q['nrmse']:.6f}"
    )
    print(f"[SELF-TEST] Baterias JSON: {recorder.json_path}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _without_ipykernel_connection_args(argv: Iterable[str]) -> List[str]:
    """
    Remove somente o par interno ``-f kernel-*.json`` injetado pelo Jupyter.

    Não usamos parse_known_args porque ele esconderia erros reais de digitação
    nos argumentos do benchmark.
    """
    values = list(argv)
    filtered: List[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "-f" and index + 1 < len(values):
            connection_file = Path(values[index + 1]).name
            if connection_file.startswith("kernel-") and connection_file.endswith(".json"):
                index += 2
                continue
        if value.startswith("-f="):
            connection_file = Path(value[3:]).name
            if connection_file.startswith("kernel-") and connection_file.endswith(".json"):
                index += 1
                continue
        filtered.append(value)
        index += 1
    return filtered


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="RIFT-LM v0.3.5 B0 + Phase 1 reference test"
    )
    parser.add_argument(
        "--mode",
        choices=["self-test", "phase1"],
        default="phase1",
        help="self-test não baixa modelo; phase1 usa Qwen por padrão",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Model ID (org/modelo) ou URL https://huggingface.co/org/modelo",
    )
    parser.add_argument(
        "--target-layer",
        default="auto",
        help="Nome do tensor .weight ou 'auto' para escolher uma camada Linear",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permite código remoto do modelo (use apenas se confiar no repositório)",
    )
    parser.add_argument(
        "--prompt",
        default="Explique em uma frase por que a memória é importante na inferência de LLMs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu ou cuda",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--out",
        default="rift_m0_test_output",
    )
    parser.add_argument(
        "--publish",
        choices=["auto", "required", "off"],
        default=os.environ.get("RIFT_PUBLISH_MODE", "auto"),
        help=(
            "auto publica quando configurado (e exige publicação no Colab); "
            "required sempre exige; off mantém somente os arquivos locais"
        ),
    )
    parser.add_argument(
        "--github-repo",
        default=None,
        help="Repositório owner/repo; alternativa: RIFT_GITHUB_REPOSITORY",
    )
    parser.add_argument(
        "--results-endpoint",
        default=None,
        help="URL HTTPS /api/results do Vercel; alternativa: RIFT_RESULTS_ENDPOINT",
    )
    parser.add_argument(
        "--github-branch",
        default=None,
        help="Branch de publicação; por padrão usa a branch default do repositório",
    )
    parser.add_argument(
        "--github-data-path",
        default=os.environ.get(
            "RIFT_GITHUB_DATA_PATH", "data/rift_test_batteries.json"
        ),
        help="Caminho do histórico dentro do repositório do dashboard",
    )
    cli_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(_without_ipykernel_connection_args(cli_args))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "self-test":
        run_synthetic_self_test(out_dir)
    else:
        run_phase1(
            out_dir=out_dir,
            model_id=args.model,
            target_layer_name=args.target_layer,
            prompt=args.prompt,
            iterations=args.iterations,
            device_str=args.device,
            trust_remote_code=args.trust_remote_code,
        )

    try:
        publish_results_if_configured(
            out_dir / "rift_test_batteries.json",
            mode=args.publish,
            results_endpoint=args.results_endpoint,
            repository=args.github_repo,
            branch=args.github_branch,
            target_path=args.github_data_path,
        )
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc
    finally:
        cleanup_colab_workspace(label="RIFT")


if __name__ == "__main__":
    main()
