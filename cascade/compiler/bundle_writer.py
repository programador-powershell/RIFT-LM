"""CASCADE Bundle M0 — arquivo físico mmap-friendly."""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from cascade.compiler.cascade_ir import make_linear_ir, validate_cascade_ir
from cascade.compiler.decompose import CascadeLinearStages

MAGIC = b"CSCD"
VERSION = 0x0003

HEADER_SIZE = 128
# magic(4) ver(H) flags(H) hdr(I) n_stages(I) ir_off(Q) st_off(Q) gate_off(Q) pay_off(Q) fsize(Q) crc(Q) reserved(Q) pad(56)
HEADER_FMT = "<4sHHIIQQQQQQQ56s"


def _pack_header(*, n_stages, ir_offset, stage_table_offset, gate_table_offset, payload_offset, file_size, checksum=0):
    return struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        0,
        HEADER_SIZE,
        n_stages,
        ir_offset,
        stage_table_offset,
        gate_table_offset,
        payload_offset,
        file_size,
        checksum,
        0,  # reserved Q
        b"\x00" * 56,
    )


def _align(n: int, a: int = 64) -> int:
    return (n + a - 1) // a * a


def _tensor_payload(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    header = {
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
    }
    meta = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = t.numpy().tobytes()
    return struct.pack("<I", len(meta)) + meta + body


def write_cascade_bundle(
    path: Path | str,
    *,
    stages: CascadeLinearStages,
    model_id: str,
    target_layer: str,
    architecture_hint: str = "dense-linear",
    gate_percentile: float = 70.0,
) -> Dict[str, Any]:
    """Escreve model.cascade mínimo com F0 INT4 + F1 low-rank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ir = make_linear_ir(
        model_id=model_id,
        architecture_hint=architecture_hint,
        target_layer=target_layer,
        cascade_ref=0,
    )
    validate_cascade_ir(ir)
    ir_bytes = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    gate = {
        "type": "ACTIVATION_SCORE_PERCENTILE_V0",
        "percentile": float(gate_percentile),
        "stage_index_controlled": 1,
    }
    gate_bytes = json.dumps(gate, separators=(",", ":")).encode("utf-8")

    # Stage table entries + payloads
    f0_payload = _tensor_payload(stages.codes) + _tensor_payload(stages.scales)
    f0_meta = json.dumps({
        "stage_id": 0,
        "stage_index": 0,
        "stage_type": "BASE_STAGE",
        "codec": "INT4_GROUP",
        "group_size": stages.group_size,
        "out_features": stages.out_features,
        "in_features": stages.in_features,
        "residency_hint": "HOT",
    }, separators=(",", ":")).encode("utf-8")
    f0_blob = struct.pack("<I", len(f0_meta)) + f0_meta + f0_payload

    f1_payload = (
        _tensor_payload(stages.u)
        + _tensor_payload(stages.s)
        + _tensor_payload(stages.v)
    )
    f1_meta = json.dumps({
        "stage_id": 1,
        "stage_index": 1,
        "stage_type": "RESIDUAL_LOWRANK",
        "codec": "FP32_LOWRANK",
        "rank": stages.rank,
        "residency_hint": "WARM",
    }, separators=(",", ":")).encode("utf-8")
    f1_blob = struct.pack("<I", len(f1_meta)) + f1_meta + f1_payload

    payloads = [f0_blob, f1_blob]

    ir_offset = HEADER_SIZE
    stage_table_offset = _align(ir_offset + 4 + len(ir_bytes))
    # each stage entry: offset u64, size u64, stage_id u32, flags u32 = 24 bytes
    STAGE_ENTRY = 24
    gate_table_offset = stage_table_offset + len(payloads) * STAGE_ENTRY
    payload_offset = _align(gate_table_offset + 4 + len(gate_bytes))

    offsets = []
    cursor = payload_offset
    for blob in payloads:
        offsets.append((cursor, len(blob)))
        cursor = _align(cursor + len(blob))

    file_size = cursor
    # header placeholders; checksum over body after header
    header = _pack_header(n_stages=len(payloads), ir_offset=ir_offset, stage_table_offset=stage_table_offset, gate_table_offset=gate_table_offset, payload_offset=payload_offset, file_size=file_size, checksum=0)
    assert len(header) == HEADER_SIZE

    parts = [bytearray(header)]
    # IR
    pad_ir = stage_table_offset - (HEADER_SIZE + 4 + len(ir_bytes))
    parts.append(struct.pack("<I", len(ir_bytes)) + ir_bytes + b"\x00" * max(0, pad_ir))
    # stage table
    for off, sz in offsets:
        parts.append(struct.pack("<QQII", off, sz, offsets.index((off, sz)), 0))
    # gate
    pad_gate = payload_offset - (gate_table_offset + 4 + len(gate_bytes))
    parts.append(struct.pack("<I", len(gate_bytes)) + gate_bytes + b"\x00" * max(0, pad_gate))
    # payloads with alignment padding
    cursor = payload_offset
    for blob in payloads:
        parts.append(blob)
        cursor += len(blob)
        aligned = _align(cursor)
        if aligned > cursor:
            parts.append(b"\x00" * (aligned - cursor))
            cursor = aligned

    data = b"".join(parts)
    checksum = zlib.crc32(data[HEADER_SIZE:]) & 0xFFFFFFFFFFFFFFFF
    # rewrite checksum at offset 64 (6th Q after smaller fields)
    # layout: 4s H H I I Q Q Q Q Q Q 56s
    # checksum is 6th Q at: 4+2+2+4+4+8*5 = 56? Let's pack properly
    header2 = _pack_header(n_stages=len(payloads), ir_offset=ir_offset, stage_table_offset=stage_table_offset, gate_table_offset=gate_table_offset, payload_offset=payload_offset, file_size=len(data), checksum=checksum)
    data = header2 + data[HEADER_SIZE:]
    path.write_bytes(data)

    return {
        "path": str(path),
        "file_size": len(data),
        "checksum": checksum,
        "stages": [
            {"stage_id": 0, "offset": offsets[0][0], "size": offsets[0][1], "type": "BASE_STAGE", "codec": "INT4_GROUP"},
            {"stage_id": 1, "offset": offsets[1][0], "size": offsets[1][1], "type": "RESIDUAL_LOWRANK", "codec": "FP32_LOWRANK"},
        ],
        "f0_bytes": stages.f0_bytes,
        "f1_bytes": stages.f1_bytes,
        "baseline_bytes": stages.baseline_bytes,
        "gate_percentile": gate_percentile,
        "model_id": model_id,
        "target_layer": target_layer,
    }
