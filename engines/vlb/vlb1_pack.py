#!/usr/bin/env python3
"""Pack VLB-DIR into a native VLB1 binary container.

PyTorch is intentionally allowed here because this is a conversion-side tool.
The produced .vlb file contains no pickle/PT payload and is intended to be
consumed by the future native VLB runtime directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List

import torch

MAGIC = b"VLB1\x00\x00\x00\x00"
VERSION = 1
ALIGNMENT = 64


def align(value: int, n: int = ALIGNMENT) -> int:
    return (value + n - 1) // n * n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes(order="C")


def encode_record(vlb_dir: Path, record: Dict[str, Any]) -> tuple[Dict[str, Any], bytes]:
    payload = torch.load(vlb_dir / record["file"], map_location="cpu", weights_only=False)
    fmt = payload["format"]

    if fmt == "Q8_G64":
        q = payload["qweight"].to(torch.int8).contiguous()
        scales = payload["scales"].to(torch.float16).contiguous()
        q_bytes = tensor_bytes(q)
        s_bytes = tensor_bytes(scales)
        raw = q_bytes + s_bytes
        meta = {
            "name": record["name"],
            "format": fmt,
            "shape": [int(x) for x in payload["shape"]],
            "group_size": int(payload["group_size"]),
            "q_bytes": len(q_bytes),
            "scale_bytes": len(s_bytes),
            "scale_dtype": "F16",
            "source_artifact_sha256": record.get("artifact_sha256"),
        }
    elif fmt == "FP16":
        t = payload["tensor"].to(torch.float16).contiguous()
        raw = tensor_bytes(t)
        meta = {
            "name": record["name"],
            "format": fmt,
            "shape": [int(x) for x in payload["shape"]],
            "tensor_bytes": len(raw),
            "dtype": "F16",
            "source_artifact_sha256": record.get("artifact_sha256"),
        }
    elif fmt == "RAW":
        t = payload["tensor"].contiguous()
        raw = tensor_bytes(t)
        meta = {
            "name": record["name"],
            "format": fmt,
            "shape": [int(x) for x in payload["shape"]],
            "tensor_bytes": len(raw),
            "dtype": str(t.dtype).replace("torch.", ""),
            "source_artifact_sha256": record.get("artifact_sha256"),
        }
    else:
        raise RuntimeError(f"unsupported VLB-DIR payload format {fmt}: {record['name']}")

    meta["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    return meta, raw


def pack(vlb_dir: Path, output: Path) -> Dict[str, Any]:
    source_manifest_path = vlb_dir / "vlb_manifest.json"
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = sorted(source["tensor_records"], key=lambda r: r["name"])

    encoded: List[tuple[Dict[str, Any], bytes]] = []
    for index, record in enumerate(records, start=1):
        meta, raw = encode_record(vlb_dir, record)
        encoded.append((meta, raw))
        if index % 50 == 0 or index == len(records):
            print(f"[VLB1] encoded {index}/{len(records)} {meta['name']}", flush=True)

    # Offsets live after fixed header + canonical JSON header. Since the JSON
    # length depends on offsets, converge deterministically.
    header: Dict[str, Any] = {
        "schema": "VLB1",
        "version": VERSION,
        "model_id": source.get("model_id"),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "quant": source.get("quant"),
        "alignment": ALIGNMENT,
        "tensor_count": len(encoded),
        "tensors": [],
    }

    header_bytes = b""
    for _ in range(8):
        base = align(8 + 8 + len(header_bytes))
        cursor = base
        tensor_meta = []
        for meta, raw in encoded:
            item = dict(meta)
            item["offset"] = cursor
            item["length"] = len(raw)
            tensor_meta.append(item)
            cursor = align(cursor + len(raw))
        header["tensors"] = tensor_meta
        new_header = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if new_header == header_bytes:
            break
        header_bytes = new_header
    else:
        raise RuntimeError("VLB1 header offset convergence failed")

    base = align(8 + 8 + len(header_bytes))
    if header["tensors"] and header["tensors"][0]["offset"] != base:
        raise RuntimeError("VLB1 internal offset mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * (base - f.tell()))
        for (meta, raw), item in zip(encoded, header["tensors"]):
            if f.tell() != item["offset"]:
                raise RuntimeError(f"VLB1 offset mismatch for {meta['name']}")
            f.write(raw)
            padded = align(f.tell())
            f.write(b"\x00" * (padded - f.tell()))

    report = {
        "schema": "VLB1_PACK_REPORT_V1",
        "model_id": source.get("model_id"),
        "vlb1_path": str(output),
        "vlb1_bytes": output.stat().st_size,
        "vlb1_sha256": sha256_file(output),
        "tensor_count": len(encoded),
        "runtime_dependency": "NONE_PYTORCH_FREE_CONTAINER",
    }
    (output.with_suffix(output.suffix + ".json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vlb-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = pack(Path(args.vlb_dir), Path(args.output))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
