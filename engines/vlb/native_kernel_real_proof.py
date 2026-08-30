#!/usr/bin/env python3
"""Native VLB kernel numerical proof on real VLB-DIR model tensors.

This is an operator proof, not an intelligence/KR100 proof. It intentionally
uses actual quantized matrices produced by the model conversion. The activation
vector is deterministic unless a captured activation is supplied in a future
proof stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

MAGIC = b"VLBK001\x00"


def build_native(native_dir: Path, build_dir: Path) -> Path:
    subprocess.check_call(["cmake", "-S", str(native_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    subprocess.check_call(["cmake", "--build", str(build_dir), "--config", "Release", "-j2"])
    candidates = [
        build_dir / "vlb-kernel-proof",
        build_dir / "Release" / "vlb-kernel-proof.exe",
        build_dir / "vlb-kernel-proof.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("vlb-kernel-proof executable not found after build")


def reference_from_payload(payload: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    shape = [int(v) for v in payload["shape"]]
    if len(shape) != 2:
        raise RuntimeError("kernel proof requires 2-D Linear matrix")
    q = payload["qweight"].float()
    scales = payload["scales"].float()
    flat = (q * scales[:, None]).reshape(-1)[: math.prod(shape)]
    weight = flat.reshape(shape).float()
    return torch.mv(weight, x.float())


def write_vector(path: Path, payload: Dict[str, Any], x: torch.Tensor, reference: torch.Tensor) -> None:
    shape = [int(v) for v in payload["shape"]]
    rows, cols = shape
    q = payload["qweight"].to(torch.int8).contiguous().reshape(-1)
    scales = payload["scales"].to(torch.float16).contiguous().reshape(-1)
    with path.open("wb") as f:
        f.write(struct.pack("<8sQQQ", MAGIC, rows, cols, int(payload["group_size"])))
        f.write(q.numpy().tobytes(order="C"))
        f.write(scales.numpy().tobytes(order="C"))
        f.write(x.to(torch.float32).contiguous().numpy().tobytes(order="C"))
        f.write(reference.to(torch.float32).contiguous().numpy().tobytes(order="C"))


def proof(vlb_dir: Path, executable: Path, tensor_limit: int) -> Dict[str, Any]:
    manifest = json.loads((vlb_dir / "vlb_manifest.json").read_text(encoding="utf-8"))
    candidates = [
        r for r in manifest["tensor_records"]
        if r.get("format") == "Q8_G64" and len(r.get("shape") or []) == 2
    ]
    if not candidates:
        raise RuntimeError("VLB-DIR contains no 2-D Q8_G64 tensors")

    # Spread selection across the actual model rather than only testing the first layer.
    if tensor_limit < len(candidates):
        indices = torch.linspace(0, len(candidates) - 1, steps=tensor_limit).round().long().tolist()
        selected = [candidates[i] for i in sorted(set(indices))]
    else:
        selected = candidates

    rows: List[Dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(20260830)

    with tempfile.TemporaryDirectory(prefix="vlb_native_kernel_proof_") as td:
        td_path = Path(td)
        for index, record in enumerate(selected):
            payload = torch.load(vlb_dir / record["file"], map_location="cpu", weights_only=False)
            shape = [int(v) for v in payload["shape"]]
            x = torch.randn(shape[1], generator=generator, dtype=torch.float32)
            reference = reference_from_payload(payload, x)
            vector = td_path / f"case_{index:04d}.bin"
            write_vector(vector, payload, x, reference)
            proc = subprocess.run(
                [str(executable), "--vector", str(vector)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            metric = json.loads(proc.stdout.strip().splitlines()[-1])
            metric.update(
                {
                    "tensor": record["name"],
                    "tensor_artifact_sha256": record.get("artifact_sha256"),
                    "activation_source": "DETERMINISTIC_KERNEL_PROBE_NOT_MODEL_ACTIVATION",
                }
            )
            rows.append(metric)
            print(
                f"[VLB native] {index + 1}/{len(selected)} {record['name']} "
                f"max_abs={metric['max_abs_error']:.8g} nrmse={metric['nrmse']:.8g} cos={metric['cosine']:.12g}",
                flush=True,
            )

    # This threshold is only a floating-point implementation equivalence gate.
    # It must never be presented as KR100 or model-quality retention.
    numeric_pass = all(
        float(row["max_abs_error"]) <= 1e-3 and float(row["nrmse"]) <= 1e-5
        for row in rows
    )
    return {
        "schema": "VLB_NATIVE_KERNEL_REAL_TENSOR_PROOF_V1",
        "model_id": manifest.get("model_id"),
        "quant": manifest.get("quant"),
        "tested_tensors": len(rows),
        "total_eligible_tensors": len(candidates),
        "numeric_threshold": {"max_abs_error": 1e-3, "nrmse": 1e-5},
        "numeric_pass": numeric_pass,
        "status": "VLB_NATIVE_KERNEL_NUMERIC_PASS" if numeric_pass else "VLB_NATIVE_KERNEL_NUMERIC_FAIL",
        "kr100_claim": False,
        "rows": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vlb-dir", required=True)
    p.add_argument("--tensor-limit", type=int, default=12)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    native_dir = here / "native"
    build_dir = Path("/tmp/vlb-native-build") if os.name != "nt" else Path(tempfile.gettempdir()) / "vlb-native-build"
    executable = build_native(native_dir, build_dir)
    report = proof(Path(args.vlb_dir), executable, max(1, args.tensor_limit))
    output = Path(args.output) if args.output else Path(args.vlb_dir) / "native_kernel_proof.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["numeric_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
