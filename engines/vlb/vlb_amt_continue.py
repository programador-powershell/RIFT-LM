#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continue VLB AMT from an already-converted VLB artifact.

This command does not reconvert or modify the base VLB artifact. It writes all
AMT state under <vlb-dir>/amt_master_v3 and verifies the base manifest/VLB1
fingerprints before and after training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vlb_amt_master import run_amt_master_reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vlb-dir", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    vlb_dir = Path(args.vlb_dir)
    if not (vlb_dir / "vlb_manifest.json").exists():
        raise SystemExit(f"VLB manifest not found: {vlb_dir / 'vlb_manifest.json'}")

    print("=" * 112)
    print("VLB AMT MASTER v3 — CONTINUATION")
    print("=" * 112)
    print("model          :", args.model)
    print("vlb base       :", vlb_dir)
    print("base mutation  : FORBIDDEN")
    print("teacher        : NONE")
    print("KL/distill     : DISABLED")
    print("state output   :", vlb_dir / "amt_master_v3")
    print()

    report = run_amt_master_reference(args.model, vlb_dir, token)
    print(json.dumps({
        "runtime_verified": report.get("runtime_verified"),
        "amt_verified": report.get("amt_verified"),
        "classification": report.get("classification"),
        "base_immutable": report.get("base_immutable"),
        "mastery_state_counts": report.get("mastery_state_counts"),
        "rounds": report.get("rounds"),
        "commits": report.get("commits"),
        "rollbacks": report.get("rollbacks"),
        "training_target_tokens": report.get("training_target_tokens"),
        "gate_threshold": report.get("gate_threshold"),
        "final_validation": report.get("final_validation"),
        "artifacts": report.get("artifacts"),
        "native_amt_certified": report.get("native_amt_certified"),
        "kr100_certified": report.get("kr100_certified"),
    }, indent=2, ensure_ascii=False))

    # This is a reference-side AMT milestone. Return success only when the AMT
    # development contract passes and the base artifact remained immutable.
    return 0 if report.get("amt_verified") and report.get("base_immutable") else 3


if __name__ == "__main__":
    raise SystemExit(main())
