#!/usr/bin/env bash
set -euo pipefail
python3 cascade_converter.py convert --input "$1" --output "$2" --ranks 8,16,32 --group-size 64
