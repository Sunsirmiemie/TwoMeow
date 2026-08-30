#!/usr/bin/env bash
set -euo pipefail

PYTHONHASHSEED=0 .venv/bin/python scripts/run_public_eval.py \
  --no-dense \
  --output results.json
