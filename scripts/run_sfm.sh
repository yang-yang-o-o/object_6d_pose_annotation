#!/usr/bin/env bash
# Default = run1 config (1600 / fast, structure quality sufficient)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
python scripts/run_sfm_hloc.py \
  --images "${IMAGES:-data/frames}" \
  --outputs "${OUT:-outputs/run1}" \
  --matcher superpoint+lightglue \
  --max_keypoints 4096 \
  --resize_max 1600
