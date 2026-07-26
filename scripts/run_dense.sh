#!/usr/bin/env bash
# Dense colored cloud for MeshLab scale picking (SfM coordinate frame).
# apt COLMAP has no CUDA → patch_match_stereo fails; use depth+pose fusion instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

# Ensure undistorted images exist (from prior MVS prep)
if [[ ! -d outputs/run1/mvs/dense/images ]]; then
  python scripts/run_dense_mvs.py \
    --images data/frames \
    --sfm_dir outputs/run1/sfm \
    --workspace outputs/run1/mvs \
    --max_image_size 1600 \
    --skip_stereo
fi

python scripts/dense_from_depth.py \
  --sfm_dir outputs/run1/mvs/dense/sparse \
  --images outputs/run1/mvs/dense/images \
  --out_ply outputs/run1/export/dense.ply \
  --stride 1
echo "Open: outputs/run1/export/dense.ply"
