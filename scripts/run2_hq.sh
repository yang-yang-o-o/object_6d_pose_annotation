#!/usr/bin/env bash
# run2: native 4K frames → high-quality SfM → CUDA MVS (with progress)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE() { echo; echo "======== [$(date '+%F %T')] $* ========"; }
PROGRESS_PY=scripts/colmap_progress.py

IMAGES=data/frames_native
OUT=outputs/run2
SFM=$OUT/sfm
EXPORT=$OUT/export
WS=$OUT/mvs_cuda
DENSE=$WS/dense
OUT_PLY=$EXPORT/dense_mvs.ply

mkdir -p "$OUT" "$EXPORT" logs
LOG=logs/run2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

STAGE "1/4 SfM (SuperPoint@2400 + LightGlue, 8192 kpts)"
source .venv/bin/activate
python -u scripts/run_sfm_hloc.py \
  --images "$IMAGES" \
  --outputs "$OUT" \
  --matcher superpoint+lightglue \
  --max_keypoints 8192 \
  --resize_max 2400

STAGE "2/4 Export poses + sparse.ply"
python -u scripts/export_poses.py \
  --sfm_dir "$SFM" \
  --out_dir "$EXPORT"

STAGE "3/4 CUDA COLMAP MVS (full resolution, geom_consistency)"
COLMAP_BIN=/usr/local/miniconda3/envs/colmap_cuda/bin/colmap
export LD_LIBRARY_PATH=/usr/local/miniconda3/envs/colmap_cuda/lib:${LD_LIBRARY_PATH:-}
: "${MAX_IMAGE_SIZE:=-1}"

mkdir -p "$WS/sparse/0"
for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
  [[ -f "$SFM/$f" ]] && cp -f "$SFM/$f" "$WS/sparse/0/"
done
rm -rf "$DENSE"
mkdir -p "$DENSE"

echo "-- undistorter (max_image_size=${MAX_IMAGE_SIZE}) --"
stdbuf -oL -eL "$COLMAP_BIN" image_undistorter \
  --image_path "$IMAGES" \
  --input_path "$WS/sparse/0" \
  --output_path "$DENSE" \
  --output_type COLMAP \
  --max_image_size "$MAX_IMAGE_SIZE" \
  2>&1 | python -u "$PROGRESS_PY"

echo "-- patch_match_stereo (GPU) --"
# Track view index across photometric+geometric passes for clearer ETA
stdbuf -oL -eL "$COLMAP_BIN" patch_match_stereo \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.gpu_index 0 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.num_iterations 5 \
  2>&1 | python -u "$PROGRESS_PY"

STAGE "4/4 stereo_fusion"
stdbuf -oL -eL "$COLMAP_BIN" stereo_fusion \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE/fused.ply" \
  2>&1 | python -u "$PROGRESS_PY"

cp -f "$DENSE/fused.ply" "$OUT_PLY"
ls -lh "$OUT_PLY" "$EXPORT/sparse.ply"
STAGE "DONE run2"
echo "sparse: $EXPORT/sparse.ply"
echo "dense:  $OUT_PLY"
echo "log:    $LOG"
