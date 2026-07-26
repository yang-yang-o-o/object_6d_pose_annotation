#!/usr/bin/env bash
# Real multi-view stereo dense reconstruction (CUDA COLMAP via conda).
# Prefer this over Depth-Anything fusion for clean object geometry.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COLMAP_BIN=/usr/local/miniconda3/envs/colmap_cuda/bin/colmap
export LD_LIBRARY_PATH=/usr/local/miniconda3/envs/colmap_cuda/lib:${LD_LIBRARY_PATH:-}

IMAGES=${IMAGES:-data/frames}
SFM=${SFM:-outputs/run1/sfm}
WS=${WS:-outputs/run1/mvs_cuda}
DENSE=$WS/dense
OUT_PLY=${OUT_PLY:-outputs/run1/export/dense_mvs.ply}

mkdir -p "$WS/sparse/0"
# copy SfM model
for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
  [[ -f "$SFM/$f" ]] && cp -f "$SFM/$f" "$WS/sparse/0/"
done

rm -rf "$DENSE"
mkdir -p "$DENSE"

# Default 1600 (run1). Full-res: MAX_IMAGE_SIZE=-1
: "${MAX_IMAGE_SIZE:=1600}"

echo "== undistorter (max_image_size=${MAX_IMAGE_SIZE}) =="
"$COLMAP_BIN" image_undistorter \
  --image_path "$IMAGES" \
  --input_path "$WS/sparse/0" \
  --output_path "$DENSE" \
  --output_type COLMAP \
  --max_image_size "$MAX_IMAGE_SIZE"

echo "== patch_match_stereo (CUDA, geom_consistency) =="
"$COLMAP_BIN" patch_match_stereo \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.gpu_index 0 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.num_iterations 5

echo "== stereo_fusion =="
"$COLMAP_BIN" stereo_fusion \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE/fused.ply"

cp -f "$DENSE/fused.ply" "$OUT_PLY"
ls -lh "$OUT_PLY"
echo "Done: $OUT_PLY"
