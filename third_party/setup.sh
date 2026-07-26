#!/usr/bin/env bash
# Fetch third_party deps that are gitignored (hloc tree + OpenMVS binaries).
# Versions pinned to what was validated with the pose annotator / SfM pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"

HLOC_URL="https://github.com/cvg/Hierarchical-Localization.git"
HLOC_REV="c13273bd0ecc2917a35910fd843712a1c6243193"
HLOC_DIR="$TP/Hierarchical-Localization"

OPENMVS_VERSION="v2.4.0"
OPENMVS_ZIP_URL="https://github.com/cdcseacave/openMVS/releases/download/${OPENMVS_VERSION}/OpenMVS_Ubuntu_x64.zip"
OPENMVS_ZIP="$TP/OpenMVS_Ubuntu_x64.zip"
OPENMVS_MARKER="$TP/DensifyPointCloud"

mkdir -p "$TP"

setup_hloc() {
  if [[ -d "$HLOC_DIR/.git" ]]; then
    cur="$(git -C "$HLOC_DIR" rev-parse HEAD)"
    if [[ "$cur" == "$HLOC_REV" ]]; then
      echo "hloc already at $HLOC_REV"
    else
      echo "Updating hloc $cur → $HLOC_REV"
      git -C "$HLOC_DIR" fetch --recurse-submodules origin
      git -C "$HLOC_DIR" checkout --recurse-submodules "$HLOC_REV"
    fi
  elif [[ -d "$HLOC_DIR" ]]; then
    echo "ERROR: $HLOC_DIR exists but is not a git clone. Remove it and re-run." >&2
    exit 1
  else
    echo "Cloning hloc @ $HLOC_REV …"
    git clone --recurse-submodules "$HLOC_URL" "$HLOC_DIR"
    git -C "$HLOC_DIR" checkout --recurse-submodules "$HLOC_REV"
  fi
}

setup_openmvs() {
  if [[ -x "$OPENMVS_MARKER" ]]; then
    echo "OpenMVS binaries already present under $TP"
    return 0
  fi
  if [[ ! -f "$OPENMVS_ZIP" ]]; then
    echo "Downloading OpenMVS ${OPENMVS_VERSION} …"
    curl -fL --retry 3 --retry-delay 2 -o "$OPENMVS_ZIP" "$OPENMVS_ZIP_URL"
  fi
  echo "Extracting $OPENMVS_ZIP → $TP"
  unzip -o "$OPENMVS_ZIP" -d "$TP"
  chmod +x "$TP"/* 2>/dev/null || true
  echo "OpenMVS ${OPENMVS_VERSION} ready"
}

setup_hloc
setup_openmvs
echo "Done. Python packages (incl. hloc/lightglue) come from: uv sync"
