#!/usr/bin/env bash
# Download SAM2.1 tiny weights used by the pose annotator.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT/models"
DEST="$DEST_DIR/sam2.1_t.pt"

# Matches the ultralytics assets tag that ships sam2.1_t.pt (verified with ultralytics==8.4.105).
URL="${SAM21_T_URL:-https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt}"

mkdir -p "$DEST_DIR"
if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
  exit 0
fi

echo "Downloading $URL → $DEST"
tmp="$(mktemp "$DEST_DIR/sam2.1_t.pt.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$URL"
mv -f "$tmp" "$DEST"
trap - EXIT
echo "OK: $DEST ($(du -h "$DEST" | awk '{print $1}'))"
