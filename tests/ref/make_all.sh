#!/usr/bin/env bash
# Regenerate every reference file. Run from anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RSCRIPT="${RSCRIPT:-$HOME/miniconda3/envs/latest-r/bin/Rscript}"
mkdir -p "$HERE/data"
cd "$REPO"
for f in "$HERE"/make_*.R; do
  echo "== $(basename "$f")"
  "$RSCRIPT" "$f"
done
