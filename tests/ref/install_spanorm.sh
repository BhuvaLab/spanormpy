#!/usr/bin/env bash
# Install SpaNorm (R, master @ c12e5e5, v1.7.15) into tests/ref/Rlib only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RBIN="${RBIN:-$HOME/miniconda3/envs/latest-r/bin}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$HERE/Rlib"
git -C "$REPO/../SpaNorm" archive c12e5e5 | tar -x -C "$TMP"
"$RBIN/R" CMD INSTALL --library="$HERE/Rlib" "$TMP"
"$RBIN/Rscript" -e "library(SpaNorm, lib.loc='$HERE/Rlib'); print(packageVersion('SpaNorm'))"
