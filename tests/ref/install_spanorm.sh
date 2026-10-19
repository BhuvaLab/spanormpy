#!/usr/bin/env bash
# Install SpaNorm (R) into tests/ref/Rlib only. SPANORM_REF is the git ref of the sibling
# ../SpaNorm repo (or ../SpaNorm-site worktree via SPANORM_REPO); default: f233b7b (v1.7.16, psi.max fix).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SPANORM_REF="${SPANORM_REF:-f233b7b}"
SPANORM_REPO="${SPANORM_REPO:-$REPO/../SpaNorm-site}"
RBIN="${RBIN:-$HOME/miniconda3/envs/latest-r/bin}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$HERE/Rlib"
git -C "$SPANORM_REPO" archive "$SPANORM_REF" | tar -x -C "$TMP"
"$RBIN/R" CMD INSTALL --library="$HERE/Rlib" "$TMP"
"$RBIN/Rscript" -e "library(SpaNorm, lib.loc='$HERE/Rlib'); print(packageVersion('SpaNorm'))"
