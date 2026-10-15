#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_ROOT/external/LIBERO}"

if [[ ! -d "$LIBERO_REPO" ]]; then
  echo "[error] LIBERO_REPO not found: $LIBERO_REPO" >&2
  echo "Clone the official LIBERO repo first." >&2
  exit 1
fi

cd "$LIBERO_REPO"
pip install -e .
pip install \
  "bddl" \
  "cloudpickle" \
  "easydict" \
  "gym" \
  "imageio[ffmpeg]" \
  "matplotlib" \
  "robosuite==1.4.1"

echo "[done] LIBERO core package installed in current env."
