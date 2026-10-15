#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

conda env create -f envs/roep_libero_core.yml
conda env create -f envs/roep_openvla_libero.yml
conda env create -f envs/roep_smolvla_libero.yml

echo "[done] base conda envs created."
echo "Next: run the post-install scripts in scripts/libero/."
