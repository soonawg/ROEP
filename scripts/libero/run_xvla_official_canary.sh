#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
XVLA_PY="${XVLA_PY:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null

export ROEP_XVLA_CKPT="${ROEP_XVLA_CKPT:-lerobot/xvla-libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

XVLA_PREFIX="$(dirname "$(dirname "$XVLA_PY")")"
export LD_LIBRARY_PATH="$XVLA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$XVLA_PY" scripts/libero/check_xvla_env.py --require-cuda --checkpoint "$ROEP_XVLA_CKPT" --smoke-load
