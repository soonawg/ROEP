#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PI0FAST_PY="${PI0FAST_PY:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null

export ROEP_PI0_FAST_CKPT="${ROEP_PI0_FAST_CKPT:-lerobot/pi0fast-libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

"$PI0FAST_PY" scripts/libero/check_pi0fast_env.py --require-cuda --checkpoint "$ROEP_PI0_FAST_CKPT" --smoke-load
