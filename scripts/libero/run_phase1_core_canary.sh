#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CORE_PY="${CORE_PY:-${HOME}/miniconda3/envs/roep_libero_core/bin/python}"
cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null
source scripts/libero/export_libero_v1_defaults.sh >/dev/null

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PROJECT_ROOT/roepbench${PYTHONPATH:+:$PYTHONPATH}"
USE_XVFB="${USE_XVFB:-0}"

"$CORE_PY" scripts/libero/check_libero_core.py

: "${ROEP_LIBERO_SUITE:?Set ROEP_LIBERO_SUITE}"
: "${ROEP_LIBERO_TASK_NAME:?Set ROEP_LIBERO_TASK_NAME}"

cmd=(
  "$CORE_PY" -m roepbench.run_eval
  --benchmark_backend libero
  --task "$ROEP_LIBERO_TASK_NAME"
  --libero_suite "$ROEP_LIBERO_SUITE"
  --libero_task_name "$ROEP_LIBERO_TASK_NAME"
  --episodes 1
  --out_dir results/libero_core_canary
)

if [[ "$USE_XVFB" == "1" ]]; then
  cmd+=(--use_xvfb)
fi

"${cmd[@]}"
