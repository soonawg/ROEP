#!/usr/bin/env bash
# shellcheck shell=bash
#
# Usage:
#   source scripts/libero/export_libero_v1_defaults.sh
#
# Fixed v1 paper-track target:
# - suite: libero_goal
# - task: open_the_middle_drawer_of_the_cabinet

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Run: source scripts/libero/export_libero_v1_defaults.sh" >&2
  exit 1
fi

export ROEP_PROJECT_ROOT="${ROEP_PROJECT_ROOT:-${HOME}/roepbench_project}"
export ROEP_LIBERO_ROOT="${ROEP_LIBERO_ROOT:-${HOME}/LIBERO}"
export ROEP_BENCHMARK_BACKEND="libero"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export ROEP_LIBERO_SUITE="${ROEP_LIBERO_SUITE:-libero_goal}"
export ROEP_LIBERO_TASK_NAME="${ROEP_LIBERO_TASK_NAME:-open_the_middle_drawer_of_the_cabinet}"
export ROEP_OPENVLA_7B_CKPT="${ROEP_OPENVLA_7B_CKPT:-$ROEP_PROJECT_ROOT/model_ckpts/openvla-7b-finetuned-libero-goal}"
export ROEP_SMOLVLA_BASE_CKPT="${ROEP_SMOLVLA_BASE_CKPT:-$ROEP_PROJECT_ROOT/model_ckpts/smolvla_libero}"

echo "[ok] LIBERO v1 defaults loaded."
echo "  - backend: $ROEP_BENCHMARK_BACKEND"
echo "  - NUMBA_DISABLE_JIT: $NUMBA_DISABLE_JIT"
echo "  - libero root: $ROEP_LIBERO_ROOT"
echo "  - suite: $ROEP_LIBERO_SUITE"
echo "  - task: $ROEP_LIBERO_TASK_NAME"
echo "  - openvla ckpt: $ROEP_OPENVLA_7B_CKPT"
echo "  - smolvla ckpt: $ROEP_SMOLVLA_BASE_CKPT"
