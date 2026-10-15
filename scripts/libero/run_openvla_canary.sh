#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null
source scripts/libero/export_libero_v1_defaults.sh >/dev/null

: "${ROEP_LIBERO_SUITE:?Set ROEP_LIBERO_SUITE}"
: "${ROEP_LIBERO_TASK_NAME:?Set ROEP_LIBERO_TASK_NAME}"
: "${ROEP_OPENVLA_7B_CKPT:?Set ROEP_OPENVLA_7B_CKPT}"

export ROEP_BENCHMARK_BACKEND=libero
export ROEP_LIBERO_RUNTIME_HOOK="${ROEP_LIBERO_RUNTIME_HOOK:-roepbench.libero.runtime_hook:run_libero_episodes}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ROEP_OPENVLA_7B_4BIT="${ROEP_OPENVLA_7B_4BIT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELS_CSV=openvla_7b
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
RUN_CORE_CANARY=0 RUN_OFFICIAL_CANARIES=0 RUN_SMOKE=1 RUN_VALIDATE=1 RUN_ADMISSION=0 RUN_MATRIX=0 MODELS_CSV=openvla_7b \
  bash scripts/run_external_libero_pipeline.sh
