#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPENVLA_REPO="${OPENVLA_REPO:-$PROJECT_ROOT/external/openvla}"
OPENVLA_PY="${OPENVLA_PY:-${HOME}/miniconda3/envs/roep_openvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null
source scripts/libero/export_libero_v1_defaults.sh >/dev/null

: "${ROEP_OPENVLA_7B_CKPT:?Set ROEP_OPENVLA_7B_CKPT}"

if [[ ! -d "$OPENVLA_REPO" ]]; then
  echo "[error] OPENVLA_REPO not found: $OPENVLA_REPO" >&2
  exit 1
fi

if [[ "$ROEP_OPENVLA_7B_CKPT" == /* && ! -d "$ROEP_OPENVLA_7B_CKPT" ]]; then
  echo "[error] OpenVLA checkpoint directory does not exist: $ROEP_OPENVLA_7B_CKPT" >&2
  echo "Download the LIBERO-goal checkpoint first." >&2
  echo "Example: hf download openvla/openvla-7b-finetuned-libero-goal --local-dir $PROJECT_ROOT/model_ckpts/openvla-7b-finetuned-libero-goal" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENVLA_OFFICIAL_LOAD_IN_4BIT="${OPENVLA_OFFICIAL_LOAD_IN_4BIT:-True}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
"$OPENVLA_PY" scripts/libero/check_openvla_env.py --require-cuda

cd "$OPENVLA_REPO"
OPENVLA_TASK_ID="${ROEP_LIBERO_TASK_ID:-0}"
if [[ -z "$OPENVLA_TASK_ID" ]]; then
  OPENVLA_TASK_ID=0
fi
OPENVLA_OFFICIAL_MAX_STEPS="${OPENVLA_OFFICIAL_MAX_STEPS:-5}"
"$OPENVLA_PY" experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$ROEP_OPENVLA_7B_CKPT" \
  --task_suite_name "$ROEP_LIBERO_SUITE" \
  --task_id "$OPENVLA_TASK_ID" \
  --episode_max_steps "$OPENVLA_OFFICIAL_MAX_STEPS" \
  --load_in_4bit "$OPENVLA_OFFICIAL_LOAD_IN_4BIT" \
  --num_trials_per_task 1 \
  --center_crop True \
  --seed 7 \
  --run_id_note roep_official_canary
