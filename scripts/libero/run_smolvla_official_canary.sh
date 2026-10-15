#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SMOLVLA_PY="${SMOLVLA_PY:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null
source scripts/libero/export_libero_v1_defaults.sh >/dev/null

: "${ROEP_SMOLVLA_BASE_CKPT:?Set ROEP_SMOLVLA_BASE_CKPT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SMOLVLA_OFFICIAL_MAX_STEPS="${SMOLVLA_OFFICIAL_MAX_STEPS:-5}"
SMOLVLA_PREFIX="$(dirname "$(dirname "$SMOLVLA_PY")")"
export LD_LIBRARY_PATH="$SMOLVLA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
smolvla_preload=(
  "$SMOLVLA_PREFIX/lib/libstdc++.so.6"
  "$SMOLVLA_PREFIX/lib/libgcc_s.so.1"
)
resolved_preload=()
for lib in "${smolvla_preload[@]}"; do
  if [[ -e "$lib" ]]; then
    resolved_preload+=("$lib")
  fi
done
if (( ${#resolved_preload[@]} > 0 )); then
  preload_joined="$(IFS=:; echo "${resolved_preload[*]}")"
  export LD_PRELOAD="${preload_joined}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ "$ROEP_SMOLVLA_BASE_CKPT" == /* && ! -e "$ROEP_SMOLVLA_BASE_CKPT" ]]; then
  echo "[error] SmolVLA checkpoint path does not exist: $ROEP_SMOLVLA_BASE_CKPT" >&2
  echo "Download a LIBERO-tuned checkpoint first." >&2
  echo "Example: hf download HuggingFaceVLA/smolvla_libero --local-dir $PROJECT_ROOT/model_ckpts/smolvla_libero" >&2
  exit 1
fi

"$SMOLVLA_PY" scripts/libero/check_smolvla_env.py --require-cuda

"$SMOLVLA_PY" -m lerobot.scripts.lerobot_eval \
  --policy.path="$ROEP_SMOLVLA_BASE_CKPT" \
  --env.type=libero \
  --env.task="$ROEP_LIBERO_SUITE" \
  --env.task_ids="[0]" \
  --env.episode_length="$SMOLVLA_OFFICIAL_MAX_STEPS" \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --policy.use_amp=false \
  --output_dir="$PROJECT_ROOT/results/libero_official_smolvla_canary"
