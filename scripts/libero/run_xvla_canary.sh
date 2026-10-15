#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
XVLA_PY="${XVLA_PY:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null

export ROEP_BENCHMARK_BACKEND=libero
export ROEP_LIBERO_SUITE="${ROEP_LIBERO_SUITE:-libero_goal}"
export ROEP_LIBERO_TASK_NAME="${ROEP_LIBERO_TASK_NAME:-put_the_bowl_on_top_of_the_cabinet}"
export ROEP_XVLA_CKPT="${ROEP_XVLA_CKPT:-lerobot/xvla-libero}"
export ROEP_XVLA_IMPL="${ROEP_XVLA_IMPL:-roepbench.model_plugins.xvla_plugin:build_adapter}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

XVLA_PREFIX="$(dirname "$(dirname "$XVLA_PY")")"
export LD_LIBRARY_PATH="$XVLA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
xvla_preload=(
  "$XVLA_PREFIX/lib/libstdc++.so.6"
  "$XVLA_PREFIX/lib/libgcc_s.so.1"
)
resolved_preload=()
for lib in "${xvla_preload[@]}"; do
  if [[ -e "$lib" ]]; then
    resolved_preload+=("$lib")
  fi
done
if (( ${#resolved_preload[@]} > 0 )); then
  preload_joined="$(IFS=:; echo "${resolved_preload[*]}")"
  export LD_PRELOAD="${preload_joined}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

"$XVLA_PY" scripts/libero/check_xvla_env.py --require-cuda

PYTHONPATH=./roepbench:${HOME}/LIBERO "$XVLA_PY" scripts/run_model_adapter_eval.py \
  --model xvla \
  --benchmark_backend libero \
  --libero_suite "$ROEP_LIBERO_SUITE" \
  --libero_task_name "$ROEP_LIBERO_TASK_NAME" \
  --stage cache \
  --task "$ROEP_LIBERO_TASK_NAME" \
  --namespace "${X_VLA_NAMESPACE:-xvla_libero_canary_v1}" \
  --episodes "${XVLA_CANARY_EPISODES:-1}" \
  --seeds "${XVLA_CANARY_SEEDS:-0}" \
  --profiles "${XVLA_CANARY_PROFILES:-medium}" \
  --scenario "${XVLA_CANARY_SCENARIO:-natural_ood_medium}" \
  --action_mode_profile "${XVLA_ACTION_MODE_PROFILE:-libero_ee_rel_pose_gripper}" \
  --seed0_dir "${ROEP_SEED0_DIR:-$PROJECT_ROOT/results/libero_seed0}" \
  --device cuda \
  --max_steps "${XVLA_CANARY_MAX_STEPS:-300}" \
  --execute
