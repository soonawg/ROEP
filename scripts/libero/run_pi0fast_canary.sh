#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PI0FAST_PY="${PI0FAST_PY:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"

cd "$PROJECT_ROOT"
source scripts/export_external_model_env.sh >/dev/null

export ROEP_BENCHMARK_BACKEND=libero
export ROEP_LIBERO_SUITE="${ROEP_LIBERO_SUITE:-libero_goal}"
export ROEP_LIBERO_TASK_NAME="${ROEP_LIBERO_TASK_NAME:-put_the_bowl_on_top_of_the_cabinet}"
export ROEP_PI0_FAST_CKPT="${ROEP_PI0_FAST_CKPT:-lerobot/pi0fast-libero}"
export ROEP_PI0_FAST_IMPL="${ROEP_PI0_FAST_IMPL:-roepbench.model_plugins.pi0_fast_plugin:build_adapter}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

PI0FAST_PREFIX="$(dirname "$(dirname "$PI0FAST_PY")")"
export LD_LIBRARY_PATH="$PI0FAST_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pi0fast_preload=(
  "$PI0FAST_PREFIX/lib/libstdc++.so.6"
  "$PI0FAST_PREFIX/lib/libgcc_s.so.1"
)
resolved_preload=()
for lib in "${pi0fast_preload[@]}"; do
  if [[ -e "$lib" ]]; then
    resolved_preload+=("$lib")
  fi
done
if (( ${#resolved_preload[@]} > 0 )); then
  preload_joined="$(IFS=:; echo "${resolved_preload[*]}")"
  export LD_PRELOAD="${preload_joined}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

"$PI0FAST_PY" scripts/libero/check_pi0fast_env.py --require-cuda

PYTHONPATH=./roepbench:${HOME}/LIBERO "$PI0FAST_PY" scripts/run_model_adapter_eval.py \
  --model pi0_fast \
  --benchmark_backend libero \
  --libero_suite "$ROEP_LIBERO_SUITE" \
  --libero_task_name "$ROEP_LIBERO_TASK_NAME" \
  --stage cache \
  --task "$ROEP_LIBERO_TASK_NAME" \
  --namespace "${PI0FAST_NAMESPACE:-pi0fast_libero_canary_v1}" \
  --episodes "${PI0FAST_CANARY_EPISODES:-1}" \
  --seeds "${PI0FAST_CANARY_SEEDS:-0}" \
  --profiles "${PI0FAST_CANARY_PROFILES:-medium}" \
  --scenario "${PI0FAST_CANARY_SCENARIO:-natural_ood_medium}" \
  --action_mode_profile "${PI0FAST_ACTION_MODE_PROFILE:-libero_ee_rel_pose_gripper}" \
  --seed0_dir "${ROEP_SEED0_DIR:-$PROJECT_ROOT/results/libero_seed0}" \
  --device cuda \
  --max_steps "${PI0FAST_CANARY_MAX_STEPS:-300}" \
  --execute
