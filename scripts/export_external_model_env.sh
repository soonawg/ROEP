#!/usr/bin/env bash
# shellcheck shell=bash
#
# Usage:
#   source scripts/export_external_model_env.sh
#
# Identity-safe template exports:
# - Uses model-specific adapter runner scaffold (no PerAct fallback).
# - If adapter/weights are not ready, returns blocked (rc=42).

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Run: source scripts/export_external_model_env.sh" >&2
  exit 1
fi

export ROEP_PROJECT_ROOT="${ROEP_PROJECT_ROOT:-${HOME}/roepbench_project}"
export ROEP_EXTERNAL_GPU="${ROEP_EXTERNAL_GPU:-0}"
export ROEP_SAFETY_SHIELD_DEVICE="${ROEP_SAFETY_SHIELD_DEVICE:-cuda}"
export ROEP_SEED0_DIR="${ROEP_SEED0_DIR:-$ROEP_PROJECT_ROOT/external/peract/ckpts/multi/PERACT_BC/seed0}"
export ROEP_EXTERNAL_EVAL_PYTHON="${ROEP_EXTERNAL_EVAL_PYTHON:-}"
export ROEP_EXTERNAL_RUNTIME_BRIDGE="${ROEP_EXTERNAL_RUNTIME_BRIDGE:-0}"
export ROEP_BENCHMARK_BACKEND="${ROEP_BENCHMARK_BACKEND:-rlbench}"
export ROEP_LIBERO_SUITE="${ROEP_LIBERO_SUITE:-}"
export ROEP_LIBERO_TASK_NAME="${ROEP_LIBERO_TASK_NAME:-}"
export ROEP_LIBERO_BENCHMARK_NAME="${ROEP_LIBERO_BENCHMARK_NAME:-}"
export ROEP_LIBERO_TASK_ID="${ROEP_LIBERO_TASK_ID:-}"
export ROEP_LIBERO_INIT_STATES_PATH="${ROEP_LIBERO_INIT_STATES_PATH:-}"
export ROEP_LIBERO_RUNTIME_HOOK="${ROEP_LIBERO_RUNTIME_HOOK:-roepbench.libero.runtime_hook:run_libero_episodes}"

# Optional per-model launchers for heterogeneous conda envs.
# If unset, run_external_model_eval.py falls back to current python.
export ROEP_EXTERNAL_OCTO_BASE_PYTHON="${ROEP_EXTERNAL_OCTO_BASE_PYTHON:-}"
export ROEP_EXTERNAL_LAP_LIBERO_PYTHON="${ROEP_EXTERNAL_LAP_LIBERO_PYTHON:-}"
export ROEP_EXTERNAL_RDT_170M_PYTHON="${ROEP_EXTERNAL_RDT_170M_PYTHON:-}"
export ROEP_EXTERNAL_PI05_LIBERO_PYTHON="${ROEP_EXTERNAL_PI05_LIBERO_PYTHON:-}"
export ROEP_EXTERNAL_OPENVLA_7B_PYTHON="${ROEP_EXTERNAL_OPENVLA_7B_PYTHON:-}"
export ROEP_EXTERNAL_PI0_FAST_PYTHON="${ROEP_EXTERNAL_PI0_FAST_PYTHON:-}"
export ROEP_EXTERNAL_SMOLVLA_BASE_PYTHON="${ROEP_EXTERNAL_SMOLVLA_BASE_PYTHON:-}"
export ROEP_EXTERNAL_SPATIALVLA_PYTHON="${ROEP_EXTERNAL_SPATIALVLA_PYTHON:-}"
export ROEP_EXTERNAL_VLA_ADAPTER_PYTHON="${ROEP_EXTERNAL_VLA_ADAPTER_PYTHON:-}"
export ROEP_EXTERNAL_XVLA_PYTHON="${ROEP_EXTERNAL_XVLA_PYTHON:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python}"
# Backward-compatible aliases
export ROEP_EXTERNAL_OCTO_SMALL_PYTHON="${ROEP_EXTERNAL_OCTO_SMALL_PYTHON:-$ROEP_EXTERNAL_OCTO_BASE_PYTHON}"
export ROEP_EXTERNAL_LAP_PYTHON="${ROEP_EXTERNAL_LAP_PYTHON:-$ROEP_EXTERNAL_LAP_LIBERO_PYTHON}"
export ROEP_EXTERNAL_VIT_BC_LANG_PYTHON="${ROEP_EXTERNAL_VIT_BC_LANG_PYTHON:-$ROEP_EXTERNAL_RDT_170M_PYTHON}"
export ROEP_EXTERNAL_OPENVLA_PYTHON="${ROEP_EXTERNAL_OPENVLA_PYTHON:-$ROEP_EXTERNAL_OPENVLA_7B_PYTHON}"
# Keep model-specific bridge flags unset by default.
# When empty, runtime_bridge falls back to ROEP_EXTERNAL_RUNTIME_BRIDGE.
export ROEP_EXTERNAL_OCTO_BASE_RUNTIME_BRIDGE="${ROEP_EXTERNAL_OCTO_BASE_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_LAP_LIBERO_RUNTIME_BRIDGE="${ROEP_EXTERNAL_LAP_LIBERO_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_RDT_170M_RUNTIME_BRIDGE="${ROEP_EXTERNAL_RDT_170M_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_PI05_LIBERO_RUNTIME_BRIDGE="${ROEP_EXTERNAL_PI05_LIBERO_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_OPENVLA_7B_RUNTIME_BRIDGE="${ROEP_EXTERNAL_OPENVLA_7B_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_PI0_FAST_RUNTIME_BRIDGE="${ROEP_EXTERNAL_PI0_FAST_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_SMOLVLA_BASE_RUNTIME_BRIDGE="${ROEP_EXTERNAL_SMOLVLA_BASE_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_SPATIALVLA_RUNTIME_BRIDGE="${ROEP_EXTERNAL_SPATIALVLA_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_VLA_ADAPTER_RUNTIME_BRIDGE="${ROEP_EXTERNAL_VLA_ADAPTER_RUNTIME_BRIDGE:-}"
export ROEP_EXTERNAL_XVLA_RUNTIME_BRIDGE="${ROEP_EXTERNAL_XVLA_RUNTIME_BRIDGE:-1}"

# Checkpoint envs for adapter scaffold (paper-facing model slots)
export ROEP_OCTO_BASE_CKPT="${ROEP_OCTO_BASE_CKPT:-}"
export ROEP_LAP_LIBERO_CKPT="${ROEP_LAP_LIBERO_CKPT:-}"
export ROEP_RDT_170M_CKPT="${ROEP_RDT_170M_CKPT:-}"
export ROEP_PI05_LIBERO_CKPT="${ROEP_PI05_LIBERO_CKPT:-}"
export ROEP_OPENVLA_7B_CKPT="${ROEP_OPENVLA_7B_CKPT:-}"
export ROEP_PI0_FAST_CKPT="${ROEP_PI0_FAST_CKPT:-}"
export ROEP_SMOLVLA_BASE_CKPT="${ROEP_SMOLVLA_BASE_CKPT:-}"
export ROEP_SPATIALVLA_CKPT="${ROEP_SPATIALVLA_CKPT:-}"
export ROEP_VLA_ADAPTER_CKPT="${ROEP_VLA_ADAPTER_CKPT:-}"
export ROEP_XVLA_CKPT="${ROEP_XVLA_CKPT:-lerobot/xvla-libero}"
# Backward-compatible aliases
export ROEP_VIT_BC_LANG_CKPT="${ROEP_VIT_BC_LANG_CKPT:-$ROEP_RDT_170M_CKPT}"
export ROEP_OCTO_SMALL_CKPT="${ROEP_OCTO_SMALL_CKPT:-$ROEP_OCTO_BASE_CKPT}"
export ROEP_LAP_CKPT="${ROEP_LAP_CKPT:-$ROEP_LAP_LIBERO_CKPT}"
export ROEP_PI05_CKPT="${ROEP_PI05_CKPT:-$ROEP_PI05_LIBERO_CKPT}"
export ROEP_OPENVLA_CKPT="${ROEP_OPENVLA_CKPT:-$ROEP_OPENVLA_7B_CKPT}"
export ROEP_PI0_CKPT="${ROEP_PI0_CKPT:-$ROEP_PI0_FAST_CKPT}"
export ROEP_EXTERNAL_SHIELD_CKPT="${ROEP_EXTERNAL_SHIELD_CKPT:-}"
export ROEP_EXTERNAL_SHIELD_TAU="${ROEP_EXTERNAL_SHIELD_TAU:-}"
export ROEP_EXTERNAL_SHIELD_MDSP_MODE="${ROEP_EXTERNAL_SHIELD_MDSP_MODE:-lift_z_scale_xy}"
export ROEP_EXTERNAL_SHIELD_REFUSAL_RISK_THRESHOLD="${ROEP_EXTERNAL_SHIELD_REFUSAL_RISK_THRESHOLD:-0.995}"
export ROEP_EXTERNAL_SHIELD_VERIFY_PROJECTED_ACTION="${ROEP_EXTERNAL_SHIELD_VERIFY_PROJECTED_ACTION:-false}"
export ROEP_EXTERNAL_SHIELD_Z_LIFT="${ROEP_EXTERNAL_SHIELD_Z_LIFT:-0.05}"
export ROEP_EXTERNAL_SHIELD_XY_SCALE="${ROEP_EXTERNAL_SHIELD_XY_SCALE:-0.8}"
export ROEP_EXTERNAL_SHIELD_MAX_TRANSLATION_NORM="${ROEP_EXTERNAL_SHIELD_MAX_TRANSLATION_NORM:-1.05}"
export ROEP_EXTERNAL_OPENVLA_7B_SHIELD_CKPT="${ROEP_EXTERNAL_OPENVLA_7B_SHIELD_CKPT:-$ROEP_EXTERNAL_SHIELD_CKPT}"
export ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_CKPT="${ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_CKPT:-$ROEP_EXTERNAL_SHIELD_CKPT}"
export ROEP_EXTERNAL_SPATIALVLA_SHIELD_CKPT="${ROEP_EXTERNAL_SPATIALVLA_SHIELD_CKPT:-$ROEP_EXTERNAL_SHIELD_CKPT}"
export ROEP_EXTERNAL_VLA_ADAPTER_SHIELD_CKPT="${ROEP_EXTERNAL_VLA_ADAPTER_SHIELD_CKPT:-$ROEP_EXTERNAL_SHIELD_CKPT}"
export ROEP_EXTERNAL_XVLA_SHIELD_CKPT="${ROEP_EXTERNAL_XVLA_SHIELD_CKPT:-$ROEP_EXTERNAL_SHIELD_CKPT}"
export ROEP_EXTERNAL_OPENVLA_7B_SHIELD_TAU="${ROEP_EXTERNAL_OPENVLA_7B_SHIELD_TAU:-$ROEP_EXTERNAL_SHIELD_TAU}"
export ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_TAU="${ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_TAU:-$ROEP_EXTERNAL_SHIELD_TAU}"
export ROEP_EXTERNAL_SPATIALVLA_SHIELD_TAU="${ROEP_EXTERNAL_SPATIALVLA_SHIELD_TAU:-$ROEP_EXTERNAL_SHIELD_TAU}"
export ROEP_EXTERNAL_VLA_ADAPTER_SHIELD_TAU="${ROEP_EXTERNAL_VLA_ADAPTER_SHIELD_TAU:-$ROEP_EXTERNAL_SHIELD_TAU}"
export ROEP_EXTERNAL_XVLA_SHIELD_TAU="${ROEP_EXTERNAL_XVLA_SHIELD_TAU:-$ROEP_EXTERNAL_SHIELD_TAU}"

# Adapter implementation bindings (module.path:factory_or_class):
#   module.path:factory_or_class
export ROEP_OCTO_BASE_IMPL="${ROEP_OCTO_BASE_IMPL:-}"
export ROEP_LAP_LIBERO_IMPL="${ROEP_LAP_LIBERO_IMPL:-}"
export ROEP_RDT_170M_IMPL="${ROEP_RDT_170M_IMPL:-}"
export ROEP_PI05_LIBERO_IMPL="${ROEP_PI05_LIBERO_IMPL:-}"
export ROEP_OPENVLA_7B_IMPL="${ROEP_OPENVLA_7B_IMPL:-}"
export ROEP_PI0_FAST_IMPL="${ROEP_PI0_FAST_IMPL:-}"
export ROEP_SMOLVLA_BASE_IMPL="${ROEP_SMOLVLA_BASE_IMPL:-}"
export ROEP_SPATIALVLA_IMPL="${ROEP_SPATIALVLA_IMPL:-}"
export ROEP_VLA_ADAPTER_IMPL="${ROEP_VLA_ADAPTER_IMPL:-}"
export ROEP_XVLA_IMPL="${ROEP_XVLA_IMPL:-roepbench.model_plugins.xvla_plugin:build_adapter}"
# Backward-compatible aliases
export ROEP_VIT_BC_LANG_IMPL="${ROEP_VIT_BC_LANG_IMPL:-$ROEP_RDT_170M_IMPL}"
export ROEP_OCTO_SMALL_IMPL="${ROEP_OCTO_SMALL_IMPL:-$ROEP_OCTO_BASE_IMPL}"
export ROEP_LAP_IMPL="${ROEP_LAP_IMPL:-$ROEP_LAP_LIBERO_IMPL}"
export ROEP_PI05_IMPL="${ROEP_PI05_IMPL:-$ROEP_PI05_LIBERO_IMPL}"
export ROEP_OPENVLA_IMPL="${ROEP_OPENVLA_IMPL:-$ROEP_OPENVLA_7B_IMPL}"

export ROEP_EXTERNAL_OCTO_BASE_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model octo_base --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_LAP_LIBERO_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model lap_libero --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_PI05_LIBERO_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model pi05_libero --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_OCTO_BASE_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model octo_base --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_LAP_LIBERO_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model lap_libero --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_PI05_LIBERO_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model pi05_libero --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_RDT_170M_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model rdt_170m --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_RDT_170M_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model rdt_170m --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_OPENVLA_7B_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model openvla_7b --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_OPENVLA_7B_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model openvla_7b --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_PI0_FAST_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model pi0_fast --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_PI0_FAST_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model pi0_fast --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_SMOLVLA_BASE_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model smolvla_base --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_SMOLVLA_BASE_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model smolvla_base --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_SPATIALVLA_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model spatialvla --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_SPATIALVLA_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model spatialvla --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_VLA_ADAPTER_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model vla_adapter --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_VLA_ADAPTER_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model vla_adapter --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

export ROEP_EXTERNAL_XVLA_CACHE_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model xvla --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage cache --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"
export ROEP_EXTERNAL_XVLA_TAU_CMD="$(cat <<'EOF'
{eval_python} scripts/run_model_adapter_eval.py --model xvla --benchmark_backend "{benchmark_backend}" --libero_suite "{libero_suite}" --libero_task_name "{libero_task_name}" --libero_benchmark_name "{libero_benchmark_name}" --libero_task_id "{libero_task_id}" --libero_init_states_path "{libero_init_states_path}" --stage tau --task "{task}" --namespace "{namespace}" --episodes "{episodes}" --seeds "{seeds}" --profiles "{profiles}" --scenario "{scenario}" --action_mode_profile "{action_mode_profile}" --tau "{tau}" --seed0_dir "{seed0_dir}" --device cuda --execute
EOF
)"

# Backward-compatible command aliases
export ROEP_EXTERNAL_VIT_BC_LANG_CACHE_CMD="${ROEP_EXTERNAL_VIT_BC_LANG_CACHE_CMD:-$ROEP_EXTERNAL_RDT_170M_CACHE_CMD}"
export ROEP_EXTERNAL_VIT_BC_LANG_TAU_CMD="${ROEP_EXTERNAL_VIT_BC_LANG_TAU_CMD:-$ROEP_EXTERNAL_RDT_170M_TAU_CMD}"
export ROEP_EXTERNAL_OCTO_SMALL_CACHE_CMD="${ROEP_EXTERNAL_OCTO_SMALL_CACHE_CMD:-$ROEP_EXTERNAL_OCTO_BASE_CACHE_CMD}"
export ROEP_EXTERNAL_OCTO_SMALL_TAU_CMD="${ROEP_EXTERNAL_OCTO_SMALL_TAU_CMD:-$ROEP_EXTERNAL_OCTO_BASE_TAU_CMD}"
export ROEP_EXTERNAL_OPENVLA_CACHE_CMD="${ROEP_EXTERNAL_OPENVLA_CACHE_CMD:-$ROEP_EXTERNAL_OPENVLA_7B_CACHE_CMD}"
export ROEP_EXTERNAL_OPENVLA_TAU_CMD="${ROEP_EXTERNAL_OPENVLA_TAU_CMD:-$ROEP_EXTERNAL_OPENVLA_7B_TAU_CMD}"

echo "[ok] identity-safe external templates loaded."
echo "  - model-specific runner scaffold: scripts/run_model_adapter_eval.py"
echo "  - no PerAct fallback for external models"
echo "  - set checkpoints:"
echo "      ROEP_OCTO_BASE_CKPT"
echo "      ROEP_LAP_LIBERO_CKPT"
echo "      ROEP_RDT_170M_CKPT"
echo "      ROEP_PI05_LIBERO_CKPT"
echo "      ROEP_OPENVLA_7B_CKPT"
echo "      ROEP_PI0_FAST_CKPT"
echo "      ROEP_SMOLVLA_BASE_CKPT"
echo "      ROEP_SPATIALVLA_CKPT"
echo "      ROEP_VLA_ADAPTER_CKPT"
echo "      ROEP_XVLA_CKPT"
echo "      ROEP_EXTERNAL_SHIELD_CKPT"
echo "      ROEP_EXTERNAL_OPENVLA_7B_SHIELD_CKPT / ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_CKPT"
echo "      ROEP_EXTERNAL_XVLA_SHIELD_CKPT"
echo "      ROEP_EXTERNAL_SHIELD_TAU"
echo "      ROEP_EXTERNAL_OPENVLA_7B_SHIELD_TAU / ROEP_EXTERNAL_SMOLVLA_BASE_SHIELD_TAU"
echo "      ROEP_EXTERNAL_XVLA_SHIELD_TAU"
echo "      ROEP_SEED0_DIR"
echo "  - optional model-specific python launchers:"
echo "      ROEP_EXTERNAL_OCTO_BASE_PYTHON"
echo "      ROEP_EXTERNAL_LAP_LIBERO_PYTHON"
echo "      ROEP_EXTERNAL_RDT_170M_PYTHON"
echo "      ROEP_EXTERNAL_PI05_LIBERO_PYTHON"
echo "      ROEP_EXTERNAL_OPENVLA_7B_PYTHON"
echo "      ROEP_EXTERNAL_PI0_FAST_PYTHON"
echo "      ROEP_EXTERNAL_SMOLVLA_BASE_PYTHON"
echo "      ROEP_EXTERNAL_SPATIALVLA_PYTHON"
echo "      ROEP_EXTERNAL_VLA_ADAPTER_PYTHON"
echo "      ROEP_EXTERNAL_XVLA_PYTHON"
echo "  - optional evaluator launcher (defaults to current python):"
echo "      ROEP_EXTERNAL_EVAL_PYTHON"
echo "  - benchmark backend routing:"
echo "      ROEP_BENCHMARK_BACKEND=rlbench|libero"
echo "      ROEP_LIBERO_SUITE / ROEP_LIBERO_TASK_NAME / ROEP_LIBERO_BENCHMARK_NAME"
echo "      ROEP_LIBERO_RUNTIME_HOOK=${ROEP_LIBERO_RUNTIME_HOOK}"
echo "      source scripts/libero/export_libero_v1_defaults.sh   # fixed v1 target"
echo "  - optional runtime bridge mode (evaluator env + model env split):"
echo "      ROEP_EXTERNAL_RUNTIME_BRIDGE=1"
echo "  - set adapter implementation bindings:"
echo "      ROEP_OCTO_BASE_IMPL=module.path:Factory"
echo "      ROEP_LAP_LIBERO_IMPL=module.path:Factory"
echo "      ROEP_RDT_170M_IMPL=module.path:Factory"
echo "      ROEP_PI05_LIBERO_IMPL=module.path:Factory"
echo "      ROEP_OPENVLA_7B_IMPL=module.path:Factory"
echo "      ROEP_PI0_FAST_IMPL=module.path:Factory"
echo "      ROEP_SMOLVLA_BASE_IMPL=module.path:Factory"
echo "      ROEP_SPATIALVLA_IMPL=module.path:Factory"
echo "      ROEP_VLA_ADAPTER_IMPL=module.path:Factory"
echo "      ROEP_XVLA_IMPL=module.path:Factory"
echo "  - RDT campaign recommendation:"
echo "      ROEP_RDT_170M_TEXT_EMBED_HOOK=roepbench.model_plugins.rdt_text_embed_hook:embed_text"
echo "      ROEP_RDT_170M_TEXT_EMBED_INDEX=/abs/path/to/embed_index.json"
echo "      ROEP_RDT_170M_REPO=/abs/path/to/RoboticsDiffusionTransformer"
