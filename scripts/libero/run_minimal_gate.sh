#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/export_external_model_env.sh" >/dev/null
source "$PROJECT_ROOT/scripts/libero/export_libero_v1_defaults.sh" >/dev/null

: "${ROEP_LIBERO_SUITE:?Set ROEP_LIBERO_SUITE}"
: "${ROEP_LIBERO_TASK_NAME:?Set ROEP_LIBERO_TASK_NAME}"

export ROEP_BENCHMARK_BACKEND=libero
export ROEP_LIBERO_RUNTIME_HOOK="${ROEP_LIBERO_RUNTIME_HOOK:-roepbench.libero.runtime_hook:run_libero_episodes}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELS_CSV="${MODELS_CSV:-openvla_7b,smolvla_base,xvla}"
export CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/libero_research_v2.yaml}"

# Minimal validation gate:
# - quick clean admission for capability sanity
# - quick perturbation to verify the OOD path
# - recovery/full preflight to fail fast on missing shield prerequisites
export PROFILES="${PROFILES:-medium}"
export RESEARCH_PROFILES="${RESEARCH_PROFILES:-medium}"
export SMOKE_EPISODES="${SMOKE_EPISODES:-1}"
export SMOKE_SEEDS="${SMOKE_SEEDS:-0}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-120}"
export ADMISSION_EPISODES="${ADMISSION_EPISODES:-2}"
export ADMISSION_SEEDS="${ADMISSION_SEEDS:-0,1}"
export ADMISSION_MAX_STEPS="${ADMISSION_MAX_STEPS:-120}"
export PERTURB_EPISODES="${PERTURB_EPISODES:-1}"
export PERTURB_SEEDS="${PERTURB_SEEDS:-0}"
export PERTURB_MAX_STEPS="${PERTURB_MAX_STEPS:-120}"
export NOISE_EPISODES="${NOISE_EPISODES:-1}"
export NOISE_SEEDS="${NOISE_SEEDS:-0}"
export NOISE_MAX_STEPS="${NOISE_MAX_STEPS:-120}"
export RECOVERY_EPISODES="${RECOVERY_EPISODES:-1}"
export RECOVERY_SEEDS="${RECOVERY_SEEDS:-0}"
export RECOVERY_MAX_STEPS="${RECOVERY_MAX_STEPS:-120}"

export RUN_CORE_CANARY="${RUN_CORE_CANARY:-0}"
export RUN_OFFICIAL_CANARIES="${RUN_OFFICIAL_CANARIES:-0}"
export RUN_SMOKE="${RUN_SMOKE:-1}"
export RUN_VALIDATE="${RUN_VALIDATE:-1}"
export RUN_ADMISSION="${RUN_ADMISSION:-1}"
export RUN_PERTURB="${RUN_PERTURB:-1}"
export RUN_NOISE_FLOOR="${RUN_NOISE_FLOOR:-0}"
export RUN_RECOVERY="${RUN_RECOVERY:-1}"
export RUN_FULL="${RUN_FULL:-1}"
export RUN_MATRIX=0
export REQUIRE_ALL_MODEL_CANARIES_FOR_ADMISSION="${REQUIRE_ALL_MODEL_CANARIES_FOR_ADMISSION:-0}"
export CONTINUE_ON_STAGE_ERROR="${CONTINUE_ON_STAGE_ERROR:-1}"
export REUSE_EXISTING_PHASE_OUTPUTS="${REUSE_EXISTING_PHASE_OUTPUTS:-1}"

export ADMISSION_ROOT="${ADMISSION_ROOT:-$PROJECT_ROOT/results/external_model_validation}"
export RESEARCH_ROOT="${RESEARCH_ROOT:-$PROJECT_ROOT/results/libero_validation}"
export NOISE_ROOT="${NOISE_ROOT:-$PROJECT_ROOT/results/libero_validation_noise}"
export ADMISSION_NS_PREFIX="${ADMISSION_NS_PREFIX:-libero_reset_min_gate_admission_v1}"
export RESEARCH_NS_PREFIX="${RESEARCH_NS_PREFIX:-libero_reset_min_gate_research_v1}"
export NOISE_NS_PREFIX="${NOISE_NS_PREFIX:-libero_reset_min_gate_noise_v1}"

echo "[minimal-gate] starting LIBERO validation profile"
echo "  MODELS_CSV=$MODELS_CSV"
echo "  TASK=$ROEP_LIBERO_TASK_NAME SUITE=$ROEP_LIBERO_SUITE"
echo "  admission: episodes=$ADMISSION_EPISODES seeds=$ADMISSION_SEEDS"
echo "  perturb:   episodes=$PERTURB_EPISODES seeds=$PERTURB_SEEDS profiles=$RESEARCH_PROFILES"
echo "  recovery:  episodes=$RECOVERY_EPISODES seeds=$RECOVERY_SEEDS full=$RUN_FULL"
echo "  admission_root=$ADMISSION_ROOT"
echo "  research_root=$RESEARCH_ROOT"

bash "$PROJECT_ROOT/scripts/run_external_libero_pipeline.sh"
