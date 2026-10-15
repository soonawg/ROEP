#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_EXE="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-roep_openvla_libero}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/envs/roep_openvla_libero.yml}"
OPENVLA_REPO="${OPENVLA_REPO:-$PROJECT_ROOT/external/openvla}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_ROOT/external/LIBERO}"
RECREATE_ENV="${RECREATE_ENV:-1}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[error] missing ${label}: $path" >&2
    exit 1
  fi
}

classify_checker_failure() {
  local log_path="$1"
  local categories=()

  if grep -q "version mismatch" "$log_path"; then
    categories+=("version_drift")
  fi
  if grep -q "import libero failed" "$log_path"; then
    categories+=("missing_libero")
  fi
  if grep -Eq "import torchvision failed|import torchaudio failed|torchvision nms probe failed" "$log_path"; then
    categories+=("torchvision_torchaudio_abi")
  fi
  if grep -q "import prismatic failed" "$log_path"; then
    categories+=("prismatic_import")
  fi

  if [[ "${#categories[@]}" -eq 0 ]]; then
    categories+=("uncategorized")
  fi
  printf '%s\n' "${categories[*]}"
}

require_path "$CONDA_EXE" "conda executable"
require_path "$ENV_FILE" "OpenVLA env spec"
require_path "$OPENVLA_REPO" "OpenVLA repo"
require_path "$LIBERO_REPO" "LIBERO repo"

cd "$PROJECT_ROOT"
set +u
eval "$("$CONDA_EXE" shell.bash hook)"
set -u

if [[ "$RECREATE_ENV" == "1" ]] && conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[openvla-rebuild] removing existing env: $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y
fi

if [[ "$RECREATE_ENV" == "1" ]]; then
  echo "[openvla-rebuild] creating env from $ENV_FILE"
  conda env create -f "$ENV_FILE"
fi

set +u
conda activate "$ENV_NAME"
set -u
export OPENVLA_REPO
export LIBERO_REPO

echo "[openvla-rebuild] running post-install"
bash "$PROJECT_ROOT/scripts/libero/post_install_openvla.sh"

echo "[openvla-rebuild] running strict env checker"
checker_log="$(mktemp)"
set +e
python "$PROJECT_ROOT/scripts/libero/check_openvla_env.py" >"$checker_log" 2>&1
checker_rc=$?
set -e
cat "$checker_log"
if [[ "$checker_rc" -ne 0 ]]; then
  categories="$(classify_checker_failure "$checker_log")"
  echo "[blocked] OpenVLA env checker failed: $categories" >&2
  rm -f "$checker_log"
  exit "$checker_rc"
fi
rm -f "$checker_log"

echo "[done] $ENV_NAME rebuilt and verified."
