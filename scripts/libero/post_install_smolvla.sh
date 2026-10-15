#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

LEROBOT_REPO="${LEROBOT_REPO:-}"
export PIP_NO_BUILD_ISOLATION=1

if ! cmake --version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
[error] cmake is not runnable in the current env.
This usually means the pip-installed cmake wrapper is broken.

Repair the env with:
  python -m pip uninstall -y cmake
  conda install -y -c conda-forge cmake ninja

Then rerun:
  bash scripts/libero/post_install_smolvla.sh
EOF
  exit 1
fi

if [[ -n "$LEROBOT_REPO" ]]; then
  if [[ ! -d "$LEROBOT_REPO" ]]; then
    echo "[error] LEROBOT_REPO not found: $LEROBOT_REPO" >&2
    exit 1
  fi
  if ! python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
  then
    echo "[error] Current env Python is too old for latest LeRobot. Recreate roep_smolvla_libero with Python 3.12." >&2
    exit 1
  fi
  cd "$LEROBOT_REPO"
  pip install -e ".[smolvla,libero]"
else
  if ! python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
  then
    echo "[error] Current env Python is too old for latest LeRobot. Recreate roep_smolvla_libero with Python 3.12." >&2
    exit 1
  fi
  pip install "lerobot[smolvla,libero]"
fi

echo "[done] SmolVLA/LeRobot + LIBERO evaluation packages installed in current env."
