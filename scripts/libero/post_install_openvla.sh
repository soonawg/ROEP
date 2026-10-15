#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPENVLA_REPO="${OPENVLA_REPO:-$PROJECT_ROOT/external/openvla}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_ROOT/external/LIBERO}"
ROEP_OPENVLA_BITSANDBYTES_VERSION="${ROEP_OPENVLA_BITSANDBYTES_VERSION:-0.43.1}"
export PIP_NO_BUILD_ISOLATION=1

if [[ ! -d "$OPENVLA_REPO" ]]; then
  echo "[error] OPENVLA_REPO not found: $OPENVLA_REPO" >&2
  echo "Clone the official OpenVLA repo first." >&2
  exit 1
fi

if [[ ! -d "$LIBERO_REPO" ]]; then
  echo "[error] LIBERO_REPO not found: $LIBERO_REPO" >&2
  echo "Clone the official LIBERO repo first." >&2
  exit 1
fi

pip install -e "$LIBERO_REPO"

cd "$OPENVLA_REPO"
pip install \
  "accelerate>=0.25.0,<1" \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla" \
  "draccus==0.8.0" \
  "json-numpy" \
  "jsonlines" \
  "matplotlib" \
  "peft==0.11.1" \
  "protobuf" \
  "rich" \
  "sentencepiece==0.1.99" \
  "tensorflow==2.15.0" \
  "tensorflow_datasets==4.9.3" \
  "tensorflow_graphics==2021.12.3" \
  "tokenizers==0.19.1" \
  "timm==0.9.10" \
  "transformers==4.40.1" \
  "wandb"
pip install -e . --no-deps
pip install -r experiments/robot/libero/libero_requirements.txt
pip install --no-deps "bitsandbytes==${ROEP_OPENVLA_BITSANDBYTES_VERSION}"

echo "[done] OpenVLA + LIBERO evaluation packages installed in current env."
echo "       pinned stack: torch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 transformers=4.40.1 timm=0.9.10 tokenizers=0.19.1"
echo "       bitsandbytes=${ROEP_OPENVLA_BITSANDBYTES_VERSION} installed with --no-deps to avoid torch stack drift."
