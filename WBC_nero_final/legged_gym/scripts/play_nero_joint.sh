#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="$(command -v "${NERO_PYTHON:-python3}")"
export LD_LIBRARY_PATH="$(dirname "$PYTHON_BIN")/../lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/nero-fixed-torch-extensions}"
exec "$PYTHON_BIN" legged_gym/scripts/play_nero_joint.py "$@"
