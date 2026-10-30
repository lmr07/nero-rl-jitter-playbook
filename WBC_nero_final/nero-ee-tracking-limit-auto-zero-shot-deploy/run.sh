#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DEPLOY_DIR="$PWD"
PYTHON_BIN="$(command -v "${NERO_DEPLOY_PYTHON:-python3}")"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Nero deployment Python environment missing: $PYTHON_BIN" >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DEPLOY_DIR"
export LD_LIBRARY_PATH="${NERO_DEPLOY_LIB_DIR:-$(dirname "$PYTHON_BIN")/../lib}"
exec "$PYTHON_BIN" -m nero_only.run "$@"
