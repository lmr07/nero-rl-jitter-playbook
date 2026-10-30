#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_PYTHON="$(command -v "${NERO_TRAIN_PYTHON:-python3}")"
DEPLOY_PYTHON="$(command -v "${NERO_DEPLOY_PYTHON:-python3}")"
export LD_LIBRARY_PATH="$(dirname "$TRAIN_PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/nero-ee-tracking-torch-extensions}"
export OMP_NUM_THREADS=1
RUN_NAME='Oct27_23-25-59_limit_continuation_10000'
RUN="logs/nero_ee_tracking_limit_v3/$RUN_NAME"
ITERATION="$($TRAIN_PYTHON -c 'from pathlib import Path; import re; run=Path("logs/nero_ee_tracking_limit_v3/Oct27_23-25-59_limit_continuation_10000"); print(max(int(m.group(1)) for p in run.iterdir() if (m:=re.fullmatch(r"model_(\d+)\.pt", p.name))))')"
echo "Preparing $RUN/model_${ITERATION}.pt"
"$TRAIN_PYTHON" nero-ee-tracking-limit-auto-zero-shot-deploy/tests/evaluate_checkpoint.py \
  --run "$RUN_NAME" --checkpoint "$ITERATION" --output nero-ee-tracking-limit-auto-zero-shot-deploy/assets/sim_gate.json \
  --sim_device cpu --rl_device cpu --headless
"$TRAIN_PYTHON" nero-ee-tracking-limit-auto-zero-shot-deploy/scripts/export_policy.py --run "$RUN_NAME" --checkpoint "$ITERATION"
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-auto-zero-shot-deploy/run.sh
env -u LD_LIBRARY_PATH "$DEPLOY_PYTHON" \
  -m unittest discover -s nero-ee-tracking-limit-auto-zero-shot-deploy/tests -p test_offline.py
"$TRAIN_PYTHON" nero-ee-tracking-limit-auto-zero-shot-deploy/tests/check_sim_parity.py \
  --sim_device cpu --rl_device cpu --headless
