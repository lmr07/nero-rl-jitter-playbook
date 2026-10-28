#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_RUN="${A2NERO_SOURCE_RUN:-Sep12_12-08-00_dr_stage1}"
SOURCE_ITER="${A2NERO_SOURCE_ITER:-39000}"
GPU_ID="${A2NERO_GPU_ID:-0}"
RUN_NAME="${A2NERO_SMOKE_RUN_NAME:-dr_stage3_minimal_smoke_from39000}"

export WANDB_MODE=disabled
export A2NERO_SOURCE_RUN="${SOURCE_RUN}"
export A2NERO_SOURCE_ITER="${SOURCE_ITER}"
export A2NERO_ADDITIONAL_ITERATIONS=2
export A2NERO_GPU_ID="${GPU_ID}"
export A2NERO_NUM_ENVS=64
export A2NERO_STAGE3_RUN_NAME="${RUN_NAME}"

bash "${SCRIPT_DIR}/train_a2nero_stage3.sh"

EXPECTED_ITER="$((SOURCE_ITER + 2))"
RUN_DIR="$(find "${REPO_ROOT}/logs/a2nero_pos_force" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${RUN_DIR}" ]]; then
    echo "Smoke run directory was not created" >&2
    exit 4
fi
CHECKPOINT="${RUN_DIR}/model_${EXPECTED_ITER}.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Expected smoke checkpoint missing: ${CHECKPOINT}" >&2
    exit 4
fi

if [[ -n "${A2NERO_PYTHON:-}" ]]; then
    PYTHON_BIN="${A2NERO_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python"
fi
"${PYTHON_BIN}" -c 'import os, sys, torch; p=sys.argv[1]; expected=int(sys.argv[2]); c=torch.load(p, map_location="cpu"); actual=int(c["iter"]); assert actual == expected, (p, actual, expected); required={"optimizer_state_dict", "adaptation_module_optimizer_state_dict", "stage3_curriculum_state", "stage3_manifest", "stage3_config"}; missing=required-set(c); assert not missing, missing; print("PASS smoke checkpoint metadata:", os.path.basename(p), "iter=", actual)' "${CHECKPOINT}" "${EXPECTED_ITER}"
echo "PASS 64-env / 2-additional-iteration Stage-3 smoke test: ${RUN_DIR}"
