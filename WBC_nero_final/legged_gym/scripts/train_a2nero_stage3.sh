#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_RUN="${A2NERO_SOURCE_RUN:-Sep12_12-08-00_dr_stage1}"
SOURCE_ITER="${A2NERO_SOURCE_ITER:-39000}"
ADDITIONAL_ITERATIONS="${A2NERO_ADDITIONAL_ITERATIONS:-8000}"
GPU_ID="${A2NERO_GPU_ID:-0}"
NUM_ENVS="${A2NERO_NUM_ENVS:-4096}"
RUN_NAME="${A2NERO_STAGE3_RUN_NAME:-dr_stage3_minimal_from39000}"

if [[ -n "${A2NERO_PYTHON:-}" ]]; then
    PYTHON_BIN="${A2NERO_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python"
fi

SOURCE_CHECKPOINT="${REPO_ROOT}/logs/a2nero_pos_force/${SOURCE_RUN}/model_${SOURCE_ITER}.pt"
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Stage-3 source checkpoint does not exist: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi
if [[ "${RUN_NAME}" == "${SOURCE_RUN}" ]]; then
    echo "Stage-3 run name must not equal the immutable source run name" >&2
    exit 2
fi

SOURCE_HASH_BEFORE="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
SOURCE_MTIME_BEFORE="$(stat -c '%Y' "${SOURCE_CHECKPOINT}")"
EXPECTED_FINAL_ITER="$((SOURCE_ITER + ADDITIONAL_ITERATIONS))"
LAUNCH_LOG_DIR="${REPO_ROOT}/logs/a2nero_pos_force"
LAUNCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${A2NERO_STAGE3_LAUNCH_LOG:-${LAUNCH_LOG_DIR}/${LAUNCH_TIMESTAMP}_${RUN_NAME}.launch.log}"
mkdir -p "${LAUNCH_LOG_DIR}"

echo "Stage-3 source iteration: ${SOURCE_ITER}"
echo "Stage-3 additional iterations: ${ADDITIONAL_ITERATIONS}"
echo "Stage-3 expected final iteration: ${EXPECTED_FINAL_ITER}"
echo "Stage-3 source SHA256: ${SOURCE_HASH_BEFORE}"
echo "Stage-3 launch log: ${LAUNCH_LOG}"
export A2NERO_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
export A2NERO_SOURCE_SHA256="${SOURCE_HASH_BEFORE}"
export A2NERO_STAGE3_RUN_NAME="${RUN_NAME}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

set +e
"${PYTHON_BIN}" "${REPO_ROOT}/legged_gym/scripts/train_a2neroposforce.py" \
    --domain_rand_stage 3 \
    --resume \
    --load_run "${SOURCE_RUN}" \
    --checkpoint "${SOURCE_ITER}" \
    --run_name "${RUN_NAME}" \
    --max_iterations "${ADDITIONAL_ITERATIONS}" \
    --num_envs "${NUM_ENVS}" \
    --sim_device "cuda:${GPU_ID}" \
    --rl_device "cuda:${GPU_ID}" \
    --headless 2>&1 | tee "${LAUNCH_LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

echo "Stage-3 training exit status: ${TRAIN_STATUS}" | tee -a "${LAUNCH_LOG}"

SOURCE_HASH_AFTER="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
SOURCE_MTIME_AFTER="$(stat -c '%Y' "${SOURCE_CHECKPOINT}")"
if [[ "${SOURCE_HASH_BEFORE}" != "${SOURCE_HASH_AFTER}" || "${SOURCE_MTIME_BEFORE}" != "${SOURCE_MTIME_AFTER}" ]]; then
    echo "ERROR: immutable source checkpoint hash or mtime changed" >&2
    exit 3
fi
echo "Source checkpoint hash and mtime unchanged."
exit "${TRAIN_STATUS}"
