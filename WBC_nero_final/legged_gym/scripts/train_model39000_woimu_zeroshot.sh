#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="${1:-stage1}"
GPU_ID="${A2NERO_GPU_ID:-0}"
NUM_ENVS="${A2NERO_NUM_ENVS:-4096}"

if [[ -n "${A2NERO_PYTHON:-}" ]]; then
    PYTHON_BIN="${A2NERO_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python3"
fi

case "${PHASE}" in
    stage1)
        DOMAIN_STAGE=4
        SOURCE_RUN="Sep12_12-08-00_dr_stage1"
        SOURCE_ITER=39000
        ADDITIONAL_ITERATIONS="${A2NERO_ADDITIONAL_ITERATIONS:-8000}"
        RUN_NAME="${A2NERO_RUN_NAME:-model39000_woimu_identified_stage1}"
        ;;
    stage2)
        DOMAIN_STAGE=5
        if [[ -z "${A2NERO_SOURCE_RUN:-}" || -z "${A2NERO_SOURCE_ITER:-}" ]]; then
            echo "Stage-2 requires A2NERO_SOURCE_RUN and A2NERO_SOURCE_ITER from the accepted Stage-1 run." >&2
            exit 2
        fi
        SOURCE_RUN="${A2NERO_SOURCE_RUN}"
        SOURCE_ITER="${A2NERO_SOURCE_ITER}"
        ADDITIONAL_ITERATIONS="${A2NERO_ADDITIONAL_ITERATIONS:-12000}"
        RUN_NAME="${A2NERO_RUN_NAME:-model39000_woimu_identified_stage2}"
        ;;
    *)
        echo "Usage: $0 stage1|stage2" >&2
        exit 2
        ;;
esac

SOURCE_CHECKPOINT="${REPO_ROOT}/logs/a2nero_pos_force/${SOURCE_RUN}/model_${SOURCE_ITER}.pt"
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Source checkpoint does not exist: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi
if [[ "${RUN_NAME}" == "${SOURCE_RUN}" ]]; then
    echo "Output run name must differ from the immutable source run." >&2
    exit 2
fi

SOURCE_HASH_BEFORE="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
SOURCE_MTIME_BEFORE="$(stat -c '%Y' "${SOURCE_CHECKPOINT}")"
export A2NERO_SOURCE_RUN="${SOURCE_RUN}"
export A2NERO_SOURCE_ITER="${SOURCE_ITER}"
export A2NERO_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
export A2NERO_SOURCE_SHA256="${SOURCE_HASH_BEFORE}"
export A2NERO_ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS}"
export A2NERO_STAGE3_RUN_NAME="${RUN_NAME}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "model39000 zero-shot continuation phase: ${PHASE}"
echo "source: ${SOURCE_CHECKPOINT}"
echo "source SHA256: ${SOURCE_HASH_BEFORE}"
echo "additional iterations: ${ADDITIONAL_ITERATIONS}"
echo "output run name: ${RUN_NAME}"

"${PYTHON_BIN}" "${REPO_ROOT}/legged_gym/scripts/train_a2neroposforce.py" \
    --domain_rand_stage "${DOMAIN_STAGE}" \
    --resume \
    --load_run "${SOURCE_RUN}" \
    --checkpoint "${SOURCE_ITER}" \
    --run_name "${RUN_NAME}" \
    --max_iterations "${ADDITIONAL_ITERATIONS}" \
    --num_envs "${NUM_ENVS}" \
    --sim_device "cuda:${GPU_ID}" \
    --rl_device "cuda:${GPU_ID}" \
    --headless

SOURCE_HASH_AFTER="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
SOURCE_MTIME_AFTER="$(stat -c '%Y' "${SOURCE_CHECKPOINT}")"
if [[ "${SOURCE_HASH_BEFORE}" != "${SOURCE_HASH_AFTER}" || "${SOURCE_MTIME_BEFORE}" != "${SOURCE_MTIME_AFTER}" ]]; then
    echo "ERROR: source checkpoint changed during continuation training." >&2
    exit 3
fi
echo "PASS: source checkpoint hash and mtime are unchanged."
