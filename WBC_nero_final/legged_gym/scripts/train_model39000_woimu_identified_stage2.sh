#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${A2NERO_SOURCE_RUN:?Set A2NERO_SOURCE_RUN to an accepted model39000 identified Stage-1 run}"
: "${A2NERO_SOURCE_ITER:?Set A2NERO_SOURCE_ITER to its accepted checkpoint iteration}"
RUN_NAME="${A2NERO_RUN_NAME:-model39000_woimu_identified_stage2}"
ADDITIONAL_ITERATIONS="${A2NERO_ADDITIONAL_ITERATIONS:-12000}"
NUM_ENVS="${A2NERO_NUM_ENVS:-4096}"
GPU_ID="${A2NERO_GPU_ID:-0}"
PYTHON_BIN="${A2NERO_PYTHON:-python}"

[[ "${A2NERO_SOURCE_RUN}" != "Sep12_12-08-00_dr_stage1" ]] || { echo "Stage-2 requires an accepted identified Stage-1 source" >&2; exit 2; }
[[ "${RUN_NAME}" != "${A2NERO_SOURCE_RUN}" ]] || { echo "Output run must not overwrite source" >&2; exit 2; }
SOURCE_CHECKPOINT="${REPO_ROOT}/logs/a2nero_pos_force/${A2NERO_SOURCE_RUN}/model_${A2NERO_SOURCE_ITER}.pt"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing ${SOURCE_CHECKPOINT}" >&2; exit 2; }
HASH_BEFORE="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
MTIME_BEFORE="$(stat -c '%Y' "${SOURCE_CHECKPOINT}")"
export A2NERO_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
export A2NERO_SOURCE_SHA256="${HASH_BEFORE}"
export A2NERO_ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS}"
export A2NERO_STAGE3_RUN_NAME="${RUN_NAME}"
verify_source_unchanged() {
  [[ "${HASH_BEFORE}" == "$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')" ]] || { echo "Source checkpoint content changed" >&2; return 3; }
  [[ "${MTIME_BEFORE}" == "$(stat -c '%Y' "${SOURCE_CHECKPOINT}")" ]] || { echo "Source checkpoint mtime changed" >&2; return 3; }
}
trap verify_source_unchanged EXIT

"${PYTHON_BIN}" "${REPO_ROOT}/legged_gym/scripts/train_a2neroposforce.py" \
  --domain_rand_stage 5 --resume --load_run "${A2NERO_SOURCE_RUN}" --checkpoint "${A2NERO_SOURCE_ITER}" \
  --run_name "${RUN_NAME}" --max_iterations "${ADDITIONAL_ITERATIONS}" --num_envs "${NUM_ENVS}" \
  --sim_device "cuda:${GPU_ID}" --rl_device "cuda:${GPU_ID}" --headless
