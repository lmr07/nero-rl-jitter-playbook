#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${A2NERO_EVAL_RUN:?Set A2NERO_EVAL_RUN to the candidate run directory}"
: "${A2NERO_EVAL_ITER:?Set A2NERO_EVAL_ITER to the candidate checkpoint iteration}"
DOMAIN_STAGE="${A2NERO_EVAL_STAGE:-5}"
NUM_ENVS="${A2NERO_EVAL_NUM_ENVS:-20}"
PYTHON_BIN="${A2NERO_PYTHON:-python}"
CHECKPOINT="${REPO_ROOT}/logs/a2nero_pos_force/${A2NERO_EVAL_RUN}/model_${A2NERO_EVAL_ITER}.pt"
OUTPUT_DIR="${A2NERO_EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/model39000_woimu_zeroshot_eval}"

[[ -f "${CHECKPOINT}" ]] || { echo "Missing ${CHECKPOINT}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"
export MODEL39000_EXPECT_CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
export MODEL39000_ISAAC_GAIT_NUM_ENVS="${NUM_ENVS}"
export MODEL39000_ISAAC_GAIT_FULL_IDENTIFIED="${A2NERO_EVAL_FULL_IDENTIFIED:-1}"

run_direction() {
    local label="$1"
    local command="$2"
    export MODEL39000_ISAAC_GAIT_COMMAND="${command}"
    export MODEL39000_ISAAC_GAIT_AUDIT_OUTPUT="${OUTPUT_DIR}/${label}.json"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_model39000_isaacgym_gait_observations.py" \
        --task a2nero_pos_force \
        --resume \
        --load_run "${A2NERO_EVAL_RUN}" \
        --checkpoint "${A2NERO_EVAL_ITER}" \
        --domain_rand_stage "${DOMAIN_STAGE}" \
        --num_envs "${NUM_ENVS}" \
        --headless
}

run_direction W "0.3,0,0"
run_direction S "-0.3,0,0"
run_direction Z "0,0.2,0"
run_direction C "0,-0.2,0"
run_direction A "0,0,0.3"
run_direction D "0,0,-0.3"

echo "Six-direction identified-plant audits written to ${OUTPUT_DIR}"
