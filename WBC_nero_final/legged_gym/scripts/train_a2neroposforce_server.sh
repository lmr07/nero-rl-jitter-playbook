#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_MODE="${WANDB_MODE:-online}"
# Keep full Episode/* curves in local TensorBoard logs. Do not stream them all
# to wandb by default, otherwise the file stream can build a long backlog.
WANDB_SYNC_TENSORBOARD="${WANDB_SYNC_TENSORBOARD:-false}"
WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
WANDB_LOG_EPISODE_METRICS="${WANDB_LOG_EPISODE_METRICS:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export WANDB_MODE
export WANDB_SYNC_TENSORBOARD
export WANDB_CONSOLE
export WANDB_LOG_EPISODE_METRICS
export PYTORCH_CUDA_ALLOC_CONF

python3 -u legged_gym/scripts/train_a2neroposforce.py \
  --task a2nero_pos_force \
  --headless \
  --sim_device "cuda:${GPU_ID}" \
  --rl_device "cuda:${GPU_ID}" \
  --num_envs "${NUM_ENVS}" \
  "$@"
