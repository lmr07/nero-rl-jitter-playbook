import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import numpy as np
from datetime import datetime
import sys
sys.path.append(unitree_rl_gym_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.envs.a2.a2nero_pos_force_config import (
    apply_a2nero_domain_rand_stage,
    load_model39000_identified_profile,
)
from legged_gym.utils import get_args, task_registry
import torch

_IDENTIFIED_PROFILE_SOURCE = load_model39000_identified_profile()["source"]
IDENTIFIED_SOURCE_RUN = _IDENTIFIED_PROFILE_SOURCE["run"]
IDENTIFIED_SOURCE_ITERATION = int(_IDENTIFIED_PROFILE_SOURCE["iteration"])


def _validate_model39000_continuation_launch(args, env_cfg, train_cfg):
    stage = int(getattr(args, "domain_rand_stage", 1))
    if stage not in (3, 4, 5):
        return
    if not args.resume:
        raise ValueError("model39000 continuation must use --resume")
    if stage in (3, 4) and args.load_run != IDENTIFIED_SOURCE_RUN:
        raise ValueError(
            f"selected Stage-3/identified Stage-1 source run must be "
            f"{IDENTIFIED_SOURCE_RUN}, got {args.load_run}"
        )
    if stage in (3, 4) and int(args.checkpoint) != IDENTIFIED_SOURCE_ITERATION:
        raise ValueError(
            "Stage-3/identified Stage-1 must resume the selected source "
            f"model_{IDENTIFIED_SOURCE_ITERATION}.pt"
        )
    if stage == 5 and (not args.load_run or int(args.checkpoint) <= IDENTIFIED_SOURCE_ITERATION):
        raise ValueError(
            "identified Stage-2 must resume an explicitly accepted identified Stage-1 "
            f"checkpoint newer than model_{IDENTIFIED_SOURCE_ITERATION}.pt"
        )
    if stage == 5 and args.load_run == IDENTIFIED_SOURCE_RUN:
        raise ValueError(
            "identified Stage-2 must not resume the original Stage-1 run"
        )
    if args.run_name == args.load_run:
        raise ValueError("Continuation output run must not overwrite the source run")
    immutable = {
        "num_single_obs": (env_cfg.env.num_single_obs, 85),
        "frame_stack": (env_cfg.env.frame_stack, 16),
        "num_pred_obs": (env_cfg.env.num_pred_obs, 27),
        "num_actions": (env_cfg.env.num_actions, 19),
        "action_scale": (env_cfg.control.action_scale, 0.25),
        "temporal_encoder": (train_cfg.policy.temporal_encoder, "mlp"),
        "adaptation_num_single_obs": (train_cfg.policy.adaptation_num_single_obs, 85),
        "latent_dim": (train_cfg.policy.latent_dim, 32),
    }
    mismatches = {
        name: values for name, values in immutable.items() if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"model_39000 85x16/19D ABI changed: {mismatches}")

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = apply_a2nero_domain_rand_stage(env_cfg, train_cfg, getattr(args, "domain_rand_stage", 1))
    _validate_model39000_continuation_launch(args, env_cfg, train_cfg)
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.headless = True
    args.task = "a2nero_pos_force"
    train(args)
