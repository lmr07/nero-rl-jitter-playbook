import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import numpy as np
from datetime import datetime
import sys
sys.path.append(unitree_rl_gym_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.headless = False
    args.task = "b2z1_pos_force"
    train(args)
