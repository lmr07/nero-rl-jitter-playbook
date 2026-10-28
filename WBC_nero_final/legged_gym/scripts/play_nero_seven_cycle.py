"""Visualize a seven-cycle checkpoint with its saved training configuration."""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym  # must precede torch
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.isaacgym_utils import sphere2cart


class SevenCyclePolicy(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.adaptation_encoder_module
        self.actor = actor_critic.actor_body

    def forward(self, history):
        return self.actor(torch.cat((history[:, -39:], self.encoder(history)), dim=-1))


def restore(cfg, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(cfg, key) and not isinstance(getattr(cfg, key), dict):
            restore(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)


def find_checkpoint(root, requested_run, requested_iteration):
    if requested_run in (None, '-1'):
        runs = [os.path.join(root, name) for name in os.listdir(root)
                if os.path.isfile(os.path.join(root, name, 'nero_seven_cycle_config.json'))]
        runs = [path for path in runs if any(re.fullmatch(r'model_\d+\.pt', name)
                                               for name in os.listdir(path))]
        if not runs:
            raise FileNotFoundError('No seven-cycle training checkpoints in ' + root)
        run = max(runs, key=lambda path: os.path.getmtime(os.path.join(path, 'nero_seven_cycle_config.json')))
    else:
        run = os.path.join(root, requested_run)
    if requested_iteration in (None, -1):
        numbers = [int(m.group(1)) for name in os.listdir(run)
                   for m in [re.fullmatch(r'model_(\d+)\.pt', name)] if m]
        if not numbers:
            raise FileNotFoundError('No numeric checkpoint in ' + run)
        requested_iteration = max(numbers)
    checkpoint = os.path.join(run, f'model_{requested_iteration}.pt')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    return checkpoint


def draw_point(env, point, color):
    origin = env.gym.get_env_origin(env.envs[0])
    offset = np.array([origin.x, origin.y, origin.z])
    center = point.detach().cpu().numpy() - offset
    axis = np.eye(3) * 0.035
    vertices = np.stack((center - axis, center + axis), axis=1).astype(np.float32)
    env.gym.add_lines(env.viewer, env.envs[0], 3, vertices,
                      np.tile(color, (3, 1)).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--no-obs-noise', action='store_true')
    opt, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'nero_seven_cycle'
    args.num_envs = args.num_envs or 1
    args.resume = False
    root = os.path.join(ROOT, 'logs', args.experiment_name or 'nero_seven_cycle_pd20_1')
    checkpoint = find_checkpoint(root, args.load_run, args.checkpoint)
    with open(os.path.join(os.path.dirname(checkpoint), 'nero_seven_cycle_config.json')) as stream:
        saved = json.load(stream)
    cfg, train = task_registry.get_cfgs(args.task)
    restore(cfg, saved['env'])
    cfg.control.behavior_version = saved['env']['control'].get('behavior_version', 1)
    cfg.rewards.scales = SimpleNamespace(**saved['env']['rewards']['scales'])
    restore(train, saved['train'])
    train.runner.resume = False
    cfg.env.num_envs = args.num_envs
    cfg.env.teleop_mode = False  # replay the deterministic training cycle
    if opt.no_obs_noise:
        cfg.noise.add_noise = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train, log_root=None)
    runner.load(checkpoint, load_optimizer=False)
    policy = SevenCyclePolicy(runner.alg.actor_critic.eval()).to(env.device).eval()
    print('Checkpoint:', checkpoint, flush=True)
    print('Green: interpolated target; yellow: segment endpoint; blue: actual tool. Esc: exit.', flush=True)
    total_error = 0.0
    with torch.no_grad():
        for step in range(opt.steps):
            action = policy(env.get_observations()['obs'])
            env.step(action)
            total_error += (env.ee_pos - env.curr_ee_goal_cart_world).norm(dim=1).mean().item()
            if env.viewer:
                env.gym.clear_lines(env.viewer)
                draw_point(env, env.curr_ee_goal_cart_world[0], [0, 1, 0])
                endpoint = env.mount_pos[0] + env._goal_center_offset() + sphere2cart(env.ee_goal_sphere)[0]
                draw_point(env, endpoint, [1, 1, 0])
                draw_point(env, env.ee_pos[0], [0, 0.5, 1])
    print('Mean tool-position error (m):', total_error / max(opt.steps, 1))
    if env.viewer:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
