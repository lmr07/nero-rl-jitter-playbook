"""Visualize a seven-joint Nero EE tracking checkpoint."""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym  # must precede torch
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.isaacgym_utils import sphere2cart


def restore(config, values):
    for key, value in values.items():
        current = getattr(config, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            restore(current, value)
        else:
            setattr(config, key, value)


def find_checkpoint(root, load_run, checkpoint):
    if load_run in (None, '-1'):
        runs = [os.path.join(root, name) for name in os.listdir(root)]
        runs = [path for path in runs if os.path.isfile(os.path.join(path, 'nero_ee_tracking_config.json'))
                and any(re.fullmatch(r'model_\d+\.pt', name) for name in os.listdir(path))]
        if not runs:
            raise FileNotFoundError('No Nero EE tracking checkpoints in ' + root)
        run = max(runs, key=lambda path: os.path.getmtime(os.path.join(path, 'nero_ee_tracking_config.json')))
    else:
        run = os.path.join(root, load_run)
    if checkpoint in (None, -1):
        numbers = [int(match.group(1)) for name in os.listdir(run)
                   for match in [re.fullmatch(r'model_(\d+)\.pt', name)] if match]
        if not numbers:
            raise FileNotFoundError('No numeric checkpoint in ' + run)
        checkpoint = max(numbers)
    path = os.path.join(run, f'model_{checkpoint}.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def draw_point(env, point, color, half_size=0.035):
    origin = env.gym.get_env_origin(env.envs[0])
    offset = np.array([origin.x, origin.y, origin.z])
    center = point.detach().cpu().numpy() - offset
    axes = np.eye(3) * half_size
    vertices = np.stack((center - axes, center + axes), axis=1).astype(np.float32)
    env.gym.add_lines(env.viewer, env.envs[0], 3, vertices,
                      np.tile(color, (3, 1)).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--no-obs-noise', action='store_true')
    parser.add_argument('--stable', action='store_true',
                        help='Load the independent stable EE tracking task')
    parser.add_argument('--limit', action='store_true',
                        help='Load the joint-limit-aware EE tracking task')
    options, remaining = parser.parse_known_args()
    if options.stable and options.limit:
        parser.error('--stable and --limit cannot be used together')
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = ('nero_ee_tracking_limit' if options.limit else
                 'nero_ee_tracking_stable' if options.stable else 'nero_ee_tracking')
    args.num_envs = args.num_envs or 1
    args.resume = False
    default_experiment = ('nero_ee_tracking_limit_v3' if options.limit else
                          'nero_ee_tracking_stable_v2' if options.stable else
                          'nero_ee_tracking_wbc_7dof')
    root = os.path.join(ROOT, 'logs', args.experiment_name or default_experiment)
    checkpoint = find_checkpoint(root, args.load_run, args.checkpoint)
    with open(os.path.join(os.path.dirname(checkpoint), 'nero_ee_tracking_config.json')) as stream:
        saved = json.load(stream)
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    restore(cfg, saved['env'])
    restore(train_cfg, saved['train'])
    cfg.env.num_envs = args.num_envs
    cfg.env.teleop_mode = False
    train_cfg.runner.resume = False
    if options.no_obs_noise:
        cfg.noise.add_noise = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.alg.actor_critic.eval()
    print('Checkpoint:', checkpoint, flush=True)
    print('Magenta: spherical origin; green: current target; yellow: sampled endpoint; '
          'blue: actual tool. Esc: exit.', flush=True)
    total_error = 0.0
    with torch.no_grad():
        for _ in range(options.steps):
            action = policy.act_inference(env.get_observations())
            env.step(action)
            total_error += (env.ee_pos - env.curr_ee_goal_cart_world).norm(dim=1).mean().item()
            if env.viewer:
                env.gym.clear_lines(env.viewer)
                draw_point(env, env.get_ee_goal_spherical_center()[0], [1, 0, 1], half_size=0.06)
                draw_point(env, env.curr_ee_goal_cart_world[0], [0, 1, 0])
                endpoint = env.get_ee_goal_spherical_center()[0] + sphere2cart(env.ee_goal_sphere)[0]
                draw_point(env, endpoint, [1, 1, 0])
                draw_point(env, env.ee_pos[0], [0, 0.5, 1])
    print('Mean tool-position error (m):', total_error / max(options.steps, 1))
    if env.viewer:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
