"""Evaluate a Nero checkpoint using its saved configuration and export its actor."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym.utils.isaacgym_utils import sphere2cart


class DeploymentPolicy(torch.nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.encoder = ac.adaptation_encoder_module
        self.actor = ac.actor_body

    def forward(self, history):
        return self.actor(torch.cat((history[:, -39:], self.encoder(history)), dim=-1))


def restore(cfg, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(cfg, key) and not isinstance(getattr(cfg, key), dict):
            restore(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)


if __name__ == '__main__':
    args = get_args()
    args.task = 'nero_fixed'
    root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'nero_fixed')
    checkpoint = get_load_path(root, load_run=args.load_run, checkpoint=args.checkpoint)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    config_path = os.path.join(os.path.dirname(checkpoint), 'nero_config.json')
    if os.path.isfile(config_path):
        with open(config_path) as stream:
            saved = json.load(stream)
        # Old checkpoints must retain their original direct-PD path.
        if 'actuator' not in saved['env']:
            env_cfg.actuator.enabled = False
        env_cfg.control.action_mode = saved['env']['control'].get('action_mode', 'absolute')
        env_cfg.control.pd_backend = saved['env']['control'].get('pd_backend', 'explicit')
        env_cfg.rewards.hold_velocity_floor = saved['env']['rewards'].get('hold_velocity_floor', 0.0)
        restore(env_cfg, saved['env'])
        restore(train_cfg, saved['train'])
        # New training-only penalties must not alter old checkpoint reward logs.
        for key in ('candidate_acceleration', 'hold_dof_vel', 'target_rate', 'target_acceleration'):
            if key not in saved['env']['rewards']['scales']:
                setattr(env_cfg.rewards.scales, key, 0.0)
    else:
        raise RuntimeError("Missing nero_config.json: cannot infer checkpoint action semantics")
    args.num_envs = args.num_envs or 1
    args.resume = False
    train_cfg.runner.resume = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_actuator = False
    env_cfg.env.teleop_mode = not args.headless
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
    runner.load(checkpoint, load_optimizer=False)
    actor = runner.alg.actor_critic.eval()
    deployment = DeploymentPolicy(actor).eval()
    example = env.get_observations()['obs']
    with torch.no_grad():
        traced = torch.jit.trace(deployment, example)
        assert torch.allclose(traced(example), deployment(example), atol=1.e-6)
    export_path = os.path.join(os.path.dirname(checkpoint), 'nero_policy.pt')
    traced.save(export_path)
    print('Exported raw-action policy (624 -> 7):', export_path)
    if env.viewer:
        print('Click the viewer to focus. I/K: up/down (+Z/-Z); Left/Right arrows: left/right (+Y/-Y); '
              'Up/Down arrows: forward/back (+X/-X). Each press requests 5 cm, clipped to sampling bounds. '
              'Yellow: endpoint; green: interpolated reference; blue: actual tool. Esc: exit.')
    error_sum, steps = 0.0, 0
    with torch.no_grad():
        while env.viewer or steps < 1000:
            env.step(deployment(env.get_observations()['obs']))
            error_sum += (env.ee_pos - env.curr_ee_goal_cart_world).norm(dim=-1).mean().item()
            steps += 1
            if env.viewer:
                env.gym.clear_lines(env.viewer)
                origin = env.gym.get_env_origin(env.envs[0])
                offset = np.array([origin.x, origin.y, origin.z])
                for point, color in ((env.curr_ee_goal_cart_world[0], [0, 1, 0]),
                                     ((env.mount_pos + sphere2cart(env.ee_goal_sphere))[0], [1, 1, 0]),
                                     (env.ee_pos[0], [0, 0.5, 1])):
                    center = point.cpu().numpy() - offset
                    axes = np.eye(3) * 0.04
                    vertices = np.stack((center - axes, center + axes), axis=1).astype(np.float32)
                    env.gym.add_lines(env.viewer, env.envs[0], 3, vertices, np.tile(color, (3, 1)).astype(np.float32))
    print('Mean position tracking error (m):', error_sum / steps)
    if env.viewer:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
