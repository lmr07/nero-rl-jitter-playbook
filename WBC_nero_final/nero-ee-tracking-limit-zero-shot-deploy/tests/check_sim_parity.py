"""Compare exported deployment observations and targets with limit-aware EE PhysX."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT.parent
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT))
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from nero_only.core import FixedPolicy


def restore(obj, values):
    for key, value in values.items():
        current = getattr(obj, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            restore(current, value)
        else:
            setattr(obj, key, value)


def main():
    args = get_args()
    args.task, args.num_envs, args.headless = 'nero_ee_tracking_limit', 1, True
    cfg, _ = task_registry.get_cfgs(args.task)
    restore(cfg, json.loads((ROOT / 'assets/nero_config.json').read_text())['env'])
    cfg.noise.add_noise = False
    cfg.env.teleop_mode = True
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    policy = FixedPolicy()
    env.reset()
    env.traj_timesteps.fill_(policy.travel_ticks)
    env.traj_total_timesteps.fill_(float('inf'))
    max_obs = max_goal = max_target = 0.
    try:
        for tick in range(400):
            if tick == 200:
                env.move_ee_goal_cart((0., 0., .05))
                assert policy.move('i')
                policy.travel_ticks = float(env.traj_timesteps[0])
            q = env.dof_pos[0, :7].cpu().numpy()
            dq = env.dof_vel[0, :7].cpu().numpy()
            obs = policy.observe(q, dq)
            max_obs = max(max_obs, float(np.max(np.abs(obs[0]-env.obs_buf[0].cpu().numpy()))))
            max_goal = max(max_goal, float(np.max(np.abs(policy.target_cart()-
                (env.curr_ee_goal_cart_world[0]-env.mount_pos[0]).cpu().numpy()))))
            with torch.inference_mode():
                raw = policy.actor(torch.from_numpy(obs)).numpy()[0]
            candidate = policy.candidate_from_raw(raw)
            _, _, reset, _ = env.step(torch.tensor(raw[None, :], dtype=torch.float32, device=env.device))
            if bool(reset[0]):
                raise RuntimeError(f'Limit-aware policy triggered a safety reset at tick {tick}')
            sent = env.drive_positions[0, :7].cpu().numpy()
            max_target = max(max_target, float(np.max(np.abs(candidate-sent))))
            policy.record_sent(sent)
            policy.advance()
        assert max_obs < 1.e-3, max_obs
        assert max_goal < 1.e-5, max_goal
        assert max_target < 1.e-5, max_target
        print('PASS: max observation error', max_obs, 'goal error m', max_goal,
              'target error rad', max_target)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
