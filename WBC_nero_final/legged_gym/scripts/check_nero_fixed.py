"""GPU integration check: FK, fixed root, torque gains, reset, goals and observations."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

args = get_args()
args.task = 'nero_fixed'
args.headless = True
args.num_envs = 32
cfg, _ = task_registry.get_cfgs(args.task)
cfg.env.episode_length_s = 2.0
cfg.goal_ee.traj_time = [0.2, 0.3]
cfg.goal_ee.hold_time = [0.1, 0.2]
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
initial_root = env.root_states.clone()
max_fk_error = 0.0
reset_count = 0
for tick in range(320):
    with torch.no_grad():
        action = 0.25 * torch.sin(torch.full((32, 7), tick * 0.03, device=env.device))
        obs, reward, done, _ = env.step(action)
    assert obs['obs'].shape == (32, 624)
    assert obs['privileged_obs'].shape == (32, 165)
    assert obs['obs_pred'].shape == (32, 27)
    assert all(torch.isfinite(v).all() for v in obs.values())
    assert torch.isfinite(reward).all()
    assert (env.torques.abs() <= env.torque_limits + 1.e-5).all()
    assert torch.allclose(env.dof_pos[:, 7], -env.dof_pos[:, 8])
    env.gym.refresh_actor_root_state_tensor(env.sim)
    assert torch.allclose(env.root_states, initial_root, atol=1.e-6)
    if not done.any():
        body = env.rigid_state[:, env.ee_body_idx]
        physical_tool = body[:, :3] + quat_apply(body[:, 3:7], env.ee_offset)
        error = (physical_tool - env.ee_pos).norm(dim=-1).max().item()
        max_fk_error = max(max_fk_error, error)
        assert error < 2.e-4, error
    else:
        assert torch.all(env.obs_history[done, :-1] == 0)
        assert torch.all(env.critic_history[done, :-1] == 0)
        assert torch.all(env.obs_history[done, -1, -6:] == 0)
        assert torch.all(env.last_candidate_delta[done] == 0)
        assert torch.all(env.hold_samples[done] == 0)
        assert torch.all(env.metric_samples[done] == 0)
        assert all(torch.isfinite(value) for value in env.extras['episode'].values())
        assert 'metric_slew_fraction' in env.extras['episode']
    reset_count += done.sum().item()
    assert not env.collision_check(torch.arange(32, device=env.device)).any()
assert reset_count >= 96
assert torch.all(env.kp[:7] == 48) and torch.all(env.kd[:7] == 1.5)
print(f'PASS: 320 steps x 32 envs; {reset_count} resets; max FK/PhysX tool error={max_fk_error:.8f} m')
env.gym.destroy_sim(env.sim)
