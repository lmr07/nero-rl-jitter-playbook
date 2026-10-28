"""Exercise the actual task reward method without importing Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT / 'legged_gym/envs/nero/nero_fixed.py').read_text())
task = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeroFixed')
method = next(n for n in task.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_reward')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<task reward>', 'exec'), namespace)
compute_reward = namespace['compute_reward']

def fixture():
    z = lambda n=1: torch.zeros(2, n) if n > 1 else torch.zeros(2)
    scales = NS(candidate_acceleration=-1., hold_dof_vel=-1.)
    env = NS(cfg=NS(rewards=NS(scales=scales, hold_error_radius=.1),
                    control=NS(action_scale=.25)), dt=.02, actuator=None,
             lower=-torch.ones(9)*3, upper=torch.ones(9)*3,
             dof_pos=z(9), dof_vel=z(9), last_vel=z(7), torques=z(9),
             actions=z(7), last_actions=z(7), candidate_actions=z(7),
             last_candidate_actions=z(7), last_candidate_delta=z(7), last_sent_delta=z(7),
             ee_pos=z(3), curr_ee_goal_cart_world=z(3),
             goal_timer=torch.tensor([20., 20.]), traj_timesteps=torch.tensor([10., 10.]),
             rew_buf=z(), episode_sums={k:z() for k in vars(scales)},
             metric_sums={k:z() for k in ('position_error_m','candidate_delta_rms_rad',
                 'adapter_gap_rms_rad','slew_fraction','hold_error_m','hold_speed_rms_rad_s',
                 'sent_speed_rms_rad_s','sent_acceleration_rms_rad_s2','sent_reversal_fraction')},
             hold_samples=z(), metric_samples=z(), zeros=z,
             _reward_tracking_ee_world=z, _reward_tracking_ee_orn=z)
    return env

class RewardTests(unittest.TestCase):
    def test_constant_ramp_has_no_reversal_penalty(self):
        env = fixture()
        env.candidate_actions.fill_(.2)
        env.last_candidate_actions.fill_(.1)
        env.last_candidate_delta.fill_(.1)
        compute_reward(env)
        torch.testing.assert_close(env.rew_buf, torch.zeros(2))
        env.last_candidate_delta.fill_(-.1)
        compute_reward(env)
        self.assertTrue((env.rew_buf < 0).all())

    def test_hold_penalty_does_not_slow_moving_reference(self):
        env = fixture()
        env.dof_vel[:, :7] = 1.
        env.goal_timer[0] = 0.
        compute_reward(env)
        self.assertEqual(env.rew_buf[0].item(), 0.)
        self.assertLess(env.rew_buf[1].item(), -.1)
        torch.testing.assert_close(env.hold_samples, torch.tensor([0., 1.]))

    def test_hold_penalty_is_smooth_and_near_target(self):
        env = fixture()
        env.dof_vel[:, :7] = 1.
        env.ee_pos[1, 0] = .3
        compute_reward(env)
        self.assertGreater(abs(env.rew_buf[0].item()), abs(env.rew_buf[1].item()) * 100)
        self.assertTrue(torch.isfinite(env.rew_buf).all())

    def test_far_hold_keeps_baseline_and_final_reversal_cost(self):
        env = fixture()
        env.cfg.rewards.hold_velocity_floor = .25
        env.ee_pos[:, 0] = 1.
        env.dof_vel[:, :7] = 1.
        compute_reward(env)
        torch.testing.assert_close(env.rew_buf, torch.full((2,), -.035))
        env = fixture()
        env.cfg.rewards.scales.target_acceleration = -1.
        env.actions.fill_(.012/.25)
        env.last_sent_delta.fill_(.012)
        compute_reward(env)
        torch.testing.assert_close(env.rew_buf, torch.zeros(2))
        env.last_sent_delta.fill_(-.012)
        compute_reward(env)
        self.assertTrue((env.rew_buf < -.5).all())

if __name__ == '__main__':
    unittest.main()
