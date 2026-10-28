"""Inject keyboard events into a headless GPU environment to check teleoperation."""
import os
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym
from isaacgym import gymapi
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.isaacgym_utils import sphere2cart, cart2sphere

args = get_args()
args.task, args.headless, args.num_envs = 'nero_fixed', True, 2
cfg, _ = task_registry.get_cfgs(args.task)
cfg.env.teleop_mode = True
cfg.env.episode_length_s = 0.1
cfg.goal_ee.traj_time = [0.2, 0.2]
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
try:
    assert set(env.TELEOP_KEYS) == {'I', 'K', 'LEFT', 'RIGHT', 'UP', 'DOWN'}
    limits = env.tensor([env.goal_ee_ranges[k] for k in ('pos_l', 'pos_p', 'pos_y')])
    def bounded(sphere):
        assert (sphere >= limits[:, 0] - 1.e-6).all()
        assert (sphere <= limits[:, 1] + 1.e-6).all()

    for key, delta in env.TELEOP_KEYS.items():
        assert hasattr(gymapi, 'KEY_' + key)
        before = sphere2cart(env.ee_goal_sphere).clone()
        reference = env.curr_ee_goal_sphere.clone()
        event = SimpleNamespace(action='nero_move_' + key, value=1)
        env.handle_viewer_action_event(event)
        requested = cart2sphere(before + env.tensor(delta))
        expected = sphere2cart(torch.max(torch.min(requested, limits[:, 1]), limits[:, 0]))
        assert torch.allclose(sphere2cart(env.ee_goal_sphere), expected, atol=1.e-6)
        assert torch.equal(env.curr_ee_goal_sphere, reference)
        assert torch.equal(env.ee_start_sphere, reference)
        env.handle_viewer_action_event(event)
        assert torch.allclose(sphere2cart(env.ee_goal_sphere), expected, atol=1.e-6)
        env.handle_viewer_action_event(SimpleNamespace(action=event.action, value=0))
    # New physical press accumulates; release did not move the target.
    before = sphere2cart(env.ee_goal_sphere).clone()
    env.handle_viewer_action_event(SimpleNamespace(action='nero_move_I', value=1))
    assert torch.allclose(sphere2cart(env.ee_goal_sphere), before + env.tensor([0, 0, .05]), atol=1.e-6)
    start, endpoint = env.ee_start_sphere.clone(), env.ee_goal_sphere.clone()
    env.goal_timer.copy_(env.traj_timesteps * .5)
    env._update_goal()
    assert torch.allclose(env.curr_ee_goal_sphere, (start + endpoint) / 2, atol=1.e-6)
    # Mid-flight retargeting keeps reference continuous and accumulates on endpoint.
    reference = env.curr_ee_goal_sphere.clone()
    env.move_ee_goal_cart([0, .05, 0])
    assert torch.equal(env.ee_start_sphere, reference)
    assert torch.equal(env.curr_ee_goal_sphere, reference)
    endpoint = env.ee_goal_sphere.clone()
    env.goal_timer.copy_(env.traj_timesteps)
    env._update_goal()
    assert torch.allclose(env.curr_ee_goal_sphere, endpoint, atol=1.e-6)
    for _ in range(30):
        obs, reward, done, _ = env.step(torch.zeros(2, 7, device=env.device))
        assert not done.any(), 'Teleop must not reset on the episode timeout'
        assert torch.allclose(env.ee_goal_sphere, endpoint), 'Automatic sampling changed endpoint'
        assert torch.allclose(env.curr_ee_goal_sphere, endpoint)
        assert all(torch.isfinite(v).all() for v in obs.values())
    # Drive every spherical coordinate past both bounds; verify all six faces
    # and every sampled point on the re-planned spherical trajectory.
    for axis in range(3):
        for side in (0, 1):
            requested = ((limits[:, 0] + limits[:, 1]) / 2).repeat(2, 1)
            requested[:, axis] = limits[axis, side] + (-0.1 if side == 0 else 0.1)
            env.move_ee_goal_cart(sphere2cart(requested) - sphere2cart(env.ee_goal_sphere))
            assert torch.allclose(env.ee_goal_sphere[:, axis], limits[axis, side].expand(2), atol=1.e-6)
            bounded(env.ee_goal_sphere)
            for fraction in torch.linspace(0, 1, 11).tolist():
                env.goal_timer.copy_(env.traj_timesteps * fraction)
                env._update_goal()
                bounded(env.curr_ee_goal_sphere)
    # Check altered checkpoint ranges also constrain the initial trajectory.
    env.goal_ee_ranges['pos_l'] = [0.51, 0.53]
    env.goal_ee_ranges['pos_p'] = [-0.1, 0.1]
    limits = env.tensor([env.goal_ee_ranges[k] for k in ('pos_l', 'pos_p', 'pos_y')])
    env.reset_idx(torch.arange(2, device=env.device))
    bounded(env.ee_start_sphere)
    bounded(env.ee_goal_sphere)
    # Automatic training goals are still sampled when teleoperation is disabled.
    cfg.env.teleop_mode = False
    env._resample_goal(torch.arange(2, device=env.device))
    assert not torch.allclose(env.ee_goal_sphere, endpoint)
    print('PASS: I/K and four arrows, bounded 5 cm requests, all six spherical bounds, '
          'bounded interpolation and initialization, key debouncing, continuous replanning, '
          'hold, timeout suppression, and automatic-mode sampling')
finally:
    env.gym.destroy_sim(env.sim)
