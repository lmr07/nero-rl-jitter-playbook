"""Train the WBC-home, two-waypoint Nero policy from scratch."""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym  # must precede torch
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path
from legged_gym.envs.nero.nero_smooth_control import configure_smooth_start


def verify_identification(cfg):
    source = os.path.join(os.path.dirname(ROOT), cfg.actuator.identification_source)
    if not os.path.isfile(source):
        print('Identification source absent on this machine; using frozen values in saved configuration:', source)
        return
    with open(source, 'rb') as stream:
        payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg.actuator.identification_sha256:
        raise RuntimeError('Identification source changed; review and update frozen delay values explicitly')
    values = [r['command_only']['delay_ms'] for r in json.loads(payload)['joints']]
    if any(abs(a - b) > 1e-6 for a, b in zip(values, cfg.actuator.identified_effective_delay_ms)):
        raise RuntimeError('Frozen per-joint delays disagree with identification data')


def save_smooth_manifest(log_dir, cfg, train_cfg):
    files = ['legged_gym/scripts/train_nero_seven_cycle.py', 'legged_gym/envs/__init__.py',
             'legged_gym/envs/nero/nero_seven_cycle.py', 'legged_gym/envs/nero/nero_fixed.py',
             'legged_gym/envs/nero/nero_seven_cycle_config.py', 'legged_gym/envs/nero/nero_fixed_config.py',
             'legged_gym/envs/nero/nero_seven_cycle_smooth_config.py',
             'legged_gym/envs/nero/nero_smooth_control.py', 'legged_gym/envs/nero/identified_position_delay.py',
             'legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py',
             'legged_gym/b2_gym_learn/ppo_cse_pf/nero_smooth_actor_critic.py',
             'legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py',
             'legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py',
             'resources/robots/a2nero/urdf/nero-fixed.urdf']
    hashes = {}
    for name in files:
        with open(os.path.join(ROOT, name), 'rb') as stream:
            hashes[name] = hashlib.sha256(stream.read()).hexdigest()
    with open(os.path.join(log_dir, 'nero_seven_cycle_config.json'), 'rb') as stream:
        config_hash = hashlib.sha256(stream.read()).hexdigest()
    manifest = {'task':'nero_seven_cycle_smooth', 'behavior_version':3,
                'policy_class':train_cfg.runner.policy_class_name, 'torch_version':torch.__version__,
                'training_kp':cfg.control.stiffness, 'training_kd':cfg.control.damping,
                'num_envs':cfg.env.num_envs, 'iterations':train_cfg.runner.max_iterations,
                'config_sha256':config_hash, 'source_sha256':hashes,
                'identification_sha256':cfg.actuator.identification_sha256,
                'sampling':'bounded deterministic mean; positive Gaussian exploration scale',
                'request':'q_home + action_scale * sampled action, captured before clipping and slew'}
    with open(os.path.join(log_dir, 'nero_training_manifest.json'), 'w') as stream:
        json.dump(manifest, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--j1-j2-kp', type=int, choices=(30, 40, 50))
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'nero_seven_cycle_smooth' if custom.smooth else 'nero_seven_cycle'
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    if custom.j1_j2_kp is not None:
        cfg.control.stiffness = dict(cfg.control.stiffness)
        for joint in ('nero_joint1', 'nero_joint2'):
            cfg.control.stiffness[joint] = float(custom.j1_j2_kp)
    if cfg.control.behavior_version == 3:
        configure_smooth_start(cfg, ROOT)
    verify_identification(cfg)
    if args.resume:
        root = os.path.join(ROOT, 'logs', args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = get_load_path(root, -1 if args.load_run in (None, '-1') else args.load_run,
                                   -1 if args.checkpoint is None else args.checkpoint)
        with open(os.path.join(os.path.dirname(checkpoint), 'nero_seven_cycle_config.json')) as stream:
            saved = json.load(stream)['env']
        for key in ('control', 'goal_ee', 'actuator', 'rewards', 'init_state',
                    'sim', 'noise', 'domain_rand'):
            if saved[key] != json.loads(json.dumps(class_to_dict(getattr(cfg, key)))):
                raise RuntimeError('Checkpoint configuration differs at ' + key + '; start a fresh run')
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if cfg.control.behavior_version == 3:
        margin = cfg.control.request_joint_margin_rad
        train_cfg.policy.request_action_lower = ((env.lower[:7]+margin-env.default_dof_pos[:7])/
                                                 cfg.control.action_scale).cpu().tolist()
        train_cfg.policy.request_action_upper = ((env.upper[:7]-margin-env.default_dof_pos[:7])/
                                                 cfg.control.action_scale).cpu().tolist()
        train_cfg.policy.request_action_initial = ((env.tensor(cfg.control.initial_position_targets)-
                                                    env.default_dof_pos[:7])/cfg.control.action_scale).cpu().tolist()
        print('Smooth v3: bounded pre-slew requests; six-fold hold penalties; quintic references; '
              'current-action shortcut masked; predictive joint-limit termination.', flush=True)
        if args.resume:
            with open(os.path.join(os.path.dirname(checkpoint), 'nero_seven_cycle_config.json')) as stream:
                saved_train = json.load(stream)['train']
            if saved_train['policy'] != json.loads(json.dumps(class_to_dict(train_cfg.policy))):
                raise RuntimeError('Checkpoint policy configuration differs; start a fresh smooth-v3 run')
    print('Nero seven-cycle: native PD Kp', [cfg.control.stiffness[f'nero_joint{i}'] for i in range(1, 8)],
          'Kd', [cfg.control.damping[f'nero_joint{i}'] for i in range(1, 8)],
          '; physics 5 ms, policy 20 ms; '
          'home -> (0.5,0,pi/4) -> home -> (0.6,pi/3,-pi/4) -> home; '
          '3 s travel / 3 s hold; WBC home and phase-diverse resets; effective delays (ms):',
          cfg.actuator.identified_effective_delay_ms, flush=True)
    if custom.smoke:
        with torch.no_grad():
            removed_rewards = {'tracking_ee_orn', 'nero_joint_pos'}
            assert removed_rewards.isdisjoint(class_to_dict(cfg.rewards.scales))
            assert removed_rewards.isdisjoint(env.episode_sums)
            expected = torch.tensor(cfg.goal_ee.fixed_waypoints_spherical, device=env.device)
            home_q = torch.tensor([cfg.init_state.default_joint_angles[f'nero_joint{i}']
                                   for i in range(1, 8)], device=env.device)
            assert torch.allclose(home_q, torch.tensor([0., -1., 0., 1.8, 0., 0., .3], device=env.device))
            assert torch.all(env.home_sphere[:, 1] < torch.pi / 2 - 0.05)
            if env.num_envs >= 4:
                assert torch.equal(env.next_waypoint[:4], torch.arange(4, device=env.device))
                anchors = torch.tensor(cfg.goal_ee.phase_start_joint_angles, device=env.device)
                assert torch.allclose(env.dof_pos[0, :7], home_q)
                assert torch.allclose(env.dof_pos[1, :7], anchors[0])
                assert torch.allclose(env.dof_pos[2, :7], home_q)
                assert torch.allclose(env.dof_pos[3, :7], anchors[1])
                assert torch.isclose(env.drive_positions[1, 1], torch.tensor(
                    cfg.goal_ee.phase_start_target_joint_angles[0][1], device=env.device))
                assert torch.allclose(env.actions[1] * cfg.control.action_scale + home_q,
                                      env.drive_positions[1, :7])
                assert torch.all((env.ee_pos[:4] - env.curr_ee_goal_cart_world[:4]).norm(dim=1) < .01)
            assert env.gym.get_sim_dof_count(env.sim) == 9 * env.num_envs
            assert torch.allclose(torch.as_tensor(env.dof_props['stiffness'][:7]),
                                  torch.tensor([cfg.control.stiffness[f'nero_joint{i}'] for i in range(1, 8)]))
            assert torch.allclose(torch.as_tensor(env.dof_props['damping'][:7]), torch.full((7,), 1.0))
            assert torch.allclose(torch.as_tensor(env.dof_props['effort'][:7]),
                                  torch.tensor(cfg.actuator.torque_limits))
            for _ in range(4):
                phase = env.next_waypoint.clone()
                env._resample_goal(torch.arange(env.num_envs, device=env.device))
                for i in range(env.num_envs):
                    goal = expected[int(phase[i] // 2)] if phase[i] % 2 == 0 else env.home_sphere[i]
                    assert torch.allclose(env.ee_goal_sphere[i], goal, atol=1e-5)
            env.reset()
            previous = env.last_actions.clone() * cfg.control.action_scale
            env.step(torch.full((env.num_envs, 7), 100., device=env.device))
            sent = env.actions * cfg.control.action_scale
            if cfg.control.behavior_version == 3:
                sent = env.issued_request-env.default_dof_pos[:7]
            limit = torch.tensor(cfg.control.target_speed_limits_rad_s, device=env.device) * env.dt
            assert torch.all((sent - previous).abs() <= limit + 1e-5)
            for _ in range(12):
                obs, reward, done, _ = env.step(torch.zeros(env.num_envs, 7, device=env.device))
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
            if cfg.control.behavior_version == 3:
                env.reset()
                for _ in range(256):
                    previous = env.default_dof_pos[:7]+cfg.control.action_scale*env.last_actions.clone()
                    obs, reward, done, _ = env.step(torch.zeros(env.num_envs, 7, device=env.device))
                    assert torch.all((env.issued_request-previous).abs() <= limit+1e-5)
                    assert torch.isfinite(reward).all()
                    assert all(torch.isfinite(value).all() for value in obs.values())
                    assert torch.all(env.smooth_history.steps[done] == 0)
                print('Smooth-v3 smoke passed: pre-slew capture, native PD, bounded targets, '
                      'smoothness rewards and clean safety resets.', flush=True)
                return
            env.reset()
            initial_phase = env.next_waypoint.clone()
            phase_counter = env.phase_cycle_count.clone()
            ids = torch.arange(env.num_envs, device=env.device)
            for tick in range(round(cfg.env.episode_length_s / env.dt)):
                obs, reward, done, _ = env.step(torch.zeros(env.num_envs, 7, device=env.device))
                assert torch.isfinite(reward).all()
                if tick + 1 in (150, 450, 750, 1050):
                    leg = (150, 450, 750, 1050).index(tick + 1) + 1
                    assert torch.equal(env.next_waypoint, (initial_phase + leg) % 4)
            assert done.all()
            if env.num_envs > 1 and cfg.goal_ee.diversify_initial_phase:
                assert torch.equal(env.next_waypoint, (ids + phase_counter) % 4)
        print('Smoke check passed: WBC home, coherent phases, target slew, native drives, delay, reward and full cycle.', flush=True)
        return
    runner, training_cfg = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    os.makedirs(runner.log_dir, exist_ok=True)
    with open(os.path.join(runner.log_dir, 'nero_seven_cycle_config.json'), 'w') as stream:
        json.dump({'env': class_to_dict(cfg), 'train': class_to_dict(training_cfg)}, stream, indent=2)
    if cfg.control.behavior_version == 3:
        save_smooth_manifest(runner.log_dir, cfg, training_cfg)
    runner.learn(num_learning_iterations=training_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
