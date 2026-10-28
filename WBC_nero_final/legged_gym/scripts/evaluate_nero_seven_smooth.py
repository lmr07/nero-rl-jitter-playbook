"""Check tracking and unfiltered actor smoothness together on a saved v3 run."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym
import numpy as np
from scipy import signal
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.play_nero_seven_cycle import find_checkpoint, restore, SevenCyclePolicy


def high_frequency_rms(values):
    filtered = signal.sosfiltfilt(signal.butter(3, 5, fs=50, btype='highpass', output='sos'), values, axis=0)
    return np.sqrt(np.mean(filtered**2, axis=0))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-pass', action='store_true')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    args.task, args.num_envs, args.headless, args.resume = 'nero_seven_cycle_smooth', 1, True, False
    checkpoint = find_checkpoint(os.path.join(ROOT, 'logs', args.experiment_name or 'nero_seven_cycle_smooth_v3'),
                                 args.load_run, args.checkpoint)
    with open(os.path.join(os.path.dirname(checkpoint), 'nero_seven_cycle_config.json')) as stream:
        saved = json.load(stream)
    if saved['env']['control']['behavior_version'] != 3:
        raise ValueError('This evaluator requires a smooth-v3 checkpoint')
    cfg, train = task_registry.get_cfgs(args.task)
    restore(cfg, saved['env'])
    cfg.rewards.scales = SimpleNamespace(**saved['env']['rewards']['scales'])
    restore(train, saved['train'])
    train.runner.resume = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_gripper_mass = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train, log_root=None)
        runner.load(checkpoint, load_optimizer=False)
        policy = SevenCyclePolicy(runner.alg.actor_critic.eval()).to(env.device).eval()
        env.reset()
        samples = []
        completed, failure = False, False
        with torch.no_grad():
            for tick in range(1350):
                goal = env.curr_ee_goal_cart_world.clone()
                holding_age = float(((env.goal_timer-env.traj_timesteps)*env.dt)[0])
                raw = policy(env.obs_buf)
                request = env.default_dof_pos[:7]+cfg.control.action_scale*raw[0]
                _, _, done, info = env.step(raw)
                if bool(done[0]):
                    position = info['terminal_joint_positions'][0]
                else:
                    position = env.dof_pos[0, :7]
                samples.append({'q':position.cpu().numpy().copy(),
                                'request':request.cpu().numpy().copy(),
                                'issued':env.issued_request[0].cpu().numpy().copy(),
                                'segment':0 if tick<150 else 1+(tick-150)//300,
                                'holding_age_s':holding_age,
                                'error_m':float((env.ee_pos[0]-goal[0]).norm()) if not bool(done[0]) else None})
                if bool(done[0]):
                    completed = bool(info['time_outs'][0]) and tick==1349
                    failure = not completed
                    break
        windows = []
        for segment in range(5):
            rows = [row for row in samples if row['segment']==segment and row['holding_age_s']>=.5
                    and row['error_m'] is not None]
            if len(rows)<40:
                windows.append({'segment':segment, 'samples':len(rows), 'passed':False})
                continue
            q = np.array([row['q'] for row in rows])
            request = np.array([row['request'] for row in rows])
            issued = np.array([row['issued'] for row in rows])
            q_rms = high_frequency_rms(q)
            request_rms = high_frequency_rms(request)
            issued_rms = high_frequency_rms(issued)
            delta = np.diff(request, axis=0)
            step95 = np.percentile(np.abs(delta), 95, axis=0)
            tracking_limit = .05 if segment in (1, 3) else .03
            final_error = max(row['error_m'] for row in rows[-50:])
            passed = bool(final_error<=tracking_limit and np.max(q_rms)<=.001
                          and np.max(request_rms)<=.002 and np.max(step95)<=.003)
            windows.append({'segment':segment, 'samples':len(rows), 'passed':passed,
                            'last_second_max_error_m':final_error, 'q_highpass5_rms_rad':q_rms.tolist(),
                            'request_highpass5_rms_rad':request_rms.tolist(),
                            'issued_highpass5_rms_rad':issued_rms.tolist(),
                            'request_step95_rad':step95.tolist()})
        result = {'checkpoint':checkpoint, 'completed_cycle':completed, 'safety_failure':failure,
                  'steps':len(samples), 'passed':completed and all(window['passed'] for window in windows),
                  'noise':False, 'randomize_gripper_mass':False, 'hold_windows':windows,
                  'scope':'one noiseless seed/initial state; does not establish hardware safety'}
        output = os.path.abspath(options.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as stream:
            json.dump(result, stream, indent=2)
        np.savez_compressed(os.path.splitext(output)[0]+'.npz',
                            q=np.array([row['q'] for row in samples]),
                            request=np.array([row['request'] for row in samples]),
                            issued=np.array([row['issued'] for row in samples]))
        print(json.dumps(result, indent=2))
        if options.require_pass and not result['passed']:
            raise SystemExit(2)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
