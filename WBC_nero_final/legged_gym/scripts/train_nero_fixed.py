"""Train Nero from scratch, or resume a checkpoint from the nero_fixed task."""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

if __name__ == '__main__':
    args = get_args()
    args.task = 'nero_fixed'
    if args.resume:
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from legged_gym.utils.helpers import get_load_path
        current, _ = task_registry.get_cfgs(args.task)
        checkpoint = get_load_path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'nero_fixed'),
                                   load_run=args.load_run, checkpoint=args.checkpoint)
        with open(os.path.join(os.path.dirname(checkpoint), 'nero_config.json')) as stream:
            saved_env = json.load(stream)['env']
            saved = saved_env['control']
        if (saved.get('pd_backend', 'explicit') != current.control.pd_backend or
                saved_env['sim']['dt'] != current.sim.dt or
                saved['decimation'] != current.control.decimation):
            raise RuntimeError('Checkpoint PD backend/timestep differs; start native-PD training from scratch')
        for key in ('action_mode', 'target_delta_limit', 'action_scale'):
            if saved.get(key) != getattr(current.control, key):
                raise RuntimeError('Checkpoint action ABI mismatch: ' + key + '; start from scratch')
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, cfg = task_registry.make_alg_runner(env, name=args.task, args=args)
    from legged_gym.utils.helpers import class_to_dict
    os.makedirs(runner.log_dir, exist_ok=True)
    with open(os.path.join(runner.log_dir, 'nero_config.json'), 'w') as stream:
        json.dump({'env': class_to_dict(env.cfg), 'train': class_to_dict(cfg)}, stream, indent=2)
    if env.robust:
        print('Nero fixed-distribution DR: no stages or performance gates.', flush=True)
        print('Action mode:', env.cfg.control.action_mode, 'delta limit:', env.cfg.control.target_delta_limit, flush=True)
        print('PD backend:', env.pd_backend, 'policy dt:', env.dt, 'physics dt:', env.cfg.sim.dt, flush=True)
        print('Reference torque limits (Nm):', env.cfg.actuator.torque_limits, flush=True)
        print('Reference velocity limits (rad/s):', env.cfg.actuator.velocity_limits, flush=True)
        print('Full configuration saved:', os.path.join(runner.log_dir, 'nero_config.json'), flush=True)
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)
