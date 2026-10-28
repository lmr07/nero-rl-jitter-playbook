"""Train the seven-joint WBC-style Nero end-effector tracking task."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym  # must precede torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path


def main(task='nero_ee_tracking'):
    args = get_args()
    args.task = task
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.resume:
        root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = get_load_path(root, load_run=args.load_run, checkpoint=args.checkpoint)
        with open(os.path.join(os.path.dirname(checkpoint), 'nero_ee_tracking_config.json')) as stream:
            saved = json.load(stream)['env']
        for key in ('init_state', 'control', 'goal_ee', 'rewards', 'asset',
                    'sim', 'noise', 'domain_rand', 'actuator'):
            if saved[key] != json.loads(json.dumps(class_to_dict(getattr(cfg, key)))):
                raise RuntimeError('Checkpoint configuration differs at ' + key + '; start a fresh run')
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)
    os.makedirs(runner.log_dir, exist_ok=True)
    with open(os.path.join(runner.log_dir, 'nero_ee_tracking_config.json'), 'w') as stream:
        json.dump({'env': class_to_dict(cfg), 'train': class_to_dict(train_cfg)}, stream, indent=2)
    print('Nero EE tracking: 7 actions; PD Kp=[40,40,20,20,20,20,20], Kd=1;',
          'home sphere=', cfg.goal_ee.ranges.init_pos_start, flush=True)
    # Match A2nero WBC: stagger episode ends so reward terms are reported
    # throughout training instead of only once per synchronized 20 s cycle.
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
