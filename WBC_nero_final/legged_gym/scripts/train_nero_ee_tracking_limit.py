"""Continue the stable Nero EE policy with joint-limit safety rewards."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym  # must precede torch
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict


SOURCE_CHECKPOINT = os.path.join(
    LEGGED_GYM_ROOT_DIR, 'logs', 'nero_ee_tracking_stable_v2',
    'Oct27_16-19-23_pitch_ramp_slew_hold_gravity', 'model_10000.pt')


def main():
    args = get_args()
    args.task = 'nero_ee_tracking_limit'
    if args.resume:
        raise ValueError('This entry point starts a fresh limit-aware continuation; --resume is unsupported')
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg)

    source = os.path.abspath(os.environ.get('NERO_SOURCE_CHECKPOINT', SOURCE_CHECKPOINT))
    digest = hashlib.sha256()
    with open(source, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    checkpoint = torch.load(source, map_location=runner.device)
    runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if runner.current_learning_iteration != 0:
        raise RuntimeError('Continuation must start at iteration zero')

    os.makedirs(runner.log_dir, exist_ok=True)
    with open(os.path.join(runner.log_dir, 'nero_ee_tracking_config.json'), 'w') as stream:
        json.dump({'env': class_to_dict(cfg), 'train': class_to_dict(train_cfg),
                   'source_checkpoint': source, 'source_sha256': digest.hexdigest(),
                   'source_iteration': checkpoint.get('iter')}, stream, indent=2)
    print('Limit-aware Nero EE tracking, source:', source, flush=True)
    print('Source SHA256:', digest.hexdigest(), 'new run:', runner.log_dir, flush=True)
    print('Starting fresh optimizer at iteration zero for',
          train_cfg.runner.max_iterations, 'iterations', flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
