"""Replay/export a single-axis policy with its saved ABI; never opens CAN."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.nero.joint_probe_utils import find_checkpoint


def restore(cfg, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(cfg, key) and not isinstance(getattr(cfg, key), dict):
            restore(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)


class JointPolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.encoder = actor.adaptation_encoder_module
        self.actor = actor.actor_body

    def forward(self, history):
        return self.actor(torch.cat((history[:, -12:], self.encoder(history)), dim=-1))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--joint", required=True, type=int, choices=range(1, 8))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--export", action="store_true")
    p.add_argument("--legacy", action="store_true", help="Replay old unrestricted 70-percent-range checkpoints")
    p.add_argument("--smooth", action="store_true", help="Replay smooth-hold reward runs")
    p.add_argument("--a2stage1", action="store_true", help="Replay A2nero Stage-1-aligned one-joint runs")
    p.add_argument("--no-obs-noise", action="store_true", help="Disable observation noise for simulation replay only")
    p.add_argument("--held-out-pose-seed", type=int)
    opt, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    args.load_run = -1 if args.load_run in (None, "-1") else args.load_run
    args.checkpoint = -1 if args.checkpoint is None else args.checkpoint
    if sum((opt.legacy, opt.smooth, opt.a2stage1)) > 1:
        p.error("--legacy, --smooth and --a2stage1 are mutually exclusive")
    experiment = ("nero_joint_a2stage1_J%d" if opt.a2stage1 else
                  "nero_joint_smooth_J%d" if opt.smooth else
                  "nero_joint_J%d" if opt.legacy else "nero_joint_slew_J%d") % opt.joint
    checkpoint = find_checkpoint(os.path.join(ROOT, "logs", experiment),
                               args.load_run, args.checkpoint)
    with open(os.path.join(os.path.dirname(checkpoint), "joint_config.json")) as f:
        saved = json.load(f)
    cfg, train = task_registry.get_cfgs("nero_joint")
    cfg.rewards.smooth_hold = False  # old manifests contain no rewards section
    restore(cfg, saved["env"])
    restore(train, saved["train"])
    if cfg.env.joint != opt.joint:
        raise ValueError("Checkpoint joint mismatch")
    if opt.no_obs_noise:
        cfg.noise.add_noise = False
        print("Simulation replay: observation noise disabled", flush=True)
    if opt.held_out_pose_seed is not None:
        cfg.env.pose_seed = opt.held_out_pose_seed
    args.num_envs = args.num_envs or 1
    args.resume = False
    train.runner.resume = False
    env, _ = task_registry.make_env("nero_joint", args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train, log_root=None)
    runner.load(checkpoint, load_optimizer=False)
    module = JointPolicy(runner.alg.actor_critic.eval()).to(env.device).eval()
    if opt.export:
        path = os.path.join(os.path.dirname(checkpoint), os.path.basename(checkpoint).replace("model_", "joint_actor_"))
        with torch.no_grad():
            example = torch.randn(8, 192, device=env.device)
            traced = torch.jit.trace(module, example)
            torch.testing.assert_close(traced(example), module(example), atol=1e-6, rtol=1e-5)
            traced.save(path)
        print("Exported 192 -> 1 single-joint actor:", path)
        print("This actor requires the new single-joint observation/action ABI; do not load into the old 624 -> 7 deployment.")
    total_error, count = 0., 0
    with torch.no_grad():
        for _ in range(opt.steps):
            target = env.reference.clone()
            obs, _, _, _ = env.step(module(env.obs_buf))
            valid = ~env.reset_buf.bool()
            total_error += (env.q[valid]-target[valid]).abs().sum().item()
            count += valid.sum().item()
    print("Checkpoint:", checkpoint, "mean joint tracking error rad:", total_error/max(count, 1))
    if env.viewer:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    env.asset_temp.cleanup()


if __name__ == "__main__":
    main()
