"""Screen several A2+Nero checkpoints in one Isaac Gym process.

The rollout matches the deterministic nominal gait audit, but reuses the
environment while swapping actor checkpoints.  This makes a dense checkpoint
screen practical without weakening the six-direction evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys


WBC_ROOT = Path(__file__).resolve().parents[2]
if str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.a2.a2nero_pos_force_config import (
    apply_a2nero_domain_rand_stage,
)
from legged_gym.scripts.audit_model39000_isaacgym_gait_observations import (
    TARGET_SPHERE,
    _configure,
)
from legged_gym.utils import get_args, task_registry


DIRECTIONS = {
    "W": (0.3, 0.0, 0.0),
    "S": (-0.3, 0.0, 0.0),
    "Z": (0.0, 0.2, 0.0),
    "C": (0.0, -0.2, 0.0),
    "A": (0.0, 0.0, 0.3),
    "D": (0.0, 0.0, -0.3),
}


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_like_play(env_cfg):
    env_cfg.env.num_envs = int(
        os.environ.get("MODEL39000_ISAAC_GAIT_NUM_ENVS", "1")
    )
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.test = True
    env_cfg.env.teleop_mode = True
    env_cfg.env.episode_length_s = 1.0e9


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return (
        WBC_ROOT
        / "logs"
        / "a2nero_pos_force"
        / run_name
        / f"model_{iteration}.pt"
    )


@torch.inference_mode()
def _rollout(env, policy, command_xyz):
    seed = int(os.environ.get("MODEL39000_SCREEN_SEED", "17800"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    obs = env.get_observations()
    env.play = True

    dt = float(env.dt)
    settle_s = float(os.environ.get("MODEL39000_SCREEN_SETTLE_S", "6.0"))
    gait_s = float(os.environ.get("MODEL39000_SCREEN_GAIT_S", "3.0"))
    final_s = float(os.environ.get("MODEL39000_SCREEN_FINAL_S", "1.0"))
    settle_ticks = int(round(settle_s / dt))
    gait_ticks = int(round(gait_s / dt))
    total_ticks = int(round((settle_s + gait_s + final_s) / dt))
    gait_stop = settle_ticks + gait_ticks
    command = torch.tensor(
        command_xyz, dtype=env.commands.dtype, device=env.device
    ).reshape(1, 3)
    zero_command = torch.zeros_like(command)
    target_sphere_value = (
        env.cfg.goal_ee.ranges.init_pos_start
        if getattr(env, "screen_match_play", False)
        else TARGET_SPHERE
    )
    target_sphere = torch.tensor(
        target_sphere_value,
        dtype=env.curr_ee_goal_sphere.dtype,
        device=env.device,
    ).reshape(1, 3)

    reset_during_gait = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    reset_count = 0
    target_steps = []
    previous_target = None
    ee_positions_base = []
    command_errors = []

    for tick in range(total_ticks):
        active = settle_ticks <= tick < gait_stop
        env.commands[:, :3] = command if active else zero_command
        env._set_teleop_ee_goal(ee_goal_sphere=target_sphere)
        action = policy(obs, {})
        target = (
            env.default_dof_pos_wo_gripper[:, :19]
            + float(env.cfg.control.action_scale) * action[:, :19]
        )
        obs, _, dones, _ = env.step(action)
        done_mask = dones.bool()
        reset_count += int(done_mask.sum().item())
        if active:
            reset_during_gait |= done_mask
            if previous_target is not None:
                target_steps.append(
                    torch.abs(target[:, 12:19] - previous_target[:, 12:19])
                    .detach()
                    .cpu()
                )
            arm_base_pos = env._get_arm_base_pos()
            ee_positions_base.append(
                quat_rotate_inverse(
                    env.base_quat, env.ee_pos - arm_base_pos
                ).detach().cpu()
            )
            body_motion = torch.cat(
                (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1
            )
            command_errors.append((body_motion - command).detach().cpu())
        previous_target = target.detach()

    step_values = torch.stack(target_steps).numpy()
    ee_values = torch.stack(ee_positions_base).numpy()
    errors = torch.stack(command_errors).numpy()
    tracking_sigma = float(env.cfg.rewards.tracking_sigma)
    tracking_reward = (
        float(env.reward_scales["tracking_lin_vel"])
        * np.exp(-np.sum(np.square(errors[:, :, :2]), axis=2) / tracking_sigma)
        + float(env.reward_scales["tracking_ang_vel"])
        * np.exp(-np.square(errors[:, :, 2]) / tracking_sigma)
    )
    ee_p2p_by_env_mm = 1000.0 * np.linalg.norm(
        np.ptp(ee_values, axis=0), axis=1
    )
    return {
        "gait_success_rate": float((~reset_during_gait).float().mean().item()),
        "reset_count_total": reset_count,
        "command_tracking_reward_mean": float(np.mean(tracking_reward)),
        "command_tracking_vector_rmse": float(
            np.sqrt(np.mean(np.square(errors)))
        ),
        "nero_target_group_p99_abs_step_rad": float(
            np.percentile(step_values, 99.0)
        ),
        "walking_ee_position_p2p_p95_mm": float(
            np.percentile(ee_p2p_by_env_mm, 95.0)
        ),
    }


def screen(args):
    run_name = os.environ.get(
        "MODEL39000_SCREEN_RUN", "Sep12_12-08-00_dr_stage1"
    )
    iterations = [
        int(value)
        for value in os.environ.get(
            "MODEL39000_SCREEN_CHECKPOINTS", "38000,39000,40000"
        ).split(",")
        if value.strip()
    ]
    if not iterations:
        raise ValueError("MODEL39000_SCREEN_CHECKPOINTS is empty")
    paths = [_checkpoint_path(run_name, iteration) for iteration in iterations]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    args.task = "a2nero_pos_force"
    args.resume = True
    args.load_run = run_name
    args.checkpoint = iterations[0]
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    match_play = _flag("MODEL39000_SCREEN_MATCH_PLAY", True)
    if match_play:
        _configure_like_play(env_cfg)
    else:
        env_cfg, train_cfg = apply_a2nero_domain_rand_stage(
            env_cfg, train_cfg, stage=4
        )
        _configure(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.screen_match_play = match_play
    if not match_play:
        # The plant has already been created with nominal properties.  Keeping
        # the Stage-3 callback enabled would redundantly rewrite every property
        # after each fall without changing nominal physics.
        env.set_stage3_nominal_override(True)
        env.stage3_enabled = False
        env._configure_stage3_actuator()
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    results = []
    for iteration, path in zip(iterations, paths):
        runner.load(str(path), load_optimizer=False)
        runner.alg.actor_critic.eval()
        directions = {
            label: _rollout(env, policy, command)
            for label, command in DIRECTIONS.items()
        }
        aggregate = {
            "min_gait_success_rate": min(
                result["gait_success_rate"] for result in directions.values()
            ),
            "mean_gait_success_rate": float(np.mean([
                result["gait_success_rate"] for result in directions.values()
            ])),
            "mean_command_tracking_reward": float(np.mean([
                result["command_tracking_reward_mean"]
                for result in directions.values()
            ])),
            "max_nero_target_step_p99_rad": max(
                result["nero_target_group_p99_abs_step_rad"]
                for result in directions.values()
            ),
            "max_walking_ee_p2p_p95_mm": max(
                result["walking_ee_position_p2p_p95_mm"]
                for result in directions.values()
            ),
            "reset_count_total": sum(
                result["reset_count_total"] for result in directions.values()
            ),
        }
        results.append({
            "iteration": iteration,
            "checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "directions": directions,
            "aggregate": aggregate,
        })
        print(json.dumps({"iteration": iteration, **aggregate}, sort_keys=True))

    output = Path(os.environ.get(
        "MODEL39000_SCREEN_OUTPUT",
        str(WBC_ROOT / "artifacts/model39000_nominal_checkpoint_screen.json"),
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "run": run_name,
        "num_envs_per_direction": int(env.num_envs),
        "deterministic": True,
        "nominal_plant": True,
        "matches_play_configuration": match_play,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    screen(get_args())
