"""Export model39000's native 85-D gait observations/actions from Isaac Gym.

Run headless with one environment.  The result uses the same group schema as
quad-deploy/scripts/audit_model39000_mujoco_gait_observations.py.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
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
from legged_gym.utils import get_args, task_registry


CHECKPOINT_SHA256 = (
    "76763078b3fe27359e70750324a4e0ed792ce93cf565cd428eceb1162e396bed"
)
TARGET_SPHERE = np.asarray(
    [0.6435060501098633, 0.7744095921516418, 0.0], dtype=np.float64
)
OBSERVATION_GROUPS = {
    "base_roll_pitch": (0, 2),
    "scaled_base_angular_velocity": (2, 5),
    "joint_position_minus_default": (5, 24),
    "scaled_joint_velocity": (24, 43),
    "previous_action": (43, 62),
    "gait_sin_cos": (62, 64),
    "ee_orientation_rpy": (64, 67),
    "scaled_commands": (67, 76),
    "target_position_base": (76, 79),
    "target_velocity_base": (79, 82),
    "target_acceleration_base": (82, 85),
}
OUTPUT = Path(
    os.environ.get(
        "MODEL39000_ISAAC_GAIT_AUDIT_OUTPUT",
        str(
            WBC_ROOT.parent
            / "quad-deploy/artifacts/mujoco/model_39000/"
            "model39000_isaacgym_gait_observation_audit.json"
        ),
    )
).expanduser()


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(array, axis=0).tolist(),
        "p01": np.percentile(array, 1.0, axis=0).tolist(),
        "median": np.median(array, axis=0).tolist(),
        "p99": np.percentile(array, 99.0, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "rms": np.sqrt(np.mean(np.square(array), axis=0)).tolist(),
        "p2p": np.ptp(array, axis=0).tolist(),
    }


def _configure(env_cfg):
    env_cfg.env.num_envs = int(
        os.environ.get("MODEL39000_ISAAC_GAIT_NUM_ENVS", "1")
    )
    env_cfg.env.test = True
    env_cfg.env.teleop_mode = True
    env_cfg.env.episode_length_s = 100.0
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.height = [0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.commands.zero_vel_cmd_prob = 1.0
    env_cfg.goal_ee.ranges.init_pos_start = TARGET_SPHERE.tolist()
    env_cfg.goal_ee.ranges.init_pos_end = TARGET_SPHERE.tolist()
    physics_dt = float(os.environ.get("MODEL39000_ISAAC_GAIT_PHYSICS_DT_S", "0.005"))
    policy_dt = 0.02
    decimation = int(round(policy_dt / physics_dt))
    if not math.isclose(decimation * physics_dt, policy_dt, abs_tol=1.0e-12):
        raise ValueError("physics dt must divide the 20 ms policy period")
    env_cfg.sim.dt = physics_dt
    env_cfg.control.decimation = decimation


def audit(args):
    if args.task != "a2nero_pos_force" or int(args.checkpoint) < 0:
        raise ValueError("select a2nero_pos_force and an explicit checkpoint")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    domain_stage = int(getattr(args, "domain_rand_stage", 1))
    env_cfg, train_cfg = apply_a2nero_domain_rand_stage(
        env_cfg, train_cfg, domain_stage
    )
    _configure(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    full_identified = os.environ.get(
        "MODEL39000_ISAAC_GAIT_FULL_IDENTIFIED", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if domain_stage in (4, 5):
        if full_identified:
            env.set_stage3_curriculum_progress(1, 1)
        else:
            env.set_stage3_nominal_override(True)
        env.reset()
    obs = env.get_observations()
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    checkpoint = (
        Path(runner.loaded_checkpoint_path).resolve()
        if hasattr(runner, "loaded_checkpoint_path")
        else (
            WBC_ROOT
            / "logs/a2nero_pos_force/Sep12_12-08-00_dr_stage1/model_39000.pt"
        ).resolve()
    )
    import hashlib
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_sha256 = os.environ.get("MODEL39000_EXPECT_CHECKPOINT_SHA256", "")
    if not expected_sha256 and int(args.checkpoint) == 39000:
        expected_sha256 = CHECKPOINT_SHA256
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise RuntimeError(
            "unexpected model39000 checkpoint SHA256: "
            f"{checkpoint_sha256} != {expected_sha256}"
        )

    env.play = True
    dt = float(env.dt)
    settle_s = 6.0
    gait_s = 3.0
    final_s = 1.0
    total_ticks = int(round((settle_s + gait_s + final_s) / dt))
    command = np.asarray(
        [float(value) for value in os.environ.get(
            "MODEL39000_ISAAC_GAIT_COMMAND", "-0.3,0,0"
        ).split(",")], dtype=np.float64
    )
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("MODEL39000_ISAAC_GAIT_COMMAND must have three values")
    observations = []
    histories = []
    actions = []
    targets = []
    positions = []
    velocities = []
    bases = []
    body_motion = []
    ee_positions = []
    ee_positions_base = []
    reset_ticks = []
    reset_during_gait = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    info = {}
    for tick in range(total_ticks):
        elapsed = tick * dt
        active = settle_s <= elapsed < settle_s + gait_s
        env.commands[:, 0:3] = torch.as_tensor(
            command if active else np.zeros(3),
            dtype=env.commands.dtype,
            device=env.device,
        ).reshape(1, 3)
        env._set_teleop_ee_goal(
            ee_goal_sphere=torch.as_tensor(
                TARGET_SPHERE,
                dtype=env.curr_ee_goal_sphere.dtype,
                device=env.device,
            ).reshape(1, 3)
        )
        action = policy(obs, info)
        target = (
            env.default_dof_pos_wo_gripper[:, :19]
            + float(env.cfg.control.action_scale) * action[:, :19]
        )
        obs, _, dones, _ = env.step(action.detach())
        if bool(torch.any(dones)):
            reset_envs = torch.nonzero(dones, as_tuple=False).flatten()
            reset_ticks.append({
                "tick": tick + 1,
                "env_ids": reset_envs.detach().cpu().tolist(),
            })
            if active:
                reset_during_gait[reset_envs] = True
        observations.append(
            obs["obs"][:, -85:].detach().cpu().numpy().copy()
        )
        histories.append(obs["obs"].detach().cpu().numpy().copy())
        actions.append(action[:, :19].detach().cpu().numpy().copy())
        targets.append(target.detach().cpu().numpy().copy())
        positions.append(env.dof_pos[:, :19].detach().cpu().numpy().copy())
        velocities.append(env.dof_vel[:, :19].detach().cpu().numpy().copy())
        bases.append(env.root_states[:, :13].detach().cpu().numpy().copy())
        body_motion.append(torch.cat(
            (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1
        ).detach().cpu().numpy().copy())
        ee_positions.append(env.ee_pos.detach().cpu().numpy().copy())
        # End-effector vibration must be measured in the same base-relative
        # frame used by the task.  World-frame p2p includes the commanded
        # chassis translation and is therefore not a vibration metric.
        arm_base_pos = env._get_arm_base_pos()
        ee_pos_base = quat_rotate_inverse(
            env.base_quat,
            env.ee_pos - arm_base_pos,
        )
        ee_positions_base.append(
            ee_pos_base.detach().cpu().numpy().copy()
        )

    start = int(round(settle_s / dt))
    stop = int(round((settle_s + gait_s) / dt))
    observation = np.asarray(observations)[start:stop]
    target = np.asarray(targets)[start:stop]
    target_step = np.diff(target, axis=0)
    ee_gait_world = np.asarray(ee_positions)[start:stop]
    ee_gait_base = np.asarray(ee_positions_base)[start:stop]
    body_motion_gait = np.asarray(body_motion)[start:stop]
    command_error = body_motion_gait - command.reshape(1, 1, 3)
    tracking_sigma = float(env.cfg.rewards.tracking_sigma)
    command_tracking_reward = (
        float(env.reward_scales["tracking_lin_vel"])
        * np.exp(
            -np.sum(np.square(command_error[:, :, :2]), axis=2)
            / tracking_sigma
        )
        + float(env.reward_scales["tracking_ang_vel"])
        * np.exp(-np.square(command_error[:, :, 2]) / tracking_sigma)
    )
    ee_p2p_by_env_mm = 1000.0 * np.linalg.norm(
        np.ptp(ee_gait_base, axis=0), axis=1
    )
    nero_target_abs_step = np.abs(target_step[:, :, 12:19])
    flattened_observation = observation.reshape(-1, observation.shape[-1])
    result = {
        "schema": "model39000_woimu.isaacgym_observation_action_audit.v1",
        "checkpoint": int(args.checkpoint),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "domain_rand_stage": domain_stage,
        "full_identified_plant": full_identified,
        "num_envs": int(env.num_envs),
        "physics_dt_s": float(env.sim_params.dt),
        "policy_dt_s": dt,
        "command_xyz": command.tolist(),
        "target_sphere": TARGET_SPHERE.tolist(),
        "reset_ticks": reset_ticks,
        "observation_layout": OBSERVATION_GROUPS,
        "observation_groups": {
            name: _distribution(flattened_observation[:, begin:end])
            for name, (begin, end) in OBSERVATION_GROUPS.items()
        },
        "stacked_history": _distribution(
            np.asarray(histories)[start:stop].reshape(-1, env.num_obs)
        ),
        "raw_action": _distribution(
            np.asarray(actions)[start:stop].reshape(-1, 19)
        ),
        "policy_target_rad": _distribution(target.reshape(-1, 19)),
        "policy_target_step_rad": _distribution(target_step.reshape(-1, 19)),
        "nero_target_max_abs_step_rad": np.max(
            nero_target_abs_step, axis=(0, 1)
        ).tolist(),
        "nero_target_group_max_abs_step_rad": float(
            np.max(nero_target_abs_step)
        ),
        "nero_target_group_p99_abs_step_rad": float(
            np.percentile(nero_target_abs_step, 99.0)
        ),
        "joint_position_rad": _distribution(
            np.asarray(positions)[start:stop].reshape(-1, 19)
        ),
        "joint_velocity_rad_s": _distribution(
            np.asarray(velocities)[start:stop].reshape(-1, 19)
        ),
        "base_state": _distribution(
            np.asarray(bases)[start:stop].reshape(-1, 13)
        ),
        "ee_position_world_m": _distribution(
            ee_gait_world.reshape(-1, 3)
        ),
        "ee_position_base_m": _distribution(
            ee_gait_base.reshape(-1, 3)
        ),
        "walking_ee_position_p2p_frame": "arm-base origin, base orientation",
        "walking_ee_position_p2p_mm_by_env": ee_p2p_by_env_mm.tolist(),
        "walking_ee_position_p2p_p95_mm": float(
            np.percentile(ee_p2p_by_env_mm, 95.0)
        ),
        "gait_success_rate": float(
            torch.mean((~reset_during_gait).float()).item()
        ),
        "command_tracking_rmse": np.sqrt(
            np.mean(np.square(command_error), axis=(0, 1))
        ).tolist(),
        "command_tracking_vector_rmse": float(
            np.sqrt(np.mean(np.square(command_error)))
        ),
        "command_tracking_reward_mean": float(
            np.mean(command_tracking_reward)
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit(get_args())
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "physics_dt_s": result["physics_dt_s"],
        "reset_ticks": result["reset_ticks"],
        "nero_target_group_max_abs_step_rad": (
            result["nero_target_group_max_abs_step_rad"]
        ),
        "nero_target_group_p99_abs_step_rad": (
            result["nero_target_group_p99_abs_step_rad"]
        ),
        "walking_ee_position_p2p_p95_mm": (
            result["walking_ee_position_p2p_p95_mm"]
        ),
        "gait_success_rate": result["gait_success_rate"],
        "command_tracking_vector_rmse": (
            result["command_tracking_vector_rmse"]
        ),
        "command_tracking_reward_mean": (
            result["command_tracking_reward_mean"]
        ),
    }, indent=2))
