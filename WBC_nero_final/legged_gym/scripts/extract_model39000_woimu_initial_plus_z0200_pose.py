"""Extract model_39000 woimu's stable +200 mm Isaac Gym startup target.

This utility is simulation-only.  It holds the training initialization
endpoint translated by +0.20 m in Cartesian Z, runs the exact 85-D no-IMU
checkpoint at zero locomotion command, and authenticates the settled 19-axis
policy target used by real-hardware prepositioning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys


WBC_ROOT = Path(__file__).resolve().parents[2]
if str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

import isaacgym  # noqa: F401 - must precede torch-backed gym modules
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


DURATION_S = float(os.environ.get("MODEL39000_EXTRACTION_DURATION_S", "12.0"))
STABLE_WINDOW_S = 1.0
Z_OFFSET_M = 0.20
NATIVE_HOLD_S = float(os.environ.get("MODEL39000_NATIVE_HOLD_S", "2.0"))
TRAJECTORY_S = 3.0
CLAMP_ROOT_EACH_POLICY_STEP = (
    os.environ.get("MODEL39000_CLAMP_ROOT_EACH_POLICY_STEP", "1") == "1"
)
CHECKPOINT_SHA256 = (
    "76763078b3fe27359e70750324a4e0ed792ce93cf565cd428eceb1162e396bed"
)
OUTPUT = Path(
    os.environ.get(
        "MODEL39000_POSE_OUTPUT",
        str(
            WBC_ROOT.parent
            / "quad-deploy/commissioning/"
            "model39000_woimu_isaacgym_initial_plus_z0200_pose.json"
        ),
    )
).expanduser()


def _sphere_to_cartesian(sphere):
    radius, pitch, yaw = np.asarray(sphere, dtype=np.float64)
    return np.asarray(
        [
            radius * math.cos(pitch) * math.cos(yaw),
            radius * math.cos(pitch) * math.sin(yaw),
            radius * math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _cartesian_to_sphere(cartesian):
    x, y, z = np.asarray(cartesian, dtype=np.float64)
    radius = float(np.linalg.norm(cartesian))
    return np.asarray(
        [radius, math.asin(z / radius), math.atan2(y, x)], dtype=np.float64
    )


def _configure(env_cfg):
    env_cfg.env.num_envs = 1
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
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.zero_vel_cmd_prob = 1.0
    physics_dt_override = os.environ.get("MODEL39000_PHYSICS_DT_S")
    if physics_dt_override is not None:
        physics_dt = float(physics_dt_override)
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError("MODEL39000_PHYSICS_DT_S must be positive and finite")
        policy_dt = 0.02
        decimation = int(round(policy_dt / physics_dt))
        if not math.isclose(
            decimation * physics_dt, policy_dt, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "MODEL39000_PHYSICS_DT_S must divide the trained 0.02 s policy period"
            )
        env_cfg.sim.dt = physics_dt
        env_cfg.control.decimation = decimation

    native_sphere = np.asarray(
        env_cfg.goal_ee.ranges.init_pos_end, dtype=np.float64
    )
    native_cart = _sphere_to_cartesian(native_sphere)
    shifted_cart = native_cart + np.asarray([0.0, 0.0, Z_OFFSET_M])
    shifted_sphere = _cartesian_to_sphere(shifted_cart)
    # Populate the recurrent history at the checkpoint's native endpoint,
    # then use the same spherical-coordinate interpolation as training.  A
    # +200 mm first-frame step would be an avoidable distribution shift.
    env_cfg.goal_ee.ranges.init_pos_start = native_sphere.tolist()
    env_cfg.goal_ee.ranges.init_pos_end = native_sphere.tolist()
    return native_sphere, native_cart, shifted_sphere, shifted_cart


def extract(args):
    if args.task != "a2nero_pos_force" or int(args.checkpoint) != 39000:
        raise ValueError(
            "use --task a2nero_pos_force --load_run "
            "Sep12_12-08-00_dr_stage1 --checkpoint 39000"
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    native_sphere, native_cart, target_sphere, target_cart = _configure(env_cfg)
    if int(env_cfg.env.num_single_obs) != 85:
        raise RuntimeError("woimu extraction did not select the 85-D observation")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    checkpoint_path = (
        Path(runner.loaded_checkpoint_path).resolve()
        if hasattr(runner, "loaded_checkpoint_path")
        else (
            WBC_ROOT
            / "logs/a2nero_pos_force/Sep12_12-08-00_dr_stage1/model_39000.pt"
        ).resolve()
    )
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_hash != CHECKPOINT_SHA256:
        raise RuntimeError(
            "model39000 woimu Isaac Gym checkpoint hash mismatch: "
            f"{actual_hash}"
        )

    env.play = True
    fixed_root_state = env.root_states[0].detach().clone()
    fixed_root_state[7:13] = 0.0
    root_env_ids = torch.zeros(1, dtype=torch.int32, device=env.device)
    policy_info = {}
    policy_hz = int(round(1.0 / float(env.dt)))
    steps = int(round(DURATION_S * policy_hz))
    window_steps = int(round(STABLE_WINDOW_S * policy_hz))
    measured_rows = []
    target_rows = []
    velocity_rows = []
    base_rows = []
    pre_clamp_base_rows = []
    goal_rows = []
    ee_world_rows = []
    goal_center_world_rows = []
    arm_base_world_rows = []
    ee_goal_world_rows = []
    ee_relative_goal_center_yaw_cart_rows = []
    ee_relative_goal_center_yaw_rows = []
    ee_relative_arm_base_rows = []
    target_relative_arm_base_rows = []
    observation_frame_rows = []
    native_measured_rows = []
    native_target_rows = []
    native_velocity_rows = []
    reset_ticks = []
    for tick in range(steps):
        env.commands[:, 0:3] = 0.0
        elapsed = tick / float(policy_hz)
        trajectory_alpha = np.clip(
            (elapsed - NATIVE_HOLD_S) / TRAJECTORY_S, 0.0, 1.0
        )
        commanded_sphere = native_sphere + trajectory_alpha * (
            target_sphere - native_sphere
        )
        env._set_teleop_ee_goal(
            ee_goal_sphere=torch.as_tensor(
                commanded_sphere,
                device=env.device,
                dtype=env.curr_ee_goal_sphere.dtype,
            ).reshape(1, 3)
        )
        actions = policy(obs, policy_info)
        if actions.shape[-1] < 19 or not bool(torch.all(torch.isfinite(actions))):
            raise RuntimeError("invalid model39000 woimu Isaac Gym action")
        policy_target = (
            env.default_dof_pos_wo_gripper[:, :19]
            + float(env.cfg.control.action_scale) * actions[:, :19]
        )
        obs, _, dones, _ = env.step(actions.detach())
        pre_clamp_root_state = env.root_states[0, :13].detach().clone()
        # Pose extraction only: remove floating-base drift while leaving all
        # 19 articulated axes under this checkpoint's actor.  No such clamp is
        # used by real deployment.
        if CLAMP_ROOT_EACH_POLICY_STEP:
            env.root_states[0] = fixed_root_state
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.root_states),
                gymtorch.unwrap_tensor(root_env_ids),
                1,
            )
        if bool(torch.any(dones)):
            reset_ticks.append(tick + 1)
        measured_rows.append(env.dof_pos[0, :19].detach().cpu().numpy().copy())
        target_rows.append(policy_target[0].detach().cpu().numpy().copy())
        velocity_rows.append(env.dof_vel[0, :19].detach().cpu().numpy().copy())
        base_rows.append(env.root_states[0, :13].detach().cpu().numpy().copy())
        pre_clamp_base_rows.append(
            pre_clamp_root_state.detach().cpu().numpy().copy()
        )
        goal_rows.append(env.curr_ee_goal_sphere[0].detach().cpu().numpy().copy())
        ee_world_rows.append(env.ee_pos[0].detach().cpu().numpy().copy())
        goal_center_world_rows.append(
            env.get_ee_goal_spherical_center()[0].detach().cpu().numpy().copy()
        )
        arm_base_world_rows.append(
            env._get_arm_base_pos()[0].detach().cpu().numpy().copy()
        )
        ee_goal_world_rows.append(
            env.curr_ee_goal_cart_world[0].detach().cpu().numpy().copy()
        )
        ee_relative_goal_center_yaw_cart_rows.append(
            quat_rotate_inverse(
                env.base_yaw_quat,
                env.ee_pos - env.get_ee_goal_spherical_center(),
            )[0].detach().cpu().numpy().copy()
        )
        ee_relative_goal_center_yaw_rows.append(
            env.ee_pos_sphe_arm[0].detach().cpu().numpy().copy()
        )
        ee_relative_arm_base_rows.append(
            quat_rotate_inverse(
                env.base_quat,
                env.ee_pos - env._get_arm_base_pos(),
            )[0].detach().cpu().numpy().copy()
        )
        target_relative_arm_base_rows.append(
            env._get_target_pos_base()[0].detach().cpu().numpy().copy()
        )
        observation_frame_rows.append(
            obs["obs"][0, -int(env.cfg.env.num_single_obs):]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        if NATIVE_HOLD_S - STABLE_WINDOW_S <= elapsed < NATIVE_HOLD_S:
            native_measured_rows.append(measured_rows[-1].copy())
            native_target_rows.append(target_rows[-1].copy())
            native_velocity_rows.append(velocity_rows[-1].copy())

    measured = np.asarray(measured_rows, dtype=np.float64)
    targets = np.asarray(target_rows, dtype=np.float64)
    velocity = np.asarray(velocity_rows, dtype=np.float64)
    base = np.asarray(base_rows, dtype=np.float64)
    pre_clamp_base = np.asarray(pre_clamp_base_rows, dtype=np.float64)
    goals = np.asarray(goal_rows, dtype=np.float64)
    ee_world = np.asarray(ee_world_rows, dtype=np.float64)
    goal_center_world = np.asarray(goal_center_world_rows, dtype=np.float64)
    arm_base_world = np.asarray(arm_base_world_rows, dtype=np.float64)
    ee_goal_world = np.asarray(ee_goal_world_rows, dtype=np.float64)
    ee_relative_goal_center_yaw_cart = np.asarray(
        ee_relative_goal_center_yaw_cart_rows, dtype=np.float64
    )
    ee_relative_goal_center_yaw_sphere = np.asarray(
        ee_relative_goal_center_yaw_rows, dtype=np.float64
    )
    ee_relative_arm_base = np.asarray(
        ee_relative_arm_base_rows, dtype=np.float64
    )
    target_relative_arm_base = np.asarray(
        target_relative_arm_base_rows, dtype=np.float64
    )
    observation_frames = np.asarray(observation_frame_rows, dtype=np.float64)
    native_measured = np.asarray(native_measured_rows, dtype=np.float64)
    native_targets = np.asarray(native_target_rows, dtype=np.float64)
    native_velocity = np.asarray(native_velocity_rows, dtype=np.float64)
    native_start = int(round((NATIVE_HOLD_S - STABLE_WINDOW_S) * policy_hz))
    native_stop = int(round(NATIVE_HOLD_S * policy_hz))
    if (
        native_measured.shape != (window_steps, 19)
        or native_targets.shape != (window_steps, 19)
        or native_velocity.shape != (window_steps, 19)
    ):
        raise RuntimeError(
            "model39000 extraction did not collect one complete native stable window"
        )
    shifted_first_tick = int(
        math.ceil((NATIVE_HOLD_S + TRAJECTORY_S + 0.5) * policy_hz)
    )
    shifted_windows = []
    for window_start in range(shifted_first_tick, steps - window_steps + 1):
        window_stop = window_start + window_steps
        measured_p2p_window = np.ptp(
            measured[window_start:window_stop], axis=0
        )
        target_p2p_window = np.ptp(
            targets[window_start:window_stop], axis=0
        )
        shifted_windows.append(
            (
                float(max(np.max(measured_p2p_window), np.max(target_p2p_window))),
                window_start,
                window_stop,
                measured_p2p_window,
                target_p2p_window,
            )
        )
    if not shifted_windows:
        raise RuntimeError("model39000 extraction has no post-trajectory stable window")
    (
        selected_shifted_p2p,
        selected_shifted_start,
        selected_shifted_stop,
        selected_shifted_measured_p2p,
        selected_shifted_target_p2p,
    ) = min(shifted_windows, key=lambda item: item[0])
    base_quat = base[-window_steps:, 3:7]
    up_z = 1.0 - 2.0 * (base_quat[:, 0] ** 2 + base_quat[:, 1] ** 2)
    tilt_deg = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    result = {
        "checkpoint": 39000,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_source": str(checkpoint_path),
        "task": args.task,
        "domain_rand_stage": 1,
        "source_kind": "isaacgym_model39000_woimu_native_start_plus_z0200",
        "observation_abi": "85d_no_base_ee_relative_imu_tail",
        "base_constraint_mode": (
            "floating_root_state_clamped_each_policy_step"
            if CLAMP_ROOT_EACH_POLICY_STEP
            else "native_floating_root"
        ),
        "root_clamp_timing": (
            "after_env_step_observation_before_next_policy_evaluation"
            if CLAMP_ROOT_EACH_POLICY_STEP
            else "not_applicable"
        ),
        "sim_device": str(args.sim_device),
        "gpu_pipeline": bool(args.use_gpu_pipeline),
        "policy_hz": policy_hz,
        "physics_dt_s": float(env.sim_params.dt),
        "control_decimation": int(env.cfg.control.decimation),
        "duration_s": DURATION_S,
        "stable_window_s": STABLE_WINDOW_S,
        "teleop_mode": True,
        "native_hold_s": NATIVE_HOLD_S,
        "training_spherical_interpolation_s": TRAJECTORY_S,
        "native_initial_ee_sphere": native_sphere.tolist(),
        "native_initial_ee_cartesian_m": native_cart.tolist(),
        "native_sim_measured_median_rad": np.median(
            native_measured, axis=0
        ).tolist(),
        "native_sim_policy_target_median_rad": np.median(
            native_targets, axis=0
        ).tolist(),
        "native_sim_measured_p2p_rad": np.ptp(
            native_measured, axis=0
        ).tolist(),
        "native_sim_policy_target_p2p_rad": np.ptp(
            native_targets, axis=0
        ).tolist(),
        "native_sim_joint_speed_absmax_rad_s": np.max(
            np.abs(native_velocity), axis=0
        ).tolist(),
        "native_sim_ee_position_world_median_xyz": np.median(
            ee_world[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_goal_center_world_median_xyz": np.median(
            goal_center_world[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_arm_base_world_median_xyz": np.median(
            arm_base_world[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_ee_goal_world_median_xyz": np.median(
            ee_goal_world[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_ee_cartesian_about_goal_center_yaw_median_xyz": np.median(
            ee_relative_goal_center_yaw_cart[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_ee_sphere_about_goal_center_median": np.median(
            ee_relative_goal_center_yaw_sphere[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_ee_relative_arm_base_median_xyz": np.median(
            ee_relative_arm_base[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_target_relative_arm_base_median_xyz": np.median(
            target_relative_arm_base[native_start:native_stop], axis=0
        ).tolist(),
        "native_sim_base_position_median_xyz": np.median(
            base[native_start:native_stop, :3], axis=0
        ).tolist(),
        "native_sim_base_quaternion_xyzw_median": np.median(
            base[native_start:native_stop, 3:7], axis=0
        ).tolist(),
        "native_sim_pre_clamp_base_position_median_xyz": np.median(
            pre_clamp_base[native_start:native_stop, :3], axis=0
        ).tolist(),
        "native_sim_pre_clamp_base_quaternion_xyzw_median": np.median(
            pre_clamp_base[native_start:native_stop, 3:7], axis=0
        ).tolist(),
        "native_sim_observation_frame_median": np.median(
            observation_frames[native_start:native_stop], axis=0
        ).tolist(),
        "startup_ee_offset_from_native_initial_m": [0.0, 0.0, Z_OFFSET_M],
        "startup_ee_sphere": target_sphere.tolist(),
        "startup_ee_cartesian_m": target_cart.tolist(),
        "selected_shifted_window_start_s": (
            selected_shifted_start / float(policy_hz)
        ),
        "selected_shifted_window_end_s": (
            selected_shifted_stop / float(policy_hz)
        ),
        "selected_shifted_max_p2p_rad": selected_shifted_p2p,
        "selected_shifted_sim_measured_median_rad": np.median(
            measured[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_policy_target_median_rad": np.median(
            targets[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_measured_p2p_rad": (
            selected_shifted_measured_p2p.tolist()
        ),
        "selected_shifted_sim_policy_target_p2p_rad": (
            selected_shifted_target_p2p.tolist()
        ),
        "selected_shifted_sim_ee_position_world_median_xyz": np.median(
            ee_world[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_goal_center_world_median_xyz": np.median(
            goal_center_world[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_arm_base_world_median_xyz": np.median(
            arm_base_world[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_ee_goal_world_median_xyz": np.median(
            ee_goal_world[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_ee_cartesian_about_goal_center_yaw_median_xyz": np.median(
            ee_relative_goal_center_yaw_cart[
                selected_shifted_start:selected_shifted_stop
            ],
            axis=0,
        ).tolist(),
        "selected_shifted_sim_ee_sphere_about_goal_center_median": np.median(
            ee_relative_goal_center_yaw_sphere[
                selected_shifted_start:selected_shifted_stop
            ],
            axis=0,
        ).tolist(),
        "selected_shifted_sim_ee_relative_arm_base_median_xyz": np.median(
            ee_relative_arm_base[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "selected_shifted_sim_target_relative_arm_base_median_xyz": np.median(
            target_relative_arm_base[selected_shifted_start:selected_shifted_stop],
            axis=0,
        ).tolist(),
        "selected_shifted_sim_base_position_median_xyz": np.median(
            base[selected_shifted_start:selected_shifted_stop, :3], axis=0
        ).tolist(),
        "selected_shifted_sim_base_quaternion_xyzw_median": np.median(
            base[selected_shifted_start:selected_shifted_stop, 3:7], axis=0
        ).tolist(),
        "selected_shifted_sim_pre_clamp_base_position_median_xyz": np.median(
            pre_clamp_base[selected_shifted_start:selected_shifted_stop, :3], axis=0
        ).tolist(),
        "selected_shifted_sim_pre_clamp_base_quaternion_xyzw_median": np.median(
            pre_clamp_base[selected_shifted_start:selected_shifted_stop, 3:7], axis=0
        ).tolist(),
        "selected_shifted_sim_observation_frame_median": np.median(
            observation_frames[selected_shifted_start:selected_shifted_stop], axis=0
        ).tolist(),
        "observation_frame_layout": {
            "base_roll_pitch": [0, 2],
            "scaled_base_angular_velocity": [2, 5],
            "joint_position_minus_default": [5, 24],
            "scaled_joint_velocity": [24, 43],
            "previous_action": [43, 62],
            "gait_sin_cos": [62, 64],
            "ee_orientation_rpy": [64, 67],
            "scaled_commands": [67, 76],
            "target_position_base": [76, 79],
            "target_velocity_base": [79, 82],
            "target_acceleration_base": [82, 85]
        },
        "joint_names": list(env.dof_names[:19]),
        "sim_measured_median_rad": np.median(
            measured[-window_steps:], axis=0
        ).tolist(),
        "sim_policy_target_median_rad": np.median(
            targets[-window_steps:], axis=0
        ).tolist(),
        "sim_measured_p2p_rad": np.ptp(
            measured[-window_steps:], axis=0
        ).tolist(),
        "sim_policy_target_p2p_rad": np.ptp(
            targets[-window_steps:], axis=0
        ).tolist(),
        "sim_joint_speed_absmax_rad_s": np.max(
            np.abs(velocity[-window_steps:]), axis=0
        ).tolist(),
        "base_position_median_xyz": np.median(
            base[-window_steps:, :3], axis=0
        ).tolist(),
        "base_tilt_deg_median": float(np.median(tilt_deg)),
        "base_tilt_deg_max": float(np.max(tilt_deg)),
        "ee_goal_sphere_median": np.median(
            goals[-window_steps:], axis=0
        ).tolist(),
        "reset_ticks": reset_ticks,
    }
    print(
        "MODEL39000 WOIMU EXTRACTION DIAGNOSTIC: "
        f"height={result['base_position_median_xyz'][2]:.3f}m, "
        f"max_tilt={result['base_tilt_deg_max']:.3f}deg, "
        f"max_measured_p2p={max(result['sim_measured_p2p_rad']):.6f}rad, "
        f"max_target_p2p={max(result['sim_policy_target_p2p_rad']):.6f}rad, "
        f"resets={reset_ticks}"
    )
    allow_unsettled_output = os.environ.get(
        "MODEL39000_ALLOW_UNSETTLED_DIAGNOSTIC", "0"
    ) == "1"
    if reset_ticks:
        raise RuntimeError("Isaac Gym reset during model39000 woimu extraction")
    if float(result["base_position_median_xyz"][2]) < 0.28:
        raise RuntimeError("model39000 woimu startup pose is not standing")
    if (
        float(result["base_tilt_deg_max"]) > 10.0
        and not allow_unsettled_output
    ):
        raise RuntimeError("model39000 woimu startup pose is not upright")
    max_p2p = max(
        result["sim_measured_p2p_rad"] + result["sim_policy_target_p2p_rad"]
    )
    result["settled_gate_passed"] = bool(max_p2p <= 0.01)
    if max_p2p > 0.01 and not allow_unsettled_output:
        raise RuntimeError(
            f"model39000 woimu startup pose did not settle: p2p={max_p2p:.6f}rad"
        )
    if not np.allclose(result["ee_goal_sphere_median"], target_sphere, atol=1e-5):
        raise RuntimeError("Isaac Gym did not hold the shifted startup goal")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    result = extract(get_args())
    print("MODEL39000 WOIMU ISAACGYM INITIAL +200MM POSE EXTRACTED")
    print(f"output={OUTPUT}")
    print(f"startup_ee_cartesian_m={result['startup_ee_cartesian_m']}")
    print(f"sim_policy_target_median_rad={result['sim_policy_target_median_rad']}")


if __name__ == "__main__":
    main()
