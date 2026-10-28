"""Train one joint, or verify the physics/observation ABI without training."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import isaacgym  # must precede torch
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.nero.joint_probe_utils import identification, pd_candidates, find_checkpoint
from nero_joint_contract import A2NERO_STAGE1_MODE


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--joint", required=True, type=int, choices=range(1, 8))
    parser.add_argument("--pose-bank-size", type=int, default=128)
    parser.add_argument("--pose-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = "nero_joint"
    args.load_run = -1 if args.load_run in (None, "-1") else args.load_run
    args.checkpoint = -1 if args.checkpoint is None else args.checkpoint
    if custom.pose_bank_size < 1:
        raise ValueError("pose-bank-size must be positive")
    if args.experiment_name:
        raise ValueError("Experiment is isolated by joint; do not override --experiment_name")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.joint = custom.joint
    env_cfg.env.pose_bank_size = custom.pose_bank_size
    env_cfg.env.pose_seed = custom.pose_seed
    source = os.path.join(ROOT, "analysis/nero_newpd_sysid/newpd_actuator_fit.json")
    _, provenance = identification(source)
    train_cfg.runner.experiment_name = "nero_joint_a2stage1_J%d" % custom.joint
    train_cfg.runner.run_name = "J%d_a2stage1_arm_physx_pd10_0p8" % custom.joint
    log_root = os.path.join(ROOT, "logs", train_cfg.runner.experiment_name)
    if args.resume:
        checkpoint = find_checkpoint(log_root, args.load_run, args.checkpoint)
        with open(os.path.join(os.path.dirname(checkpoint), "joint_config.json")) as f:
            saved = json.load(f)
        for key in ("joint", "pose_seed", "pose_bank_size"):
            if saved["env"]["env"][key] != getattr(env_cfg.env, key):
                raise ValueError("Resume mismatch: " + key)
        if saved["env"]["control"] != class_to_dict(env_cfg.control):
            raise ValueError("Resume action / PD / delay ABI mismatch")
        if saved["env"].get("rewards") != class_to_dict(env_cfg.rewards):
            raise ValueError("Resume reward configuration mismatch; start a fresh run")
        # Registry's generic resolver also accepts export filenames; pin the exact model.
        args.load_run = os.path.basename(os.path.dirname(checkpoint))
        args.checkpoint = int(os.path.basename(checkpoint)[len("model_"):-3])
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    print("Single-joint setup:", json.dumps(dict(
        joint=custom.joint, goal_range=[env.goal_lower, env.goal_upper], reference_speed=env.speed,
        command_delta_rad=env.command_delta,
        simulation_dt=env.sim_params.dt, policy_dt=env.dt, kp=env.cfg.control.kp, kd=env.cfg.control.kd,
        action_mode=env.cfg.control.action_mode, action_scale=env.cfg.control.action_scale,
        training_delay_s=env.cfg.control.delay_s,
        random_fixed_poses=len(env.pose_bank)), indent=2), flush=True)
    if custom.smoke:
        # Exercise random actions, fractional delay, timeout/reset and native drives.
        initial_rigid_count = env.gym.get_sim_rigid_body_count(env.sim)
        assert env.gym.get_sim_dof_count(env.sim) == env.num_envs
        env.episode_length_buf.fill_(env.max_episode_length-2)
        with torch.no_grad():
            for tick in range(160):
                previous_target = env.issued_target.clone()
                previous_delta = env.last_target_delta.clone()
                previous_dq = env.dq.clone()
                previous_action = env.actions.clone()
                reference = env.reference.clone()
                ref_vel = env.ref_vel.clone()
                holding = env.clock >= env.travel
                action = torch.empty(env.num_envs, 1, device=env.device).uniform_(-.3, .3)
                obs, reward, done, _ = env.step(action)
                valid = ~done.bool()
                delta = (env.issued_target-previous_target)[valid].abs()
                assert (delta <= env.command_delta + 1e-6).all()
                assert (delta / env.dt < env.command_delta / env.dt / env.cfg.control.command_speed_fraction).all()
                assert obs["obs"].shape == (env.num_envs, 192)
                assert obs["privileged_obs"].shape == (env.num_envs, env.cfg.env.num_privileged_obs)
                assert all(torch.isfinite(x).all() for x in obs.values())
                assert torch.isfinite(reward).all()
                # Check the actual reward path, including signs, normalization and history.
                e = env.q - reference
                dq = env.dq
                vmax = env.speed * 2
                if env.cfg.control.action_mode == A2NERO_STAGE1_MODE:
                    r = env.cfg.rewards
                    span = env.upper-env.lower
                    soft_lower = env.lower+(1-r.soft_dof_pos_limit)*span/2
                    soft_upper = env.upper-(1-r.soft_dof_pos_limit)*span/2
                    violation = (soft_lower-env.q).clamp(min=0)+(env.q-soft_upper).clamp(min=0)
                    expected = env.dt*(r.tracking*torch.exp(-2*e.abs()/r.tracking_sigma)
                        + r.position_error*e.abs()
                        + r.action_rate*(env.actions-previous_action).square()
                        + r.dof_vel*dq.square()
                        + r.dof_acc*((dq-previous_dq)/env.dt).square()
                        + r.dof_pos_limits*violation)
                else:
                    expected = env.dt * (
                        4*torch.exp(-(e/.15)**2) - 2*e.square()
                        - .04*(dq-ref_vel).square()
                        - env.cfg.rewards.hold_velocity*holding*dq.square()
                        - .01*((env.actions-previous_action)/env.dt).square()
                        - .00005*env.forces.square()
                        - .2*(dq.abs()-vmax).clamp(min=0).square()
                        - env.cfg.rewards.target_acceleration*((env.last_target_delta-previous_delta)/env.command_delta).square()
                        - env.cfg.rewards.adapter_gap*(action[:, 0].clamp(-1, 1)-env.actions).square())
                torch.testing.assert_close(reward[valid], expected[valid], atol=1e-5, rtol=1e-5)
                assert (env.last_target_delta[done.bool()] == 0).all()
                assert env.ref_vel.abs().max() <= env.speed + 1e-5
                assert env.reference.min() >= env.goal_lower - 1e-5
                assert env.reference.max() <= env.goal_upper + 1e-5
                assert env.q.min() >= env.lower-.051 and env.q.max() <= env.upper+.051
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actors[0])
        if env.cfg.control.action_mode == A2NERO_STAGE1_MODE:
            strength = props["stiffness"][0]/env.cfg.control.kp
            assert env.cfg.domain_rand.arm_motor_strength_range[0] <= strength <= env.cfg.domain_rand.arm_motor_strength_range[1]
            assert np.allclose(props["damping"], env.cfg.control.kd*strength)
        else:
            assert np.allclose(props["stiffness"], env.cfg.control.kp)
            assert np.allclose(props["damping"], env.cfg.control.kd)
        print("SMOKE PASS: J%d; one DOF/actor, %d bodies, resets, finite observations, ranges, PD." %
              (custom.joint, initial_rigid_count), flush=True)
    else:
        runner, cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
        os.makedirs(runner.log_dir, exist_ok=True)
        manifest = dict(env=class_to_dict(env.cfg), train=class_to_dict(cfg),
                        identification_for_offline_pd_candidates_only=provenance,
                        locked_pose_bank=env.pose_bank.tolist(),
                        action="A2nero Stage-1 q_default+0.25*raw_action; physical target clipped to URDF and slew-limited to 0.99*vmax per 20ms",
                        observation="16 frames of [q-q_default,0.05*dq,q_ref-q_default,0.05*dq_ref,previous_sent_action,locked_q1..q7]; actor q/dq noisy, critic true")
        with open(os.path.join(runner.log_dir, "joint_config.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(runner.log_dir, "pd_candidates_OFFLINE_ONLY.json"), "w") as f:
            json.dump(pd_candidates(source, training_kp=env.cfg.control.kp,
                                    training_kd=env.cfg.control.kd), f, indent=2)
        runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)
    if env.viewer:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    env.asset_temp.cleanup()


if __name__ == "__main__":
    main()
