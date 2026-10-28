#!/usr/bin/env python3
"""Strict ABI and forward-pass check for the immutable A2+Nero checkpoint."""

import argparse
import hashlib
import os
import re
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic

# Load the config submodule without executing legged_gym.envs.__init__, whose
# legacy registry eagerly imports Isaac Gym.  ABI checking is intentionally a
# CPU/PyTorch-only operation.
if "legged_gym.envs" not in sys.modules:
    envs_package = types.ModuleType("legged_gym.envs")
    envs_package.__path__ = [os.path.join(REPO_ROOT, "legged_gym", "envs")]
    sys.modules["legged_gym.envs"] = envs_package
from legged_gym.envs.a2.a2nero_pos_force_config import (
    A2NEROPosForceRoughCfg,
    A2NEROPosForceRoughCfgPPO,
)


EXPECTED_SHAPES = {
    "adaptation_encoder_module.mlp.0.weight": (512, 1360),
    "adaptation_decoder_module.4.weight": (27, 64),
    "actor_body.0.weight": (512, 117),
    "actor_body.6.weight": (19, 128),
    "critic_body.0.weight": (512, 473),
    "std": (19,),
}
EXPECTED_LABELS = [
    "ee_pos_base_loss",
    "ee_vel_base_loss",
    "tracking_error_base_loss",
    "future_ee_pos_base_loss",
    "future_tracking_error_base_loss",
]
EXPECTED_DIMS = [3, 3, 3, 9, 9]
EXPECTED_DEFAULTS = {
    "FL_hip_joint": 0.15,
    "FL_thigh_joint": 0.67,
    "FL_calf_joint": -1.32,
    "FR_hip_joint": -0.15,
    "FR_thigh_joint": 0.67,
    "FR_calf_joint": -1.32,
    "RL_hip_joint": 0.15,
    "RL_thigh_joint": 0.67,
    "RL_calf_joint": -1.32,
    "RR_hip_joint": -0.15,
    "RR_thigh_joint": 0.67,
    "RR_calf_joint": -1.32,
    "nero_joint1": 0.0,
    "nero_joint2": -1.0,
    "nero_joint3": 0.0,
    "nero_joint4": 1.8,
    "nero_joint5": 0.0,
    "nero_joint6": 0.0,
    "nero_joint7": 0.3,
}
EXPECTED_EE_QUAT = [0.70710678, 0.0, 0.0, 0.70710678]


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(path):
    env_cfg = A2NEROPosForceRoughCfg()
    train_cfg = A2NEROPosForceRoughCfgPPO()

    _assert_equal(env_cfg.env.num_single_obs, 85, "num_single_obs")
    _assert_equal(env_cfg.env.frame_stack, 16, "frame_stack")
    _assert_equal(env_cfg.env.num_observations, 1360, "stacked observation")
    _assert_equal(env_cfg.env.num_pred_obs, 27, "num_pred_obs")
    _assert_equal(env_cfg.env.num_actions, 19, "num_actions")
    _assert_equal(env_cfg.control.action_scale, 0.25, "action_scale")
    _assert_equal(env_cfg.env.num_privileged_obs, 441, "critic privileged observation")
    _assert_equal(train_cfg.policy.temporal_encoder, "mlp", "temporal encoder")
    _assert_equal(train_cfg.policy.adaptation_num_single_obs, 85, "adaptation_num_single_obs")
    _assert_equal(train_cfg.policy.latent_dim, 32, "latent_dim")
    _assert_equal(train_cfg.policy.adaptation_labels, EXPECTED_LABELS, "adaptation labels")
    _assert_equal(train_cfg.policy.adaptation_dims, EXPECTED_DIMS, "adaptation dims")
    _assert_equal(env_cfg.asset.ee_orientation_body_to_goal_frame_quat, EXPECTED_EE_QUAT, "EE goal-frame quaternion")
    _assert_equal(env_cfg.goal_ee.ranges.orn_r, [0.0, 0.0], "EE roll command range")
    _assert_equal(env_cfg.goal_ee.ranges.orn_p, [0.0, 0.0], "EE pitch command range")
    _assert_equal(env_cfg.goal_ee.ranges.orn_y, [0.0, 0.0], "EE yaw command range")
    _assert_equal(env_cfg.rewards.scales.tracking_ee_orn, 2.0, "EE orientation reward scale")
    for name, value in EXPECTED_DEFAULTS.items():
        _assert_equal(env_cfg.init_state.default_joint_angles[name], value, f"default pose {name}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    for key, shape in EXPECTED_SHAPES.items():
        _assert_equal(tuple(state_dict[key].shape), shape, key)

    model = ActorCritic(
        env_cfg.env.num_observations,
        env_cfg.env.num_privileged_obs,
        env_cfg.env.num_pred_obs,
        env_cfg.env.num_single_obs,
        env_cfg.env.num_actions,
        actor_hidden_dims=list(train_cfg.policy.actor_hidden_dims),
        critic_hidden_dims=list(train_cfg.policy.critic_hidden_dims),
        temporal_encoder=train_cfg.policy.temporal_encoder,
        adaptation_num_single_obs=train_cfg.policy.adaptation_num_single_obs,
        latent_dim=train_cfg.policy.latent_dim,
        critic_use_latent=train_cfg.policy.critic_use_latent,
        adaptation_labels=list(train_cfg.policy.adaptation_labels),
        adaptation_dims=list(train_cfg.policy.adaptation_dims),
        adaptation_weights=list(train_cfg.policy.adaptation_weights),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.inference_mode():
        observations = torch.zeros(2, 1360)
        latent = model.adaptation_encoder_module(observations)
        decoded = model.adaptation_decoder_module(latent)
        actor_input = torch.cat((observations[:, -85:], latent), dim=-1)
        actions = model.actor_body(actor_input)
        values = model.critic_body(torch.zeros(2, 473))
    _assert_equal(tuple(latent.shape), (2, 32), "encoder forward")
    _assert_equal(tuple(decoded.shape), (2, 27), "decoder forward")
    _assert_equal(tuple(actions.shape), (2, 19), "actor forward")
    _assert_equal(tuple(values.shape), (2, 1), "critic forward")

    filename_match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
    filename_iteration = int(filename_match.group(1)) if filename_match else None
    metadata_iteration = int(checkpoint.get("iter", 0))
    if filename_iteration and metadata_iteration == 0:
        print(
            "WARNING: legacy checkpoint metadata iter=0; continuation runner "
            f"must recover source iteration {filename_iteration} from the filename."
        )
    print(f"PASS strict checkpoint ABI: {path}")
    print(f"  sha256={_sha256(path)}")
    print(f"  encoder={tuple(latent.shape)} decoder={tuple(decoded.shape)} actor={tuple(actions.shape)}")
    print(f"  action_std=[{float(state_dict['std'].min()):.8f}, {float(state_dict['std'].max()):.8f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="logs/a2nero_pos_force/Aug21_10-46-25_dr_stage1/model_17800.pt",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    check_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
