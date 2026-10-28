from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from collections import deque
import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from legged_gym.utils.isaacgym_utils import euler_from_quat, sphere2cart, cart2sphere
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg
from legged_gym.envs.a2.a2nero_stage3_actuator import (
    A2NERO_URDF_HARD_LOWER,
    A2NERO_URDF_HARD_UPPER,
    A2NeroStage3Actuator,
)
from legged_gym.envs.a2.a2nero_stage3_randomization import A2_LEG_BODY_NAMES
from legged_gym.envs.a2.a2nero_stage3_sensors import (
    A2NeroIdentifiedSensorModel,
    A2NeroUnobservedWristFlexModel,
)

import matplotlib.pyplot as plt



INDEX_EE_POS_RADIUS_CMD = 3
INDEX_EE_POS_PITCH_CMD = 4
INDEX_EE_POS_YAW_CMD = 5
INDEX_EE_ROLL_CMD = 6
INDEX_EE_PITCH_CMD =7
INDEX_EE_YAW_CMD = 8
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

def quat_conjugate_xyzw(quat):
    quat_conj = quat.clone()
    quat_conj[:, :3] *= -1
    return quat_conj

def quat_mul_xyzw(q, r):
    x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return torch.stack((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dim=1)

class LeggedRobot_a2nero_pos_force(BaseTask):
    def __init__(self, cfg: A2NEROPosForceRoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """ 
        self.cfg = cfg
        self.stage3_enabled = bool(getattr(getattr(cfg, "stage3", None), "enabled", False))
        self.identified_stage = int(getattr(getattr(cfg, "stage3", None), "identified_stage", 0))
        self.stage3_nominal_override = False
        self.stage3_curriculum_progress = 0.0
        self.stage3_curriculum_frozen = False
        self.stage3_frozen_progress = None
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.draw_collision_bbox = True
        self.draw_world_origin = True
        self.draw_ee_goal_center = True
        self.debug_curve = False
        if self.debug_curve:
            self.FL_joint_index = 2
            self.FR_joint_index = 5
            self.RL_joint_index = 8
            self.RR_joint_index = 11
            plt.ion() 
            fig, ((self.ax1, self.ax4), (self.ax2, self.ax5), (self.ax3, self.ax6)) = plt.subplots(3, 2)
            self.t_data, self.FL_action, self.FL_state, self.FL_ref, self.FR_action, self.FR_state, self.FR_ref = [], [], [], [], [], [], []
            self.RL_action, self.RL_state, self.RL_ref, self.RR_action, self.RR_state, self.RR_ref = [], [], [], [], [], []
            self.FL_action_line, = self.ax1.plot([], [], 'b-', label="FL_action")       
            self.FL_state_line, = self.ax1.plot([], [], 'r-', label="FL_state")
            self.FL_ref_line, = self.ax1.plot([], [], 'y-', label="FL_ref")
            self.FR_action_line, = self.ax2.plot([], [], 'b-', label="FR_action")
            self.FR_state_line, = self.ax2.plot([], [], 'r-', label="FR_state")
            self.FR_ref_line, = self.ax2.plot([], [], 'y-', label="FR_ref")
            self.RL_action_line, = self.ax4.plot([], [], 'b-', label="RL_action")       
            self.RL_state_line, = self.ax4.plot([], [], 'r-', label="RL_state")
            self.RL_ref_line, = self.ax4.plot([], [], 'y-', label="RL_ref")
            self.RR_action_line, = self.ax5.plot([], [], 'b-', label="RR_action")
            self.RR_state_line, = self.ax5.plot([], [], 'r-', label="RR_state")
            self.RR_ref_line, = self.ax5.plot([], [], 'y-', label="RR_ref")
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_pred_obs = self.cfg.env.num_pred_obs
        self.num_single_obs = self.cfg.env.num_single_obs
        self.obs_pred = torch.zeros(self.num_envs, self.num_pred_obs, device=self.device, dtype=torch.float)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._initialize_identified_reward_curriculum()
        self.init_done = True
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.raw_actions.copy_(
            torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        )
        self._configure_stage3_actuator()
        gait_active = (self.gait_indices != 0.0) | self.get_walking_cmd_mask()
        actuator_tick = self.stage3_actuator.advance(
            self.raw_actions,
            self.dof_pos_wo_gripper,
            gait_active=gait_active,
        )
        # The action channel in the 85-D observation is the command actually
        # sent by the software adapter, never the raw actor target or delayed
        # plant target.
        self.actions.copy_(actuator_tick.previous_action)

        if not self.headless:
            self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            self.gym.simulate(self.sim)
            # CPU PhysX is asynchronous as well: fetch the step that was just
            # submitted before refreshing tensor-backed DOF state.  Fetching
            # before simulate leaves extraction/CPU playback one unresolved
            # step behind and can make every recorded joint position appear
            # frozen even while policy targets change.
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._enforce_gripper_symmetry()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.obs_pred = torch.clip(self.obs_pred, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.gripper_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 0:3]
        self.gripper_velocity = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 7:10]
        self._update_ee_state()
        self._update_ee_imu_state()
        
        self._post_physics_step_callback()


        # update ee goal
        self.update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            self._draw_ee_goal_traj()
            if self.draw_collision_bbox:
                self._draw_collision_bbox()
            self._draw_curve()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.base_euler_xyz[:,1])>1.0, torch.abs(self.base_euler_xyz[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._randomize_stage3_contact_props(env_ids)
        self._reset_stage3_sensor_biases(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.raw_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        if hasattr(self, "last_ee_vel"):
            self.last_ee_vel[env_ids] = self.ee_vel[env_ids]
        if hasattr(self, "last_ee_imu_vel"):
            self.last_ee_imu_vel[env_ids] = self.ee_imu_vel[env_ids]
        if hasattr(self, "last_ee_acc"):
            self.last_ee_acc[env_ids] = 0.
        if hasattr(self, "last_target_pos_base"):
            target_pos_base = self._get_target_pos_base()
            self.last_target_pos_base[env_ids] = target_pos_base[env_ids]
            self.last_target_vel_base[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        if hasattr(self, "identified_last_contacts"):
            self.identified_last_contacts[env_ids] = False
            self.identified_touchdown[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.goal_timer[env_ids] = 0.

        self._configure_stage3_actuator()
        self.stage3_actuator.reset(
            self.dof_pos_wo_gripper,
            env_ids=env_ids,
            true_dof_pos=self.dof_pos_wo_gripper,
            true_dof_vel=self.dof_vel_wo_gripper,
        )
        if self.identified_sensor is not None:
            base_seed = torch.zeros(self.num_envs, 6, device=self.device)
            ee_seed = torch.zeros_like(base_seed)
            self.identified_sensor.reset(
                env_ids,
                self.dof_pos_wo_gripper,
                self.dof_vel_wo_gripper,
                base_seed,
                ee_seed,
            )
            self.identified_prev_base_orientation[env_ids] = (
                self.get_body_orientation()[env_ids]
            )
            self.identified_prev_ee_orientation[env_ids] = (
                self.get_ee_orientation_rpy()[env_ids]
            )
        if self.unobserved_wrist_flex is not None:
            self.unobserved_wrist_flex.reset(env_ids)
        # The plant's internal delay queue is seeded from feedback above, but
        # the source checkpoint's policy-facing previous-action history starts
        # at zero after every reset. Do not leak the internal seed into the
        # observation ABI.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles

    def get_ee_goal_heading_quat(self):
        goal_heading = torch.atan2(self.curr_ee_goal_cart[:, 1], self.curr_ee_goal_cart[:, 0])
        goal_heading_quat = quat_from_euler_xyz(0 * goal_heading, 0 * goal_heading, goal_heading)
        return quat_mul_xyzw(self.base_yaw_quat, goal_heading_quat)

    def get_ee_orientation_rpy(self):
        ee_quat_in_goal_heading = quat_mul_xyzw(quat_conjugate_xyzw(self.get_ee_goal_heading_quat()), self.get_ee_goal_frame_quat())
        return get_euler_xyz_tensor(ee_quat_in_goal_heading)

    def get_ee_goal_frame_quat(self):
        return quat_mul_xyzw(self.ee_orn, self.ee_link_to_goal_frame_quat)

    def _update_ee_state(self):
        ee_body_state = self.rigid_state[:, self.ee_position_body_idx, :]
        ee_offset_world = quat_apply(ee_body_state[:, 3:7], self.ee_position_body_offset)
        ee_lin_vel = ee_body_state[:, 7:10] + torch.cross(ee_body_state[:, 10:13], ee_offset_world, dim=1)
        ee_orn_body_state = self.rigid_state[:, self.ee_orientation_body_idx, :]
        self.ee_pos = ee_body_state[:, 0:3] + ee_offset_world
        self.ee_orn = ee_orn_body_state[:, 3:7]
        ee_ang_vel = ee_orn_body_state[:, 10:13]
        self.ee_vel = torch.cat((ee_lin_vel, ee_ang_vel), dim=1)

    def _update_ee_imu_state(self):
        """Update motion at the configured point relative to the IMU center of mass."""
        imu_body_state = self.rigid_state[:, self.ee_imu_body_idx, :]
        imu_com_to_sensor_world = quat_apply(
            imu_body_state[:, 3:7], self.ee_imu_com_to_sensor_offset
        )
        imu_ang_vel = imu_body_state[:, 10:13]
        imu_lin_vel = imu_body_state[:, 7:10] + torch.cross(
            imu_ang_vel, imu_com_to_sensor_world, dim=1
        )
        self.ee_imu_vel = torch.cat((imu_lin_vel, imu_ang_vel), dim=1)

    def _get_arm_base_pos(self):
        return self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)

    def _get_target_pos_base(self, arm_base_pos=None):
        if arm_base_pos is None:
            arm_base_pos = self._get_arm_base_pos()
        return quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        
    def compute_observations(self):
        """ Computes observations
        """

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)


        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        diff = self.dof_pos[:, :12] - self.ref_dof_pos

        arm_base_pos = self._get_arm_base_pos()
        target_pos_base = self._get_target_pos_base(arm_base_pos)
        target_vel_base = (target_pos_base - self.last_target_pos_base) / self.dt
        target_acc_base = (target_vel_base - self.last_target_vel_base) / self.dt

        ee_pos_base = quat_rotate_inverse(self.base_quat, self.ee_pos - arm_base_pos)
        ee_vel_base = quat_rotate_inverse(self.base_quat, self.ee_vel[:, 0:3] - self.root_states[:, 7:10])
        tracking_error_base = target_pos_base - ee_pos_base

        # model39000 has no end-effector IMU channels, but its first five
        # observations still come from the base IMU and its proprioception is
        # transported asynchronously on hardware. Model those training-side
        # latencies without changing the 85-D observation order or scale.
        base_lin_acc_base = quat_rotate_inverse(
            self.base_quat,
            (self.root_states[:, 7:10] - self.last_root_vel[:, 0:3]) / self.dt,
        )
        base_imu = torch.cat((self.base_ang_vel, base_lin_acc_base), dim=-1)
        base_imu = base_imu + self.stage3_base_imu_bias
        student_dof_pos = self.dof_pos_wo_gripper
        student_dof_vel = self.dof_vel_wo_gripper
        student_base_ang_vel = self.base_ang_vel
        current_base_orientation = self.get_body_orientation()
        student_base_orientation = current_base_orientation
        if self.identified_sensor is not None and self._stage3_progress() > 0.10:
            # The EE argument is deliberately equal to base IMU: no synthetic
            # EE/relative-IMU information is exposed to this ablation policy.
            student_dof_pos, student_dof_vel, modeled_base_imu, _, _ = (
                self.identified_sensor.advance(
                    self.dof_pos_wo_gripper,
                    self.dof_vel_wo_gripper,
                    base_imu,
                    base_imu,
                )
            )
            student_base_ang_vel = modeled_base_imu[:, :3]
            orientation_age_blend = torch.clamp(
                self.identified_sensor.a2_age / self.dt, 0.0, 1.0
            )
            student_base_orientation = (
                current_base_orientation * (1.0 - orientation_age_blend)
                + self.identified_prev_base_orientation * orientation_age_blend
            )

        ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        # Spherical to cartesian coordinates in the arm base frame 
        radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
        pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
        yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        self.ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)
        ee_orn_rpy = self.get_ee_orientation_rpy()
        student_ee_orientation = ee_orn_rpy
        if self.identified_sensor is not None and self._stage3_progress() > 0.10:
            ee_orientation_age_blend = torch.clamp(
                self.identified_sensor.nero_age / self.dt, 0.0, 1.0
            )
            student_ee_orientation = (
                ee_orn_rpy * (1.0 - ee_orientation_age_blend)
                + self.identified_prev_ee_orientation
                * ee_orientation_age_blend
            )
        
        self.privileged_obs_buf = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    ee_orn_rpy, # 3
                                    diff, # 12
                                    self.mass_params_tensor, # 24
                                    self.friction_coeffs_tensor, #  1
                                    self.motor_strength[:, :19] - 1, # 19
                                    stance_mask, # 4
                                    contact_mask, # 4
                                    self.projected_gravity, # 3
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_dofs], # dim 19
                                    (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_dofs], # dim 19
                                    self.actions[:, :19], # dim 19
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    self.commands * self.commands_scale, # dim 9
                                    ),dim=-1)
        obs_pred = torch.cat((
                                    ee_pos_base, # 3
                                    ee_vel_base, # 3
                                    tracking_error_base, # 3
                                    torch.zeros(self.num_envs, self.cfg.env.num_pred_obs - 9, device=self.device), # future targets are filled from rollout storage
                                    ),dim=-1)
        
        obs_buf = torch.cat(( student_base_orientation,  # dim 2
                                    student_base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    (student_dof_pos - self.default_dof_pos_wo_gripper) * self.obs_scales.dof_pos, # dim 19
                                    student_dof_vel * self.obs_scales.dof_vel, # dim 19
                                    self.actions[:, :19], # dim 19
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    student_ee_orientation, # dim 3
                                    self.commands * self.commands_scale, # dim 9
                                    target_pos_base, # dim 3
                                    target_vel_base, # dim 3
                                    target_acc_base, # dim 3
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.obs_pred = obs_pred.clone()
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        self.last_target_pos_base[:] = target_pos_base
        self.last_target_vel_base[:] = target_vel_base
        self.last_ee_acc[:] = (self.ee_vel[:, 0:3] - self.last_ee_vel[:, 0:3]) / self.dt
        self.identified_prev_base_orientation.copy_(current_base_orientation)
        self.identified_prev_ee_orientation.copy_(ee_orn_rpy)
        self.last_ee_vel[:] = self.ee_vel
        self.last_ee_imu_vel[:] = self.ee_imu_vel

    def get_observations(self):
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain(self.cfg.terrain, )
        self._create_trimesh()
        # self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _draw_curve(self):
        if self.debug_curve:

            self.ax1.set_ylim(self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()+1.5)
            self.ax1.set_title(f"FL_joint_{self.FL_joint_index}")
            self.ax1.set_xlabel("time / s")
            self.ax1.legend(loc="upper right")
            self.ax2.set_ylim(self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item()+1.5)
            self.ax2.set_title(f"FL_joint_{self.FR_joint_index}")
            self.ax2.set_xlabel("time / s")
            self.ax2.legend(loc="upper right")

            self.ax3.set_ylim(self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()+1.5)
            self.ax3.set_title(f"front_ref_motion")
            self.ax3.set_xlabel("time / s")
            self.ax3.legend(loc="upper right")

            self.ax4.set_ylim(self.default_dof_pos_wo_gripper[0, self.RL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.RL_joint_index].item()+1.5)
            self.ax4.set_title(f"FL_joint_{self.RL_joint_index}")
            self.ax4.set_xlabel("time / s")
            self.ax4.legend(loc="upper right")
            self.ax5.set_ylim(self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item()+1.5)
            self.ax5.set_title(f"FL_joint_{self.RR_joint_index}")
            self.ax5.set_xlabel("time / s")
            self.ax5.legend(loc="upper right")

            self.ax6.set_ylim(-.5, 1.5)
            self.ax6.set_title(f"ront_ref_motion")
            self.ax6.set_xlabel("time / s")
            self.ax6.legend(loc="upper right")

            self.t_data.append(self.global_steps)
            self.FL_action.append(self.actions[0, self.FL_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item())
            self.FL_state.append(self.dof_pos[0, self.FL_joint_index].item())
            self.FL_ref.append(self.ref_dof_pos[0, self.FL_joint_index].item())
            self.FR_action.append(self.actions[0, self.FR_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item())
            self.FR_state.append(self.dof_pos[0, self.FR_joint_index].item())
            self.FR_ref.append(self.ref_dof_pos[0, self.FR_joint_index].item())

            # stand_mask = self._get_gait_phase()
            self.RL_action.append(self.actions[0, self.RL_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item())
            self.RL_state.append(self.dof_pos[0, self.RL_joint_index].item())
            self.RL_ref.append(self.ref_dof_pos[0, self.RL_joint_index].item())
            self.RR_action.append(self.actions[0, self.RR_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item())
            self.RR_state.append(self.dof_pos[0, self.RR_joint_index].item())
            self.RR_ref.append(self.ref_dof_pos[0, self.RR_joint_index].item())
            
            if len(self.t_data) > 100:
                self.t_data.pop(0)
                self.FL_action.pop(0)
                self.FL_state.pop(0)
                self.FL_ref.pop(0)
                self.FR_action.pop(0)
                self.FR_state.pop(0)
                self.FR_ref.pop(0)
                self.RL_action.pop(0)
                self.RL_state.pop(0)
                self.RL_ref.pop(0)
                self.RR_action.pop(0)
                self.RR_state.pop(0)
                self.RR_ref.pop(0)
            self.FL_action_line.set_data(self.t_data, self.FL_action)
            self.FL_state_line.set_data(self.t_data, self.FL_state)
            self.FL_ref_line.set_data(self.t_data, self.FL_ref)
            self.FR_action_line.set_data(self.t_data, self.FR_action)
            self.FR_state_line.set_data(self.t_data, self.FR_state)
            self.FR_ref_line.set_data(self.t_data, self.FR_ref)

            self.RL_action_line.set_data(self.t_data, self.RL_action)
            self.RL_state_line.set_data(self.t_data, self.RL_state)
            self.RL_ref_line.set_data(self.t_data, self.RL_ref)
            self.RR_action_line.set_data(self.t_data, self.RR_action)
            self.RR_state_line.set_data(self.t_data, self.RR_state)
            self.RR_ref_line.set_data(self.t_data, self.RR_ref)
            self.ax1.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax2.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax3.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax4.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax5.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax6.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            plt.pause(0.001)
            # print(self.commands)
    def _draw_collision_bbox(self):

        center = self.ee_goal_center_offset
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = self.base_yaw_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], 0), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

           
    def _draw_ee_goal_curr(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))
        upper_arm_pose = self.get_ee_goal_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.ee_pos

        if self.draw_world_origin:
            sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
            sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
            gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        ee_goal_frame_quat = self.get_ee_goal_frame_quat()

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            curr_pose = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), 
                                         r=gymapi.Quat(ee_goal_frame_quat[i, 0], ee_goal_frame_quat[i, 1], ee_goal_frame_quat[i, 2], ee_goal_frame_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], curr_pose)

            if self.draw_ee_goal_center:
                sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
                gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3)

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_ee_goal_traj(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze(0)
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += self.get_ee_goal_spherical_center()[:, :, None]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        fixed_stage3_props = bool(
            self.stage3_enabled
            and getattr(self.cfg.stage3, "fixed_rigid_props_at_creation", False)
        )
        if fixed_stage3_props:
            if env_id == 0:
                friction_range = self.cfg.stage3.friction_range
                num_buckets = 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1],
                    (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for shape in props:
                shape.friction = float(self.friction_coeffs[env_id].item())
            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        elif self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        else:
            self.friction_coeffs = torch.ones([self.num_envs, 1, 1])
        return props
    

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _stage3_progress(self):
        if not self.stage3_enabled or self.stage3_nominal_override:
            return 0.0
        progress = float(self.stage3_curriculum_progress)
        if self.stage3_curriculum_frozen and self.stage3_frozen_progress is not None:
            progress = min(progress, float(self.stage3_frozen_progress))
        return min(max(progress, 0.0), 1.0)

    def _stage3_randomization_factor(self):
        if self.identified_stage:
            curriculum = self.cfg.stage3.identified_profile.get("curriculum", {})
            ramp_start = float(
                curriculum.get("randomization_start_fraction", 0.10)
            )
            ramp_end = float(curriculum.get(
                "randomization_end_fraction",
                0.60 if self.identified_stage == 1 else 0.75,
            ))
        else:
            ramp_start = 0.10
            ramp_end = 0.40
        if not 0.0 <= ramp_start < ramp_end <= 1.0:
            raise ValueError(
                "invalid Stage3 randomization curriculum: "
                f"start={ramp_start}, end={ramp_end}"
            )
        return min(
            max(
                (self._stage3_progress() - ramp_start)
                / (ramp_end - ramp_start),
                0.0,
            ),
            1.0,
        )

    def _initialize_identified_reward_curriculum(self):
        self.stage3_reward_start_scales = {}
        self.stage3_reward_target_scales = {}
        if not self.identified_stage:
            return
        start_scales = self.cfg.stage3.identified_reward_start_scales
        target_scales = self.cfg.stage3.identified_reward_target_scales
        for name, target_scale in target_scales.items():
            if name not in self.reward_scales:
                raise RuntimeError(
                    f"identified reward '{name}' was removed before curriculum setup"
                )
            self.stage3_reward_start_scales[name] = (
                float(start_scales[name]) * self.dt
            )
            self.stage3_reward_target_scales[name] = (
                float(target_scale) * self.dt
            )
        self._configure_identified_reward_scales()

    def _stage3_reward_factor(self):
        if not self.identified_stage:
            return 1.0
        curriculum = self.cfg.stage3.identified_profile.get("curriculum", {})
        ramp_end = float(curriculum.get("reward_ramp_end_fraction", 0.10))
        if not 0.0 < ramp_end <= 1.0:
            raise ValueError(f"invalid reward_ramp_end_fraction={ramp_end}")
        return min(max(self._stage3_progress() / ramp_end, 0.0), 1.0)

    def _configure_identified_reward_scales(self):
        if not getattr(self, "stage3_reward_target_scales", None):
            return
        factor = self._stage3_reward_factor()
        for name, target_scale in self.stage3_reward_target_scales.items():
            start_scale = self.stage3_reward_start_scales[name]
            self.reward_scales[name] = self._lerp(
                start_scale, target_scale, factor
            )
        self.cfg.rewards.max_contact_force = self._lerp(
            float(self.cfg.stage3.source_max_contact_force),
            float(self.cfg.stage3.final_max_contact_force),
            factor,
        )

    def _stage3_adapter_factor(self):
        return min(max((self._stage3_progress() - 0.40) / 0.40, 0.0), 1.0)

    @staticmethod
    def _lerp(low, high, factor):
        return low + (high - low) * factor

    def _configure_stage3_actuator(self):
        progress = self._stage3_progress()
        random_factor = self._stage3_randomization_factor()
        adapter_factor = self._stage3_adapter_factor()
        actuator = self.stage3_actuator
        if self.identified_stage:
            actuator.enabled = False
            actuator.configure_identified(self.stage3_enabled and progress > 0.10)
            actuator.identified_factor = random_factor
            if self.identified_sensor is not None:
                self.identified_sensor.factor = random_factor
            flex_model = getattr(self, "unobserved_wrist_flex", None)
            if flex_model is not None:
                flex_model.factor = random_factor
            return
        actuator.enabled = bool(self.stage3_enabled and progress > 0.10)

        if progress < 0.40:
            actuator.nero_max_step = self.cfg.stage3.nero_feedback_envelope_start
        else:
            actuator.nero_max_step = self._lerp(
                self.cfg.stage3.nero_max_step_start,
                self.cfg.stage3.nero_max_step_final,
                adapter_factor,
            )
        actuator.a2_stationary_feedback_limit = self._lerp(
            self.cfg.stage3.a2_stationary_feedback_envelope_start,
            self.cfg.stage3.a2_stationary_feedback_envelope_final,
            adapter_factor,
        )
        actuator.a2_active_feedback_limit = self._lerp(
            self.cfg.stage3.a2_active_feedback_envelope_start,
            self.cfg.stage3.a2_active_feedback_envelope_final,
            adapter_factor,
        )
        actuator.nero_feedback_limit = self._lerp(
            self.cfg.stage3.nero_feedback_envelope_start,
            self.cfg.stage3.nero_feedback_envelope_final,
            adapter_factor,
        )
        a2_delayed = random_factor * self.cfg.stage3.a2_delay_probabilities[1]
        nero_delayed = random_factor * self.cfg.stage3.nero_delay_probabilities[1]
        actuator.set_delay_probabilities(
            a2_probabilities=[1.0 - a2_delayed, a2_delayed],
            nero_probabilities=[1.0 - nero_delayed, nero_delayed],
        )

    def set_stage3_curriculum_progress(self, local_iteration, total_iterations):
        total = max(int(total_iterations), 1)
        requested = min(max(float(local_iteration) / float(total), 0.0), 1.0)
        if self.stage3_curriculum_frozen and self.stage3_frozen_progress is not None:
            requested = min(requested, float(self.stage3_frozen_progress))
        self.stage3_curriculum_progress = requested
        self._configure_stage3_actuator()
        self._configure_identified_reward_scales()

    def freeze_stage3_curriculum(self):
        if not self.stage3_curriculum_frozen:
            self.stage3_curriculum_frozen = True
            self.stage3_frozen_progress = float(self.stage3_curriculum_progress)

    def set_stage3_nominal_override(self, enabled):
        self.stage3_nominal_override = bool(enabled)
        self._configure_stage3_actuator()
        self._configure_identified_reward_scales()

    def get_stage3_curriculum_state(self):
        return {
            "progress": float(self.stage3_curriculum_progress),
            "reward_factor": float(self._stage3_reward_factor()),
            "randomization_factor": float(self._stage3_randomization_factor()),
            "frozen": bool(self.stage3_curriculum_frozen),
            "frozen_progress": self.stage3_frozen_progress,
        }

    def load_stage3_curriculum_state(self, state):
        if not state:
            return
        self.stage3_curriculum_progress = float(state.get("progress", 0.0))
        self.stage3_curriculum_frozen = bool(state.get("frozen", False))
        self.stage3_frozen_progress = state.get("frozen_progress")
        self._configure_stage3_actuator()
        self._configure_identified_reward_scales()

    def get_stage3_config(self):
        config = {
            "enabled": bool(self.stage3_enabled),
            "identified_stage": int(self.identified_stage),
            "rigid_body_lifecycle": (
                "fixed_at_actor_creation"
                if getattr(self.cfg.stage3, "fixed_rigid_props_at_creation", False)
                else "legacy"
            ),
            "num_single_obs": int(self.cfg.env.num_single_obs),
            "frame_stack": int(self.cfg.env.frame_stack),
            "num_pred_obs": int(self.cfg.env.num_pred_obs),
            "num_actions": int(self.cfg.env.num_actions),
            "action_scale": float(self.cfg.control.action_scale),
            "noise_level": float(self.cfg.noise.noise_level),
            "delay": {
                "a2": [list(self.cfg.stage3.a2_delay_values), list(self.cfg.stage3.a2_delay_probabilities)],
                "nero": [list(self.cfg.stage3.nero_delay_values), list(self.cfg.stage3.nero_delay_probabilities)],
            },
        }
        if self.identified_stage:
            profile = self.cfg.stage3.identified_profile
            config["identified_plant"] = {
                "actuator": profile["actuator"],
                "sensor": profile["sensor"],
                "contact": profile["contact"],
                "previous_action_source": "actual_sent_target",
                "deployment_filter_required": False,
            }
        return config

    def _sample_scaled_range(self, value_range, factor, shape):
        low = 1.0 + (float(value_range[0]) - 1.0) * factor
        high = 1.0 + (float(value_range[1]) - 1.0) * factor
        return torch_rand_float(low, high, shape, device=self.device)

    def _randomize_dof_props(self, env_ids):
        if not self.stage3_enabled or len(env_ids) == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        factor = self._stage3_randomization_factor()
        if factor <= 0.0:
            self.motor_strength[ids] = 1.0
            self.stage3_kp_scale[ids] = 1.0
            self.stage3_kd_scale[ids] = 1.0
            return
        count = len(ids)
        if self.identified_stage:
            for indices, value_range in (
                ([0, 3, 6, 9], self.cfg.stage3.a2_hip_torque_scale_range),
                ([1, 4, 7, 10], self.cfg.stage3.a2_thigh_torque_scale_range),
                ([2, 5, 8, 11], self.cfg.stage3.a2_calf_torque_scale_range),
            ):
                joint_ids = torch.tensor(indices, device=self.device)
                self.motor_strength[ids[:, None], joint_ids] = self._sample_scaled_range(
                    value_range, factor, (count, len(indices))
                )
        else:
            self.motor_strength[ids, :12] = self._sample_scaled_range(
                self.cfg.stage3.a2_torque_scale_range, factor, (count, 12)
            )
        self.motor_strength[ids, 12:] = self._sample_scaled_range(
            self.cfg.stage3.nero_torque_scale_range, factor, (count, 7)
        )
        self.stage3_kp_scale[ids, :12] = self._sample_scaled_range(
            self.cfg.stage3.a2_kp_scale_range, factor, (count, 12)
        )
        self.stage3_kd_scale[ids, :12] = self._sample_scaled_range(
            self.cfg.stage3.a2_kd_scale_range, factor, (count, 12)
        )
        self.stage3_kp_scale[ids, 12:] = self._sample_scaled_range(
            self.cfg.stage3.nero_kp_scale_range, factor, (count, 7)
        )
        self.stage3_kd_scale[ids, 12:] = self._sample_scaled_range(
            self.cfg.stage3.nero_kd_scale_range, factor, (count, 7)
        )

    def _reset_stage3_sensor_biases(self, env_ids):
        if len(env_ids) == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        factor = self._stage3_randomization_factor()
        if not self.stage3_enabled or factor <= 0.0:
            self.stage3_base_imu_bias[ids] = 0.0
            self.stage3_ee_imu_bias[ids] = 0.0
            return
        count = len(ids)
        bg = self.cfg.stage3.base_gyro_bias_range
        ba = self.cfg.stage3.base_acc_bias_range
        eg = self.cfg.stage3.ee_gyro_bias_range
        ea = self.cfg.stage3.ee_acc_bias_range
        self.stage3_base_imu_bias[ids, :3] = torch_rand_float(bg[0] * factor, bg[1] * factor, (count, 3), device=self.device)
        self.stage3_base_imu_bias[ids, 3:] = torch_rand_float(ba[0] * factor, ba[1] * factor, (count, 3), device=self.device)
        self.stage3_ee_imu_bias[ids, :3] = torch_rand_float(eg[0] * factor, eg[1] * factor, (count, 3), device=self.device)
        self.stage3_ee_imu_bias[ids, 3:] = torch_rand_float(ea[0] * factor, ea[1] * factor, (count, 3), device=self.device)
                
    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1], size=(1, ))
            props[self.robot_base_idx].mass += float(rand_mass[0])
        else:
            rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_leg_mass:
            rng_link = self.cfg.domain_rand.leg_mass_scale_range
            rand_leg_masses = []
            for idx in self.stage3_leg_body_indices:
                rand_link_mass = float(np.random.uniform(rng_link[0], rng_link[1])) * props[idx].mass
                props[idx].mass += rand_link_mass
                rand_leg_masses.append(rand_link_mass)
            # Preserve the legacy 19-slot privileged-observation ABI without
            # assigning arm bodies the semantics of leg links.
            rand_leg_masses = np.asarray(rand_leg_masses + [0.0] * (19 - len(rand_leg_masses)))
        else:
            rand_leg_masses = np.zeros(19)

        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.stage3_payload_body_idx].mass += float(gripper_rand_mass[0])
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[self.robot_base_idx].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        
        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass, rand_leg_masses])
        return props, mass_params

    def _process_stage3_fixed_rigid_body_props(self, props):
        """Sample source-compatible mass properties once at actor creation."""
        cfg = self.cfg.stage3
        base_mass_delta = float(np.random.uniform(*cfg.added_base_mass_range))
        base_com_delta = np.asarray([
            np.random.uniform(*cfg.base_com_range_x),
            np.random.uniform(*cfg.base_com_range_y),
            np.random.uniform(*cfg.base_com_range_z),
        ])
        payload_delta = float(np.random.uniform(*cfg.gripper_payload_range))
        leg_mass_deltas = np.zeros(len(self.stage3_leg_body_indices))

        base = props[self.robot_base_idx]
        base.mass += base_mass_delta
        base.com += gymapi.Vec3(*base_com_delta)
        for slot, body_idx in enumerate(self.stage3_leg_body_indices):
            relative_delta = float(np.random.uniform(*cfg.leg_mass_scale_range))
            mass_delta = float(props[body_idx].mass) * relative_delta
            props[body_idx].mass += mass_delta
            leg_mass_deltas[slot] = mass_delta
        props[self.stage3_payload_body_idx].mass += payload_delta

        padded_leg_deltas = np.concatenate((leg_mass_deltas, np.zeros(3)))
        mass_params = np.concatenate((
            np.asarray([base_mass_delta]),
            base_com_delta,
            np.asarray([payload_delta]),
            padded_leg_deltas,
        ))
        return props, mass_params

    def _randomize_stage3_contact_props(self, env_ids):
        """Ramp contact DR on reset without rewriting mass or inertia."""
        if not self.stage3_enabled or not self.identified_stage or len(env_ids) == 0:
            return
        factor = self._stage3_randomization_factor()
        cfg = self.cfg.stage3
        contact_cfg = cfg.identified_profile["contact"]
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            if factor <= 0.0 and not self.stage3_contact_props_dirty[env_id]:
                continue
            shapes = self.gym.get_actor_rigid_shape_properties(
                self.envs[env_id], self.actor_handles[env_id]
            )
            nominal = self.stage3_nominal_body_props[env_id]
            restitution = float(np.random.uniform(
                cfg.restitution_range[0] * factor,
                cfg.restitution_range[1] * factor,
            ))
            stiffness_scale = float(np.random.uniform(
                self._lerp(1.0, contact_cfg["foot_stiffness_scale"][0], factor),
                self._lerp(1.0, contact_cfg["foot_stiffness_scale"][1], factor),
            ))
            damping_scale = float(np.random.uniform(
                self._lerp(1.0, contact_cfg["foot_damping_scale"][0], factor),
                self._lerp(1.0, contact_cfg["foot_damping_scale"][1], factor),
            ))
            for shape_index in self.stage3_foot_shape_indices:
                shape = shapes[shape_index]
                source_compliance = nominal["shape_compliance"][shape_index]
                source_restitution = nominal["shape_restitution"][shape_index]
                if factor <= 0.0:
                    shape.compliance = source_compliance
                    shape.restitution = source_restitution
                else:
                    target_compliance = max(source_compliance, 1.0e-6) / stiffness_scale
                    shape.compliance = self._lerp(
                        source_compliance, target_compliance, factor
                    )
                    shape.restitution = min(
                        max(restitution / damping_scale, 0.0), 1.0
                    )
            self.gym.set_actor_rigid_shape_properties(
                self.envs[env_id], self.actor_handles[env_id], shapes
            )
            self.stage3_contact_props_dirty[env_id] = factor > 0.0
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # Stage-3 actuator/IMU parameters are episode-persistent and therefore
        # are sampled only from reset_idx, not when commands are resampled.
        if not self.stage3_enabled:
            self._randomize_dof_props(env_ids)
        self._step_contact_targets()
        if self.unobserved_wrist_flex is not None:
            ee_acc_world = (
                self.ee_vel[:, 0:3] - self.last_ee_vel[:, 0:3]
            ) / self.dt
            base_acc_world = (
                self.root_states[:, 7:10] - self.last_root_vel[:, 0:3]
            ) / self.dt
            self.unobserved_wrist_flex.advance(
                ee_acc_world - base_acc_world
            )
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 5.0
        self.identified_touchdown = contact_now & ~self.identified_last_contacts
        self.identified_last_contacts.copy_(contact_now)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _step_contact_targets(self):
        cycle_time = self.cfg.rewards.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt / cycle_time, 1.0)
        self.gait_indices[standing_mask] = 0

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.env.teleop_mode:
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        zero_cmd_mask = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) < self.cfg.commands.zero_vel_cmd_prob
        self.commands[env_ids, :3] *= ~zero_cmd_mask.unsqueeze(1)

        # set small commands to zero
        non_stop_sign = (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)
        self.commands[env_ids, :3] *= non_stop_sign.unsqueeze(1)

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    def set_gripper_target(self, opening):
        """Set the single logical gripper opening used by both finger DOFs."""
        target = torch.as_tensor(opening, dtype=torch.float, device=self.device)
        if target.numel() == 1:
            target = target.expand(self.num_envs)
        target = target.reshape(self.num_envs, 1)
        lower, upper = self.cfg.control.gripper_opening_range
        self.gripper_target_pos[:] = torch.clamp(target, lower, upper)

    def _enforce_gripper_symmetry(self):
        """Project the two sliders onto the real gripper's one-DOF linkage."""
        q1 = self.dof_pos[:, self.gripper_joint1_idx]
        q2 = self.dof_pos[:, self.gripper_joint2_idx]
        dq1 = self.dof_vel[:, self.gripper_joint1_idx]
        dq2 = self.dof_vel[:, self.gripper_joint2_idx]
        lower, upper = self.cfg.control.gripper_opening_range
        opening = torch.clamp(0.5 * (q1 - q2), lower, upper)
        opening_vel = 0.5 * (dq1 - dq2)
        self.dof_pos[:, self.gripper_joint1_idx] = opening
        self.dof_pos[:, self.gripper_joint2_idx] = -opening
        self.dof_vel[:, self.gripper_joint1_idx] = opening_vel
        self.dof_vel[:, self.gripper_joint2_idx] = -opening_vel
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # The policy target has already advanced once at the policy boundary.
        # Physics substeps only evaluate the fixed delayed plant target.  Motor
        # strength scales the complete PD torque, never the position offset.
        default_torques = self.stage3_actuator.compute_torques(
            self.dof_pos_wo_gripper,
            self.dof_vel_wo_gripper,
            kp=self.p_gains * self.stage3_kp_scale,
            kd=self.d_gains * self.stage3_kd_scale,
            motor_strength=self.motor_strength,
            torque_limits=self.torque_limits[:self.num_actions],
        )
        num_gripper_dofs = self.cfg.env.num_gripper_dofs
        gripper_targets = torch.cat(
            (self.gripper_target_pos, -self.gripper_target_pos),
            dim=-1,
        )
        gripper_pos = self.dof_pos[:, -num_gripper_dofs:]
        gripper_vel = self.dof_vel[:, -num_gripper_dofs:]
        gripper_torques = (
            self.cfg.control.gripper_stiffness
            * (gripper_targets - gripper_pos)
            - self.cfg.control.gripper_damping * gripper_vel
        )

        torques = torch.cat([default_torques, gripper_torques], dim=-1)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_orn_r = torch_rand_float(self.goal_ee_ranges["orn_r"][0], self.goal_ee_ranges["orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_orn_p = torch_rand_float(self.goal_ee_ranges["orn_p"][0], self.goal_ee_ranges["orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_orn_y = torch_rand_float(self.goal_ee_ranges["orn_y"][0], self.goal_ee_ranges["orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_rpy[env_ids, :] = torch.cat([ee_goal_orn_r, ee_goal_orn_p, ee_goal_orn_y], dim=-1)

    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.cfg.env.teleop_mode and is_init:
            self.curr_ee_goal_sphere[:] = self.init_start_ee_sphere[:]
            self.curr_ee_goal_orn_rpy[:] = self.init_start_ee_orn_rpy[:]
            self.ee_start_sphere[:] = self.curr_ee_goal_sphere[:]
            self.ee_goal_sphere[:] = self.curr_ee_goal_sphere[:]
            self.ee_start_orn_rpy[:] = self.curr_ee_goal_orn_rpy[:]
            self.ee_goal_orn_rpy[:] = self.curr_ee_goal_orn_rpy[:]
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)
            self.commands[:, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)] = self.curr_ee_goal_orn_rpy.view(self.num_envs,3)
            return
        elif self.cfg.env.teleop_mode:
            return

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()

            ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
                    # Spherical to cartesian coordinates in the arm base frame 
            radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
            pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
            yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
            ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)

            if is_init:
                if self.global_steps < 0 * 24 and not self.play:
                    self.ee_start_orn_rpy[env_ids] = self.init_start_ee_orn_rpy[:]
                    self.ee_goal_orn_rpy[env_ids] = self.init_start_ee_orn_rpy[:]
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
                    self.ee_start_orn_rpy[env_ids] = self.init_start_ee_orn_rpy[:]
                    self.ee_goal_orn_rpy[env_ids] = self.init_end_ee_orn_rpy[:]
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
            else:
                if self.global_steps < 0 * 24 and not self.play:
                    self.ee_start_orn_rpy[env_ids] = self.init_start_ee_orn_rpy[:]
                    self.ee_goal_orn_rpy[env_ids] = self.init_start_ee_orn_rpy[:]
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
                    self.ee_start_orn_rpy[env_ids] = self.ee_goal_orn_rpy[env_ids].clone()
                    self._resample_ee_goal_orn_once(env_ids)
                    self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
                    for i in range(10):
                        self._resample_ee_goal_sphere_once(env_ids)
                        collision_mask = self.collision_check(env_ids)
                        env_ids = env_ids[collision_mask]
                        if len(env_ids) == 0:
                            break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
            self.curr_ee_goal_orn_rpy[:] = torch.lerp(self.ee_start_orn_rpy, self.ee_goal_orn_rpy, t[:, None])
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)
            self.commands[:, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)] = self.curr_ee_goal_orn_rpy.view(self.num_envs,3)

        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        # TODO: add twisting motion by fixing yaw at traj start
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + ee_goal_cart_yaw_global

        ee_goal_orn_local_quat = quat_from_euler_xyz(self.curr_ee_goal_orn_rpy[:, 0], self.curr_ee_goal_orn_rpy[:, 1], self.curr_ee_goal_orn_rpy[:, 2])
        self.ee_goal_orn_quat = quat_mul_xyzw(self.get_ee_goal_heading_quat(), ee_goal_orn_local_quat)
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        
        if len(resample_id) > 0 and self.stop_update_goal:
            # set these env commands as 0
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 1] = 0
            self.commands[resample_id, 2] = 0
            
            
        self._resample_ee_goal(resample_id)
    
    def get_ee_goal_spherical_center(self):
        center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids, :12] = self.default_dof_pos[:, :12]* torch_rand_float(0.95, 1.05, (len(env_ids), 12), device=self.device)
        self.dof_pos[env_ids, 12:19] += torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions-12), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # base orientation
        rand_yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :]  

        # base velocities
        init_vel_range = self.cfg.init_state.init_vel_perturb_range
        self.root_states[env_ids, 7:13] = torch_rand_float(-init_vel_range, init_vel_range, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            self.root_states[:, 7:9] * 2.5,
            self.root_states[:, 7:9]
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = noise_scales.gravity * noise_level
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:5+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[5+self.num_actions:5+self.num_actions*2] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[5+self.num_actions*2:5+self.num_actions*3] = 0. # previous actions
        noise_vec[5+self.num_actions*3:] = 0. # phase, commands, targets

        return noise_vec

    #----------------------------------------
    def _identified_joint_delay_ranges(self):
        if self.identified_stage != 2:
            return None
        delay_ms = self.cfg.stage3.identified_profile["actuator"]["delay_ms"]

        def ticks(name):
            bounds = delay_ms[name]
            tick_ms = self.dt * 1000.0
            return [
                int(np.floor(bounds[0] / tick_ms)),
                int(np.ceil(bounds[1] / tick_ms)),
            ]

        ranges = [None] * 19
        for indices, name in (
            ([0, 3, 6, 9], "a2_hip"),
            ([1, 4, 7, 10], "a2_thigh"),
            ([2, 5, 8, 11], "a2_calf"),
            ([12, 13, 17, 18], "nero_outer"),
            ([14, 15, 16], "nero_inner"),
        ):
            for index in indices:
                ranges[index] = ticks(name)
        return ranges

    def _identified_tail_ticks(self):
        if self.identified_stage != 2:
            return (0, 0)
        bounds = self.cfg.stage3.identified_profile["actuator"]["tail_ms"]
        tick_ms = self.dt * 1000.0
        return (
            int(np.floor(bounds[0] / tick_ms)),
            int(np.ceil(bounds[1] / tick_ms)),
        )

    def _identified_jitter_ticks(self):
        if self.identified_stage != 2:
            return (0, 0)
        actuator_cfg = self.cfg.stage3.identified_profile["actuator"]
        bounds = actuator_cfg.get(
            "per_tick_jitter_ms", actuator_cfg.get("jitter_ms", [0.0, 0.0])
        )
        tick_ms = self.dt * 1000.0
        return (
            int(np.floor(bounds[0] / tick_ms)),
            int(np.ceil(bounds[1] / tick_ms)),
        )

    def _identified_tau(self, key):
        if self.identified_stage != 2:
            return (0.0, 0.0)
        bounds = self.cfg.stage3.identified_profile["actuator"][key]
        return (float(bounds[0]) / 1000.0, float(bounds[1]) / 1000.0)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_dofs]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_dofs]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.arm_base_offset = torch.tensor([0.16, 0., 0.142], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.gripper_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 0:3]
        self.gripper_velocity = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 7:10]
        # ee info
        self._update_ee_state()
        self._update_ee_imu_state()
        self.ee_link_to_goal_frame_quat = torch.tensor(self.cfg.asset.ee_orientation_body_to_goal_frame_quat, device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        # target_ee info
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_start_orn_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_orn_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_pos_sphe_arm = torch.zeros(self.num_envs, 3, device=self.device)
        ee_goal_orn_local_quat = quat_from_euler_xyz(self.curr_ee_goal_orn_rpy[:, 0], self.curr_ee_goal_orn_rpy[:, 1], self.curr_ee_goal_orn_rpy[:, 2])
        self.ee_goal_orn_quat = quat_mul_xyzw(self.get_ee_goal_heading_quat(), ee_goal_orn_local_quat)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device, dtype=torch.float).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device, dtype=torch.float).unsqueeze(0)
        self.init_start_ee_orn_rpy = torch.tensor(self.cfg.goal_ee.ranges.init_orn_start, device=self.device, dtype=torch.float).unsqueeze(0)
        self.init_end_ee_orn_rpy = torch.tensor(self.cfg.goal_ee.ranges.init_orn_end, device=self.device, dtype=torch.float).unsqueeze(0)
        
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        target_pos_base = self._get_target_pos_base()
        self.last_target_pos_base = target_pos_base.clone()
        self.last_target_vel_base = torch.zeros_like(target_pos_base)
        self.last_ee_vel = self.ee_vel.clone()
        self.last_ee_imu_vel = self.ee_imu_vel.clone()
        self.last_ee_acc = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.raw_actions = torch.zeros_like(self.actions)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = self.root_states[:, 7:13].clone()
        self.last_torques = torch.zeros_like(self.torques)
        self.stage3_kp_scale = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.stage3_kd_scale = torch.ones_like(self.stage3_kp_scale)
        self.stage3_base_imu_bias = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
        self.stage3_ee_imu_bias = torch.zeros_like(self.stage3_base_imu_bias)
        self.identified_prev_base_orientation = self.get_body_orientation().clone()
        self.identified_prev_ee_orientation = self.get_ee_orientation_rpy().clone()
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.identified_last_contacts = torch.zeros_like(self.last_contacts)
        self.identified_touchdown = torch.zeros_like(self.last_contacts)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.ee_sphe_radius_cmd, 
                                            self.obs_scales.ee_sphe_pitch_cmd,
                                            self.obs_scales.ee_sphe_yaw_cmd,
                                            self.obs_scales.end_effector_roll_cmd, 
                                            self.obs_scales.end_effector_pitch_cmd,
                                            self.obs_scales.end_effector_yaw_cmd,], device=self.device, requires_grad=False,)
        
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_dofs, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:, :-self.cfg.env.num_gripper_dofs]
        self.stage3_actuator = A2NeroStage3Actuator(
            self.num_envs,
            self.default_dof_pos_wo_gripper.squeeze(0),
            A2NERO_URDF_HARD_LOWER,
            A2NERO_URDF_HARD_UPPER,
            device=self.device,
            enabled=False,
            action_scale=self.cfg.control.action_scale,
            raw_action_clip=self.cfg.normalization.clip_actions,
            a2_delay_options=self.cfg.stage3.a2_delay_values,
            a2_delay_probabilities=self.cfg.stage3.a2_delay_probabilities,
            nero_delay_options=self.cfg.stage3.nero_delay_values,
            nero_delay_probabilities=self.cfg.stage3.nero_delay_probabilities,
            identified_enabled=False,
            policy_dt=self.dt,
            joint_delay_ranges_ticks=self._identified_joint_delay_ranges(),
            per_tick_jitter_ticks=self._identified_jitter_ticks(),
            tail_probability=(
                float(self.cfg.stage3.identified_profile["actuator"].get("tail_probability", 0.0))
                if self.identified_stage
                else 0.0
            ),
            tail_delay_ticks=self._identified_tail_ticks(),
            a2_tau_range_s=self._identified_tau("a2_tau_ms"),
            nero_tau_range_s=self._identified_tau("nero_tau_ms"),
        )
        self.identified_sensor = None
        self.unobserved_wrist_flex = None
        if self.identified_stage:
            sensor_cfg = dict(self.cfg.stage3.identified_profile["sensor"])
            self.identified_sensor = A2NeroIdentifiedSensorModel(
                self.num_envs,
                self.device,
                self.dt,
                stage=3 + self.identified_stage,
                config={
                    **sensor_cfg,
                    "sensor": sensor_cfg,
                    "wrist_flex_enabled": False,
                    # model39000 has no EE IMU; keep this physically
                    # unobserved path disabled instead of fabricating input.
                    "wrist_flex": {},
                },
            )
        if self.identified_stage == 2:
            self.unobserved_wrist_flex = A2NeroUnobservedWristFlexModel(
                self.num_envs,
                self.device,
                self.dt,
                self.cfg.stage3.identified_profile["unobserved_wrist_flex"],
            )
        self.nero_joint_indices = [self.dof_names.index(f"nero_joint{i}") for i in range(1, 8)]
        self.nero_joint1_idx = self.nero_joint_indices[0]
        self.nero_joint2_idx = self.nero_joint_indices[1]
        self.nero_joint3_idx = self.nero_joint_indices[2]
        self.nero_joint4_idx = self.nero_joint_indices[3]
        self.nero_joint5_idx = self.nero_joint_indices[4]
        self.nero_joint6_idx = self.nero_joint_indices[5]
        self.nero_joint7_idx = self.nero_joint_indices[6]
        self.gripper_joint1_idx = self.gripper_dof_indices[0]
        self.gripper_joint2_idx = self.gripper_dof_indices[1]
        self.gripper_target_pos = self.default_dof_pos[
            :, self.gripper_joint1_idx:self.gripper_joint1_idx + 1
        ].repeat(self.num_envs, 1).clone()
        
        # gait
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.foot_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]

        self.global_steps = 0

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        if self.identified_stage and not hasattr(rigid_shape_props_asset[0], "compliance"):
            raise RuntimeError(
                "This Isaac Gym build cannot apply the identified foot compliance model"
            )

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        configured_gripper_dof_names = list(self.cfg.asset.gripper_dof_names)
        if len(configured_gripper_dof_names) != self.cfg.env.num_gripper_dofs:
            raise RuntimeError(
                "env.num_gripper_dofs does not match asset.gripper_dof_names: "
                f"{self.cfg.env.num_gripper_dofs} != {len(configured_gripper_dof_names)}"
            )
        missing_gripper_dofs = [name for name in configured_gripper_dof_names if name not in self.dof_names]
        if missing_gripper_dofs:
            raise RuntimeError(
                f"Configured gripper DOFs {missing_gripper_dofs} were not found in the asset. "
                f"Available DOFs: {self.dof_names}"
            )
        self.gripper_dof_indices = [self.dof_names.index(name) for name in configured_gripper_dof_names]
        expected_gripper_indices = list(range(len(self.dof_names) - len(configured_gripper_dof_names), len(self.dof_names)))
        if self.gripper_dof_indices != expected_gripper_indices:
            raise RuntimeError(
                "Gripper DOFs must be the final asset DOFs because observations and actions use contiguous slices. "
                f"Configured indices: {self.gripper_dof_indices}; expected: {expected_gripper_indices}"
            )
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_dofs]
        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]
        ee_position_body_name = getattr(self.cfg.asset, "ee_position_body_name", self.cfg.asset.gripper_name)
        ee_position_body_offset = getattr(self.cfg.asset, "ee_position_body_offset", [0.0, 0.0, 0.0])
        ee_orientation_body_name = getattr(self.cfg.asset, "ee_orientation_body_name", ee_position_body_name)
        self.ee_position_body_idx = self.body_names_to_idx[ee_position_body_name]
        self.ee_orientation_body_idx = self.body_names_to_idx[ee_orientation_body_name]
        self.ee_position_body_offset = torch.tensor(
            ee_position_body_offset,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        ee_imu_body_name = self.cfg.asset.ee_imu_body_name
        if ee_imu_body_name not in self.body_names_to_idx:
            raise RuntimeError(
                f"Configured end-effector IMU body '{ee_imu_body_name}' was not found. "
                "Ensure its fixed joint has dont_collapse=\"true\" in the URDF. "
                f"Available rigid bodies: {body_names}"
            )
        self.ee_imu_body_idx = self.body_names_to_idx[ee_imu_body_name]
        self.ee_imu_com_to_sensor_offset = torch.tensor(
            self.cfg.asset.ee_imu_com_to_sensor_offset,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        self.robot_base_idx = [index for index, body_name in enumerate(body_names) if body_name == "base_link"][0]
        missing_leg_bodies = [name for name in A2_LEG_BODY_NAMES if name not in self.body_names_to_idx]
        if missing_leg_bodies:
            raise RuntimeError(
                f"Missing explicit A2 leg rigid bodies {missing_leg_bodies}; "
                f"imported bodies are {body_names}"
            )
        self.stage3_leg_body_indices = [self.body_names_to_idx[name] for name in A2_LEG_BODY_NAMES]
        payload_body_name = "nero_link7"
        if payload_body_name not in self.body_names_to_idx:
            raise RuntimeError(f"Stage3 gripper payload body '{payload_body_name}' is missing")
        self.stage3_payload_body_idx = self.body_names_to_idx[payload_body_name]
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.stage3_foot_shape_indices = []
        if self.identified_stage:
            if not hasattr(self.gym, "get_asset_rigid_body_shape_indices"):
                raise RuntimeError(
                    "This Isaac Gym build cannot resolve per-foot collision shapes"
                )
            body_shape_ranges = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
            for foot_name in feet_names:
                shape_range = body_shape_ranges[self.body_names_to_idx[foot_name]]
                self.stage3_foot_shape_indices.extend(
                    range(int(shape_range.start), int(shape_range.start + shape_range.count))
                )
            if not self.stage3_foot_shape_indices:
                raise RuntimeError(
                    "Identified profile could not resolve any foot collision shapes"
                )
        thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.stage3_nominal_body_props = []
        self.stage3_contact_props_dirty = [False] * self.num_envs

        self.mass_params_tensor = torch.zeros(self.num_envs, 24, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_envs):

            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            actor_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_handle, actor_handle
            )
            self.stage3_nominal_body_props.append({
                "masses": [float(prop.mass) for prop in body_props],
                "coms": [(float(prop.com.x), float(prop.com.y), float(prop.com.z)) for prop in body_props],
                "shape_friction": [float(prop.friction) for prop in actor_shape_props],
                "shape_restitution": [float(prop.restitution) for prop in actor_shape_props],
                "shape_compliance": [
                    float(getattr(prop, "compliance", 0.0))
                    for prop in actor_shape_props
                ],
            })
            if (
                self.stage3_enabled
                and getattr(self.cfg.stage3, "fixed_rigid_props_at_creation", False)
            ):
                body_props, mass_params = self._process_stage3_fixed_rigid_body_props(
                    body_props
                )
            else:
                body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat([
                    torch_rand_float(self.cfg.domain_rand.leg_motor_strength_range[0], self.cfg.domain_rand.leg_motor_strength_range[1], (self.num_envs, 12), device=self.device),
                    torch_rand_float(self.cfg.domain_rand.arm_motor_strength_range[0], self.cfg.domain_rand.arm_motor_strength_range[1], (self.num_envs, 7), device=self.device)
                ], dim=1)
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)
        
        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level  # start from 0
        if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _parse_cfg(self, cfg):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
     
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        
        self.action_delay = self.cfg.env.action_delay

        self.stop_update_goal = False

    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask


    def  _get_phase(self):
        phase = self.gait_indices
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd
        
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        # FL RR foot stance
        stance_mask[:, 0] = sin_pos_l >= 0
        stance_mask[:, 3] = sin_pos_l >= 0
        # FR RL foot stance
        stance_mask[:, 1] = sin_pos_r < 0
        stance_mask[:, 2] = sin_pos_r < 0

        return stance_mask
    

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd
        repeat_default_pos = self.default_dof_pos[:, :12].repeat(self.num_envs, 1)
        # self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = repeat_default_pos.clone()
        scale_1 = self.cfg.rewards.target_joint_pos_scale / (1 - self.cfg.rewards.target_joint_pos_thd)
        scale_2 = scale_1 * 2
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = sin_pos_l[sin_pos_l > 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, 1] -= sin_pos_l * scale_1 # FL_thigh_joint
        self.ref_dof_pos[:, 2] += sin_pos_l * scale_2 # FL_calf_joint
        self.ref_dof_pos[:, 10] -= sin_pos_l * scale_1 # RR_thigh_joint
        self.ref_dof_pos[:, 11] += sin_pos_l * scale_2 # RR_calf_joint

        sin_pos_r[sin_pos_r < 0] = sin_pos_r[sin_pos_r < 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, 4] += sin_pos_r * scale_1 # FR_thigh_joint
        self.ref_dof_pos[:, 5] -= sin_pos_r * scale_2 # FR_calf_joint
        self.ref_dof_pos[:, 7] += sin_pos_r * scale_1 # RL_thigh_joint
        self.ref_dof_pos[:, 8] -= sin_pos_r * scale_2 # RL_calf_joint
 
    def subscribe_viewer_keyboard_events(self):
        super().subscribe_viewer_keyboard_events()
        if self.cfg.env.teleop_mode:
            self.teleop_motion_keys = {
                "forward": False,
                "reverse": False,
                "left": False,
                "right": False,
                "turn_left": False,
                "turn_right": False,
            }
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_x")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_y")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "ee_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "ee_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "ee_forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "ee_backward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "ee_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "ee_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "ee_roll_pos")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "ee_roll_neg")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "ee_pitch_pos")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "ee_pitch_neg")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "ee_yaw_pos")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "ee_yaw_neg")

    def _clip_ee_goal_sphere_to_training_range(self, ee_goal_sphere):
        lower = torch.tensor([self.goal_ee_ranges["pos_l"][0],
                              self.goal_ee_ranges["pos_p"][0],
                              self.goal_ee_ranges["pos_y"][0]], device=self.device)
        upper = torch.tensor([self.goal_ee_ranges["pos_l"][1],
                              self.goal_ee_ranges["pos_p"][1],
                              self.goal_ee_ranges["pos_y"][1]], device=self.device)
        return torch.max(torch.min(ee_goal_sphere, upper), lower)

    def _clip_ee_goal_orn_to_training_range(self, ee_goal_orn_rpy):
        lower = torch.tensor([self.goal_ee_ranges["orn_r"][0],
                              self.goal_ee_ranges["orn_p"][0],
                              self.goal_ee_ranges["orn_y"][0]], device=self.device)
        upper = torch.tensor([self.goal_ee_ranges["orn_r"][1],
                              self.goal_ee_ranges["orn_p"][1],
                              self.goal_ee_ranges["orn_y"][1]], device=self.device)
        return torch.max(torch.min(ee_goal_orn_rpy, upper), lower)

    def _set_teleop_ee_goal(self, ee_goal_sphere=None, ee_goal_orn_rpy=None):
        if ee_goal_sphere is not None:
            self.curr_ee_goal_sphere[:] = self._clip_ee_goal_sphere_to_training_range(ee_goal_sphere)
            self.ee_start_sphere[:] = self.curr_ee_goal_sphere[:]
            self.ee_goal_sphere[:] = self.curr_ee_goal_sphere[:]
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs, 3)

        if ee_goal_orn_rpy is not None:
            self.curr_ee_goal_orn_rpy[:] = self._clip_ee_goal_orn_to_training_range(ee_goal_orn_rpy)
            self.ee_start_orn_rpy[:] = self.curr_ee_goal_orn_rpy[:]
            self.ee_goal_orn_rpy[:] = self.curr_ee_goal_orn_rpy[:]
            self.commands[:, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)] = self.curr_ee_goal_orn_rpy.view(self.num_envs, 3)

    def _teleop_move_ee_cart(self, delta_x=0.0, delta_y=0.0, delta_z=0.0):
        ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart[:, 0] += delta_x
        ee_goal_cart[:, 1] += delta_y
        ee_goal_cart[:, 2] += delta_z
        self._set_teleop_ee_goal(ee_goal_sphere=cart2sphere(ee_goal_cart))

    def _teleop_move_ee_z(self, delta_z):
        self._teleop_move_ee_cart(delta_z=delta_z)

    def _teleop_roll_ee(self, delta_roll):
        ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy.clone()
        ee_goal_orn_rpy[:, 0] = wrap_to_pi(ee_goal_orn_rpy[:, 0] + delta_roll)
        self._set_teleop_ee_goal(ee_goal_orn_rpy=ee_goal_orn_rpy)

    def _teleop_pitch_ee(self, delta_pitch):
        ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy.clone()
        ee_goal_orn_rpy[:, 1] = wrap_to_pi(ee_goal_orn_rpy[:, 1] + delta_pitch)
        self._set_teleop_ee_goal(ee_goal_orn_rpy=ee_goal_orn_rpy)

    def _teleop_yaw_ee(self, delta_yaw):
        ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy.clone()
        ee_goal_orn_rpy[:, 2] = wrap_to_pi(ee_goal_orn_rpy[:, 2] + delta_yaw)
        self._set_teleop_ee_goal(ee_goal_orn_rpy=ee_goal_orn_rpy)

    def _update_teleop_motion_command(self):
        self._ensure_teleop_motion_keys()

        command_x = 0.0
        if self.teleop_motion_keys.get("forward", False) != self.teleop_motion_keys.get("reverse", False):
            command_x = self.command_ranges["lin_vel_x"][1] if self.teleop_motion_keys.get("forward", False) else self.command_ranges["lin_vel_x"][0]

        command_y = 0.0
        if self.teleop_motion_keys.get("left", False) != self.teleop_motion_keys.get("right", False):
            command_y = self.command_ranges["lin_vel_y"][1] if self.teleop_motion_keys.get("left", False) else self.command_ranges["lin_vel_y"][0]

        command_yaw = 0.0
        if self.teleop_motion_keys.get("turn_left", False) != self.teleop_motion_keys.get("turn_right", False):
            command_yaw = self.command_ranges["ang_vel_yaw"][1] if self.teleop_motion_keys.get("turn_left", False) else self.command_ranges["ang_vel_yaw"][0]

        self.commands[:, 0] = command_x
        self.commands[:, 1] = command_y
        self.commands[:, 2] = command_yaw

    def _ensure_teleop_motion_keys(self):
        if not hasattr(self, "teleop_motion_keys"):
            self.teleop_motion_keys = {
                "forward": False,
                "reverse": False,
                "left": False,
                "right": False,
                "turn_left": False,
                "turn_right": False,
            }
    
    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)

        motion_actions = {"forward", "reverse", "left", "right", "turn_left", "turn_right"}
        if evt.action in motion_actions:
            self._ensure_teleop_motion_keys()
            self.teleop_motion_keys[evt.action] = evt.value > 0
            self._update_teleop_motion_command()
            return

        if evt.action == "stop_x" and evt.value > 0:
            self._ensure_teleop_motion_keys()
            self.teleop_motion_keys["forward"] = False
            self.teleop_motion_keys["reverse"] = False
            self._update_teleop_motion_command()
            return
        elif evt.action == "stop_y" and evt.value > 0:
            self._ensure_teleop_motion_keys()
            self.teleop_motion_keys["left"] = False
            self.teleop_motion_keys["right"] = False
            self._update_teleop_motion_command()
            return
        elif evt.action == "stop_angular" and evt.value > 0:
            self._ensure_teleop_motion_keys()
            self.teleop_motion_keys["turn_left"] = False
            self.teleop_motion_keys["turn_right"] = False
            self._update_teleop_motion_command()
            return
        
        if evt.value <= 0:
            return

        if evt.action == "ee_up":
            self._teleop_move_ee_z(self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_down":
            self._teleop_move_ee_z(-self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_forward":
            self._teleop_move_ee_cart(delta_x=self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_backward":
            self._teleop_move_ee_cart(delta_x=-self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_left":
            self._teleop_move_ee_cart(delta_y=self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_right":
            self._teleop_move_ee_cart(delta_y=-self.cfg.env.teleop_ee_pos_step)
        elif evt.action == "ee_roll_pos":
            self._teleop_roll_ee(self.cfg.env.teleop_ee_orn_step)
        elif evt.action == "ee_roll_neg":
            self._teleop_roll_ee(-self.cfg.env.teleop_ee_orn_step)
        elif evt.action == "ee_pitch_pos":
            self._teleop_pitch_ee(self.cfg.env.teleop_ee_orn_step)
        elif evt.action == "ee_pitch_neg":
            self._teleop_pitch_ee(-self.cfg.env.teleop_ee_orn_step)
        elif evt.action == "ee_yaw_pos":
            self._teleop_yaw_ee(self.cfg.env.teleop_ee_orn_step)
        elif evt.action == "ee_yaw_neg":
            self._teleop_yaw_ee(-self.cfg.env.teleop_ee_orn_step)
        # print(evt.action, self.commands)

        

    #------------ reward functions----------------
    def _get_ee_tracking_acc_terms(self):
        ee_error = self.curr_ee_goal_cart_world - self.ee_pos
        ee_error_norm = torch.norm(ee_error, dim=1, keepdim=True)
        target_dir = ee_error / torch.clamp(
            ee_error_norm,
            min=self.cfg.rewards.ee_tracking_direction_eps,
        )

        ee_acc = (self.ee_vel[:, 0:3] - self.last_ee_vel[:, 0:3]) / self.dt
        tangent_acc = torch.sum(ee_acc * target_dir, dim=1)
        normal_acc = ee_acc - tangent_acc.unsqueeze(1) * target_dir

        last_tangent_acc = torch.sum(self.last_ee_acc * target_dir, dim=1)
        tangent_jerk = (tangent_acc - last_tangent_acc) / self.dt
        return ee_error_norm.squeeze(1), normal_acc, tangent_jerk

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - self.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew

    def _reward_ee_normal_acc(self):
        _, normal_acc, _ = self._get_ee_tracking_acc_terms()
        normal_acc_norm = torch.norm(normal_acc, dim=1)
        penalty = torch.square(normal_acc_norm / self.cfg.rewards.ee_normal_acc_sigma)
        return torch.clamp(penalty, max=self.cfg.rewards.ee_acc_penalty_clip)

    def _reward_ee_tangent_acc_smooth(self):
        _, _, tangent_jerk = self._get_ee_tracking_acc_terms()
        penalty = torch.square(tangent_jerk / self.cfg.rewards.ee_tangent_jerk_sigma)
        return torch.clamp(penalty, max=self.cfg.rewards.ee_acc_penalty_clip)
    
    def _reward_tracking_ee_orn(self):
        ee_orn_error = wrap_to_pi(self.get_ee_orientation_rpy() - self.curr_ee_goal_orn_rpy)
        ee_orn_error = torch.sum(torch.square(ee_orn_error * self.orn_error_scale), dim=1)
        return torch.exp(-ee_orn_error / self.cfg.rewards.tracking_ee_orn_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_lin_acc_z(self):
        """Penalize high-frequency base acceleration along the world Z axis."""
        lin_acc_z = (
            self.root_states[:, 9] - self.last_root_vel[:, 2]
        ) / self.dt
        penalty = torch.square(
            lin_acc_z / self.cfg.rewards.base_lin_acc_z_norm
        )
        return torch.clamp(
            penalty, max=self.cfg.rewards.base_acc_penalty_clip
        )

    def _reward_base_ang_acc_yaw(self):
        """Penalize high-frequency acceleration about the body yaw axis."""
        last_ang_vel_body = quat_rotate_inverse(
            self.base_quat, self.last_root_vel[:, 3:6]
        )
        yaw_acc = (
            self.base_ang_vel[:, 2] - last_ang_vel_body[:, 2]
        ) / self.dt
        penalty = torch.square(
            yaw_acc / self.cfg.rewards.base_ang_acc_yaw_norm
        )
        return torch.clamp(
            penalty, max=self.cfg.rewards.base_acc_penalty_clip
        )
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_roll(self):
        # Penalize non flat base orientation
        roll = self.get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error
    
    def _reward_pitch(self):
        # Penalize non flat base orientation
        pitch = self.get_body_orientation()[:, 1]
        error = torch.abs(pitch)
        return error
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques)[:, :12], dim=1)
    
    def _reward_torques_arm(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques)[:, 12:19], dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, :12], dim=1)

    def _reward_dof_vel_arm(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, 12:19], dim=1)
        
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, :12], dim=1)
        return energy
    
    def _reward_energy_square_stand(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, :12], dim=1)

        walking_flag = self.get_walking_cmd_mask()
        energy[walking_flag] = 0
        return energy
    
    def _reward_energy_square_arm(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, 12:19], dim=1)
        return energy
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, :12] / self.dt), dim=1)
    
    def _reward_dof_acc_arm(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, 12:19] / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, :12], dim=1)
    
    def _reward_action_rate_arm(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 12:19], dim=1)

    def _reward_action_acc_arm(self):
        """Penalize Nero target curvature that excites unobserved wrist modes."""
        second_difference = (
            self.actions + self.last_last_actions - 2.0 * self.last_actions
        )[:, 12:19]
        return torch.sum(torch.square(second_difference), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits[:, :19], dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_torque_limits_leg(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques[:,:12]) - self.torque_limits[:12] * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_torque_limits_arm(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques[:,12:19]) - self.torque_limits[12:19] * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_penalty(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        lin_vel_error = torch.sum(torch.abs(self.commands[:, :2] - self.base_ang_vel[:, :2]), dim=1)
        # print(ang_vel_error)
        penalty = torch.zeros_like(lin_vel_error).to(self.device)
        penalty[lin_vel_error>0.4] += 0.5 
        return penalty
    
    def _reward_feet_vel_xy(self):
         # Penalize xy axis feet linear velocity
        foot_velocities = torch.norm(self.foot_velocities[:, :, :2], dim=2).view(self.num_envs, -1)
        mean_velocity = torch.mean(foot_velocities, dim=1)
        return mean_velocity
    
    def _reward_feet_pos_xy(self):
        # Penalize xy axis feet linear velocity
        feet_pos_xy = self.rigid_state[:, self.feet_indices, :2]
        thigh_pos_xy = self.rigid_state[:, self.thigh_indices, :2]
        diff = torch.norm(feet_pos_xy-thigh_pos_xy, dim=2).view(self.num_envs, -1)
        mean_diff = torch.mean(diff, dim=1)
        return mean_diff
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_ang_penalty(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        # print(ang_vel_error)
        penalty = torch.zeros_like(ang_vel_error).to(self.device)
        penalty[ang_vel_error>0.4] += 0.5 
        return penalty
    
    def _reward_feet_height(self):
        feet_height = self.rigid_state[:, self.feet_indices[:2], 2] # Only front feet
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.10, max=0)
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew
    
    def _reward_feet_height_high(self):
        feet_height = self.rigid_state[:, self.feet_indices, 2]
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.20, min=0)
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew
    
    def _reward_feet_height_high_standing(self):
        feet_height = self.rigid_state[:, self.feet_indices, 2]
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.05, min=0)
        cmd_walk_flag = self.get_walking_cmd_mask()
        rew[cmd_walk_flag] = 0
        return rew

    def _reward_feet_hind_height(self):
        feet_height = self.rigid_state[:, self.feet_indices[2:], 2] # Only front feet
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.10, max=0)
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew
        
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= self.get_walking_cmd_mask() #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[self.get_walking_cmd_mask()] = 0.
        return rew

    def _reward_walking_dof(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[~self.get_walking_cmd_mask()] = 0.
        return rew
    
    def _reward_walking_ref_dof(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(joint_pos - pos_target)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.2)
        rew[~self.get_walking_cmd_mask()] = 0.
        return rew
    
    def _reward_walking_ref_swing_dof(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        # Penalize motion at zero commands
        stand_mask = self._get_gait_phase()
        stand_mask = torch.stack([stand_mask, stand_mask,stand_mask],2).reshape(self.num_envs, 12)
        dof_error = torch.abs(joint_pos - pos_target)[:, :12]
        dof_error[stand_mask==1] = 0
        dof_error = torch.sum(dof_error, dim=1)
        rew = torch.exp(-dof_error*0.2)
        rew[~self.get_walking_cmd_mask()] = 0.
        return rew
    
    def _reward_walking_ref_stand_dof(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        # Penalize motion at zero commands
        stand_mask = self._get_gait_phase()
        stand_mask = torch.stack([stand_mask, stand_mask,stand_mask],2).reshape(self.num_envs, 12)
        dof_error = torch.abs(joint_pos - pos_target)[:, :12]
        dof_error[stand_mask==0] = 0
        dof_error = torch.sum(dof_error, dim=1)
        rew = torch.exp(-dof_error*0.5) - 1
        rew[~self.get_walking_cmd_mask()] = 0.
        return rew
    
    def _reward_ref_dof_leg(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(joint_pos - pos_target)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.1)
        return rew
        
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos - pos_target)
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[~self.get_walking_cmd_mask()] = 0.
        return r
    
    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
        return rew

    def _reward_nero_joint1_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint1_idx] - self.default_dof_pos[:, self.nero_joint1_idx])

    def _reward_nero_joint2_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint2_idx] - self.default_dof_pos[:, self.nero_joint2_idx])

    def _reward_nero_joint3_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint3_idx] - self.default_dof_pos[:, self.nero_joint3_idx])

    def _reward_nero_joint4_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint4_idx] - self.default_dof_pos[:, self.nero_joint4_idx])

    def _reward_nero_joint5_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint5_idx] - self.default_dof_pos[:, self.nero_joint5_idx])

    def _reward_nero_joint6_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint6_idx] - self.default_dof_pos[:, self.nero_joint6_idx])

    def _reward_nero_joint7_pos(self):
        return torch.square(self.dof_pos[:, self.nero_joint7_idx] - self.default_dof_pos[:, self.nero_joint7_idx])

    def _reward_gripper_joint1_pos(self):
        return torch.square(self.dof_pos[:, self.gripper_joint1_idx] - self.default_dof_pos[:, self.gripper_joint1_idx])

    def _reward_gripper_joint2_pos(self):
        return torch.square(self.dof_pos[:, self.gripper_joint2_idx] - self.default_dof_pos[:, self.gripper_joint2_idx])
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_touchdown_vertical_velocity(self):
        """Penalize downward foot speed on the first contact policy tick."""
        foot_vz = self.rigid_state[:, self.feet_indices, 9]
        downward_speed = torch.clamp(-foot_vz, min=0.0)
        return torch.sum(
            torch.square(downward_speed) * self.identified_touchdown.float(), dim=1
        )

    def _reward_nero_touchdown_phase_power(self):
        """Discourage arm power injection at leg impact without freezing Nero."""
        touchdown = torch.any(self.identified_touchdown, dim=1).float()
        nero_power = self.torques[:, 12:19] * self.dof_vel[:, 12:19]
        return touchdown * torch.mean(torch.square(nero_power / 10.0), dim=1)

    def _reward_unobserved_wrist_flex(self):
        """Penalize an unobserved structural deflection estimate during gait."""
        if self.unobserved_wrist_flex is None:
            return torch.zeros(self.num_envs, device=self.device)
        normalizer = float(
            self.cfg.stage3.identified_profile["unobserved_wrist_flex"]
            ["deflection_norm_m"]
        )
        penalty = torch.sum(
            torch.square(
                self.unobserved_wrist_flex.total_deflection / normalizer
            ),
            dim=1,
        )
        return penalty * self.get_walking_cmd_mask().float()

    def _reward_feet_height_symmetry(self):
        # penalize asymmetry feet height
        feet_height = self.rigid_state[:, self.feet_indices, 2] # Only front feet
        rew = abs(feet_height[:,0] - feet_height[:,3]) + abs(feet_height[:,1] - feet_height[:,2])
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew
    
    def _reward_alive(self):
        return 1.
    
    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.rigid_state[:, self.feet_indices, 7:10]).sum(dim=-1)
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        dragging_vel = foot_forces * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew

    
    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.torques - self.last_torques)[:, :12], dim=1)
        return rew
    
    def _reward_delta_torques_arm(self):
        rew = torch.sum(torch.square(self.torques - self.last_torques)[:, 12:19], dim=1)
        return rew
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:, :12], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:, :12], dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions)[:, :12], dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)
    
    def _reward_feet_contact_number_walking(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        reward = torch.mean(reward, dim=1)

        cmd_stop_flag = ~self.get_walking_cmd_mask()
        reward[cmd_stop_flag] = 0
        return reward
    
    def _reward_feet_contact_number_standing(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 0, -1.)
        reward = torch.mean(reward, dim=1)
        cmd_walking_flag = self.get_walking_cmd_mask()
        
        reward[cmd_walking_flag] = 0
        return reward
    
