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
from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfg

import matplotlib.pyplot as plt



INDEX_EE_POS_RADIUS_CMD = 3
INDEX_EE_POS_PITCH_CMD = 4
INDEX_EE_POS_YAW_CMD = 5
INDEX_EE_ROLL_CMD = 6
INDEX_EE_PITCH_CMD =7
INDEX_EE_YAW_CMD = 8
INDEX_EE_FORCE_X = 9
INDEX_EE_FORCE_Y = 10
INDEX_EE_FORCE_Z = 11
INDEX_BASE_FORCE_X = 12
INDEX_BASE_FORCE_Y = 13
INDEX_BASE_FORCE_Z = 14

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobot_b2z1_pos_force(BaseTask):
    def __init__(self, cfg: B2Z1PosForceRoughCfg, sim_params, physics_engine, sim_device, headless):
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
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
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
        self.init_done = True
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

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
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.global_steps > self.cfg.commands.force_start_step * 24:
                # push gripper
                self._push_gripper(torch.arange(self.num_envs, device=self.device))      
                # push robot base 
                # self._push_robot_base(torch.arange(self.num_envs, device=self.device)) 
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
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
            self._draw_collision_bbox()
            self._draw_ee_force()
            self._draw_base_force()
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
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.goal_timer[env_ids] = 0.

        # force control
        self.forces[env_ids, self.gripper_idx, :3] = 0.
        self.selected_env_ids_gripper_cmd[env_ids] = 0
        self.selected_env_ids_gripper_ext[env_ids] = 0
        self.push_end_time_gripper_cmd[env_ids] = 0.
        self.force_target_gripper_cmd[env_ids, :3] = 0.
        self.force_target_gripper_ext[env_ids, :3] = 0.
        self.push_duration_gripper_cmd[env_ids] = 0.
        self.current_Fxyz_gripper_cmd[env_ids, :3] = 0.



        self.forces[env_ids, self.robot_base_idx, :3] = 0.
        self.selected_env_ids_base_cmd[env_ids] = 0
        self.selected_env_ids_base_ext[env_ids] = 0
        self.push_end_time_base_cmd[env_ids] = 0.
        self.force_target_base_cmd[env_ids, :3] = 0.
        self.force_target_base_ext[env_ids, :3] = 0.
        self.push_duration_base_cmd[env_ids] = 0.
        self.current_Fxyz_base_cmd[env_ids, :3] = 0.
        
        self.commands[env_ids, INDEX_EE_FORCE_X] = 0.0
        self.commands[env_ids, INDEX_EE_FORCE_Y] = 0.0
        self.commands[env_ids, INDEX_EE_FORCE_Z] = 0.0

        self.commands[env_ids, INDEX_BASE_FORCE_X] = 0.0
        self.commands[env_ids, INDEX_BASE_FORCE_Y] = 0.0
        self.commands[env_ids, INDEX_BASE_FORCE_Z] = 0.0
        

        # Reset push gripper 
        if self.cfg.commands.push_gripper_stators:
            self.forces[env_ids, self.gripper_idx, :3] = 0.
            self.selected_env_ids_gripper_cmd[env_ids] = 0
            self.selected_env_ids_gripper_ext[env_ids] = 0
            self.push_end_time_gripper_cmd[env_ids] = 0.

        # Reset push robot base 
        if self.cfg.commands.push_robot_base:
            self.forces[env_ids, self.robot_base_idx, :3] = 0.
            self.selected_env_ids_base_cmd[env_ids] = 0
            self.selected_env_ids_base_ext[env_ids] = 0
            self.push_end_time_base_cmd[env_ids] = 0.

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

        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)

        ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        # Spherical to cartesian coordinates in the arm base frame 
        radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
        pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
        yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        self.ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)
        
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        forces_global_gripper = self.forces[:, self.gripper_idx, 0:3]
        self.forces_local[:, self.gripper_idx] = quat_rotate_inverse(base_quat_world_indep, forces_global_gripper).view(self.num_envs, 3)

        forces_global_base = self.forces[:, self.robot_base_idx, 0:3]
        self.forces_local[:, self.robot_base_idx] = quat_rotate_inverse(base_quat_world_indep, forces_global_base).view(self.num_envs, 3)
        

        # offset

        forces_global = self.forces[:, self.gripper_idx, 0:3]
        forces_cmd = self.current_Fxyz_gripper_cmd
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_offset = (forces_global + forces_cmd_global)
        curr_ee_goal_cart_world_offset = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
        ee_goal_offset_local_cart = quat_rotate_inverse(self.base_yaw_quat, curr_ee_goal_cart_world_offset - self.get_ee_goal_spherical_center())
        ee_goal_offset_local_sphere = cart2sphere(ee_goal_offset_local_cart)

        self.privileged_obs_buf = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    self.forces_local[:, self.gripper_idx] * self.obs_scales.ee_force, # 3
                                    self.forces_local[:, self.robot_base_idx] * self.obs_scales.base_force, # 3
                                    diff, # 12
                                    self.mass_params_tensor, # 22
                                    self.friction_coeffs_tensor, #  1
                                    self.motor_strength[:, :17] - 1, # 17
                                    stance_mask, # 4
                                    contact_mask, # 4
                                    self.projected_gravity, # 3
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    self.actions[:, :17], # dim 17
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    (self.commands * self.commands_scale)[:, :15], # dim 15
                                    # base_lin_vel_offset * self.obs_scales.lin_vel, # dim 2
                                    ee_goal_offset_local_sphere[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    ee_goal_offset_local_sphere[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    ee_goal_offset_local_sphere[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    ),dim=-1)
        obs_pred = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    self.forces_local[:, self.gripper_idx] * self.obs_scales.ee_force, # 3
                                    self.forces_local[:, self.robot_base_idx] * self.obs_scales.base_force, # 3
                                    ),dim=-1)
        
        obs_buf = torch.cat(( self.get_body_orientation(),  # dim 2
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    self.actions[:, :17], # dim 17
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    (self.commands * self.commands_scale)[:, :15], # dim 15
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

        sphere_geom_4 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))
        upper_arm_pose = self.get_ee_goal_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_state[:, self.gripper_idx, :3]

        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)


        forces_global = self.forces[:, self.gripper_idx, 0:3]
        forces_cmd = self.current_Fxyz_gripper_cmd
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_offset = (forces_global + forces_cmd_global)
        curr_ee_goal_cart_world_offset = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

            sphere_pose_4 = gymapi.Transform(gymapi.Vec3(curr_ee_goal_cart_world_offset[i, 0], curr_ee_goal_cart_world_offset[i, 1], curr_ee_goal_cart_world_offset[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_4, self.gym, self.viewer, self.envs[i], sphere_pose_4) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_ee_force(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom_arrow_1 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))
        arrow_color_1 = [0, 0, 1]
        sphere_geom_arrow_2 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
        arrow_color_2 = [0, 1, 0]
        ee_pose = self.rigid_state[:, self.gripper_idx, :3]
        forces_global = self.forces[:, self.gripper_idx, 0:3] / 100
        forces_global_norm = torch.norm(forces_global, dim=-1, keepdim=True)
        target_forces_global = forces_global / (forces_global_norm + 1e-5)
        forces_cmd = self.current_Fxyz_gripper_cmd / 100
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_cmd_norm = torch.norm(forces_cmd_global, dim=-1, keepdim=True)
        target_forces_cmd = forces_cmd_global / (forces_cmd_norm + 1e-5)
        for i in range(self.num_envs):

            start_pos = ee_pose[i].cpu().numpy()
            arrow_direction = target_forces_global[i].cpu().numpy()
            arrow_length = forces_global_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_1, arrow_color_1]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_1, self.gym, self.viewer, self.envs[i], head_pose)

            start_pos = ee_pose[i].cpu().numpy()
            arrow_direction = target_forces_cmd[i].cpu().numpy()
            arrow_length = forces_cmd_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_2, arrow_color_2]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_2, self.gym, self.viewer, self.envs[i], head_pose)
    
    def _draw_base_force(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom_arrow_1 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))
        arrow_color_1 = [0, 0, 1]
        sphere_geom_arrow_2 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
        arrow_color_2 = [0, 1, 0]
        base_pose = self.rigid_state[:, self.robot_base_idx, :3]
        forces_global = self.forces[:, self.robot_base_idx, 0:3] / 100
        forces_global_norm = torch.norm(forces_global, dim=-1, keepdim=True)
        target_forces_global = forces_global / (forces_global_norm + 1e-5)
        forces_cmd = self.current_Fxyz_base_cmd / 100
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_cmd_norm = torch.norm(forces_cmd_global, dim=-1, keepdim=True)
        target_forces_cmd = forces_cmd_global / (forces_cmd_norm + 1e-5)
        for i in range(self.num_envs):

            start_pos = base_pose[i].cpu().numpy() + [0, 0, 0.5]
            arrow_direction = target_forces_global[i].cpu().numpy()
            arrow_length = forces_global_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_1, arrow_color_1]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_1, self.gym, self.viewer, self.envs[i], head_pose)

            start_pos = base_pose[i].cpu().numpy() + [0, 0, 0.5]
            arrow_direction = target_forces_cmd[i].cpu().numpy()
            arrow_length = forces_cmd_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_2, arrow_color_2]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_2, self.gym, self.viewer, self.envs[i], head_pose)
            
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
        if self.cfg.domain_rand.randomize_friction:
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

    def _randomize_dof_props(self, env_ids):
        if self.cfg.commands.randomize_gripper_force_gains:
            min_kp, max_kp = self.cfg.commands.gripper_force_kp_range
            min_kd, max_kd = self.cfg.commands.gripper_force_kd_range
            self.gripper_force_kps[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_kp - min_kp) + min_kp
            if self.cfg.commands.gripper_prop_kd > 0:
                self.gripper_force_kds[env_ids, :] = self.gripper_force_kps[env_ids, :] * self.cfg.commands.gripper_prop_kd
            else:
                self.gripper_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                        requires_grad=False).unsqueeze(1) * (
                                                    max_kd - min_kd) + min_kd
        if self.cfg.commands.randomize_base_force_gains:
            min_kp, max_kp = self.cfg.commands.base_force_kp_range
            min_kd, max_kd = self.cfg.commands.base_force_kd_range
            self.base_force_kps[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_kp - min_kp) + min_kp
            self.base_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                        requires_grad=False).unsqueeze(1) * (
                                                    max_kd - min_kd) + min_kd
                
    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_leg_mass:
            rng_link = self.cfg.domain_rand.leg_mass_scale_range
            rand_leg_masses = []
            for idx in range(17):
                rand_link_mass = np.random.uniform(rng_link[0], rng_link[1], size=(1, )) * props[idx].mass
                props[idx].mass += rand_link_mass
                rand_leg_masses.append(rand_link_mass)
            rand_leg_masses = np.concatenate(rand_leg_masses)
        else:
            rand_leg_masses = np.zeros(17)

        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[1].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        
        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass, rand_leg_masses])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._step_contact_targets()

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
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.motor_strength * self.cfg.control.action_scale

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        last_torque = 64. * (torch.zeros([actions_scaled.shape[0], 1], device=actions_scaled.device) + self.default_dof_pos[:, -2] - self.dof_pos[:, -2:-1]) - 1.5 * self.dof_vel[:, -2:-1]
        gripper_torque = 64. * (torch.zeros([actions_scaled.shape[0], 1], device=actions_scaled.device) + self.default_dof_pos[:, -1] - self.dof_pos[:, -1:]) - 1.5 * self.dof_vel[:, -1:]
        
        torques = torch.cat([default_torques, last_torque, gripper_torque], dim=-1)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.cfg.env.teleop_mode and is_init:
            self.curr_ee_goal_sphere[:] = self.init_start_ee_sphere[:]
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)
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
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
            else:
                if self.global_steps < 0 * 24 and not self.play:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
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
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)

        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        # TODO: add twisting motion by fixing yaw at traj start
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + ee_goal_cart_yaw_global
        
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2, default_pitch + self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)
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
        self.dof_pos[env_ids, :12] = self.default_dof_pos[:, :12]* torch_rand_float(0.5, 1.5, (len(env_ids), 12), device=self.device)
        self.dof_pos[env_ids, 12:17] += torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_actions-12), device=self.device)
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
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
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

    def _push_gripper(self, env_ids_all):
        """ Randomly pushes the gripper stators. Emulates an impulse by setting a randomized gripper stator velocity.
        """
        
        if self.cfg.commands.push_gripper_stators:
            # cmd force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:
                
                self.freed_envs_gripper_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_cmd
                min_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[0]
                max_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[1]

                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 0] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 1] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 2] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                push_duration_gripper_cmd = torch_rand_float(self.push_duration_gripper_cmd_min, self.push_duration_gripper_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_gripper_cmd = torch.clip(push_duration_gripper_cmd, max=(self.push_interval_gripper_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_gripper_cmd
                self.push_duration_gripper_cmd[new_selected_env_ids_cmd] = push_duration_gripper_cmd
                
                self.selected_env_ids_gripper_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_cmd == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] < (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :3] = (self.force_target_gripper_cmd[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_X] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :2], dim=1)
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Y] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0])
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Z] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 2]
 
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] > (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :3] = self.force_target_gripper_cmd[env_ids_apply_push_step2, :3] - (self.force_target_gripper_cmd[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    # World frame
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_X] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :2], dim=1)
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Y] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0])
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Z] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] >= (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper + self.push_duration_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_cmd[env_ids_to_reset] = 0
                    self.force_target_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_cmd[env_ids_to_reset] = 0.
                    self.push_duration_gripper_cmd[env_ids_to_reset] = 0.
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_X] = 0.0
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_Y] = 0.0
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_Z] = 0.0
                    self.push_interval_gripper_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_cmd_min), int(self.push_interval_gripper_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_cmd[self.freed_envs_gripper_cmd] = 0
            self.force_target_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.current_Fxyz_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.push_end_time_gripper_cmd[self.freed_envs_gripper_cmd] = 0.
            self.push_duration_gripper_cmd[self.freed_envs_gripper_cmd] = 0. 
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_X] = 0.0
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_Y] = 0.0
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_Z] = 0.0


            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_ext[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_gripper_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_ext
                min_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[0]
                max_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[1]

                self.force_target_gripper_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 2] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                push_duration_gripper_ext = torch_rand_float(self.push_duration_gripper_ext_min, self.push_duration_gripper_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_gripper_ext = torch.clip(push_duration_gripper_ext, max=(self.push_interval_gripper_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_gripper_ext
                self.push_duration_gripper_ext[new_selected_env_ids_ext] = push_duration_gripper_ext
                
                self.selected_env_ids_gripper_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_ext == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] < (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.forces[env_ids_apply_push_step1, self.gripper_idx, :3] = (self.force_target_gripper_ext[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] > (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2, self.gripper_idx, :3] = self.force_target_gripper_ext[env_ids_apply_push_step2, :3] - (self.force_target_gripper_ext[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] >= (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper + self.push_duration_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]                                        
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_ext[env_ids_to_reset] = 0
                    self.force_target_gripper_ext[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_ext[env_ids_to_reset] = 0.
                    self.push_duration_gripper_ext[env_ids_to_reset] = 0.
                    self.push_interval_gripper_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_ext_min), int(self.push_interval_gripper_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_ext[self.freed_envs_gripper_ext] = 0
            self.force_target_gripper_ext[self.freed_envs_gripper_ext, :3] = 0.
            self.push_end_time_gripper_ext[self.freed_envs_gripper_ext] = 0.
            self.push_duration_gripper_ext[self.freed_envs_gripper_ext] = 0. 
            
            self.forces[self.freed_envs_gripper_ext, self.gripper_idx, :3] = 0

            

    def _push_robot_base(self, env_ids_all):
        """ 
            Randomly pushes the robot base.
        """
        if self.cfg.commands.push_robot_base:
            # cmd force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_base_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:
                
                self.freed_envs_base_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.base_forced_prob_cmd
                min_force_cmd = self.cfg.commands.max_push_force_xyz_base_cmd[0]
                max_force_cmd = self.cfg.commands.max_push_force_xyz_base_cmd[1]

                self.force_target_base_cmd[new_selected_env_ids_cmd, 0] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_base_cmd[new_selected_env_ids_cmd, 1] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_base_cmd[new_selected_env_ids_cmd, 2] = torch.zeros((len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                push_duration_base_cmd = torch_rand_float(self.push_duration_base_cmd_min, self.push_duration_base_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_base_cmd = torch.clip(push_duration_base_cmd, max=(self.push_interval_base_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_base)/2).to(self.device)
                self.push_end_time_base_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_base_cmd
                self.push_duration_base_cmd[new_selected_env_ids_cmd] = push_duration_base_cmd
                
                self.selected_env_ids_base_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_base_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_base_cmd == 1]

                # Step 1: apply force from 0 to force_target_base_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] < (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.current_Fxyz_base_cmd[env_ids_apply_push_step1, :3] = (self.force_target_base_cmd[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_base_cmd[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_X] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_base_cmd[env_ids_apply_push_step1, :2], dim=1)
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_Y] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 0])
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_Z] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 2]
 
                # Step 2: apply force from force_target_base_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] > (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1] + self.settling_time_force_base).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                    self.current_Fxyz_base_cmd[env_ids_apply_push_step2, :3] = self.force_target_base_cmd[env_ids_apply_push_step2, :3] - (self.force_target_base_cmd[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_base_cmd[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_base), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    # World frame
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_X] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_base_cmd[env_ids_apply_push_step2, :2], dim=1)
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_Y] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 0])
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_Z] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] >= (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1] + self.settling_time_force_base + self.push_duration_base_cmd[self.selected_env_ids_base_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_base_cmd[env_ids_to_reset] = 0
                    self.force_target_base_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_base_cmd[env_ids_to_reset, :3] = 0.
                    self.push_end_time_base_cmd[env_ids_to_reset] = 0.
                    self.push_duration_base_cmd[env_ids_to_reset] = 0.
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_X] = 0.0
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_Y] = 0.0
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_Z] = 0.0
                    self.push_interval_base_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_base_cmd_min), int(self.push_interval_base_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_base_cmd[self.freed_envs_base_cmd] = 0
            self.force_target_base_cmd[self.freed_envs_base_cmd, :3] = 0.
            self.current_Fxyz_base_cmd[self.freed_envs_base_cmd, :3] = 0.
            self.push_end_time_base_cmd[self.freed_envs_base_cmd] = 0.
            self.push_duration_base_cmd[self.freed_envs_base_cmd] = 0. 
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_X] = 0.0
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_Y] = 0.0
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_Z] = 0.0

            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_base_ext[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_base_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.base_forced_prob_ext
                min_force_ext = self.cfg.commands.max_push_force_xyz_base_ext[0]
                max_force_ext = self.cfg.commands.max_push_force_xyz_base_ext[1]

                self.force_target_base_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_base_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_base_ext[new_selected_env_ids_ext, 2] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) * self.cfg.commands.force_z_base_ext_scale
                push_duration_base_ext = torch_rand_float(self.push_duration_base_ext_min, self.push_duration_base_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_base_ext = torch.clip(push_duration_base_ext, max=(self.push_interval_base_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_base)/2).to(self.device)
                self.push_end_time_base_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_base_ext
                self.push_duration_base_ext[new_selected_env_ids_ext] = push_duration_base_ext
                
                self.selected_env_ids_base_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_base_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_base_ext == 1]

                # Step 1: apply force from 0 to force_target_base_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] < (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_ext[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.forces[env_ids_apply_push_step1, self.robot_base_idx, :3] = (self.force_target_base_ext[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_base_ext[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_base_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] > (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1] + self.settling_time_force_base).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_ext[env_ids_apply_push_step2].unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2, self.robot_base_idx, :3] = self.force_target_base_ext[env_ids_apply_push_step2, :3] - (self.force_target_base_ext[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_base_ext[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_base), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] >= (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1] + self.settling_time_force_base + self.push_duration_base_ext[self.selected_env_ids_base_ext == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_base_ext[env_ids_to_reset] = 0
                    self.force_target_base_ext[env_ids_to_reset, :3] = 0.
                    self.push_end_time_base_ext[env_ids_to_reset] = 0.
                    self.push_duration_base_ext[env_ids_to_reset] = 0.
                    self.push_interval_base_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_base_ext_min), int(self.push_interval_base_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_base_ext[self.freed_envs_base_ext] = 0
            self.force_target_base_ext[self.freed_envs_base_ext, :3] = 0.
            self.push_end_time_base_ext[self.freed_envs_base_ext] = 0.
            self.push_duration_base_ext[self.freed_envs_base_ext] = 0. 

            self.forces[self.freed_envs_base_ext, self.robot_base_idx, :3] = 0

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
        noise_vec[5+self.num_actions*3:] = 0. # commands

        return noise_vec

    #----------------------------------------
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
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.arm_base_offset = torch.tensor([0.2, 0., 0.225], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.gripper_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 0:3]
        self.gripper_velocity = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 7:10]
        # ee info
        self.ee_pos = self.rigid_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_state[:, self.gripper_idx, 7:]
        
        # target_ee info
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_pos_sphe_arm = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        
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
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros_like(self.torques)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.ee_sphe_radius_cmd, 
                                            self.obs_scales.ee_sphe_pitch_cmd,
                                            self.obs_scales.ee_sphe_yaw_cmd,
                                            self.obs_scales.end_effector_roll_cmd, 
                                            self.obs_scales.end_effector_pitch_cmd,
                                            self.obs_scales.end_effector_yaw_cmd,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.base_force,
                                            self.obs_scales.base_force,
                                            self.obs_scales.base_force,], device=self.device, requires_grad=False,) # TODO change this
        
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)

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
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:, :-self.cfg.env.num_gripper_joints]
        
        # gait
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.foot_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]

        # force control

        # Push gripper 
        self.freed_envs_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_gripper_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_gripper_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval_gripper_cmd = torch.randint(int(self.push_interval_gripper_cmd_min), int(self.push_interval_gripper_cmd_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_interval_gripper_ext = torch.randint(int(self.push_interval_gripper_ext_min), int(self.push_interval_gripper_ext_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_gripper_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_end_time_gripper_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_gripper_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Push robot 
        self.freed_envs_base_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_base_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_base_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_base_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval_base_cmd = torch.randint(int(self.push_interval_base_cmd_min), int(self.push_interval_base_cmd_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_interval_base_ext = torch.randint(int(self.push_interval_base_ext_min), int(self.push_interval_base_ext_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_base_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_end_time_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.base_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.forces_local = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)

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

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]
        self.robot_base_idx = [index for index, body_name in enumerate(body_names) if body_name == "base_link"][0]
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
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

        self.mass_params_tensor = torch.zeros(self.num_envs, 22, dtype=torch.float, device=self.device, requires_grad=False)
        
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
                    torch_rand_float(self.cfg.domain_rand.arm_motor_strength_range[0], self.cfg.domain_rand.arm_motor_strength_range[1], (self.num_envs, 5), device=self.device)
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
        # Gripper 
        self.push_interval_gripper_cmd_min = np.ceil(self.cfg.commands.push_gripper_interval_s_cmd[0] / self.dt)
        self.push_interval_gripper_cmd_max = np.ceil(self.cfg.commands.push_gripper_interval_s_cmd[1] / self.dt)
        self.push_interval_gripper_ext_min = np.ceil(self.cfg.commands.push_gripper_interval_s_ext[0] / self.dt)
        self.push_interval_gripper_ext_max = np.ceil(self.cfg.commands.push_gripper_interval_s_ext[1] / self.dt)
        self.push_duration_gripper_cmd_min = np.ceil(self.cfg.commands.push_gripper_duration_s_cmd[0] / self.dt)
        self.push_duration_gripper_cmd_max = np.ceil(self.cfg.commands.push_gripper_duration_s_cmd[1] / self.dt)
        self.push_duration_gripper_ext_min = np.ceil(self.cfg.commands.push_gripper_duration_s_ext[0] / self.dt)
        self.push_duration_gripper_ext_max = np.ceil(self.cfg.commands.push_gripper_duration_s_ext[1] / self.dt)
        self.settling_time_force_gripper = np.ceil(self.cfg.commands.settling_time_force_gripper_s / self.dt)
        # Base 
        self.push_interval_base_cmd_min = np.ceil(self.cfg.commands.push_base_interval_s_cmd[0] / self.dt)
        self.push_interval_base_cmd_max = np.ceil(self.cfg.commands.push_base_interval_s_cmd[1] / self.dt)
        self.push_interval_base_ext_min = np.ceil(self.cfg.commands.push_base_interval_s_ext[0] / self.dt)
        self.push_interval_base_ext_max = np.ceil(self.cfg.commands.push_base_interval_s_ext[1] / self.dt)
        self.push_duration_base_cmd_min = np.ceil(self.cfg.commands.push_base_duration_s_cmd[0] / self.dt)
        self.push_duration_base_cmd_max = np.ceil(self.cfg.commands.push_base_duration_s_cmd[1] / self.dt)
        self.push_duration_base_ext_min = np.ceil(self.cfg.commands.push_base_duration_s_ext[0] / self.dt)
        self.push_duration_base_ext_max = np.ceil(self.cfg.commands.push_base_duration_s_ext[1] / self.dt)
        self.settling_time_force_base = np.ceil(self.cfg.commands.settling_time_force_base_s / self.dt)
        
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
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_x")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_y")

    
    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)

        if evt.value <= 0:
            return
        
        if evt.action == "stop_x":
            self.commands[:, 0] = 0
        elif evt.action == "forward":
            self.commands[:, 0] += 0.1
        elif evt.action == "reverse":
            self.commands[:, 0] -= 0.1
        
        elif evt.action == "stop_y":
            self.commands[:, 1] = 0
        elif evt.action == "left":
            self.commands[:, 1] += 0.1
        elif evt.action == "right":
            self.commands[:, 1] -= 0.1

        if evt.action == "stop_angular":
            self.commands[:, 2] = 0
        if evt.action == "turn_left":
            self.commands[:, 2] += 0.1
        elif evt.action == "turn_right":
            self.commands[:, 2] -= 0.1
        # print(evt.action, self.commands)

        

    #------------ reward functions----------------
    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - self.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew
    
    def _reward_tracking_ee_force_world(self):
        forces_global = self.forces[:, self.gripper_idx, 0:3]
        forces_cmd = self.current_Fxyz_gripper_cmd
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_offset = (forces_global + forces_cmd_global)
        curr_ee_goal_cart_world_offset = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
       
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - curr_ee_goal_cart_world_offset), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
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
        return torch.sum(torch.square(self.torques)[:, 12:17], dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, :12], dim=1)

    def _reward_dof_vel_arm(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, 12:17], dim=1)
        
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, :12], dim=1)
        return energy
    
    def _reward_energy_square_stand(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, :12], dim=1)

        walking_flag = self.get_walking_cmd_mask()
        energy[walking_flag] = 0
        return energy
    
    def _reward_energy_square_arm(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel)[:, 12:17], dim=1)
        return energy
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, :12] / self.dt), dim=1)
    
    def _reward_dof_acc_arm(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, 12:17] / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, :12], dim=1)
    
    def _reward_action_rate_arm(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 12:17], dim=1)
    
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
        return torch.sum(out_of_limits[:, :17], dim=1)

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
        return torch.sum((torch.abs(self.torques[:,12:17]) - self.torque_limits[12:17] * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_force_world(self):
        forces_global_base = self.forces[:, self.robot_base_idx, 0:3]
        forces_local_base = quat_rotate_inverse(self.base_yaw_quat, forces_global_base).view(self.num_envs, 3)
    
        forces_cmd_local = self.current_Fxyz_base_cmd
        forces_offset = (forces_local_base + forces_cmd_local)
        base_lin_vel_offset = (forces_offset / self.base_force_kds)[:, :2] + self.commands[:, :2]


        non_stop_sign = (torch.abs(base_lin_vel_offset[:, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(base_lin_vel_offset[:, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_yaw_clip)
        base_lin_vel_offset[:, :3] *= non_stop_sign.unsqueeze(1)

        lin_vel_error = torch.sum(torch.square(base_lin_vel_offset - self.base_lin_vel[:, :2]), dim=1)
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
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
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
        rew = torch.sum(torch.square(self.torques - self.last_torques)[:, 12:17], dim=1)
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
    
    def _reward_ee_force_x(self):
        
        xy_forces_global = self.forces[:, self.gripper_idx, 0:3]
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)
        
        force_magn_meas = (xy_forces_local[:, 0]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_X]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff*force_magn_error)
    
    def _reward_ee_force_y(self):
        
        xy_forces_global = self.forces[:, self.gripper_idx, 0:3]
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)

        force_magn_meas = (xy_forces_local[:, 1]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_Y]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff*force_magn_error)
    
    def _reward_ee_force_z(self):

        force_magn_meas = (self.forces[:, self.gripper_idx, 2]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_Z]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff*force_magn_error)

    def _reward_ee_force_magnitude_x_pen(self):
        
        force_magn_meas = torch.abs(self.forces[:, self.gripper_idx, 0]).view(self.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        return force_magn_error
    
    def _reward_ee_force_magnitude_y_pen(self):
       
        force_magn_meas = torch.abs(self.forces[:, self.gripper_idx, 1]).view(self.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        return force_magn_error