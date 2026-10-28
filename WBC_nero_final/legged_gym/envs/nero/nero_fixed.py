"""Seven-action fixed-base arm task using the project's PPO observation API."""
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_mul, quat_from_euler_xyz
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.nero.nero_actuator import NeroActuator
from legged_gym.envs.a2.legged_robot_a2nero_pos_force import LeggedRobot_a2nero_pos_force as WholeBody
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.isaacgym_utils import sphere2cart, cart2sphere


class NeroFixed(BaseTask):
    # Cartesian displacements in the mounting-centered task frame, in metres.
    TELEOP_KEYS = {
        'I': (0.0, 0.0, 0.05), 'K': (0.0, 0.0, -0.05),
        'LEFT': (0.0, 0.05, 0.0), 'RIGHT': (0.0, -0.05, 0.0),
        'UP': (0.05, 0.0, 0.0), 'DOWN': (-0.05, 0.0, 0.0),
    }
    # Share the exact spherical sampler, path collision check, heading convention
    # and tool orientation rewards, without inheriting any leg-indexed code.
    _resample_ee_goal_sphere_once = WholeBody._resample_ee_goal_sphere_once
    _resample_ee_goal_orn_once = WholeBody._resample_ee_goal_orn_once
    collision_check = WholeBody.collision_check
    get_ee_goal_heading_quat = WholeBody.get_ee_goal_heading_quat
    get_ee_goal_frame_quat = WholeBody.get_ee_goal_frame_quat
    get_ee_orientation_rpy = WholeBody.get_ee_orientation_rpy
    _reward_tracking_ee_world = WholeBody._reward_tracking_ee_world
    _reward_tracking_ee_orn = WholeBody._reward_tracking_ee_orn
    _enforce_gripper_symmetry = WholeBody._enforce_gripper_symmetry

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.pd_backend = getattr(cfg.control, 'pd_backend', 'explicit')
        if self.pd_backend not in ('explicit', 'physx_position'):
            raise ValueError('Unsupported Nero PD backend: ' + self.pd_backend)
        self.native_pd = self.pd_backend == 'physx_position'
        if self.native_pd and physics_engine != gymapi.SIM_PHYSX:
            raise ValueError('Native Nero position drives require PhysX')
        self.robust = bool(getattr(getattr(cfg, "actuator", None), "enabled", False))
        self.dt = cfg.control.decimation * sim_params.dt
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_single_obs = cfg.env.num_single_obs
        self.num_pred_obs = cfg.env.num_pred_obs
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, 9, 2)
        self.dof_pos, self.dof_vel = self.dof_state[..., 0], self.dof_state[..., 1]
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(self.num_envs, 13)
        self.rigid_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, -1, 13)
        self.default_dof_pos = self.tensor([cfg.init_state.default_joint_angles[n] for n in self.dof_names])
        self.lower = self.tensor(self.dof_props['lower'].copy())
        self.upper = self.tensor(self.dof_props['upper'].copy())
        self.torque_limits = self.tensor(self.dof_props['effort'].copy())
        self.kp = self.tensor([cfg.control.stiffness[n] for n in self.dof_names[:7]] + [cfg.control.gripper_stiffness] * 2)
        self.kd = self.tensor([cfg.control.damping[n] for n in self.dof_names[:7]] + [cfg.control.gripper_damping] * 2)
        self.gripper_joint1_idx, self.gripper_joint2_idx = 7, 8
        self.actions = self.zeros(7)
        self.last_actions = self.zeros(7)
        self.candidate_actions = self.zeros(7)
        self.last_candidate_actions = self.zeros(7)
        self.last_candidate_delta = self.zeros(7)
        self.last_sent_delta = self.zeros(7)
        self.incremental = getattr(cfg.control, "action_mode", "absolute") == "incremental_v4"
        if getattr(cfg.control, "action_mode", "absolute") not in ("absolute", "incremental_v4"):
            raise ValueError("Unsupported Nero action mode")
        if self.incremental and not (0 < cfg.control.target_delta_limit < cfg.actuator.max_target_step):
            raise ValueError("Incremental target limit must be below protective slew limit")
        self.last_vel = self.zeros(7)
        self.torques = self.zeros(9)
        if self.native_pd:
            self.drive_positions = self.default_dof_pos.repeat(self.num_envs, 1)
            self.drive_velocities = self.zeros(9)
            self.dof_force = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).view(self.num_envs, 9)
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.drive_velocities))
        self.motor_strength = torch.ones(self.num_envs, 7, device=self.device)
        self.base_yaw_quat = self.tensor([0, 0, 0, 1]).repeat(self.num_envs, 1)
        self.ee_link_to_goal_frame_quat = self.tensor(cfg.asset.ee_orientation_body_to_goal_frame_quat).repeat(self.num_envs, 1)
        self.ee_offset = self.tensor(cfg.asset.ee_position_body_offset).repeat(self.num_envs, 1)
        self.orn_error_scale = self.tensor(cfg.goal_ee.orn_error_scale)
        self.goal_ee_ranges = class_to_dict(cfg.goal_ee.ranges)
        self.num_collision_check_samples = cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        self.collision_upper_limits = self.tensor(cfg.goal_ee.collision_upper_limits)
        self.collision_lower_limits = self.tensor(cfg.goal_ee.collision_lower_limits)
        self.underground_limit = cfg.goal_ee.underground_limit
        for name in ('ee_start_sphere', 'ee_goal_sphere', 'curr_ee_goal_sphere', 'curr_ee_goal_cart',
                     'ee_start_orn_rpy', 'ee_goal_orn_rpy', 'curr_ee_goal_orn_rpy',
                     'last_target_pos', 'last_target_vel'):
            setattr(self, name, self.zeros(3))
        self.goal_timer = self.zeros()
        self.traj_timesteps = self.zeros()
        self.traj_total_timesteps = self.zeros()
        self.obs_history = self.zeros(cfg.env.frame_stack, self.num_single_obs)
        self.critic_history = self.zeros(cfg.env.c_frame_stack, cfg.env.single_num_privileged_obs)
        self.obs_pred = self.zeros(self.num_pred_obs)
        self.episode_sums = {k: self.zeros() for k in class_to_dict(cfg.rewards.scales)}
        self.metric_sums = {k: self.zeros() for k in (
            'position_error_m', 'candidate_delta_rms_rad', 'adapter_gap_rms_rad',
            'slew_fraction', 'hold_error_m', 'hold_speed_rms_rad_s',
            'sent_speed_rms_rad_s', 'sent_acceleration_rms_rad_s2', 'sent_reversal_fraction')}
        self.hold_samples = self.zeros()
        self.metric_samples = self.zeros()
        self.reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._init_fk()
        self.actuator = NeroActuator(self.num_envs, sim_params.dt, self.dt, cfg.actuator,
            cfg.domain_rand.randomize_actuator, self.default_dof_pos, self.lower, self.upper,
            pd_backend=self.pd_backend) if self.robust else None
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_goal()
        self.compute_observations()
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*cfg.viewer.pos), gymapi.Vec3(*cfg.viewer.lookat))

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float, device=self.device)

    def zeros(self, *shape):
        return torch.zeros((self.num_envs,) + shape, device=self.device)

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane)
        self.asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        options = gymapi.AssetOptions()
        for key in ('fix_base_link', 'collapse_fixed_joints', 'flip_visual_attachments', 'armature',
                    'disable_gravity', 'angular_damping', 'linear_damping', 'thickness',
                    'replace_cylinder_with_capsule', 'max_angular_velocity', 'max_linear_velocity'):
            setattr(options, key, getattr(self.cfg.asset, key))
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS if self.native_pd else gymapi.DOF_MODE_EFFORT
        options.use_mesh_materials = True
        asset = self.gym.load_asset(self.sim, os.path.dirname(self.asset_path), os.path.basename(self.asset_path), options)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        expected = [f'nero_joint{i}' for i in range(1, 8)] + ['gripper_joint1', 'gripper_joint2']
        if self.dof_names != expected:
            raise ValueError(f'Nero DOF order mismatch: {self.dof_names}')
        self.dof_props = self.gym.get_asset_dof_properties(asset)
        self.dof_props['driveMode'].fill(options.default_dof_drive_mode)
        self.dof_props['stiffness'].fill(0)
        self.dof_props['damping'].fill(0)
        if self.native_pd:
            self.dof_props['stiffness'] = [self.cfg.control.stiffness[n] for n in expected[:7]] + [self.cfg.control.gripper_stiffness] * 2
            self.dof_props['damping'] = [self.cfg.control.damping[n] for n in expected[:7]] + [self.cfg.control.gripper_damping] * 2
        if self.robust or getattr(self.cfg.actuator, 'force_reference_limits', False):
            # Override URDF metadata AND the limits used for applied torque.
            self.dof_props['effort'][:7] = self.cfg.actuator.torque_limits
            self.dof_props['velocity'][:7] = self.cfg.actuator.velocity_limits
        body_names = self.gym.get_asset_rigid_body_names(asset)
        self.ee_body_idx = body_names.index('nero_link7')
        self.envs, self.actors = [], []
        spacing = self.cfg.env.env_spacing
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), int(math.sqrt(self.num_envs)))
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
            pose.r = gymapi.Quat(*self.cfg.init_state.rot)
            actor = self.gym.create_actor(env, asset, pose, 'nero', i, self.cfg.asset.self_collisions, 0)
            self.gym.set_actor_dof_properties(env, actor, self.dof_props)
            if self.native_pd:
                self.gym.enable_actor_dof_force_sensors(env, actor)
            randomize_mass = self.robust and self.cfg.domain_rand.randomize_link_mass
            if randomize_mass or self.cfg.domain_rand.randomize_gripper_mass:
                props = self.gym.get_actor_rigid_body_properties(env, actor)
                if randomize_mass:
                    for prop in props:
                        scale = np.random.uniform(*self.cfg.domain_rand.link_mass_scale_range)
                        prop.mass *= scale
                        for row in (prop.inertia.x, prop.inertia.y, prop.inertia.z):
                            row.x *= scale; row.y *= scale; row.z *= scale
                        prop.invMass = 1.0 / prop.mass if prop.mass > 0 else 0.0
                if self.cfg.domain_rand.randomize_gripper_mass:
                    prop = props[self.ee_body_idx]
                    scale = 1 + np.random.uniform(*self.cfg.domain_rand.gripper_added_mass_range) / prop.mass
                    prop.mass *= scale
                    for row in (prop.inertia.x, prop.inertia.y, prop.inertia.z):
                        row.x *= scale; row.y *= scale; row.z *= scale
                    prop.invMass = 1.0 / prop.mass
                # Preserve explicitly scaled positive inertia rather than deriving it from meshes.
                self.gym.set_actor_rigid_body_properties(env, actor, props, recomputeInertia=False)
            self.envs.append(env)
            self.actors.append(actor)

    def _init_fk(self):
        """FK also gives valid tool observations immediately after indexed reset."""
        urdf = ET.parse(self.asset_path).getroot()
        self.fk_origins = []
        for name in self.dof_names[:7]:
            joint = urdf.find(f"joint[@name='{name}']")
            xyz = self.tensor([float(x) for x in joint.find('origin').get('xyz').split()]).repeat(self.num_envs, 1)
            rpy = self.tensor([float(x) for x in joint.find('origin').get('rpy').split()])
            quat = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3]).repeat(self.num_envs, 1)
            axis = self.tensor([float(x) for x in joint.find('axis').get('xyz').split()])
            self.fk_origins.append((xyz, quat, axis))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.mount_pos = self.root_states[:, :3].clone()
        self.mount_quat = self.root_states[:, 3:7].clone()

    def _update_ee_state(self):
        pos, orn = self.mount_pos.clone(), self.mount_quat.clone()
        for i, (xyz, origin_quat, axis) in enumerate(self.fk_origins):
            pos = pos + quat_apply(orn, xyz)
            half = self.dof_pos[:, i:i+1] * 0.5
            rotation = torch.cat((torch.sin(half) * axis, torch.cos(half)), -1)
            orn = quat_mul(quat_mul(orn, origin_quat), rotation)
        self.ee_orn = orn
        self.ee_pos = pos + quat_apply(orn, self.ee_offset)
        body = self.rigid_state[:, self.ee_body_idx]
        self.ee_vel = body[:, 7:10] + torch.cross(body[:, 10:13], quat_apply(orn, self.ee_offset), dim=-1)
        self.ee_vel[self.reset_mask] = 0

    def get_ee_goal_spherical_center(self):
        return self.mount_pos

    def subscribe_viewer_keyboard_events(self):
        super().subscribe_viewer_keyboard_events()
        for key in self.TELEOP_KEYS:
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, getattr(gymapi, 'KEY_' + key), 'nero_move_' + key)

    def handle_viewer_action_event(self, evt):
        if evt.action.startswith('nero_move_'):
            if not self.cfg.env.teleop_mode:
                return
            key = evt.action[len('nero_move_'):]
            if key not in self.TELEOP_KEYS:
                return
            if not hasattr(self, '_teleop_pressed'):
                self._teleop_pressed = set()
            if evt.value <= 0:
                self._teleop_pressed.discard(key)
            elif key not in self._teleop_pressed:
                self._teleop_pressed.add(key)
                self.move_ee_goal_cart(self.TELEOP_KEYS[key])
            return
        super().handle_viewer_action_event(evt)

    def _clip_goal_to_sampling_range(self, sphere):
        limits = self.tensor([self.goal_ee_ranges[k] for k in ('pos_l', 'pos_p', 'pos_y')])
        return torch.max(torch.min(sphere, limits[:, 1]), limits[:, 0])

    def move_ee_goal_cart(self, displacement):
        """Move the endpoint within the training spherical sampling bounds.

        Consecutive presses accumulate on the endpoint, including while moving.
        A boundary may shorten the requested 5 cm displacement.
        """
        if not self.cfg.env.teleop_mode:
            return
        endpoint = sphere2cart(self.ee_goal_sphere) + self.tensor(displacement)
        if not torch.isfinite(endpoint).all() or (endpoint.norm(dim=-1) < 1.e-6).any():
            print('Ignored target at the spherical-coordinate origin or non-finite target.')
            return
        self.ee_start_sphere.copy_(self.curr_ee_goal_sphere)
        self.ee_start_orn_rpy.copy_(self.curr_ee_goal_orn_rpy)
        requested_sphere = cart2sphere(endpoint)
        self.ee_goal_sphere.copy_(self._clip_goal_to_sampling_range(requested_sphere))
        limited = not torch.allclose(self.ee_goal_sphere, requested_sphere, atol=1.e-7, rtol=0)
        endpoint = sphere2cart(self.ee_goal_sphere)
        self.goal_timer.zero_()
        self.traj_timesteps.uniform_(*self.cfg.goal_ee.traj_time).div_(self.dt)
        self.traj_total_timesteps.fill_(float('inf'))
        print('Nero target xyz (m):', endpoint[0].tolist(),
              'interpolation (s):', round(self.traj_timesteps[0].item() * self.dt, 2),
              '(clipped to sampling bounds)' if limited else '')

    def _resample_goal(self, ids, initial=False):
        if not len(ids) or (self.cfg.env.teleop_mode and not initial):
            return
        if initial:
            self.ee_start_sphere[ids] = self.tensor(self.cfg.goal_ee.ranges.init_pos_start)
            self.ee_goal_sphere[ids] = self.tensor(self.cfg.goal_ee.ranges.init_pos_end)
            self.ee_start_orn_rpy[ids] = self.tensor(self.cfg.goal_ee.ranges.init_orn_start)
            self.ee_goal_orn_rpy[ids] = self.tensor(self.cfg.goal_ee.ranges.init_orn_end)
            if self.cfg.env.teleop_mode:
                self.ee_start_sphere[ids] = self._clip_goal_to_sampling_range(self.ee_start_sphere[ids])
                self.ee_goal_sphere[ids] = self._clip_goal_to_sampling_range(self.ee_goal_sphere[ids])
        else:
            self.ee_start_sphere[ids] = self.ee_goal_sphere[ids]
            self.ee_start_orn_rpy[ids] = self.ee_goal_orn_rpy[ids]
            self._resample_ee_goal_orn_once(ids)
            pending = ids
            for _ in range(10):
                self._resample_ee_goal_sphere_once(pending)
                pending = pending[self.collision_check(pending)]
                if not len(pending):
                    break
            # Never accept an invalid path when rejection sampling is exhausted.
            self.ee_goal_sphere[pending] = self.ee_start_sphere[pending]
        self.goal_timer[ids] = 0
        self.traj_timesteps[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.goal_ee.traj_time) / self.dt
        self.traj_total_timesteps[ids] = self.traj_timesteps[ids] + torch.empty(len(ids), device=self.device).uniform_(*self.cfg.goal_ee.hold_time) / self.dt

    def _update_goal(self):
        t = (self.goal_timer / self.traj_timesteps).clamp(0, 1).unsqueeze(-1)
        self.curr_ee_goal_sphere.copy_(torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t))
        self.curr_ee_goal_orn_rpy.copy_(torch.lerp(self.ee_start_orn_rpy, self.ee_goal_orn_rpy, t))
        self.curr_ee_goal_cart.copy_(sphere2cart(self.curr_ee_goal_sphere))
        self.curr_ee_goal_cart_world = self.mount_pos + self.curr_ee_goal_cart

    def _reset_native_drive_properties(self, ids):
        """Episode-constant native drive DR, uploaded only when episodes reset."""
        if self.actuator is not None:
            a = self.actuator
            stiffness = a.kp_scale[ids] * self.kp[:7]
            damping = a.kd_scale[ids] * self.kd[:7] + a.viscous[ids]
            friction = a.sample(self.cfg.actuator.native_joint_friction, len(ids), 0.)
            values = torch.stack((stiffness, damping, a.torque_cap[ids], a.speed_cap[ids], friction), 1).cpu().numpy()
        else:
            values = None
        for row, i in enumerate(ids.cpu().tolist()):
            props = self.dof_props.copy()
            if values is not None:
                for col, key in enumerate(('stiffness', 'damping', 'effort', 'velocity', 'friction')):
                    props[key][:7] = values[row, col]
            elif self.cfg.domain_rand.randomize_motor:
                strength = self.motor_strength[i].cpu().numpy()
                props['stiffness'][:7] *= strength
                props['damping'][:7] *= strength
            self.gym.set_actor_dof_properties(self.envs[i], self.actors[i], props)

    def reset_idx(self, ids):
        if not len(ids):
            return
        self.dof_pos[ids] = self.default_dof_pos
        self.dof_vel[ids] = 0
        actors = ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(actors), len(ids))
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.candidate_actions[ids] = 0
        self.last_candidate_actions[ids] = 0
        self.last_candidate_delta[ids] = 0
        self.last_sent_delta[ids] = 0
        for value in self.metric_sums.values():
            value[ids] = 0
        self.hold_samples[ids] = 0
        self.metric_samples[ids] = 0
        if self.actuator is not None:
            self.actuator.reset(ids, self.dof_pos[ids, :7], self.dof_vel[ids, :7])
        self.last_vel[ids] = 0
        self.obs_history[ids] = 0
        self.critic_history[ids] = 0
        self.episode_length_buf[ids] = 0
        self.reset_mask[ids] = True
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength[ids] = torch.empty(len(ids), 7, device=self.device).uniform_(*self.cfg.domain_rand.arm_motor_strength_range)
        if self.native_pd:
            self._reset_native_drive_properties(ids)
            self.drive_positions[ids] = self.default_dof_pos
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.drive_positions),
                gymtorch.unwrap_tensor(actors), len(ids))
            command_delay = getattr(self, 'command_delay', None)
            if command_delay is not None:
                command_delay.reset(ids, self.drive_positions[ids, :7])
        self._resample_goal(ids, initial=True)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_goal()
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        self.extras = {}
        self.actions.copy_(actions.to(self.device).clamp(-self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions))
        target = self.default_dof_pos.repeat(self.num_envs, 1)
        if self.incremental:
            previous = (self.actuator.host_target if self.actuator is not None else
                        self.default_dof_pos[:7] + self.cfg.control.action_scale * self.last_actions)
            target[:, :7] = previous + self.cfg.control.target_delta_limit * torch.tanh(self.actions)
        else:
            target[:, :7] += self.cfg.control.action_scale * self.actions
        target = torch.max(torch.min(target, self.upper), self.lower)
        self.candidate_actions.copy_((target[:, :7] - self.default_dof_pos[:7]) / self.cfg.control.action_scale)
        if self.actuator is not None:
            target[:, :7] = self.actuator.submit(target[:, :7], incremental=self.incremental)
        self.actions.copy_((target[:, :7] - self.default_dof_pos[:7]) / self.cfg.control.action_scale)
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.native_pd:
                self.drive_positions.copy_(target)
                if self.actuator is not None:
                    self.drive_positions[:, :7] = self.actuator.advance_target()
                command_delay = getattr(self, 'command_delay', None)
                if command_delay is not None:
                    self.drive_positions[:, :7] = command_delay.advance(target[:, :7])
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.drive_positions))
            else:
                self.torques = self.kp * (target - self.dof_pos) - self.kd * self.dof_vel
                if self.actuator is not None:
                    self.torques[:, :7] = self.actuator.torque(self.dof_pos[:, :7], self.dof_vel[:, :7], self.kp[:7], self.kd[:7])
                else:
                    self.torques[:, :7] *= self.motor_strength
                self.torques = torch.max(torch.min(self.torques, self.torque_limits), -self.torque_limits)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if self.native_pd:
                self.gym.refresh_dof_force_tensor(self.sim)
                # Native generalized force measurement includes constraint forces;
                # it is not the clipped motor command used by the legacy backend.
                self.torques.copy_(self.dof_force)
            self._enforce_gripper_symmetry()
            if self.actuator is not None:
                self.actuator.feedback(self.dof_pos[:, :7], self.dof_vel[:, :7])
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self._update_ee_state()
        # Reward the command seen by the policy before advancing to the next one.
        self.compute_reward()
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        if self.cfg.env.teleop_mode:
            self.time_out_buf.zero_()
        invalid = ~torch.isfinite(self.dof_state).all(dim=(1, 2))
        self.reset_buf = self.time_out_buf | invalid
        if hasattr(self, 'task_failure_buf'):
            self.reset_buf |= self.task_failure_buf
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.extras['time_outs'] = self.time_out_buf & ~invalid
        if hasattr(self, 'task_failure_buf'):
            self.extras['time_outs'] &= ~self.task_failure_buf
        self.last_sent_delta.copy_((self.actions - self.last_actions) * self.cfg.control.action_scale)
        self.last_actions.copy_(self.actions)
        self.last_candidate_delta.copy_(self.candidate_actions - self.last_candidate_actions)
        self.last_candidate_actions.copy_(self.candidate_actions)
        self.last_vel.copy_(self.dof_vel[:, :7])
        self.goal_timer += 1
        self._resample_goal((self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten())
        if len(ids):
            self.extras['episode'] = {k: v[ids].mean() / self.cfg.env.episode_length_s for k, v in self.episode_sums.items()}
            if hasattr(self, 'task_failure_buf'):
                self.extras['terminal_joint_positions'] = self.dof_pos[ids, :7].clone()
                self.extras['terminal_joint_velocities'] = self.dof_vel[ids, :7].clone()
                self.extras['episode']['metric_episode_duration_s'] = (self.episode_length_buf[ids]*self.dt).float().mean()
                self.extras['episode']['metric_cycle_completion'] = self.extras['time_outs'][ids].float().mean()
                self.extras['episode']['metric_failure_episode_fraction'] = (
                    self.task_failure_buf[ids] | invalid[ids]).float().mean()
            for key, value in self.metric_sums.items():
                count = self.hold_samples[ids] if key.startswith('hold_') else self.metric_samples[ids]
                valid = count > 0
                # Average only episodes that actually visited the holding phase.
                self.extras['episode']['metric_' + key] = (
                    (value[ids] / count.clamp(min=1)) * valid).sum() / valid.sum().clamp(min=1)
            self.extras['episode']['metric_hold_fraction'] = (
                self.hold_samples[ids] / self.metric_samples[ids].clamp(min=1)).mean()
            for value in self.episode_sums.values():
                value[ids] = 0
            self.reset_idx(ids)
        self._update_goal()
        self.compute_observations()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def compute_reward(self):
        soft_lower = (self.lower + self.upper) * 0.5 - (self.upper - self.lower) * 0.45
        soft_upper = (self.lower + self.upper) * 0.5 + (self.upper - self.lower) * 0.45
        error = (self.ee_pos - self.curr_ee_goal_cart_world).norm(dim=-1)
        candidate_delta = self.candidate_actions - self.last_candidate_actions
        holding = self.goal_timer >= self.traj_timesteps
        floor = getattr(self.cfg.rewards, "hold_velocity_floor", 0.0)
        hold_weight = holding.float() * (floor + (1 - floor) * torch.exp(
            -(error / getattr(self.cfg.rewards, "hold_error_radius", 0.10)).square()))
        sent_delta = (self.actions - self.last_actions) * self.cfg.control.action_scale
        sent_acceleration = sent_delta - self.last_sent_delta
        delta_norm = getattr(self.cfg.control, "target_delta_limit", 0.012)
        terms = {
            'tracking_ee_world': self._reward_tracking_ee_world(),
            'ee_position_error': error,
            'action_rate': (self.actions - self.last_actions).square().sum(-1),
            'target_rate': (sent_delta / delta_norm).square().sum(-1),
            'target_acceleration': (sent_acceleration / delta_norm).square().sum(-1),
            'candidate_rate': candidate_delta.square().sum(-1),
            # Second difference per 20 ms actor tick, in normalized action units.
            'candidate_acceleration': (candidate_delta - self.last_candidate_delta).square().sum(-1),
            'hold_dof_vel': hold_weight * self.dof_vel[:, :7].square().sum(-1),
            'adapter_gap': (self.candidate_actions - self.actions).square().sum(-1),
            'velocity_limits': ((self.dof_vel[:, :7].abs() - self.actuator.speed_cap).clamp(min=0).square().sum(-1)
                                if self.actuator is not None else self.zeros()),
            'dof_vel': self.dof_vel[:, :7].square().sum(-1),
            'dof_acc': ((self.dof_vel[:, :7] - self.last_vel) / self.dt).square().sum(-1),
            'torques': self.torques[:, :7].square().sum(-1),
            'dof_pos_limits': ((soft_lower[:7] - self.dof_pos[:, :7]).clamp(min=0) + (self.dof_pos[:, :7] - soft_upper[:7]).clamp(min=0)).sum(-1),
        }
        if getattr(self.cfg.rewards.scales, 'tracking_ee_orn', 0.):
            terms['tracking_ee_orn'] = self._reward_tracking_ee_orn()
        self.rew_buf.zero_()
        for key, value in terms.items():
            contribution = value * getattr(self.cfg.rewards.scales, key, 0.) * self.dt
            self.rew_buf += contribution
            if key in self.episode_sums:
                self.episode_sums[key] += contribution
        scale = self.cfg.control.action_scale
        # PPO initially randomizes episode_length_buf; count real samples here.
        self.metric_samples += 1
        self.metric_sums['sent_speed_rms_rad_s'] += sent_delta.square().mean(-1).sqrt() / self.dt
        self.metric_sums['sent_acceleration_rms_rad_s2'] += sent_acceleration.square().mean(-1).sqrt() / self.dt**2
        self.metric_sums['sent_reversal_fraction'] += ((sent_delta * self.last_sent_delta < 0) &
            (sent_delta.abs() > 0.001) & (self.last_sent_delta.abs() > 0.001)).float().mean(-1)
        self.metric_sums['position_error_m'] += error
        self.metric_sums['candidate_delta_rms_rad'] += candidate_delta.square().mean(-1).sqrt() * scale
        self.metric_sums['adapter_gap_rms_rad'] += (self.candidate_actions - self.actions).square().mean(-1).sqrt() * scale
        slew = (self.actions - self.last_actions).abs() * scale
        limit = self.cfg.actuator.max_target_step if self.actuator is not None else float('inf')
        if getattr(self, 'target_step_limit', None) is not None:
            limit = self.target_step_limit
        self.metric_sums['slew_fraction'] += (slew >= limit - 1.e-6).float().mean(-1)
        self.hold_samples += holding.float()
        self.metric_sums['hold_error_m'] += error * holding
        self.metric_sums['hold_speed_rms_rad_s'] += self.dof_vel[:, :7].square().mean(-1).sqrt() * holding

    def compute_observations(self):
        self._update_ee_state()
        # Observation target and EE state must use the same mount origin.
        target = self.curr_ee_goal_cart_world - self.mount_pos
        vel = (target - self.last_target_pos) / self.dt
        vel[self.reset_mask] = 0
        acc = (vel - self.last_target_vel) / self.dt
        acc[self.reset_mask] = 0
        q = self.dof_pos[:, :7] - self.default_dof_pos[:7]
        dq = self.dof_vel[:, :7] * self.cfg.normalization.obs_scales.dof_vel
        commands_scale = self.tensor([0.5, 1.0, 1.3, 0.5, 0.5, 0.5])
        commands = torch.cat((self.curr_ee_goal_sphere, self.curr_ee_goal_orn_rpy), -1) * commands_scale
        clean = torch.cat((q, dq, self.actions, self.get_ee_orientation_rpy(), commands, target, vel, acc), -1)
        ee_local = self.ee_pos - self.mount_pos
        error = target - ee_local
        privileged = torch.cat((clean, ee_local, self.ee_vel, error, self.motor_strength - 1), -1)
        noisy = clean.clone()
        if self.actuator is not None:
            noisy[:, :7] = self.actuator.measured_q - self.default_dof_pos[:7]
            noisy[:, 7:14] = self.actuator.measured_dq * self.cfg.normalization.obs_scales.dof_vel
        if self.cfg.noise.add_noise:
            noisy[:, :7] += torch.randn_like(q) * self.cfg.noise.noise_scales.dof_pos * self.cfg.noise.noise_level
            noisy[:, 7:14] += torch.randn_like(dq) * self.cfg.noise.noise_scales.dof_vel * self.cfg.normalization.obs_scales.dof_vel * self.cfg.noise.noise_level
        if self.actuator is not None:
            # Actor FK must use the same delayed/noisy joint cohort, not true physics state.
            true_orn = self.ee_orn
            orn = self.mount_quat.clone()
            for i, (_, origin_quat, axis) in enumerate(self.fk_origins):
                half = (noisy[:, i:i+1] + self.default_dof_pos[i]) * 0.5
                rotation = torch.cat((torch.sin(half) * axis, torch.cos(half)), -1)
                orn = quat_mul(quat_mul(orn, origin_quat), rotation)
            self.ee_orn = orn
            noisy[:, 21:24] = self.get_ee_orientation_rpy()
            self.ee_orn = true_orn
        self.obs_history = torch.roll(self.obs_history, -1, dims=1)
        self.critic_history = torch.roll(self.critic_history, -1, dims=1)
        self.obs_history[:, -1] = noisy
        self.critic_history[:, -1] = privileged
        self.obs_buf = self.obs_history.flatten(1).clamp(-100, 100)
        self.privileged_obs_buf = self.critic_history.flatten(1).clamp(-100, 100)
        self.obs_pred = torch.cat((ee_local, self.ee_vel, error, self.zeros(18)), -1)
        self.last_target_pos.copy_(target)
        self.last_target_vel.copy_(vel)
        self.reset_mask.zero_()

    def get_observations(self):
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}
