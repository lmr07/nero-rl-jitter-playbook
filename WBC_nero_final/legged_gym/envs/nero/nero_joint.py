"""One movable DOF, six true fixed joints, native PhysX PD and lumped command lag."""
import math
import os
import tempfile

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from .joint_probe_utils import fixed_pose_urdf, joint_limits, TORQUE_LIMITS, VELOCITY_LIMITS
from nero_joint_contract import RATE_LIMITED_MODE, LEGACY_MODE, A2NERO_STAGE1_MODE, limit_position_target


class NeroJoint(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.j = cfg.env.joint - 1
        if not 0 <= self.j < 7:
            raise ValueError("joint must be 1..7")
        if abs(sim_params.dt - .005) > 1e-9 or cfg.control.decimation != 4 or sim_params.substeps != 1:
            raise ValueError("Joint probe requires 200 Hz simulation / 50 Hz actor")
        self.dt = .02
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self.num_single_obs = cfg.env.num_single_obs
        self.num_pred_obs = cfg.env.num_pred_obs
        self.asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        lo, hi, radius = joint_limits(self.asset_path)
        self.lower, self.upper, self.radius = float(lo[self.j]), float(hi[self.j]), float(radius[self.j])
        if cfg.control.action_mode not in (RATE_LIMITED_MODE, LEGACY_MODE, A2NERO_STAGE1_MODE):
            raise ValueError("Unknown single-joint action ABI")
        self.a2nero_stage1 = cfg.control.action_mode == A2NERO_STAGE1_MODE
        self.rate_limited = cfg.control.action_mode in (RATE_LIMITED_MODE, A2NERO_STAGE1_MODE)
        if self.rate_limited:
            if cfg.control.goal_range_fraction != 1. or not 0 < cfg.control.command_speed_fraction < 1:
                raise ValueError("v2 requires full URDF ranges and command speed strictly below vmax")
            sample_lo, sample_hi = lo, hi
        else:
            # Old checkpoints keep their original sampling/action ABI on replay.
            sample_lo, sample_hi = -radius, radius
        self.goal_lower, self.goal_upper = float(sample_lo[self.j]), float(sample_hi[self.j])
        self.default_q = float(cfg.init_state.default_joint_angles["nero_joint%d" % (self.j + 1)])
        self.command_delta = (cfg.control.command_speed_fraction * VELOCITY_LIMITS[self.j] * self.dt
                              if self.rate_limited else None)
        self.speed = .5 * VELOCITY_LIMITS[self.j]
        self.delay_steps = cfg.control.delay_s / sim_params.dt
        self.delay_integer = int(math.floor(self.delay_steps))
        self.delay_fraction = self.delay_steps - self.delay_integer
        self.asset_temp = tempfile.TemporaryDirectory(prefix="nero-joint-assets-")
        rng = np.random.RandomState(cfg.env.pose_seed)
        self.pose_bank = rng.uniform(sample_lo, sample_hi, (min(cfg.env.pose_bank_size, cfg.env.num_envs), 7))
        self.pose_bank[:, self.j] = 0
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.a2nero_stage1:
            self.motor_scales = torch.tensor(self._motor_scales, device=self.device, dtype=torch.float)
            self.payload_masses = torch.tensor(self._payload_masses, device=self.device, dtype=torch.float)
        self.state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, 1, 2)
        self.q, self.dq = self.state[:, 0, 0], self.state[:, 0, 1]
        self.forces = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).view(self.num_envs)
        self.actor_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.locked = torch.tensor(self.pose_bank[np.arange(self.num_envs) % len(self.pose_bank)],
                                   device=self.device, dtype=torch.float)
        self.actions = self.zeros()
        self.previous = self.zeros()
        self.last_target_delta = self.zeros()
        self.last_dq = self.zeros()
        self.command = self.zeros()
        self.issued_target = self.zeros()  # pre-delay position command, also represented by actions
        self.queue = self.zeros(max(self.delay_integer + 2, 3))
        self.reference = self.zeros()
        self.ref_vel = self.zeros()
        self.start = self.zeros()
        self.end = self.zeros()
        self.clock = self.zeros()
        self.travel = self.zeros()
        self.total = self.zeros()
        self.history = self.zeros(cfg.env.frame_stack, self.num_single_obs)
        self.error_sum = self.zeros()
        self.metric_count = self.zeros()
        self.smooth_metrics = {key: self.zeros() for key in (
            "hold_count", "hold_speed_squared", "target_acceleration_squared",
            "adapter_gap_squared", "command_reversal_fraction")}
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], gymapi.Vec3(2, -2, 2), gymapi.Vec3(0, 0, .8))

    def zeros(self, *shape):
        return torch.zeros((self.num_envs,) + shape, device=self.device)

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # Isolation experiment: no ground/contact forces; base remains fixed.
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        # Keep nero_link7 as its own rigid body in the A2nero-aligned profile so
        # payload randomization acts at the same physical link as A2nero.
        options.collapse_fixed_joints = not self.a2nero_stage1
        options.disable_gravity = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.armature = self.cfg.asset.armature
        options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        options.use_mesh_materials = True
        options.linear_damping = 0.
        options.angular_damping = 0.
        assets = []
        for k, pose in enumerate(self.pose_bank):
            path = os.path.join(self.asset_temp.name, "%d.urdf" % k)
            fixed_pose_urdf(self.asset_path, path, self.j + 1, pose)
            asset = self.gym.load_asset(self.sim, self.asset_temp.name, os.path.basename(path), options)
            if self.gym.get_asset_dof_names(asset) != ["nero_joint%d" % (self.j+1)]:
                raise RuntimeError("Inactive joints were not physically fixed")
            assets.append(asset)
        self.envs, self.actors = [], []
        self._motor_scales, self._payload_masses = [], []
        for k in range(self.num_envs):
            asset = assets[k % len(assets)]
            env = self.gym.create_env(self.sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), max(1, int(math.sqrt(self.num_envs))))
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
            pose.r = gymapi.Quat(*self.cfg.init_state.rot)
            actor = self.gym.create_actor(env, asset, pose, "nero_joint", k, 1, 0)
            props = self.gym.get_asset_dof_properties(asset)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            motor_scale = (np.random.uniform(*self.cfg.domain_rand.arm_motor_strength_range)
                           if self.a2nero_stage1 and self.cfg.domain_rand.randomize_motor else 1.)
            # Native position drive cannot multiply generated torque externally;
            # scaling both gains reproduces A2nero's motor-strength multiplier
            # before the unchanged physical effort cap.
            props["stiffness"].fill(self.cfg.control.kp * motor_scale)
            props["damping"].fill(self.cfg.control.kd * motor_scale)
            props["effort"].fill(TORQUE_LIMITS[self.j])
            props["velocity"].fill(VELOCITY_LIMITS[self.j])
            props["friction"].fill(0.)
            self.gym.set_actor_dof_properties(env, actor, props)
            payload_mass = 0.
            if self.a2nero_stage1 and (self.cfg.domain_rand.randomize_link_mass or
                                       self.cfg.domain_rand.randomize_gripper_mass):
                bodies = self.gym.get_actor_rigid_body_properties(env, actor)
                names = self.gym.get_actor_rigid_body_names(env, actor)
                if self.cfg.domain_rand.randomize_gripper_mass and "nero_link7" not in names:
                    raise RuntimeError("A2nero payload body nero_link7 missing after fixed-pose conversion")
                for body, name in zip(bodies, names):
                    if self.cfg.domain_rand.randomize_link_mass:
                        body.mass *= np.random.uniform(*self.cfg.domain_rand.link_mass_scale_range)
                    if self.cfg.domain_rand.randomize_gripper_mass and name == "nero_link7":
                        payload_mass = np.random.uniform(*self.cfg.domain_rand.gripper_added_mass_range)
                        body.mass += payload_mass
                self.gym.set_actor_rigid_body_properties(env, actor, bodies, recomputeInertia=True)
            self._motor_scales.append(motor_scale)
            self._payload_masses.append(payload_mass)
            self.gym.enable_actor_dof_force_sensors(env, actor)
            self.envs.append(env)
            self.actors.append(actor)

    def sample_goal(self, ids):
        self.start[ids] = self.reference[ids]
        self.end[ids] = torch.empty(len(ids), device=self.device).uniform_(self.goal_lower, self.goal_upper)
        # Whole policy ticks, so reference speed never exceeds 0.5*v_max.
        self.travel[ids] = torch.ceil((self.end[ids]-self.start[ids]).abs()/self.speed/self.dt).clamp(min=1)
        hold_range = self.cfg.env.hold_time if self.a2nero_stage1 else (4., 7.)
        hold = torch.empty(len(ids), device=self.device).uniform_(*hold_range)
        self.total[ids] = self.travel[ids] + torch.ceil(hold/self.dt)
        self.clock[ids] = 0
        self.ref_vel[ids] = (self.end[ids]-self.start[ids]) / (self.travel[ids]*self.dt)

    def reset_idx(self, ids):
        if not len(ids):
            return
        self.q[ids] = torch.empty(len(ids), device=self.device).uniform_(self.goal_lower, self.goal_upper)
        self.dq[ids] = 0
        indices = self.actor_ids[ids].contiguous()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.state),
                                            gymtorch.unwrap_tensor(indices), len(ids))
        self.command[ids] = self.q[ids]
        self.issued_target[ids] = self.q[ids]
        self.queue[ids] = self.q[ids, None]
        # Normalized absolute action whose target equals the reset position.
        self.actions[ids] = ((self.q[ids]-self.default_q)/self.cfg.control.action_scale if self.a2nero_stage1
                             else 2*(self.q[ids]-self.lower)/(self.upper-self.lower)-1)
        self.previous[ids] = self.actions[ids]
        self.last_target_delta[ids] = 0
        self.last_dq[ids] = 0
        self.reference[ids] = self.q[ids]
        self.sample_goal(ids)
        self.episode_length_buf[ids] = 0
        self.history[ids] = 0
        self.error_sum[ids] = 0
        self.metric_count[ids] = 0
        for value in self.smooth_metrics.values():
            value[ids] = 0

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        self.previous.copy_(self.actions)
        clip = self.cfg.control.action_clip if self.a2nero_stage1 else 1.
        self.actions = actions[:, 0].to(self.device).clamp(-clip, clip)
        target = (self.default_q + self.cfg.control.action_scale*self.actions if self.a2nero_stage1
                  else self.lower + (self.actions+1)*.5*(self.upper-self.lower))
        requested_action = self.actions.clone()
        previous_target = self.issued_target.clone()
        if self.rate_limited:
            target = limit_position_target(target, self.issued_target, self.lower, self.upper, self.command_delta)
            # The actor observes its actual issued target, not an unreachable raw request.
            self.actions = ((target-self.default_q)/self.cfg.control.action_scale if self.a2nero_stage1
                            else 2*(target-self.lower)/(self.upper-self.lower)-1)
        target_delta = target - previous_target
        self.issued_target.copy_(target)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.queue = torch.cat((target[:, None], self.queue[:, :-1]), dim=1)
            n, f = self.delay_integer, self.delay_fraction
            # Fractional-delay approximation at 5 ms resolution, not a second transport lag.
            self.command = ((1-f)*self.queue[:, n] + f*self.queue[:, n+1]).contiguous()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.command))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        if not torch.isfinite(self.state).all() or not torch.isfinite(self.forces).all():
            raise RuntimeError("Non-finite PhysX state/force in joint probe; aborting training")
        self.episode_length_buf += 1
        error = self.q-self.reference
        moving = self.clock < self.travel
        velocity_error = self.dq-self.ref_vel
        if self.a2nero_stage1:
            r = self.cfg.rewards
            span = self.upper-self.lower
            soft_lower = self.lower + (1-r.soft_dof_pos_limit)*span/2
            soft_upper = self.upper - (1-r.soft_dof_pos_limit)*span/2
            soft_violation = (soft_lower-self.q).clamp(min=0) + (self.q-soft_upper).clamp(min=0)
            self.rew_buf = self.dt * (
                r.tracking*torch.exp(-2*error.abs()/r.tracking_sigma)
                + r.position_error*error.abs()
                + r.action_rate*(self.actions-self.previous).square()
                + r.dof_vel*self.dq.square()
                + r.dof_acc*((self.dq-self.last_dq)/self.dt).square()
                + r.dof_pos_limits*soft_violation)
        else:
            self.rew_buf = self.dt * (
                4*torch.exp(-(error/.15)**2) - 2*error.square()
                - .04*velocity_error.square()
                - .05*(~moving)*self.dq.square()
                - .01*((self.actions-self.previous)/self.dt).square()
                - .00005*self.forces.square()
                - .2*(self.dq.abs()-VELOCITY_LIMITS[self.j]).clamp(min=0).square())
        if not self.a2nero_stage1 and getattr(self.cfg.rewards, "smooth_hold", False):
            # Preserve the old reward when replaying a pre-smoothing manifest.
            # Gap uses normalized absolute-position units, independent of joint span.
            gap = requested_action - self.actions
            acceleration = (target_delta - self.last_target_delta) / self.command_delta
            self.rew_buf -= self.dt * (
                (self.cfg.rewards.hold_velocity - .05) * (~moving) * self.dq.square()
                + self.cfg.rewards.target_acceleration * acceleration.square()
                + self.cfg.rewards.adapter_gap * gap.square())
        self.smooth_metrics["hold_count"] += (~moving).float()
        self.smooth_metrics["hold_speed_squared"] += (~moving) * self.dq.square()
        self.smooth_metrics["adapter_gap_squared"] += (requested_action - self.actions).square()
        step_scale = self.command_delta if self.rate_limited else self.speed * self.dt
        self.smooth_metrics["target_acceleration_squared"] += ((target_delta-self.last_target_delta)/step_scale).square()
        self.smooth_metrics["command_reversal_fraction"] += (
            (target_delta*self.last_target_delta < 0) &
            (target_delta.abs() > 1e-4) & (self.last_target_delta.abs() > 1e-4)).float()
        self.last_target_delta.copy_(target_delta)
        self.last_dq.copy_(self.dq)
        self.error_sum += error.abs()
        self.metric_count += 1
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        invalid = ~torch.isfinite(self.state).all(dim=2).all(dim=1)
        self.reset_buf = self.time_out_buf | invalid | (self.q < self.lower-.05) | (self.q > self.upper+.05)
        self.extras = {"time_outs": self.time_out_buf.clone()}
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.extras["episode"] = {"tracking_mae_rad": (self.error_sum[ids]/self.metric_count[ids]).mean()}
            for key, value in self.smooth_metrics.items():
                if key == "hold_count":
                    continue
                divisor = (self.smooth_metrics["hold_count"] if key == "hold_speed_squared"
                           else self.metric_count)
                self.extras["episode"]["metric_" + key] = (value[ids]/divisor[ids].clamp(min=1)).mean()
        self.clock += 1
        self.reference = torch.lerp(self.start, self.end, (self.clock/self.travel).clamp(max=1))
        self.ref_vel = torch.where(self.clock < self.travel, (self.end-self.start)/(self.travel*self.dt), self.zeros())
        self.sample_goal((self.clock >= self.total).nonzero(as_tuple=False).flatten())
        self.reset_idx(ids)
        self.compute_observations()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        if self.a2nero_stage1:
            # A2nero proprioception: default-relative q, 0.05*dq, sent action.
            # Reference and fixed-pose coordinates remain task-specific channels.
            q_obs = self.q - self.default_q
            dq_obs = self.dq * 0.05
            if self.cfg.noise.add_noise:
                noise = self.cfg.noise.noise_level
                # The A2nero Stage-1 implementation applies noise_level in the
                # scale vector and again when adding Gaussian noise.
                q_obs = q_obs + torch.randn_like(q_obs)*self.cfg.noise.noise_scales.dof_pos*noise**2
                dq_obs = dq_obs + torch.randn_like(dq_obs)*self.cfg.noise.noise_scales.dof_vel*0.05*noise**2
            frame = torch.cat((q_obs[:, None], dq_obs[:, None], self.reference[:, None]-self.default_q,
                               self.ref_vel[:, None]*0.05, self.actions[:, None], self.locked), dim=1)
            critic_frame = torch.cat(((self.q-self.default_q)[:, None], (self.dq*0.05)[:, None],
                                      self.reference[:, None]-self.default_q, self.ref_vel[:, None]*0.05,
                                      self.actions[:, None], self.locked), dim=1)
        else:
            frame = torch.cat((self.q[:, None], self.dq[:, None], self.reference[:, None],
                               self.ref_vel[:, None], self.actions[:, None], self.locked), dim=1)
            critic_frame = frame
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1] = frame
        self.obs_buf = self.history.flatten(1)
        critic_parts = [critic_frame, self.command[:, None], self.queue[:, :3]]
        if self.a2nero_stage1:
            critic_parts += [self.motor_scales[:, None]-1, self.payload_masses[:, None]]
        self.privileged_obs_buf = torch.cat(critic_parts, dim=1)
        self.obs_pred = torch.stack((self.q, self.dq, self.reference-self.q), dim=1)

    def get_observations(self):
        return dict(obs=self.obs_buf, privileged_obs=self.privileged_obs_buf, obs_pred=self.obs_pred)
