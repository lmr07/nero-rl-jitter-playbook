"""Seven-joint Nero, native PhysX PD, WBC home and phase-diverse fixed cycle."""
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import sphere2cart, cart2sphere
from .identified_position_delay import IdentifiedPositionDelay
from .nero_fixed import NeroFixed
from .nero_smooth_control import SmoothRequestHistory, robust_square, configure_smooth_start


class NeroSevenCycle(NeroFixed):
    def get_ee_goal_heading_quat(self):
        # Identical to atan2(sphere2cart(y), sphere2cart(x)) away from the
        # pole, but stable at the vertical q=0 home where x=y=0.
        yaw = self.curr_ee_goal_sphere[:, 2]
        zero = torch.zeros_like(yaw)
        return quat_from_euler_xyz(zero, zero, yaw)

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        if abs(sim_params.dt - 0.005) > 1e-9 or cfg.control.decimation != 4 or sim_params.substeps != 1:
            raise ValueError('Nero seven-cycle needs 200 Hz physics and 50 Hz policy')
        if cfg.control.pd_backend != 'physx_position' or cfg.control.action_mode != 'absolute':
            raise ValueError('Nero seven-cycle requires native PD and absolute position actions')
        self.behavior_version = getattr(cfg.control, 'behavior_version', 1)
        if self.behavior_version not in (1, 2, 3):
            raise ValueError('Unsupported seven-cycle behavior version')
        if self.behavior_version == 3:
            configure_smooth_start(cfg, LEGGED_GYM_ROOT_DIR)
        if self.behavior_version >= 2 and (
            len(cfg.control.target_speed_limits_rad_s) != 7 or min(cfg.control.target_speed_limits_rad_s) <= 0
        ):
            raise ValueError('The seven sent-target speed limits must be positive')
        self.command_delay = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 27/.02 can round slightly above 1350 in float32; each environment
        # resets after exactly one complete (possibly phase-shifted) cycle.
        self.max_episode_length = round(cfg.env.episode_length_s / self.dt)
        self.command_delay = IdentifiedPositionDelay(
            self.num_envs, cfg.actuator.identified_effective_delay_ms, sim_params.dt, self.device)
        self.command_delay.reset(torch.arange(self.num_envs, device=self.device), self.drive_positions[:, :7])
        self.target_step_limit = (self.tensor(cfg.control.target_speed_limits_rad_s) * self.dt
                                  if self.behavior_version >= 2 else None)
        self.previous_ee_velocity = self.zeros(3)
        self.previous_tangent_acc = self.zeros()
        if self.behavior_version == 3:
            self.smooth_history = SmoothRequestHistory(self.num_envs, self.device)
            self.pre_slew_request = self.drive_positions[:, :7].clone()
            self.bounded_request = self.pre_slew_request.clone()
            self.issued_request = self.pre_slew_request.clone()
            self.hold_joint_anchor = self.dof_pos[:, :7].clone()
            self.hold_anchor_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.hold_settle_time = self.zeros()
            self.hold_anchor_lost_time = self.zeros()
            self.task_failure_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.smooth_history.reset(torch.arange(self.num_envs, device=self.device), self.pre_slew_request)
            for name in ('request_speed_rms_rad_s', 'request_acceleration_rms_rad_s2',
                         'request_jerk_rms_rad_s3', 'request_gap_rms_rad', 'request_slew_fraction',
                         'hold_request_speed_rms_rad_s', 'hold_joint_drift_rms_rad',
                         'hold_anchor_fraction', 'task_failure_fraction'):
                self.metric_sums[name] = self.zeros()

    def _goal_center_offset(self):
        return self.tensor([0.0, 0.0, -self.cfg.goal_ee.center_below_mount_m])

    def _resample_goal(self, ids, initial=False):
        if not len(ids):
            return
        if not hasattr(self, 'next_waypoint'):
            self.next_waypoint = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.home_sphere = self.zeros(3)
            self.home_orn_rpy = self.zeros(3)
        if initial:
            # Calculate the actual WBC-home tool point using the same FK and
            # tool offset used by the position-tracking reward.
            self._update_ee_state()
            home_cart = self.ee_pos[ids] - self.mount_pos[ids] - self._goal_center_offset()
            self.home_sphere[ids] = cart2sphere(home_cart)
            if self.behavior_version == 1:
                pole = home_cart[:, :2].norm(dim=1) < 1e-4
                self.home_sphere[ids[pole], 2] = 0.0
            # Home is now away from the spherical pole; retain its measured
            # orientation as an observation target, not as a reward term.
            self.curr_ee_goal_sphere[ids] = self.home_sphere[ids]
            self.home_orn_rpy[ids] = self.get_ee_orientation_rpy()[ids]
            self.next_waypoint[ids] = 0
            self.ee_start_sphere[ids] = self.home_sphere[ids]
            self.ee_goal_sphere[ids] = self.home_sphere[ids]
            self.ee_start_orn_rpy[ids] = self.home_orn_rpy[ids]
            self.ee_goal_orn_rpy[ids] = self.home_orn_rpy[ids]
            self.traj_timesteps[ids] = 1  # no travel; hold at WBC-home tool position
            self.traj_total_timesteps[ids] = round(3.0 / self.dt) - 1
        else:
            # Home -> P1 -> Home -> P2 -> Home, repeated after reset.
            waypoints = self.tensor(self.cfg.goal_ee.fixed_waypoints_spherical)
            phase = self.next_waypoint[ids]
            self.ee_start_sphere[ids] = self.ee_goal_sphere[ids]
            self.ee_start_orn_rpy[ids] = self.ee_goal_orn_rpy[ids]
            self.ee_goal_sphere[ids] = self.home_sphere[ids]
            self.ee_goal_orn_rpy[ids] = self.home_orn_rpy[ids]
            first = phase == 0
            second = phase == 2
            self.ee_goal_sphere[ids[first]] = waypoints[0]
            self.ee_goal_sphere[ids[second]] = waypoints[1]
            self.ee_goal_orn_rpy[ids[first | second]] = 0
            self.next_waypoint[ids] = (phase + 1) % 4
            self.traj_timesteps[ids] = round(3.0 / self.dt)
            self.traj_total_timesteps[ids] = round(6.0 / self.dt) - 1
        self.goal_timer[ids] = 0
        if hasattr(self, 'hold_anchor_ready'):
            self.hold_anchor_ready[ids] = False
            self.hold_settle_time[ids] = 0
            self.hold_anchor_lost_time[ids] = 0

    def _update_goal(self):
        t = (self.goal_timer / self.traj_timesteps).clamp(0, 1).unsqueeze(-1)
        if self.behavior_version == 3:
            t = t.pow(3)*(10+t*(-15+6*t))
        self.curr_ee_goal_sphere.copy_(torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t))
        self.curr_ee_goal_orn_rpy.copy_(torch.lerp(self.ee_start_orn_rpy, self.ee_goal_orn_rpy, t))
        # The original WBC uses a sphere centered 0.142 m below the flange.
        # Keep the inherited observation's Cartesian channel mount-relative.
        self.curr_ee_goal_cart.copy_(sphere2cart(self.curr_ee_goal_sphere) + self._goal_center_offset())
        self.curr_ee_goal_cart_world = self.mount_pos + self.curr_ee_goal_cart

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if not len(ids):
            return
        if self.behavior_version == 1:
            if hasattr(self, 'previous_ee_velocity'):
                self.previous_ee_velocity[ids] = 0
                self.previous_tangent_acc[ids] = 0
            return
        if not hasattr(self, 'phase_cycle_count'):
            self.phase_cycle_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # The physical one-environment replay always begins at home. Training
        # environments cover four coherent hold phases and rotate across them
        # on successive resets. Never offset a goal timer without its pose.
        phase = ((ids + self.phase_cycle_count[ids]) % 4
                 if self.cfg.goal_ee.diversify_initial_phase and self.num_envs > 1
                 else torch.zeros_like(ids))
        self.phase_cycle_count[ids] += 1
        if self.behavior_version == 3:
            self.drive_positions[ids, :7] = self.tensor(self.cfg.control.initial_position_targets)
            home_action = (self.drive_positions[ids, :7]-self.default_dof_pos[:7])/self.cfg.control.action_scale
            for name in ('actions', 'last_actions', 'candidate_actions', 'last_candidate_actions'):
                getattr(self, name)[ids] = home_action
        self.next_waypoint[ids] = phase
        self.traj_timesteps[ids] = 1
        self.traj_total_timesteps[ids] = round(3.0 / self.dt) - 1
        if not hasattr(self, 'phase_start_anchors'):
            self.phase_start_anchors = self.tensor(self.cfg.goal_ee.phase_start_joint_angles)
            self.phase_start_targets = self.tensor(self.cfg.goal_ee.phase_start_target_joint_angles)
            self.phase_start_waypoints = self.tensor(self.cfg.goal_ee.fixed_waypoints_spherical)
        anchors = self.phase_start_anchors
        targets = self.phase_start_targets
        waypoints = self.phase_start_waypoints
        moved = []
        for phase_id, anchor_id in ((1, 0), (3, 1)):
            selected = ids[phase == phase_id]
            if not len(selected):
                continue
            anchor = anchors[anchor_id]
            self.dof_pos[selected, :7] = anchor
            self.dof_vel[selected, :7] = 0
            self.drive_positions[selected, :7] = targets[anchor_id]
            anchor_action = (targets[anchor_id] - self.default_dof_pos[:7]) / self.cfg.control.action_scale
            for name in ('actions', 'last_actions', 'candidate_actions', 'last_candidate_actions'):
                getattr(self, name)[selected] = anchor_action
            self.ee_start_sphere[selected] = waypoints[anchor_id]
            self.ee_goal_sphere[selected] = waypoints[anchor_id]
            self.ee_start_orn_rpy[selected] = 0
            self.ee_goal_orn_rpy[selected] = 0
            moved.append(selected)
        if moved or self.behavior_version == 3:
            selected = ids if self.behavior_version == 3 else torch.cat(moved)
            actors = selected.to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(actors), len(selected))
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.drive_positions), gymtorch.unwrap_tensor(actors), len(selected))
            if self.command_delay is not None:
                self.command_delay.reset(selected, self.drive_positions[selected, :7])
        if hasattr(self, 'previous_ee_velocity'):
            self.previous_ee_velocity[ids] = 0
            self.previous_tangent_acc[ids] = 0
        if hasattr(self, 'smooth_history'):
            target = self.drive_positions[ids, :7]
            self.smooth_history.reset(ids, target)
            self.pre_slew_request[ids] = target
            self.bounded_request[ids] = target
            self.hold_joint_anchor[ids] = self.dof_pos[ids, :7]
            self.hold_anchor_ready[ids] = False
            self.hold_settle_time[ids] = 0
            self.hold_anchor_lost_time[ids] = 0
            self.task_failure_buf[ids] = False

    def step(self, actions):
        if self.behavior_version == 1:
            return super().step(actions)
        # Clamp the command actually sent to PhysX. This same task step is used
        # by visualization; a real-arm adapter must apply these limits too.
        scale = self.cfg.control.action_scale
        limit = self.target_step_limit
        previous = self.default_dof_pos[:7] + scale * self.last_actions
        requested = self.default_dof_pos[:7] + scale * actions.to(self.device).clamp(
            -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        if self.behavior_version == 3:
            self.pre_slew_request.copy_(self.default_dof_pos[:7] + scale*actions.to(self.device))
            margin = self.cfg.control.request_joint_margin_rad
            requested = torch.max(torch.min(requested, self.upper[:7]-margin), self.lower[:7]+margin)
            self.bounded_request.copy_(requested)
        limited = previous + (requested - previous).clamp(-limit, limit)
        if self.behavior_version == 3:
            self.issued_request.copy_(limited)
        return super().step((limited - self.default_dof_pos[:7]) / scale)

    def compute_reward(self):
        if self.behavior_version == 3 and not (
            torch.isfinite(self.dof_state).all() and torch.isfinite(self.pre_slew_request).all()
            and torch.isfinite(self.ee_pos).all() and torch.isfinite(self.ee_vel).all()
            and torch.isfinite(self.torques).all()
        ):
            raise FloatingPointError('Non-finite smooth-v3 state/request; refusing to corrupt PPO rewards or history')
        # Base class supplies position tracking, speed and PD-state terms.
        super().compute_reward()
        error = self.curr_ee_goal_cart_world - self.ee_pos
        direction = error / error.norm(dim=1, keepdim=True).clamp(min=self.cfg.rewards.ee_tracking_direction_eps)
        acceleration = (self.ee_vel - self.previous_ee_velocity) / self.dt
        tangent_acc = (acceleration * direction).sum(dim=1)
        normal_acc = acceleration - tangent_acc[:, None] * direction
        normal = (normal_acc.norm(dim=1) / self.cfg.rewards.ee_normal_acc_sigma).square()
        jerk = ((tangent_acc - self.previous_tangent_acc) / self.dt / self.cfg.rewards.ee_tangent_jerk_sigma).square()
        normal = normal.clamp(max=self.cfg.rewards.ee_acc_penalty_clip)
        jerk = jerk.clamp(max=self.cfg.rewards.ee_acc_penalty_clip)
        scales = self.cfg.rewards.scales
        for name, term in (('ee_normal_acc', normal), ('ee_tangent_acc_smooth', jerk)):
            contribution = getattr(scales, name) * term * self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        if self.behavior_version == 1:
            self.previous_ee_velocity.copy_(self.ee_vel)
            self.previous_tangent_acc.copy_(tangent_acc)
            return
        holding = (self.goal_timer >= self.traj_timesteps).float()
        close = torch.exp(-(error.norm(dim=1) / self.cfg.rewards.hold_error_radius).square())
        hold_weight = holding * (self.cfg.rewards.hold_velocity_floor +
                                 (1 - self.cfg.rewards.hold_velocity_floor) * close)
        sent_delta = (self.actions - self.last_actions) * self.cfg.control.action_scale
        norm = self.cfg.rewards.target_rate_norm_rad
        rate = (sent_delta / norm).square().mean(dim=1) * hold_weight
        reversal = torch.relu(-sent_delta * self.last_sent_delta).mean(dim=1) / norm**2 * hold_weight
        home_hold = holding * ((self.ee_goal_sphere - self.home_sphere).norm(dim=1) < 1e-4).float()
        home_posture = (self.dof_pos[:, :7] - self.default_dof_pos[:7]).square().sum(dim=1) * home_hold
        for name, term in (('hold_target_rate', rate), ('hold_target_reversal', reversal),
                           ('home_posture', home_posture)):
            contribution = getattr(scales, name) * term * self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        if self.behavior_version == 3:
            self._compute_smooth_reward()
        self.previous_ee_velocity.copy_(self.ee_vel)
        self.previous_tangent_acc.copy_(tangent_acc)

    def _compute_smooth_reward(self):
        cfg = self.cfg.rewards
        holding = self.goal_timer >= self.traj_timesteps
        ramp = ((self.goal_timer-self.traj_timesteps)*self.dt/cfg.hold_ramp_s).clamp(0, 1)
        multiplier = 1 + (cfg.hold_smooth_multiplier-1)*ramp
        previous = self.default_dof_pos[:7] + self.cfg.control.action_scale*self.last_actions
        target = self.default_dof_pos[:7] + self.cfg.control.action_scale*self.actions
        terms, metrics = self.smooth_history.measure(self.pre_slew_request, target, previous,
                                                    self.last_sent_delta, self.dt, cfg)
        for name in terms:
            terms[name] *= multiplier
        error = (self.ee_pos-self.curr_ee_goal_cart_world).norm(dim=-1)
        lost = self.hold_anchor_ready & (error > cfg.hold_anchor_release_error_m)
        self.hold_anchor_lost_time.copy_(torch.where(lost, self.hold_anchor_lost_time+self.dt,
                                                     torch.zeros_like(self.hold_anchor_lost_time)))
        self.hold_anchor_ready &= self.hold_anchor_lost_time < cfg.hold_anchor_release_s
        settled = holding & (error < cfg.hold_anchor_error_m) & (
            self.dof_vel[:, :7].abs().amax(dim=-1) < cfg.hold_anchor_speed_rad_s)
        self.hold_settle_time.copy_(torch.where(settled, self.hold_settle_time+self.dt,
                                               torch.zeros_like(self.hold_settle_time)))
        self.hold_anchor_ready &= holding
        acquire = holding & ~self.hold_anchor_ready & (self.hold_settle_time >= cfg.hold_anchor_settle_s)
        self.hold_joint_anchor[acquire] = self.dof_pos[acquire, :7]
        self.hold_anchor_ready |= acquire
        anchor_error = self.dof_pos[:, :7]-self.hold_joint_anchor
        terms['hold_joint_velocity'] = robust_square(self.dof_vel[:, :7]/cfg.hold_joint_speed_norm_rad_s).mean(-1)*ramp
        terms['hold_joint_acceleration'] = robust_square((self.dof_vel[:, :7]-self.last_vel)/(
            self.dt*cfg.hold_joint_acceleration_norm_rad_s2)).mean(-1)*ramp
        terms['hold_joint_anchor'] = robust_square(anchor_error/cfg.hold_anchor_norm_rad).mean(-1)*self.hold_anchor_ready
        prediction = self.dof_pos[:, :7] + self.dof_vel[:, :7]*cfg.limit_prediction_s
        lower_error = (self.lower[:7]+cfg.limit_margin_rad-prediction).clamp(min=0)
        upper_error = (prediction-self.upper[:7]+cfg.limit_margin_rad).clamp(min=0)
        terms['predictive_joint_limits'] = robust_square((lower_error+upper_error)/cfg.limit_penalty_norm_rad).amax(-1)
        approach_lower = (self.dof_vel[:, :7] < -cfg.stop_speed_rad_s) & (
            prediction < self.lower[:7]+cfg.stop_margin_rad)
        approach_upper = (self.dof_vel[:, :7] > cfg.stop_speed_rad_s) & (
            prediction > self.upper[:7]-cfg.stop_margin_rad)
        position_failure = (self.dof_pos[:, :7] < self.lower[:7]+.01) | (
            self.dof_pos[:, :7] > self.upper[:7]-.01)
        self.task_failure_buf.copy_((approach_lower | approach_upper | position_failure).any(-1))
        terms['safety_termination'] = self.task_failure_buf.float()/self.dt
        for name, term in terms.items():
            contribution = getattr(cfg.scales, name)*term*self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        for name, value in metrics.items():
            self.metric_sums[name] += value
        self.metric_sums['request_slew_fraction'] += (
            (self.pre_slew_request-previous).abs() > self.target_step_limit).float().mean(-1)
        self.metric_sums['hold_request_speed_rms_rad_s'] += metrics['request_speed_rms_rad_s']*holding
        self.metric_sums['hold_joint_drift_rms_rad'] += anchor_error.square().mean(-1).sqrt()*self.hold_anchor_ready
        self.metric_sums['hold_anchor_fraction'] += self.hold_anchor_ready.float()
        self.metric_sums['task_failure_fraction'] += self.task_failure_buf.float()
