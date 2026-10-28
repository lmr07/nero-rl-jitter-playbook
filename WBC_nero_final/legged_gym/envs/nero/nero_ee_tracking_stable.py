"""Rate-limited seven-axis EE tracking with explicit hold stabilization."""
import torch
from isaacgym import gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import sphere2cart

from .nero_ee_tracking import NeroEETracking
from .nero_smooth_control import gravity_position_targets, robust_square


class NeroEETrackingStable(NeroEETracking):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        if cfg.control.action_mode != 'absolute' or cfg.control.pd_backend != 'physx_position':
            raise ValueError('Stable EE tracking requires absolute actions and native PhysX position PD')
        if abs(sim_params.dt - 0.005) > 1.e-9 or cfg.control.decimation != 4:
            raise ValueError('Stable EE tracking requires 5 ms physics and 20 ms policy steps')
        if len(cfg.control.target_speed_limits_rad_s) != 7 or min(cfg.control.target_speed_limits_rad_s) <= 0:
            raise ValueError('Seven positive target speed limits are required')
        path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        calculated = gravity_position_targets(path, cfg.init_state.default_joint_angles,
                                              cfg.control.stiffness, cfg.init_state.rot, cfg.sim.gravity)
        if max(abs(a - b) for a, b in zip(calculated, cfg.control.initial_position_targets)) > 1.e-5:
            raise ValueError('Saved gravity-compensated home target disagrees with the URDF or PD gains')
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.target_step_limit = self.tensor(cfg.control.target_speed_limits_rad_s) * self.dt
        self.pre_slew_request = self.drive_positions[:, :7].clone()
        self.previous_raw_request = self.pre_slew_request.clone()
        self.previous_sent_request = self.pre_slew_request.clone()
        self.issued_request = self.pre_slew_request.clone()
        self.hold_joint_anchor = self.dof_pos[:, :7].clone()
        self.hold_anchor_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_settle_time = self.zeros()
        self.hold_anchor_lost_time = self.zeros()
        for name in ('request_speed_rms_rad_s', 'request_gap_rms_rad', 'hold_anchor_fraction'):
            self.metric_sums[name] = self.zeros()

    def _resample_goal(self, ids, initial=False):
        super()._resample_goal(ids, initial=initial)
        if not len(ids):
            return
        if initial:
            # Orientation is expressed in the heading frame of the home goal.
            self.curr_ee_goal_sphere[ids] = self.ee_start_sphere[ids]
            self.curr_ee_goal_cart[ids] = sphere2cart(self.ee_start_sphere[ids])
            self._update_ee_state()
            measured = self.get_ee_orientation_rpy()[ids]
            configured = self.tensor(self.cfg.goal_ee.ranges.init_orn_start)
            if (measured - configured).abs().max() > 1.e-3:
                raise ValueError('Initial tool orientation no longer matches the saved pitch ramp')
            self.ee_start_orn_rpy[ids] = measured
        if hasattr(self, 'hold_anchor_ready'):
            self.hold_anchor_ready[ids] = False
            self.hold_settle_time[ids] = 0
            self.hold_anchor_lost_time[ids] = 0

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if not len(ids):
            return
        target = self.tensor(self.cfg.control.initial_position_targets)
        self.drive_positions[ids, :7] = target
        actors = ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.drive_positions), gymtorch.unwrap_tensor(actors), len(ids))
        home_action = (target - self.default_dof_pos[:7]) / self.cfg.control.action_scale
        for name in ('actions', 'last_actions', 'candidate_actions', 'last_candidate_actions'):
            getattr(self, name)[ids] = home_action
        self.last_candidate_delta[ids] = 0
        self.last_sent_delta[ids] = 0
        if hasattr(self, 'pre_slew_request'):
            for name in ('pre_slew_request', 'previous_raw_request',
                         'previous_sent_request', 'issued_request'):
                getattr(self, name)[ids] = target
            self.hold_joint_anchor[ids] = self.dof_pos[ids, :7]
            self.hold_anchor_ready[ids] = False
            self.hold_settle_time[ids] = 0
            self.hold_anchor_lost_time[ids] = 0

    def step(self, actions):
        scale = self.cfg.control.action_scale
        raw = self.default_dof_pos[:7] + scale * actions.to(self.device).clamp(
            -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        self.pre_slew_request.copy_(raw)
        margin = getattr(self.cfg.control, 'target_position_margin_rad', None)
        if margin is None:
            lower, upper = self.lower[:7], self.upper[:7]
        else:
            margin = self.tensor(margin)
            lower, upper = self.lower[:7] + margin, self.upper[:7] - margin
        bounded = torch.max(torch.min(raw, upper), lower)
        previous = self.drive_positions[:, :7].clone()
        self.previous_sent_request.copy_(previous)
        issued = previous + (bounded - previous).clamp(-self.target_step_limit, self.target_step_limit)
        self.issued_request.copy_(issued)
        return super().step((issued - self.default_dof_pos[:7]) / scale)

    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        holding = self.goal_timer >= self.traj_timesteps
        position_error = (self.ee_pos - self.curr_ee_goal_cart_world).norm(dim=1)
        orientation_error = self.get_ee_orientation_rpy() - self.curr_ee_goal_orn_rpy
        orientation_error = torch.atan2(torch.sin(orientation_error), torch.cos(orientation_error)).norm(dim=1)
        joint_speed = self.dof_vel[:, :7]
        settled = (holding & (position_error < cfg.hold_anchor_pos_error_m)
                   & (orientation_error < cfg.hold_anchor_orn_error_rad)
                   & (joint_speed.abs().amax(dim=1) < cfg.hold_anchor_joint_speed_rad_s))
        self.hold_settle_time = torch.where(settled, self.hold_settle_time + self.dt,
                                            torch.zeros_like(self.hold_settle_time))
        new_anchor = ~self.hold_anchor_ready & (self.hold_settle_time >= cfg.hold_anchor_settle_s)
        self.hold_joint_anchor[new_anchor] = self.dof_pos[new_anchor, :7]
        self.hold_anchor_ready |= new_anchor
        lost = self.hold_anchor_ready & (position_error > cfg.hold_anchor_release_pos_error_m)
        self.hold_anchor_lost_time = torch.where(lost, self.hold_anchor_lost_time + self.dt,
                                                torch.zeros_like(self.hold_anchor_lost_time))
        release = self.hold_anchor_lost_time >= cfg.hold_anchor_release_s
        self.hold_anchor_ready[release] = False
        self.hold_settle_time[release] = 0
        self.hold_anchor_lost_time[release] = 0

        request_delta = self.pre_slew_request - self.previous_raw_request
        sent_delta = self.issued_request - self.previous_sent_request
        acceleration = (joint_speed - self.last_vel) / self.dt
        terms = {
            'pre_slew_request_rate': robust_square(
                request_delta / (self.dt * cfg.request_speed_norm_rad_s)).mean(dim=1),
            'pre_slew_request_gap': robust_square(
                (self.pre_slew_request - self.issued_request) / cfg.request_gap_norm_rad).mean(dim=1),
            'hold_joint_velocity': joint_speed.square().sum(dim=1) * holding,
            'hold_joint_acceleration': (acceleration / cfg.hold_acceleration_norm_rad_s2).square().mean(dim=1) * holding,
            'hold_target_rate': robust_square(
                sent_delta / (self.dt * cfg.hold_target_speed_norm_rad_s)).mean(dim=1) * holding,
            'hold_joint_anchor': robust_square(
                (self.dof_pos[:, :7] - self.hold_joint_anchor) / cfg.hold_anchor_norm_rad
            ).mean(dim=1) * self.hold_anchor_ready,
        }
        for name, value in terms.items():
            contribution = value * getattr(cfg.scales, name) * self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        self.metric_sums['request_speed_rms_rad_s'] += request_delta.square().mean(dim=1).sqrt() / self.dt
        self.metric_sums['request_gap_rms_rad'] += (
            self.pre_slew_request - self.issued_request).square().mean(dim=1).sqrt()
        self.metric_sums['hold_anchor_fraction'] += (self.hold_anchor_ready & holding).float()
        self.previous_raw_request.copy_(self.pre_slew_request)
