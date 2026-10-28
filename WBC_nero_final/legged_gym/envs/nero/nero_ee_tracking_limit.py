"""Limit-aware stable EE tracking with the same predictive guard as hardware."""
import torch

from .nero_ee_tracking_stable import NeroEETrackingStable
from .nero_smooth_control import robust_square


class NeroEETrackingLimit(NeroEETrackingStable):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        margins = cfg.control.target_position_margin_rad
        if len(margins) != 7 or min(margins) <= 0:
            raise ValueError('A positive target margin is required for every arm joint')
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        margin = self.tensor(margins)
        if ((self.lower[:7] + margin) >= (self.upper[:7] - margin)).any():
            raise ValueError('Joint target margins leave no command range')
        target = self.tensor(cfg.control.initial_position_targets)
        if ((target < self.lower[:7] + margin) | (target > self.upper[:7] - margin)).any():
            raise ValueError('Home gravity target violates a target margin')
        self.task_failure_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in ('predicted_limit_violation_rad', 'requested_margin_fraction',
                     'task_failure_fraction'):
            self.metric_sums[name] = self.zeros()

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if hasattr(self, 'task_failure_buf'):
            self.task_failure_buf[ids] = False

    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        q, dq = self.dof_pos[:, :7], self.dof_vel[:, :7]
        prediction = q + dq * cfg.limit_prediction_s
        lower_penetration = (self.lower[:7] + cfg.limit_reward_margin_rad - prediction).clamp(min=0)
        upper_penetration = (prediction - self.upper[:7] + cfg.limit_reward_margin_rad).clamp(min=0)
        penetration = lower_penetration + upper_penetration
        margin = self.tensor(self.cfg.control.target_position_margin_rad)
        safe_lower, safe_upper = self.lower[:7] + margin, self.upper[:7] - margin
        request = self.pre_slew_request
        request_excess = (safe_lower - request).clamp(min=0) + (request - safe_upper).clamp(min=0)
        approaching_lower = (dq < -cfg.limit_stop_speed_rad_s) & (
            prediction < self.lower[:7] + cfg.limit_stop_margin_rad)
        approaching_upper = (dq > cfg.limit_stop_speed_rad_s) & (
            prediction > self.upper[:7] - cfg.limit_stop_margin_rad)
        at_limit = (q < self.lower[:7] + 0.01) | (q > self.upper[:7] - 0.01)
        self.task_failure_buf.copy_((approaching_lower | approaching_upper | at_limit).any(dim=1))
        terms = {
            'predictive_joint_limits': robust_square(
                penetration / cfg.limit_penalty_norm_rad).amax(dim=1),
            'requested_target_margin': robust_square(
                request_excess / cfg.target_margin_penalty_norm_rad).mean(dim=1),
            'safety_termination': self.task_failure_buf.float() / self.dt,
        }
        for name, value in terms.items():
            contribution = value * getattr(cfg.scales, name) * self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        self.metric_sums['predicted_limit_violation_rad'] += penetration.amax(dim=1)
        self.metric_sums['requested_margin_fraction'] += (request_excess > 0).float().mean(dim=1)
        self.metric_sums['task_failure_fraction'] += self.task_failure_buf.float()
