"""WBC-style random end-effector tracking for the fixed seven-joint Nero."""
import torch

from .nero_fixed import NeroFixed


class NeroEETracking(NeroFixed):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        home_error = (self.ee_pos - self.get_ee_goal_spherical_center() -
                      self.tensor(cfg.goal_ee.home_xyz_from_center)).norm(dim=1).max()
        if home_error > 1.e-3:
            raise ValueError(f'Nero home FK no longer matches the configured WBC coordinates: {home_error.item()} m')
        self.previous_ee_velocity = self.zeros(3)
        self.previous_ee_acceleration = self.zeros(3)
        self.previous_action_delta = self.zeros(7)

    def get_ee_goal_spherical_center(self):
        # The A2nero WBC center is at A2 base +(0.16, 0, 0.5), while the
        # fixed Nero root is at A2 base +(0.16, 0, 0.142).
        return self.mount_pos + self.tensor((0.0, 0.0, -self.cfg.goal_ee.center_below_mount_m))

    def _update_goal(self):
        super()._update_goal()
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + self.curr_ee_goal_cart

    def _resample_goal(self, ids, initial=False):
        if not len(ids) or (self.cfg.env.teleop_mode and not initial):
            return
        if initial:
            self.ee_start_sphere[ids] = self.tensor(self.cfg.goal_ee.ranges.init_pos_start)
            self.ee_goal_sphere[ids] = self.tensor(self.cfg.goal_ee.ranges.init_pos_end)
            self.ee_start_orn_rpy[ids] = self.tensor(self.cfg.goal_ee.ranges.init_orn_start)
            self.ee_goal_orn_rpy[ids] = self.tensor(self.cfg.goal_ee.ranges.init_orn_end)
        else:
            self.ee_start_sphere[ids] = self.ee_goal_sphere[ids]
            self.ee_start_orn_rpy[ids] = self.ee_goal_orn_rpy[ids]
            self._resample_ee_goal_orn_once(ids)
            pending = ids
            # Match the WBC sampler: sample each spherical component uniformly,
            # then retry only paths rejected by its ten-point collision test.
            for _ in range(10):
                self._resample_ee_goal_sphere_once(pending)
                pending = pending[self.collision_check(pending)]
                if not len(pending):
                    break
        self.goal_timer[ids] = 0
        self.traj_timesteps[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.goal_ee.traj_time) / self.dt
        self.traj_total_timesteps[ids] = self.traj_timesteps[ids] + torch.empty(len(ids), device=self.device).uniform_(*self.cfg.goal_ee.hold_time) / self.dt

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if hasattr(self, 'previous_ee_velocity'):
            self.previous_ee_velocity[ids] = 0
            self.previous_ee_acceleration[ids] = 0
            self.previous_action_delta[ids] = 0

    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        direction = self.curr_ee_goal_cart_world - self.ee_pos
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=cfg.ee_tracking_direction_eps)
        acceleration = (self.ee_vel - self.previous_ee_velocity) / self.dt
        tangent_acceleration = (acceleration * direction).sum(dim=1)
        normal_acceleration = acceleration - tangent_acceleration[:, None] * direction
        previous_tangent = (self.previous_ee_acceleration * direction).sum(dim=1)
        tangent_jerk = (tangent_acceleration - previous_tangent) / self.dt
        action_delta = self.actions - self.last_actions
        arm_vel_limit = self.tensor(self.cfg.actuator.velocity_limits)
        arm_effort_limit = self.torque_limits[:7]
        terms = {
            'ee_normal_acc': (normal_acceleration.norm(dim=1) / cfg.ee_normal_acc_sigma).square().clamp(max=cfg.ee_acc_penalty_clip),
            'ee_tangent_acc_smooth': (tangent_jerk / cfg.ee_tangent_jerk_sigma).square().clamp(max=cfg.ee_acc_penalty_clip),
            'action_acc_arm': (action_delta - self.previous_action_delta).square().sum(dim=1),
            'energy_square_arm': (self.torques[:, :7] * self.dof_vel[:, :7]).square().sum(dim=1),
            'dof_vel_limits_arm': (self.dof_vel[:, :7].abs() - arm_vel_limit).clamp(min=0, max=1).sum(dim=1),
            'torque_limits_arm': (self.torques[:, :7].abs() - 0.9 * arm_effort_limit).clamp(min=0).sum(dim=1),
        }
        for i in range(7):
            terms[f'nero_joint{i + 1}_pos'] = (self.dof_pos[:, i] - self.default_dof_pos[i]).square()
        for name, value in terms.items():
            contribution = value * getattr(cfg.scales, name) * self.dt
            self.rew_buf += contribution
            self.episode_sums[name] += contribution
        self.previous_ee_velocity.copy_(self.ee_vel)
        self.previous_ee_acceleration.copy_(acceleration)
        self.previous_action_delta.copy_(action_delta)
