"""Fixed-distribution Nero host/CAN/drive/feedback model (torch only, no hardware)."""
import math
import torch


class NeroActuator:
    def __init__(self, n, dt, policy_dt, cfg, randomize, default, lower, upper, pd_backend='explicit'):
        self.n, self.dt, self.policy_dt, self.cfg = n, dt, policy_dt, cfg
        self.native_pd = pd_backend == 'physx_position'
        self.randomize = randomize
        self.device = default.device
        self.default, self.lower, self.upper = default[:7], lower[:7], upper[:7]
        self._validate()
        self.rows = torch.arange(n, device=self.device)[:, None]
        self.cols = torch.arange(7, device=self.device)[None, :]
        self.clock = torch.zeros(n, 1, device=self.device, dtype=torch.long)
        self.ring_index = 0
        self.ring_size = math.ceil(max(cfg.drive_delay_s[1], cfg.feedback_transport_s[1]) / dt) + 2
        self.command_history = torch.zeros(self.ring_size, n, 7, device=self.device)
        self.state_history = torch.zeros(self.ring_size, n, 7, 2, device=self.device)
        for name in ('initial_target', 'host_target', 'tx_target', 'drive_target', 'measured_q', 'measured_dq',
                     'filtered_torque', 'kp_scale', 'kd_scale', 'torque_cap', 'speed_cap',
                     'friction', 'viscous', 'alpha', 'position_bias'):
            setattr(self, name, torch.zeros(n, 7, device=self.device))
        for name in ('tx_period', 'rx_period', 'next_tx', 'next_rx', 'drive_delay', 'feedback_delay'):
            dtype = torch.float if self.native_pd and name in ('tx_period', 'rx_period', 'next_tx', 'next_rx') else torch.long
            setattr(self, name, torch.zeros(n, 7, dtype=dtype, device=self.device))
        self.host_elapsed = torch.zeros(n, 1, device=self.device)
        self.reset(torch.arange(n, device=self.device), self.default.expand(n, -1), torch.zeros(n, 7, device=self.device))

    def _validate(self):
        c = self.cfg
        if self.native_pd:
            bounds = c.native_joint_friction
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError('Invalid native joint friction coefficient range')
        for key in ('kp_scale', 'kd_scale', 'torque_limit_scale', 'velocity_limit_scale',
                    'torque_lag_s', 'coulomb_friction_nm', 'viscous_friction',
                    'tx_period_s', 'feedback_period_s', 'drive_delay_s', 'feedback_transport_s', 'position_bias_rad'):
            bounds = getattr(c, key)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError('Invalid Nero actuator range: ' + key)
            if key != 'position_bias_rad' and bounds[0] < 0:
                raise ValueError('Negative Nero actuator range: ' + key)
        for key in ('torque_limits', 'velocity_limits'):
            values = getattr(c, key)
            if len(values) != 7 or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError(key + ' must contain seven positive finite values')
        if not (0 < c.torque_limit_scale[0] <= c.torque_limit_scale[1] <= 1 and
                0 < c.velocity_limit_scale[0] <= c.velocity_limit_scale[1] <= 1):
            raise ValueError('Randomized capacities must be in (0, 1] of reference limits')
        if c.tx_period_s[0] < self.dt or c.feedback_period_s[0] < self.dt:
            raise ValueError('Transport period shorter than physics step')
        if (not math.isfinite(self.dt) or self.dt <= 0 or
                abs(self.policy_dt - .02) > 1.e-7 or self.dt > (.005 if self.native_pd else .0002) + 1.e-9 or
                abs(self.policy_dt / self.dt - round(self.policy_dt / self.dt)) > 1.e-6):
            raise ValueError('Nero requires 20 ms policy; maximum step is 5 ms for PhysX PD, 0.2 ms for explicit PD')
        if not (c.max_target_step > 0 and c.feedback_envelope > 0 and c.takeover_s >= 0):
            raise ValueError('Invalid deployment adapter limits')
        if not 0 < c.speed_taper_fraction <= 1 or c.joint_frame_spacing_s < 0:
            raise ValueError('Invalid speed taper or frame spacing')
        if c.position_quantum <= 0 or c.velocity_quantum <= 0 or c.friction_smoothing_velocity <= 0:
            raise ValueError('Quantization/smoothing values must be positive')

    def sample(self, bounds, count, nominal=None):
        if self.randomize:
            return torch.empty(count, 7, device=self.device).uniform_(*bounds)
        value = (bounds[0] + bounds[1]) * .5 if nominal is None else nominal
        return torch.full((count, 7), value, device=self.device)

    def reset(self, ids, q, dq):
        c, m = self.cfg, len(ids)
        self.clock[ids] = 0
        self.host_elapsed[ids] = 0
        for name in ('initial_target', 'host_target', 'tx_target', 'drive_target'):
            getattr(self, name)[ids] = q
        self.command_history[:, ids] = q
        self.state_history[:, ids, :, 0] = q
        self.state_history[:, ids, :, 1] = dq
        self.kp_scale[ids] = self.sample(c.kp_scale, m, 1.)
        self.kd_scale[ids] = self.sample(c.kd_scale, m, 1.)
        self.torque_cap[ids] = self.sample(c.torque_limit_scale, m, 1.) * torch.tensor(c.torque_limits, device=self.device)
        self.speed_cap[ids] = self.sample(c.velocity_limit_scale, m, 1.) * torch.tensor(c.velocity_limits, device=self.device)
        self.friction[ids] = self.sample(c.coulomb_friction_nm, m, 0.)
        self.viscous[ids] = self.sample(c.viscous_friction, m, 0.)
        self.position_bias[ids] = self.sample(c.position_bias_rad, m, 0.)
        tau = self.sample(c.torque_lag_s, m, 0.)
        self.alpha[ids] = torch.where(tau > 0, 1 - torch.exp(-self.dt / tau.clamp(min=1.e-12)), torch.ones_like(tau))
        self.filtered_torque[ids] = 0
        for name, bounds, nominal in (('tx_period', c.tx_period_s, .02), ('rx_period', c.feedback_period_s, .005)):
            ticks = self.sample(bounds, m, nominal) / self.dt
            getattr(self, name)[ids] = ticks.clamp(min=1) if self.native_pd else ticks.round().long().clamp(min=1)
        for name, bounds in (('drive_delay', c.drive_delay_s), ('feedback_delay', c.feedback_transport_s)):
            ticks = self.sample(bounds, m, 0.) / self.dt
            # At 5 ms, substep delays are not resolvable. Stochastic rounding
            # preserves their mean over episodes instead of making all zero.
            ticks = torch.floor(ticks + torch.rand_like(ticks)) if self.native_pd and self.randomize else ticks.round()
            getattr(self, name)[ids] = ticks.long()
        # One shared bus phase per environment, with sequential joint offsets.
        if self.randomize:
            tx_phase = torch.rand(m, 1, device=self.device) * self.tx_period[ids, :1]
            rx_phase = torch.rand(m, 1, device=self.device) * self.rx_period[ids, :1]
        else:
            tx_phase = torch.zeros(m, 1, device=self.device)
            rx_phase = torch.zeros(m, 1, device=self.device)
        offset = self.cols * c.joint_frame_spacing_s / self.dt
        if not self.native_pd:
            offset = offset.round().long()
            tx_phase, rx_phase = tx_phase.long(), rx_phase.long()
        self.next_tx[ids] = (tx_phase + offset) % self.tx_period[ids]
        self.next_rx[ids] = (rx_phase + offset) % self.rx_period[ids]
        self.measured_q[ids] = self.quantize(q + self.position_bias[ids], c.position_quantum)
        self.measured_dq[ids] = self.quantize(dq, c.velocity_quantum)

    @staticmethod
    def quantize(value, quantum):
        return torch.round(value / quantum) * quantum

    def submit(self, candidate, incremental=False):
        """Exactly the deployment blend -> slew -> feedback envelope -> joint limits."""
        c = self.cfg
        blend = (self.host_elapsed / max(c.takeover_s, 1.e-12)).clamp(max=1.) if c.takeover_s else 1.
        origin = self.host_target if incremental else self.initial_target
        target = origin + blend * (candidate - origin)
        target = torch.maximum(torch.minimum(target, self.upper), self.lower)
        target = torch.maximum(torch.minimum(target, self.host_target + c.max_target_step), self.host_target - c.max_target_step)
        target = torch.maximum(torch.minimum(target, self.measured_q + c.feedback_envelope), self.measured_q - c.feedback_envelope)
        self.host_target.copy_(torch.maximum(torch.minimum(target, self.upper), self.lower))
        self.host_elapsed += self.policy_dt
        return self.host_target

    def advance_target(self):
        """Advance command transport once per physics tick, without computing PD."""
        due = self.clock >= self.next_tx
        self.tx_target.copy_(torch.where(due, self.host_target, self.tx_target))
        self.next_tx += due.long() * self.tx_period
        self.command_history[self.ring_index].copy_(self.tx_target)
        delayed = (self.ring_index - self.drive_delay) % self.ring_size
        self.drive_target.copy_(self.command_history[delayed, self.rows, self.cols])
        return self.drive_target

    def torque(self, q, dq, kp, kd):
        self.advance_target()
        requested = self.kp_scale * kp * (self.drive_target - q) - self.kd_scale * kd * dq
        requested = requested.clamp(-self.torque_cap, self.torque_cap)
        self.filtered_torque += self.alpha * (requested - self.filtered_torque)
        # Taper only motoring torque near the speed ceiling. Never remove braking.
        motoring = self.filtered_torque * dq > 0
        fraction = ((self.speed_cap - dq.abs()) / (self.cfg.speed_taper_fraction * self.speed_cap)).clamp(0., 1.)
        motor = self.filtered_torque * torch.where(motoring, fraction, torch.ones_like(fraction))
        passive = self.friction * torch.tanh(dq / self.cfg.friction_smoothing_velocity) + self.viscous * dq
        return (motor - passive).clamp(-self.torque_cap, self.torque_cap)

    def feedback(self, q, dq):
        self.clock += 1
        self.ring_index = (self.ring_index + 1) % self.ring_size
        self.state_history[self.ring_index, :, :, 0].copy_(q)
        self.state_history[self.ring_index, :, :, 1].copy_(dq)
        idx = (self.ring_index - self.feedback_delay) % self.ring_size
        state = self.state_history[idx, self.rows, self.cols]
        due = self.clock >= self.next_rx
        self.measured_q.copy_(torch.where(due, self.quantize(state[..., 0] + self.position_bias, self.cfg.position_quantum), self.measured_q))
        self.measured_dq.copy_(torch.where(due, self.quantize(state[..., 1], self.cfg.velocity_quantum), self.measured_dq))
        self.next_rx += due.long() * self.rx_period
