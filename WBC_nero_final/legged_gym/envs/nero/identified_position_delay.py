"""Per-joint, fractional command delay for Isaac Gym native position drives.

The delay is applied once to position targets at the 5 ms physics boundary.
It does not delay observations again: the source identification reported a
combined command/feedback alignment, not separately identifiable latencies.
"""
import math
import torch


class IdentifiedPositionDelay:
    def __init__(self, num_envs, delay_ms, physics_dt, device):
        if len(delay_ms) != 7 or any(not math.isfinite(x) or x < 0 for x in delay_ms):
            raise ValueError('Expected seven finite non-negative delays in milliseconds')
        self.delay_ms = tuple(float(x) for x in delay_ms)
        self.dt = float(physics_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Invalid physics dt')
        ticks = torch.tensor(self.delay_ms, dtype=torch.float32, device=device) / (1000 * self.dt)
        self.lower = torch.floor(ticks).long()
        self.fraction = (ticks - self.lower).view(1, 7)
        self.size = int(torch.ceil(ticks.max()).item()) + 2
        self.history = torch.zeros(self.size, num_envs, 7, device=device)
        self.slot = 0
        self.env_index = torch.arange(num_envs, device=device)[:, None]
        self.joint_index = torch.arange(7, device=device)[None, :]

    def reset(self, ids, position):
        if position.shape != (len(ids), 7):
            raise ValueError('Reset position must have shape [num_reset_envs, 7]')
        self.history[:, ids] = position.unsqueeze(0)

    def advance(self, target):
        if target.shape != self.history.shape[1:]:
            raise ValueError('Target must have shape [num_envs, 7]')
        self.history[self.slot].copy_(target)
        newer = self.history[(self.slot - self.lower) % self.size, self.env_index, self.joint_index]
        older = self.history[(self.slot - self.lower - 1) % self.size, self.env_index, self.joint_index]
        self.slot = (self.slot + 1) % self.size
        return newer * (1 - self.fraction) + older * self.fraction
