"""Pure-PyTorch Stage-3 actuator model for the A2 + Nero policy.

The class in this module deliberately has no Isaac Gym imports.  It advances
exactly once per *policy* tick.  A simulator may call :meth:`compute_torques`
multiple times afterwards (once per physics substep) without advancing the
rate adapters or delay queues again.

The public tensors make the control boundaries explicit:

``raw_action -> q_default + 0.25 * raw_action -> sent_target -> plant_target -> torque``

``observed_feedback`` is used by the safety envelopes.  ``true_dof_pos`` and
``true_dof_vel`` are used only by the PD plant.  In particular, delayed plant
targets never leak into the policy's previous-action observation.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float], float]

NUM_ACTIONS = 19
NUM_A2_ACTIONS = 12
NUM_NERO_ACTIONS = 7

# Isaac Gym preserves the actuated-joint order of this repository's URDF.
# These constants are provided for independent tools/tests; the environment
# should still pass the limits read from its loaded asset so an asset mismatch
# is detected rather than hidden.
A2NERO_ACTUATED_DOF_NAMES: Tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "nero_joint1",
    "nero_joint2",
    "nero_joint3",
    "nero_joint4",
    "nero_joint5",
    "nero_joint6",
    "nero_joint7",
)

A2NERO_URDF_HARD_LOWER: Tuple[float, ...] = (
    -1.01,
    -2.34,
    -2.77,
    -1.01,
    -2.34,
    -2.77,
    -1.01,
    -1.56,
    -2.77,
    -1.01,
    -1.56,
    -2.77,
    -2.70526,
    -1.74,
    -2.75,
    -1.01,
    -2.75,
    -0.73,
    -1.5707963,
)

A2NERO_URDF_HARD_UPPER: Tuple[float, ...] = (
    1.01,
    3.15,
    -0.54,
    1.01,
    3.15,
    -0.54,
    1.01,
    3.94,
    -0.54,
    1.01,
    3.94,
    -0.54,
    2.70526,
    1.74,
    2.75,
    2.14,
    2.75,
    0.95,
    1.5707963,
)


@dataclass(frozen=True)
class ActuatorTick:
    """Snapshot returned by one policy-tick advance.

    Targets are physical joint angles.  ``previous_action`` is the normalized
    sent command placed in the next observation, with any configured static
    target bias removed.
    """

    raw_action: torch.Tensor
    policy_target: torch.Tensor
    bounded_target: torch.Tensor
    rate_limited_target: torch.Tensor
    sent_target: torch.Tensor
    plant_target: torch.Tensor
    observed_feedback: torch.Tensor
    previous_action: torch.Tensor
    a2_delay_ticks: torch.Tensor
    nero_delay_ticks: torch.Tensor
    a2_feedback_scale: torch.Tensor
    nero_rate_scale: torch.Tensor
    a2_feedback_saturation_fraction: torch.Tensor
    nero_rate_saturation_fraction: torch.Tensor
    nero_feedback_saturation_fraction: torch.Tensor
    joint_delay_ticks: torch.Tensor


class A2NeroStage3Actuator:
    """Stateful 19-axis Stage-3 target adapter and delayed PD plant.

    Args:
        num_envs: Number of independent vectorized environments.
        default_dof_pos: Checkpoint-compatible 19-axis default pose.
        hard_lower/hard_upper: Absolute URDF hard limits, not reward soft
            limits.  Both are required so callers cannot silently substitute
            the 90-percent reward envelope.
        enabled: When false, reproduce the legacy Stage-1/2 target mapping:
            ``default + action_scale * clip(raw_action)``.  Safety adapters,
            static bias, and delays are bypassed.
        joint_limit_margin: Deployment safety margin subtracted from both
            sides of the URDF interval.  May be scalar or 19-vector.
    """

    def __init__(
        self,
        num_envs: int,
        default_dof_pos: TensorLike,
        hard_lower: TensorLike,
        hard_upper: TensorLike,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        enabled: bool = True,
        action_scale: float = 0.25,
        raw_action_clip: TensorLike = 100.0,
        joint_limit_margin: TensorLike = 0.0,
        nero_max_step: float = 0.005,
        a2_stationary_feedback_limit: float = 0.30,
        a2_active_feedback_limit: float = 0.40,
        nero_feedback_limit: float = 0.10,
        a2_delay_options: Sequence[int] = (0, 1),
        a2_delay_probabilities: Sequence[float] = (0.85, 0.15),
        nero_delay_options: Sequence[int] = (0, 1),
        nero_delay_probabilities: Sequence[float] = (0.30, 0.70),
        static_target_bias: TensorLike = 0.0,
        identified_enabled: bool = False,
        policy_dt: float = 0.02,
        joint_delay_ranges_ticks: Optional[Sequence[Sequence[int]]] = None,
        per_tick_jitter_ticks: Tuple[int, int] = (0, 0),
        tail_probability: float = 0.0,
        tail_delay_ticks: Tuple[int, int] = (0, 0),
        a2_tau_range_s: Tuple[float, float] = (0.0, 0.0),
        nero_tau_range_s: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive, got {action_scale}")
        if nero_max_step <= 0.0:
            raise ValueError(f"nero_max_step must be positive, got {nero_max_step}")
        if a2_stationary_feedback_limit <= 0.0:
            raise ValueError("a2_stationary_feedback_limit must be positive")
        if a2_active_feedback_limit <= 0.0:
            raise ValueError("a2_active_feedback_limit must be positive")
        if nero_feedback_limit <= 0.0:
            raise ValueError("nero_feedback_limit must be positive")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.enabled = bool(enabled)
        self.action_scale = float(action_scale)
        self.nero_max_step = float(nero_max_step)
        self.a2_stationary_feedback_limit = float(a2_stationary_feedback_limit)
        self.a2_active_feedback_limit = float(a2_active_feedback_limit)
        self.nero_feedback_limit = float(nero_feedback_limit)
        self.identified_enabled = bool(identified_enabled)
        self.identified_factor = 1.0
        self.policy_dt = float(policy_dt)
        if self.policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")
        self.per_tick_jitter_ticks = tuple(int(v) for v in per_tick_jitter_ticks)
        self.tail_probability = float(tail_probability)
        self.tail_delay_ticks = tuple(int(v) for v in tail_delay_ticks)
        self.a2_tau_range_s = tuple(float(v) for v in a2_tau_range_s)
        self.nero_tau_range_s = tuple(float(v) for v in nero_tau_range_s)
        for label, bounds in (
            ("per_tick_jitter_ticks", self.per_tick_jitter_ticks),
            ("tail_delay_ticks", self.tail_delay_ticks),
        ):
            if len(bounds) != 2 or bounds[0] < 0 or bounds[0] > bounds[1]:
                raise ValueError(f"{label} must satisfy 0 <= low <= high")
        for label, bounds in (
            ("a2_tau_range_s", self.a2_tau_range_s),
            ("nero_tau_range_s", self.nero_tau_range_s),
        ):
            if len(bounds) != 2 or bounds[0] < 0.0 or bounds[0] > bounds[1]:
                raise ValueError(f"{label} must satisfy 0 <= low <= high")
        if not 0.0 <= self.tail_probability <= 1.0:
            raise ValueError("tail_probability must be in [0, 1]")

        self.default_dof_pos = self._joint_row(default_dof_pos, "default_dof_pos")
        self.hard_lower = self._joint_row(hard_lower, "hard_lower")
        self.hard_upper = self._joint_row(hard_upper, "hard_upper")
        self.joint_limit_margin = self._joint_row(joint_limit_margin, "joint_limit_margin")
        if torch.any(self.joint_limit_margin < 0.0):
            raise ValueError("joint_limit_margin must be non-negative")
        self.safe_lower = self.hard_lower + self.joint_limit_margin
        self.safe_upper = self.hard_upper - self.joint_limit_margin
        if torch.any(self.safe_lower > self.safe_upper):
            bad = torch.nonzero(self.safe_lower > self.safe_upper, as_tuple=False).flatten().tolist()
            raise ValueError(f"joint_limit_margin collapses hard-limit interval at indices {bad}")

        self.raw_action_clip = self._joint_row(raw_action_clip, "raw_action_clip")
        if torch.any(self.raw_action_clip < 0.0):
            raise ValueError("raw_action_clip must be non-negative")
        self.static_target_bias = self._joint_row(static_target_bias, "static_target_bias")

        self._a2_delay_options, self._a2_delay_probabilities = self._delay_distribution(
            a2_delay_options, a2_delay_probabilities, "A2"
        )
        self._nero_delay_options, self._nero_delay_probabilities = self._delay_distribution(
            nero_delay_options, nero_delay_probabilities, "Nero"
        )
        legacy_max_delay = max(
            int(torch.max(self._a2_delay_options).item()),
            int(torch.max(self._nero_delay_options).item()),
        )
        if joint_delay_ranges_ticks is None:
            self.joint_delay_ranges_ticks = None
            identified_max_delay = 0
        else:
            ranges = torch.as_tensor(joint_delay_ranges_ticks, device=self.device, dtype=torch.long)
            if tuple(ranges.shape) != (NUM_ACTIONS, 2) or torch.any(ranges < 0) or torch.any(ranges[:, 0] > ranges[:, 1]):
                raise ValueError("joint_delay_ranges_ticks must have shape [19,2] with 0 <= low <= high")
            self.joint_delay_ranges_ticks = ranges
            identified_max_delay = int(ranges[:, 1].max().item()) + max(self.per_tick_jitter_ticks) + max(self.tail_delay_ticks)
        self.max_delay_ticks = max(legacy_max_delay, identified_max_delay)

        batch_shape = (self.num_envs, NUM_ACTIONS)
        default_batch = self.default_dof_pos.expand(batch_shape).clone()
        self.raw_action = torch.zeros(batch_shape, device=self.device, dtype=self.dtype)
        self.policy_target = default_batch.clone()
        self.bounded_target = default_batch.clone()
        self.rate_limited_target = default_batch.clone()
        self.sent_target = default_batch.clone()
        self.previous_sent_target = default_batch.clone()
        self.plant_target = default_batch.clone()
        self.observed_feedback = default_batch.clone()
        self.previous_action = torch.zeros(batch_shape, device=self.device, dtype=self.dtype)

        self.true_dof_pos = default_batch.clone()
        self.true_dof_vel = torch.zeros_like(default_batch)
        self.position_error = torch.zeros_like(default_batch)
        self.nominal_torque = torch.zeros_like(default_batch)
        self.unclipped_torque = torch.zeros_like(default_batch)
        self.torque = torch.zeros_like(default_batch)

        self.a2_delay_ticks = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.nero_delay_ticks = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.joint_delay_ticks = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device, dtype=torch.long)
        self.actuator_tau_s = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device, dtype=self.dtype)
        self._delay_queue = default_batch.unsqueeze(0).repeat(self.max_delay_ticks + 1, 1, 1)

        self.a2_feedback_scale = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.nero_rate_scale = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.a2_feedback_saturation_fraction = torch.zeros(
            self.num_envs, device=self.device, dtype=self.dtype
        )
        self.nero_rate_saturation_fraction = torch.zeros_like(
            self.a2_feedback_saturation_fraction
        )
        self.nero_feedback_saturation_fraction = torch.zeros_like(
            self.a2_feedback_saturation_fraction
        )

        # A newly constructed pipeline is already in a valid reset state.
        self.reset(observed_feedback=default_batch)

    def configure_identified(self, enabled: bool) -> None:
        """Enable the isolated model39000 plant without changing legacy Stage-3."""
        self.identified_enabled = bool(enabled)

    def _joint_row(self, value: TensorLike, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.numel() == 1:
            tensor = tensor.reshape(1, 1).expand(1, NUM_ACTIONS)
        elif tensor.numel() == NUM_ACTIONS:
            tensor = tensor.reshape(1, NUM_ACTIONS)
        else:
            raise ValueError(
                f"{name} must be scalar or contain {NUM_ACTIONS} values, got shape {tuple(tensor.shape)}"
            )
        return tensor.clone()

    def _batch(self, value: TensorLike, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.ndim == 1 and tensor.numel() == NUM_ACTIONS and self.num_envs == 1:
            tensor = tensor.unsqueeze(0)
        expected = (self.num_envs, NUM_ACTIONS)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _broadcast_batch(self, value: TensorLike, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.numel() == 1:
            return tensor.reshape(1, 1).expand(self.num_envs, NUM_ACTIONS)
        if tensor.numel() == NUM_ACTIONS:
            return tensor.reshape(1, NUM_ACTIONS).expand(self.num_envs, NUM_ACTIONS)
        expected = (self.num_envs, NUM_ACTIONS)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must be scalar, a {NUM_ACTIONS}-vector, or shape {expected}; "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    def _delay_distribution(
        self,
        options: Sequence[int],
        probabilities: Sequence[float],
        label: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(options) == 0 or len(options) != len(probabilities):
            raise ValueError(f"{label} delay options/probabilities must have the same non-zero length")
        option_tensor = torch.as_tensor(options, device=self.device, dtype=torch.long)
        if torch.any(option_tensor < 0):
            raise ValueError(f"{label} delay options must be non-negative")
        probability_tensor = torch.as_tensor(
            probabilities, device=self.device, dtype=self.dtype
        )
        if torch.any(probability_tensor < 0.0) or float(probability_tensor.sum().item()) <= 0.0:
            raise ValueError(f"{label} delay probabilities must be non-negative with positive sum")
        probability_tensor = probability_tensor / probability_tensor.sum()
        return option_tensor, probability_tensor

    def set_delay_probabilities(
        self,
        *,
        a2_probabilities: Optional[Sequence[float]] = None,
        nero_probabilities: Optional[Sequence[float]] = None,
    ) -> None:
        """Update curriculum probabilities without changing episode delays.

        Existing environments retain their sampled delays; the new
        distribution is used only by subsequent :meth:`reset` calls.
        """

        if a2_probabilities is not None:
            _, probabilities = self._delay_distribution(
                self._a2_delay_options.tolist(), a2_probabilities, "A2"
            )
            self._a2_delay_probabilities = probabilities
        if nero_probabilities is not None:
            _, probabilities = self._delay_distribution(
                self._nero_delay_options.tolist(), nero_probabilities, "Nero"
            )
            self._nero_delay_probabilities = probabilities

    def _env_ids(self, env_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError(f"env_ids out of range for num_envs={self.num_envs}: {ids.tolist()}")
        return ids

    def _reset_feedback(self, feedback: TensorLike, env_ids: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(feedback, device=self.device, dtype=self.dtype)
        if tensor.ndim == 1 and tensor.numel() == NUM_ACTIONS:
            return tensor.reshape(1, NUM_ACTIONS).expand(len(env_ids), NUM_ACTIONS)
        if tuple(tensor.shape) == (len(env_ids), NUM_ACTIONS):
            return tensor
        if tuple(tensor.shape) == (self.num_envs, NUM_ACTIONS):
            return tensor[env_ids]
        raise ValueError(
            "observed_feedback must be a 19-vector, a full [num_envs, 19] batch, "
            f"or an [len(env_ids), 19] batch; got {tuple(tensor.shape)}"
        )

    def _sample_delay(
        self,
        options: torch.Tensor,
        probabilities: torch.Tensor,
        count: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        sampled_indices = torch.multinomial(
            probabilities,
            count,
            replacement=True,
            generator=generator,
        )
        return options[sampled_indices]

    def _explicit_delay(
        self,
        value: Union[int, torch.Tensor, Sequence[int]],
        env_ids: torch.Tensor,
        allowed: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.long)
        if tensor.numel() == 1:
            tensor = tensor.reshape(1).expand(len(env_ids))
        elif tuple(tensor.shape) == (self.num_envs,):
            tensor = tensor[env_ids]
        elif tuple(tensor.shape) != (len(env_ids),):
            raise ValueError(
                f"{label} delay must be scalar, [num_envs], or [len(env_ids)], got {tuple(tensor.shape)}"
            )
        valid = (tensor.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1)
        if not torch.all(valid):
            bad = tensor[~valid].tolist()
            raise ValueError(f"{label} delays {bad} are not in configured options {allowed.tolist()}")
        return tensor

    def reset(
        self,
        observed_feedback: TensorLike,
        env_ids: Optional[torch.Tensor] = None,
        *,
        a2_delay_ticks: Optional[Union[int, torch.Tensor, Sequence[int]]] = None,
        nero_delay_ticks: Optional[Union[int, torch.Tensor, Sequence[int]]] = None,
        generator: Optional[torch.Generator] = None,
        true_dof_pos: Optional[TensorLike] = None,
        true_dof_vel: Optional[TensorLike] = None,
    ) -> None:
        """Reset selected episodes and seed every target-delay slot from feedback.

        Delay values are sampled only here and therefore remain fixed for the
        whole episode.  Explicit delay arguments are useful for deterministic
        play and trace tests.
        """

        ids = self._env_ids(env_ids)
        if len(ids) == 0:
            return
        # Clone because callers may pass one of this object's public buffers
        # back into reset; indexed assignment from an overlapping view is
        # rejected by PyTorch.
        feedback = self._reset_feedback(observed_feedback, ids).clone()

        if a2_delay_ticks is None:
            sampled_a2 = self._sample_delay(
                self._a2_delay_options,
                self._a2_delay_probabilities,
                len(ids),
                generator,
            )
        else:
            sampled_a2 = self._explicit_delay(
                a2_delay_ticks, ids, self._a2_delay_options, "A2"
            )
        if nero_delay_ticks is None:
            sampled_nero = self._sample_delay(
                self._nero_delay_options,
                self._nero_delay_probabilities,
                len(ids),
                generator,
            )
        else:
            sampled_nero = self._explicit_delay(
                nero_delay_ticks, ids, self._nero_delay_options, "Nero"
            )
        self.a2_delay_ticks[ids] = sampled_a2
        self.nero_delay_ticks[ids] = sampled_nero
        if self.identified_enabled:
            if self.joint_delay_ranges_ticks is None:
                # Identified Stage-1 retains independent episode samples for
                # A2 and Nero while broadcasting them to their member joints.
                self.joint_delay_ticks[ids, :NUM_A2_ACTIONS] = sampled_a2.unsqueeze(1)
                self.joint_delay_ticks[ids, NUM_A2_ACTIONS:] = sampled_nero.unsqueeze(1)
            else:
                lows = self.joint_delay_ranges_ticks[:, 0].unsqueeze(0)
                highs = self.joint_delay_ranges_ticks[:, 1].unsqueeze(0)
                uniform = torch.rand((len(ids), NUM_ACTIONS), device=self.device, generator=generator)
                sampled = lows + torch.floor(uniform * (highs - lows + 1).to(self.dtype)).to(torch.long)
                self.joint_delay_ticks[ids] = sampled
            for start, end, tau_range in (
                (0, NUM_A2_ACTIONS, self.a2_tau_range_s),
                (NUM_A2_ACTIONS, NUM_ACTIONS, self.nero_tau_range_s),
            ):
                low, high = tau_range
                if high <= 0.0:
                    self.actuator_tau_s[ids, start:end] = 0.0
                else:
                    self.actuator_tau_s[ids, start:end] = low + (high - low) * torch.rand(
                        (len(ids), end - start), device=self.device, generator=generator
                    )
        else:
            self.joint_delay_ticks[ids, :NUM_A2_ACTIONS] = sampled_a2.unsqueeze(1)
            self.joint_delay_ticks[ids, NUM_A2_ACTIONS:] = sampled_nero.unsqueeze(1)
            self.actuator_tau_s[ids] = 0.0

        # A reset has no actor output yet.  Seed raw/policy consistently from
        # feedback so the previous-action observation has no artificial jump.
        seed_action = (feedback - self.default_dof_pos) / self.action_scale
        seed_policy_target = self.default_dof_pos + self.action_scale * seed_action

        self.raw_action[ids] = seed_action
        self.policy_target[ids] = seed_policy_target
        self.bounded_target[ids] = feedback
        self.rate_limited_target[ids] = feedback
        self.sent_target[ids] = feedback
        self.previous_sent_target[ids] = feedback
        self.plant_target[ids] = feedback
        self.observed_feedback[ids] = feedback
        self.previous_action[ids] = seed_action
        self._delay_queue[:, ids, :] = feedback.unsqueeze(0)

        self.a2_feedback_scale[ids] = 1.0
        self.nero_rate_scale[ids] = 1.0
        self.a2_feedback_saturation_fraction[ids] = 0.0
        self.nero_rate_saturation_fraction[ids] = 0.0
        self.nero_feedback_saturation_fraction[ids] = 0.0

        if true_dof_pos is not None:
            self.true_dof_pos[ids] = self._reset_feedback(true_dof_pos, ids).clone()
        if true_dof_vel is not None:
            self.true_dof_vel[ids] = self._reset_feedback(true_dof_vel, ids).clone()

    @property
    def delay_history(self) -> torch.Tensor:
        """The target-delay queue, ordered oldest to newest."""

        return self._delay_queue

    @staticmethod
    def seed_observation_history(
        current_observation: torch.Tensor,
        frame_stack: int = 16,
        *,
        flatten: bool = False,
    ) -> torch.Tensor:
        """Repeat a valid current frame instead of manufacturing zero history."""

        if current_observation.ndim != 2:
            raise ValueError(
                "current_observation must have shape [num_envs, single_observation_dim]"
            )
        if int(frame_stack) <= 0:
            raise ValueError(f"frame_stack must be positive, got {frame_stack}")
        history = current_observation.unsqueeze(1).repeat(1, int(frame_stack), 1)
        if flatten:
            return history.reshape(current_observation.shape[0], -1)
        return history

    @staticmethod
    def seed_history_in_place(
        history: Sequence[torch.Tensor],
        env_ids: torch.Tensor,
        current_observation: torch.Tensor,
    ) -> None:
        """Seed deque/list frames for selected envs with one valid observation."""

        ids = torch.as_tensor(env_ids, device=current_observation.device, dtype=torch.long)
        for frame in history:
            frame[ids] = current_observation[ids]

    def _gait_active(self, gait_active: Union[bool, torch.Tensor, Sequence[bool]]) -> torch.Tensor:
        active = torch.as_tensor(gait_active, device=self.device, dtype=torch.bool)
        if active.numel() == 1:
            return active.reshape(1).expand(self.num_envs)
        if tuple(active.shape) != (self.num_envs,):
            raise ValueError(
                f"gait_active must be scalar or shape ({self.num_envs},), got {tuple(active.shape)}"
            )
        return active

    @staticmethod
    def _independent_clamp(
        target: torch.Tensor,
        feedback: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        return torch.maximum(torch.minimum(target, feedback + limit), feedback - limit)

    def _push_delay_queue(self, sent_target: torch.Tensor) -> torch.Tensor:
        if self.max_delay_ticks > 0:
            self._delay_queue[:-1].copy_(self._delay_queue[1:].clone())
        self._delay_queue[-1].copy_(sent_target)

        env_index = torch.arange(self.num_envs, device=self.device)
        a2_queue_index = self.max_delay_ticks - self.a2_delay_ticks
        nero_queue_index = self.max_delay_ticks - self.nero_delay_ticks
        plant = torch.empty_like(sent_target)
        plant[:, :NUM_A2_ACTIONS] = self._delay_queue[
            a2_queue_index, env_index, :NUM_A2_ACTIONS
        ]
        plant[:, NUM_A2_ACTIONS:] = self._delay_queue[
            nero_queue_index, env_index, NUM_A2_ACTIONS:
        ]
        return plant

    def _push_identified_delay_queue(self, sent_target: torch.Tensor) -> torch.Tensor:
        if self.max_delay_ticks > 0:
            self._delay_queue[:-1].copy_(self._delay_queue[1:].clone())
        self._delay_queue[-1].copy_(sent_target)
        delays = torch.round(self.joint_delay_ticks.to(self.dtype) * self.identified_factor).to(torch.long)
        jitter_low, jitter_high = self.per_tick_jitter_ticks
        if jitter_high > 0 and self.identified_factor > 0.0:
            # Scheduling jitter belongs to one controller frame. Independent
            # per-joint jitter would fabricate a 19-axis phase scramble that
            # is absent from both hardware transports.
            delays = delays + torch.randint(
                jitter_low,
                jitter_high + 1,
                (self.num_envs, 1),
                device=self.device,
            )
        if self.tail_probability > 0.0 and self.identified_factor > 0.0:
            tail_mask = (
                torch.rand((self.num_envs, 1), device=self.device)
                < self.tail_probability * self.identified_factor
            )
            tail_low, tail_high = self.tail_delay_ticks
            tails = torch.randint(
                tail_low,
                tail_high + 1,
                (self.num_envs, 1),
                device=self.device,
            )
            delays = delays + torch.where(tail_mask, tails, torch.zeros_like(tails))
        delays = torch.clamp(delays, max=self.max_delay_ticks)
        queue_indices = self.max_delay_ticks - delays
        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(delays)
        joint_indices = torch.arange(NUM_ACTIONS, device=self.device).unsqueeze(0).expand_as(delays)
        delayed = self._delay_queue[queue_indices, env_indices, joint_indices]
        tau = self.actuator_tau_s * self.identified_factor
        alpha = torch.where(
            tau > 0.0,
            1.0 - torch.exp(-self.policy_dt / torch.clamp(tau, min=1.0e-6)),
            torch.ones_like(tau),
        )
        return self.plant_target + alpha * (delayed - self.plant_target)

    def advance(
        self,
        raw_action: TensorLike,
        observed_feedback: TensorLike,
        gait_active: Union[bool, torch.Tensor, Sequence[bool]],
    ) -> ActuatorTick:
        """Advance the adapters and target-delay queue by one policy tick."""

        actor_action = self._batch(raw_action, "raw_action")
        feedback = self._batch(observed_feedback, "observed_feedback")
        clipped_action = torch.maximum(
            torch.minimum(actor_action, self.raw_action_clip), -self.raw_action_clip
        )
        policy_target = self.default_dof_pos + self.action_scale * clipped_action

        if self.identified_enabled:
            # Identified continuation preserves the checkpoint's target map.
            # Delay and plant dynamics affect physics, while previous_action
            # remains the actual software-sent target (before plant latency).
            bounded_target = policy_target
            rate_limited_target = policy_target
            sent_target = policy_target
            plant_target = self._push_identified_delay_queue(sent_target)
            action_observation = (sent_target - self.default_dof_pos - self.static_target_bias) / self.action_scale
            a2_scale = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
            nero_scale = torch.ones_like(a2_scale)
            a2_saturation = torch.zeros_like(a2_scale)
            nero_rate_saturation = torch.zeros_like(a2_scale)
            nero_feedback_saturation = torch.zeros_like(a2_scale)
        elif not self.enabled:
            # Exact legacy Stage-1/2 mapping.  In particular, do not introduce
            # a hard-limit clamp or a nominal delay into an old checkpoint run.
            bounded_target = policy_target
            rate_limited_target = policy_target
            sent_target = policy_target
            plant_target = policy_target
            a2_scale = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
            nero_scale = torch.ones_like(a2_scale)
            a2_saturation = torch.zeros_like(a2_scale)
            nero_rate_saturation = torch.zeros_like(a2_scale)
            nero_feedback_saturation = torch.zeros_like(a2_scale)
            # Keep the diagnostic history meaningful even though delay is bypassed.
            if self.max_delay_ticks > 0:
                self._delay_queue[:-1].copy_(self._delay_queue[1:].clone())
            self._delay_queue[-1].copy_(sent_target)
            action_observation = clipped_action
        else:
            # Stage-3 deliberately adds no target offset and no new hard-limit
            # semantics.  The checkpoint's position mapping stays exactly
            # q_default + action_scale * raw_action; software adapters operate
            # on that target only.
            bounded_target = policy_target

            active = self._gait_active(gait_active)
            a2_feedback = feedback[:, :NUM_A2_ACTIONS]
            a2_target = bounded_target[:, :NUM_A2_ACTIONS]
            a2_delta = a2_target - a2_feedback
            a2_max_delta = torch.amax(torch.abs(a2_delta), dim=1)
            a2_active_scale = torch.clamp(
                self.a2_active_feedback_limit / torch.clamp(a2_max_delta, min=1.0e-12),
                max=1.0,
            )
            a2_active_target = a2_feedback + a2_active_scale.unsqueeze(1) * a2_delta
            a2_stationary_target = self._independent_clamp(
                a2_target, a2_feedback, self.a2_stationary_feedback_limit
            )
            a2_sent = torch.where(active.unsqueeze(1), a2_active_target, a2_stationary_target)
            # A single direction-preserving scale is meaningful only for the
            # active-gait branch; stationary mode intentionally clips joints
            # independently and is summarized by its saturation fraction.
            a2_scale = torch.where(
                active,
                a2_active_scale,
                torch.ones_like(a2_active_scale),
            )

            nero_target = bounded_target[:, NUM_A2_ACTIONS:]
            nero_previous = self.previous_sent_target[:, NUM_A2_ACTIONS:]
            nero_delta = nero_target - nero_previous
            nero_max_delta = torch.amax(torch.abs(nero_delta), dim=1)
            nero_scale = torch.clamp(
                self.nero_max_step / torch.clamp(nero_max_delta, min=1.0e-12),
                max=1.0,
            )
            nero_rate_target = nero_previous + nero_scale.unsqueeze(1) * nero_delta
            nero_sent = self._independent_clamp(
                nero_rate_target,
                feedback[:, NUM_A2_ACTIONS:],
                self.nero_feedback_limit,
            )

            rate_limited_target = torch.cat((a2_target, nero_rate_target), dim=1)
            sent_target = torch.cat((a2_sent, nero_sent), dim=1)
            plant_target = self._push_delay_queue(sent_target)
            action_observation = (
                sent_target - self.default_dof_pos
            ) / self.action_scale

            tolerance = 1.0e-7
            a2_saturation = torch.mean(
                (torch.abs(a2_sent - a2_target) > tolerance).to(self.dtype), dim=1
            )
            nero_rate_saturation = torch.mean(
                (torch.abs(nero_rate_target - nero_target) > tolerance).to(self.dtype), dim=1
            )
            nero_feedback_saturation = torch.mean(
                (torch.abs(nero_sent - nero_rate_target) > tolerance).to(self.dtype), dim=1
            )

        # Keep the preallocated tensors alive.  Environment rollout runs under
        # torch.inference_mode(); rebinding here would make reset state an
        # inference tensor that cannot later be updated in-place.
        self.raw_action.copy_(clipped_action)
        self.policy_target.copy_(policy_target)
        self.bounded_target.copy_(bounded_target)
        self.rate_limited_target.copy_(rate_limited_target)
        self.sent_target.copy_(sent_target)
        self.plant_target.copy_(plant_target)
        self.observed_feedback.copy_(feedback)
        self.previous_action.copy_(action_observation)
        self.previous_sent_target.copy_(sent_target)
        self.a2_feedback_scale.copy_(a2_scale)
        self.nero_rate_scale.copy_(nero_scale)
        self.a2_feedback_saturation_fraction.copy_(a2_saturation)
        self.nero_rate_saturation_fraction.copy_(nero_rate_saturation)
        self.nero_feedback_saturation_fraction.copy_(nero_feedback_saturation)

        return ActuatorTick(
            raw_action=self.raw_action.clone(),
            policy_target=self.policy_target.clone(),
            bounded_target=self.bounded_target.clone(),
            rate_limited_target=self.rate_limited_target.clone(),
            sent_target=self.sent_target.clone(),
            plant_target=self.plant_target.clone(),
            observed_feedback=self.observed_feedback.clone(),
            previous_action=self.previous_action.clone(),
            a2_delay_ticks=self.a2_delay_ticks.clone(),
            nero_delay_ticks=self.nero_delay_ticks.clone(),
            a2_feedback_scale=self.a2_feedback_scale.clone(),
            nero_rate_scale=self.nero_rate_scale.clone(),
            a2_feedback_saturation_fraction=self.a2_feedback_saturation_fraction.clone(),
            nero_rate_saturation_fraction=self.nero_rate_saturation_fraction.clone(),
            nero_feedback_saturation_fraction=self.nero_feedback_saturation_fraction.clone(),
            joint_delay_ticks=self.joint_delay_ticks.clone(),
        )

    # A concise alias for callers that conceptualize this as a policy step.
    step = advance

    def compute_torques(
        self,
        true_dof_pos: TensorLike,
        true_dof_vel: TensorLike,
        kp: TensorLike,
        kd: TensorLike,
        *,
        motor_strength: TensorLike = 1.0,
        torque_limits: Optional[TensorLike] = None,
        equilibrium_offset: TensorLike = 0.0,
    ) -> torch.Tensor:
        """Compute delayed-target PD torque from the unobserved true state.

        Motor strength scales the complete nominal PD torque and can therefore
        never alter ``policy_target``, ``sent_target``, or ``plant_target``.
        Torque-limit clipping is deliberately the final operation.
        """

        true_pos = self._batch(true_dof_pos, "true_dof_pos")
        true_vel = self._batch(true_dof_vel, "true_dof_vel")
        kp_batch = self._broadcast_batch(kp, "kp")
        kd_batch = self._broadcast_batch(kd, "kd")
        strength_batch = self._broadcast_batch(motor_strength, "motor_strength")
        equilibrium_offset_batch = self._broadcast_batch(
            equilibrium_offset, "equilibrium_offset"
        )

        position_error = self.plant_target + equilibrium_offset_batch - true_pos
        nominal_torque = kp_batch * position_error - kd_batch * true_vel
        unclipped_torque = strength_batch * nominal_torque
        if torque_limits is None:
            torque = unclipped_torque
        else:
            limit_batch = torch.abs(self._broadcast_batch(torque_limits, "torque_limits"))
            torque = torch.maximum(torch.minimum(unclipped_torque, limit_batch), -limit_batch)

        self.true_dof_pos.copy_(true_pos)
        self.true_dof_vel.copy_(true_vel)
        self.position_error.copy_(position_error)
        self.nominal_torque.copy_(nominal_torque)
        self.unclipped_torque.copy_(unclipped_torque)
        self.torque.copy_(torque)
        return self.torque


__all__ = [
    "A2NERO_ACTUATED_DOF_NAMES",
    "A2NERO_URDF_HARD_LOWER",
    "A2NERO_URDF_HARD_UPPER",
    "ActuatorTick",
    "A2NeroStage3Actuator",
    "NUM_ACTIONS",
    "NUM_A2_ACTIONS",
    "NUM_NERO_ACTIONS",
]
