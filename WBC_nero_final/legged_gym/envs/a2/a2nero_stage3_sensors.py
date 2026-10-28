"""Pure PyTorch kinematics helpers for the A2NERO Stage-3 sensor model.

Conventions:

* quaternions are ``xyzw`` and rotate vectors from a local frame to world;
* linear/angular velocities and accelerations passed to point kinematics are
  world-frame values at the rigid body's center of mass;
* IMU vectors are ``[gyro_xyz, gravity_removed_linear_acceleration_xyz]``.

No deployment board-axis correction is baked into these functions.  The
simulation's configured URDF IMU link is already the nominal sensor frame, so
applying a hardware-only ``diag(1, -1, -1)`` here would be a double transform.
"""

from typing import Optional, Sequence, Tuple, Union

import torch


TensorOrScalar = Union[torch.Tensor, float, Sequence[float]]


class A2NeroUnobservedWristFlexModel:
    """Training-only flexible-wrist state that is never added to observations.

    The rigid URDF cannot express the measured 5.5--14.5 Hz link/mount modes
    without adding policy-visible DOFs and breaking the model39000 ABI.  This
    oscillator maps rigid end-effector/base relative acceleration to an
    unobserved deflection estimate used only by a reward.  Consequently the
    no-IMU student must learn not to excite the mode; it cannot close a loop on
    synthetic information that will be absent on hardware.
    """

    def __init__(self, num_envs, device, dt, config):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.config = config
        self.factor = 1.0
        self.num_modes = len(config["frequency_bands_hz"])
        self.displacement = torch.zeros(
            self.num_envs, self.num_modes, 3, device=self.device
        )
        self.velocity = torch.zeros_like(self.displacement)
        self.total_deflection = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.frequency_hz = torch.zeros(
            self.num_envs, self.num_modes, 1, device=self.device
        )
        self.damping_ratio = torch.zeros_like(self.frequency_hz)
        self.drive_gain = torch.ones_like(self.frequency_hz)
        self.reset(torch.arange(self.num_envs, device=self.device))

    @staticmethod
    def _uniform(bounds, shape, device):
        return float(bounds[0]) + (
            float(bounds[1]) - float(bounds[0])
        ) * torch.rand(shape, device=device)

    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids) == 0:
            return
        self.displacement[ids] = 0.0
        self.velocity[ids] = 0.0
        self.total_deflection[ids] = 0.0
        for mode, bounds in enumerate(self.config["frequency_bands_hz"]):
            self.frequency_hz[ids, mode] = self._uniform(
                bounds, (len(ids), 1), self.device
            )
        self.damping_ratio[ids] = self._uniform(
            self.config["damping_ratio"],
            (len(ids), self.num_modes, 1),
            self.device,
        )
        self.drive_gain[ids] = self._uniform(
            self.config.get("drive_gain", [1.0, 1.0]),
            (len(ids), self.num_modes, 1),
            self.device,
        )

    def advance(self, relative_linear_acceleration):
        excitation = torch.as_tensor(
            relative_linear_acceleration,
            device=self.device,
            dtype=self.displacement.dtype,
        )
        if tuple(excitation.shape) != (self.num_envs, 3):
            raise ValueError(
                "relative_linear_acceleration must have shape "
                f"({self.num_envs}, 3), got {tuple(excitation.shape)}"
            )
        omega = 2.0 * torch.pi * self.frequency_hz
        # Four substeps are sufficient for the highest 14.5 Hz mode at the
        # fixed 50 Hz policy rate and avoid explicit-Euler energy growth.
        sub_dt = self.dt / 4.0
        factor = min(max(float(self.factor), 0.0), 1.0)
        force = (
            factor
            * self.drive_gain
            * excitation.unsqueeze(1)
            / self.num_modes
        )
        for _ in range(4):
            acceleration = (
                force
                - 2.0 * self.damping_ratio * omega * self.velocity
                - omega * omega * self.displacement
            )
            self.velocity += sub_dt * acceleration
            self.displacement += sub_dt * self.velocity
        self.total_deflection.copy_(torch.sum(self.displacement, dim=1))
        return self.total_deflection


class A2NeroIdentifiedSensorModel:
    """Policy-rate asynchronous student sensor model for identified DR.

    It deliberately returns only modeled student channels.  Callers keep the
    original rigid-body tensors for privileged/teacher observations.
    """

    def __init__(self, num_envs, device, dt, *, stage, config):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.stage = int(stage)
        self.config = config
        self.factor = 1.0
        self.time = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_dof_pos = torch.zeros(self.num_envs, 19, device=self.device)
        self.prev_dof_vel = torch.zeros_like(self.prev_dof_pos)
        self.prev_base_imu = torch.zeros(self.num_envs, 6, device=self.device)
        self.prev_ee_imu = torch.zeros_like(self.prev_base_imu)
        self.held_base_imu = torch.zeros_like(self.prev_base_imu)
        self.held_ee_imu = torch.zeros_like(self.prev_base_imu)
        self.gap_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.imu_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.lpf_state = torch.zeros_like(self.prev_base_imu)
        self.flex_x = torch.zeros(self.num_envs, 3, 6, device=self.device)
        self.flex_v = torch.zeros_like(self.flex_x)
        self._sample_episode(torch.arange(self.num_envs, device=self.device))

    @staticmethod
    def _uniform(bounds, shape, device):
        return float(bounds[0]) + (float(bounds[1]) - float(bounds[0])) * torch.rand(shape, device=device)

    def _sample_episode(self, ids):
        if len(ids) == 0:
            return
        sensor = self.config["sensor"]
        n = len(ids)
        self.a2_age = getattr(self, "a2_age", torch.zeros(self.num_envs, 1, device=self.device))
        self.nero_age = getattr(self, "nero_age", torch.zeros_like(self.a2_age))
        self.ee_age = getattr(self, "ee_age", torch.zeros_like(self.a2_age))
        self.dropout = getattr(self, "dropout", torch.zeros_like(self.a2_age))
        self.lpf_hz = getattr(self, "lpf_hz", torch.ones_like(self.a2_age))
        self.a2_age[ids] = self._uniform(sensor["a2_age_ms"], (n,1), self.device) / 1000.0
        self.nero_age[ids] = self._uniform(sensor["nero_age_ms"], (n,1), self.device) / 1000.0
        self.ee_age[ids] = self._uniform(sensor["ee_imu_age_ms"], (n,1), self.device) / 1000.0
        dropout_bounds = sensor.get("dropout_probability", [0.0, 0.0])
        self.dropout[ids] = self._uniform(dropout_bounds, (n,1), self.device)
        self.lpf_hz[ids] = self._uniform(sensor.get("relative_imu_lpf_hz", [1000,1000]), (n,1), self.device)
        self.imu_period = getattr(self, "imu_period", torch.full_like(self.a2_age, self.dt))
        period_ms = sensor.get("imu_period_ms", [self.dt*1000.0, self.dt*1000.0])
        self.imu_period[ids] = self._uniform(period_ms, (n,1), self.device) / 1000.0
        self.imu_phase[ids] = torch.rand(n,1,device=self.device) * self.imu_period[ids]

        frequencies = self.config.get("wrist_flex", {}).get(
            "frequency_bands_hz", [[5.5,7.0],[8.5,10.5],[11.5,14.5]]
        )
        self.flex_freq = getattr(self, "flex_freq", torch.zeros(self.num_envs, 3, 1, device=self.device))
        self.flex_zeta = getattr(self, "flex_zeta", torch.zeros_like(self.flex_freq))
        for mode, band in enumerate(frequencies):
            self.flex_freq[ids, mode] = self._uniform(band, (n,1), self.device)
        damping = self.config.get("wrist_flex", {}).get("damping_ratio", [0.05,0.25])
        self.flex_zeta[ids] = self._uniform(damping, (n,3,1), self.device)
        self.flex_axis_gain = getattr(self, "flex_axis_gain", torch.ones(self.num_envs,1,6,device=self.device))
        gain = self.config.get("wrist_flex", {}).get("axis_gain", [1.0,1.0])
        self.flex_axis_gain[ids] = self._uniform(gain, (n,1,6), self.device)
        self.flex_axis_mix = getattr(self, "flex_axis_mix", torch.zeros(self.num_envs,1,device=self.device))
        misalignment = self.config.get("wrist_flex", {}).get("axis_misalignment_deg", [0.0,0.0])
        self.flex_axis_mix[ids] = torch.tan(self._uniform(misalignment, (n,1), self.device) * torch.pi / 180.0)
        colored = self.config.get("colored_frequencies_hz", [1.5625,3.125,4.6875,6.25])
        self.colored_freq = torch.as_tensor(colored, device=self.device).reshape(1,-1,1)
        modes = len(colored)
        self.colored_phase = getattr(self, "colored_phase", torch.zeros(self.num_envs,modes,6,device=self.device))
        self.colored_amp = getattr(self, "colored_amp", torch.zeros_like(self.colored_phase))
        self.colored_phase[ids] = (2.0 * torch.pi * torch.rand(n,modes,6,device=self.device) - torch.pi)
        gyro = self.config.get("gyro_rms", [0.0,0.0]); accel = self.config.get("accel_rms", [0.0,0.0])
        amp = torch.cat((self._uniform(gyro,(n,3),self.device), self._uniform(accel,(n,3),self.device)), dim=1)
        self.colored_amp[ids] = amp[:,None,:] * (2.0 / max(modes,1)) ** 0.5

    def reset(self, ids, dof_pos, dof_vel, base_imu, ee_imu):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._sample_episode(ids)
        self.prev_dof_pos[ids] = dof_pos[ids]
        self.prev_dof_vel[ids] = dof_vel[ids]
        self.prev_base_imu[ids] = base_imu[ids]
        self.prev_ee_imu[ids] = ee_imu[ids]
        self.held_base_imu[ids] = base_imu[ids]
        self.held_ee_imu[ids] = ee_imu[ids]
        self.flex_x[ids] = 0.0; self.flex_v[ids] = 0.0; self.gap_ticks[ids] = 0

    def advance(self, dof_pos, dof_vel, base_imu, ee_imu):
        # Sensor age is asynchronous per sample, not an episode-persistent
        # fixed delay. Bounds stay episode-configured and ABI-neutral.
        sensor = self.config["sensor"]
        factor = min(max(float(self.factor), 0.0), 1.0)
        self.a2_age = factor * self._uniform(sensor["a2_age_ms"], (self.num_envs,1), self.device) / 1000.0
        self.nero_age = factor * self._uniform(sensor["nero_age_ms"], (self.num_envs,1), self.device) / 1000.0
        self.imu_phase = torch.remainder(self.imu_phase + self.dt, self.imu_period)
        ee_transport_age = self._uniform(sensor["ee_imu_age_ms"], (self.num_envs,1), self.device) / 1000.0
        self.ee_age = factor * torch.clamp(
            self.imu_phase + ee_transport_age,
            max=float(sensor["ee_imu_age_ms"][1]) / 1000.0,
        )
        a2_blend = torch.clamp(self.a2_age / self.dt, 0.0, 1.0)
        nero_blend = torch.clamp(self.nero_age / self.dt, 0.0, 1.0)
        student_pos = torch.cat((
            dof_pos[:,:12] * (1-a2_blend) + self.prev_dof_pos[:,:12] * a2_blend,
            dof_pos[:,12:] * (1-nero_blend) + self.prev_dof_pos[:,12:] * nero_blend,
        ), dim=1)
        student_vel = torch.cat((
            dof_vel[:,:12] * (1-a2_blend) + self.prev_dof_vel[:,:12] * a2_blend,
            dof_vel[:,12:] * (1-nero_blend) + self.prev_dof_vel[:,12:] * nero_blend,
        ), dim=1)
        ee_blend = torch.clamp(self.ee_age / self.dt, 0.0, 1.0)
        modeled_base = base_imu * (1-a2_blend) + self.prev_base_imu * a2_blend
        modeled_ee = ee_imu * (1-ee_blend) + self.prev_ee_imu * ee_blend
        relative = modeled_ee - modeled_base
        raw_relative = relative

        phase = 2.0 * torch.pi * self.time[:,None,:] * self.colored_freq + self.colored_phase
        relative = relative + factor * torch.sum(self.colored_amp * torch.sin(phase), dim=1)
        if self.stage >= 5 and bool(self.config.get("wrist_flex_enabled", True)):
            # Stable explicit integration of three independently sampled wrist
            # modes, excited by relative sensor acceleration/gyro innovation.
            excitation = (relative - self.lpf_state).unsqueeze(1)
            omega = 2.0 * torch.pi * self.flex_freq
            sub_dt = self.dt / 4.0
            for _ in range(4):
                flex_a = excitation - 2.0*self.flex_zeta*omega*self.flex_v - omega*omega*self.flex_x
                self.flex_v += sub_dt * flex_a
                self.flex_x += sub_dt * self.flex_v
            flex = torch.sum(self.flex_x * self.flex_axis_gain, dim=1)
            # Small-angle cross-axis mixing realizes the sampled mounting
            # misalignment without changing channel count.
            flex = flex + self.flex_axis_mix * torch.roll(flex, shifts=1, dims=1)
            relative = relative + factor * flex
            alpha = 1.0 - torch.exp(-2.0 * torch.pi * self.lpf_hz * self.dt)
            self.lpf_state += alpha * (relative - self.lpf_state)
            relative = (1.0-factor) * raw_relative + factor * self.lpf_state
            gap_cfg = self.config.get("gap_probability", 0.0)
            new_gap = torch.rand(self.num_envs, device=self.device) < factor * float(gap_cfg)
            gap_ms = self.config.get("gap_ms", [0,0])
            sampled_gap = torch.randint(max(1,int(gap_ms[0]/(self.dt*1000))), max(2,int(gap_ms[1]/(self.dt*1000))+1), (self.num_envs,), device=self.device)
            self.gap_ticks = torch.where(new_gap, sampled_gap, torch.clamp(self.gap_ticks-1,min=0))
            drop = (torch.rand(self.num_envs,device=self.device) < factor*self.dropout.squeeze(1)) | (self.gap_ticks > 0)
            modeled_base = apply_sample_hold(modeled_base, self.held_base_imu, drop)
            modeled_ee = modeled_base + apply_sample_hold(relative, self.held_ee_imu-self.held_base_imu, drop)
        self.held_base_imu.copy_(modeled_base); self.held_ee_imu.copy_(modeled_ee)
        self.prev_dof_pos.copy_(dof_pos); self.prev_dof_vel.copy_(dof_vel)
        self.prev_base_imu.copy_(base_imu); self.prev_ee_imu.copy_(ee_imu)
        self.time += self.dt
        return student_pos, student_vel, modeled_base, modeled_ee, modeled_ee-modeled_base


def _as_tensor_like(value: TensorOrScalar, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _normalized_quaternion_xyzw(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    if quaternion_xyzw.shape[-1] != 4:
        raise ValueError("Quaternion tensors must have final dimension 4 (xyzw)")
    norm = torch.linalg.vector_norm(quaternion_xyzw, dim=-1, keepdim=True)
    if bool(torch.any(norm <= torch.finfo(quaternion_xyzw.dtype).eps).item()):
        raise ValueError("Zero-length quaternion is not valid")
    return quaternion_xyzw / norm


def quat_rotate_xyzw(
    quaternion_xyzw: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate ``vector`` from local coordinates to world coordinates."""

    if vector.shape[-1] != 3:
        raise ValueError("Vector tensors must have final dimension 3")
    quaternion_xyzw = _normalized_quaternion_xyzw(quaternion_xyzw)
    q_vector = quaternion_xyzw[..., :3]
    q_w = quaternion_xyzw[..., 3:4]
    # torch.linalg.cross requires equal ranks even for otherwise broadcastable
    # inputs (for example q=[N,4], vector=[3]), so broadcast explicitly.
    q_vector, vector = torch.broadcast_tensors(q_vector, vector)
    q_w, _ = torch.broadcast_tensors(q_w, vector[..., :1])
    first_cross = torch.cross(q_vector, vector, dim=-1)
    return vector + 2.0 * (
        q_w * first_cross + torch.cross(q_vector, first_cross, dim=-1)
    )


def quat_rotate_inverse_xyzw(
    quaternion_xyzw: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate a world-coordinate vector into the quaternion's local frame."""

    quaternion_xyzw = _normalized_quaternion_xyzw(quaternion_xyzw)
    conjugate = torch.cat(
        (-quaternion_xyzw[..., :3], quaternion_xyzw[..., 3:4]), dim=-1
    )
    return quat_rotate_xyzw(conjugate, vector)


def point_offset_world(
    body_orientation_xyzw: torch.Tensor, point_offset_body: torch.Tensor
) -> torch.Tensor:
    """Transform a COM-to-sensor offset from body coordinates to world."""

    return quat_rotate_xyzw(body_orientation_xyzw, point_offset_body)


def point_velocity_world(
    com_linear_velocity_world: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    body_orientation_xyzw: torch.Tensor,
    point_offset_body: torch.Tensor,
) -> torch.Tensor:
    """Velocity of a fixed sensor point: ``v_com + omega x r``."""

    offset_world = point_offset_world(body_orientation_xyzw, point_offset_body)
    return com_linear_velocity_world + torch.cross(
        angular_velocity_world, offset_world, dim=-1
    )


def point_acceleration_world(
    com_linear_acceleration_world: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    angular_acceleration_world: torch.Tensor,
    body_orientation_xyzw: torch.Tensor,
    point_offset_body: torch.Tensor,
) -> torch.Tensor:
    """Acceleration of a rigidly fixed sensor point.

    ``a_point = a_com + alpha x r + omega x (omega x r)``.  The final term is
    centripetal acceleration and is required even at constant angular speed.
    """

    offset_world = point_offset_world(body_orientation_xyzw, point_offset_body)
    tangential = torch.cross(angular_acceleration_world, offset_world, dim=-1)
    centripetal = torch.cross(
        angular_velocity_world,
        torch.cross(angular_velocity_world, offset_world, dim=-1),
        dim=-1,
    )
    return com_linear_acceleration_world + tangential + centripetal


def finite_difference(
    current: torch.Tensor, previous: torch.Tensor, dt: TensorOrScalar
) -> torch.Tensor:
    """Finite difference with a checked positive timestep."""

    dt_tensor = _as_tensor_like(dt, current)
    if bool(torch.any(dt_tensor <= 0.0).item()):
        raise ValueError("dt must be positive")
    return (current - previous) / dt_tensor


def specific_force_sensor_frame(
    linear_acceleration_world: torch.Tensor,
    sensor_orientation_xyzw: torch.Tensor,
    gravity_world: TensorOrScalar = (0.0, 0.0, -9.81),
) -> torch.Tensor:
    """Ideal raw accelerometer specific force ``R^-1 (a_world - gravity)``."""

    gravity = _as_tensor_like(gravity_world, linear_acceleration_world)
    return quat_rotate_inverse_xyzw(
        sensor_orientation_xyzw, linear_acceleration_world - gravity
    )


def remove_gravity_from_specific_force(
    specific_force_sensor: torch.Tensor,
    sensor_orientation_xyzw: torch.Tensor,
    gravity_world: TensorOrScalar = (0.0, 0.0, -9.81),
) -> torch.Tensor:
    """Recover gravity-removed kinematic acceleration in sensor coordinates."""

    gravity = _as_tensor_like(gravity_world, specific_force_sensor)
    gravity_sensor = quat_rotate_inverse_xyzw(sensor_orientation_xyzw, gravity)
    return specific_force_sensor + gravity_sensor


def gravity_removed_imu(
    angular_velocity_world: torch.Tensor,
    linear_acceleration_world: torch.Tensor,
    sensor_orientation_xyzw: torch.Tensor,
    gravity_world: TensorOrScalar = (0.0, 0.0, -9.81),
) -> torch.Tensor:
    """Build ``[gyro, gravity-removed linear acceleration]`` in sensor axes."""

    gyro_sensor = quat_rotate_inverse_xyzw(
        sensor_orientation_xyzw, angular_velocity_world
    )
    specific_force = specific_force_sensor_frame(
        linear_acceleration_world, sensor_orientation_xyzw, gravity_world
    )
    linear_acceleration_sensor = remove_gravity_from_specific_force(
        specific_force, sensor_orientation_xyzw, gravity_world
    )
    return torch.cat((gyro_sensor, linear_acceleration_sensor), dim=-1)


def relative_imu(ee_imu: torch.Tensor, base_imu: torch.Tensor) -> torch.Tensor:
    """Derive the relative channel exactly once from modeled EE and base IMUs."""

    if ee_imu.shape != base_imu.shape or ee_imu.shape[-1] != 6:
        raise ValueError("base_imu and ee_imu must have identical (..., 6) shapes")
    return ee_imu - base_imu


def quantize(value: torch.Tensor, step: TensorOrScalar) -> torch.Tensor:
    """Round to the nearest configurable quantum; zero disables quantization."""

    step_tensor = _as_tensor_like(step, value)
    if bool(torch.any(step_tensor < 0.0).item()):
        raise ValueError("Quantization step must be non-negative")
    enabled = step_tensor > 0.0
    safe_step = torch.where(enabled, step_tensor, torch.ones_like(step_tensor))
    quantized = torch.round(value / safe_step) * safe_step
    return torch.where(enabled, quantized, value)


def apply_sample_hold(
    current_sample: torch.Tensor,
    previous_sample: torch.Tensor,
    hold_mask: torch.Tensor,
) -> torch.Tensor:
    """Select a previous sample for held environments without mutating state."""

    if current_sample.shape != previous_sample.shape:
        raise ValueError("current_sample and previous_sample must have equal shapes")
    mask = hold_mask.to(device=current_sample.device, dtype=torch.bool)
    while mask.ndim < current_sample.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, previous_sample, current_sample)


def apply_bias_scale_and_quantization(
    value: torch.Tensor,
    *,
    bias: Optional[TensorOrScalar] = None,
    multiplicative_scale: Optional[TensorOrScalar] = None,
    quantization_step: TensorOrScalar = 0.0,
) -> torch.Tensor:
    """Apply episode-persistent scale/bias, then optional quantization."""

    modeled = value
    if multiplicative_scale is not None:
        modeled = modeled * _as_tensor_like(multiplicative_scale, value)
    if bias is not None:
        modeled = modeled + _as_tensor_like(bias, value)
    return quantize(modeled, quantization_step)


__all__ = [
    "apply_bias_scale_and_quantization",
    "apply_sample_hold",
    "finite_difference",
    "gravity_removed_imu",
    "point_acceleration_world",
    "point_offset_world",
    "point_velocity_world",
    "quantize",
    "quat_rotate_inverse_xyzw",
    "quat_rotate_xyzw",
    "relative_imu",
    "remove_gravity_from_specific_force",
    "specific_force_sensor_frame",
]
