"""Linux SocketCAN BCM writer for deterministic Nero MIT keepalive frames."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

import can
import numpy as np

from pyAgxArm.utiles.numeric_codec import NumericCodec as nc


NERO_MIT_FIRST_ID = 0x15A
NERO_MIT_LAST_ID = 0x160


def build_nero_mit_frames(
    robot,
    target: Sequence[float],
    *,
    kp: float | Sequence[float],
    kd: float | Sequence[float],
) -> list[can.Message]:
    """Pack seven zero-velocity/zero-feedforward frames as the SDK.

    ``kp`` and ``kd`` may be scalars or seven-axis vectors.  Per-axis gains
    are needed for bumpless cumulative enable: a joint first releases its
    brake with zero position stiffness while joints already commissioned keep
    holding their measured anchors.
    """
    positions = np.asarray(target, dtype=np.float64)
    if positions.shape != (7,) or not np.all(np.isfinite(positions)):
        raise ValueError(f"Nero MIT target must be a finite seven-vector: {positions}")
    if np.any(positions < -12.5) or np.any(positions > 12.5):
        raise ValueError(f"Nero MIT target exceeds protocol position range: {positions}")

    def gain_vector(value, *, name: str, lower: float, upper: float):
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(7, float(values), dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Nero MIT {name} must be a finite scalar or seven-vector: "
                f"{values}"
            )
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(
                f"Nero MIT {name} is outside [{lower:g}, {upper:g}]: "
                f"{values}"
            )
        return values

    kp_values = gain_vector(kp, name="Kp", lower=0.0, upper=500.0)
    kd_values = gain_vector(kd, name="Kd", lower=-5.0, upper=5.0)

    velocity_uint = nc.FloatToUint(0.0, -45.0, 45.0, 12)
    # Nero V1.20 inherits the V1.11 MIT payload: t_ff is a 12-bit value over
    # [-16, 16] N*m.  The pre-V1.11 8-bit encoding would put integer 127 into
    # this 12-bit field, which the drive interprets as roughly -15 N*m rather
    # than zero.
    torque_uint = nc.FloatToUint(0.0, -16.0, 16.0, 12)
    frames: list[can.Message] = []
    for zero_index, position in enumerate(positions):
        joint_index = zero_index + 1
        protocol_msg = robot._parser._make_joint_mit_ctrl_msg(
            joint_index=joint_index,
            p_des=nc.FloatToUint(float(position), -12.5, 12.5, 16),
            v_des=velocity_uint,
            kp=nc.FloatToUint(float(kp_values[zero_index]), 0.0, 500.0, 12),
            kd=nc.FloatToUint(float(kd_values[zero_index]), -5.0, 5.0, 12),
            t_ff=torque_uint,
        )
        frame = robot._parser.pack(protocol_msg)
        expected_id = NERO_MIT_FIRST_ID + joint_index - 1
        if frame is None or int(frame.arbitration_id) != expected_id:
            actual = None if frame is None else hex(int(frame.arbitration_id))
            raise RuntimeError(
                f"SDK packed unexpected Nero J{joint_index} MIT CAN ID: {actual}"
            )
        frames.append(frame)
    return frames


class NeroMitKernelWriter:
    """Keep seven Nero MIT frames periodic in the Linux CAN BCM scheduler.

    Policy code only replaces the current payloads. Periodic transmission is
    performed by the kernel and therefore does not depend on Python thread/GIL
    scheduling or on the SDK's receive socket.
    """

    def __init__(self, can_interface: str, *, period_s: float = 0.010):
        if not 0.005 <= float(period_s) <= 0.020:
            raise ValueError("Nero MIT kernel period must be in [0.005, 0.020] s")
        self.can_interface = str(can_interface)
        self.period_s = float(period_s)
        self._bus = None
        self._tasks: list[object] = []
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._update_count = 0
        self._last_update_monotonic: float | None = None
        self._max_update_gap_s = 0.0

    @staticmethod
    def _validate_frames(frames: Sequence[can.Message]) -> list[can.Message]:
        values = list(frames)
        ids = [int(frame.arbitration_id) for frame in values]
        expected = list(range(NERO_MIT_FIRST_ID, NERO_MIT_LAST_ID + 1))
        if ids != expected:
            raise ValueError(f"Nero MIT frames must have IDs {expected}, got {ids}")
        if any(frame.is_extended_id or len(frame.data) != 8 for frame in values):
            raise ValueError("Nero MIT frames must be seven classic 8-byte CAN frames")
        return values

    def start(self, frames: Sequence[can.Message]) -> None:
        values = self._validate_frames(frames)
        with self._lock:
            if self._tasks:
                raise RuntimeError("Nero MIT kernel writer is already running")
            self._error = None
            self._bus = can.Bus(
                interface="socketcan",
                channel=self.can_interface,
                # Enable SocketCAN local echo so the independent passive
                # receiver can prove that BCM frames reached the CAN stack.
                local_loopback=True,
                receive_own_messages=False,
            )
            try:
                self._tasks = []
                for frame in values:
                    self._tasks.append(self._bus.send_periodic(
                        frame,
                        period=self.period_s,
                        store_task=False,
                    ))
            except BaseException as exc:
                self._error = exc
                self._stop_locked()
                raise
            now = time.monotonic()
            self._last_update_monotonic = now
            self._max_update_gap_s = 0.0
            self._update_count = 1

    def update(self, frames: Sequence[can.Message]) -> None:
        values = self._validate_frames(frames)
        with self._lock:
            if not self._tasks:
                raise RuntimeError("Nero MIT kernel writer is not running")
            try:
                for task, frame in zip(self._tasks, values, strict=True):
                    task.modify_data(frame)
            except BaseException as exc:
                self._error = exc
                raise
            now = time.monotonic()
            if self._last_update_monotonic is not None:
                self._max_update_gap_s = max(
                    self._max_update_gap_s, now - self._last_update_monotonic
                )
            self._last_update_monotonic = now
            self._update_count += 1

    def _stop_locked(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            try:
                task.stop()
            except Exception:
                pass
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.shutdown()
            except Exception:
                pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    @property
    def running(self) -> bool:
        with self._lock:
            return len(self._tasks) == 7 and self._error is None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def health_summary(self) -> str:
        with self._lock:
            return (
                f"kernel_bcm_running={len(self._tasks) == 7}, "
                f"period_ms={1000.0 * self.period_s:.1f}, "
                f"target_updates={self._update_count}, "
                f"max_target_update_gap_ms={1000.0 * self._max_update_gap_s:.1f}, "
                f"error={self._error!r}"
            )
