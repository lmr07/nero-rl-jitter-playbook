"""Safety helpers for Nero V112 commissioning over the external CAN bus."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence

import numpy as np


ALL_DISABLED = [False] * 7
NERO_HANDOFF_DRIFT_LIMIT_RAD = 0.025
NERO_HANDOFF_POSITION_STEP_LIMIT_RAD = 0.015
NERO_HANDOFF_HIGH_SPEED_RAD_S = 1.0
NERO_HANDOFF_HARD_SPEED_RAD_S = 2.5
NERO_HANDOFF_HIGH_SPEED_DRIFT_RAD = 0.010
NERO_HANDOFF_HIGH_SPEED_MAX_CONSECUTIVE = 1


def evaluate_nero_handoff_motion(
    measured: Sequence[float],
    velocity: Sequence[float],
    hold: Sequence[float],
    previous_measured: Sequence[float],
    high_speed_streak: Sequence[int],
) -> dict[str, object]:
    """Classify one fresh CPV-to-MIT cohort without trusting velocity alone.

    Nero can publish one native-velocity impulse at a mode boundary while its
    position changes by only a few encoder quanta. Such a sample is not enough
    to prove physical instability, but it resets the later continuous-stability
    window. Position drift/step, an extreme velocity, or a second consecutive
    >1 rad/s cohort remains an immediate stop.
    """
    position = np.asarray(measured, dtype=np.float64)
    speed_signed = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(hold, dtype=np.float64)
    previous = np.asarray(previous_measured, dtype=np.float64)
    streak = np.asarray(high_speed_streak, dtype=np.int64)
    if any(array.shape != (7,) for array in (
        position, speed_signed, target, previous, streak
    )):
        raise ValueError("Nero handoff motion guard requires seven-axis vectors")
    if not all(np.all(np.isfinite(array)) for array in (
        position, speed_signed, target, previous
    )):
        raise ValueError("Nero handoff motion guard requires finite feedback")
    if np.any(streak < 0):
        raise ValueError("Nero handoff speed streak cannot be negative")

    drift = np.abs(position - target)
    position_step = np.abs(position - previous)
    speed = np.abs(speed_signed)
    high = speed > NERO_HANDOFF_HIGH_SPEED_RAD_S
    next_streak = np.where(high, streak + 1, 0)

    violation: dict[str, object] | None = None
    candidates = (
        ("drift", drift > NERO_HANDOFF_DRIFT_LIMIT_RAD, drift),
        (
            "position_step",
            position_step > NERO_HANDOFF_POSITION_STEP_LIMIT_RAD,
            position_step,
        ),
        ("hard_speed", speed > NERO_HANDOFF_HARD_SPEED_RAD_S, speed),
        (
            "high_speed_with_drift",
            high & (drift > NERO_HANDOFF_HIGH_SPEED_DRIFT_RAD),
            speed,
        ),
        (
            "consecutive_high_speed",
            next_streak > NERO_HANDOFF_HIGH_SPEED_MAX_CONSECUTIVE,
            speed,
        ),
    )
    for reason, mask, ranking in candidates:
        if np.any(mask):
            masked = np.where(mask, ranking, -np.inf)
            violation = {
                "reason": reason,
                "joint": int(np.argmax(masked)),
            }
            break

    return {
        "drift": drift,
        "position_step": position_step,
        "speed": speed,
        "high_speed_streak": next_streak,
        "isolated_high_speed": high & (next_streak == 1),
        "violation": violation,
    }


def parse_firmware_version(value: str) -> tuple[int, int]:
    """Parse the Nero controller's numeric ``major.minor`` version."""
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(value))
    if match is None:
        raise ValueError(f"Invalid Nero firmware version: {value!r}")
    return int(match.group(1)), int(match.group(2))


def require_firmware_at_least(
    robot,
    minimum: tuple[int, int],
    timeout_s: float,
) -> str:
    """Read the controller version and reject an incompatible SDK selection.

    Immediately reconnecting after a verified disable can leave the V1.20
    controller streaming motor feedback before its request/reply firmware
    service answers.  Treat a missing software-version field as transient
    inside the caller's existing bounded startup timeout.  A real version is
    still parsed and rejected immediately when it is too old or malformed.
    """
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Nero firmware query timeout must be positive")
    deadline = time.monotonic() + timeout
    attempts = 0
    last_firmware = None
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        attempts += 1
        firmware = robot.get_firmware(
            timeout=min(remaining, 1.0),
            min_interval=0.0,
        )
        last_firmware = firmware
        if firmware is not None and "software_version" in firmware:
            version_text = str(firmware["software_version"])
            version = parse_firmware_version(version_text)
            if version < tuple(minimum):
                raise RuntimeError(
                    f"Nero firmware {version_text} is older than required "
                    f"{minimum[0]}.{minimum[1]:02d}"
                )
            return version_text
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(
        "Nero firmware query returned no software version after "
        f"{attempts} attempt(s) in {timeout:.1f}s; last response={last_firmware!r}. "
        "Motor feedback alone does not prove that the controller request/reply "
        "service recovered after a prior mode/control failure. Do not retry "
        "active control: "
        "power-cycle the Nero controller, reinitialize SocketCAN, and pass a "
        "read-only firmware/feedback check first"
    )


def motor_position_velocity_snapshot(
    robot,
) -> tuple[list[float], list[float], list[float]] | None:
    """Return complete V1.20 motor position, velocity and timestamps."""
    positions: list[float] = []
    velocities: list[float] = []
    stamps: list[float] = []
    for joint_index in range(1, 8):
        state = robot.get_motor_states(joint_index)
        if state is None:
            return None
        position = float(state.msg.position)
        velocity = float(state.msg.velocity)
        stamp = float(state.timestamp)
        if not all(math.isfinite(value) for value in (position, velocity, stamp)):
            return None
        positions.append(position)
        velocities.append(velocity)
        stamps.append(stamp)
    return positions, velocities, stamps


def external_joint_position_snapshot(
    robot,
) -> tuple[list[float], list[float]] | None:
    """Return raw 0x511..0x517 positions and receive timestamps.

    This channel is deliberately kept separate from ``get_joint_angles()``:
    the commissioned hardware reports it in the internal motor coordinate
    system, whose sign is opposite to the Web/URDF coordinate on all joints.
    """
    feedback = robot.get_external_joint_states()
    if feedback is None:
        return None
    positions = [float(value) for value in feedback.msg]
    stamp = float(feedback.timestamp)
    if len(positions) != 7 or not all(math.isfinite(value) for value in positions):
        return None
    if not math.isfinite(stamp):
        return None
    # The aggregate timestamp is the oldest of the seven component frames.
    return positions, [stamp] * 7


def wait_for_live_external_joint_positions(
    robot,
    timeout_s: float,
    *,
    max_age_s: float = 0.20,
) -> tuple[list[float], list[float]]:
    """Require a complete, recent, advancing 0x511..0x517 snapshot."""
    deadline = time.monotonic() + timeout_s
    baseline_stamp: float | None = None
    last_stamp: float | None = None
    while time.monotonic() < deadline:
        snapshot = external_joint_position_snapshot(robot)
        if snapshot is not None:
            positions, stamps = snapshot
            last_stamp = min(stamps)
            if baseline_stamp is None:
                baseline_stamp = last_stamp
            elif (
                last_stamp > baseline_stamp
                and feedback_timestamp_age_s(last_stamp) <= max_age_s
            ):
                return positions, stamps
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.01)
    raise TimeoutError(
        "Incomplete, stale, or non-advancing Nero external joint feedback "
        f"(0x511..0x517); baseline={baseline_stamp}, last={last_stamp}"
    )


def feedback_timestamp_age_s(stamp: float) -> float:
    """Return receive age for either wall-clock or monotonic SDK timestamps."""
    stamp = float(stamp)
    if not math.isfinite(stamp):
        return math.inf
    candidates = (time.time() - stamp, time.monotonic() - stamp)
    # python-can normally supplies Unix wall-clock timestamps, while test and
    # alternate backends may use a monotonic clock. Pick the plausible clock
    # without treating a large negative cross-clock difference as fresh.
    nonnegative = [age for age in candidates if age >= -0.01]
    if not nonnegative:
        return math.inf
    return max(0.0, min(nonnegative))


def motor_position_snapshot(robot) -> tuple[list[float], list[float]] | None:
    """Return the seven live 0x251..0x257 motor positions and timestamps.

    V112's controller-level 0x2A5..0x2A9 joint-angle frames can remain
    timestamp-fresh but position-frozen in MOVE_MIT.  The per-drive high-speed
    frames are the feedback source belonging to the MIT passthrough loop.
    """
    positions: list[float] = []
    stamps: list[float] = []
    for joint_index in range(1, 8):
        state = robot.get_motor_states(joint_index)
        if state is None:
            return None
        position = float(state.msg.position)
        stamp = float(state.timestamp)
        if not (math.isfinite(position) and math.isfinite(stamp)):
            return None
        positions.append(position)
        stamps.append(stamp)
    return positions, stamps


def wait_for_live_motor_positions(
    robot,
    timeout_s: float,
    *,
    max_age_s: float = 0.20,
) -> tuple[list[float], list[float]]:
    """Require a complete, recent and advancing 0x251..0x257 snapshot."""
    deadline = time.monotonic() + timeout_s
    baseline_stamps: list[float] | None = None
    last_stamps: list[float] | None = None
    while time.monotonic() < deadline:
        snapshot = motor_position_snapshot(robot)
        if snapshot is not None:
            positions, stamps = snapshot
            last_stamps = stamps
            if baseline_stamps is None:
                baseline_stamps = list(stamps)
            elif (
                all(new > old for new, old in zip(stamps, baseline_stamps, strict=True))
                and all(feedback_timestamp_age_s(stamp) <= max_age_s for stamp in stamps)
            ):
                return positions, stamps
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.01)
    raise TimeoutError(
        "Incomplete, stale, or non-advancing Nero MIT motor feedback "
        f"(0x251..0x257); baseline={baseline_stamps}, last={last_stamps}"
    )


def format_status_diagnostics(robot) -> str:
    """Return a best-effort one-line controller/driver diagnostic snapshot."""
    parts: list[str] = []
    try:
        arm_status = robot.get_arm_status()
    except Exception as exc:  # Diagnostics must not hide the original failure.
        parts.append(f"0x2A1=error({exc})")
    else:
        if arm_status is None:
            parts.append("0x2A1=unavailable")
        else:
            msg = arm_status.msg
            parts.append(
                "0x2A1("
                f"ctrl_mode={getattr(msg, 'ctrl_mode', 'unknown')}, "
                f"arm_status={getattr(msg, 'arm_status', 'unknown')}, "
                f"mode={getattr(msg, 'mode_feedback', 'unknown')})"
            )

    for joint_index in range(1, 8):
        try:
            state = robot.get_driver_states(joint_index)
        except Exception as exc:  # See comment above: preserve original error.
            parts.append(f"J{joint_index}=error({exc})")
            continue
        if state is None:
            parts.append(f"J{joint_index}=unavailable")
            continue
        msg = state.msg
        foc_status = getattr(msg, "foc_status", None)
        try:
            flags = vars(foc_status) if foc_status is not None else {}
        except TypeError:
            flags = {"value": str(foc_status)}
        status_code = getattr(msg, "foc_status_code", None)
        status_text = (
            "unknown" if status_code is None else f"0x{int(status_code):02X}"
        )
        parts.append(
            f"J{joint_index}(status={status_text}, "
            f"voltage={getattr(msg, 'vol', 'unknown')}, "
            f"driver_temp={getattr(msg, 'foc_temp', 'unknown')}, "
            f"motor_temp={getattr(msg, 'motor_temp', 'unknown')}, "
            f"flags={flags})"
        )
    return "; ".join(parts)


def driver_enable_snapshot(robot) -> tuple[list[bool], list[float]] | None:
    """Return a complete enable-bit/timestamp snapshot for 0x261..0x267."""
    statuses: list[bool] = []
    stamps: list[float] = []
    for joint_index in range(1, 8):
        state = robot.get_driver_states(joint_index)
        if state is None:
            return None
        statuses.append(bool(state.msg.foc_status.driver_enable_status))
        stamps.append(float(state.timestamp))
    return statuses, stamps


def refresh_disabled_position_targets(
    robot,
    target,
    measured,
) -> float:
    """Refresh only not-yet-enabled axes from current physical feedback.

    V1.20 cumulative J1..J7 enable handshakes can take several seconds.  A
    fixed seven-axis target captured before J1 is enabled is therefore stale
    by the time a gravity-loaded later joint is enabled.  Follow physical
    feedback only while an axis is disabled, then freeze that axis at the
    last measured position as soon as its enable bit is acknowledged.
    """
    if len(target) != 7 or len(measured) != 7:
        raise ValueError("Nero disabled-target refresh requires seven axes")
    snapshot = driver_enable_snapshot(robot)
    if snapshot is None:
        raise RuntimeError(
            "Cannot refresh Nero disabled targets without driver feedback"
        )
    statuses = snapshot[0]
    maximum_change = 0.0
    for index, enabled in enumerate(statuses):
        if enabled:
            continue
        value = float(measured[index])
        previous = float(target[index])
        if not math.isfinite(value) or not math.isfinite(previous):
            raise RuntimeError(
                f"Invalid Nero J{index + 1} disabled-target feedback"
            )
        maximum_change = max(maximum_change, abs(value - previous))
        target[index] = value
    return maximum_change


def wait_for_external_feedback(robot, timeout_s: float):
    """Wait for complete joint-angle and per-driver external-CAN feedback."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        joints = robot.get_joint_angles()
        snapshot = driver_enable_snapshot(robot)
        arm_status = robot.get_arm_status()
        if joints is not None and snapshot is not None and arm_status is not None:
            return joints, snapshot, arm_status
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.01)
    raise TimeoutError(
        "Incomplete Nero V112 external feedback "
        "(0x2A1/0x2A5..0x2A9/0x261..0x267)"
    )


def enter_can_control_disabled(
    robot,
    timeout_s: float,
    *,
    stable_s: float = 0.5,
) -> None:
    """Enter CAN_CTRL with the hardware-proven zero-filled 0x151 frame.

    This function refuses to transmit unless all seven joints report disabled.
    It sends no enable or target command.
    """
    snapshot = driver_enable_snapshot(robot)
    arm_status = robot.get_arm_status()
    if snapshot is None or arm_status is None:
        raise RuntimeError("Cannot switch Nero mode without complete feedback")
    if snapshot[0] != ALL_DISABLED:
        raise RuntimeError(
            "CAN-control switch requires all seven joints disabled; "
            f"status={snapshot[0]}"
        )

    initial_status_stamp = float(arm_status.timestamp)
    msg = robot._MSG_ModeCtrl(
        ctrl_mode=0x01,
        move_mode=0x00,
        move_spd_rate_ctrl=0,
        mit_mode=0x00,
        residence_time=0,
        installation_pos=0x00,
        enable_can_push=0x00,
    )
    robot._send_msg(msg)

    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    last_mode = None
    while time.monotonic() < deadline:
        snapshot = driver_enable_snapshot(robot)
        arm_status = robot.get_arm_status()
        if snapshot is None or arm_status is None:
            stable_since = None
        else:
            last_mode = arm_status.msg.ctrl_mode
            if snapshot[0] != ALL_DISABLED:
                raise RuntimeError(
                    "A Nero joint became enabled during CAN-control switch: "
                    f"{snapshot[0]}"
                )
            if (
                arm_status.msg.ctrl_mode == robot.ARM_STATUS.CtrlMode.CAN_CTRL
                and float(arm_status.timestamp) > initial_status_stamp
            ):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_s:
                    return
            else:
                stable_since = None
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.02)
    raise TimeoutError(f"Nero did not enter stable CAN_CTRL; last mode={last_mode}")


def clear_emergency_stop_disabled(
    robot,
    timeout_s: float,
    *,
    stable_s: float = 0.5,
    retry_interval_s: float = 0.10,
) -> None:
    """Retry the documented reset while all joints stay disabled.

    Nero V112 may not act on a single external-CAN 0x150 reset frame.  The
    command is idempotent and does not enable motors or contain a target, so
    retry it at a bounded rate while fresh 0x261..0x267 feedback continues to
    prove that every joint remains disabled.
    """
    if retry_interval_s <= 0.0:
        raise ValueError("retry_interval_s must be positive")
    snapshot = driver_enable_snapshot(robot)
    arm_status = robot.get_arm_status()
    if snapshot is None or arm_status is None:
        raise RuntimeError("Cannot reset Nero without complete feedback")
    if snapshot[0] != ALL_DISABLED:
        raise RuntimeError(
            "Nero reset requires all seven joints disabled; "
            f"status={snapshot[0]}"
        )
    initial_status_stamp = float(arm_status.timestamp)

    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    last_arm_status = None
    next_reset_time = 0.0
    reset_attempts = 0
    while time.monotonic() < deadline:
        snapshot = driver_enable_snapshot(robot)
        arm_status = robot.get_arm_status()
        if snapshot is None or arm_status is None:
            stable_since = None
        else:
            last_arm_status = arm_status.msg.arm_status
            if snapshot[0] != ALL_DISABLED:
                raise RuntimeError(
                    "A Nero joint became enabled during controller reset: "
                    f"{snapshot[0]}"
                )
            if (
                arm_status.msg.arm_status
                != robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP
                and float(arm_status.timestamp) > initial_status_stamp
            ):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_s:
                    return
            else:
                stable_since = None
                now = time.monotonic()
                if now >= next_reset_time:
                    robot.reset()
                    reset_attempts += 1
                    next_reset_time = now + retry_interval_s
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.02)
    raise TimeoutError(
        "Nero emergency stop did not clear while disabled; "
        f"last arm status={last_arm_status}, reset attempts={reset_attempts}"
    )


def handoff_enabled_mit_to_cpv(
    robot,
    target: Sequence[float],
    timeout_s: float,
    *,
    control_dt_s: float = 0.02,
) -> list[float]:
    """Switch a measured enabled MIT hold to CPV without a target jump.

    Real Nero V1.20 ignores MIT-to-CPV mode requests after all joints have
    already been disabled.  Cleanup must therefore preseed the current
    seven-axis target, prove MOVE_CPV while all seven axes are still enabled,
    and only then request disable.  This helper sends no new motion target:
    callers pass the freshly measured MIT pose and it is held continuously
    across the mode boundary.
    """
    values = [float(value) for value in target]
    if len(values) != 7 or any(not math.isfinite(value) for value in values):
        raise ValueError(
            "Nero MIT-to-CPV cleanup target must contain seven finite values"
        )
    if timeout_s <= 0.0 or control_dt_s <= 0.0:
        raise ValueError("Nero MIT-to-CPV cleanup timing must be positive")
    snapshot = driver_enable_snapshot(robot)
    status = robot.get_arm_status()
    feedback = robot.get_joint_angles()
    if (
        snapshot is None
        or snapshot[0] != [True] * 7
        or status is None
        or feedback is None
    ):
        actual = None if snapshot is None else snapshot[0]
        raise RuntimeError(
            "Nero MIT-to-CPV cleanup requires all seven joints enabled and "
            f"complete feedback; enable status={actual}"
        )
    initial_status_stamp = float(status.timestamp)
    last_feedback_stamp = float(feedback.timestamp)

    def send_target() -> None:
        for joint_index, position in enumerate(values, start=1):
            robot.move_cpv_pos(joint_index, position)

    # CPV writes are inert while MOVE_MIT is active, but preload the exact
    # physical pose so selecting CPV cannot expose a stale controller target.
    send_target()
    robot.set_motion_mode("cpv")
    deadline = time.monotonic() + float(timeout_s)
    next_mode_request = time.monotonic() + 0.20
    resumed_samples = 0
    latest = list(values)
    last_mode = status.msg.mode_feedback
    while time.monotonic() < deadline:
        started = time.monotonic()
        send_target()
        snapshot = driver_enable_snapshot(robot)
        status = robot.get_arm_status()
        feedback = robot.get_joint_angles()
        if snapshot is None or snapshot[0] != [True] * 7:
            actual = None if snapshot is None else snapshot[0]
            raise RuntimeError(
                "Nero enable state changed during MIT-to-CPV cleanup: "
                f"{actual}"
            )
        if status is not None:
            last_mode = status.msg.mode_feedback
        if feedback is not None and status is not None:
            mode_ready = (
                status.msg.mode_feedback
                == robot.ARM_STATUS.ModeFeedback.MOVE_CPV
                and float(status.timestamp) > initial_status_stamp
            )
            feedback_stamp = float(feedback.timestamp)
            if mode_ready and feedback_stamp > last_feedback_stamp:
                measured = [float(value) for value in feedback.msg]
                if len(measured) == 7 and all(
                    math.isfinite(value) for value in measured
                ):
                    latest = measured
                    last_feedback_stamp = feedback_stamp
                    resumed_samples += 1
                    # The aggregate 0x2A5..0x2A9 sample can straddle the mode
                    # boundary.  Accept only the following complete sample.
                    if resumed_samples >= 2:
                        return latest
        now = time.monotonic()
        if now >= next_mode_request:
            robot.set_motion_mode("cpv")
            next_mode_request = now + 0.20
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        remaining = control_dt_s - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
    raise TimeoutError(
        "Nero did not complete enabled measured-pose MIT-to-CPV cleanup; "
        f"last mode={last_mode}, resumed samples={resumed_samples}"
    )


def restore_disabled_read_service(
    robot,
    timeout_s: float,
    *,
    minimum_firmware: tuple[int, int] = (1, 21),
    retry_interval_s: float = 0.20,
) -> str:
    """Prove the read service after a verified enabled MIT-to-CPV handoff.

    V1.20 can keep streaming 0x251..0x257 after an MIT abort while its
    firmware request/reply service remains unavailable.  Hardware testing
    also shows it ignores a first MIT-to-CPV request after all axes are
    disabled.  Normal cleanup therefore calls :func:`handoff_enabled_mit_to_cpv`
    first; this disabled check then proves both the CPV state and firmware
    service before another active run is allowed. Do not issue a controller
    reset here: real V1.20 testing showed repeated disabled resets can change
    MOVE_CPV back to MOVE_MIT and make the following enable service reject
    otherwise valid J1 requests. A new SDK connection performs the mandatory
    firmware preflight before any later active command.
    """
    if timeout_s <= 0.0 or retry_interval_s <= 0.0:
        raise ValueError("Nero read-service recovery timing must be positive")
    snapshot = driver_enable_snapshot(robot)
    status = robot.get_arm_status()
    if snapshot is None or status is None:
        raise RuntimeError(
            "Cannot restore Nero read service without complete feedback"
        )
    if snapshot[0] != ALL_DISABLED:
        raise RuntimeError(
            "Nero read-service recovery requires all joints disabled; "
            f"status={snapshot[0]}"
        )
    initial_status_stamp = float(status.timestamp)
    deadline = time.monotonic() + float(timeout_s)
    next_mode_request = 0.0
    last_mode = None
    last_firmware = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_mode_request:
            robot.set_motion_mode("cpv")
            next_mode_request = now + retry_interval_s
        snapshot = driver_enable_snapshot(robot)
        status = robot.get_arm_status()
        if snapshot is None or status is None:
            time.sleep(0.02)
            continue
        if snapshot[0] != ALL_DISABLED:
            raise RuntimeError(
                "A Nero joint enabled during read-service recovery: "
                f"status={snapshot[0]}"
            )
        last_mode = status.msg.mode_feedback
        remaining = max(0.01, deadline - time.monotonic())
        firmware = robot.get_firmware(
            timeout=min(remaining, 0.25),
            min_interval=0.0,
        )
        last_firmware = firmware
        if firmware is not None and "software_version" in firmware:
            version_text = str(firmware["software_version"])
            version = parse_firmware_version(version_text)
            if version < tuple(minimum_firmware):
                raise RuntimeError(
                    f"Nero firmware {version_text} is older than required "
                    f"{minimum_firmware[0]}.{minimum_firmware[1]:02d}"
                )
            if (
                status.msg.mode_feedback
                == robot.ARM_STATUS.ModeFeedback.MOVE_CPV
                and float(status.timestamp) > initial_status_stamp
            ):
                return version_text
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.02)
    raise TimeoutError(
        "Nero read service is unavailable after verified disable; "
        f"last mode={last_mode}, last firmware={last_firmware!r}. Disconnect, "
        "wait for the controller service to settle, then pass a new disabled "
        "firmware-service check before another active run. Power-cycle only "
        "if that fresh preflight still fails"
    )


def wait_for_exact_enable_status(
    robot,
    expected: list[bool],
    initial_stamps: list[float],
    timeout_s: float,
    command: Callable[[], object],
    *,
    retry_interval_s: float | None = None,
    keepalive: Callable[[], object] | None = None,
    keepalive_interval_s: float = 0.02,
) -> list[float]:
    """Send a motor command and require a fresh exact seven-bit result.

    When ``retry_interval_s`` is provided, resend the same idempotent
    per-joint request at a bounded rate until feedback proves the requested
    state.  ``keepalive`` optionally maintains an already-seeded CPV target
    while cumulative enable is in progress, so joints enabled earlier in the
    sequence never experience a position-command gap.
    """
    if len(expected) != 7 or len(initial_stamps) != 7:
        raise ValueError("Nero enable status and timestamps must contain seven values")
    if retry_interval_s is not None and retry_interval_s <= 0.0:
        raise ValueError("retry_interval_s must be positive")
    if keepalive_interval_s <= 0.0:
        raise ValueError("keepalive_interval_s must be positive")
    command()
    attempts = 1
    next_command_time = (
        None
        if retry_interval_s is None
        else time.monotonic() + retry_interval_s
    )
    deadline = time.monotonic() + timeout_s
    next_keepalive_time = (
        None if keepalive is None else time.monotonic()
    )
    last_status = None
    while time.monotonic() < deadline:
        snapshot = driver_enable_snapshot(robot)
        if snapshot is not None:
            statuses, stamps = snapshot
            last_status = statuses
            unexpected = [
                index + 1
                for index, (actual, wanted) in enumerate(
                    zip(statuses, expected, strict=True)
                )
                if actual and not wanted
            ]
            if unexpected:
                raise RuntimeError(
                    f"Unexpected Nero joints enabled: {unexpected}; status={statuses}"
                )
            if statuses == expected and all(
                new > old
                for new, old in zip(stamps, initial_stamps, strict=True)
            ):
                return stamps
        now = time.monotonic()
        if next_keepalive_time is not None and now >= next_keepalive_time:
            keepalive()
            next_keepalive_time = now + keepalive_interval_s
        if next_command_time is not None and now >= next_command_time:
            command()
            attempts += 1
            next_command_time = now + float(retry_interval_s)
        if robot.has_comm_error():
            raise RuntimeError(f"Nero CAN error: {robot.get_comm_error()}")
        time.sleep(0.01)
    raise TimeoutError(
        f"Expected Nero enable bits {expected}, last observed {last_status}, "
        f"command attempts={attempts}"
    )


def enable_all_individual_verified(
    robot,
    timeout_s: float,
    *,
    keepalive: Callable[[], object] | None = None,
    keepalive_interval_s: float = 0.02,
) -> None:
    """Enable J1..J7 individually, verifying each cumulative state.

    The V112 all-joint/broadcast 0x471 request is intermittent on the real
    external-CAN interface.  Per-joint indices 1..7 were commissioned on the
    hardware and provide an unambiguous intermediate state after every frame.
    Callers must seed all position targets while disabled before invoking this
    helper. Active callers should pass a keepalive callback; it is invoked
    immediately before every per-joint enable request as well as during its
    acknowledgment wait, allowing MIT callers to refresh still-disabled axes.
    """
    snapshot = driver_enable_snapshot(robot)
    if snapshot is None:
        raise RuntimeError("Cannot enable Nero without complete driver feedback")
    if snapshot[0] != ALL_DISABLED:
        raise RuntimeError(
            "Individual Nero enable requires all seven joints disabled; "
            f"status={snapshot[0]}"
        )

    stamps = snapshot[1]
    expected = list(ALL_DISABLED)
    for joint_index in range(1, 8):
        if keepalive is not None:
            keepalive()
        expected[joint_index - 1] = True

        def send_enable(index=joint_index):
            msg = robot._MSG_MotorEnableDisableConfig(
                joint_index=index,
                enable_flag=2,
            )
            robot._send_msg(msg)

        try:
            stamps = wait_for_exact_enable_status(
                robot,
                list(expected),
                stamps,
                timeout_s,
                send_enable,
                retry_interval_s=0.10,
                keepalive=keepalive,
                keepalive_interval_s=keepalive_interval_s,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Nero J{joint_index} enable failed: {exc}; "
                f"diagnostics: {format_status_diagnostics(robot)}"
            ) from exc


def request_individual_disable(robot, duration_s: float = 1.0) -> None:
    """Repeatedly place per-joint disable frames after any queued command."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        for joint_index in range(1, 8):
            robot.disable(joint_index=joint_index)
        time.sleep(0.05)


def wait_for_stable_all_disabled(
    robot,
    timeout_s: float = 2.0,
    stable_s: float = 0.5,
    *,
    initial_stamps: list[float] | None = None,
) -> bool:
    """Require continuously all-disabled feedback plus one fresh full cycle.

    Driver-state frames can arrive more slowly than the 50 Hz control/ polling
    loop.  Cached all-disabled samples therefore remain valid between CAN
    updates and must not reset the quiet-period timer.  Success still requires
    every one of the seven 0x261..0x267 timestamps to advance after a known
    baseline, so stale cached state alone can never pass.  Cleanup callers pass
    the snapshot taken *before* sending disable: V112 status frames can be
    event-driven and may finish updating during the disable-request window.
    """
    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    if initial_stamps is not None and len(initial_stamps) != 7:
        raise ValueError("initial_stamps must contain seven values")
    baseline_stamps = (
        None if initial_stamps is None else list(initial_stamps)
    )
    while time.monotonic() < deadline:
        snapshot = driver_enable_snapshot(robot)
        if snapshot is None:
            stable_since = None
            if initial_stamps is None:
                baseline_stamps = None
        else:
            statuses, stamps = snapshot
            if statuses == ALL_DISABLED:
                if stable_since is None:
                    stable_since = time.monotonic()
                    if baseline_stamps is None:
                        baseline_stamps = list(stamps)
                elif (
                    time.monotonic() - stable_since >= stable_s
                    and baseline_stamps is not None
                    and all(
                        new > old
                        for new, old in zip(
                            stamps, baseline_stamps, strict=True
                        )
                    )
                ):
                    return True
            else:
                stable_since = None
                if initial_stamps is None:
                    baseline_stamps = None
        if robot.has_comm_error():
            return False
        time.sleep(0.02)
    return False


def disable_all_individual_verified(
    robot,
    *,
    request_s: float = 1.0,
    timeout_s: float = 2.0,
    stable_s: float = 0.5,
) -> bool:
    """Drive every enabled joint to a verified disabled state.

    Real V112 hardware can stop accepting later per-joint requests after the
    first part of a sequential MIT-mode shutdown.  Every retry therefore sends
    one all-joint disable (index 8 in the Nero protocol), followed on the next
    slot by one still-enabled joint.  Feedback, rather than the SDK method's
    immediate cached return value, decides completion.  No e-stop, mode, or
    motion-target frame is sent here.
    """
    snapshot = driver_enable_snapshot(robot)
    if snapshot is None:
        return False
    overall_deadline = time.monotonic() + max(timeout_s, request_s + 0.10)
    initially_enabled = list(snapshot[0])
    initial_stamps = list(snapshot[1])
    next_request = 0.0
    send_broadcast = True
    round_robin = 0
    stable_since: float | None = None
    while time.monotonic() < overall_deadline:
        snapshot = driver_enable_snapshot(robot)
        if snapshot is None:
            stable_since = None
        else:
            statuses, stamps = snapshot
            if statuses == ALL_DISABLED:
                transitioned = all(
                    (not was_enabled) or new > old
                    for was_enabled, new, old in zip(
                        initially_enabled, stamps, initial_stamps, strict=True
                    )
                )
                if transitioned:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    elif time.monotonic() - stable_since >= stable_s:
                        return True
            else:
                stable_since = None

            now = time.monotonic()
            if now >= next_request and statuses != ALL_DISABLED:
                enabled_indices = [
                    index + 1 for index, enabled in enumerate(statuses) if enabled
                ]
                if send_broadcast:
                    # Nero's SDK maps joint_index=255 to protocol motor index 8.
                    robot.disable(joint_index=255)
                else:
                    joint_index = enabled_indices[round_robin % len(enabled_indices)]
                    robot.disable(joint_index=joint_index)
                    round_robin += 1
                send_broadcast = not send_broadcast
                next_request = now + 0.05
        if robot.has_comm_error():
            return False
        time.sleep(0.01)
    return False
