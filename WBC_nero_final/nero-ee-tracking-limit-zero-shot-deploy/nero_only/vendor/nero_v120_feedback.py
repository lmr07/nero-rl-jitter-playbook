"""Independent passive SocketCAN feedback for Nero firmware V1.20.

The vendor SDK exposes the 0x251..0x257 stream through mutable parser-cache
objects.  That is convenient for ordinary applications, but it cannot provide
an atomic seven-joint cohort and a parser-thread failure can leave apparently
valid cached objects behind.  This receiver owns a second read-only SocketCAN
socket, maps the kernel timestamp of each physical frame into the monotonic
clock, and publishes one locked snapshot to the real-time environment.
"""

from __future__ import annotations

import math
import threading
import time

import can


class NeroV120Feedback:
    """Passively receive and atomically snapshot Nero V1.20 motor feedback."""

    FIRST_ID = 0x251
    LAST_ID = 0x257

    def __init__(self, can_interface: str, on_frame=None):
        self.can_interface = str(can_interface)
        self.on_frame = on_frame
        self._bus = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._positions = [math.nan] * 7
        self._velocities = [math.nan] * 7
        self._stamps = [math.nan] * 7
        self._counts = [0] * 7
        self._feedback_max_gaps = [0.0] * 7
        self._mit_tx_stamps = [math.nan] * 7
        self._mit_tx_counts = [0] * 7
        self._mit_tx_max_gaps = [0.0] * 7
        self._driver_stamps = [math.nan] * 7
        self._driver_max_gaps = [0.0] * 7
        self._driver_payloads = [None] * 7
        self._arm_status_stamp = math.nan
        self._arm_status_max_gap = 0.0
        self._arm_status_payload = None
        self._max_reader_queue_delay = 0.0
        self._error: BaseException | None = None

    @staticmethod
    def decode_payload(data) -> tuple[float, float]:
        """Decode documented V1.20 velocity and position fields."""
        payload = bytes(data)
        if len(payload) != 8:
            raise ValueError(f"Nero high-speed payload must be 8 bytes, got {len(payload)}")
        velocity = int.from_bytes(payload[0:2], "big", signed=True) * 1.0e-3
        position = int.from_bytes(payload[4:8], "big", signed=True) * 1.0e-3
        return float(position), float(velocity)

    @staticmethod
    def kernel_stamp_to_monotonic(
        kernel_wall_stamp: float,
        *,
        dequeue_monotonic: float,
        dequeue_wall: float,
    ) -> float:
        """Map a SocketCAN kernel wall timestamp onto the monotonic clock."""
        stamp = float(kernel_wall_stamp)
        if not math.isfinite(stamp) or stamp <= 0.0:
            return float(dequeue_monotonic)
        mapped = stamp + (float(dequeue_monotonic) - float(dequeue_wall))
        if mapped > float(dequeue_monotonic) + 0.010:
            return float(dequeue_monotonic)
        return mapped

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._bus = can.Bus(
            interface="socketcan",
            channel=self.can_interface,
        )
        self._thread = threading.Thread(
            target=self._read_loop,
            name="nero-v120-passive-feedback",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                msg = self._bus.recv(timeout=0.05)
                if msg is None or msg.is_extended_id or msg.is_error_frame:
                    continue
                can_id = int(msg.arbitration_id)
                dequeue_monotonic = time.monotonic()
                kernel_wall_stamp = float(getattr(msg, "timestamp", 0.0))
                # python-can SocketCAN timestamps are CLOCK_REALTIME. Mapping
                # the kernel event time prevents a delayed reader from making
                # an old queued frame appear freshly transmitted.
                stamp = self.kernel_stamp_to_monotonic(
                    kernel_wall_stamp,
                    dequeue_monotonic=dequeue_monotonic,
                    dequeue_wall=time.time(),
                )
                if self.on_frame is not None:
                    self.on_frame(msg, stamp, dequeue_monotonic)
                queue_delay = max(0.0, dequeue_monotonic - stamp)
                with self._lock:
                    self._max_reader_queue_delay = max(
                        self._max_reader_queue_delay, queue_delay
                    )
                if self.FIRST_ID <= can_id <= self.LAST_ID:
                    position, velocity = self.decode_payload(msg.data)
                    index = can_id - self.FIRST_ID
                    with self._lock:
                        previous = self._stamps[index]
                        if math.isfinite(previous):
                            self._feedback_max_gaps[index] = max(
                                self._feedback_max_gaps[index], stamp - previous
                            )
                        self._positions[index] = position
                        self._velocities[index] = velocity
                        self._stamps[index] = stamp
                        self._counts[index] += 1
                elif 0x15A <= can_id <= 0x160:
                    index = can_id - 0x15A
                    with self._lock:
                        previous = self._mit_tx_stamps[index]
                        if math.isfinite(previous):
                            self._mit_tx_max_gaps[index] = max(
                                self._mit_tx_max_gaps[index], stamp - previous
                            )
                        self._mit_tx_stamps[index] = stamp
                        self._mit_tx_counts[index] += 1
                elif 0x261 <= can_id <= 0x267:
                    index = can_id - 0x261
                    with self._lock:
                        previous = self._driver_stamps[index]
                        if math.isfinite(previous):
                            self._driver_max_gaps[index] = max(
                                self._driver_max_gaps[index], stamp - previous
                            )
                        self._driver_stamps[index] = stamp
                        self._driver_payloads[index] = bytes(msg.data).hex().upper()
                elif can_id == 0x2A1:
                    with self._lock:
                        if math.isfinite(self._arm_status_stamp):
                            self._arm_status_max_gap = max(
                                self._arm_status_max_gap,
                                stamp - self._arm_status_stamp,
                            )
                        self._arm_status_stamp = stamp
                        self._arm_status_payload = bytes(msg.data).hex().upper()
        except BaseException as exc:
            if not self._stop.is_set():
                self._error = exc

    def snapshot(
        self,
    ) -> tuple[list[float], list[float], list[float], list[int]] | None:
        """Return one atomic complete J1..J7 snapshot, or ``None`` initially."""
        with self._lock:
            positions = list(self._positions)
            velocities = list(self._velocities)
            stamps = list(self._stamps)
            counts = list(self._counts)
        if min(counts) < 1 or not all(
            math.isfinite(value) for value in positions + velocities + stamps
        ):
            return None
        return positions, velocities, stamps, counts

    def wait_for_complete(self, timeout_s: float) -> None:
        """Wait until every physical 0x251..0x257 ID has been received."""
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.snapshot() is not None:
                return
            if self._error is not None:
                raise RuntimeError(
                    f"Nero passive SocketCAN receiver failed: {self._error}"
                ) from self._error
            time.sleep(0.005)
        raise TimeoutError(
            "No complete Nero V1.20 physical 0x251..0x257 feedback cohort; "
            + self.health_summary()
        )

    def reset_mit_tx_diagnostics(self) -> None:
        with self._lock:
            self._mit_tx_stamps = [math.nan] * 7
            self._mit_tx_counts = [0] * 7
            self._mit_tx_max_gaps = [0.0] * 7

    def reset_timing_diagnostics(self) -> None:
        with self._lock:
            self._feedback_max_gaps = [0.0] * 7
            self._mit_tx_max_gaps = [0.0] * 7
            self._driver_max_gaps = [0.0] * 7
            self._arm_status_max_gap = 0.0
            self._max_reader_queue_delay = 0.0

    def timing_snapshot(self) -> dict[str, list[float] | float | list[int]]:
        with self._lock:
            return {
                "feedback_counts": list(self._counts),
                "feedback_max_gaps_s": list(self._feedback_max_gaps),
                "mit_tx_counts": list(self._mit_tx_counts),
                "mit_tx_max_gaps_s": list(self._mit_tx_max_gaps),
                "driver_max_gaps_s": list(self._driver_max_gaps),
                "arm_status_max_gap_s": float(self._arm_status_max_gap),
                "max_reader_queue_delay_s": float(self._max_reader_queue_delay),
            }

    def diagnostic_snapshot(self) -> dict[str, object]:
        """Return one locked, structured transport snapshot.

        Unlike :meth:`snapshot`, this accessor deliberately returns incomplete
        state during a fault.  A passive diagnostic recorder needs the last
        timestamp and count for every RX/TX family after the controller has
        stopped publishing; requiring a fresh complete cohort would erase the
        evidence that distinguishes a node-side stall from a host TX stall.
        The returned lists are copies and may be inspected without holding the
        receiver lock.
        """
        with self._lock:
            return {
                "feedback_positions_rad": list(self._positions),
                "feedback_velocities_rad_s": list(self._velocities),
                "feedback_stamps_s": list(self._stamps),
                "feedback_counts": list(self._counts),
                "feedback_max_gaps_s": list(self._feedback_max_gaps),
                "mit_tx_stamps_s": list(self._mit_tx_stamps),
                "mit_tx_counts": list(self._mit_tx_counts),
                "mit_tx_max_gaps_s": list(self._mit_tx_max_gaps),
                "driver_stamps_s": list(self._driver_stamps),
                "driver_max_gaps_s": list(self._driver_max_gaps),
                "driver_payloads": [
                    None if payload is None else str(payload)
                    for payload in self._driver_payloads
                ],
                "arm_status_stamp_s": float(self._arm_status_stamp),
                "arm_status_max_gap_s": float(self._arm_status_max_gap),
                "arm_status_payload": self._arm_status_payload,
                "max_reader_queue_delay_s": float(self._max_reader_queue_delay),
                "thread_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "error": None if self._error is None else repr(self._error),
            }

    @staticmethod
    def decode_driver_status_payload(payload) -> dict[str, object]:
        """Decode one documented 0x261..0x267 low-speed status frame."""
        data = bytes.fromhex(payload) if isinstance(payload, str) else bytes(payload)
        if len(data) != 8:
            raise ValueError(
                f"Nero driver-status payload must be 8 bytes, got {len(data)}"
            )
        status = int(data[5])
        return {
            "voltage_v": 0.1 * int.from_bytes(data[0:2], "big", signed=False),
            "driver_temp_c": int.from_bytes(data[2:4], "big", signed=True),
            "motor_temp_c": int.from_bytes(data[4:5], "big", signed=True),
            "status_code": status,
            "voltage_too_low": bool(status & (1 << 0)),
            "motor_overheating": bool(status & (1 << 1)),
            "driver_overcurrent": bool(status & (1 << 2)),
            "driver_overheating": bool(status & (1 << 3)),
            "collision_status": bool(status & (1 << 4)),
            "driver_error_status": bool(status & (1 << 5)),
            "driver_enable_status": bool(status & (1 << 6)),
            "stall_status": bool(status & (1 << 7)),
            "bus_current_a": 0.001
            * int.from_bytes(data[6:8], "big", signed=False),
        }

    @staticmethod
    def decode_arm_status_payload(payload) -> dict[str, int]:
        """Decode one documented 0x2A1 controller-status frame."""
        data = bytes.fromhex(payload) if isinstance(payload, str) else bytes(payload)
        if len(data) != 8:
            raise ValueError(
                f"Nero arm-status payload must be 8 bytes, got {len(data)}"
            )
        return {
            "ctrl_mode": int(data[0]),
            "arm_status": int(data[1]),
            "mode_feedback": int(data[2]),
            "teach_status": int(data[3]),
            "motion_status": int(data[4]),
            "trajectory_num": int(data[5]),
            "err_code": int.from_bytes(data[6:8], "big", signed=False),
        }

    def driver_voltage_snapshot(
        self,
    ) -> tuple[list[float], list[float]] | None:
        """Return atomic J1..J7 driver voltages and physical frame stamps.

        Driver-status IDs 0x261..0x267 encode supply voltage as an unsigned
        big-endian uint16 in bytes 0..1 at 0.1 V/count. Return ``None`` until
        all seven physical status frames have been observed so callers cannot
        mistake a partial cohort for a healthy power rail.
        """
        with self._lock:
            payloads = list(self._driver_payloads)
            stamps = list(self._driver_stamps)
        if any(payload is None for payload in payloads) or not all(
            math.isfinite(stamp) for stamp in stamps
        ):
            return None
        decoded = [self.decode_driver_status_payload(payload) for payload in payloads]
        voltages = [float(status["voltage_v"]) for status in decoded]
        return voltages, stamps

    def driver_status_payload_snapshot(self) -> tuple[list[str], list[float]]:
        """Return the latest raw J1..J7 status payloads and frame stamps.

        This accessor is diagnostic-only.  Keeping the raw bytes beside an
        undervoltage decision makes it possible to distinguish a decoded
        physical status frame (for example byte 1 == 0x02, or 0.2 V) from a
        stale SDK cache or an accidental field interpretation.
        """
        with self._lock:
            payloads = [
                "None" if payload is None else str(payload)
                for payload in self._driver_payloads
            ]
            stamps = list(self._driver_stamps)
        return payloads, stamps

    def wait_for_mit_tx(self, timeout_s: float, *, minimum_count: int = 2) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                counts = list(self._mit_tx_counts)
            if min(counts) >= int(minimum_count):
                return
            if self._error is not None:
                raise RuntimeError(
                    f"Nero passive SocketCAN receiver failed: {self._error}"
                ) from self._error
            time.sleep(0.002)
        raise TimeoutError(
            "Kernel-periodic Nero MIT frames were not observed on all seven IDs; "
            + self.health_summary()
        )

    def mit_tx_is_recent(self, max_age_s: float) -> bool:
        with self._lock:
            counts = list(self._mit_tx_counts)
            stamps = list(self._mit_tx_stamps)
        now = time.monotonic()
        return min(counts) >= 2 and all(
            math.isfinite(stamp) and now - stamp <= float(max_age_s)
            for stamp in stamps
        )

    def feedback_is_recent(self, max_age_s: float) -> bool:
        snapshot = self.snapshot()
        if snapshot is None:
            return False
        now = time.monotonic()
        return all(now - stamp <= float(max_age_s) for stamp in snapshot[2])

    def wait_for_runtime_stream(self, timeout_s: float, *, max_age_s: float) -> None:
        """Wait for a new, recent RX cohort and all kernel-periodic MIT IDs."""
        with self._lock:
            initial_feedback = list(self._counts)
            initial_tx = list(self._mit_tx_counts)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                feedback_counts = list(self._counts)
                tx_counts = list(self._mit_tx_counts)
            feedback_advanced = all(
                new > old
                for new, old in zip(
                    feedback_counts, initial_feedback, strict=True
                )
            )
            tx_advanced = all(
                new > old
                for new, old in zip(tx_counts, initial_tx, strict=True)
            )
            if (
                feedback_advanced
                and tx_advanced
                and self.feedback_is_recent(max_age_s)
                and self.mit_tx_is_recent(max_age_s)
            ):
                return
            if self._error is not None:
                raise RuntimeError(
                    f"Nero passive SocketCAN receiver failed: {self._error}"
                ) from self._error
            time.sleep(0.002)
        raise TimeoutError(
            "Nero V1.20 RX/TX stream did not recover after the blocking A2 "
            "startup; " + self.health_summary()
        )

    def health_summary(self) -> str:
        snapshot = self.snapshot()
        with self._lock:
            counts = list(self._counts)
            stamps = list(self._stamps)
            feedback_max_gaps = list(self._feedback_max_gaps)
            mit_counts = list(self._mit_tx_counts)
            mit_stamps = list(self._mit_tx_stamps)
            mit_max_gaps = list(self._mit_tx_max_gaps)
            driver_stamps = list(self._driver_stamps)
            driver_max_gaps = list(self._driver_max_gaps)
            driver_payloads = list(self._driver_payloads)
            arm_status_stamp = self._arm_status_stamp
            arm_status_max_gap = self._arm_status_max_gap
            arm_status_payload = self._arm_status_payload
            max_reader_queue_delay = self._max_reader_queue_delay
        now = time.monotonic()
        feedback_ages_ms = [
            None if not math.isfinite(stamp) else 1000.0 * max(0.0, now - stamp)
            for stamp in stamps
        ]
        thread_alive = bool(self._thread is not None and self._thread.is_alive())
        def age_vector_ms(values):
            return [
                None
                if not math.isfinite(stamp)
                else round(1000.0 * max(0.0, now - stamp), 1)
                for stamp in values
            ]
        return (
            f"passive_socket_alive={thread_alive}, complete={snapshot is not None}, "
            f"counts={counts}, ages_ms="
            f"{[None if age is None else round(age, 1) for age in feedback_ages_ms]}, "
            f"feedback_max_gaps_ms="
            f"{[round(1000.0 * value, 1) for value in feedback_max_gaps]}, "
            f"physical_mit_tx_counts={mit_counts}, "
            f"physical_mit_tx_ages_ms={age_vector_ms(mit_stamps)}, "
            f"physical_mit_tx_max_gaps_ms="
            f"{[round(1000.0 * value, 1) for value in mit_max_gaps]}, "
            f"driver_ages_ms={age_vector_ms(driver_stamps)}, "
            f"driver_max_gaps_ms="
            f"{[round(1000.0 * value, 1) for value in driver_max_gaps]}, "
            f"driver_payloads={driver_payloads}, arm_status_age_ms="
            f"{age_vector_ms([arm_status_stamp])[0]}, arm_status_payload="
            f"{arm_status_payload}, arm_status_max_gap_ms="
            f"{1000.0 * arm_status_max_gap:.1f}, "
            f"max_reader_queue_delay_ms="
            f"{1000.0 * max_reader_queue_delay:.1f}, "
            f"error={self._error!r}"
        )

    @property
    def error(self) -> BaseException | None:
        return self._error

    def close(self) -> None:
        self._stop.set()
        bus = self._bus
        if bus is not None:
            try:
                bus.shutdown()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._bus = None
