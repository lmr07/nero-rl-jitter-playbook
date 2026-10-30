"""Nero-only V1.21 pure-MIT backend; owns no A2 or IMU connection."""
import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
from .feedback_clock import DriverClock
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from .vendor.nero_mit_kernel_writer import NeroMitKernelWriter, build_nero_mit_frames
from .vendor.nero_v120_feedback import NeroV120Feedback
from .vendor.nero_v112_safety import (
    driver_enable_snapshot, enter_can_control_disabled, require_firmware_at_least,
    enable_all_individual_verified, disable_all_individual_verified,
    refresh_disabled_position_targets, evaluate_nero_handoff_motion, feedback_timestamp_age_s,
)


def can_health(interface):
    result = subprocess.run(['ip','-j','-details','link','show','dev',interface],
                            text=True,capture_output=True,timeout=2)
    if result.returncode:
        available = sorted(p.name for p in Path('/sys/class/net').iterdir() if (p/'type').read_text().strip() == '280')
        raise RuntimeError(f'CAN interface {interface!r} unavailable. Current CAN interfaces: {available}. '
            'For the commissioned dual-PEAK wiring, restore the Nero interface with '
            'sudo python3 scripts/prepare_nero_can.py --apply. Do not guess can0/can1. '
            + result.stderr.strip())
    data = json.loads(result.stdout)[0]
    info = data['linkinfo']['info_data']
    if info['state'] != 'ERROR-ACTIVE' or info['bittiming']['bitrate'] != 1000000:
        raise RuntimeError('Nero requires ERROR-ACTIVE at 1 Mbps: '+str(info))
    return info


def predictive_limit_risk(q, dq, lower, upper, horizon_s, margin_rad):
    """Indices whose measured motion approaches a hard joint limit."""
    q, dq = np.asarray(q), np.asarray(dq)
    predicted = q + dq*horizon_s
    return np.flatnonzero(((dq < -.15) & (predicted < np.asarray(lower)+margin_rad)) |
                          ((dq > .15) & (predicted > np.asarray(upper)-margin_rad)))


class NeroHardware:
    def __init__(self, interface, policy, trace=None):
        self.interface,self.policy = interface,policy
        self.trace = trace
        self.command_seq = 0
        self.driver_clock = DriverClock()
        self.last_feedback = None
        self.robot = self.receiver = self.writer = None
        self.lock = None
        self.may_enabled = False
        self.active = False
        self.policy_running = False
        self.last_counts = None
        self.last_sent = None
        self.last_tick = time.monotonic()
        self.watch_stop = threading.Event()
        self.fault = threading.Event()
        self.thread = None

    def connect(self):
        can_health(self.interface)
        # This lock prevents concurrent instances of this independent deployment.
        path = Path('/tmp')/('nero-only-'+self.interface+'.lock')
        fd = os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        self.lock = os.fdopen(fd,'w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.receiver = NeroV120Feedback(self.interface, on_frame=self.trace.can_frame if self.trace else None)
        self.receiver.start()
        self.receiver.wait_for_complete(5)
        time.sleep(.15)
        # Existing A2nero does not share our lock. Refuse its live MIT traffic.
        if any(self.receiver.timing_snapshot()['mit_tx_counts']):
            raise RuntimeError('Another Nero MIT sender is active; stop that deployment first')
        cfg = create_agx_arm_config(robot=ArmModel.NERO,firmeware_version=NeroFW.V121,
            interface='socketcan',channel=self.interface,bitrate=1000000,enable_check_can=True)
        self.robot = AgxArmFactory.create_arm(cfg)
        self.robot.set_joint_limits_enabled(True)
        self.robot.connect()
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            bits = driver_enable_snapshot(self.robot)
            if bits is not None:
                break
            time.sleep(.01)
        if bits is None or bits[0] != [False]*7:
            raise RuntimeError('Nero must have all seven joints disabled before takeover')
        version = require_firmware_at_least(self.robot,(1,21),5)
        q,dq = self.read()
        print('Nero firmware:',version,'identity J1..J7 feedback; q=',q)
        return q,dq

    def read(self,timeout=.18,require_advance=True):
        read_enter = time.monotonic()
        deadline = read_enter+timeout
        while time.monotonic()<deadline:
            if self.fault.is_set():
                raise RuntimeError('Nero watchdog stopped the arm')
            if self.receiver.error:
                raise RuntimeError('Nero receiver failed') from self.receiver.error
            snapshot = self.receiver.snapshot()
            if snapshot:
                q,dq,stamps,counts = snapshot
                advanced = self.last_counts is None or all(a>b for a,b in zip(counts,self.last_counts))
                if time.monotonic()-min(stamps)<=.03 and max(stamps)-min(stamps)<=.03 and (advanced or not require_advance):
                    self.last_counts = counts
                    self.last_feedback = {'read_enter_s':read_enter,'read_ready_s':time.monotonic(),
                        'joint_rx_stamps_s':list(stamps),'joint_rx_counts':list(counts)}
                    if self.trace:
                        self.trace.check()
                        self.trace.emit('feedback_used', **self.last_feedback, q=list(q), dq=list(dq))
                    q,dq = np.array(q),np.array(dq)
                    if not np.isfinite(np.r_[q,dq]).all():
                        raise RuntimeError('Non-finite Nero feedback')
                    if (q < self.policy.lower-.02).any() or (q > self.policy.upper+.02).any():
                        bad = np.flatnonzero((q < self.policy.lower-.02) | (q > self.policy.upper+.02))
                        raise RuntimeError('Measured joint beyond URDF limits: '
                            + ', '.join(f'J{j+1}={q[j]:.3f} rad, range [{self.policy.lower[j]:.3f}, '
                                        f'{self.policy.upper[j]:.3f}]' for j in bad))
                    if np.max(np.abs(dq))>5:
                        raise RuntimeError('Measured Nero speed exceeds 5 rad/s')
                    if self.policy_running:
                        risk = predictive_limit_risk(q,dq,self.policy.lower,self.policy.upper,
                            float(self.policy.guards['limit_predict_horizon_s']),
                            float(self.policy.guards['limit_predict_margin_rad']))
                        monitored = np.asarray(self.policy.guards['limit_predict_joints'],dtype=int)-1
                        risk = np.intersect1d(risk,monitored)
                        if len(risk):
                            raise RuntimeError('Measured motion predicts joint-limit approach: '
                                + ', '.join(f'J{j+1} q={q[j]:.3f} dq={dq[j]:.3f}' for j in risk))
                    if self.active:
                        enabled = driver_enable_snapshot(self.robot)
                        status = self.robot.get_arm_status()
                        if enabled is None or enabled[0] != [True]*7 or status is None:
                            raise RuntimeError('Nero enable/status feedback missing')
                        ages=self.driver_clock.check(enabled,self.receiver.diagnostic_snapshot(),time.monotonic())
                        if self.trace:self.trace.emit('sdk_driver_progress',sdk_stamps=enabled[1],monotonic_progress_age_s=ages,wall_s=time.time())
                        if status.msg.mode_feedback != self.robot.ARM_STATUS.ModeFeedback.MOVE_MIT:
                            raise RuntimeError('Nero left MOVE_MIT')
                    return q,dq
            time.sleep(.001)
        raise TimeoutError('No fresh, fully advancing Nero feedback cohort within 180 ms')

    def send(self,target):
        if self.fault.is_set():
            raise RuntimeError('Cannot send after a watchdog stop')
        target = np.asarray(target,dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Invalid MIT target')
        if (target < self.policy.lower).any() or (target > self.policy.upper).any():
            raise ValueError('MIT target outside trained URDF limits')
        frames = build_nero_mit_frames(self.robot,target,kp=self.policy.kp,kd=self.policy.kd)
        self.command_seq += 1
        seq = self.command_seq
        if self.trace:
            self.trace.emit('command_submit', seq=seq, q_target=target.tolist(),
                frames=[{'can_id':int(f.arbitration_id),'data':bytes(f.data).hex()} for f in frames])
        if self.writer is not None and self.writer.running:
            self.writer.update(frames)
        else:
            for j,q in enumerate(target):
                self.robot.move_mit(j+1,p_des=float(q),v_des=0,kp=float(self.policy.kp[j]),kd=float(self.policy.kd[j]),t_ff=0)
        if self.trace:
            self.trace.emit('command_update_done', seq=seq)
        self.last_sent = target.copy()
        self.last_tick = time.monotonic()

    def start(self):
        status = self.robot.get_arm_status()
        allowed = {self.robot.ARM_STATUS.ArmStatus.NORMAL,self.robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED}
        if status is None or status.msg.arm_status not in allowed:
            raise RuntimeError('Nero not fault-free; reset the controller before starting')
        enter_can_control_disabled(self.robot,5)
        q,_ = self.read()
        self.robot.set_auto_set_motion_mode_enabled(False)
        self.send(q)
        self.robot.set_motion_mode('mit')
        stable,deadline = None,time.monotonic()+5
        while time.monotonic()<deadline:
            q,_ = self.read()
            self.send(q)
            bits = driver_enable_snapshot(self.robot)
            status = self.robot.get_arm_status()
            if bits is None or bits[0] != [False]*7:
                raise RuntimeError('Unexpected enable during disabled MIT setup')
            if status and status.msg.mode_feedback == self.robot.ARM_STATUS.ModeFeedback.MOVE_MIT:
                stable = time.monotonic() if stable is None else stable
                if time.monotonic()-stable >= .2:
                    break
            else:
                stable = None
            time.sleep(.02)
        else:
            raise TimeoutError('Disabled Nero never entered stable MOVE_MIT')
        self.writer = NeroMitKernelWriter(self.interface,period_s=.02)
        self.writer.start(build_nero_mit_frames(self.robot,q,kp=self.policy.kp,kd=self.policy.kd))
        self.receiver.wait_for_mit_tx(.3)
        anchor = q.copy()
        initial = anchor.copy()
        previous = q.copy()
        streak = np.zeros(7,dtype=int)
        def keepalive():
            nonlocal previous,streak
            measured,velocity = self.read(require_advance=False)
            refresh_disabled_position_targets(self.robot,anchor,measured)
            if np.max(np.abs(anchor-initial))>.10:
                raise RuntimeError('Brake-release anchor drift exceeds 0.10 rad')
            bits = driver_enable_snapshot(self.robot)
            if bits is None:
                raise RuntimeError('Missing enable feedback during brake release')
            mask = np.array(bits[0])
            result = evaluate_nero_handoff_motion(measured,np.where(mask,velocity,0),
                np.where(mask,anchor,measured),np.where(mask,previous,measured),streak)
            previous,streak = measured.copy(),np.array(result['high_speed_streak'])
            if result['violation']:
                raise RuntimeError('Brake-release motion guard: '+str(result['violation']))
            self.send(anchor)
        keepalive()
        self.may_enabled = True
        enable_all_individual_verified(self.robot,5,keepalive=keepalive,keepalive_interval_s=.02)
        self.active = True
        self.last_tick = time.monotonic()
        self.thread = threading.Thread(target=self._watchdog,daemon=True)
        self.thread.start()
        # Preserve a measured-anchor settle interval before any preposition.
        for _ in range(50):
            keepalive()
            time.sleep(.02)

    def _watchdog(self):
        while not self.watch_stop.wait(.01):
            if time.monotonic()-self.last_tick>.18 or not self.receiver.feedback_is_recent(.18):
                if self.trace:self.trace.emit('watchdog_stop',command_age_s=time.monotonic()-self.last_tick,
                    feedback=self.receiver.diagnostic_snapshot())
                self.fault.set()
                self.writer.stop()
                try:
                    self.robot.electronic_emergency_stop()
                except Exception:
                    pass
                return

    def guarded_target(self,candidate,q,max_step=.02,envelope=.20):
        """Satisfy joint limits, slew and feedback envelope at the same time."""
        lower = np.maximum.reduce((self.policy.target_lower, self.last_sent-max_step, q-envelope))
        upper = np.minimum.reduce((self.policy.target_upper, self.last_sent+max_step, q+envelope))
        if (lower > upper).any():
            bad = np.flatnonzero(lower > upper)
            raise RuntimeError('No safe target satisfies both slew and feedback guards: '
                               + ', '.join(f'J{j+1}' for j in bad))
        return np.clip(candidate, lower, upper)

    def preposition(self,duration=8):
        start,_ = self.read()
        begun = time.monotonic()
        settled_since = None
        while True:
            tick = time.monotonic()
            q,dq = self.read()
            t = min((tick-begun)/duration,1)
            blend = t*t*t*(10+t*(-15+6*t))
            candidate = start+blend*(self.policy.default-start)
            self.send(self.guarded_target(candidate,q,max_step=.01,envelope=.20))
            settled = t == 1 and np.max(np.abs(q-self.policy.default)) < .10 and np.max(np.abs(dq)) < .05
            settled_since = (settled_since or tick) if settled else None
            if settled_since is not None and tick-settled_since >= .5:
                return
            if tick-begun>duration+5:
                raise RuntimeError('Nero did not converge to the policy initial pose')
            time.sleep(max(0,.02-(time.monotonic()-tick)))

    def close(self,emergency=False):
        self.watch_stop.set()
        if self.thread:
            self.thread.join(timeout=.3)
        cleanup_ok = True
        disabled_verified = False
        if self.trace:self.trace.emit('cleanup_begin',emergency=emergency,may_enabled=self.may_enabled)
        try:
            if self.robot and self.may_enabled:
                if emergency or self.fault.is_set():
                    if self.writer: self.writer.stop()
                    self.robot.electronic_emergency_stop()
                else:
                    q,_ = self.read(require_advance=False)
                    self.send(q)
                cleanup_ok = disable_all_individual_verified(self.robot,request_s=1,timeout_s=15,stable_s=.5)
                if not cleanup_ok:
                    raise RuntimeError('Nero disable was not verified')
                self.may_enabled = False
                disabled_verified = True
        except BaseException as exc:
            cleanup_ok = False
            print('Nero cleanup issue:',repr(exc))
            if self.trace:self.trace.emit('cleanup_exception',error=repr(exc))
            if self.robot and self.may_enabled:
                if self.writer: self.writer.stop()
                try:
                    self.robot.electronic_emergency_stop()
                    cleanup_ok = disable_all_individual_verified(self.robot,request_s=1,timeout_s=15,stable_s=.5)
                    if cleanup_ok:
                        self.may_enabled=False
                        disabled_verified=True
                except Exception as failure:
                    print('Emergency cleanup failed:',failure)
        finally:
            if self.writer: self.writer.stop()
            if self.receiver: self.receiver.close()
            if self.robot: self.robot.disconnect()
            if self.lock: self.lock.close()
        if self.trace:self.trace.emit('cleanup_end',disabled_verified=disabled_verified,cleanup_ok=cleanup_ok)
        if not cleanup_ok:
            raise RuntimeError('Could not verify that Nero is disabled; inspect hardware before restarting')
