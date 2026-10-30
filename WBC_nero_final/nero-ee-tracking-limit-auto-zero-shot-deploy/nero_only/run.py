"""Limit-aware automatic EE tracking; offline by default; --act moves Nero."""
import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty
import numpy as np
import torch
from .core import FixedPolicy, ROOT, sphere_to_cart

MAX_ACTOR_SECONDS = 60.0


def check_post_gain_home(policy, q, dq, target):
    """Validate the measured takeover state after the identified-gain switch."""
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    target = np.asarray(target, dtype=float)
    if any(v.shape != (7,) for v in (q, dq, target)) or not np.isfinite(np.r_[q, dq, target]).all():
        raise RuntimeError('Invalid post-gain joint feedback or MIT target; actor NOT started')
    home_error = np.abs(q - policy.default)
    speed = np.abs(dq)
    tracking_error = np.abs(target - q)
    allowed_tracking = np.minimum(policy.feedback_envelope, .20)
    outside_target = (target < policy.target_lower) | (target > policy.target_upper)
    if ((home_error > .10).any() or (speed > .08).any()
        or (tracking_error > allowed_tracking).any() or outside_target.any()):
        raise RuntimeError(
            'Post-gain home check failed; actor NOT started: '
            f'max home error={home_error.max():.4f} rad (J{home_error.argmax()+1}, limit .10); '
            f'max speed={speed.max():.4f} rad/s (J{speed.argmax()+1}, limit .08); '
            f'max target-feedback gap={tracking_error.max():.4f} rad '
            f'(J{tracking_error.argmax()+1}, per-joint limit {allowed_tracking[tracking_error.argmax()]:.4f}); '
            f'target margin violations={np.flatnonzero(outside_target).tolist()}')
    return {'max_home_error_rad': float(home_error.max()),
            'max_speed_rad_s': float(speed.max()),
            'max_target_feedback_gap_rad': float(tracking_error.max())}


class Keyboard:
    def __enter__(self):
        self.old = None
        self.buffer = b''
        self.escape_at = None
        if sys.stdin.isatty():
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self
    def __exit__(self,*args):
        if self.old is not None:
            termios.tcsetattr(sys.stdin,termios.TCSADRAIN,self.old)
    def poll(self):
        if self.old is None:
            return []
        keys=[]
        while select.select([sys.stdin],[],[],0)[0]:
            data=os.read(sys.stdin.fileno(),128)
            if not data: break
            self.buffer+=data
        arrows={b'\x1b[A':'up',b'\x1b[B':'down',b'\x1b[C':'right',b'\x1b[D':'left'}
        while self.buffer:
            if self.buffer.startswith(b'\x1b'):
                if any(sequence.startswith(self.buffer) for sequence in arrows) and len(self.buffer)<3:
                    self.escape_at = time.monotonic() if self.escape_at is None else self.escape_at
                    if time.monotonic()-self.escape_at<.08: break
                    keys.append('exit'); self.buffer=self.buffer[1:]; self.escape_at=None
                elif self.buffer[:3] in arrows:
                    keys.append(arrows[self.buffer[:3]]); self.buffer=self.buffer[3:]; self.escape_at=None
                else:
                    keys.append('exit'); self.buffer=self.buffer[1:]
            else:
                key=chr(self.buffer[0]).lower(); self.buffer=self.buffer[1:]
                keys.append('exit' if key=='q' else key)
        return keys


def offline(policy, seconds=20.0, auto_goals=False):
    reference=torch.load(policy.assets/'export_reference.pt',map_location='cpu',weights_only=True)
    with torch.inference_mode():
        error=(policy.actor(reference['obs'])-reference['action']).abs().max().item()
    assert error<1.e-5,error
    if auto_goals:
        policy.enable_auto_goals()
    q=policy.default.copy()
    for i in range(max(1000, round(seconds/policy.dt))):
        candidate=policy.infer(q,np.zeros(7))
        desired,_=policy.takeover_candidate(candidate,i*policy.dt)
        lower=np.maximum.reduce((policy.target_lower,policy.last_sent-policy.max_target_step,
                                 q-policy.feedback_envelope))
        upper=np.minimum.reduce((policy.target_upper,policy.last_sent+policy.max_target_step,
                                 q+policy.feedback_envelope))
        assert np.all(lower<=upper)
        target=np.clip(desired,lower,upper)
        assert np.isfinite(target).all()
        assert (target>=policy.target_lower).all() and (target<=policy.target_upper).all()
        policy.record_sent(target)
        policy.advance()
    sampled_goals = policy.segment
    policy.reset()
    gain=json.loads((policy.assets/'gain_profile.json').read_text())
    print(json.dumps({'mode':'offline; no CAN opened','goal_mode':'automatic',
          'offline_sampled_goals':sampled_goals,'run':policy.manifest['source_run'],
          'checkpoint':policy.manifest['checkpoint'],'shape':[624,7],
          'export_max_abs_error':error,'training_kp':policy.training_kp.tolist(),
          'training_kd':policy.training_kd.tolist(), 'hardware_kp':gain['kp'], 'hardware_kd':gain['kd'],
          'alpha':gain['alpha'],'beta':gain['beta'],'validation_r2':gain['validation_r2'],
          'python':sys.executable, 'action_mode':policy.action_mode,
          'target_step_limit_rad':policy.max_target_step.tolist(),
          'feedback_envelope_rad':policy.feedback_envelope.tolist(),
          'training_pd_backend':policy.manifest['training_pd_backend'],
          'hardware_control':policy.manifest['control'],
          'gain_trial':{'ramp_seconds':gain['ramp_seconds'],'hold_seconds':gain['hold_seconds']}},indent=2))


def read_only(interface,seconds):
    from .vendor.nero_v120_feedback import NeroV120Feedback
    from .hardware import can_health
    print('PASSIVE RX ONLY: no SDK connection, motor enable or target transmissions')
    print(can_health(interface))
    receiver=NeroV120Feedback(interface)
    try:
        receiver.start(); receiver.wait_for_complete(5)
        started=time.monotonic()
        while time.monotonic()-started<seconds:
            snapshot=receiver.snapshot()
            if not snapshot or not receiver.feedback_is_recent(.03):
                raise RuntimeError('Nero feedback unavailable/stale')
            print('q=',np.round(snapshot[0],4),'dq=',np.round(snapshot[1],4))
            time.sleep(.5)
        print(receiver.health_summary())
    finally:
        receiver.close()


def active(policy,args):
    from .hardware import NeroHardware
    folder=ROOT/'logs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'session.json').write_text(json.dumps({'manifest':policy.manifest,'args':vars(args)},indent=2))
    from .timing import Trace
    trace=Trace(folder/'timing.jsonl')
    hardware=NeroHardware(args.can,policy,trace=trace)
    emergency=False
    try:
        hardware.connect()
        hardware.start()
        hardware.preposition()
        from .gain_trial import apply_trial
        with Keyboard() as gain_keyboard:
            apply_trial(hardware,policy,trace,gain_keyboard)
        if args.gain_hold_only:
            print('Gain hold-only test complete; actor was NOT run.')
            return
        q, dq = hardware.read()
        result = check_post_gain_home(policy, q, dq, hardware.last_sent)
        trace.emit('post_gain_home_check_passed', q=q.tolist(), dq=dq.tolist(),
                   target=hardware.last_sent.tolist(), **result)
        policy.reset(sent=hardware.last_sent)
        if args.auto_goals:
            policy.enable_auto_goals()
        # Seed history before starting exactly as training does. The first
        # inference reuses this observation rather than inserting it twice.
        q,dq=hardware.read()
        initial_obs=policy.observe(q,dq)
        started=time.monotonic()
        hardware.policy_running=True
        if args.auto_goals:
            print('Policy takeover: 3 seconds. Automatic sampled EE goals; Q/Esc/Ctrl+C: stop.')
        else:
            print('Policy takeover: 3 seconds. I/K, arrows: 5 cm EE goal steps; Q/Esc/Ctrl+C: stop.')
        print('CSV-equivalent JSONL log:',folder/'control.jsonl')
        with Keyboard() as keyboard, (folder/'control.jsonl').open('w',buffering=1) as log:
            tick_count=0
            while tick_count < round(args.seconds/policy.dt) and time.monotonic()-started < args.seconds:
                tick=time.monotonic()
                trace.emit("policy_tick", tick=tick_count)
                keys=keyboard.poll()
                if 'exit' in keys: break
                if not args.auto_goals:
                    for key in keys:
                        if policy.move(key):
                            print('Target xyz (mount frame):', np.round(policy.target_cart(), 4),
                                  'endpoint sphere:', np.round(policy.endpoint, 4), flush=True)
                trace.emit("inference_begin", tick=tick_count)
                if tick_count==0:
                    with torch.inference_mode():
                        raw=policy.actor(torch.from_numpy(initial_obs)).numpy()[0]
                    candidate=policy.candidate_from_raw(raw)
                else:
                    q,dq=hardware.read()
                    trace.emit("actor_input_ready", tick=tick_count)
                    candidate=policy.infer(q,dq)
                trace.emit("inference_end", tick=tick_count)
                desired,blend=policy.takeover_candidate(candidate,tick_count*policy.dt)
                target=hardware.guarded_target(desired,q,max_step=policy.max_target_step,envelope=policy.feedback_envelope)
                hardware.send(target)
                policy.record_sent(target)
                ee,_=policy.fk(q)
                log.write(json.dumps({'t':tick-started,'q':q.tolist(),'dq':dq.tolist(),
                    'candidate':candidate.tolist(),'sent':target.tolist(),
                    'reference_xyz':policy.target_cart().tolist(),
                    'endpoint_xyz':(policy.goal_center_offset+sphere_to_cart(policy.endpoint)).tolist(),
                    'segment':policy.segment,'goal_timer':policy.timer,'ee_fk':ee.tolist(),
                    'error_m':float(np.linalg.norm(ee-policy.target_cart())),
                    'adapter_max_delta_rad':float(np.max(np.abs(target-candidate))),'takeover':blend,
                    'host_monotonic':tick,'command_seq':hardware.command_seq,'feedback_timing':hardware.last_feedback,
                    'kp':policy.kp.tolist(),'kd':policy.kd.tolist()})+'\n')
                previous_segment = policy.segment
                policy.advance()
                if policy.segment != previous_segment:
                    trace.emit('goal_sample', segment=policy.segment,
                               endpoint_sphere=policy.endpoint.tolist(),
                               travel_s=policy.travel_ticks*policy.dt,
                               hold_s=(policy.total_ticks-policy.travel_ticks)*policy.dt)
                    print('Sampled EE goal', policy.segment, 'sphere:',
                          np.round(policy.endpoint, 4), 'xyz (mount frame):',
                          np.round(policy.goal_center_offset+sphere_to_cart(policy.endpoint), 4),
                          flush=True)
                tick_count+=1
                time.sleep(max(0,policy.dt-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        print('Operator stop; measured-pose hold then verified disable')
    except BaseException:
        emergency=True
        raise
    finally:
        # Ignore a second Ctrl+C only while the bounded, verified shutdown runs.
        old_sigint=signal.signal(signal.SIGINT,signal.SIG_IGN)
        old_sigterm=signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            hardware.close(emergency=emergency)
        finally:
            try:trace.close()
            finally:
                signal.signal(signal.SIGINT,old_sigint)
                signal.signal(signal.SIGTERM,old_sigterm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--act',action='store_true',help='Enable and move only Nero; uses pure MIT')
    modes.add_argument('--read-only',action='store_true',help='Passive CAN feedback check; no command transmission')
    parser.add_argument('--gain-hold-only',action='store_true',help='With --act: finish after guarded gain switch, no actor')
    parser.add_argument('--can',default='can_nero')
    parser.add_argument('--seconds',type=float,default=5.0,
                        help='Actor duration, up to 60 s with continuous held/operator goals')
    args=parser.parse_args()
    args.auto_goals = True
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= MAX_ACTOR_SECONDS:
        parser.error('--seconds must be in (0, 60]')
    if args.gain_hold_only and not args.act: parser.error('--gain-hold-only requires --act')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy=FixedPolicy(ROOT/'assets')
    # Warm and verify the exported graph before any hardware startup.
    offline(policy,args.seconds,auto_goals=args.auto_goals)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.read_only:
        read_only(args.can,args.seconds)
    elif args.act:
        active(policy,args)

if __name__=='__main__': main()
