"""Check inverse-identified MIT gains with actor disabled before takeover."""
import json
import time
import numpy as np
from .core import ROOT

def profile(assets=ROOT/'assets'):
    p = json.loads((assets/'gain_profile.json').read_text())
    fit = json.loads((assets/'identification_fit.json').read_text())
    rows = sorted(fit['joints'], key=lambda row: row['joint'])
    if [row['joint'] for row in rows] != list(range(1, 8)):
        raise RuntimeError('Missing joint identification')
    training = json.loads((assets/'nero_config.json').read_text())['env']['control']
    for key, field, index in (('kp', 'stiffness', 0), ('kd', 'damping', 1)):
        actual = np.asarray(p[key], dtype=float)
        expected = np.asarray([training[field][f'nero_joint{joint}']/row['command_only']['train']['coeff'][index]
                               for joint, row in enumerate(rows, 1)])
        if actual.shape != (7,) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-12):
            raise RuntimeError('Gain profile differs from inverse identification: '+key)
    return p

def equilibrium_target(anchor, initial_target, initial_kp, current_kp):
    """Preserve the static P effort at the measured takeover pose.

    At v=0, the effective stiffness multiplier cancels in
    alpha*kp_new*(target_new-anchor) = alpha*kp_old*(target_old-anchor).
    This is a fixed-target calibration for the gain test, not torque feedforward.
    """
    anchor = np.asarray(anchor, dtype=float)
    initial_target = np.asarray(initial_target, dtype=float)
    initial_kp = np.asarray(initial_kp, dtype=float)
    current_kp = np.asarray(current_kp, dtype=float)
    if any(v.shape != (7,) for v in (anchor, initial_target, initial_kp, current_kp)):
        raise ValueError('Expected seven joint values')
    if not np.isfinite(np.r_[anchor, initial_target, initial_kp, current_kp]).all() or (current_kp <= 0).any():
        raise ValueError('Invalid equilibrium conversion')
    return anchor + (initial_kp/current_kp)*(initial_target-anchor)

def apply_trial(hw, policy, trace, keyboard, clock=time):
    p = profile(policy.assets)
    initial_target = hw.last_sent.copy()
    anchor, anchor_velocity = hw.read()
    if np.max(np.abs(anchor-initial_target)) > p['max_tracking_error_rad'] or np.max(np.abs(anchor_velocity)) > 0.05:
        raise RuntimeError('Preposition was not settled; actor NOT started')
    old_kp, old_kd = policy.kp.copy(), policy.kd.copy()
    kp, kd = np.asarray(p['kp']), np.asarray(p['kd'])
    start = clock.monotonic()
    print('Actor OFF: 5 s gain ramp + 5 s measured-pose hold.', flush=True)
    trace.emit('gain_trial_begin', profile=p, initial_target=initial_target.tolist(),
               anchor=anchor.tolist(), anchor_velocity=anchor_velocity.tolist())
    while True:
        tick = clock.monotonic()
        elapsed = tick-start
        if 'exit' in keyboard.poll():
            raise KeyboardInterrupt()
        q, dq = hw.read()
        x = min(elapsed/p['ramp_seconds'], 1.)
        blend = x*x*(3-2*x)
        next_kp = old_kp + blend*(kp-old_kp)
        next_kd = old_kd + blend*(kd-old_kd)
        target = equilibrium_target(anchor, initial_target, old_kp, next_kp)
        drift, speed = np.abs(q-anchor), np.abs(dq)
        target_step = np.abs(target-hw.last_sent)
        target_error = np.abs(target-q)
        invalid = ((target < policy.target_lower).any() or (target > policy.target_upper).any()
                   or (target_step > policy.max_target_step+1e-9).any()
                   or (target_error > np.minimum(policy.feedback_envelope, .20)).any())
        if np.max(drift) > p['max_tracking_error_rad'] or np.max(speed) > p['max_speed_rad_s'] or invalid:
            trace.emit('gain_trial_failed', elapsed_s=elapsed, q=q.tolist(), dq=dq.tolist(),
                       anchor=anchor.tolist(), target=target.tolist(), drift_rad=drift.tolist(),
                       speed_rad_s=speed.tolist(), target_step_rad=target_step.tolist(),
                       target_error_rad=target_error.tolist(), kp=policy.kp.tolist(), kd=policy.kd.tolist())
            raise RuntimeError('Inverse-gain pose hold failed: '
                f'J{np.argmax(drift)+1} drift={np.max(drift):.4f} rad; '
                f'J{np.argmax(speed)+1} speed={np.max(speed):.4f} rad/s; '
                f'target envelope/step/limit violation={invalid}; actor NOT started')
        policy.kp, policy.kd = next_kp, next_kd
        hw.send(target)
        trace.emit('gain_trial_sample', q=q.tolist(), dq=dq.tolist(), anchor=anchor.tolist(),
                   target=target.tolist(), kp=policy.kp.tolist(), kd=policy.kd.tolist())
        if elapsed >= p['ramp_seconds'] + p['hold_seconds']:
            break
        clock.sleep(max(0, .02-(clock.monotonic()-tick)))
    trace.emit('gain_trial_passed', kp=policy.kp.tolist(), kd=policy.kd.tolist())
    print('Measured-pose gain check passed. Kp=',policy.kp,'Kd=',policy.kd,
          'final MIT target=',hw.last_sent,flush=True)
