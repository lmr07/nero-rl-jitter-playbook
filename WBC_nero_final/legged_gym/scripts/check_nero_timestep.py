"""Open-loop GPU timestep comparison; outputs traces, never connects hardware.

NERO_PROBE_DT sets the candidate dt. The local validation override is restricted
to this diagnostic; the production actuator still enforces its validated limit.
"""
import os
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
from isaacgym import gymtorch
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.nero.nero_actuator import NeroActuator

dt = float(os.environ.get('NERO_PROBE_DT', '.000125'))
out = Path(os.environ.get('NERO_PROBE_OUTPUT', '/tmp/nero_dt_' + str(dt)))
args = get_args()
args.task, args.num_envs, args.headless, args.seed = 'nero_fixed', 64, True, 123
cfg, _ = task_registry.get_cfgs(args.task)
cfg.sim.dt, cfg.control.decimation = dt, round(.02 / dt)
cfg.noise.add_noise = False
cfg.env.teleop_mode = True
cfg.env.episode_length_s = 60
original_validate = NeroActuator._validate
def probe_validate(self):
    actual, policy_dt = self.dt, self.policy_dt
    try:
        self.dt = .000125
        self.policy_dt = .02
        original_validate(self)
    finally:
        self.dt = actual
        self.policy_dt = policy_dt
NeroActuator._validate = probe_validate
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
try:
    poses = env.tensor([[.006,.022,.078,.033,-.16,-.046,-.277],
                       [0,-1,0,1.8,0,0,.3],
                       [.036,1.073,-.079,1.812,-.017,-.054,-1.21]])
    initial = poses[torch.arange(env.num_envs, device=env.device) % 3]
    env.dof_pos[:, :7] = initial
    env.dof_vel.zero_()
    ids = torch.arange(env.num_envs, device=env.device)
    ids32 = ids.to(torch.int32)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state),
                                        gymtorch.unwrap_tensor(ids32), env.num_envs)
    env.actuator.reset(ids, initial, torch.zeros_like(initial))
    env.actuator.host_elapsed.fill_(3.)
    tail_peak = torch.zeros((), device=env.device)
    feedback = env.actuator.feedback
    def capture(q, dq):
        if 120 <= tick < 180:
            tail_peak.copy_(torch.maximum(tail_peak, dq.abs().max()))
        feedback(q, dq)
    env.actuator.feedback = capture
    q_trace, dq_trace, ee_trace = [], [], []
    torch.cuda.synchronize()
    begun = time.perf_counter()
    with torch.no_grad():
        for tick in range(256):
            target = initial.clone()
            if tick >= 180:
                target[:, 1] += .03
            if tick >= 216:
                target[:, 5] += .03
            env.step((target - env.default_dof_pos[:7]) / cfg.control.action_scale)
            q_trace.append(env.dof_pos[:, :7].clone())
            dq_trace.append(env.dof_vel[:, :7].clone())
            ee_trace.append((env.ee_pos - env.mount_pos).clone())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begun
    q, dq, ee = [torch.stack(x).cpu().numpy() for x in (q_trace, dq_trace, ee_trace)]
    report = {'dt_s': dt, 'seconds': elapsed, 'policy_ticks': 256, 'envs': 64,
              'finite': bool(np.isfinite(q).all() and np.isfinite(dq).all()),
              'max_hold_substep_speed_rad_s': tail_peak.item(),
              'max_hold_q_ptp_rad': float(np.ptp(q[120:180], axis=0).max()),
              'max_hold_dq_rad_s': float(np.abs(dq[120:180]).max())}
    np.savez(str(out) + '.npz', q=q, dq=dq, ee=ee)
    Path(str(out) + '.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
finally:
    env.gym.destroy_sim(env.sim)
