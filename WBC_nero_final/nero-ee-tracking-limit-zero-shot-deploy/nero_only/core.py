"""Limit-aware EE tracking ABI. No CAN, A2, IMU or Isaac Gym dependencies."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

ROOT = Path(__file__).resolve().parents[1]


def sphere_to_cart(s):
    radius, pitch, yaw = np.asarray(s, dtype=float)
    return radius * np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])


def cart_to_sphere(p):
    p = np.asarray(p, dtype=float)
    radius = np.linalg.norm(p)
    if radius < 1.e-6 or not np.isfinite(p).all():
        raise ValueError('Target must be finite and away from the mounting origin')
    return np.array([radius, np.arcsin(np.clip(p[2]/radius, -1, 1)), np.arctan2(p[1],p[0])])


def training_rpy(rotation):
    # Match Isaac Gym get_euler_xyz rather than changing the Euler branch.
    x,y,z,w = rotation.as_quat()
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])


class FixedPolicy:
    """Limit-aware EE observation and target-slew ABI, without Isaac Gym or CAN."""

    KEY_STEPS = {
        'i': (0., 0., .05), 'k': (0., 0., -.05),
        'left': (0., .05, 0.), 'right': (0., -.05, 0.),
        'up': (.05, 0., 0.), 'down': (-.05, 0., 0.),
    }

    def __init__(self, assets=ROOT/'assets'):
        self.assets = Path(assets)
        self.manifest = json.loads((self.assets/'manifest.json').read_text())
        for name,digest in self.manifest['sha256'].items():
            if hashlib.sha256((self.assets/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Deployment artifact was modified: '+name)
        self.cfg = json.loads((self.assets/'nero_config.json').read_text())['env']
        assert self.cfg['env']['num_single_obs'] == 39
        assert self.cfg['env']['frame_stack'] == 16
        assert self.cfg['env']['num_actions'] == 7
        self.dt = self.cfg['sim']['dt'] * self.cfg['control']['decimation']
        assert abs(self.dt - .02) < 1.e-9
        self.names = [f'nero_joint{i}' for i in range(1,8)]
        self.default = np.array([self.cfg['init_state']['default_joint_angles'][k] for k in self.names])
        self.training_kp = np.array([self.cfg['control']['stiffness'][k] for k in self.names])
        self.training_kd = np.array([self.cfg['control']['damping'][k] for k in self.names])
        self.kp = np.asarray(self.manifest['startup_gains']['kp'], dtype=float).copy()
        self.kd = np.asarray(self.manifest['startup_gains']['kd'], dtype=float).copy()
        self.scale = self.cfg['control']['action_scale']
        self.action_mode = self.cfg['control'].get('action_mode')
        if self.action_mode != 'absolute' or self.cfg['control']['pd_backend'] != 'physx_position':
            raise RuntimeError('This deployment requires stable EE absolute/native-PD semantics')
        self.max_target_step = np.asarray(self.cfg['control']['target_speed_limits_rad_s'], dtype=float) * self.dt
        self.guards = json.loads((self.assets/'guard_profile.json').read_text())
        self.feedback_envelope = np.asarray(self.guards['feedback_envelope_rad'], dtype=float)
        self.takeover_s = 3.0
        if (self.feedback_envelope.shape != (7,) or not np.all(np.isfinite(self.feedback_envelope))
            or not np.all(self.max_target_step > 0) or not np.all(self.feedback_envelope > 0)
            or not np.all(self.feedback_envelope < 1.0)):
            raise RuntimeError('Invalid seven-cycle target limits')
        self.chain = []
        lo,hi = [],[]
        urdf = ET.parse(self.assets/'nero-fixed.urdf').getroot()
        for name in self.names:
            joint = urdf.find(f"joint[@name='{name}']")
            origin = joint.find('origin')
            xyz = np.array([float(x) for x in origin.get('xyz').split()])
            rot = R.from_euler('xyz',[float(x) for x in origin.get('rpy').split()])
            axis = np.array([float(x) for x in joint.find('axis').get('xyz').split()])
            self.chain.append((xyz,rot,axis))
            lo.append(float(joint.find('limit').get('lower')))
            hi.append(float(joint.find('limit').get('upper')))
        self.lower,self.upper = np.array(lo),np.array(hi)
        self.target_margin = np.asarray(self.cfg['control']['target_position_margin_rad'], dtype=float)
        if (self.target_margin.shape != (7,) or not np.isfinite(self.target_margin).all()
            or (self.target_margin <= 0).any()):
            raise RuntimeError('Invalid trained joint target margins')
        self.target_lower = self.lower + self.target_margin
        self.target_upper = self.upper - self.target_margin
        if (self.target_lower >= self.target_upper).any():
            raise RuntimeError('Trained target margins leave no command range')
        self.mount_rotation = R.from_quat(self.cfg['init_state']['rot'])
        self.tool_rotation = R.from_quat(self.cfg['asset']['ee_orientation_body_to_goal_frame_quat'])
        self.tool_offset = np.array(self.cfg['asset']['ee_position_body_offset'])
        self.goal_center_offset = np.array([0., 0., -float(self.cfg['goal_ee']['center_below_mount_m'])])
        self.ranges = self.cfg['goal_ee']['ranges']
        self.bounds = np.array([self.ranges[k] for k in ('pos_l', 'pos_p', 'pos_y')])
        self.initial_target = np.asarray(self.cfg['control']['initial_position_targets'], dtype=float)
        if self.initial_target.shape != (7,) or self.bounds.shape != (3, 2):
            raise RuntimeError('Invalid stable EE target configuration')
        if ((self.initial_target < self.target_lower).any()
            or (self.initial_target > self.target_upper).any()):
            raise RuntimeError('Home target violates trained joint target margins')
        self.rng = np.random.default_rng(7600)
        self.actor = torch.jit.load(str(self.assets/'policy.pt'),map_location='cpu').eval()
        self.reset()

    def reset(self, sent=None):
        self.history = np.zeros((16,39),dtype=np.float32)
        self.last_sent = self.initial_target.copy() if sent is None else np.asarray(sent, dtype=float).copy()
        if self.last_sent.shape != (7,) or not np.isfinite(self.last_sent).all():
            raise ValueError('Invalid initial sent target')
        self.previous_action = (self.last_sent-self.default)/self.scale
        home_pos, home_rot = self.fk(self.default)
        self.home = np.asarray(self.ranges['init_pos_start'], dtype=float)
        self.home_orn = np.asarray(self.ranges['init_orn_start'], dtype=float)
        if np.linalg.norm(sphere_to_cart(self.home)+self.goal_center_offset-home_pos) > .001:
            raise RuntimeError('Saved initial goal does not match Nero URDF FK')
        home_heading = R.from_euler('z', np.arctan2(home_pos[1], home_pos[0]))
        if np.max(np.abs(training_rpy(home_heading.inv() * home_rot)-self.home_orn)) > .001:
            raise RuntimeError('Saved initial orientation does not match Nero URDF FK')
        self.start = self.home.copy()
        self.endpoint = self.home.copy()
        self.current = self.home.copy()
        self.orn_start = self.home_orn.copy()
        self.orn_end = np.asarray(self.ranges['init_orn_end'], dtype=float)
        self.orn = self.home_orn.copy()
        self.timer = 0
        self.travel_ticks = round(2.0 / self.dt)
        self.segment = 0  # initial pitch ramp; then operator-commanded goals
        self.last_target = self.target_cart()
        self.last_vel = np.zeros(3)
        self.first = True

    def target_cart(self):
        return sphere_to_cart(self.current) + self.goal_center_offset

    def fk(self,q):
        pos,rotation = np.zeros(3),self.mount_rotation
        for angle,(xyz,origin,axis) in zip(q,self.chain):
            pos = pos + rotation.apply(xyz)
            rotation = rotation * origin * R.from_rotvec(angle*axis)
        return pos+rotation.apply(self.tool_offset), rotation*self.tool_rotation

    def advance(self):
        """Match the stable task's linear spherical and orientation interpolation."""
        self.timer += 1
        t = min(self.timer / self.travel_ticks, 1.0)
        self.current = self.start * (1-t) + self.endpoint * t
        self.orn = self.orn_start * (1-t) + self.orn_end * t

    def move(self, key):
        if key not in self.KEY_STEPS:
            return False
        if self.segment == 0 and self.timer < self.travel_ticks:
            return False
        requested = cart_to_sphere(sphere_to_cart(self.endpoint) + self.KEY_STEPS[key])
        next_endpoint = np.clip(requested, self.bounds[:, 0], self.bounds[:, 1])
        self.start = self.current.copy()
        self.endpoint = next_endpoint
        self.orn_start = self.orn.copy()
        self.orn_end = np.zeros(3)
        self.timer = 0
        self.travel_ticks = self.rng.uniform(*self.cfg['goal_ee']['traj_time']) / self.dt
        self.segment += 1
        return True

    def observe(self,q,dq):
        q,dq = np.asarray(q),np.asarray(dq)
        if q.shape != (7,) or dq.shape != (7,) or not np.isfinite(np.r_[q,dq]).all():
            raise ValueError('Invalid seven-axis joint feedback')
        target = self.target_cart()
        vel = (target-self.last_target)/self.dt
        acc = (vel-self.last_vel)/self.dt
        if self.first:
            vel[:],acc[:] = 0,0
        pos,orn = self.fk(q)
        heading = R.from_euler('z',np.arctan2(target[1],target[0]))
        rpy = training_rpy(heading.inv()*orn)
        command = np.r_[self.current,self.orn]*[.5,1,1.3,.5,.5,.5]
        frame = np.r_[q-self.default,dq*self.cfg['normalization']['obs_scales']['dof_vel'],
                      self.previous_action,rpy,command,target,vel,acc].astype(np.float32)
        assert frame.shape == (39,)
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        self.last_target,self.last_vel = target.copy(),vel.copy()
        self.first = False
        return np.clip(self.history.reshape(1,624),-100,100).copy()

    def infer(self,q,dq):
        obs = self.observe(q,dq)
        with torch.inference_mode():
            raw = self.actor(torch.from_numpy(obs)).numpy()[0]
        return self.candidate_from_raw(raw)

    def candidate_from_raw(self, raw):
        raw = np.asarray(raw)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise RuntimeError('Invalid actor output')
        clipped = np.clip(raw,-self.cfg['normalization']['clip_actions'],self.cfg['normalization']['clip_actions'])
        requested = np.clip(self.default + self.scale*clipped, self.target_lower, self.target_upper)
        return np.clip(requested, self.last_sent-self.max_target_step, self.last_sent+self.max_target_step)

    def takeover_candidate(self, candidate, elapsed):
        blend = min(max(elapsed / self.takeover_s, 0.), 1.) if self.takeover_s else 1.
        return self.last_sent + blend * (candidate - self.last_sent), blend

    def record_sent(self,target):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Invalid sent target')
        self.last_sent = target.copy()
        self.previous_action = (target-self.default)/self.scale
