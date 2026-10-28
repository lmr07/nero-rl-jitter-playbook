"""Physical-unit request accounting before clipping and slew limiting."""
import torch
import xml.etree.ElementTree as ET


def robust_square(value):
    return 2 * (torch.sqrt(1 + value.square()) - 1)


@torch.enable_grad()
def gravity_position_targets(urdf_path, angles, stiffness, base_quaternion, gravity):
    root = ET.parse(urdf_path).getroot()
    names = [f'nero_joint{joint}' for joint in range(1, 8)]
    position = torch.tensor([angles[name] for name in names], dtype=torch.float64, requires_grad=True)
    identity = torch.eye(3, dtype=torch.float64)

    def vector(value):
        return torch.tensor([float(component) for component in value.split()], dtype=torch.float64)

    def rotation(axis, angle):
        axis = axis/axis.norm()
        axis_x, axis_y, axis_z = axis
        skew = torch.stack((torch.stack((axis_x*0, -axis_z, axis_y)),
                            torch.stack((axis_z, axis_x*0, -axis_x)),
                            torch.stack((-axis_y, axis_x, axis_x*0))))
        return torch.cos(angle)*identity + (1-torch.cos(angle))*axis[:, None]*axis[None, :] + torch.sin(angle)*skew

    quaternion = torch.tensor(base_quaternion, dtype=torch.float64)
    quaternion = quaternion/quaternion.norm()
    axis_norm = quaternion[:3].norm()
    base_rotation = (identity if axis_norm < 1e-12 else
                     rotation(quaternion[:3]/axis_norm, 2*torch.atan2(axis_norm, quaternion[3])))
    joints = list(root.findall('joint'))
    child_links = {joint.find('child').get('link') for joint in joints}
    root_link = next(link.get('name') for link in root.findall('link') if link.get('name') not in child_links)
    poses = {root_link:(torch.zeros(3, dtype=torch.float64), base_rotation)}
    while joints:
        progressed = False
        for joint in joints[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            parent_position, parent_rotation = poses[parent]
            origin = joint.find('origin')
            offset = vector(origin.get('xyz', '0 0 0')) if origin is not None else torch.zeros(3, dtype=torch.float64)
            rpy = vector(origin.get('rpy', '0 0 0')) if origin is not None else torch.zeros(3, dtype=torch.float64)
            origin_rotation = (rotation(vector('0 0 1'), rpy[2]) @ rotation(vector('0 1 0'), rpy[1])
                               @ rotation(vector('1 0 0'), rpy[0]))
            link_position = parent_position+parent_rotation@offset
            link_rotation = parent_rotation@origin_rotation
            name = joint.get('name')
            angle = position[names.index(name)] if name in names else torch.tensor(angles.get(name, 0.), dtype=torch.float64)
            axis_element = joint.find('axis')
            axis = vector(axis_element.get('xyz')) if axis_element is not None else vector('1 0 0')
            if joint.get('type') in ('revolute', 'continuous'):
                link_rotation = link_rotation@rotation(axis, angle)
            elif joint.get('type') == 'prismatic':
                link_position = link_position+link_rotation@axis*angle
            poses[joint.find('child').get('link')] = (link_position, link_rotation)
            joints.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF joints do not form a connected tree')
    potential = position.sum()*0
    gravity_vector = torch.tensor(gravity, dtype=torch.float64)
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue
        mass = float(inertial.find('mass').get('value'))
        origin = inertial.find('origin')
        center = vector(origin.get('xyz', '0 0 0')) if origin is not None else vector('0 0 0')
        link_position, link_rotation = poses[link.get('name')]
        potential = potential-mass*torch.dot(gravity_vector, link_position+link_rotation@center)
    effort = torch.autograd.grad(potential, position)[0]
    gains = torch.tensor([stiffness[name] for name in names], dtype=torch.float64)
    return (position+effort/gains).detach().tolist()


def configure_smooth_start(cfg, root_dir):
    path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=root_dir)
    home = dict(cfg.init_state.default_joint_angles)
    cfg.control.initial_position_targets = tuple(gravity_position_targets(
        path, home, cfg.control.stiffness, cfg.init_state.rot, cfg.sim.gravity))
    targets = []
    for anchor in cfg.goal_ee.phase_start_joint_angles:
        pose = dict(home)
        pose.update({f'nero_joint{joint}':angle for joint, angle in enumerate(anchor, 1)})
        targets.append(tuple(gravity_position_targets(path, pose, cfg.control.stiffness,
                                                       cfg.init_state.rot, cfg.sim.gravity)))
    cfg.goal_ee.phase_start_target_joint_angles = tuple(targets)


class SmoothRequestHistory:
    def __init__(self, num_envs, device):
        self.request = torch.zeros(num_envs, 7, device=device)
        self.request_delta = torch.zeros_like(self.request)
        self.request_second_delta = torch.zeros_like(self.request)
        self.target_second_delta = torch.zeros_like(self.request)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids, target):
        self.request[ids] = target
        self.request_delta[ids] = 0
        self.request_second_delta[ids] = 0
        self.target_second_delta[ids] = 0
        self.steps[ids] = 0

    def measure(self, request, target, previous_target, previous_target_delta, dt, norms):
        request_delta = request - self.request
        request_second = request_delta - self.request_delta
        request_third = request_second - self.request_second_delta
        target_delta = target - previous_target
        target_second = target_delta - previous_target_delta
        target_third = target_second - self.target_second_delta
        first_valid = (self.steps > 0).float()
        second_valid = (self.steps > 1).float()
        third_valid = (self.steps > 2).float()
        terms = {
            'request_gap': robust_square((request-target)/norms.request_gap_norm_rad).mean(-1),
            'request_rate': robust_square(request_delta/(dt*norms.request_speed_norm_rad_s)).mean(-1)*first_valid,
            'request_acceleration': robust_square(request_second/(dt**2*norms.acceleration_norm_rad_s2)).mean(-1)*second_valid,
            'request_jerk': robust_square(request_third/(dt**3*norms.jerk_norm_rad_s3)).mean(-1)*third_valid,
            'sent_target_rate': robust_square(target_delta/(dt*norms.request_speed_norm_rad_s)).mean(-1),
            'sent_target_acceleration': robust_square(target_second/(dt**2*norms.acceleration_norm_rad_s2)).mean(-1)*first_valid,
            'sent_target_jerk': robust_square(target_third/(dt**3*norms.jerk_norm_rad_s3)).mean(-1)*second_valid,
            'target_reversal': torch.relu(-target_delta*previous_target_delta).mean(-1)/(dt*norms.request_speed_norm_rad_s)**2,
        }
        metrics = {
            'request_speed_rms_rad_s': request_delta.square().mean(-1).sqrt()/dt*first_valid,
            'request_acceleration_rms_rad_s2': request_second.square().mean(-1).sqrt()/dt**2*second_valid,
            'request_jerk_rms_rad_s3': request_third.square().mean(-1).sqrt()/dt**3*third_valid,
            'request_gap_rms_rad': (request-target).square().mean(-1).sqrt(),
        }
        self.request.copy_(request)
        self.request_delta.copy_(request_delta)
        self.request_second_delta.copy_(request_second)
        self.target_second_delta.copy_(target_second)
        self.steps += 1
        return terms, metrics
