import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import sys
sys.path.append(unitree_rl_gym_path)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.envs.a2.a2nero_pos_force_config import (
    apply_a2nero_domain_rand_stage,
)
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.isaacgym_utils import sphere2cart, cart2sphere
from legged_gym.utils.math import wrap_to_pi

import json
import numpy as np
import subprocess
import threading
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


UNITY_TO_ROBOT_ROT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


class VRTeleopGateway:
    def __init__(self, path, host, port, output_hz, udp_timeout):
        self.path = path
        self.host = host
        self.port = port
        self.output_hz = output_hz
        self.udp_timeout = udp_timeout
        self.process = None
        self.thread = None
        self.lock = threading.Lock()
        self.latest_command = None
        self.latest_monotonic_time = None
        self.last_status_line = None
        self.last_error = None

    def start(self):
        command = [
            sys.executable,
            "-u",
            self.path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--output-hz",
            str(self.output_hz),
            "--udp-timeout",
            str(self.udp_timeout),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[VR teleop] started gateway: {' '.join(command)}")

    def _read_loop(self):
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                self.last_status_line = line
                print(f"[VR gateway] {line}")
                continue

            if command.get("type") != "dog_arm_teleop":
                continue

            with self.lock:
                self.latest_command = command
                self.latest_monotonic_time = time.monotonic()

        return_code = self.process.poll()
        self.last_error = f"gateway exited with code {return_code}"
        print(f"[VR teleop] {self.last_error}")

    def get_latest(self):
        with self.lock:
            if self.latest_command is None:
                return None, None
            return self.latest_command, time.monotonic() - self.latest_monotonic_time

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


def _clip_scalar(value, value_range):
    return max(value_range[0], min(value_range[1], value))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float_triplet(values, default=(0.0, 0.0, 0.0)):
    if not isinstance(values, list) or len(values) != 3:
        return list(default)
    try:
        return [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return list(default)


def _as_float_quat_xyzw(values):
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        quat = np.array([float(values[0]), float(values[1]), float(values[2]), float(values[3])], dtype=np.float64)
    except (TypeError, ValueError):
        return None

    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return None
    return quat / norm


def _quat_xyzw_conjugate(quat):
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_xyzw_mul(q, r):
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_xyzw_to_rot(quat):
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rot_to_rpy(rot):
    pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _unity_relative_quat_to_robot_rpy(current_quat, zero_quat):
    delta_unity_quat = _quat_xyzw_mul(current_quat, _quat_xyzw_conjugate(zero_quat))
    delta_unity_quat /= np.linalg.norm(delta_unity_quat)
    delta_unity_rot = _quat_xyzw_to_rot(delta_unity_quat)
    delta_robot_rot = UNITY_TO_ROBOT_ROT @ delta_unity_rot @ UNITY_TO_ROBOT_ROT.T
    return _rot_to_rpy(delta_robot_rot)


def _zero_vr_motion(env, state):
    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    state["arm_anchor_cart"] = None
    state["arm_anchor_orn_rpy"] = None
    state["arm_anchor_unity_quat"] = None


def apply_vr_teleop_command(env, state, command, command_age, timeout_s, orientation_scale):
    timed_out = command is None or command_age is None or command_age > timeout_s or command.get("timeout", False)
    if timed_out:
        _zero_vr_motion(env, state)
        return

    dog = command.get("dog", {})
    vx = _clip_scalar(_safe_float(dog.get("vx_mps", 0.0)), env.command_ranges["lin_vel_x"])
    vy = _clip_scalar(_safe_float(dog.get("vy_mps", 0.0)), env.command_ranges["lin_vel_y"])
    wz = _clip_scalar(_safe_float(dog.get("wz_radps", 0.0)), env.command_ranges["ang_vel_yaw"])
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = wz

    arm = command.get("arm", {})
    if not arm.get("enabled", False):
        state["arm_anchor_cart"] = None
        state["arm_anchor_orn_rpy"] = None
        state["arm_anchor_unity_quat"] = None
        return

    if arm.get("just_enabled", False) or state.get("arm_anchor_cart") is None:
        state["arm_anchor_cart"] = sphere2cart(env.curr_ee_goal_sphere).clone()

    delta_position = torch.tensor(
        _as_float_triplet(arm.get("delta_position_robot_m")),
        dtype=torch.float,
        device=env.device,
    ).view(1, 3)
    target_cart = state["arm_anchor_cart"] + delta_position.repeat(env.num_envs, 1)
    env._set_teleop_ee_goal(ee_goal_sphere=cart2sphere(target_cart))

    unity_quat = _as_float_quat_xyzw(arm.get("target_orientation_unity_xyzw"))
    if unity_quat is None:
        return

    if arm.get("just_enabled", False) or state.get("arm_anchor_unity_quat") is None:
        state["arm_anchor_unity_quat"] = unity_quat
        state["arm_anchor_orn_rpy"] = env.curr_ee_goal_orn_rpy.clone()

    delta_rpy = torch.tensor(
        _unity_relative_quat_to_robot_rpy(unity_quat, state["arm_anchor_unity_quat"]),
        dtype=torch.float,
        device=env.device,
    ).view(1, 3)
    target_orn_rpy = wrap_to_pi(state["arm_anchor_orn_rpy"] + orientation_scale * delta_rpy.repeat(env.num_envs, 1))
    env._set_teleop_ee_goal(ee_goal_orn_rpy=target_orn_rpy)



def play(args):
    record_dir = os.getenv("A2NERO_RECORD_DIR")
    record_steps = int(os.getenv("A2NERO_RECORD_STEPS", "500"))
    record_stride = max(1, int(os.getenv("A2NERO_RECORD_STRIDE", "1")))
    record_follow_camera = os.getenv("A2NERO_RECORD_FOLLOW_CAMERA", "1") != "0"
    record_nominal = os.getenv("A2NERO_RECORD_NOMINAL", "1") != "0"
    record_command = None
    if os.getenv("A2NERO_RECORD_COMMAND_X") is not None:
        record_command = (
            float(os.getenv("A2NERO_RECORD_COMMAND_X", "0")),
            float(os.getenv("A2NERO_RECORD_COMMAND_Y", "0")),
            float(os.getenv("A2NERO_RECORD_COMMAND_YAW", "0")),
        )
    camera_distance = float(os.getenv("A2NERO_RECORD_CAMERA_DISTANCE", "1.8"))
    camera_height = float(os.getenv("A2NERO_RECORD_CAMERA_HEIGHT", "1.2"))
    if record_dir:
        record_dir = os.path.abspath(record_dir)
        os.makedirs(record_dir, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    domain_rand_stage = int(getattr(args, "domain_rand_stage", 1))
    full_identified_dr = domain_rand_stage in (4, 5)
    if full_identified_dr:
        env_cfg, train_cfg = apply_a2nero_domain_rand_stage(
            env_cfg,
            train_cfg,
            domain_rand_stage,
        )
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # Preserve the historical nominal visualization for legacy Stage 1/2.
    # Identified Stage 4/5 explicitly means full profile DR, including the
    # checkpoint-compatible observation noise configuration.
    if not full_identified_dr:
        env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.test = True
    env_cfg.env.teleop_mode = args.VR_teleop_mode if record_dir else True
    env_cfg.env.episode_length_s = 1e9
    if record_dir and args.seed is not None:
        # make_env only applies --seed through the training config, so set the
        # environment seed explicitly for reproducible checkpoint recordings.
        env_cfg.seed = int(args.seed)

    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.draw_collision_bbox = False
    env.draw_world_origin = False
    env.draw_ee_goal_center = False
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.stage3_load_optimizer_state = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if full_identified_dr:
        # A checkpoint stores the training progress at which it was saved, but
        # a full-DR visualization must be unambiguous for every checkpoint.
        # Set progress to one and reset after loading so episode-persistent
        # motor, delay, contact and sensor samples are all drawn at full scale.
        env.set_stage3_curriculum_progress(1, 1)
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(env_ids)
        obs = env.get_observations()
        print(
            "[play] full identified DR enabled: "
            f"stage={domain_rand_stage}, "
            f"state={env.get_stage3_curriculum_state()}"
        )

    # Stage-3 best checkpoints are selected with deterministic nominal
    # evaluation.  Use the same execution mode for a recording unless the
    # caller explicitly requests the curriculum state stored in the model.
    if (
        record_dir
        and record_nominal
        and not full_identified_dr
        and hasattr(env, "set_stage3_nominal_override")
    ):
        env.set_stage3_nominal_override(True)
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(env_ids)
        if record_command is not None:
            env.commands[:, 0] = record_command[0]
            env.commands[:, 1] = record_command[1]
            env.commands[:, 2] = record_command[2]
        zero_actions = torch.zeros(
            env.num_envs, env.num_actions, device=env.device, requires_grad=False
        )
        env.step(zero_actions)
        obs = env.get_observations()
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:

        import copy
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        os.makedirs(path, exist_ok=True)
        adaptation_module_path = os.path.join(path, 'adaptation_module.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_encoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_module_path)
        print('Exported policy as jit script to: ', adaptation_module_path)

        adaptation_decoder_path = os.path.join(path, 'adaptation_decoder.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_decoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_decoder_path)
        print('Exported policy as jit script to: ', adaptation_decoder_path)

        actor_body_path = os.path.join(path, 'actor_body.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor_body).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(actor_body_path)
        print('Exported policy as jit script to: ', actor_body_path)

    if VISUAL_PRED:
        fig_linvel = plt.figure()
        ax_linvel = fig_linvel.add_subplot(111, projection='3d')
        vector1_linvel = np.array([0, 0, 0])
        vector2_linvel = np.array([0, 0, 0])
        ax_linvel.set_xlim([-2, 2])
        ax_linvel.set_ylim([-2, 2])
        ax_linvel.set_zlim([-2, 2])
        ax_linvel.set_xlabel('X axis')
        ax_linvel.set_ylabel('Y axis')
        ax_linvel.set_zlabel('Z axis')
        line1_linvel, = ax_linvel.plot([0, vector1_linvel[0]], [0, vector1_linvel[1]], [0, vector1_linvel[2]], marker='o', label='linvel_pred')
        line2_linvel, = ax_linvel.plot([0, vector2_linvel[0]], [0, vector2_linvel[1]], [0, vector2_linvel[2]], marker='o', label='linvel_gt')
        ax_linvel.legend()

        fig_eepos = plt.figure()
        ax_eepos = fig_eepos.add_subplot(111, projection='3d')
        vector1_eepos = np.array([0, 0, 0])
        vector2_eepos = np.array([0, 0, 0])
        ax_eepos.set_xlim([-2, 2])
        ax_eepos.set_ylim([-2, 2])
        ax_eepos.set_zlim([-2, 2])
        ax_eepos.set_xlabel('X axis')
        ax_eepos.set_ylabel('Y axis')
        ax_eepos.set_zlabel('Z axis')
        line1_eepos, = ax_eepos.plot([0, vector1_eepos[0]], [0, vector1_eepos[1]], [0, vector1_eepos[2]], marker='o', label='eepos_pred')
        line2_eepos, = ax_eepos.plot([0, vector2_eepos[0]], [0, vector2_eepos[1]], [0, vector2_eepos[2]], marker='o', label='eepos_gt')
        ax_eepos.legend()
    

    vr_gateway = None
    vr_state = {"arm_anchor_cart": None, "arm_anchor_orn_rpy": None, "arm_anchor_unity_quat": None}
    if args.VR_teleop_mode:
        if not os.path.exists(args.vr_gateway_path):
            raise FileNotFoundError(f"VR teleop gateway not found: {args.vr_gateway_path}")
        vr_gateway = VRTeleopGateway(
            path=args.vr_gateway_path,
            host=args.vr_gateway_host,
            port=args.vr_gateway_port,
            output_hz=args.vr_gateway_output_hz,
            udp_timeout=args.vr_gateway_udp_timeout,
        )
        vr_gateway.start()

    env.play = True
    policy_info = {}
    try:
        recorded_frames = 0
        for i in range(100*int(env.max_episode_length)):
            if record_command is not None:
                env.commands[:, 0] = record_command[0]
                env.commands[:, 1] = record_command[1]
                env.commands[:, 2] = record_command[2]
            if vr_gateway is not None:
                vr_command, vr_command_age = vr_gateway.get_latest()
                apply_vr_teleop_command(
                    env,
                    vr_state,
                    vr_command,
                    vr_command_age,
                    args.vr_command_timeout,
                    args.vr_arm_orientation_scale,
                )

            actions = policy(obs, policy_info)
            # breakpoint()
            if FIX_COMMAND:
                env.commands[:, 0] = 0.    # 1.0
                env.commands[:, 1] = 0.
                env.commands[:, 2] = 0.0
                env.commands[:, 3] = 0.
                # env.gait_indices[:] = 0.
            obs, rews, dones, infos = env.step(actions.detach())
            if record_dir and i % record_stride == 0:
                if env.viewer is None:
                    raise RuntimeError("Checkpoint recording requires a non-headless viewer")
                if record_follow_camera:
                    root = env.root_states[0, :3].detach().cpu().numpy()
                    env.set_camera(
                        [
                            root[0] - camera_distance,
                            root[1] - camera_distance,
                            root[2] + camera_height,
                        ],
                        [root[0], root[1], root[2] + 0.35],
                    )
                frame_path = os.path.join(
                    record_dir, "frame_{:05d}.png".format(recorded_frames)
                )
                env.gym.write_viewer_image_to_file(env.viewer, frame_path)
                recorded_frames += 1
                if recorded_frames >= record_steps:
                    print(
                        "Recorded {} frames to {}".format(recorded_frames, record_dir)
                    )
                    break
            if torch.any(dones):
                vr_state["arm_anchor_cart"] = None
                vr_state["arm_anchor_orn_rpy"] = None
                vr_state["arm_anchor_unity_quat"] = None
            if VISUAL_PRED:
                pass
                # linvel_pred = policy_info["latents"][0, 0:3]
                # vector1_linvel = linvel_pred
                # vector2_linvel = env.base_lin_vel[0].detach().cpu().numpy() * env.obs_scales.lin_vel
                # line1_linvel.set_data([0, vector1_linvel[0]], [0, vector1_linvel[1]])
                # line1_linvel.set_3d_properties([0, vector1_linvel[2]])

                # line2_linvel.set_data([0, vector2_linvel[0]], [0, vector2_linvel[1]])
                # line2_linvel.set_3d_properties([0, vector2_linvel[2]])

                # eepos_pred = policy_info["latents"][0, 3:6]
                # vector1_eepos = eepos_pred
                
                # vector2_eepos = np.array([env.ee_pos_sphe_arm[0,0].detach().cpu().numpy() * env.obs_scales.ee_sphe_radius_cmd,
                #                                 env.ee_pos_sphe_arm[0,1].detach().cpu().numpy() * env.obs_scales.ee_sphe_pitch_cmd,
                #                                 env.ee_pos_sphe_arm[0,2].detach().cpu().numpy() * env.obs_scales.ee_sphe_yaw_cmd])
                # line1_eepos.set_data([0, vector1_eepos[0]], [0, vector1_eepos[1]])
                # line1_eepos.set_3d_properties([0, vector1_eepos[2]])
                # line2_eepos.set_data([0, vector2_eepos[0]], [0, vector2_eepos[1]])
                # line2_eepos.set_3d_properties([0, vector2_eepos[2]])
                # plt.draw()
                # plt.pause(0.001)
    finally:
        if vr_gateway is not None:
            vr_gateway.close()

if __name__ == '__main__':
    EXPORT_POLICY = os.getenv("A2NERO_EXPORT_POLICY", "1") != "0"
    RECORD_FRAMES = bool(os.getenv("A2NERO_RECORD_DIR"))
    MOVE_CAMERA = False
    FIX_COMMAND = False
    VISUAL_PRED = False
    args = get_args()
    play(args)
