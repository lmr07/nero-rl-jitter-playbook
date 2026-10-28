"""Fixed-base Nero: original arm gains, tool frame and spherical commands."""
import math
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg as Original
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class NeroFixedCfg(LeggedRobotCfg):
    class env:
        num_envs = 2048
        num_actions = 7
        num_single_obs = 39
        frame_stack = 16
        num_observations = 39 * 16
        num_pred_obs = 27
        c_frame_stack = 3
        single_num_privileged_obs = 55
        num_privileged_obs = 55 * 3
        episode_length_s = 20.0
        env_spacing = 2.0
        teleop_mode = False
        send_timeouts = True

    class init_state:
        pos = [0.0, 0.0, 0.8]
        # Preserve the original installation yaw relative to the task frame.
        rot = [0.0, 0.0, math.sin(3.141 / 2), math.cos(3.141 / 2)]
        default_joint_angles = {k: v for k, v in Original.init_state.default_joint_angles.items()
                                if k.startswith(('nero_', 'gripper_'))}

    class control(Original.control):
        stiffness = {f'nero_joint{i}': 48.0 for i in range(1, 8)}
        damping = {f'nero_joint{i}': 1.5 for i in range(1, 8)}
        action_mode = "incremental_v4"
        pd_backend = "physx_position"
        target_delta_limit = 0.012  # rad / 20 ms = 0.6 rad/s nominal target speed
        decimation = 4  # 4 * 5 ms = 20 ms actor tick

    class asset(Original.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a2nero/urdf/nero-fixed.urdf'
        name = 'nero_fixed'
        fix_base_link = True
        default_dof_drive_mode = 1  # gymapi.DOF_MODE_POS
        # Keep tool frame body available without changing its inertia.
        collapse_fixed_joints = True

    class goal_ee:
        traj_time = Original.goal_ee.traj_time
        hold_time = [4.0, 7.0]  # fixed distribution: more settled-target experience
        num_collision_check_samples = Original.goal_ee.num_collision_check_samples
        orn_error_scale = Original.goal_ee.orn_error_scale
        # Coordinates in a task frame centered at the Nero mounting flange,
        # with axes parallel to the original A2 body at zero yaw.
        class ranges(Original.goal_ee.ranges):
            init_pos_start = [0.54, 0.15, 0.0]
            init_pos_end = [0.54, 0.0, 0.0]
            pos_l = [0.35, 0.85]
            pos_p = [-0.15, 1.40]
            pos_y = [-1.60, 1.60]
        collision_upper_limits = [0.15, 0.12, 0.05]
        collision_lower_limits = [-0.15, -0.12, -0.8]
        underground_limit = -0.70

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

    class actuator:
        enabled = True
        # User's reference limits, inferred from manufacturer material; not measured.
        limits_source = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a2nero/urdf/a2-nero.urdf'
        torque_limits = [24.0, 24.0, 16.0, 16.0, 8.0, 8.0, 8.0]
        velocity_limits = [3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
        # Fixed distributions throughout training; samples drawn each episode.
        kp_scale = [0.85, 1.15]
        kd_scale = [0.80, 1.20]
        torque_limit_scale = [0.80, 1.00]
        velocity_limit_scale = [0.85, 1.00]
        # Native drives own torque generation: no external torque filter/taper.
        torque_lag_s = [0.0, 0.0]
        coulomb_friction_nm = [0.0, 0.0]  # legacy explicit-PD path only
        native_joint_friction = [0.0, 0.05]  # PhysX coefficient, not Nm; assumed DR
        viscous_friction = [0.0, 0.05]
        friction_smoothing_velocity = 0.10  # avoid a stiff near-zero friction slope
        speed_taper_fraction = 0.20
        tx_period_s = [0.020, 0.02025]
        feedback_period_s = [0.005, 0.005]
        joint_frame_spacing_s = 0.00025
        drive_delay_s = [0.0, 0.002]
        feedback_transport_s = [0.0, 0.002]
        position_bias_rad = [-0.001, 0.001]
        position_quantum = 0.001
        velocity_quantum = 0.001
        max_target_step = 0.02
        feedback_envelope = 0.20
        takeover_s = 3.0

    class domain_rand:
        # Separate P/D variation replaces a shared motor-strength multiplier.
        randomize_motor = False
        arm_motor_strength_range = [1.0, 1.0]
        randomize_actuator = True
        randomize_link_mass = True
        link_mass_scale_range = [0.95, 1.05]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.12]

    class noise(LeggedRobotCfg.noise):
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            # In addition to encoder quantization and episode-constant bias.
            dof_pos = 0.001
            dof_vel = 0.02

    class rewards:
        tracking_ee_sigma = 0.40
        tracking_ee_orn_sigma = 0.25
        hold_error_radius = 0.10
        hold_velocity_floor = 0.25  # holding penalty remains active away from goal
        class scales:
            tracking_ee_world = 4.0
            # Dense guidance from the folded reset pose, where the exponential
            # tracking reward alone has very little gradient.
            ee_position_error = -2.0
            tracking_ee_orn = 2.0
            action_rate = -0.01
            candidate_rate = -0.01
            adapter_gap = -0.04
            candidate_acceleration = -0.002
            hold_dof_vel = -0.30
            target_rate = -0.01  # final delta / target_delta_limit
            target_acceleration = -0.05  # final second difference / target_delta_limit
            velocity_limits = -0.1
            dof_vel = -2.e-4
            dof_acc = -4.5e-7
            torques = -5.e-6
            dof_pos_limits = -10.0

    class viewer:
        pos = [2.0, -2.0, 1.7]
        lookat = [0.3, 0.0, 1.0]


class NeroFixedCfgPPO(A2NEROPosForceRoughCfgPPO):
    class policy(A2NEROPosForceRoughCfgPPO.policy):
        adaptation_num_single_obs = 39
    class runner(A2NEROPosForceRoughCfgPPO.runner):
        experiment_name = 'nero_fixed'
        run_name = 'nero_pd48_1p5_dr_v5_physx_5ms'
        stage3_enabled = False  # no curriculum/gating inherited from A2nero workflows
        max_iterations = 10000
