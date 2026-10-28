"""Seven-axis fixed-base task with the A2Nero spherical task frame."""
import math
from .nero_fixed_config import NeroFixedCfg, NeroFixedCfgPPO
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg as WBC


class NeroSevenCycleCfg(NeroFixedCfg):
    class env(NeroFixedCfg.env):
        # 3 s at home, then four 3 s travel + 3 s hold legs.
        episode_length_s = 27.0

    class init_state(NeroFixedCfg.init_state):
        # WBC_6_9_7_9_4 arm home. Zero actor action means this joint posture.
        default_joint_angles = {**{f'nero_joint{i}': WBC.init_state.default_joint_angles[f'nero_joint{i}']
                                    for i in range(1, 8)},
                                'gripper_joint1': 0.05, 'gripper_joint2': -0.05}

    class control(NeroFixedCfg.control):
        behavior_version = 2  # old saved runs without this field replay as v1
        stiffness = {f'nero_joint{i}': 20.0 for i in range(1, 8)}
        damping = {f'nero_joint{i}': 1.0 for i in range(1, 8)}
        action_mode = 'absolute'
        pd_backend = 'physx_position'
        decimation = 4
        action_scale = WBC.control.action_scale
        # Half the reference joint speed limits, applied to the sent position
        # target at each 20 ms policy tick in both training and visualization.
        target_speed_limits_rad_s = (1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0)

    class goal_ee(NeroFixedCfg.goal_ee):
        traj_time = [3.0, 3.0]
        hold_time = [3.0, 3.0]
        # Old WBC center is A2 base XY + [0.16, 0, 0.50]. Its Nero flange is
        # at A2 base + [0.16, 0, 0.142], hence the -0.142 m mount-relative Z.
        center_below_mount_m = 0.142
        fixed_waypoints_spherical = ((0.5, 0.0, math.pi / 4),
                                     (0.6, math.pi / 3, -math.pi / 4))
        # Position sampling is deterministic: home -> P1 -> home -> P2 -> home.
        # Parallel training environments start at coherent, different cycle
        # phases; a one-environment replay always starts at home.
        diversify_initial_phase = True
        # Position-only URDF FK solutions, >=0.25 rad from hard joint limits.
        # Used only to initialize P1/P2 hold phases, never as real-arm commands.
        phase_start_joint_angles = ((-1.6670, -1.4900, 2.3211, 1.4074, 0.1337, -0.0421, 0.1233),
                                    (0.0752, -0.5872, -0.6270, 1.8900, -0.0883, 0.0168, 0.5396))
        # The P1 J2 native drive sags to its lower stop if initialized with
        # q_des=q. A fixed-target PhysX sweep at Kp=20/Kd=1 found that -0.80
        # rad command keeps the P1 tool within ~5 cm after settling. These are
        # simulator reset commands only; the actor still controls all targets.
        phase_start_target_joint_angles = ((-1.6670, -0.8000, 2.3211, 1.4074, 0.1337, -0.0421, 0.1233),
                                           (0.0752, -0.5872, -0.6270, 1.8900, -0.0883, 0.0168, 0.5396))

    class actuator(NeroFixedCfg.actuator):
        enabled = False  # no deployment takeover or extra feedback lag
        force_reference_limits = True  # avoid the official URDF's inflated caps
        # 2026-10-23 command-only MIT fit, in joint order. The effective delay
        # includes command transport, drive response and feedback alignment.
        identified_effective_delay_ms = (5.5, 7.0, 8.0, 7.5, 9.0, 9.5, 10.0)
        identification_sha256 = 'b7f0565b552f6091c926cab34c28f1d496c82ee1f7739037d05af973729642fe'
        identification_source = 'nero_identification/datasets/nero_eq20/run_20261023_103746/gain_delay_fit_with_frequency.json'

    class domain_rand(NeroFixedCfg.domain_rand):
        randomize_motor = False  # Kp=20, Kd=1 remain exact
        randomize_actuator = False
        randomize_link_mass = False
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.12]  # WBC DR Stage-1

    class noise(NeroFixedCfg.noise):
        add_noise = True
        noise_level = 0.7
        class noise_scales(WBC.noise.noise_scales):
            # Fixed provisional observation noise: 2.1 mrad position and
            # 0.07 rad/s velocity, instead of WBC's 1.05 rad/s velocity noise.
            dof_pos = 0.003
            dof_vel = 0.10

    class rewards:
        # WBC_6_9_7_9_4 used 1.0 m; this checkout's later A2 config uses 0.4.
        tracking_ee_sigma = 1.0
        ee_tracking_direction_eps = WBC.rewards.ee_tracking_direction_eps
        ee_normal_acc_sigma = WBC.rewards.ee_normal_acc_sigma
        ee_tangent_jerk_sigma = WBC.rewards.ee_tangent_jerk_sigma
        ee_acc_penalty_clip = WBC.rewards.ee_acc_penalty_clip
        hold_error_radius = 0.10
        hold_velocity_floor = 0.10
        target_rate_norm_rad = 0.03  # 1.5 rad/s over the 20 ms policy interval
        class scales:
            tracking_ee_world = WBC.rewards.scales.tracking_ee_world
            ee_position_error = -2.0
            action_rate = WBC.rewards.scales.action_rate_arm
            dof_vel = WBC.rewards.scales.dof_vel_arm
            dof_acc = WBC.rewards.scales.dof_acc_arm
            dof_pos_limits = WBC.rewards.scales.dof_pos_limits
            ee_normal_acc = WBC.rewards.scales.ee_normal_acc
            ee_tangent_acc_smooth = WBC.rewards.scales.ee_tangent_acc_smooth
            hold_dof_vel = -0.10
            hold_target_rate = -0.20
            hold_target_reversal = -0.10
            home_posture = -0.50


class NeroSevenCycleCfgPPO(NeroFixedCfgPPO):
    class algorithm(NeroFixedCfgPPO.algorithm):
        learning_rate = 2.0e-4
        learning_rate_upper_bound = 3.0e-4
        learning_rate_lower_bound = 1.0e-5
        entropy_coef = 0.003
        num_learning_epochs = 4

    class runner(NeroFixedCfgPPO.runner):
        experiment_name = 'nero_seven_cycle_pd20_1'
        run_name = 'wbc_home_phase_diverse_hold_stable_v2'
        num_steps_per_env = 48
        max_iterations = 20000  # 48 steps/update keeps the old 40k x 24 sample budget
