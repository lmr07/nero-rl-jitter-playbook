"""Aggressive pre-slew and hold stabilization training, independent of v2 runs."""
from .nero_seven_cycle_config import NeroSevenCycleCfg, NeroSevenCycleCfgPPO


class NeroSevenCycleSmoothCfg(NeroSevenCycleCfg):
    class control(NeroSevenCycleCfg.control):
        behavior_version = 3
        stiffness = {f'nero_joint{joint}': (40.0 if joint <= 2 else 20.0) for joint in range(1, 8)}
        damping = {f'nero_joint{joint}': 1.0 for joint in range(1, 8)}
        request_joint_margin_rad = .10

    class goal_ee(NeroSevenCycleCfg.goal_ee):
        interpolation = 'quintic'

    class rewards(NeroSevenCycleCfg.rewards):
        tracking_ee_sigma = .40
        request_gap_norm_rad = .04
        request_speed_norm_rad_s = 1.5
        acceleration_norm_rad_s2 = 40.0
        jerk_norm_rad_s3 = 1000.0
        hold_smooth_multiplier = 6.0
        hold_ramp_s = .50
        hold_joint_speed_norm_rad_s = .10
        hold_joint_acceleration_norm_rad_s2 = 5.0
        hold_anchor_error_m = .04
        hold_anchor_speed_rad_s = .08
        hold_anchor_settle_s = .30
        hold_anchor_norm_rad = .015
        hold_anchor_release_error_m = .10
        hold_anchor_release_s = .50
        limit_prediction_s = .15
        limit_margin_rad = .12
        limit_penalty_norm_rad = .08
        stop_margin_rad = .08
        stop_speed_rad_s = .15

        class scales(NeroSevenCycleCfg.rewards.scales):
            tracking_ee_world = 8.0
            ee_position_error = -8.0
            action_rate = -.03
            dof_vel = -.02
            dof_acc = -5.e-6
            dof_pos_limits = -25.0
            ee_normal_acc = -.20
            ee_tangent_acc_smooth = -.10
            hold_dof_vel = 0.0
            hold_target_rate = 0.0
            hold_target_reversal = 0.0
            home_posture = -1.0
            request_gap = -2.0
            request_rate = -.40
            request_acceleration = -.75
            request_jerk = -.03
            sent_target_rate = -.40
            sent_target_acceleration = -.75
            sent_target_jerk = -.05
            target_reversal = -1.0
            hold_joint_velocity = -3.0
            hold_joint_acceleration = -.50
            hold_joint_anchor = -1.0
            predictive_joint_limits = -10.0
            safety_termination = -20.0


class NeroSevenCycleSmoothCfgPPO(NeroSevenCycleCfgPPO):
    class policy(NeroSevenCycleCfgPPO.policy):
        init_noise_std = .01

    class algorithm(NeroSevenCycleCfgPPO.algorithm):
        learning_rate = 1.e-4
        learning_rate_upper_bound = 1.5e-4
        entropy_coef = .0001

    class runner(NeroSevenCycleCfgPPO.runner):
        policy_class_name = 'NeroSmoothActorCritic'
        experiment_name = 'nero_seven_cycle_smooth_v3'
        run_name = 'j1j2_pd40_1_preslew_hold'
        max_iterations = 10000
