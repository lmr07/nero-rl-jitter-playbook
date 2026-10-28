"""Stable seven-axis EE tracking; independent from the original WBC-style runs."""
import math

from .nero_ee_tracking_config import NeroEETrackingCfg, NeroEETrackingCfgPPO


class NeroEETrackingStableCfg(NeroEETrackingCfg):
    class domain_rand(NeroEETrackingCfg.domain_rand):
        # The saved home PD equilibrium assumes the nominal URDF mass.
        randomize_gripper_mass = False

    class control(NeroEETrackingCfg.control):
        target_speed_limits_rad_s = (1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0)
        # Current URDF gravity equilibrium at the unchanged WBC home angles.
        # Recomputed and checked against the URDF when the environment starts.
        initial_position_targets = (0.0, -0.8966083540921852, -0.0011076309926653974,
                                    1.629563833607229, 0.0005273786979047471,
                                    -0.00001756119250868184, 0.27841557917203463)

    class goal_ee(NeroEETrackingCfg.goal_ee):
        class ranges(NeroEETrackingCfg.goal_ee.ranges):
            # Initial joint angles stay unchanged; ramp their measured tool
            # pitch to the horizontal goal during the first 1--3 s segment.
            init_orn_start = [0.0, -0.4707963, 0.0]
            init_orn_end = [0.0, 0.0, 0.0]
            pos_p = [-math.pi / 5, 2 * math.pi / 5]

    class rewards(NeroEETrackingCfg.rewards):
        request_speed_norm_rad_s = 1.5
        request_gap_norm_rad = 0.04
        hold_target_speed_norm_rad_s = 1.5
        hold_acceleration_norm_rad_s2 = 5.0
        hold_anchor_norm_rad = 0.05
        hold_anchor_pos_error_m = 0.03
        hold_anchor_orn_error_rad = 0.10
        hold_anchor_joint_speed_rad_s = 0.08
        hold_anchor_settle_s = 0.30
        hold_anchor_release_pos_error_m = 0.10
        hold_anchor_release_s = 0.50

        class scales(NeroEETrackingCfg.rewards.scales):
            dof_vel = -0.004
            pre_slew_request_rate = -0.10
            pre_slew_request_gap = -0.10
            hold_joint_velocity = -0.05
            hold_joint_acceleration = -0.01
            hold_target_rate = -0.10
            hold_joint_anchor = -0.20


class NeroEETrackingStableCfgPPO(NeroEETrackingCfgPPO):
    class runner(NeroEETrackingCfgPPO.runner):
        experiment_name = 'nero_ee_tracking_stable_v2'
        run_name = 'pitch_ramp_slew_hold_gravity'
