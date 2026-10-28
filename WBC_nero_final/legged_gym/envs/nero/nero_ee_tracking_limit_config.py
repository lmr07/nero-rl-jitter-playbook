"""Independent limit-aware continuation of stable seven-axis EE tracking."""
from .nero_ee_tracking_stable_config import NeroEETrackingStableCfg, NeroEETrackingStableCfgPPO


class NeroEETrackingLimitCfg(NeroEETrackingStableCfg):
    class control(NeroEETrackingStableCfg.control):
        # The physical home trial approached the J7 upper bound while the
        # endpoint stayed fixed. Keep extra braking room at that joint.
        target_position_margin_rad = (0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.30)

    class rewards(NeroEETrackingStableCfg.rewards):
        limit_prediction_s = 0.15
        limit_reward_margin_rad = 0.22
        limit_penalty_norm_rad = 0.08
        limit_stop_margin_rad = 0.08
        limit_stop_speed_rad_s = 0.15
        target_margin_penalty_norm_rad = 0.05

        class scales(NeroEETrackingStableCfg.rewards.scales):
            dof_pos_limits = -25.0
            predictive_joint_limits = -5.0
            requested_target_margin = -0.20
            safety_termination = -20.0


class NeroEETrackingLimitCfgPPO(NeroEETrackingStableCfgPPO):
    class runner(NeroEETrackingStableCfgPPO.runner):
        experiment_name = 'nero_ee_tracking_limit_v3'
        run_name = 'predicted_limits_j7_margin'
