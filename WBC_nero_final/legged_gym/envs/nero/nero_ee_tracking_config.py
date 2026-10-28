"""Seven controlled Nero joints with the original A2nero EE command frame."""
import math

from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg as WBC
from .nero_fixed_config import NeroFixedCfg, NeroFixedCfgPPO


# URDF FK at q=[0,-1,0,1.8,0,0,0.3], including the link7 tool offset.
# Coordinates are relative to the WBC spherical center at A2 base +(0.16,0,0.50).
HOME_XYZ_FROM_CENTER = (0.091498974344, -0.000054198929, 0.716365196962)
HOME_SPHERE = (0.722184990609, 1.443757450432, -0.000592344591)


class NeroEETrackingCfg(NeroFixedCfg):
    class init_state(NeroFixedCfg.init_state):
        # A2 base is at z=0.5; the Nero flange is at +(0.16,0,0.142).
        pos = [0.16, 0.0, 0.642]
        default_joint_angles = {k: WBC.init_state.default_joint_angles[k]
                                for k in (*[f'nero_joint{i}' for i in range(1, 8)],
                                          'gripper_joint1', 'gripper_joint2')}

    class control(NeroFixedCfg.control):
        action_mode = 'absolute'
        stiffness = {f'nero_joint{i}': (40.0 if i <= 2 else 20.0) for i in range(1, 8)}
        damping = {f'nero_joint{i}': 1.0 for i in range(1, 8)}
        action_scale = WBC.control.action_scale

    class goal_ee(NeroFixedCfg.goal_ee):
        traj_time = WBC.goal_ee.traj_time
        hold_time = WBC.goal_ee.hold_time
        center_below_mount_m = 0.142
        home_xyz_from_center = HOME_XYZ_FROM_CENTER
        collision_upper_limits = WBC.goal_ee.collision_upper_limits
        collision_lower_limits = WBC.goal_ee.collision_lower_limits
        underground_limit = WBC.goal_ee.underground_limit

        class ranges(WBC.goal_ee.ranges):
            init_pos_start = list(HOME_SPHERE)
            init_pos_end = list(HOME_SPHERE)
            pos_l = [0.4, 0.7]
            pos_p = [-2 * math.pi / 5, 2 * math.pi / 5]
            pos_y = [-3 * math.pi / 5, 3 * math.pi / 5]
            orn_r = [0.0, 0.0]
            orn_p = [0.0, 0.0]
            orn_y = [0.0, 0.0]

    class actuator(NeroFixedCfg.actuator):
        enabled = False
        force_reference_limits = True

    class domain_rand(NeroFixedCfg.domain_rand):
        randomize_motor = False
        randomize_actuator = False
        randomize_link_mass = False
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.12]

    class noise(NeroFixedCfg.noise):
        noise_level = 0.7
        class noise_scales(NeroFixedCfg.noise.noise_scales):
            dof_pos = WBC.noise.noise_scales.dof_pos
            dof_vel = WBC.noise.noise_scales.dof_vel

    class rewards:
        tracking_ee_sigma = WBC.rewards.tracking_ee_sigma
        tracking_ee_orn_sigma = WBC.rewards.tracking_ee_orn_sigma
        ee_tracking_direction_eps = WBC.rewards.ee_tracking_direction_eps
        ee_normal_acc_sigma = WBC.rewards.ee_normal_acc_sigma
        ee_tangent_jerk_sigma = WBC.rewards.ee_tangent_jerk_sigma
        ee_acc_penalty_clip = WBC.rewards.ee_acc_penalty_clip

        class scales:
            tracking_ee_world = WBC.rewards.scales.tracking_ee_world
            tracking_ee_orn = WBC.rewards.scales.tracking_ee_orn
            # Freeze the arm-side defaults from WBC_6_9_7_9_4_woimu_2.
            action_rate = -0.02
            action_acc_arm = -0.015
            dof_vel = -4.e-4
            dof_acc = -7.e-7
            dof_pos_limits = WBC.rewards.scales.dof_pos_limits
            ee_normal_acc = WBC.rewards.scales.ee_normal_acc
            ee_tangent_acc_smooth = WBC.rewards.scales.ee_tangent_acc_smooth
            energy_square_arm = -1.e-4
            dof_vel_limits_arm = -0.1
            torque_limits_arm = -0.005
            nero_joint1_pos = WBC.rewards.scales.nero_joint1_pos
            nero_joint2_pos = WBC.rewards.scales.nero_joint2_pos
            nero_joint3_pos = WBC.rewards.scales.nero_joint3_pos
            nero_joint4_pos = WBC.rewards.scales.nero_joint4_pos
            nero_joint5_pos = WBC.rewards.scales.nero_joint5_pos
            nero_joint6_pos = WBC.rewards.scales.nero_joint6_pos
            nero_joint7_pos = WBC.rewards.scales.nero_joint7_pos


class NeroEETrackingCfgPPO(NeroFixedCfgPPO):
    class runner(NeroFixedCfgPPO.runner):
        experiment_name = 'nero_ee_tracking_wbc_7dof'
        run_name = 'wbc_rewards_pd40_20_home_fk'
