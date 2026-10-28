"""Independent single-axis absolute-position policy; fixed distributions, no stages."""
from .nero_fixed_config import NeroFixedCfg, NeroFixedCfgPPO
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg as A2Cfg
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfgPPO as A2PPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from nero_joint_contract import A2NERO_STAGE1_MODE, COMMAND_SPEED_FRACTION


class NeroJointCfg(LeggedRobotCfg):
    init_state = NeroFixedCfg.init_state
    asset = NeroFixedCfg.asset
    sim = NeroFixedCfg.sim
    viewer = NeroFixedCfg.viewer

    class domain_rand:
        # A2nero Stage-1 arm-side distributions, sampled once per actor.
        randomize_motor = True
        arm_motor_strength_range = [0.92, 1.08]
        randomize_link_mass = False  # A2nero Stage-1 varies leg mass, not Nero link mass.
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.12]
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class noise:
        add_noise = True
        noise_level = 0.7
        class noise_scales:
            dof_pos = A2Cfg.noise.noise_scales.dof_pos
            dof_vel = A2Cfg.noise.noise_scales.dof_vel

    class env(NeroFixedCfg.env):
        num_envs = A2Cfg.env.num_envs
        num_actions = 1
        num_single_obs = 12  # q, dq, ref, ref_vel, previous action, seven locked-pose coordinates
        frame_stack = 16
        num_observations = 12 * 16
        num_privileged_obs = 18  # true frame, command, queue, motor strength and payload
        num_pred_obs = 3
        episode_length_s = A2Cfg.env.episode_length_s
        joint = 1
        pose_seed = 42
        pose_bank_size = 128
        hold_time = A2Cfg.goal_ee.hold_time

    class rewards:
        # A2nero Stage-1 arm regularizers; tracking remains a one-DOF task.
        tracking_sigma = A2Cfg.rewards.tracking_ee_sigma
        tracking = A2Cfg.rewards.scales.tracking_ee_world
        action_rate = A2Cfg.rewards.scales.action_rate_arm
        dof_vel = A2Cfg.rewards.scales.dof_vel_arm
        dof_acc = A2Cfg.rewards.scales.dof_acc_arm
        dof_pos_limits = A2Cfg.rewards.scales.dof_pos_limits
        soft_dof_pos_limit = A2Cfg.rewards.soft_dof_pos_limit
        # Goal-space equivalent of the fixed-base arm's dense tracking guidance.
        position_error = -2.0
        smooth_hold = False

    class control:
        kp = 10.
        kd = 0.8
        decimation = 4
        action_mode = A2NERO_STAGE1_MODE
        action_scale = A2Cfg.control.action_scale
        action_clip = A2Cfg.normalization.clip_actions
        command_speed_fraction = COMMAND_SPEED_FRACTION
        goal_range_fraction = 1.0
        pd_backend = "physx_position"
        # A2nero Stage-1 has no added plant command delay.
        delay_s = 0.0


class NeroJointCfgPPO(NeroFixedCfgPPO):
    class policy(NeroFixedCfgPPO.policy):
        adaptation_num_single_obs = 12
        adaptation_labels = ["joint_state"]
        adaptation_dims = [3]
        adaptation_weights = [1.]
        actor_hidden_dims = A2PPO.policy.actor_hidden_dims
        critic_hidden_dims = A2PPO.policy.critic_hidden_dims
        latent_dim = A2PPO.policy.latent_dim
        aux_future_horizon = 0
        init_noise_std = A2PPO.policy.init_noise_std if hasattr(A2PPO.policy, "init_noise_std") else 1.0

    class runner(NeroFixedCfgPPO.runner):
        experiment_name = "nero_joint_a2stage1"
        run_name = "a2stage1_arm_pd10_0p8"
        max_iterations = 5000
