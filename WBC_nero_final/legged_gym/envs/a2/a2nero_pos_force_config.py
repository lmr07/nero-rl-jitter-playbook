from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import os
import json
from pathlib import Path


def load_model39000_identified_profile():
    """Load the ABI-preserving identified continuation profile for model39000."""
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "training_profiles"
        / "model39000_woimu_training_identification.json"
    )
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    if profile.get("profile") != "model39000_woimu_identified_stage1_stage2_v1":
        raise RuntimeError(f"Unexpected model39000 profile in {profile_path}")
    if int(profile.get("policy_hz", 0)) != 50:
        raise RuntimeError("The identified dynamics require the checkpoint's 50 Hz policy rate")
    return profile


def _apply_identified_values(env_cfg, profile, identified_stage):
    data = profile["stage1" if identified_stage == 1 else "stage2"]
    env_cfg.stage3.enabled = True
    env_cfg.stage3.identified_stage = identified_stage
    env_cfg.stage3.identified_profile = data
    env_cfg.stage3.curriculum_total_iterations = data["additional_iterations"]
    rigid = data["rigid_body"]
    env_cfg.stage3.friction_range = rigid["friction"]
    env_cfg.stage3.restitution_range = rigid["restitution"]
    env_cfg.stage3.added_base_mass_range = rigid["added_base_mass_kg"]
    env_cfg.stage3.base_com_range_x = rigid["base_com_x"]
    env_cfg.stage3.base_com_range_y = rigid["base_com_y"]
    env_cfg.stage3.base_com_range_z = rigid["base_com_z"]
    env_cfg.stage3.leg_mass_scale_range = rigid["leg_mass_relative_delta"]
    env_cfg.stage3.gripper_payload_range = rigid["gripper_payload_kg"]
    actuator = data["actuator"]
    env_cfg.stage3.a2_hip_torque_scale_range = actuator["motor_strength"]["a2_hip"]
    env_cfg.stage3.a2_thigh_torque_scale_range = actuator["motor_strength"]["a2_thigh"]
    env_cfg.stage3.a2_calf_torque_scale_range = actuator["motor_strength"]["a2_calf"]
    env_cfg.stage3.nero_torque_scale_range = actuator["motor_strength"]["nero"]
    env_cfg.stage3.a2_kp_scale_range = actuator["kp"]["a2"]
    env_cfg.stage3.a2_kd_scale_range = actuator["kd"]["a2"]
    env_cfg.stage3.nero_kp_scale_range = actuator["kp"]["nero"]
    env_cfg.stage3.nero_kd_scale_range = actuator["kd"]["nero"]
    if identified_stage == 2:
        # Stage 2 uses the per-joint-group delay model, not the legacy scalar queue.
        env_cfg.stage3.a2_delay_values = [0]
        env_cfg.stage3.a2_delay_probabilities = [1.0]
        env_cfg.stage3.nero_delay_values = [0]
        env_cfg.stage3.nero_delay_probabilities = [1.0]
    else:
        delay = actuator["delay"]
        env_cfg.stage3.a2_delay_values = delay["a2"]["values"]
        env_cfg.stage3.a2_delay_probabilities = delay["a2"]["probabilities"]
        env_cfg.stage3.nero_delay_values = delay["nero"]["values"]
        env_cfg.stage3.nero_delay_probabilities = delay["nero"]["probabilities"]
    # Preserve the checkpoint reward contract at continuation iteration zero.
    # The report values are curriculum targets, not values to switch on before
    # the first rollout.
    env_cfg.stage3.source_max_contact_force = float(env_cfg.rewards.max_contact_force)
    env_cfg.stage3.final_max_contact_force = float(
        data["contact"]["max_contact_force_n"]
    )
    env_cfg.stage3.identified_reward_start_scales = {
        name: float(getattr(env_cfg.rewards.scales, name, 0.0))
        for name in data["rewards"]
    }
    env_cfg.stage3.identified_reward_target_scales = {
        name: float(scale) for name, scale in data["rewards"].items()
    }
    env_cfg.rewards.max_contact_force = env_cfg.stage3.final_max_contact_force
    for name, scale in data["rewards"].items():
        setattr(env_cfg.rewards.scales, name, scale)

class A2NEROPosForceRoughCfg( LeggedRobotCfg ):

    class goal_ee:
        
        num_commands = 3
        traj_time = [1, 3]
        # Keep the arm static for several gait cycles so the locomotion policy
        # learns pose-dependent compensation for the shifted whole-body COM.
        hold_time = [2.0, 5.0]
        collision_upper_limits = [0.20, 0.16, -0.05]
        collision_lower_limits = [-0.5, -0.16, -0.5]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
        
        class sphere_center:
            x_offset = 0.16 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.5 # Relative to terrain
        
        class ranges:
            init_pos_start = [0.46, np.pi/4, 0]
            # init_pos_end = [0.66, np.pi/6, 0]
            # init_pos_end = [0.66, -1 * np.pi / 3, 0]
            init_pos_end = [0.46, 0, 0]
            init_orn_start = [0.0, 0.0, 0.0]
            init_orn_end = [0.0, 0.0, 0.0]
            pos_l = [0.35, 0.85]
            pos_p = [-2 * np.pi / 5, 2 * np.pi / 5]
            pos_y = [-3 * np.pi/ 5, 3 * np.pi /5]

            # The default task keeps the gripper centerline horizontal.  The
            # orientation reward below constrains all three goal-frame angles
            # to this zero-RPY target during training.
            orn_r = [0.0, 0.0]
            orn_p = [0.0, 0.0]
            orn_y = [0.0, 0.0]
            
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.15,   # [rad]
            'FL_thigh_joint': 0.67,     # [rad]
            'FL_calf_joint': -1.32,   # [rad]

            'RL_hip_joint': 0.15,   # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'RL_calf_joint': -1.32,    # [rad]

            'FR_hip_joint': -0.15,  # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'FR_calf_joint': -1.32,  # [rad]

            'RR_hip_joint': -0.15,   # [rad]
            'RR_thigh_joint': 0.67,   # [rad]
            'RR_calf_joint': -1.32,    # [rad]

            'nero_joint1': 0.0,
            'nero_joint2': -1.0,
            'nero_joint3': 0.0, # -0.63,
            'nero_joint4': 1.8, # -0.84,
            'nero_joint5': 0.0,
            'nero_joint6': 0.0, # 0.0,
            'nero_joint7': 0.3,
            # One logical opening coordinate is mapped to these two physical
            # finger joints as +opening and -opening by the environment.
            'gripper_joint1': 0.05,
            'gripper_joint2': -0.05,
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.0

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.2, 2.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_leg_mass = True
        leg_mass_scale_range = [-0.20, 0.20]
        randomize_motor = True
        leg_motor_strength_range = [0.85, 1.15]
        arm_motor_strength_range = [0.85, 1.15]
        
        randomize_rigids_after_start = False # True
        randomize_restitution = False # True
        restitution_range = [0.0, 1.0]

        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.2]
        # randomize_arm_friction = True
        # arm_friction_range = [0.0, 0.2]
        # randomize_arm_ema = True
        # arm_ema_range = [0.05, 0.25]

        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.8


    class env( LeggedRobotCfg.env ):

        # The real gripper has one actuator/independent coordinate. The URDF
        # keeps two physical finger DOFs so both fingers can move symmetrically.
        num_gripper_joints = 1
        num_gripper_dofs = 2
        num_actions = 19
        num_torques = 19
        frame_stack = 16
        c_frame_stack = 3
        # 85-D policy observation: the former 18-D base/EE/relative IMU tail
        # is intentionally omitted for the no-end-effector-IMU ablation.
        num_single_obs = 85
        num_pred_obs = 27
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 147
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = False
        frequencies = 1.0

        action_delay = 3 # Not used, assigned in code
        teleop_mode = False
        teleop_ee_pos_step = 0.05
        teleop_ee_orn_step = 5 * np.pi / 180

    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            imu_gyro = 0.05
            imu_acc = 0.3
            relative_imu_gyro = 0.07
            relative_imu_acc = 0.5

    class stage3:
        """Minimal continuation-only sim-to-real settings.

        These values are kept outside the legacy observation/network config so
        continuation training cannot change the model_39000 85x16/19D ABI.
        """
        enabled = False
        identified_stage = 0
        identified_profile = None
        # The source DR Stage-1 sampled friction and rigid-body properties once
        # per actor. Rebuilding inertia during episode reset changes its physics
        # contract even when the continuation randomization factor is zero.
        fixed_rigid_props_at_creation = False
        curriculum_total_iterations = 8000
        curriculum_progress = 0.0
        source_max_contact_force = 200.0
        final_max_contact_force = 200.0

        a2_delay_values = [0, 1]
        a2_delay_probabilities = [0.85, 0.15]
        nero_delay_values = [0, 1]
        nero_delay_probabilities = [0.30, 0.70]

        nero_max_step_start = 0.02
        nero_max_step_final = 0.005
        a2_stationary_feedback_envelope_start = 25.0
        a2_stationary_feedback_envelope_final = 0.30
        a2_active_feedback_envelope_start = 25.0
        a2_active_feedback_envelope_final = 0.40
        nero_feedback_envelope_start = 25.0
        nero_feedback_envelope_final = 0.10

        friction_range = [0.7, 1.3]
        added_base_mass_range = [-0.5, 2.0]
        base_com_range_x = [-0.015, 0.015]
        base_com_range_y = [-0.015, 0.015]
        base_com_range_z = [-0.01, 0.01]
        leg_mass_scale_range = [-0.05, 0.05]
        gripper_payload_range = [0.0, 0.12]

        a2_torque_scale_range = [0.95, 1.05]
        nero_torque_scale_range = [0.94, 1.06]
        a2_kp_scale_range = [0.95, 1.05]
        a2_kd_scale_range = [0.95, 1.05]
        nero_kp_scale_range = [0.95, 1.05]
        nero_kd_scale_range = [0.90, 1.10]

        base_gyro_bias_range = [-0.01, 0.01]
        base_acc_bias_range = [-0.10, 0.10]
        ee_gyro_bias_range = [-0.015, 0.015]
        ee_acc_bias_range = [-0.15, 0.15]

        evaluation_interval = 200
        evaluation_seed = 17800
        baseline_ratio_floor = 0.90

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 9 # lin_vel_x, lin_vel_y, ang_vel_yaw, ee position, ee orientation
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.3, 0.3] # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
        ang_vel_yaw_clip = 0.2
        ang_vel_pitch_clip = 0.5
        # Commands below this threshold are treated as standing by gait rewards.
        # Keep the deadband small enough that low-speed backward commands train
        # as walking commands rather than conflicting with standing behavior.
        lin_vel_x_clip = 0.05
        lin_vel_y_clip = 0.05


        zero_vel_cmd_prob = 0.1
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.05] # [0.04, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 100, 'thigh': 100, 'calf': 150, 'nero_joint1': 48, 'nero_joint2': 48, 'nero_joint3': 48, 'nero_joint4': 48, 'nero_joint5': 48, 'nero_joint6': 48, 'nero_joint7': 48, 'gripper_joint1': 64, 'gripper_joint2': 64}  # [N*m/rad]
        damping = {'hip': 4, 'thigh': 4, 'calf': 6, 'nero_joint1': 1.5, 'nero_joint2': 1.5, 'nero_joint3': 1.5, 'nero_joint4': 1.5, 'nero_joint5': 1.5, 'nero_joint6': 1.5, 'nero_joint7': 1.5, 'gripper_joint1': 1.5, 'gripper_joint2': 1.5}     # [N*m*s/rad]
        gripper_stiffness = 64.0
        gripper_damping = 1.5
        gripper_opening_range = [0.0, 0.05]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a2nero/urdf/a2-nero.urdf'
        name = "a2nero"
        foot_name = "foot"
        thigh_name = "thigh"
        gripper_name = "gripper_link2"
        gripper_dof_names = ["gripper_joint1", "gripper_joint2"]
        ee_position_body_name = "nero_link7"
        ee_position_body_offset = [0.17803, 0.0, -0.0235]
        ee_orientation_body_name = "nero_link7"
        # Align goal-frame +X with the gripper centerline (nero_link7 +X), so
        # zero goal-frame pitch means that the gripper centerline is horizontal.
        ee_orientation_body_to_goal_frame_quat = [0.70710678, 0.0, 0.0, 0.70710678]
        ee_imu_body_name = "imu"
        # Sensor sampling point relative to the IMU rigid body's center of mass,
        # expressed in the IMU body frame: 6.2 mm along local +Z.
        ee_imu_com_to_sensor_offset = [0.0, 0.0, 0.0062]
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        gait_vel_sigma = 2.0
        gait_force_sigma = 2.0
        kappa_gait_probs = 0.07

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        # Sharpen base velocity tracking to penalize pose-dependent steady-state
        # drift, especially motion opposite to a low-speed command.
        tracking_sigma = 0.12 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 0.4
        tracking_ee_orn_sigma = 0.25
        ee_tracking_direction_eps = 1e-4
        ee_normal_acc_sigma = 5.0
        ee_tangent_jerk_sigma = 80.0
        ee_acc_penalty_clip = 100.0
        # Normalize and clip finite-difference base acceleration penalties so
        # occasional impact spikes cannot dominate the PPO update.
        base_lin_acc_z_norm = 10.0       # [m/s^2]
        base_ang_acc_yaw_norm = 10.0     # [rad/s^2]
        base_acc_penalty_clip = 10.0
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        base_height_target = 0.50
        max_contact_force = 150. # forces above this value are penalized

        cycle_time = 0.64
        target_joint_pos_scale = 0.17
        target_joint_pos_thd = 0.5

        class scales:

            feet_contact_number = 2.0

            tracking_lin_vel = 3.0
            tracking_ang_vel = 2.0

            torques = -5.e-6 
            stand_still = 0.5 
            # Reduce competition from the command-independent reference gait so
            # leg actions can compensate for different static arm poses.
            ref_dof_leg = 0.5
            alive = 1.5
            lin_vel_z = -3.0
            orientation = -1.5
            roll = -0.5
            pitch = -0.4
            feet_air_time = 0.3
            feet_height = 0.2
            ang_vel_xy = -0.15
            # Directly suppress high-frequency vertical and yaw oscillations.
            base_lin_acc_z = -0.05
            base_ang_acc_yaw = -0.05
            dof_acc = -7.5e-7
            dof_vel = -8.e-4
            dof_acc_arm = -4.5e-7 
            dof_vel_arm = -2.e-4
            collision = -5. 
            action_rate = -0.04
            action_rate_arm = -0.01 
            dof_pos_limits = -10.0 
            torque_limits = -0.005
            hip_pos = -0.5  
            nero_joint1_pos = -0.02
            nero_joint2_pos = -0.01
            nero_joint3_pos = -0.2
            nero_joint4_pos = -0.02
            nero_joint5_pos = -0.02
            nero_joint6_pos = -0.02
            nero_joint7_pos = -0.02
            gripper_joint1_pos = -0.2
            gripper_joint2_pos = -0.2
            feet_drag = -0.0008 
            feet_contact_forces = -0.004
            base_height = -5.0
            feet_pos_xy = -0.5
            feet_height_high = -25
            
            arm_termination = 0.
            tracking_ee_sphere = 0.
            tracking_ee_world = 4.0
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = 0.
            ee_normal_acc = -0.05
            ee_tangent_acc_smooth = -0.01
            arm_orientation = 0.
            arm_energy_abs_sum = 0.
            tracking_ee_orn = 2.0
            tracking_ee_orn_ry = 0.

class A2NEROPosForceRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # Both the actor and adaptation encoder consume the same 85-D
        # observation, with no base/EE/relative IMU inputs.
        temporal_encoder = "mlp"
        adaptation_num_single_obs = 85
        latent_dim = 32
        critic_use_latent = True
        aux_future_horizon = 3
        adaptation_labels = [
            "ee_pos_base_loss",
            "ee_vel_base_loss",
            "tracking_error_base_loss",
            "future_ee_pos_base_loss",
            "future_tracking_error_base_loss",
        ]
        adaptation_dims = [3, 3, 3, 9, 9]
        adaptation_weights = [1.0, 0.5, 1.0, 0.7, 0.7]
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'a2nero_pos_force'


def apply_a2nero_domain_rand_stage(env_cfg, train_cfg=None, stage=1):
    """Apply two-stage sim-to-real domain randomization for A2NERO training."""
    env_cfg.stage3.enabled = False
    if train_cfg is not None:
        train_cfg.algorithm.entropy_coef = 0.01
        train_cfg.runner.stage3_enabled = False
    if stage == 1:
        env_cfg.noise.add_noise = True
        env_cfg.noise.noise_level = 0.7

        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [0.6, 1.4]

        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.added_mass_range = [-1.0, 3.0]

        env_cfg.domain_rand.randomize_base_com = True
        env_cfg.domain_rand.added_com_range_x = [-0.03, 0.03]
        env_cfg.domain_rand.added_com_range_y = [-0.03, 0.03]
        env_cfg.domain_rand.added_com_range_z = [-0.03, 0.03]

        env_cfg.domain_rand.randomize_leg_mass = True
        env_cfg.domain_rand.leg_mass_scale_range = [-0.05, 0.05]

        env_cfg.domain_rand.randomize_motor = True
        env_cfg.domain_rand.leg_motor_strength_range = [0.90, 1.10]
        env_cfg.domain_rand.arm_motor_strength_range = [0.92, 1.08]

        env_cfg.domain_rand.randomize_gripper_mass = True
        env_cfg.domain_rand.gripper_added_mass_range = [0.0, 0.12]

        env_cfg.domain_rand.push_robots = False
        stage_name = "dr_stage1"
    elif stage == 2:
        env_cfg.noise.add_noise = True
        env_cfg.noise.noise_level = 1.0

        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [0.35, 2.0]

        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.added_mass_range = [-1.5, 5.0]

        env_cfg.domain_rand.randomize_base_com = True
        env_cfg.domain_rand.added_com_range_x = [-0.06, 0.06]
        env_cfg.domain_rand.added_com_range_y = [-0.06, 0.06]
        env_cfg.domain_rand.added_com_range_z = [-0.05, 0.05]

        env_cfg.domain_rand.randomize_leg_mass = True
        env_cfg.domain_rand.leg_mass_scale_range = [-0.10, 0.10]

        env_cfg.domain_rand.randomize_motor = True
        env_cfg.domain_rand.leg_motor_strength_range = [0.85, 1.15]
        env_cfg.domain_rand.arm_motor_strength_range = [0.88, 1.12]

        env_cfg.domain_rand.randomize_gripper_mass = True
        env_cfg.domain_rand.gripper_added_mass_range = [0.0, 0.20]

        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.push_interval_s = 8
        env_cfg.domain_rand.max_push_vel_xy = 0.5
        stage_name = "dr_stage2"
    elif stage == 3:
        # Stage-3 physics starts nominal.  The environment introduces only the
        # conservative ranges above through the explicit progress curriculum.
        env_cfg.stage3.enabled = True
        env_cfg.stage3.fixed_rigid_props_at_creation = True
        env_cfg.stage3.curriculum_progress = 0.0
        env_cfg.noise.add_noise = True
        env_cfg.noise.noise_level = 0.7

        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_leg_mass = False
        env_cfg.domain_rand.randomize_motor = False
        env_cfg.domain_rand.randomize_gripper_mass = False
        env_cfg.domain_rand.push_robots = False
        stage_name = "dr_stage3_minimal_from39000"
    elif stage in (4, 5):
        profile = load_model39000_identified_profile()
        identified_stage = stage - 3
        _apply_identified_values(env_cfg, profile, identified_stage)
        env_cfg.stage3.fixed_rigid_props_at_creation = True
        env_cfg.stage3.curriculum_progress = 0.0
        env_cfg.noise.add_noise = True
        env_cfg.noise.noise_level = 0.7 if stage == 4 else 1.0
        # The explicit identified path owns all randomization. Disable legacy
        # callbacks so no distribution is accidentally applied twice.
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_leg_mass = False
        env_cfg.domain_rand.randomize_motor = False
        env_cfg.domain_rand.randomize_gripper_mass = False
        env_cfg.domain_rand.push_robots = False
        stage_name = f"model39000_woimu_identified_stage{identified_stage}"
    else:
        raise ValueError(f"Unsupported domain randomization stage: {stage}")

    if train_cfg is not None:
        if stage in (3, 4, 5):
            train_cfg.algorithm.entropy_coef = 0.003
            # Preserve the converged Stage1 policy while keeping the PPO and
            # network/checkpoint structures unchanged.
            train_cfg.algorithm.freeze_action_std = True
            train_cfg.algorithm.learning_rate_upper_bound = 2.25e-5
            train_cfg.algorithm.adaptation_module_learning_rate = 1.0e-6
            train_cfg.algorithm.adaptation_policy_gradient_scale = 0.0
            train_cfg.runner.stage3_enabled = True
            train_cfg.runner.stage3_source_iteration = int(os.getenv("A2NERO_SOURCE_ITER", "39000"))
            train_cfg.runner.stage3_additional_iterations = int(os.getenv("A2NERO_ADDITIONAL_ITERATIONS", "8000"))
            train_cfg.algorithm.adaptation_warmup_iterations = max(
                1, int(0.10 * train_cfg.runner.stage3_additional_iterations)
            )
            train_cfg.runner.stage3_source_run = os.getenv(
                "A2NERO_SOURCE_RUN", "Sep12_12-08-00_dr_stage1"
            )
            train_cfg.runner.stage3_source_checkpoint = os.getenv(
                "A2NERO_SOURCE_CHECKPOINT", ""
            )
            train_cfg.runner.stage3_source_sha256 = os.getenv("A2NERO_SOURCE_SHA256", "")
            train_cfg.runner.stage3_evaluation_interval = env_cfg.stage3.evaluation_interval
            train_cfg.runner.stage3_evaluation_seed = env_cfg.stage3.evaluation_seed
            train_cfg.runner.stage3_baseline_ratio_floor = env_cfg.stage3.baseline_ratio_floor
            train_cfg.runner.stage3_action_scale = env_cfg.control.action_scale
            train_cfg.runner.stage3_action_position_rms_warning = 0.05
            train_cfg.runner.stage3_action_drift_reference_observations = 256
            train_cfg.runner.stage3_cap_learning_rate_to_checkpoint = True
            train_cfg.runner.stage3_hard_stop_on_failure = True
            if stage in (4, 5):
                profile = load_model39000_identified_profile()
                data = profile["stage1" if stage == 4 else "stage2"]
                ppo = data["ppo"]
                train_cfg.algorithm.entropy_coef = ppo["entropy_coef"]
                train_cfg.algorithm.learning_rate_upper_bound = ppo["actor_learning_rate_cap"]
                train_cfg.algorithm.learning_rate_lower_bound = ppo.get(
                    "actor_learning_rate_floor", 1.0e-8
                )
                train_cfg.algorithm.adaptation_module_learning_rate = ppo["adaptation_learning_rate"]
                train_cfg.algorithm.freeze_action_std = bool(ppo.get("freeze_action_std", False))
                train_cfg.algorithm.actor_warmup_iterations = max(
                    1,
                    int(
                        train_cfg.runner.stage3_additional_iterations
                        * ppo.get("actor_warmup_fraction", 0.0)
                    ),
                )
                train_cfg.algorithm.adaptation_optimizer_modules_only = bool(
                    ppo.get("adaptation_optimizer_modules_only", False)
                )
                train_cfg.algorithm.source_action_anchor_coef = float(
                    ppo.get("source_action_anchor_coef", 0.0)
                )
                train_cfg.algorithm.adaptation_action_anchor_coef = float(
                    ppo.get("adaptation_action_anchor_coef", 0.0)
                )
                train_cfg.runner.stage3_load_optimizer_state = bool(
                    ppo.get("load_optimizer_state", True)
                )
                train_cfg.runner.stage3_nominal_evaluation_enabled = bool(
                    ppo.get("nominal_evaluation_enabled", True)
                )
                train_cfg.runner.stage3_additional_iterations = int(os.getenv(
                    "A2NERO_ADDITIONAL_ITERATIONS",
                    str(data["additional_iterations"]),
                ))
                train_cfg.algorithm.freeze_action_std_iterations = int(
                    train_cfg.runner.stage3_additional_iterations
                    * ppo.get("freeze_action_std_first_fraction", 0.0)
                )
                train_cfg.runner.stage3_source_iteration = (
                    int(profile["source"]["iteration"])
                    if stage == 4
                    else int(os.getenv("A2NERO_SOURCE_ITER", "-1"))
                )
                train_cfg.algorithm.adaptation_warmup_iterations = max(
                    1,
                    int(
                        train_cfg.runner.stage3_additional_iterations
                        * ppo.get("adaptation_warmup_fraction", 0.10)
                    ),
                )
                train_cfg.runner.stage3_source_run = (
                    profile["source"]["run"]
                    if stage == 4
                    else os.getenv("A2NERO_SOURCE_RUN", "")
                )
                train_cfg.runner.stage3_evaluation_interval = (
                    ppo["evaluation_interval_iterations"]
                    if train_cfg.runner.stage3_nominal_evaluation_enabled
                    else 0
                )
                train_cfg.runner.stage3_hard_stop_on_failure = bool(
                    train_cfg.runner.stage3_nominal_evaluation_enabled
                )
                train_cfg.runner.max_iterations = train_cfg.runner.stage3_additional_iterations
        else:
            train_cfg.runner.max_iterations = 40000
        if not train_cfg.runner.run_name:
            train_cfg.runner.run_name = stage_name

    return env_cfg, train_cfg

  
