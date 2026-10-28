from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class B2Z1PosForceRoughCfg( LeggedRobotCfg ):

    class goal_ee:
        
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.25, 0.2, -0.15]
        collision_lower_limits = [-0.7, -0.2, -0.8]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
        
        class sphere_center:
            x_offset = 0.2 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.8 # Relative to terrain
        
        class ranges:
            init_pos_start = [0.66, np.pi/4, 0]
            # init_pos_end = [0.66, np.pi/6, 0]
            # init_pos_end = [0.66, -1 * np.pi / 3, 0]
            init_pos_end = [0.66, 0, 0]
            pos_l = [0.35, 0.95]
            pos_p = [-2 * np.pi / 5, 2 * np.pi / 5]
            pos_y = [-3 * np.pi/ 5, 3 * np.pi /5]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
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

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -1.5, # -0.63,
            'z1_wrist_angle': 0, # -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57, # 0.0,
            'z1_jointGripper': -0.785,
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 2.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_leg_mass = False
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

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.8


    class env( LeggedRobotCfg.env ):

        num_gripper_joints = 2
        num_actions = 17
        num_torques = 17
        frame_stack = 32
        c_frame_stack = 3
        num_single_obs = 73
        num_pred_obs = 12
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 149
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = False
        frequencies = 1.0

        action_delay = 3 # Not used, assigned in code
        teleop_mode = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 15 # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
        ang_vel_yaw_clip = 0.2
        ang_vel_pitch_clip = 0.5
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1


        zero_vel_cmd_prob = 0.3
        zero_vel_cmd_prob_after_force = 0.8

        # Push gripper 
        push_gripper_stators = True
        push_gripper_interval_s_cmd = [3.5, 9.0]
        push_gripper_duration_s_cmd = [1.0, 3.0]
        gripper_forced_prob_cmd = 0.8
        push_gripper_interval_s_ext = [3.5, 9.0]
        push_gripper_duration_s_ext = [1.0, 3.0]
        gripper_forced_prob_ext = 0.8
        randomize_gripper_force_gains = True
        gripper_force_kp_range = [200., 200.]
        gripper_force_kd_range = [3.0, 3.0]
        gripper_prop_kd = 0.1

        max_push_force_xyz_gripper_cmd = [-60, 60]
        max_push_force_xyz_gripper_ext = [-60, 60]

        settling_time_force_gripper_s = 1.0

        # Push base
        push_robot_base = True
        push_base_interval_s_cmd = [3.5, 9.0]
        push_base_duration_s_cmd = [1.0, 3.0]
        base_forced_prob_cmd = 0.8
        push_base_interval_s_ext = [6.0, 12.0]
        push_base_duration_s_ext = [1.0, 3.0]
        base_forced_prob_ext = 0.8
        randomize_base_force_gains = True
        base_force_kp_range = [200., 200.]
        base_force_kd_range = [200.0, 200.0]
        base_prop_kd = 0.1

        max_push_force_xyz_base_cmd = [-50, 50]
        max_push_force_xyz_base_ext = [-50, 50]

        force_z_base_ext_scale = 0.1

        settling_time_force_base_s = 3.0

        force_start_step = 8000
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
        stiffness = {'hip': 300., 'thigh': 300, 'calf': 500, 'z1_waist': 64., 'z1_shoulder': 128., 'z1_elbow': 64., 'z1_wrist_angle': 64., 'z1_forearm_roll': 64., 'z1_wrist_rotate': 64., 'z1_jointGripper': 64., }  # [N*m/rad]
        damping = {'hip': 7.5, 'thigh': 7.5, 'calf': 12.5, 'z1_waist': 1.5, 'z1_shoulder': 3.0, 'z1_elbow': 1.5, 'z1_wrist_angle': 1.5, 'z1_forearm_roll': 1.5, 'z1_wrist_rotate': 1.5, 'z1_jointGripper': 1.5, }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08

    class asset( LeggedRobotCfg.asset ):
        file = '/home/lhao/UniFP_a2/resources/robots/a2piper/urdf/a2piper.urdf'
        name = "b2z1"
        foot_name = "foot"
        thigh_name = "thigh"
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        gait_vel_sigma = 2.0
        gait_force_sigma = 2.0
        kappa_gait_probs = 0.07

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1.0
        soft_dof_pos_limit = 0.8 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        base_height_target = 0.50
        max_contact_force = 200. # forces above this value are penalized

        cycle_time = 0.64
        target_joint_pos_scale = 0.17
        target_joint_pos_thd = 0.5

        sigma_force = 1/50
        
        class scales:

            feet_contact_number = 2.0

            tracking_lin_vel_force_world = 2.0
            tracking_ang_vel = 1.0 

            torques = -5.e-6 
            stand_still = 0.5 
            ref_dof_leg = 1.0
            alive = 1.5
            lin_vel_z = -1.5 
            feet_air_time = 1.0 
            feet_height = 1.0 
            ang_vel_xy = -0.02 
            dof_acc = -2.5e-7 
            dof_vel = -8.e-4
            dof_acc_arm = -4.5e-7 
            dof_vel_arm = -2.e-4
            collision = -5. 
            action_rate = -0.02 
            action_rate_arm = -0.045 
            dof_pos_limits = -10.0 
            torque_limits = -0.005
            hip_pos = -0.5  
            feet_drag = -0.0008 
            feet_contact_forces = -0.001 
            base_height = -2.0
            feet_pos_xy = -0.5
            feet_height_high = -15
            
            arm_termination = 0.
            tracking_ee_sphere = 0.
            tracking_ee_force_world = 2.0
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = 0.
            arm_orientation = 0.
            arm_energy_abs_sum = 0.
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = 0.

class B2Z1PosForceRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'b2z1_pos_force'

  
