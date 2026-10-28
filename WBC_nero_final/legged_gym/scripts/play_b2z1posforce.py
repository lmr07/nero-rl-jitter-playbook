import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import sys
sys.path.append(unitree_rl_gym_path)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.test = True

    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    
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
        fig_ee_force = plt.figure()
        ax_ee_force = fig_ee_force.add_subplot(111, projection='3d')
        vector1_ee_force = np.array([0, 0, 0])
        vector2_ee_force = np.array([0, 0, 0])
        ax_ee_force.set_xlim([-0.7, 0.7])
        ax_ee_force.set_ylim([-0.7, 0.7])
        ax_ee_force.set_zlim([-0.7, 0.7])
        ax_ee_force.set_xlabel('X axis')
        ax_ee_force.set_ylabel('Y axis')
        ax_ee_force.set_zlabel('Z axis')
        line1_ee_force, = ax_ee_force.plot([0, vector1_ee_force[0]], [0, vector1_ee_force[1]], [0, vector1_ee_force[2]], marker='o', label='ee_force_pred')
        line2_ee_force, = ax_ee_force.plot([0, vector2_ee_force[0]], [0, vector2_ee_force[1]], [0, vector2_ee_force[2]], marker='o', label='ee_force_gt')
        ax_ee_force.legend()

        fig_base_force = plt.figure()
        ax_base_force = fig_base_force.add_subplot(111, projection='3d')
        vector1_base_force = np.array([0, 0, 0])
        vector2_base_force = np.array([0, 0, 0])
        ax_base_force.set_xlim([-0.7, 0.7])
        ax_base_force.set_ylim([-0.7, 0.7])
        ax_base_force.set_zlim([-0.7, 0.7])
        ax_base_force.set_xlabel('X axis')
        ax_base_force.set_ylabel('Y axis')
        ax_base_force.set_zlabel('Z axis')
        line1_base_force, = ax_base_force.plot([0, vector1_base_force[0]], [0, vector1_base_force[1]], [0, vector1_base_force[2]], marker='o', label='base_force_pred')
        line2_base_force, = ax_base_force.plot([0, vector2_base_force[0]], [0, vector2_base_force[1]], [0, vector2_base_force[2]], marker='o', label='base_force_gt')
        ax_base_force.legend()

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
    

    env.play = True
    policy_info = {}
    for i in range(100*int(env.max_episode_length)):
        actions = policy(obs, policy_info)
        # breakpoint()
        if FIX_COMMAND:
            env.commands[:, 0] = 0.    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.
            # env.gait_indices[:] = 0.
        obs, rews, dones, infos = env.step(actions.detach())
        if VISUAL_PRED:
            ee_force_pred = policy_info["latents"][0, 6:9]
            vector1_ee_force = ee_force_pred
            vector2_ee_force = env.forces_local[0, env.gripper_idx].detach().cpu().numpy() * env.obs_scales.ee_force
            print("ee_force_pred:", ee_force_pred*100)
            print("ee_force_ext:", vector2_ee_force*100)
            line1_ee_force.set_data([0, vector1_ee_force[0]], [0, vector1_ee_force[1]])
            line1_ee_force.set_3d_properties(np.array([0, vector1_ee_force[2]]))

            line2_ee_force.set_data([0, vector2_ee_force[0]], [0, vector2_ee_force[1]])
            line2_ee_force.set_3d_properties(np.array([0, vector2_ee_force[2]]))

            base_force_pred = policy_info["latents"][0, 9:12]
            vector1_base_force = base_force_pred
            vector2_base_force = env.forces_local[0,env.robot_base_idx].detach().cpu().numpy() * env.obs_scales.base_force
            print("ee_base_pred:", base_force_pred*100)
            print("ee_base_ext:", vector2_base_force*100)
            # line1_base_force.set_data([0, vector1_base_force[0]], [0, vector1_base_force[1]])
            # line1_base_force.set_3d_properties([0, vector1_base_force[2]])

            # line2_base_force.set_data([0, vector2_base_force[0]], [0, vector2_base_force[1]])
            # line2_base_force.set_3d_properties([0, vector2_base_force[2]])
            

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

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = False
    VISUAL_PRED = False
    args = get_args()
    play(args)
