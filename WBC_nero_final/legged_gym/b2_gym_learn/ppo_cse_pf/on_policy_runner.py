# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import json
import re
import warnings
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
try:
    import wandb
except ImportError:
    wandb = None
from datetime import datetime
from legged_gym.b2_gym_learn.ppo_cse_pf.ppo import PPO
from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.nero_smooth_actor_critic import NeroSmoothActorCritic
from legged_gym.b2_gym_learn.env import VecEnv


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Stage3HealthGateError(RuntimeError):
    """Controlled stop used when continuation no longer preserves the task."""


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        self.env.num_privileged_obs,
                                                        self.env.num_pred_obs,
                                                        self.env.num_single_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_pred_obs],
            [self.env.num_actions],
            aux_future_horizon=self.policy_cfg.get("aux_future_horizon", 0),
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.wandb_run = None
        self.wandb_sync_tensorboard = False
        self.wandb_log_episode_metrics = False
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.source_learning_iteration = int(self.cfg.get("stage3_source_iteration", 0))
        self.stage3_additional_iterations = int(
            self.cfg.get("stage3_additional_iterations", self.cfg.get("max_iterations", 0))
        )
        self.stage3_baseline = None
        self.stage3_best = None
        self.stage3_manifest = None
        self.stage3_source_observation_reference = None
        self.stage3_source_action_reference = None
        self.stage3_stop_result = None
        self.total_reward = 0
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            wandb_mode = os.getenv("WANDB_MODE", "online").lower()
            wandb_project = os.getenv("WANDB_PROJECT", "UniFP_a2")
            self.wandb_sync_tensorboard = _env_flag("WANDB_SYNC_TENSORBOARD", default=False)
            self.wandb_log_episode_metrics = _env_flag("WANDB_LOG_EPISODE_METRICS", default=False)
            wandb_console = os.getenv("WANDB_CONSOLE", "off")
            # Keep W&B artifacts outside the TensorBoard run directory to avoid
            # recursive syncing when sync_tensorboard is enabled.
            default_wandb_dir = os.path.join(os.path.dirname(self.log_dir), "wandb")
            wandb_dir = os.getenv("WANDB_DIR", default_wandb_dir)

            if wandb is None:
                print("wandb is not installed. Continuing with TensorBoard only.")
            elif wandb_mode != "disabled":
                try:
                    wandb_settings = wandb.Settings(console=wandb_console)
                    self.wandb_run = wandb.init(
                        project=wandb_project,
                        sync_tensorboard=self.wandb_sync_tensorboard,
                        name=self.wandb_run_name,
                        config=self.all_cfg,
                        dir=wandb_dir,
                        mode=wandb_mode,
                        settings=wandb_settings,
                    )
                    if not self.wandb_sync_tensorboard:
                        self.wandb_run.define_metric("train/timesteps")
                        self.wandb_run.define_metric("*", step_metric="train/timesteps")
                except Exception as exc:
                    self.wandb_run = None
                    print(f"W&B init failed ({exc}). Continuing with TensorBoard only.")
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if self.cfg.get("stage3_enabled", False):
                self._write_stage3_manifest()
        if (
            self.cfg.get("stage3_enabled", False)
            and self.cfg.get("stage3_nominal_evaluation_enabled", True)
            and self.stage3_baseline is None
            and not _env_flag("A2NERO_SKIP_NOMINAL_EVAL", default=False)
        ):
            self.stage3_baseline = self._run_nominal_evaluation(
                self.current_learning_iteration
            )
            self.stage3_best = dict(self.stage3_baseline)
            self._append_stage3_evaluation(self.stage3_baseline, baseline=True)
            self.save(os.path.join(self.log_dir, "model_best.pt"))
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_pred = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_pred"]
        obs, privileged_obs, obs_pred = obs.to(self.device), privileged_obs.to(self.device), obs_pred.to(
            self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            local_iteration = max(it - self.source_learning_iteration, 0)
            if self.cfg.get("stage3_enabled", False):
                self.alg.set_continuation_iteration(local_iteration)
            if hasattr(self.env, "set_stage3_curriculum_progress"):
                self.env.set_stage3_curriculum_progress(
                    local_iteration, self.stage3_additional_iterations
                )
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, obs_pred)
                    with torch.no_grad():
                        latent = self.alg.actor_critic.get_student_latent(obs)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, obs_pred = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_pred"]

                    obs, privileged_obs, obs_pred, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_pred.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(privileged_obs, obs)
            
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_adaptation_losses_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            completed_iteration = it + 1
            self.current_learning_iteration = completed_iteration
            if completed_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(completed_iteration)))
            evaluation_interval = int(self.cfg.get("stage3_evaluation_interval", 200))
            if (
                self.cfg.get("stage3_enabled", False)
                and self.stage3_baseline is not None
                and evaluation_interval > 0
                and completed_iteration % evaluation_interval == 0
            ):
                result = self._run_nominal_evaluation(completed_iteration)
                healthy = self._apply_stage3_health_gate(result)
                self._append_stage3_evaluation(result, baseline=False)
                if self._is_better_stage3_checkpoint(result):
                    self.stage3_best = dict(result)
                    self.save(os.path.join(self.log_dir, "model_best.pt"))
                if not healthy and self.cfg.get("stage3_hard_stop_on_failure", False):
                    self.stage3_stop_result = dict(result)
                    failed_path = os.path.join(
                        self.log_dir, f"model_{completed_iteration}_failed.pt"
                    )
                    self.save(failed_path)
                    if self.writer is not None:
                        self.writer.flush()
                    raise Stage3HealthGateError(
                        "Stage3 nominal health gate failed at iteration "
                        f"{completed_iteration}; saved {failed_path}"
                    )
                refreshed = self.env.get_observations()
                obs = refreshed["obs"].to(self.device)
                privileged_obs = refreshed["privileged_obs"].to(self.device)
                obs_pred = refreshed["obs_pred"].to(self.device)
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    @torch.inference_mode()
    def _run_nominal_evaluation(self, iteration):
        """Run the deterministic six-direction check used by the play script.

        The continuation environment normally owns random velocity and arm-goal
        curricula.  Those are useful training stresses, but they must not define
        whether the source policy still performs its nominal deployed gait.  A
        previous implementation evaluated random training goals (and injected
        a zero-action warm-up step), which made stable play/robot checkpoints
        look as if they fell frequently.
        """
        seed = int(self.cfg.get("stage3_evaluation_seed", 17800))
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = self.alg.actor_critic.training
        was_add_noise = bool(self.env.add_noise)
        was_teleop_mode = bool(self.env.cfg.env.teleop_mode)
        was_play = bool(self.env.play)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.env.set_stage3_nominal_override(True)
        self.env.add_noise = False
        self.env.cfg.env.teleop_mode = True
        self.env.play = True
        all_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(all_ids)
        zero_actions = torch.zeros(
            self.env.num_envs, self.env.num_actions, device=self.env.device
        )
        safe_ee_sphere = torch.as_tensor(
            self.env.cfg.goal_ee.ranges.init_pos_start,
            dtype=self.env.curr_ee_goal_sphere.dtype,
            device=self.env.device,
        ).reshape(1, 3)
        self.env._set_teleop_ee_goal(ee_goal_sphere=safe_ee_sphere)
        self.env.commands[:, :3] = 0.0
        observations = self.env.get_observations()["obs"].to(self.device)
        returns = torch.zeros(self.env.num_envs, device=self.device)
        reset_during_gait = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )
        self.alg.actor_critic.eval()
        initial_actions = self.alg.actor_critic.act_student(observations)
        if (
            self.stage3_source_observation_reference is None
            or self.stage3_source_action_reference is None
        ):
            reference_count = min(
                int(self.cfg.get("stage3_action_drift_reference_observations", 256)),
                int(observations.shape[0]),
            )
            self.stage3_source_observation_reference = (
                observations[:reference_count].detach().cpu().clone()
            )
            self.stage3_source_action_reference = (
                initial_actions[:reference_count].detach().cpu().clone()
            )
        reference_observations = self.stage3_source_observation_reference.to(self.device)
        reference_current_actions = self.alg.actor_critic.act_student(
            reference_observations
        )
        action_position_rms_drift = self._action_position_rms_drift(
            reference_current_actions,
            self.stage3_source_action_reference.to(self.device),
            float(self.cfg.get("stage3_action_scale", 0.25)),
        )

        directions = torch.tensor(
            [
                [0.3, 0.0, 0.0],
                [-0.3, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, -0.2, 0.0],
                [0.0, 0.0, 0.3],
                [0.0, 0.0, -0.3],
            ],
            dtype=self.env.commands.dtype,
            device=self.env.device,
        )
        direction_index = torch.remainder(all_ids, len(directions))
        per_env_command = directions[direction_index]
        settle_ticks = int(round(6.0 / float(self.env.dt)))
        gait_ticks = int(round(3.0 / float(self.env.dt)))
        total_ticks = int(round(10.0 / float(self.env.dt)))
        gait_stop = settle_ticks + gait_ticks
        reset_count = 0
        with torch.inference_mode():
            for step_index in range(total_ticks):
                gait_active = settle_ticks <= step_index < gait_stop
                self.env.commands[:, :3] = (
                    per_env_command if gait_active else 0.0
                )
                self.env._set_teleop_ee_goal(ee_goal_sphere=safe_ee_sphere)
                actions = (
                    initial_actions
                    if step_index == 0
                    else self.alg.actor_critic.act_student(observations)
                )
                obs_dict, rewards, dones, _ = self.env.step(actions)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device).bool()
                returns += rewards
                reset_count += int(dones.sum().item())
                if gait_active:
                    reset_during_gait |= dones
                observations = obs_dict["obs"].to(self.device)
        direction_labels = ("W", "S", "Z", "C", "A", "D")
        direction_success = {}
        for direction_id, label in enumerate(direction_labels):
            mask = direction_index == direction_id
            if torch.any(mask):
                direction_success[label] = float(
                    (~reset_during_gait[mask]).float().mean().item()
                )
        result = {
            "iteration": int(iteration),
            "seed": seed,
            "num_envs": int(self.env.num_envs),
            "evaluation": "deterministic_play_equivalent_six_direction",
            "success_rate": float((~reset_during_gait).float().mean().item()),
            "min_direction_success_rate": min(direction_success.values()),
            "direction_success_rate": direction_success,
            "reset_count_total": reset_count,
            "mean_return": float(returns.mean().item()),
            "action_position_rms_drift": action_position_rms_drift,
        }

        self.env.add_noise = was_add_noise
        self.env.cfg.env.teleop_mode = was_teleop_mode
        self.env.play = was_play
        self.env.set_stage3_nominal_override(False)
        self.env.reset_idx(all_ids)
        self.env.step(zero_actions)
        if was_training:
            self.alg.actor_critic.train()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        return result

    @staticmethod
    def _action_position_rms_drift(current_actions, source_actions, action_scale):
        if current_actions.shape != source_actions.shape:
            raise ValueError(
                "current/source deterministic action shapes differ: "
                f"{tuple(current_actions.shape)} != {tuple(source_actions.shape)}"
            )
        position_delta = float(action_scale) * (current_actions - source_actions)
        return float(torch.sqrt(torch.mean(position_delta.square())).item())

    def _write_stage3_manifest(self):
        config = self.env.get_stage3_config()
        manifest = {
            "version": 1,
            "source_run": self.cfg.get("stage3_source_run"),
            "source_iteration": self.source_learning_iteration,
            "source_checkpoint": self.cfg.get("stage3_source_checkpoint"),
            "source_sha256": self.cfg.get("stage3_source_sha256"),
            "additional_iterations": self.stage3_additional_iterations,
            "expected_final_iteration": (
                self.source_learning_iteration + self.stage3_additional_iterations
            ),
            "strict_checkpoint_load": True,
            "optimizer_state_restored": bool(
                self.cfg.get("stage3_load_optimizer_state", True)
            ),
            "checkpoint_learning_rate": float(self.alg.optimizer.param_groups[0]["lr"]),
            "entropy_coef": float(self.alg.entropy_coef),
            "optimization_safety": {
                "nominal_evaluation_enabled": bool(
                    self.cfg.get("stage3_nominal_evaluation_enabled", True)
                ),
                "freeze_action_std": bool(self.alg.freeze_action_std),
                "learning_rate_upper_bound": self.alg.learning_rate_upper_bound,
                "learning_rate_lower_bound": self.alg.learning_rate_lower_bound,
                "actor_warmup_iterations": self.alg.actor_warmup_iterations,
                "adaptation_warmup_iterations": self.alg.adaptation_warmup_iterations,
                "adaptation_policy_gradient_scale": self.alg.adaptation_policy_gradient_scale,
                "adaptation_optimizer_modules_only": (
                    self.alg.adaptation_optimizer_modules_only
                ),
                "source_action_anchor_coef": self.alg.source_action_anchor_coef,
                "adaptation_action_anchor_coef": (
                    self.alg.adaptation_action_anchor_coef
                ),
                "adaptation_learning_rate": float(
                    self.alg.adaptation_module_optimizer.param_groups[0]["lr"]
                ),
                "action_position_rms_warning": self.cfg.get(
                    "stage3_action_position_rms_warning"
                ),
                "action_drift_reference_observations": self.cfg.get(
                    "stage3_action_drift_reference_observations", 256
                ),
                "hard_stop_on_failure": bool(
                    self.cfg.get("stage3_hard_stop_on_failure", False)
                ),
            },
            "network": {
                "temporal_encoder": self.policy_cfg.get("temporal_encoder"),
                "adaptation_num_single_obs": self.policy_cfg.get("adaptation_num_single_obs"),
                "latent_dim": self.policy_cfg.get("latent_dim"),
                "actor_hidden_dims": self.policy_cfg.get("actor_hidden_dims"),
                "critic_hidden_dims": self.policy_cfg.get("critic_hidden_dims"),
            },
            "best_checkpoint_rule": (
                "disabled: continuation nominal evaluation was explicitly disabled"
                if not self.cfg.get("stage3_nominal_evaluation_enabled", True)
                else (
                    "play-equivalent deterministic six-direction nominal gate; eligible "
                    "iff minimum-direction success, aggregate success and mean return are "
                    "each within the configured degradation ratio of baseline; rank by "
                    "minimum-direction success, aggregate success, then mean return"
                )
            ),
        }
        self.stage3_manifest = manifest
        with open(
            os.path.join(self.log_dir, "stage3_config.json"), "w", encoding="utf-8"
        ) as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)
        with open(
            os.path.join(self.log_dir, "stage3_manifest.json"), "w", encoding="utf-8"
        ) as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)

    def _append_stage3_evaluation(self, result, baseline=False):
        record = dict(result)
        record["baseline"] = bool(baseline)
        path = os.path.join(self.log_dir, "stage3_nominal_evaluations.jsonl")
        with open(path, "a", encoding="utf-8") as evaluation_file:
            evaluation_file.write(json.dumps(record, sort_keys=True) + "\n")
        if baseline:
            with open(
                os.path.join(self.log_dir, "stage1_deterministic_baseline.json"),
                "w",
                encoding="utf-8",
            ) as baseline_file:
                json.dump(record, baseline_file, indent=2, sort_keys=True)

    def _stage3_result_is_healthy(self, result):
        floor = float(self.cfg.get("stage3_baseline_ratio_floor", 0.90))
        baseline_success = float(self.stage3_baseline["success_rate"])
        baseline_min_direction = float(
            self.stage3_baseline.get(
                "min_direction_success_rate", baseline_success
            )
        )
        baseline_return = float(self.stage3_baseline["mean_return"])
        success_ok = (
            baseline_success <= 1.0e-12
            or float(result["success_rate"]) >= floor * baseline_success
        )
        min_direction_ok = (
            baseline_min_direction <= 1.0e-12
            or float(result.get("min_direction_success_rate", result["success_rate"]))
            >= floor * baseline_min_direction
        )
        minimum_return = baseline_return - (1.0 - floor) * abs(baseline_return)
        return_ok = float(result["mean_return"]) >= minimum_return
        return success_ok and min_direction_ok and return_ok

    def _apply_stage3_health_gate(self, result):
        drift_warning_threshold = self.cfg.get("stage3_action_position_rms_warning")
        drift_warning = (
            drift_warning_threshold is not None
            and float(result.get("action_position_rms_drift", 0.0))
            > float(drift_warning_threshold)
        )
        result["action_drift_warning"] = bool(drift_warning)
        if drift_warning:
            warnings.warn(
                "Stage3 deterministic source-action drift is above its warning threshold: "
                f"{result['action_position_rms_drift']:.6f} > "
                f"{float(drift_warning_threshold):.6f}; task health gate remains based on "
                "nominal success and return",
                RuntimeWarning,
            )
        healthy = self._stage3_result_is_healthy(result)
        result["healthy"] = bool(healthy)
        if not healthy and hasattr(self.env, "freeze_stage3_curriculum"):
            self.env.freeze_stage3_curriculum()
            warning_path = os.path.join(self.log_dir, "stage3_curriculum_frozen.json")
            with open(warning_path, "w", encoding="utf-8") as warning_file:
                json.dump(
                    {
                        "reason": (
                            "nominal success rate or mean return below Stage1 health floor"
                        ),
                        "baseline": self.stage3_baseline,
                        "evaluation": result,
                        "curriculum": self.env.get_stage3_curriculum_state(),
                    },
                    warning_file,
                    indent=2,
                    sort_keys=True,
                )
        return healthy

    def _is_better_stage3_checkpoint(self, result):
        if not self._stage3_result_is_healthy(result):
            return False
        if self.stage3_best is None:
            return True
        candidate = (
            float(result.get("min_direction_success_rate", result["success_rate"])),
            float(result["success_rate"]),
            float(result["mean_return"]),
        )
        incumbent = (
            float(self.stage3_best.get(
                "min_direction_success_rate", self.stage3_best["success_rate"]
            )),
            float(self.stage3_best["success_rate"]),
            float(self.stage3_best["mean_return"]),
        )
        return candidate > incumbent

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        episode_metrics = {}

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    val = ep_info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.Tensor([val])
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor)
                value_item = value.item()
                episode_metrics[key] = value_item
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value_item:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        mean_std_item = mean_std.item()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        mean_reward = statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) > 0 else None
        mean_episode_length = statistics.mean(locs['lenbuffer']) if len(locs['rewbuffer']) > 0 else None

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_adaptation_module_loss', locs['mean_adaptation_module_loss'], locs['it'])
        self.writer.add_scalar(
            'Loss/source_action_anchor',
            self.alg.last_action_anchor_loss,
            locs['it'],
        )

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if getattr(self.alg, 'last_kl_mean', None) is not None:
            self.writer.add_scalar('Loss/kl_mean', self.alg.last_kl_mean, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std_item, locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if mean_reward is not None and mean_episode_length is not None:
            self.writer.add_scalar('Train/mean_reward', mean_reward, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_episode_length, locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', mean_episode_length, self.tot_time)

        if self.wandb_run is not None and not self.wandb_sync_tensorboard:
            wandb_metrics = {
                'train/timesteps': self.tot_timesteps,
                'train/iteration': locs['it'],
                'loss/value_function': locs['mean_value_loss'],
                'loss/surrogate': locs['mean_surrogate_loss'],
                'loss/adaptation_module': locs['mean_adaptation_module_loss'],
                'loss/learning_rate': self.alg.learning_rate,
                'policy/mean_noise_std': mean_std_item,
                'perf/total_fps': fps,
                'perf/collection_time': locs['collection_time'],
                'perf/learning_time': locs['learn_time'],
                'perf/iteration_time': iteration_time,
                'perf/total_time': self.tot_time,
            }
            if mean_reward is not None and mean_episode_length is not None:
                wandb_metrics.update({
                    'train/mean_reward': mean_reward,
                    'train/mean_episode_length': mean_episode_length,
                })
            if self.wandb_log_episode_metrics:
                for key, value in episode_metrics.items():
                    wandb_metrics[f'episode/{key}'] = value
            self.wandb_run.log(wandb_metrics)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if mean_reward is not None and mean_episode_length is not None:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std_item:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {mean_episode_length:.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std_item:.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        if mean_reward is not None:
            self.total_reward = mean_reward

    def save(self, path, infos=None):
        filename_match = re.fullmatch(r"model_(\d+)(?:_[^.]+)?\.pt", os.path.basename(path))
        if filename_match and int(filename_match.group(1)) != int(self.current_learning_iteration):
            raise ValueError(
                f"checkpoint filename iteration {filename_match.group(1)} does not match "
                f"internal iteration {self.current_learning_iteration}"
            )
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'adaptation_module_optimizer_state_dict': self.alg.adaptation_module_optimizer.state_dict(),
            'source_policy_state_dict': self.alg.source_policy_state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.env, "get_stage3_curriculum_state"):
            checkpoint['stage3_curriculum_state'] = self.env.get_stage3_curriculum_state()
        if hasattr(self.env, "get_stage3_config"):
            checkpoint['stage3_config'] = self.env.get_stage3_config()
        checkpoint['stage3_manifest'] = self.stage3_manifest
        checkpoint['stage3_baseline'] = self.stage3_baseline
        checkpoint['stage3_best'] = self.stage3_best
        checkpoint['stage3_source_observation_reference'] = (
            self.stage3_source_observation_reference
        )
        checkpoint['stage3_source_action_reference'] = self.stage3_source_action_reference
        checkpoint['stage3_stop_result'] = self.stage3_stop_result
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=True)
        if (
            self.alg.source_action_anchor_coef > 0.0
            or self.alg.adaptation_action_anchor_coef > 0.0
        ):
            source_policy_state = loaded_dict.get('source_policy_state_dict')
            if source_policy_state is None:
                self.alg.capture_source_policy()
            else:
                self.alg.load_source_policy_state_dict(source_policy_state)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'adaptation_module_optimizer_state_dict' in loaded_dict:
                self.alg.adaptation_module_optimizer.load_state_dict(
                    loaded_dict['adaptation_module_optimizer_state_dict']
                )
        optimizer_lr = float(self.alg.optimizer.param_groups[0]['lr'])
        self.alg.learning_rate = optimizer_lr
        if self.cfg.get('stage3_cap_learning_rate_to_checkpoint', False):
            configured_cap = self.alg.learning_rate_upper_bound
            checkpoint_cap = (
                optimizer_lr
                if configured_cap is None
                or optimizer_lr <= float(configured_cap) * (1.0 + 1.0e-12)
                else float(configured_cap)
            )
            self.alg.set_learning_rate_upper_bound(checkpoint_cap)
        loaded_iteration = int(loaded_dict.get('iter', 0))
        filename_match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
        if loaded_iteration == 0 and filename_match:
            recovered_iteration = int(filename_match.group(1))
            warnings.warn(
                f"checkpoint metadata iter=0; recovering source iteration as {recovered_iteration} "
                "from the checkpoint filename",
                RuntimeWarning,
            )
            loaded_iteration = recovered_iteration
        self.current_learning_iteration = loaded_iteration
        if loaded_dict.get('stage3_curriculum_state') is not None and hasattr(
            self.env, 'load_stage3_curriculum_state'
        ):
            self.env.load_stage3_curriculum_state(loaded_dict['stage3_curriculum_state'])
        self.stage3_baseline = loaded_dict.get('stage3_baseline')
        self.stage3_best = loaded_dict.get('stage3_best')
        self.stage3_manifest = loaded_dict.get('stage3_manifest')
        self.stage3_source_observation_reference = loaded_dict.get(
            'stage3_source_observation_reference'
        )
        self.stage3_source_action_reference = loaded_dict.get(
            'stage3_source_action_reference'
        )
        self.stage3_stop_result = loaded_dict.get('stage3_stop_result')
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
