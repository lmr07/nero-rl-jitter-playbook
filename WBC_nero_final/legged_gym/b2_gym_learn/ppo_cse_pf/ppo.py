import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.rollout_storage import RolloutStorage


class Adaptation_Args():
    adaptation_module_learning_rate = 1.e-5
    num_adaptation_module_substeps = 1
    adaptation_batch_size = 64


class PPO:
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 learning_rate_upper_bound=None,
                 learning_rate_lower_bound=1.0e-8,
                 freeze_action_std=False,
                 freeze_action_std_iterations=0,
                 actor_warmup_iterations=0,
                 adaptation_module_learning_rate=Adaptation_Args.adaptation_module_learning_rate,
                 adaptation_warmup_iterations=0,
                 adaptation_policy_gradient_scale=1.0,
                 adaptation_optimizer_modules_only=False,
                 source_action_anchor_coef=0.0,
                 adaptation_action_anchor_coef=0.0,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.learning_rate_upper_bound = (
            None
            if learning_rate_upper_bound is None
            else float(learning_rate_upper_bound)
        )
        self.learning_rate_lower_bound = float(learning_rate_lower_bound)
        if self.learning_rate_lower_bound <= 0.0:
            raise ValueError("learning_rate_lower_bound must be positive")
        if (
            self.learning_rate_upper_bound is not None
            and self.learning_rate_lower_bound > self.learning_rate_upper_bound
        ):
            raise ValueError(
                "learning_rate_lower_bound cannot exceed learning_rate_upper_bound"
            )
        self.freeze_action_std = bool(freeze_action_std)
        self.freeze_action_std_iterations = max(int(freeze_action_std_iterations), 0)
        self.actor_warmup_iterations = max(int(actor_warmup_iterations), 0)
        self.actor_updates_enabled = self.actor_warmup_iterations == 0
        self.adaptation_warmup_iterations = max(int(adaptation_warmup_iterations), 0)
        self.adaptation_policy_gradient_scale = float(adaptation_policy_gradient_scale)
        if not 0.0 <= self.adaptation_policy_gradient_scale <= 1.0:
            raise ValueError("adaptation_policy_gradient_scale must be in [0, 1]")
        self.adaptation_updates_enabled = self.adaptation_warmup_iterations == 0
        self.adaptation_optimizer_modules_only = bool(
            adaptation_optimizer_modules_only
        )
        self.source_action_anchor_coef = float(source_action_anchor_coef)
        self.adaptation_action_anchor_coef = float(adaptation_action_anchor_coef)
        if self.source_action_anchor_coef < 0.0 or self.adaptation_action_anchor_coef < 0.0:
            raise ValueError("action anchor coefficients must be non-negative")
        self.source_adaptation_encoder = None
        self.source_actor_body = None
        self.last_action_anchor_loss = 0.0

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        if self.freeze_action_std or self.freeze_action_std_iterations > 0:
            # Keep the checkpoint's per-axis exploration distribution exactly.
            # The parameter remains in the optimizer param groups so legacy Adam
            # state can still be restored without changing checkpoint structure.
            self.actor_critic.std.requires_grad_(False)
        
        self.adaptation_labels = self.actor_critic.adaptation_labels
        self.adaptation_dims = self.actor_critic.adaptation_dims
        self.adaptation_weights = self.actor_critic.adaptation_weights
        
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        adaptation_parameters = self.actor_critic.parameters()
        if self.adaptation_optimizer_modules_only:
            adaptation_parameters = list(
                self.actor_critic.adaptation_encoder_module.parameters()
            ) + list(self.actor_critic.adaptation_decoder_module.parameters())
        self.adaptation_module_optimizer = optim.Adam(
            adaptation_parameters,
            lr=float(adaptation_module_learning_rate),
        )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def set_learning_rate_upper_bound(self, upper_bound):
        """Cap adaptive scheduling without replacing restored Adam state."""
        self.learning_rate_upper_bound = float(upper_bound)
        self.learning_rate = min(self.learning_rate, self.learning_rate_upper_bound)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min(
                float(param_group['lr']), self.learning_rate_upper_bound
            )

    def set_continuation_iteration(self, local_iteration):
        """Enable auxiliary adaptation only after the nominal warm-up."""
        self.actor_updates_enabled = (
            int(local_iteration) >= self.actor_warmup_iterations
        )
        self.adaptation_updates_enabled = (
            int(local_iteration) >= self.adaptation_warmup_iterations
        )
        freeze_std_now = self.freeze_action_std or (
            int(local_iteration) < self.freeze_action_std_iterations
        )
        self.actor_critic.std.requires_grad_(not freeze_std_now)

    def capture_source_policy(self):
        """Freeze the loaded checkpoint's deterministic policy as an anchor."""
        self.source_adaptation_encoder = copy.deepcopy(
            self.actor_critic.adaptation_encoder_module
        ).to(self.device)
        self.source_actor_body = copy.deepcopy(
            self.actor_critic.actor_body
        ).to(self.device)
        self.source_adaptation_encoder.eval()
        self.source_actor_body.eval()
        for module in (self.source_adaptation_encoder, self.source_actor_body):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def source_policy_state_dict(self):
        if self.source_adaptation_encoder is None or self.source_actor_body is None:
            return None
        return {
            "adaptation_encoder": self.source_adaptation_encoder.state_dict(),
            "actor_body": self.source_actor_body.state_dict(),
        }

    def load_source_policy_state_dict(self, state_dict):
        self.capture_source_policy()
        self.source_adaptation_encoder.load_state_dict(
            state_dict["adaptation_encoder"], strict=True
        )
        self.source_actor_body.load_state_dict(
            state_dict["actor_body"], strict=True
        )

    def _source_action_mean(self, observations):
        if self.source_adaptation_encoder is None or self.source_actor_body is None:
            raise RuntimeError(
                "source action anchoring is enabled before a source policy was captured"
            )
        with torch.no_grad():
            latent = self.source_adaptation_encoder(observations)
            source_input = torch.cat((
                observations[:, -self.actor_critic.num_obs_now:], latent
            ), dim=-1)
            return self.source_actor_body(source_input)

    def _freeze_warmup_actor_gradients(self):
        if self.actor_updates_enabled:
            return
        for parameter in self.actor_critic.actor_body.parameters():
            parameter.grad = None
        self.actor_critic.std.grad = None

    def _clear_non_adaptation_gradients(self):
        """Keep the auxiliary optimizer isolated from policy/value modules."""
        if not self.adaptation_optimizer_modules_only:
            return
        for module in (
            self.actor_critic.actor_body,
            self.actor_critic.critic_body,
        ):
            for parameter in module.parameters():
                parameter.grad = None
        self.actor_critic.std.grad = None

    def _scale_adaptation_policy_gradients(self):
        """Prevent PPO from rapidly moving the source checkpoint's latent map."""
        scale = self.adaptation_policy_gradient_scale
        modules = tuple(
            module
            for module in (
                getattr(self.actor_critic, 'adaptation_encoder_module', None),
                getattr(self.actor_critic, 'adaptation_decoder_module', None),
            )
            if module is not None
        )
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is None:
                    continue
                if scale == 0.0:
                    parameter.grad = None
                elif scale != 1.0:
                    parameter.grad.mul_(scale)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_pred_shape, action_shape, aux_future_horizon=0):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_pred_shape,
            action_shape,
            self.device,
            aux_future_horizon=aux_future_horizon,
        )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_pred):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, observations=obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observation_preds = obs_pred
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, last_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, observations=last_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_action_anchor_loss = 0
        kl_total = 0.0
        kl_updates = 0
        
        mean_adaptation_losses = {}
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(self.adaptation_labels, self.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, obs_pred_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, observations=obs_batch, masks=masks_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                action_anchor_loss = torch.zeros((), device=self.device)
                if self.source_action_anchor_coef > 0.0:
                    source_mu_batch = self._source_action_mean(obs_batch)
                    action_anchor_loss = F.mse_loss(mu_batch, source_mu_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        kl_total += kl_mean.item()
                        kl_updates += 1

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(
                                self.learning_rate_lower_bound,
                                self.learning_rate / 1.5,
                            )
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            upper_bound = (
                                1e-2
                                if self.learning_rate_upper_bound is None
                                else self.learning_rate_upper_bound
                            )
                            self.learning_rate = min(
                                upper_bound, self.learning_rate * 1.5
                            )
                        if self.learning_rate_upper_bound is not None:
                            self.learning_rate = min(
                                self.learning_rate, self.learning_rate_upper_bound
                            )
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + self.source_action_anchor_coef * action_anchor_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self._scale_adaptation_policy_gradients()
                self._freeze_warmup_actor_gradients()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_action_anchor_loss += action_anchor_loss.item()

                data_size = critic_obs_batch.shape[0]
                num_train = int(data_size // 5 * 4)

                # Adaptation module gradient step, only update concurrent state estimation module, not policy network
                if len(self.adaptation_labels) > 0 and self.adaptation_updates_enabled:

                    for epoch in range(Adaptation_Args.num_adaptation_module_substeps):

                        adaptation_pred = self.actor_critic.get_student_latent(obs_batch)
                        with torch.no_grad():
                            adaptation_target = obs_pred_batch
                        adaptation_loss = 0
                        for idx, (label, length, weight) in enumerate(zip(self.adaptation_labels, self.adaptation_dims, self.adaptation_weights)):

                            start, end = label_start_end[label]
                            selection_indices = torch.linspace(start, end - 1, steps=end - start, dtype=torch.long)

                            idx_adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices] * weight,
                                                            adaptation_target[:, selection_indices] * weight)
                            mean_adaptation_losses[label] += idx_adaptation_loss.item()

                            adaptation_loss += idx_adaptation_loss

                        if self.adaptation_action_anchor_coef > 0.0:
                            current_latent = self.actor_critic._encode_latent(obs_batch)
                            current_mean = self.actor_critic.actor_body(torch.cat((
                                obs_batch[:, -self.actor_critic.num_obs_now:],
                                current_latent,
                            ), dim=-1))
                            source_mean = self._source_action_mean(obs_batch)
                            adaptation_loss = (
                                adaptation_loss
                                + self.adaptation_action_anchor_coef
                                * F.mse_loss(current_mean, source_mean)
                            )

                        self.adaptation_module_optimizer.zero_grad()
                        adaptation_loss.backward()
                        self._clear_non_adaptation_gradients()
                        self.adaptation_module_optimizer.step()

                        mean_adaptation_module_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.last_action_anchor_loss = mean_action_anchor_loss / num_updates
        self.last_kl_mean = kl_total / kl_updates if kl_updates else None
        mean_adaptation_module_loss /= (num_updates * Adaptation_Args.num_adaptation_module_substeps * Adaptation_Args.adaptation_batch_size)
        for label in self.adaptation_labels:
            mean_adaptation_losses[label] /= (num_updates * Adaptation_Args.num_adaptation_module_substeps * Adaptation_Args.adaptation_batch_size)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_adaptation_losses
