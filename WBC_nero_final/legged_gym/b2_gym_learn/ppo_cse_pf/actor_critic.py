import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):

    adaptation_module_encoder_hidden_dims = [512, 256, 128]

    adaptation_module_decoder_hidden_dims = [128, 64]
    
    adaptation_labels = ["base_velocity_loss", "gripper_pos_loss", "force_ee_loss", "force_base_loss"]
    adaptation_dims = [3, 3, 3, 3]
    adaptation_weights = [.2, .2, 1.0, 1.0]

    adaptation_labels_no_force = ["base_velocity_loss", "gripper_pos_loss", "gripper_orn_loss"]
    adaptation_dims_no_force = [3, 3, 3]
    adaptation_weights_no_force = [.2, .2, 1.0]


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self,  num_obs,
                        num_privileged_obs,
                        num_obs_pred,
                        num_single_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        temporal_encoder = kwargs.pop("temporal_encoder", "mlp")
        temporal_history_len = kwargs.pop("temporal_history_len", 16)
        temporal_embed_dim = kwargs.pop("temporal_embed_dim", 128)
        temporal_num_heads = kwargs.pop("temporal_num_heads", 4)
        temporal_dropout = kwargs.pop("temporal_dropout", 0.0)
        temporal_imu_dim = kwargs.pop("temporal_imu_dim", 18)
        temporal_imu_start = kwargs.pop("temporal_imu_start", None)
        adaptation_num_single_obs = kwargs.pop("adaptation_num_single_obs", None)
        latent_dim = kwargs.pop("latent_dim", None)
        critic_use_latent = kwargs.pop("critic_use_latent", False)
        adaptation_labels = kwargs.pop("adaptation_labels", None)
        adaptation_dims = kwargs.pop("adaptation_dims", None)
        adaptation_weights = kwargs.pop("adaptation_weights", None)
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()
        
        if adaptation_labels is not None and adaptation_dims is not None:
            self.adaptation_labels = list(adaptation_labels)
            self.adaptation_dims = list(adaptation_dims)
            self.adaptation_weights = list(adaptation_weights) if adaptation_weights is not None else [1.0] * len(self.adaptation_labels)
        elif num_obs_pred == sum(AC_Args.adaptation_dims_no_force):
            self.adaptation_labels = list(AC_Args.adaptation_labels_no_force)
            self.adaptation_dims = list(AC_Args.adaptation_dims_no_force)
            self.adaptation_weights = list(AC_Args.adaptation_weights_no_force)
        else:
            self.adaptation_labels = list(AC_Args.adaptation_labels)
            self.adaptation_dims = list(AC_Args.adaptation_dims)
            self.adaptation_weights = list(AC_Args.adaptation_weights)

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_obs_pred = num_obs_pred
        self.num_frames = int(num_obs / num_single_obs)
        self.num_latent_dim = int(latent_dim) if latent_dim is not None else self.num_frames * 2
        self.num_obs_now = num_single_obs
        self.critic_use_latent = critic_use_latent

        activation = get_activation(activation)

        # Adaptation module
        if temporal_encoder == "mha":
            self.adaptation_encoder_module = MHAEncoder(
                num_obs=self.num_obs,
                num_single_obs=self.num_obs_now,
                latent_dim=self.num_latent_dim,
                history_len=temporal_history_len,
                embed_dim=temporal_embed_dim,
                num_heads=temporal_num_heads,
                imu_dim=temporal_imu_dim,
                imu_start=temporal_imu_start,
                dropout=temporal_dropout,
                activation=activation,
            )
        else:
            self.adaptation_encoder_module = HistoryMLPEncoder(
                num_obs=self.num_obs,
                num_single_obs=self.num_obs_now,
                adaptation_num_single_obs=adaptation_num_single_obs,
                hidden_dims=AC_Args.adaptation_module_encoder_hidden_dims,
                latent_dim=self.num_latent_dim,
                activation=activation,
            )


        adaptation_module_decoder_layers = []
        adaptation_module_decoder_layers.append(nn.Linear(self.num_latent_dim, AC_Args.adaptation_module_decoder_hidden_dims[0]))
        adaptation_module_decoder_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_decoder_hidden_dims)):
            if l == len(AC_Args.adaptation_module_decoder_hidden_dims) - 1:
                adaptation_module_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l], self.num_obs_pred))
            else:
                adaptation_module_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l],
                              AC_Args.adaptation_module_decoder_hidden_dims[l + 1]))
                adaptation_module_decoder_layers.append(activation)
        self.adaptation_decoder_module = nn.Sequential(*adaptation_module_decoder_layers)



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_obs_now + self.num_latent_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_input_dim = self.num_privileged_obs + (self.num_latent_dim if self.critic_use_latent else 0)
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Encoder Module: {self.adaptation_encoder_module}")
        print(f"Adaptation Decoder Module: {self.adaptation_decoder_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _encode_latent(self, observations):
        latent = self.adaptation_encoder_module(observations)
        return latent

    def update_distribution(self, observations):
        latent = self._encode_latent(observations)
        mean = self.actor_body(torch.cat((observations[:, -self.num_obs_now:], latent), dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["obs_pred"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs"], policy_info=policy_info)

    def act_student(self, observations, policy_info={}):
        latent = self._encode_latent(observations)
        actions_mean = self.actor_body(torch.cat((observations[:, -self.num_obs_now:], latent), dim=-1))
        obs_pred = self.adaptation_decoder_module(latent)
        policy_info["latents"] = obs_pred.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, critic_observations, pred_obs, policy_info={}):
        actions_mean = self.actor_body(critic_observations)
        policy_info["latents"] = pred_obs
        return actions_mean

    def evaluate(self, critic_observations, observations=None, **kwargs):
        if self.critic_use_latent:
            if observations is None:
                raise ValueError("observations must be provided when critic_use_latent=True")
            latent = self._encode_latent(observations)
            critic_observations = torch.cat((critic_observations, latent), dim=-1)
        value = self.critic_body(critic_observations)
        return value

    def get_student_latent(self, observations):
        latent = self._encode_latent(observations)
        obs_pred = self.adaptation_decoder_module(latent)
        return obs_pred


class HistoryMLPEncoder(nn.Module):
    """MLP encoder over selected per-frame proprioceptive observations."""

    def __init__(
        self,
        num_obs,
        num_single_obs,
        adaptation_num_single_obs,
        hidden_dims,
        latent_dim,
        activation,
    ):
        super().__init__()
        self.num_single_obs = int(num_single_obs)
        self.num_frames = int(num_obs / num_single_obs)
        self.adaptation_num_single_obs = (
            self.num_single_obs
            if adaptation_num_single_obs is None
            else int(adaptation_num_single_obs)
        )
        if not 0 < self.adaptation_num_single_obs <= self.num_single_obs:
            raise ValueError(
                "adaptation_num_single_obs must be in [1, num_single_obs], got "
                f"{self.adaptation_num_single_obs} for num_single_obs={self.num_single_obs}"
            )

        layers = [
            nn.Linear(
                self.num_frames * self.adaptation_num_single_obs,
                hidden_dims[0],
            ),
            activation,
        ]
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], latent_dim))
            else:
                layers.extend((
                    nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]),
                    activation,
                ))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations):
        obs_seq = observations.reshape(
            observations.shape[0], self.num_frames, self.num_single_obs
        )
        proprioceptive_obs = obs_seq[:, :, :self.adaptation_num_single_obs]
        return self.mlp(proprioceptive_obs.reshape(observations.shape[0], -1))


class MHAEncoder(nn.Module):
    def __init__(
        self,
        num_obs,
        num_single_obs,
        latent_dim,
        history_len,
        embed_dim,
        num_heads,
        imu_dim,
        imu_start,
        dropout,
        activation,
    ):
        super().__init__()
        self.num_single_obs = num_single_obs
        self.num_frames = int(num_obs / num_single_obs)
        self.history_len = min(int(history_len), self.num_frames)
        self.imu_dim = int(imu_dim)
        self.imu_start = num_single_obs - self.imu_dim if imu_start is None else int(imu_start)

        if self.imu_start < 0 or self.imu_start + self.imu_dim > num_single_obs:
            raise ValueError(
                f"Invalid MHA IMU slice [{self.imu_start}, {self.imu_start + self.imu_dim}) "
                f"for num_single_obs={num_single_obs}"
            )

        self.obs_proj = nn.Linear(num_single_obs, embed_dim)
        self.imu_proj = nn.Linear(self.imu_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation,
            nn.Linear(embed_dim, latent_dim),
        )

    def forward(self, observations):
        obs_seq = observations.view(observations.shape[0], self.num_frames, self.num_single_obs)
        obs_seq = obs_seq[:, -self.history_len:, :]
        imu_seq = obs_seq[:, :, self.imu_start:self.imu_start + self.imu_dim]

        q = self.obs_proj(obs_seq) + self.pos_embedding
        kv = self.imu_proj(imu_seq) + self.pos_embedding
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        z = self.norm(attn_out + q)[:, -1, :]
        return self.out(z)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
