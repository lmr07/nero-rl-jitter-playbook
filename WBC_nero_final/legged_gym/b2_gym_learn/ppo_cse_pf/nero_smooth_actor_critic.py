"""Versioned Nero policy with bounded requests and no current-action shortcut."""
import torch
from torch import nn
from torch.distributions import Normal
from .actor_critic import ActorCritic


class BoundedRequestHead(nn.Module):
    def __init__(self, body, lower, upper, num_obs_now, initial=None):
        super().__init__()
        lower = torch.as_tensor(lower, dtype=torch.float32)
        upper = torch.as_tensor(upper, dtype=torch.float32)
        if lower.shape != (7,) or upper.shape != (7,) or not torch.isfinite(torch.cat((lower, upper))).all():
            raise ValueError('Expected seven finite request action bounds')
        if not torch.all((lower < 0) & (upper > 0)):
            raise ValueError('The default pose must be strictly within request bounds')
        self.body = body
        center = (lower + upper)/2
        radius = (upper - lower)/2
        initial = torch.zeros_like(center) if initial is None else torch.as_tensor(initial, dtype=torch.float32)
        if initial.shape != (7,) or not torch.all((initial > lower) & (initial < upper)):
            raise ValueError('Initial gravity-compensated request must be within request bounds')
        mask = torch.ones(body[0].in_features)
        if num_obs_now != 39:
            raise ValueError('Nero smooth policy requires the 39-value frame')
        mask[14:21] = 0
        self.register_buffer('input_mask', mask)
        self.register_buffer('center', center)
        self.register_buffer('radius', radius)
        self.register_buffer('zero_shift', torch.atanh((initial-center)/radius))

    def forward(self, observations):
        return self.center + self.radius*torch.tanh(self.body(observations*self.input_mask) + self.zero_shift)


class NeroSmoothActorCritic(ActorCritic):
    def __init__(self, *args, request_action_lower, request_action_upper, request_action_initial=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_noise_std = .002
        if torch.any(self.std <= self.minimum_noise_std):
            raise ValueError('Initial exploration std must exceed the smooth-policy floor')
        with torch.no_grad():
            self.std.copy_(torch.log(torch.expm1(self.std-self.minimum_noise_std)))
        nn.init.orthogonal_(self.actor_body[-1].weight, gain=.01)
        nn.init.zeros_(self.actor_body[-1].bias)
        self.actor_body = BoundedRequestHead(self.actor_body, request_action_lower, request_action_upper,
                                             self.num_obs_now, request_action_initial)

    def update_distribution(self, observations):
        latent = self._encode_latent(observations)
        mean = self.actor_body(torch.cat((observations[:, -self.num_obs_now:], latent), dim=-1))
        scale = self.minimum_noise_std + torch.nn.functional.softplus(self.std)
        self.distribution = Normal(mean, mean*0.+scale)
