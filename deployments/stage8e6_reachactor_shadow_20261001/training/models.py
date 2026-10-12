"""Actor network shared by BC pretraining and the future SAC stage."""

import torch
from torch import nn


class SACActor(nn.Module):
    """Gaussian SAC actor with a deterministic BC-compatible action API."""

    architecture_version = "sac_actor_mlp_v1"
    log_std_min = -20.0
    log_std_max = 2.0

    def __init__(self, state_dim=12, action_dim=6, hidden_dim=256,
                 log_std_init=-2.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(log_std_init))
        self.freeze_log_std()

    def freeze_log_std(self):
        """Keep log_std out of the supervised BC optimizer."""
        for parameter in self.log_std_head.parameters():
            parameter.requires_grad = False

    def unfreeze_log_std(self):
        """Enable the uncertainty branch for a later SAC implementation."""
        for parameter in self.log_std_head.parameters():
            parameter.requires_grad = True

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(
                state, dtype=self.mean_head.weight.dtype,
                device=self.mean_head.weight.device,
            )
        else:
            state = state.to(
                device=self.mean_head.weight.device,
                dtype=self.mean_head.weight.dtype,
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                "SACActor expected last dimension {}, got {}".format(
                    self.state_dim, state.shape[-1]
                )
            )
        feature = self.trunk(state)
        return self.mean_head(feature), self.log_std_head(feature)

    def deterministic_action(self, state):
        """Return normalized deterministic action in [-1, 1]^action_dim."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)

    def sample(self, state, eps=1e-6):
        """Sample a tanh-squashed Gaussian action for a future SAC target.

        This is an inference/target utility only; Stage 3B never optimizes the
        actor.  ``log_prob`` includes the standard tanh Jacobian correction.
        """
        mean, log_std = self.forward(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh)
        correction = 2.0 * (
            torch.log(torch.tensor(2.0, device=pre_tanh.device,
                                   dtype=pre_tanh.dtype))
            - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        if not torch.isfinite(action).all() or not torch.isfinite(log_prob).all():
            raise FloatingPointError("SACActor.sample produced NaN/Inf")
        return action, log_prob, torch.tanh(mean)
