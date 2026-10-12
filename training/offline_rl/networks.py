"""Twin Q networks used by Stage 3B (no actor/critic trainer here)."""

import torch
from torch import nn


class SACQNetwork(nn.Module):
    """Q(s,a) MLP with independent parameters for every instance."""

    architecture_version = "sac_q_mlp_v1"

    def __init__(self, state_dim=12, action_dim=6, hidden_dim=256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, state, action):
        if not torch.is_tensor(state) or not torch.is_tensor(action):
            raise TypeError("state and action must be torch tensors")
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state must have shape [B,{}]".format(self.state_dim))
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [B,{}]".format(self.action_dim))
        if state.shape[0] != action.shape[0]:
            raise ValueError("state/action batch sizes differ")
        value = self.net(torch.cat((state, action), dim=1))
        if value.shape != (state.shape[0], 1):
            raise RuntimeError("SACQNetwork output shape is not [B,1]")
        return value


def freeze_target_network(network):
    """Disable gradients for a target critic and put it in eval mode."""
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


@torch.no_grad()
def hard_update(target, source):
    """Copy online parameters into a target network."""
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target, source, tau):
    """Polyak update: target <- tau*source + (1-tau)*target."""
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0,1]")
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

