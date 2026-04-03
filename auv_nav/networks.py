"""Neural network building blocks for recurrent RL agents."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ImportError as exc:
    torch = None
    nn = None
    Normal = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "This module requires PyTorch. Install `torch` to use it."
        ) from _TORCH_IMPORT_ERROR


class MLP(_ModuleBase):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class RecurrentEncoder(_ModuleBase):
    def __init__(self, obs_dim: int, encoder_dim: int, hidden_dim: int) -> None:
        require_torch()
        super().__init__()
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(encoder_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        obs_seq: "torch.Tensor",
        hidden: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        emb = self.obs_mlp(obs_seq)
        feat_seq, hidden_out = self.gru(emb, hidden)
        return feat_seq, hidden_out

    def step(
        self,
        obs: "torch.Tensor",
        hidden: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        obs_seq = obs.unsqueeze(1)
        feat_seq, hidden_out = self.forward(obs_seq, hidden)
        return feat_seq[:, 0], hidden_out
