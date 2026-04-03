"""Neural network building blocks shared across RL agents."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = None
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
