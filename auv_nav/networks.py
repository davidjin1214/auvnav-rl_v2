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


def build_hidden_layers(
    in_dim: int,
    hidden_dim: int,
    use_layernorm: bool = False,
    dropout_rate: float = 0.0,
) -> "list[nn.Module]":
    """Build two hidden Linear layers with optional LayerNorm and Dropout."""
    layers: list = []
    for i in range(2):
        in_features = in_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
    return layers


class MLP(_ModuleBase):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        layers = build_hidden_layers(in_dim, hidden_dim, use_layernorm, dropout_rate)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)
