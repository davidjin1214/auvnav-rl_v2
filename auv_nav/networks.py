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
    num_hidden_layers: int = 2,
) -> "list[nn.Module]":
    """Build repeated hidden blocks with optional LayerNorm and Dropout."""
    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")
    layers: list = []
    for i in range(num_hidden_layers):
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
        num_hidden_layers: int = 2,
    ) -> None:
        require_torch()
        super().__init__()
        layers = build_hidden_layers(
            in_dim,
            hidden_dim,
            use_layernorm,
            dropout_rate,
            num_hidden_layers=num_hidden_layers,
        )
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)
