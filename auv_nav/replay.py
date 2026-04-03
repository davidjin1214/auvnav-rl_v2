"""Transition replay buffer for off-policy RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .networks import require_torch

try:
    import torch
except ImportError:
    torch = None


@dataclass(slots=True)
class TransitionReplayConfig:
    capacity: int = 1_000_000


class TransitionReplay:
    """Standard transition replay buffer for feedforward off-policy agents."""

    def __init__(self, obs_dim: int, action_dim: int, config: TransitionReplayConfig) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        capacity = max(1, int(config.capacity))
        self.observations = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.observations[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.costs[idx] = float(cost)
        self.next_observations[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.config.capacity
        self.size = min(self.size + 1, self.config.capacity)

    def ready(self, batch_size: int) -> bool:
        return self.size >= max(1, int(batch_size))

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "capacity": self.config.capacity,
            },
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "costs": self.costs,
            "dones": self.dones,
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_config = state.get("config")
        if saved_config is not None:
            expected = TransitionReplayConfig(**saved_config)
            if expected != self.config:
                raise ValueError(
                    "Replay config mismatch on load: "
                    f"saved={expected}, current={self.config}."
                )
        if int(state.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError("Observation dimension mismatch on replay load.")
        if int(state.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Action dimension mismatch on replay load.")

        self.observations[...] = np.asarray(state["observations"], dtype=np.float32)
        self.next_observations[...] = np.asarray(state["next_observations"], dtype=np.float32)
        self.actions[...] = np.asarray(state["actions"], dtype=np.float32)
        self.rewards[...] = np.asarray(state["rewards"], dtype=np.float32)
        self.costs[...] = np.asarray(state["costs"], dtype=np.float32)
        self.dones[...] = np.asarray(state["dones"], dtype=np.float32)
        self.ptr = int(state.get("ptr", 0))
        self.size = int(state.get("size", 0))

    def sample_batch(self, batch_size: int, device: "torch.device") -> "dict[str, torch.Tensor]":
        require_torch()
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        batch_size = max(1, int(batch_size))
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "costs": torch.as_tensor(self.costs[indices], device=device),
            "next_obs": torch.as_tensor(self.next_observations[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }
