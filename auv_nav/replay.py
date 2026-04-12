"""Transition replay buffer for off-policy RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    privileged_obs_dim: int = 0   # 0 = disabled


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
        # Optional privileged observation buffer.
        if config.privileged_obs_dim > 0:
            self.privileged_obs: np.ndarray | None = np.zeros(
                (capacity, config.privileged_obs_dim), dtype=np.float32
            )
        else:
            self.privileged_obs = None

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
        privileged_obs: np.ndarray | None = None,
    ) -> None:
        idx = self.ptr
        self.observations[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.costs[idx] = float(cost)
        self.next_observations[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx] = float(done)
        if self.privileged_obs is not None:
            if privileged_obs is not None:
                self.privileged_obs[idx] = np.asarray(privileged_obs, dtype=np.float32)
            # else: slot stays as zeros (safe default)
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

    @classmethod
    def from_npz(cls, path: str | Path) -> "TransitionReplay":
        """Create a read-only TransitionReplay pre-filled from a .npz file.

        The .npz must contain arrays: obs, actions, rewards, costs, next_obs, dones.
        """
        data = np.load(str(path))
        obs = data["obs"]
        actions = data["actions"]
        n_transitions, obs_dim = obs.shape
        action_dim = actions.shape[1]
        config = TransitionReplayConfig(capacity=n_transitions)
        replay = cls(obs_dim, action_dim, config)
        replay.observations[:n_transitions] = obs.astype(np.float32)
        replay.actions[:n_transitions] = actions.astype(np.float32)
        replay.rewards[:n_transitions] = data["rewards"].astype(np.float32)
        replay.costs[:n_transitions] = data["costs"].astype(np.float32)
        replay.next_observations[:n_transitions] = data["next_obs"].astype(np.float32)
        replay.dones[:n_transitions] = data["dones"].astype(np.float32)
        replay.size = n_transitions
        replay.ptr = 0  # offline buffer is read-only
        return replay

    def sample_batch(self, batch_size: int, device: "torch.device") -> "dict[str, torch.Tensor]":
        require_torch()
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        batch_size = max(1, int(batch_size))
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs":      torch.as_tensor(self.observations[indices], device=device),
            "actions":  torch.as_tensor(self.actions[indices], device=device),
            "rewards":  torch.as_tensor(self.rewards[indices], device=device),
            "costs":    torch.as_tensor(self.costs[indices], device=device),
            "next_obs": torch.as_tensor(self.next_observations[indices], device=device),
            "dones":    torch.as_tensor(self.dones[indices], device=device),
        }
        if self.privileged_obs is not None:
            batch["privileged_obs"] = torch.as_tensor(
                self.privileged_obs[indices], device=device
            )
        return batch


class DualBufferSampler:
    """RLPD-style symmetric sampling from offline + online replay buffers."""

    def __init__(
        self,
        offline_buffer: TransitionReplay,
        online_buffer: TransitionReplay,
        offline_ratio: float = 0.5,
    ) -> None:
        self.offline = offline_buffer
        self.online = online_buffer
        self.offline_ratio = offline_ratio

    def ready(self, batch_size: int) -> bool:
        n_online = max(1, int(batch_size * (1 - self.offline_ratio)))
        return self.online.ready(n_online) and len(self.offline) > 0

    def sample_batch(
        self, batch_size: int, device: "torch.device"
    ) -> "dict[str, torch.Tensor]":
        require_torch()
        n_offline = int(batch_size * self.offline_ratio)
        n_online = batch_size - n_offline
        offline_batch = self.offline.sample_batch(n_offline, device)
        online_batch = self.online.sample_batch(n_online, device)
        return {
            k: torch.cat([offline_batch[k], online_batch[k]], dim=0)
            for k in online_batch
        }
