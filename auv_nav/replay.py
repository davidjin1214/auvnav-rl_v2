"""Episode-based replay buffer for recurrent agents."""

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


@dataclass(slots=True)
class SequenceReplayConfig:
    max_episodes: int = 5000
    burn_in: int = 8
    train_seq_len: int = 32


class EpisodeSequenceReplay:
    """Episode replay with subsequence sampling for recurrent agents."""

    def __init__(self, config: SequenceReplayConfig) -> None:
        self.config = config
        self.episodes: list[dict[str, np.ndarray]] = []
        self.transition_count = 0

    def __len__(self) -> int:
        return self.transition_count

    def _window_count(self, episode: dict[str, np.ndarray]) -> int:
        min_required = self.config.burn_in + self.config.train_seq_len
        total_transitions = int(episode["actions"].shape[0])
        return max(0, total_transitions - min_required + 1)

    def add_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if observations.ndim != 2:
            raise ValueError("observations must have shape (T+1, obs_dim).")
        if actions.ndim != 2:
            raise ValueError("actions must have shape (T, act_dim).")
        if rewards.ndim != 1 or costs.ndim != 1 or dones.ndim != 1:
            raise ValueError("rewards, costs, and dones must have shape (T,).")
        if observations.shape[0] != actions.shape[0] + 1:
            raise ValueError("observations length must equal actions length + 1.")
        if (
            actions.shape[0] != rewards.shape[0]
            or actions.shape[0] != costs.shape[0]
            or actions.shape[0] != dones.shape[0]
        ):
            raise ValueError("actions, rewards, costs, and dones must have matching lengths.")

        episode = {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "costs": np.asarray(costs, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        self.episodes.append(episode)
        self.transition_count += int(actions.shape[0])

        while len(self.episodes) > self.config.max_episodes:
            removed = self.episodes.pop(0)
            self.transition_count -= int(removed["actions"].shape[0])

    def ready(self, batch_size: int) -> bool:
        batch_size = max(1, int(batch_size))
        available_windows = sum(self._window_count(ep) for ep in self.episodes)
        return available_windows >= batch_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "max_episodes": self.config.max_episodes,
                "burn_in": self.config.burn_in,
                "train_seq_len": self.config.train_seq_len,
            },
            "episodes": self.episodes,
            "transition_count": self.transition_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_config = state.get("config")
        if saved_config is not None:
            expected = SequenceReplayConfig(**saved_config)
            if expected != self.config:
                raise ValueError(
                    "Replay config mismatch on load: "
                    f"saved={expected}, current={self.config}."
                )
        self.episodes = state.get("episodes", [])
        self.transition_count = int(state.get("transition_count", 0))

    def sample_batch(self, batch_size: int, device: "torch.device") -> "dict[str, torch.Tensor]":
        require_torch()
        batch_size = max(1, int(batch_size))
        seq_transitions = self.config.burn_in + self.config.train_seq_len
        candidates = []
        candidate_weights = []
        for episode in self.episodes:
            window_count = self._window_count(episode)
            if window_count > 0:
                candidates.append(episode)
                candidate_weights.append(window_count)
        if not candidates:
            raise RuntimeError("Not enough episodes to sample a recurrent batch.")

        weights = np.asarray(candidate_weights, dtype=np.float64)
        weights /= weights.sum()

        obs_list = []
        act_list = []
        rew_list = []
        cost_list = []
        done_list = []

        sampled_episode_indices = np.random.choice(
            len(candidates),
            size=batch_size,
            replace=True,
            p=weights,
        )
        for idx in sampled_episode_indices:
            episode = candidates[idx]
            window_count = candidate_weights[idx]
            start = int(np.random.randint(0, window_count))
            stop = start + seq_transitions

            obs_list.append(episode["obs"][start:stop + 1])
            act_list.append(episode["actions"][start:stop])
            rew_list.append(episode["rewards"][start:stop])
            cost_list.append(episode["costs"][start:stop])
            done_list.append(episode["dones"][start:stop])

        batch = {
            "obs": torch.as_tensor(np.stack(obs_list), device=device),
            "actions": torch.as_tensor(np.stack(act_list), device=device),
            "rewards": torch.as_tensor(np.stack(rew_list), device=device),
            "costs": torch.as_tensor(np.stack(cost_list), device=device),
            "dones": torch.as_tensor(np.stack(done_list), device=device),
        }
        return batch
