"""TD3+BC agent for pure offline reinforcement learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .networks import MLP, require_torch
from .sac import AsymmetricQNetwork, QNetwork

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

_ModuleBase = nn.Module if nn is not None else object
_no_grad = torch.no_grad if torch is not None else (lambda: (lambda f: f))


@dataclass(slots=True)
class TD3BCConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256
    grad_clip_norm: float = 10.0
    use_layernorm: bool = False
    dropout_rate: float = 0.0
    privileged_obs_dim: int = 0
    privileged_actor_update_mode: Literal["zeros", "batch"] = "zeros"
    normalizer_eps: float = 1e-3


@dataclass(slots=True)
class ObservationNormalizerState:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-3

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
            "eps": float(self.eps),
        }


class ObservationNormalizer:
    def __init__(
        self,
        state: ObservationNormalizerState,
        device: str | "torch.device" = "cpu",
    ) -> None:
        require_torch()
        self.state = ObservationNormalizerState(
            mean=np.asarray(state.mean, dtype=np.float32),
            std=np.asarray(state.std, dtype=np.float32),
            eps=float(state.eps),
        )
        self.device = torch.device(device)
        self.mean_t = torch.as_tensor(self.state.mean, dtype=torch.float32, device=self.device)
        self.std_t = torch.as_tensor(self.state.std, dtype=torch.float32, device=self.device)

    @classmethod
    def from_observations(
        cls,
        observations: np.ndarray,
        *,
        eps: float = 1e-3,
        device: str | "torch.device" = "cpu",
    ) -> "ObservationNormalizer":
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected observations with shape [N, D], got {obs.shape}.")
        mean = obs.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = obs.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, float(eps))
        return cls(
            ObservationNormalizerState(mean=mean, std=std, eps=float(eps)),
            device=device,
        )

    @classmethod
    def identity(
        cls,
        obs_dim: int,
        *,
        eps: float = 1e-3,
        device: str | "torch.device" = "cpu",
    ) -> "ObservationNormalizer":
        return cls(
            ObservationNormalizerState(
                mean=np.zeros(obs_dim, dtype=np.float32),
                std=np.ones(obs_dim, dtype=np.float32),
                eps=float(eps),
            ),
            device=device,
        )

    def normalize_tensor(self, obs: "torch.Tensor") -> "torch.Tensor":
        return (obs - self.mean_t) / self.std_t

    def normalize_numpy(self, obs: np.ndarray) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        return (obs_array - self.state.mean) / self.state.std

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.state.mean.astype(np.float32).tolist(),
            "std": self.state.std.astype(np.float32).tolist(),
            "eps": float(self.state.eps),
        }

    def json_dict(self) -> dict[str, Any]:
        return self.state.to_json_dict()


class DeterministicActor(_ModuleBase):
    def __init__(self, config: TD3BCConfig) -> None:
        require_torch()
        super().__init__()
        self.net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.action_dim,
            use_layernorm=config.use_layernorm,
            dropout_rate=config.dropout_rate,
        )

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.net(obs))


class TD3BCAgent:
    def __init__(
        self,
        config: TD3BCConfig,
        *,
        obs_normalizer: ObservationNormalizer | None = None,
        device: str | "torch.device" = "cpu",
    ) -> None:
        require_torch()
        self.config = config
        self.device = torch.device(device)
        self.obs_normalizer = (
            obs_normalizer
            if obs_normalizer is not None
            else ObservationNormalizer.identity(
                config.obs_dim,
                eps=config.normalizer_eps,
                device=self.device,
            )
        )
        critic_cls = AsymmetricQNetwork if config.privileged_obs_dim > 0 else QNetwork

        self.actor = DeterministicActor(config).to(self.device)
        self.actor_target = DeterministicActor(config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = critic_cls(config).to(self.device)
        self.q2 = critic_cls(config).to(self.device)
        self.q1_target = critic_cls(config).to(self.device)
        self.q2_target = critic_cls(config).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.update_count = 0

    def reset_policy_state(self) -> None:
        return None

    def _normalize_obs(self, obs: "torch.Tensor") -> "torch.Tensor":
        return self.obs_normalizer.normalize_tensor(obs)

    def _actor_privileged_obs(
        self,
        batch: dict[str, "torch.Tensor"],
    ) -> "torch.Tensor | None":
        if self.config.privileged_obs_dim <= 0:
            return None
        if self.config.privileged_actor_update_mode == "batch":
            privileged_obs = batch.get("privileged_obs")
            if privileged_obs is None:
                raise ValueError(
                    "TD3+BC actor update requested batch privileged_obs, "
                    "but the batch does not provide it."
                )
            return privileged_obs
        return None

    @_no_grad()
    def act(
        self,
        obs: np.ndarray,
        policy_state: None = None,
        deterministic: bool = False,
    ) -> "tuple[np.ndarray, None]":
        _ = policy_state, deterministic
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        is_batched = obs_t.ndim > 1
        if not is_batched:
            obs_t = obs_t.unsqueeze(0)
        action = self.actor(self._normalize_obs(obs_t))
        action_np = action.cpu().numpy().astype(np.float32)
        if not is_batched:
            action_np = action_np[0]
        return action_np, None

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for src, tgt in zip(self.actor.parameters(), self.actor_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)
            for src, tgt in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)
            for src, tgt in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)

    def update(self, batch: "dict[str, torch.Tensor]") -> dict[str, float]:
        obs = self._normalize_obs(batch["obs"])
        next_obs = self._normalize_obs(batch["next_obs"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        privileged_obs = batch.get("privileged_obs")
        next_privileged_obs = batch.get("next_privileged_obs")

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            q1_next = self.q1_target(next_obs, next_actions, next_privileged_obs)
            q2_next = self.q2_target(next_obs, next_actions, next_privileged_obs)
            q_target = rewards + self.config.gamma * (1.0 - dones) * torch.min(q1_next, q2_next)

        q1_pred = self.q1(obs, actions, privileged_obs)
        q2_pred = self.q2(obs, actions, privileged_obs)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.grad_clip_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.grad_clip_norm)
        self.q2_opt.step()

        self.update_count += 1
        should_update_actor = (self.update_count % max(1, self.config.policy_freq)) == 0
        actor_privileged_obs = self._actor_privileged_obs(batch)

        if should_update_actor:
            pi = self.actor(obs)
            q_pi = self.q1(obs, pi, actor_privileged_obs)
            lambda_coef = self.config.alpha / q_pi.abs().mean().detach().clamp_min(1e-6)
            bc_loss = F.mse_loss(pi, actions)
            actor_loss = -lambda_coef * q_pi.mean() + bc_loss

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()
            self._soft_update_targets()
            actor_loss_value = float(actor_loss.item())
            bc_loss_value = float(bc_loss.item())
            lambda_value = float(lambda_coef.item())
            mean_q_value = float(q_pi.detach().mean().item())
        else:
            with torch.no_grad():
                pi = self.actor(obs)
                q_pi = self.q1(obs, pi, actor_privileged_obs)
                lambda_coef = self.config.alpha / q_pi.abs().mean().clamp_min(1e-6)
                bc_loss = F.mse_loss(pi, actions)
                actor_loss = -lambda_coef * q_pi.mean() + bc_loss
            actor_loss_value = float(actor_loss.item())
            bc_loss_value = float(bc_loss.item())
            lambda_value = float(lambda_coef.item())
            mean_q_value = float(q_pi.mean().item())

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "critic_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
            "actor_loss": actor_loss_value,
            "bc_loss": bc_loss_value,
            "mean_q": mean_q_value,
            "target_q": float(q_target.detach().mean().item()),
            "td_abs_error": float((q1_pred.detach() - q_target.detach()).abs().mean().item()),
            "lambda": lambda_value,
            "policy_updated": float(1.0 if should_update_actor else 0.0),
        }

    def save(self, path: str) -> None:
        require_torch()
        payload = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "update_count": self.update_count,
            "config": asdict(self.config),
            "obs_normalizer": self.obs_normalizer.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        require_torch()
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        if "actor_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
        if "q1_opt" in payload:
            self.q1_opt.load_state_dict(payload["q1_opt"])
        if "q2_opt" in payload:
            self.q2_opt.load_state_dict(payload["q2_opt"])
        self.update_count = int(payload.get("update_count", 0))

        normalizer_state = payload.get("obs_normalizer")
        if normalizer_state is not None:
            self.obs_normalizer = ObservationNormalizer(
                ObservationNormalizerState(
                    mean=np.asarray(normalizer_state["mean"], dtype=np.float32),
                    std=np.asarray(normalizer_state["std"], dtype=np.float32),
                    eps=float(normalizer_state.get("eps", self.config.normalizer_eps)),
                ),
                device=self.device,
            )
