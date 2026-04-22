"""ReBRAC agent for pure offline reinforcement learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .networks import MLP, require_torch
from .sac import AsymmetricQNetwork, QNetwork
from .td3bc import (
    ObservationNormalizer,
    ObservationNormalizerState,
)

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
class ReBRACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256
    grad_clip_norm: float = 10.0
    actor_use_layernorm: bool = False
    critic_use_layernorm: bool = True
    dropout_rate: float = 0.0
    privileged_obs_dim: int = 0
    privileged_actor_update_mode: Literal["zeros", "batch"] = "zeros"
    normalizer_eps: float = 1e-3
    normalize_q: bool = True


class DeterministicActor(_ModuleBase):
    def __init__(self, config: ReBRACConfig) -> None:
        require_torch()
        super().__init__()
        self.net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.action_dim,
            use_layernorm=config.actor_use_layernorm,
            dropout_rate=config.dropout_rate,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.net(obs))


class ReBRACPolicy:
    """Lightweight deterministic policy wrapper for evaluation workers."""

    def __init__(
        self,
        config: ReBRACConfig,
        *,
        actor_state: dict[str, Any],
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
        self.actor = DeterministicActor(config).to(self.device)
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        device: str | "torch.device" = "cpu",
    ) -> "ReBRACPolicy":
        config = ReBRACConfig(**payload["config"])
        normalizer_state = payload.get("obs_normalizer")
        obs_normalizer = None
        if normalizer_state is not None:
            obs_normalizer = ObservationNormalizer(
                ObservationNormalizerState(
                    mean=np.asarray(normalizer_state["mean"], dtype=np.float32),
                    std=np.asarray(normalizer_state["std"], dtype=np.float32),
                    eps=float(normalizer_state.get("eps", config.normalizer_eps)),
                ),
                device=device,
            )
        return cls(
            config,
            actor_state=payload["actor"],
            obs_normalizer=obs_normalizer,
            device=device,
        )

    def reset_policy_state(self) -> None:
        return None

    @_no_grad()
    def act(
        self,
        obs: np.ndarray,
        policy_state: None = None,
        deterministic: bool = True,
    ) -> "tuple[np.ndarray, None]":
        _ = policy_state, deterministic
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        is_batched = obs_t.ndim > 1
        if not is_batched:
            obs_t = obs_t.unsqueeze(0)
        action = self.actor(self.obs_normalizer.normalize_tensor(obs_t))
        action_np = action.cpu().numpy().astype(np.float32)
        if not is_batched:
            action_np = action_np[0]
        return action_np, None


class ReBRACAgent:
    def __init__(
        self,
        config: ReBRACConfig,
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
        self._last_actor_metrics = {
            "actor_loss": float("nan"),
            "bc_loss": float("nan"),
            "mean_q": float("nan"),
            "lambda": float("nan"),
        }
        self._has_actor_metrics = False

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
                    "ReBRAC actor update requested batch privileged_obs, "
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

    def _actor_loss_terms(
        self,
        obs: "torch.Tensor",
        actions: "torch.Tensor",
        actor_privileged_obs: "torch.Tensor | None",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        pi = self.actor(obs)
        q_pi = torch.min(
            self.q1(obs, pi, actor_privileged_obs),
            self.q2(obs, pi, actor_privileged_obs),
        )
        if self.config.normalize_q:
            lambda_coef = q_pi.abs().mean().detach().clamp_min(1e-6).reciprocal()
        else:
            lambda_coef = torch.ones((), dtype=torch.float32, device=obs.device)
        bc_loss = (pi - actions).pow(2).sum(dim=-1).mean()
        actor_loss = -lambda_coef * q_pi.mean() + self.config.actor_bc_coef * bc_loss
        return actor_loss, bc_loss, q_pi, lambda_coef

    def update(self, batch: "dict[str, torch.Tensor]") -> dict[str, float]:
        obs = self._normalize_obs(batch["obs"])
        next_obs = self._normalize_obs(batch["next_obs"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_actions_data = batch.get("next_actions")
        if next_actions_data is None:
            raise ValueError(
                "ReBRAC requires batch['next_actions']; recollect the dataset or "
                "load it via TransitionReplay.from_npz so the field can be derived."
            )
        privileged_obs = batch.get("privileged_obs")
        next_privileged_obs = batch.get("next_privileged_obs")

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(
                self.q1_target(next_obs, next_actions, next_privileged_obs),
                self.q2_target(next_obs, next_actions, next_privileged_obs),
            )
            critic_penalty = (next_actions - next_actions_data).pow(2).sum(dim=-1)
            q_target = rewards + self.config.gamma * (1.0 - dones) * (
                target_q - self.config.critic_bc_coef * critic_penalty
            )

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
            actor_loss, bc_loss, q_pi, lambda_coef = self._actor_loss_terms(
                obs,
                actions,
                actor_privileged_obs,
            )
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()
            self._soft_update_targets()
            self._last_actor_metrics = {
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "mean_q": float(q_pi.detach().mean().item()),
                "lambda": float(lambda_coef.item()),
            }
            self._has_actor_metrics = True
        elif not self._has_actor_metrics:
            with torch.no_grad():
                actor_loss, bc_loss, q_pi, lambda_coef = self._actor_loss_terms(
                    obs,
                    actions,
                    actor_privileged_obs,
                )
            self._last_actor_metrics = {
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "mean_q": float(q_pi.mean().item()),
                "lambda": float(lambda_coef.item()),
            }
            self._has_actor_metrics = True

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "critic_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
            "actor_loss": self._last_actor_metrics["actor_loss"],
            "bc_loss": self._last_actor_metrics["bc_loss"],
            "mean_q": self._last_actor_metrics["mean_q"],
            "target_q": float(q_target.detach().mean().item()),
            "critic_penalty": float(critic_penalty.detach().mean().item()),
            "td_abs_error": float((q1_pred.detach() - q_target.detach()).abs().mean().item()),
            "lambda": self._last_actor_metrics["lambda"],
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

    def export_policy_payload(self) -> dict[str, Any]:
        require_torch()
        actor_state = {
            key: value.detach().cpu().clone()
            for key, value in self.actor.state_dict().items()
        }
        return {
            "algo": "rebrac",
            "config": asdict(self.config),
            "actor": actor_state,
            "obs_normalizer": self.obs_normalizer.state_dict(),
        }

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
