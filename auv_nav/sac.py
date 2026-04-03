"""Standard feedforward Soft Actor-Critic baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .networks import MLP, require_torch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
except ImportError:
    torch = None
    nn = None
    F = None
    Normal = None

_ModuleBase = nn.Module if nn is not None else object
_no_grad = torch.no_grad if torch is not None else (lambda: (lambda f: f))


@dataclass(slots=True)
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    init_alpha: float = 0.2
    target_entropy: float | None = None
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    batch_size: int = 256
    updates_per_step: int = 1
    grad_clip_norm: float = 10.0


class SquashedGaussianActor(_ModuleBase):
    def __init__(self, config: SACConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)

    def distribution(self, obs: "torch.Tensor") -> "Normal":
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(
            self.log_std(features),
            self.config.log_std_min,
            self.config.log_std_max,
        )
        return Normal(mean, torch.exp(log_std))

    def sample(
        self,
        obs: "torch.Tensor",
        deterministic: bool = False,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        dist = self.distribution(obs)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_prob = torch.zeros_like(action[..., 0])
        else:
            log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class QNetwork(_ModuleBase):
    def __init__(self, config: SACConfig) -> None:
        require_torch()
        super().__init__()
        self.q = MLP(config.obs_dim + config.action_dim, config.hidden_dim, 1)

    def forward(self, obs: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([obs, action], dim=-1)
        return self.q(x).squeeze(-1)


class SACAgent:
    def __init__(
        self,
        config: SACConfig,
        device: str | "torch.device" = "cpu",
    ) -> None:
        require_torch()
        self.config = config
        self.device = torch.device(device)

        if config.target_entropy is None:
            self.target_entropy = -float(config.action_dim)
        else:
            self.target_entropy = float(config.target_entropy)

        self.actor = SquashedGaussianActor(config).to(self.device)
        self.q1 = QNetwork(config).to(self.device)
        self.q2 = QNetwork(config).to(self.device)
        self.q1_target = QNetwork(config).to(self.device)
        self.q2_target = QNetwork(config).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(
            np.log(config.init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.update_count = 0

    @property
    def alpha(self) -> "torch.Tensor":
        return self.log_alpha.exp()

    def reset_policy_state(self) -> None:
        return None

    @_no_grad()
    def act(
        self,
        obs: np.ndarray,
        policy_state: None = None,
        deterministic: bool = False,
    ) -> "tuple[np.ndarray, None]":
        _ = policy_state
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(obs_t, deterministic=deterministic)
        return action[0].cpu().numpy().astype(np.float32), None

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for src, tgt in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)
            for src, tgt in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)

    def update(self, batch: "dict[str, torch.Tensor]") -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            q_target = rewards + self.config.gamma * (1.0 - dones) * q_next

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
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

        pi_action, log_prob = self.actor.sample(obs)
        q1_pi = self.q1(obs, pi_action)
        q2_pi = self.q2(obs, pi_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update_targets()
        self.update_count += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "mean_q": float(q_pi.detach().mean().item()),
            "mean_log_prob": float(log_prob.detach().mean().item()),
        }

    def save(self, path: str) -> None:
        require_torch()
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_count": self.update_count,
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        require_torch()
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
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
        if "alpha_opt" in payload:
            self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.update_count = int(payload.get("update_count", 0))
