"""GRU Residual SAC agent with optional safety critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .networks import (
    MLP,
    RecurrentEncoder,
    require_torch,
)
from .baselines import CrossCurrentResidualPrior

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
class GRUResidualSACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    encoder_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    lambda_lr: float = 3e-4
    gamma: float = 0.995
    cost_gamma: float = 0.995
    tau: float = 0.005
    init_alpha: float = 0.2
    init_lambda: float = 1e-2
    target_entropy: float | None = None
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    burn_in: int = 8
    train_seq_len: int = 32
    batch_size: int = 32
    updates_per_step: int = 1
    grad_clip_norm: float = 10.0
    residual_l2_weight: float = 1e-3
    behavior_cloning_weight: float = 0.0
    behavior_cloning_decay_steps: int = 50_000
    use_residual_prior: bool = True
    use_safety_critic: bool = True
    cost_limit: float = 0.02  # average per-step cost target (same scale as env cost signal)


class ResidualGaussianActor(_ModuleBase):
    def __init__(self, config: GRUResidualSACConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.encoder = RecurrentEncoder(config.obs_dim, config.encoder_dim, config.hidden_dim)
        self.mean_head = MLP(config.hidden_dim, config.hidden_dim, config.action_dim)
        self.log_std_head = MLP(config.hidden_dim, config.hidden_dim, config.action_dim)

    def encode_sequence(
        self, obs_seq: "torch.Tensor", hidden: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        return self.encoder(obs_seq, hidden)

    def encode_step(
        self, obs: "torch.Tensor", hidden: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        return self.encoder.step(obs, hidden)

    def distribution_from_features(self, features: "torch.Tensor") -> "Normal":
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample_from_features(
        self,
        features: "torch.Tensor",
        prior_action: "torch.Tensor",
        deterministic: bool = False,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        dist = self.distribution_from_features(features)
        residual_pre_tanh = dist.mean if deterministic else dist.rsample()
        raw_residual = torch.tanh(residual_pre_tanh)
        
        # Safe residual_max to avoid log_prob explosion.
        # Clamping safe_prior to 0.99 ensures residual_max >= 0.01,
        # preventing the entropy term (-log(residual_max)) from exploding.
        safe_prior = prior_action.clamp(-0.99, 0.99)
        residual_max = (1.0 - safe_prior.abs()).clamp_min(1e-3)
        
        residual = raw_residual * residual_max
        action = (prior_action + residual).clamp(-1.0, 1.0)  # numerical safety only

        if deterministic:
            log_prob = torch.zeros_like(action[..., 0])
            base_log_prob = torch.zeros_like(action[..., 0])
        else:
            # base_log_prob is the log-prob in residual space before prior scaling.
            base_log_prob = dist.log_prob(residual_pre_tanh).sum(dim=-1)
            base_log_prob = base_log_prob - torch.log(1.0 - raw_residual.pow(2) + 1e-6).sum(dim=-1)

            # log_prob is the full log-prob in action space after prior scaling.
            log_prob = base_log_prob - torch.log(residual_max).sum(dim=-1)
            
        return action, log_prob, residual, base_log_prob


class RecurrentQNetwork(_ModuleBase):
    def __init__(self, config: GRUResidualSACConfig) -> None:
        require_torch()
        super().__init__()
        self.encoder = RecurrentEncoder(config.obs_dim, config.encoder_dim, config.hidden_dim)
        self.q_head = MLP(config.hidden_dim + config.action_dim, config.hidden_dim, 1)

    def encode_sequence(
        self, obs_seq: "torch.Tensor", hidden: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        return self.encoder(obs_seq, hidden)

    def q_from_features(self, features: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([features, action], dim=-1)
        return self.q_head(x).squeeze(-1)


class GRUResidualSACAgent:
    def __init__(
        self,
        config: GRUResidualSACConfig,
        prior: CrossCurrentResidualPrior | None = None,
        device: str | "torch.device" = "cpu",
    ) -> None:
        require_torch()
        self.config = config
        self.device = torch.device(device)
        self.prior = prior if config.use_residual_prior else None

        if config.target_entropy is None:
            self.target_entropy = -float(config.action_dim)
        else:
            self.target_entropy = float(config.target_entropy)

        self.actor = ResidualGaussianActor(config).to(self.device)
        self.q1 = RecurrentQNetwork(config).to(self.device)
        self.q2 = RecurrentQNetwork(config).to(self.device)
        self.q1_target = RecurrentQNetwork(config).to(self.device)
        self.q2_target = RecurrentQNetwork(config).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        if config.use_safety_critic:
            self.qc = RecurrentQNetwork(config).to(self.device)
            self.qc_target = RecurrentQNetwork(config).to(self.device)
            self.qc_target.load_state_dict(self.qc.state_dict())
        else:
            self.qc = None
            self.qc_target = None

        self.log_alpha = torch.tensor(
            np.log(config.init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        if config.use_safety_critic:
            self.log_lambda = torch.tensor(
                np.log(config.init_lambda), dtype=torch.float32, device=self.device, requires_grad=True
            )
        else:
            self.log_lambda = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        if config.use_safety_critic and self.qc is not None and self.log_lambda is not None:
            self.qc_opt = torch.optim.Adam(self.qc.parameters(), lr=config.critic_lr)
            self.lambda_opt = torch.optim.Adam([self.log_lambda], lr=config.lambda_lr)
        else:
            self.qc_opt = None
            self.lambda_opt = None
        self.update_count = 0

    @property
    def alpha(self) -> "torch.Tensor":
        return self.log_alpha.exp()

    @property
    def lagrange_lambda(self) -> "torch.Tensor":
        if self.log_lambda is None:
            return torch.tensor(0.0, device=self.device)
        return self.log_lambda.exp()

    def reset_policy_state(self) -> "torch.Tensor":
        return self.reset_hidden()

    def reset_hidden(self) -> "torch.Tensor":
        return torch.zeros(1, 1, self.config.hidden_dim, device=self.device)

    def _prior_action_tensor(self, obs: "torch.Tensor") -> "torch.Tensor":
        if self.prior is None:
            return torch.zeros(*obs.shape[:-1], self.config.action_dim, device=obs.device)
        return self.prior.action_tensor(obs)

    @_no_grad()
    def act(
        self,
        obs: np.ndarray,
        hidden: "torch.Tensor | None",
        deterministic: bool = False,
    ) -> "tuple[np.ndarray, torch.Tensor]":
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feat, hidden_out = self.actor.encode_step(obs_t, hidden)
        prior_action = self._prior_action_tensor(obs_t)
        action, _, _, _ = self.actor.sample_from_features(feat, prior_action, deterministic=deterministic)
        return action[0].cpu().numpy().astype(np.float32), hidden_out

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for src, tgt in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)
            for src, tgt in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)
            if self.qc is not None and self.qc_target is not None:
                for src, tgt in zip(self.qc.parameters(), self.qc_target.parameters(), strict=True):
                    tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)

    def current_behavior_cloning_weight(self) -> float:
        base = float(self.config.behavior_cloning_weight)
        if base <= 0.0:
            return 0.0
        decay_steps = max(1, int(self.config.behavior_cloning_decay_steps))
        ratio = max(0.0, 1.0 - (self.update_count / decay_steps))
        return base * ratio

    def update(self, batch: "dict[str, torch.Tensor]") -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        dones = batch["dones"]

        burn_in = self.config.burn_in
        obs_t = obs[:, :-1]
        obs_tp1 = obs[:, 1:]
        act_t = actions
        rew_t = rewards
        cost_t = costs
        done_t = dones

        q1_feat, _ = self.q1.encode_sequence(obs)
        q2_feat, _ = self.q2.encode_sequence(obs)
        with torch.no_grad():
            q1_target_feat, _ = self.q1_target.encode_sequence(obs)
            q2_target_feat, _ = self.q2_target.encode_sequence(obs)
        actor_feat, _ = self.actor.encode_sequence(obs)
        if self.qc is not None and self.qc_target is not None:
            qc_feat, _ = self.qc.encode_sequence(obs)
            with torch.no_grad():
                qc_target_feat, _ = self.qc_target.encode_sequence(obs)
        else:
            qc_feat = None
            qc_target_feat = None

        q1_t = q1_feat[:, :-1][:, burn_in:]
        q2_t = q2_feat[:, :-1][:, burn_in:]
        q1_tp1_target = q1_target_feat[:, 1:][:, burn_in:]
        q2_tp1_target = q2_target_feat[:, 1:][:, burn_in:]
        actor_t = actor_feat[:, :-1][:, burn_in:]
        actor_tp1 = actor_feat[:, 1:][:, burn_in:]

        obs_t = obs_t[:, burn_in:]
        obs_tp1 = obs_tp1[:, burn_in:]
        act_t = act_t[:, burn_in:]
        rew_t = rew_t[:, burn_in:]
        cost_t = cost_t[:, burn_in:]
        done_t = done_t[:, burn_in:]

        prior_t = self._prior_action_tensor(obs_t)
        prior_tp1 = self._prior_action_tensor(obs_tp1)

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample_from_features(actor_tp1, prior_tp1)
            q1_next = self.q1_target.q_from_features(q1_tp1_target, next_action)
            q2_next = self.q2_target.q_from_features(q2_tp1_target, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            q_target = rew_t + self.config.gamma * (1.0 - done_t) * q_next

        q1_pred = self.q1.q_from_features(q1_t, act_t)
        q2_pred = self.q2.q_from_features(q2_t, act_t)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        qc_loss = torch.zeros_like(q1_loss)

        if self.qc is not None and self.qc_target is not None and qc_feat is not None and qc_target_feat is not None:
            qc_t = qc_feat[:, :-1][:, burn_in:]
            qc_tp1_target = qc_target_feat[:, 1:][:, burn_in:]
            with torch.no_grad():
                # For safety critic, entropy doesn't usually apply, but we use the same next_action
                qc_next = self.qc_target.q_from_features(qc_tp1_target, next_action)
                qc_target = cost_t + self.config.cost_gamma * (1.0 - done_t) * qc_next
            qc_pred = self.qc.q_from_features(qc_t, act_t)
            qc_loss = F.mse_loss(qc_pred, qc_target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.grad_clip_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.grad_clip_norm)
        self.q2_opt.step()

        if self.qc_opt is not None and self.qc is not None:
            self.qc_opt.zero_grad(set_to_none=True)
            qc_loss.backward()
            nn.utils.clip_grad_norm_(self.qc.parameters(), self.config.grad_clip_norm)
            self.qc_opt.step()

        pi_action, log_prob, residual, base_log_prob = self.actor.sample_from_features(actor_t, prior_t)
        q1_pi = self.q1.q_from_features(q1_t.detach(), pi_action)
        q2_pi = self.q2.q_from_features(q2_t.detach(), pi_action)
        q_pi = torch.min(q1_pi, q2_pi)

        # Use the same action-space entropy objective in the Bellman target, actor update,
        # and temperature update so recurrent training remains self-consistent.
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        residual_l2 = residual.pow(2).mean()

        if self.qc is not None and qc_feat is not None:
            qc_pi = self.qc.q_from_features(qc_feat[:, :-1][:, burn_in:].detach(), pi_action)
            safety_penalty = self.lagrange_lambda.detach() * qc_pi.mean()
        else:
            qc_pi = torch.zeros_like(q_pi)
            safety_penalty = torch.zeros_like(actor_loss)

        actor_total_loss = (
            actor_loss
            + safety_penalty
            + self.config.residual_l2_weight * residual_l2
        )

        self.actor_opt.zero_grad(set_to_none=True)
        actor_total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        lambda_loss = torch.zeros_like(alpha_loss)
        if self.lambda_opt is not None and self.log_lambda is not None and self.qc is not None:
            mean_cost_q = qc_pi.detach().mean()
            mean_cost_per_step = mean_cost_q * (1.0 - self.config.cost_gamma)
            lambda_loss = -(self.log_lambda * (mean_cost_per_step - self.config.cost_limit))
            self.lambda_opt.zero_grad(set_to_none=True)
            lambda_loss.backward()
            self.lambda_opt.step()
            with torch.no_grad():
                self.log_lambda.clamp_(max=2.0)  # lambda <= e^2 ≈ 7.4

        self.soft_update_targets()
        self.update_count += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "actor_total_loss": float(actor_total_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "qc_loss": float(qc_loss.item()),
            "lambda_loss": float(lambda_loss.item()),
            "lagrange_lambda": float(self.lagrange_lambda.detach().item()),
            "mean_cost_q": float(qc_pi.detach().mean().item()),
            "mean_cost_per_step": float(qc_pi.detach().mean().item() * (1.0 - self.config.cost_gamma)),
            "mean_q": float(q_pi.detach().mean().item()),
            "mean_log_prob": float(log_prob.detach().mean().item()),
            "mean_base_log_prob": float(base_log_prob.detach().mean().item()),
            "residual_l2": float(residual_l2.detach().item()),
        }

    def save(self, path: str) -> None:
        require_torch()
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "qc": self.qc.state_dict() if self.qc is not None else None,
            "qc_target": self.qc_target.state_dict() if self.qc_target is not None else None,
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "qc_opt": self.qc_opt.state_dict() if self.qc_opt is not None else None,
            "alpha_opt": self.alpha_opt.state_dict(),
            "lambda_opt": self.lambda_opt.state_dict() if self.lambda_opt is not None else None,
            "log_alpha": self.log_alpha.detach().cpu(),
            "log_lambda": self.log_lambda.detach().cpu() if self.log_lambda is not None else None,
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
        if self.qc is not None and payload.get("qc") is not None:
            self.qc.load_state_dict(payload["qc"])
        if self.qc_target is not None and payload.get("qc_target") is not None:
            self.qc_target.load_state_dict(payload["qc_target"])
        if "actor_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
        if "q1_opt" in payload:
            self.q1_opt.load_state_dict(payload["q1_opt"])
        if "q2_opt" in payload:
            self.q2_opt.load_state_dict(payload["q2_opt"])
        if self.qc_opt is not None and payload.get("qc_opt") is not None:
            self.qc_opt.load_state_dict(payload["qc_opt"])
        if "alpha_opt" in payload:
            self.alpha_opt.load_state_dict(payload["alpha_opt"])
        if self.lambda_opt is not None and payload.get("lambda_opt") is not None:
            self.lambda_opt.load_state_dict(payload["lambda_opt"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        if self.log_lambda is not None and payload.get("log_lambda") is not None:
            self.log_lambda.data.copy_(payload["log_lambda"].to(self.device))
        self.update_count = int(payload.get("update_count", 0))
