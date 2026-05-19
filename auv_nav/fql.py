r"""Flow Q-Learning (FQL) agent for offline reinforcement learning.

Reference: Park, Li, Levine, "Flow Q-Learning", ICML 2025.

Design contract: see ``docs/fql_pytorch_port_design.md`` (v1.0, 2026-05-19).

Key design choices anchored in the design doc:

* **Critic is inherited from ReBRAC / SAC** (``auv_nav.sac.QNetwork``) — the
  paper claim "FQL beats ReBRAC on multi-modal sub-optimal data" stays clean
  only if the critic is held fixed across the two algorithms.
* **Behavior teacher** is a flow-matching velocity field
  :math:`v_\theta(x_t, t \mid s)` trained with linear-interpolation paths.
* **Student** is a 1-step deterministic actor distilled from the teacher's
  ODE integration plus Q-guidance (ReBRAC Finding iii Q-normalisation).
* **No critic-side BC penalty** (FQL puts the entire BC anchor in the
  student loss via teacher distillation).
* **No asymmetric critic** (sensor-envelope chapter is out of scope for the
  FQL paper).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .networks import MLP, polyak_update, require_torch
from .sac import QNetwork
from .td3bc import ObservationNormalizer, ObservationNormalizerState

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
class FQLConfig:
    """Configuration for the FQL agent.

    Field layout mirrors :class:`auv_nav.rebrac.ReBRACConfig` so the critic
    (``QNetwork``) — which is shared via the ``_resolve_*`` helpers in
    ``auv_nav/sac.py`` — picks up identical regularisation defaults.
    """

    # --- core dimensions (required) ---
    obs_dim: int
    action_dim: int

    # --- network architecture ---
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # --- FQL-specific ---
    flow_steps: int = 10
    flow_time_embed_dim: int = 32
    distill_alpha_bc: float = 1.0

    # --- learning rates ---
    teacher_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # --- RL hyperparams (TD3-style) ---
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256
    grad_clip_norm: float = 10.0

    # --- regularisation (inherit ReBRAC Finding iv) ---
    critic_use_layernorm: bool = True
    actor_use_layernorm: bool = False
    teacher_use_layernorm: bool = False
    dropout_rate: float = 0.0

    # --- Q normalisation (inherit ReBRAC Finding iii) ---
    normalizer_eps: float = 1e-3
    normalize_q: bool = True


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal, DDPM-style)
# ---------------------------------------------------------------------------


class TimeEmbed(_ModuleBase):
    """Sinusoidal time embedding for flow-matching diffusion time.

    Output ``embed`` is the concatenation of ``sin(t * freq_k)`` and
    ``cos(t * freq_k)`` for ``k = 0 .. half - 1`` where
    ``freq_k = exp(-log(10000) * k / half)``.

    Parameters
    ----------
    embed_dim:
        Total embedding dimension. Must be even.
    """

    def __init__(self, embed_dim: int) -> None:
        require_torch()
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"time_embed_dim must be even, got {embed_dim}"
            )
        if embed_dim <= 0:
            raise ValueError(
                f"time_embed_dim must be positive, got {embed_dim}"
            )
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / float(half)
        )
        # registered buffer so it moves with .to(device) but is not learnable
        self.register_buffer("freqs", freqs)

    def forward(self, t: "torch.Tensor") -> "torch.Tensor":
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError(
                "TimeEmbed expects t of shape [B] or [B, 1], "
                f"got shape {tuple(t.shape)}"
            )
        # t: [B, 1], freqs: [half] → broadcast to [B, half]
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Flow matching teacher
# ---------------------------------------------------------------------------


class FlowMatchingTeacher(_ModuleBase):
    """Velocity field ``v_theta(x_t, t | s)`` parameterising the behaviour
    distribution as a flow from ``N(0, I)`` to the data action.

    Linear interpolation path::

        x_t = (1 - t) * x_0 + t * a,   v_target = a - x_0,
        x_0 ~ N(0, I),  a is the data action.

    Notes
    -----
    The output is **unbounded** (no tanh) on purpose; the student learns the
    bounded ``[-1, 1]`` action via its own tanh head plus the BC distillation
    term.
    """

    def __init__(self, config: FQLConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.time_embed = TimeEmbed(config.flow_time_embed_dim)
        input_dim = (
            config.action_dim + config.obs_dim + config.flow_time_embed_dim
        )
        self.net = MLP(
            input_dim,
            config.hidden_dim,
            config.action_dim,
            use_layernorm=config.teacher_use_layernorm,
            dropout_rate=config.dropout_rate,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        x_t: "torch.Tensor",  # [B, A]
        t: "torch.Tensor",    # [B, 1] or [B]
        obs: "torch.Tensor",  # [B, O]
    ) -> "torch.Tensor":      # [B, A]
        # Validate the one shape mismatch PyTorch silently broadcasts past:
        # torch.cat would happily merge [B1, *] with [B2, *] in dim=-1 by
        # treating them as a single dim-0 mismatch error far from here.
        if x_t.shape[0] != obs.shape[0]:
            raise ValueError(
                "x_t and obs must have the same batch size, "
                f"got {x_t.shape[0]} vs {obs.shape[0]}"
            )
        t_embed = self.time_embed(t)
        x_in = torch.cat([x_t, obs, t_embed], dim=-1)
        return self.net(x_in)

    @_no_grad()
    def integrate(
        self,
        obs: "torch.Tensor",
        n_steps: int | None = None,
    ) -> "torch.Tensor":
        """Euler ODE integration from ``x_0 ~ N(0, I)`` to ``x_1``."""
        if n_steps is None:
            n_steps = self.config.flow_steps
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if obs.ndim != 2:
            raise ValueError(
                f"obs must have shape [B, obs_dim], got {tuple(obs.shape)}"
            )
        batch = obs.shape[0]
        x = torch.randn(batch, self.config.action_dim, device=obs.device)
        dt = 1.0 / float(n_steps)
        for i in range(n_steps):
            t = torch.full(
                (batch, 1),
                fill_value=float(i) * dt,
                device=obs.device,
                dtype=x.dtype,
            )
            v = self.forward(x, t, obs)
            x = x + dt * v
        return x


# ---------------------------------------------------------------------------
# Distilled student (1-step deterministic actor)
# ---------------------------------------------------------------------------


class DistilledStudent(_ModuleBase):
    r"""One-step deterministic actor :math:`\pi_\phi(s) \to a \in [-1, 1]^A`.

    Mirrors :class:`auv_nav.rebrac.DeterministicActor` 1:1; only the config
    type differs.
    """

    def __init__(self, config: FQLConfig) -> None:
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


# ---------------------------------------------------------------------------
# Policy wrapper for evaluation workers
# ---------------------------------------------------------------------------


class FQLPolicy:
    """Lightweight deterministic policy wrapper for evaluation workers.

    Only the student network is loaded — teacher and critics are not needed
    at evaluation time, which is the central efficiency claim of FQL.
    """

    def __init__(
        self,
        config: FQLConfig,
        *,
        student_state: dict[str, Any],
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
        self.student = DistilledStudent(config).to(self.device)
        self.student.load_state_dict(student_state)
        self.student.eval()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        device: str | "torch.device" = "cpu",
    ) -> "FQLPolicy":
        config = FQLConfig(**payload["config"])
        normalizer_state = payload.get("obs_normalizer")
        obs_normalizer: ObservationNormalizer | None = None
        if normalizer_state is not None:
            obs_normalizer = ObservationNormalizer(
                ObservationNormalizerState(
                    mean=np.asarray(normalizer_state["mean"], dtype=np.float32),
                    std=np.asarray(normalizer_state["std"], dtype=np.float32),
                    eps=float(
                        normalizer_state.get("eps", config.normalizer_eps)
                    ),
                ),
                device=device,
            )
        return cls(
            config,
            student_state=payload["student"],
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
        action = self.student(self.obs_normalizer.normalize_tensor(obs_t))
        action_np = action.cpu().numpy().astype(np.float32)
        if not is_batched:
            action_np = action_np[0]
        return action_np, None


# ---------------------------------------------------------------------------
# FQL agent (training)
# ---------------------------------------------------------------------------


class FQLAgent:
    """Flow Q-Learning offline RL agent.

    See :class:`auv_nav.rebrac.ReBRACAgent` for the sibling implementation
    that this class mirrors in API surface (so ``scripts/train_offline.py``,
    checkpoint pipeline, and ``evaluate_offline_policy_parallel`` can dispatch
    via :func:`auv_nav.offline_registry.make_agent`).
    """

    def __init__(
        self,
        config: FQLConfig,
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

        # Teacher: trained on flow-matching loss every step.
        self.teacher = FlowMatchingTeacher(config).to(self.device)

        # Student: deterministic 1-step actor.
        self.student = DistilledStudent(config).to(self.device)
        self.student_target = DistilledStudent(config).to(self.device)
        self.student_target.load_state_dict(self.student.state_dict())

        # Critics: inherited from ReBRAC / SAC (vanilla, no privileged input).
        self.q1 = QNetwork(config).to(self.device)
        self.q2 = QNetwork(config).to(self.device)
        self.q1_target = QNetwork(config).to(self.device)
        self.q2_target = QNetwork(config).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.teacher_opt = torch.optim.Adam(
            self.teacher.parameters(), lr=config.teacher_lr
        )
        self.student_opt = torch.optim.Adam(
            self.student.parameters(), lr=config.actor_lr
        )
        self.q1_opt = torch.optim.Adam(
            self.q1.parameters(), lr=config.critic_lr
        )
        self.q2_opt = torch.optim.Adam(
            self.q2.parameters(), lr=config.critic_lr
        )

        self.update_count = 0
        self._last_actor_metrics: dict[str, float] = {
            "actor_loss": float("nan"),
            "bc_loss": float("nan"),
            "mean_q": float("nan"),
            "lambda": float("nan"),
            "student_grad_norm": float("nan"),
        }
        self._has_actor_metrics = False

    # --- helpers -------------------------------------------------------

    def reset_policy_state(self) -> None:
        return None

    def _normalize_obs(self, obs: "torch.Tensor") -> "torch.Tensor":
        return self.obs_normalizer.normalize_tensor(obs)

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        polyak_update(self.student, self.student_target, tau)
        polyak_update(self.q1, self.q1_target, tau)
        polyak_update(self.q2, self.q2_target, tau)

    def _teacher_loss(
        self,
        obs: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> "torch.Tensor":
        """Flow-matching MSE on the linear interpolation path."""
        batch = actions.shape[0]
        t = torch.rand(batch, 1, device=actions.device, dtype=actions.dtype)
        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t) * x_0 + t * actions
        v_target = actions - x_0
        v_pred = self.teacher(x_t, t, obs)
        return F.mse_loss(v_pred, v_target)

    def _compute_actor_quantities(
        self,
        obs: "torch.Tensor",
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Compute the student-actor objective and its components.

        Returns ``(actor_loss, bc_loss, q_min, lambda_coef)``.  Caller controls
        whether this runs inside ``torch.no_grad()`` (sentinel path) or with
        gradients (training path).
        """
        a_teacher = self.teacher.integrate(
            obs, n_steps=self.config.flow_steps
        )  # always @no_grad inside FlowMatchingTeacher.integrate
        a_student = self.student(obs)
        q_min = torch.min(
            self.q1(obs, a_student), self.q2(obs, a_student)
        )
        if self.config.normalize_q:
            lambda_coef = (
                q_min.abs().mean().detach().clamp_min(1e-6).reciprocal()
            )
        else:
            lambda_coef = torch.ones(
                (), dtype=torch.float32, device=obs.device
            )
        bc_loss = (a_student - a_teacher).pow(2).mean()
        actor_loss = (
            -lambda_coef * q_min.mean()
            + self.config.distill_alpha_bc * bc_loss
        )
        return actor_loss, bc_loss, q_min, lambda_coef

    # --- act -----------------------------------------------------------

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
        action = self.student(self._normalize_obs(obs_t))
        action_np = action.cpu().numpy().astype(np.float32)
        if not is_batched:
            action_np = action_np[0]
        return action_np, None

    # --- update --------------------------------------------------------

    def update(
        self,
        batch: "dict[str, torch.Tensor]",
    ) -> dict[str, float]:
        obs = self._normalize_obs(batch["obs"])
        next_obs = self._normalize_obs(batch["next_obs"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # --- 1. Teacher update (flow-matching) -------------------------
        teacher_loss = self._teacher_loss(obs, actions)
        self.teacher_opt.zero_grad(set_to_none=True)
        teacher_loss.backward()
        teacher_grad_norm = nn.utils.clip_grad_norm_(
            self.teacher.parameters(), self.config.grad_clip_norm
        )
        self.teacher_opt.step()

        # --- 2. Critic update (TD3-style, no critic-side BC) -----------
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.policy_noise
            noise = noise.clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = (
                self.student_target(next_obs) + noise
            ).clamp(-1.0, 1.0)
            target_q = torch.min(
                self.q1_target(next_obs, next_actions),
                self.q2_target(next_obs, next_actions),
            )
            q_target = rewards + self.config.gamma * (1.0 - dones) * target_q

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_grad_norm = nn.utils.clip_grad_norm_(
            self.q1.parameters(), self.config.grad_clip_norm
        )
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_grad_norm = nn.utils.clip_grad_norm_(
            self.q2.parameters(), self.config.grad_clip_norm
        )
        self.q2_opt.step()

        # --- 3. Student update (every policy_freq steps) ---------------
        self.update_count += 1
        should_update_actor = (
            self.update_count % max(1, self.config.policy_freq)
        ) == 0

        if should_update_actor:
            actor_loss, bc_loss, q_min, lambda_coef = (
                self._compute_actor_quantities(obs)
            )
            self.student_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            student_grad_norm = nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.grad_clip_norm
            )
            self.student_opt.step()
            self._soft_update_targets()
            self._last_actor_metrics = {
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "mean_q": float(q_min.detach().mean().item()),
                "lambda": float(lambda_coef.item()),
                "student_grad_norm": float(student_grad_norm.item()),
            }
            self._has_actor_metrics = True
        elif not self._has_actor_metrics:
            # Sentinel populated only on the first ``update()`` call when
            # ``policy_freq > 1``, so CSV columns are non-NaN immediately.
            with torch.no_grad():
                actor_loss, bc_loss, q_min, lambda_coef = (
                    self._compute_actor_quantities(obs)
                )
            self._last_actor_metrics = {
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "mean_q": float(q_min.mean().item()),
                "lambda": float(lambda_coef.item()),
                "student_grad_norm": float("nan"),
            }
            self._has_actor_metrics = True

        return {
            "loss_flow": float(teacher_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "critic_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
            "target_q": float(q_target.detach().mean().item()),
            "td_abs_error": float(
                (q1_pred.detach() - q_target.detach()).abs().mean().item()
            ),
            "actor_loss": self._last_actor_metrics["actor_loss"],
            "bc_loss": self._last_actor_metrics["bc_loss"],
            "mean_q": self._last_actor_metrics["mean_q"],
            "lambda": self._last_actor_metrics["lambda"],
            "teacher_grad_norm": float(teacher_grad_norm.item()),
            "q1_grad_norm": float(q1_grad_norm.item()),
            "q2_grad_norm": float(q2_grad_norm.item()),
            "student_grad_norm": self._last_actor_metrics["student_grad_norm"],
            "policy_updated": float(1.0 if should_update_actor else 0.0),
        }

    # --- save / load ---------------------------------------------------

    def save(self, path: str) -> None:
        require_torch()
        payload = {
            "algorithm": "fql",
            "config": asdict(self.config),
            "teacher": self.teacher.state_dict(),
            "student": self.student.state_dict(),
            "student_target": self.student_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "teacher_opt": self.teacher_opt.state_dict(),
            "student_opt": self.student_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "update_count": self.update_count,
            "obs_normalizer": self.obs_normalizer.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        require_torch()
        payload = torch.load(path, map_location=self.device)
        self.teacher.load_state_dict(payload["teacher"])
        self.student.load_state_dict(payload["student"])
        self.student_target.load_state_dict(payload["student_target"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        if "teacher_opt" in payload:
            self.teacher_opt.load_state_dict(payload["teacher_opt"])
        if "student_opt" in payload:
            self.student_opt.load_state_dict(payload["student_opt"])
        if "q1_opt" in payload:
            self.q1_opt.load_state_dict(payload["q1_opt"])
        if "q2_opt" in payload:
            self.q2_opt.load_state_dict(payload["q2_opt"])
        self.update_count = int(payload.get("update_count", 0))

        normalizer_state = payload.get("obs_normalizer")
        if normalizer_state is not None:
            self.obs_normalizer = ObservationNormalizer(
                ObservationNormalizerState(
                    mean=np.asarray(
                        normalizer_state["mean"], dtype=np.float32
                    ),
                    std=np.asarray(normalizer_state["std"], dtype=np.float32),
                    eps=float(
                        normalizer_state.get(
                            "eps", self.config.normalizer_eps
                        )
                    ),
                ),
                device=self.device,
            )

    def export_policy_payload(self) -> dict[str, Any]:
        require_torch()
        student_state = {
            key: value.detach().cpu().clone()
            for key, value in self.student.state_dict().items()
        }
        return {
            "algo": "fql",
            "config": asdict(self.config),
            "student": student_state,
            "obs_normalizer": self.obs_normalizer.state_dict(),
        }
