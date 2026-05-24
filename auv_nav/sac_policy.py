"""SAC checkpoint as a behavior policy for offline data collection.

Wraps a trained ``SquashedGaussianActor`` so it satisfies the same minimal
interface as the rule-based baselines in :mod:`auv_nav.baselines`:

    policy.act(env, obs) -> np.ndarray

Unlike the rule-based baselines, the SAC actor was trained on the full
history-stacked observation and must receive that stacked obs at collect time.
The ``uses_stacked_obs = True`` attribute signals the collector to skip the
single-step unwrap path in :func:`scripts.collect_offline_data._collect_episode`.

This module is the only place in :mod:`auv_nav` that hard-imports torch beyond
:mod:`auv_nav.sac` / :mod:`auv_nav.networks`; ``baselines.py`` stays torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import PlanarRemusEnv
from .sac import SACConfig, SquashedGaussianActor


@dataclass
class SACCheckpointPolicy:
    """Thin behavior-policy wrapper around a frozen ``SquashedGaussianActor``.

    Attributes
    ----------
    actor
        The loaded ``SquashedGaussianActor`` in eval mode.
    device
        Torch device the actor lives on (typically CPU for collection).
    deterministic
        If True, ``act`` returns ``tanh(mean)``; else samples from the Gaussian.
    obs_dim, action_dim
        Cached from the actor's ``SACConfig`` for sanity-check use by the
        collector.
    uses_stacked_obs
        Signals the collector to pass the full history-stacked observation
        rather than the decoded single-step obs (always True for SAC).
    """

    actor: SquashedGaussianActor
    device: torch.device
    deterministic: bool
    obs_dim: int
    action_dim: int
    uses_stacked_obs: bool = True

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
    ) -> "SACCheckpointPolicy":
        """Load a SAC checkpoint and freeze the actor for inference.

        The checkpoint payload (written by :meth:`auv_nav.sac.SACAgent.save`)
        includes a ``"config"`` dict that is an ``asdict`` of ``SACConfig``.
        We filter that dict against the *current* ``SACConfig`` field set so
        ckpts saved by a newer SACConfig (with extra fields) still load.
        """
        torch_device = torch.device(device)
        # weights_only=False: ckpt payload includes nested dicts (optimizer
        # state, config); we trust our own training output.
        payload: dict[str, Any] = torch.load(
            str(ckpt_path), map_location=torch_device, weights_only=False
        )
        if "actor" not in payload or "config" not in payload:
            raise ValueError(
                f"Checkpoint {ckpt_path!s} is missing required keys "
                f"'actor'/'config'; got keys={sorted(payload.keys())}."
            )

        raw_cfg = dict(payload["config"])
        known = {f.name for f in fields(SACConfig)}
        filtered = {k: v for k, v in raw_cfg.items() if k in known}
        missing_required = {"obs_dim", "action_dim"} - filtered.keys()
        if missing_required:
            raise ValueError(
                f"Checkpoint config missing required SACConfig fields: "
                f"{sorted(missing_required)}."
            )
        cfg = SACConfig(**filtered)

        actor = SquashedGaussianActor(cfg).to(torch_device)
        actor.load_state_dict(payload["actor"])
        actor.eval()
        for param in actor.parameters():
            param.requires_grad_(False)

        return cls(
            actor=actor,
            device=torch_device,
            deterministic=bool(deterministic),
            obs_dim=int(cfg.obs_dim),
            action_dim=int(cfg.action_dim),
        )

    def act(
        self,
        env: PlanarRemusEnv | None,
        obs: np.ndarray,
    ) -> np.ndarray:
        """Return a single-step action for the given stacked observation.

        ``env`` is accepted to match the rule-based baseline signature but is
        not used; the SAC actor reads obs only.
        """
        _ = env
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 1:
            raise ValueError(
                f"SACCheckpointPolicy expects a 1-D obs vector; got shape={obs_arr.shape}."
            )
        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs dim {obs_arr.shape[0]} does not match actor obs_dim={self.obs_dim}; "
                f"check --probe-layout / --history-length / --objective matches ckpt training config."
            )
        obs_tensor = torch.from_numpy(obs_arr).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = self.actor.sample(obs_tensor, deterministic=self.deterministic)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)


def resolve_trainer_state_path(ckpt_path: str | Path) -> Path | None:
    """Autodetect ``trainer_state.json`` via the ``checkpoints/<X>`` ↔ ``experiments/<X>`` mirror.

    Returns the resolved path if it exists, otherwise None. The collector
    falls back to fail-fast on None unless ``--sac-skip-trainer-state-check``
    is set explicitly.
    """
    ckpt = Path(ckpt_path).resolve()
    parts = list(ckpt.parts)
    try:
        idx = parts.index("checkpoints")
    except ValueError:
        return None
    mirrored = list(parts)
    mirrored[idx] = "experiments"
    candidate = Path(*mirrored[:-1]) / "trainer_state.json"
    return candidate if candidate.exists() else None
