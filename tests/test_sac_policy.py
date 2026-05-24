"""Tests for the SAC checkpoint collector adapter.

Five tests, by spec:
    (a) shape / range with the locally checked-in SAC ckpt
    (b) deterministic vs stochastic semantics
    (c) Layer 1 obs_dim mismatch fail-fast
    (d) Layer 2 trainer_state protocol-mismatch fail-fast
    (e) End-to-end collector run with stacked obs (regression for the
        obs-unwrap bug that would otherwise feed single-step obs to the actor).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from auv_nav.sac_policy import SACCheckpointPolicy
from scripts.collect_offline_data import collect


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_CKPT = (
    _REPO_ROOT
    / "checkpoints"
    / "local_profiles"
    / "arrival_v2_cross_u10_12k_seed46"
    / "agent_best.pt"
)
_LOCAL_TRAINER_STATE = (
    _REPO_ROOT
    / "experiments"
    / "local_profiles"
    / "arrival_v2_cross_u10_12k_seed46"
    / "trainer_state.json"
)
_WAKE_DUMMY = _REPO_ROOT / "wake_data" / "wake_dummy_roi.npy"

_HAS_LOCAL_CKPT = _LOCAL_CKPT.exists() and _LOCAL_TRAINER_STATE.exists()
_skip_if_no_local_ckpt = pytest.mark.skipif(
    not _HAS_LOCAL_CKPT,
    reason="local SAC ckpt + trainer_state.json not available",
)


def _make_sac_args(
    output_dir: Path,
    *,
    sac_ckpt: Path,
    sac_trainer_state: Path | None = None,
    sac_skip_trainer_state_check: bool = False,
    episodes: int = 2,
    deterministic: bool = True,
) -> SimpleNamespace:
    """Build the argparse.Namespace equivalent used by collect() in SAC mode."""
    return SimpleNamespace(
        policy=None,
        policy_mixture=None,
        flow=_WAKE_DUMMY,
        probe_layout="s0",
        history_length=4,
        difficulty=None,
        task_geometry="cross_stream",
        target_speed=1.5,
        objective="arrival_v2",
        energy_cost_gain=None,
        safety_cost_gain=None,
        episodes=episodes,
        seed=0,
        action_noise_std=0.0,
        action_noise_clip=0.5,
        num_workers=1,
        output_dir=str(output_dir),
        sac_ckpt=str(sac_ckpt),
        sac_deterministic=deterministic,
        sac_device="cpu",
        sac_trainer_state=str(sac_trainer_state) if sac_trainer_state else None,
        sac_skip_trainer_state_check=sac_skip_trainer_state_check,
    )


# (a) Action shape / range -------------------------------------------------
@_skip_if_no_local_ckpt
def test_sac_policy_action_shape_and_range() -> None:
    policy = SACCheckpointPolicy.from_checkpoint(
        _LOCAL_CKPT, device="cpu", deterministic=True
    )
    obs = np.zeros(policy.obs_dim, dtype=np.float32)
    action = policy.act(env=None, obs=obs)
    assert action.shape == (policy.action_dim,)
    assert action.dtype == np.float32
    assert ((action >= -1.0) & (action <= 1.0)).all()
    assert policy.obs_dim == 48
    assert policy.action_dim == 2
    assert policy.uses_stacked_obs is True


# (b) Deterministic vs stochastic semantics --------------------------------
@_skip_if_no_local_ckpt
def test_deterministic_stable_stochastic_varies() -> None:
    det = SACCheckpointPolicy.from_checkpoint(_LOCAL_CKPT, deterministic=True)
    sto = SACCheckpointPolicy.from_checkpoint(_LOCAL_CKPT, deterministic=False)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=det.obs_dim).astype(np.float32)

    # Deterministic: identical regardless of torch RNG.
    torch.manual_seed(0)
    a_det_0 = det.act(None, obs)
    torch.manual_seed(7)
    a_det_1 = det.act(None, obs)
    assert np.allclose(a_det_0, a_det_1)

    # Stochastic: different samples for different torch RNG seeds.
    torch.manual_seed(0)
    a_sto_0 = sto.act(None, obs)
    torch.manual_seed(7)
    a_sto_1 = sto.act(None, obs)
    assert not np.allclose(a_sto_0, a_sto_1, atol=1e-4)


# (c) Layer 1 obs_dim mismatch fail-fast -----------------------------------
def test_layer1_obs_dim_mismatch_raises(tmp_path: Path) -> None:
    fake_ckpt_path = tmp_path / "fake_obs_dim_99.pt"
    fake_payload: dict[str, Any] = {
        "actor": {},
        "config": {"obs_dim": 99, "action_dim": 2},
    }
    torch.save(fake_payload, str(fake_ckpt_path))

    # s0 / history=1 / arrival_v1 → env obs_dim = 10; ckpt claims 99 → must raise.
    args = SimpleNamespace(
        policy=None,
        policy_mixture=None,
        flow=_WAKE_DUMMY,
        probe_layout="s0",
        history_length=1,
        difficulty=None,
        task_geometry="downstream",
        target_speed=1.0,
        objective="arrival_v1",
        energy_cost_gain=None,
        safety_cost_gain=None,
        episodes=1,
        seed=0,
        action_noise_std=0.0,
        action_noise_clip=0.5,
        num_workers=1,
        output_dir=str(tmp_path / "out"),
        sac_ckpt=str(fake_ckpt_path),
        sac_deterministic=True,
        sac_device="cpu",
        sac_trainer_state=None,
        sac_skip_trainer_state_check=True,  # bypass Layer 2 — testing Layer 1 only
    )
    with pytest.raises(ValueError, match="obs_dim"):
        collect(args)


# (d) Layer 2 trainer_state mismatch fail-fast -----------------------------
@_skip_if_no_local_ckpt
def test_layer2_trainer_state_mismatch_raises(tmp_path: Path) -> None:
    # Fake trainer_state with wrong probe_layout (claims s1, CLI says s0).
    fake_trainer_state = {
        "algorithm": "sac",
        "probe_layout": "s1",  # mismatch
        "history_length": 4,
        "reward_objective": "arrival_v2",
    }
    fake_ts_path = tmp_path / "fake_trainer_state.json"
    fake_ts_path.write_text(json.dumps(fake_trainer_state), encoding="utf-8")

    args = _make_sac_args(
        tmp_path / "out",
        sac_ckpt=_LOCAL_CKPT,
        sac_trainer_state=fake_ts_path,
    )
    with pytest.raises(ValueError, match="probe_layout"):
        collect(args)


# (e) End-to-end stacked-obs path regression -------------------------------
@_skip_if_no_local_ckpt
def test_collector_e2e_passes_stacked_obs(tmp_path: Path) -> None:
    output_dir = tmp_path / "sac_e2e"
    args = _make_sac_args(
        output_dir,
        sac_ckpt=_LOCAL_CKPT,
        sac_trainer_state=_LOCAL_TRAINER_STATE,  # explicit to avoid autodetect dependency
        episodes=2,
    )
    collect(args)

    with np.load(output_dir / "transitions.npz") as data:
        obs = data["obs"].copy()
        actions = data["actions"].copy()
        behavior_codes = data["behavior_policy_codes"].copy()

    # The critical regression check: obs second-dim must match the ckpt's
    # obs_dim (48). If _collect_episode stripped to single-step obs (12 = s0
    # arrival_v2 per-frame), the actor would have received the wrong shape
    # and either crashed or silently produced garbage actions.
    assert obs.shape[1] == 48, (
        f"Expected stacked-obs dim 48 (s0/h=4 arrival_v2), got {obs.shape[1]}. "
        f"This usually means _collect_episode handed the actor single_obs."
    )
    assert actions.shape[1] == 2
    assert ((actions >= -1.0) & (actions <= 1.0)).all()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["behavior_source"] == "sac_checkpoint"
    assert metadata["policy"] == "sac_checkpoint"
    assert metadata["sac_deterministic"] is True
    assert metadata["sac_device"] == "cpu"
    assert metadata["sac_agent_config"]["obs_dim"] == 48
    assert metadata["sac_trainer_state_snapshot"]["probe_layout"] == "s0"
    assert metadata["sac_trainer_state_snapshot"]["history_length"] == 4
    assert metadata["sac_trainer_state_snapshot"]["reward_objective"] == "arrival_v2"

    sac_code = metadata["behavior_policy_vocab"]["sac_checkpoint"]
    assert (behavior_codes == sac_code).all()
