"""Unit tests for the sanity card writer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.write_sanity_card import (
    derive_sanity_card,
    write_sanity_card,
)


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Synthetic 3-episode dataset (lengths 5, 7, 4 = 16 transitions)."""
    n = 16
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, 10), dtype=np.float32)
    actions = rng.standard_normal((n, 2), dtype=np.float32)
    rewards = rng.standard_normal(n, dtype=np.float32)
    next_obs = rng.standard_normal((n, 10), dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[[4, 11, 15]] = 1.0
    privileged_obs = rng.standard_normal((n, 2), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "transitions.npz",
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        privileged_obs=privileged_obs,
    )
    metadata = {
        "policy": "crosscomp",
        "obs_dim": 10,
        "action_dim": 2,
        "privileged_obs_dim": 2,
        "num_episodes": 3,
        "num_transitions": 16,
        "success_rate": 0.667,
        "mean_return": 12.5,
        "std_return": 3.1,
        "mean_episode_length": 5.33,
        "flow_path": "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_derive_card_pulls_seven_fields(tmp_dataset_dir: Path) -> None:
    card = derive_sanity_card(tmp_dataset_dir)
    assert card["collector_success_rate"] == pytest.approx(0.667)
    assert card["collector_mean_return"] == pytest.approx(12.5)
    assert card["episode_length_mean"] == pytest.approx((5 + 7 + 4) / 3)
    assert card["episode_length_std"] > 0
    assert card["obs_dim"] == 10
    assert card["n_transitions"] == 16
    assert card["privileged_obs_present"] is True
    assert card["flow_file"] == "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"


def test_obs_dim_consistency_check(tmp_dataset_dir: Path) -> None:
    """Card writer must verify obs_dim matches probe_layout when available.

    Default fixture has no `history_length` in metadata → treated as 1, so
    obs_dim=10 matches the s0 base dim.
    """
    card = derive_sanity_card(tmp_dataset_dir, expected_probe_layout="s0")
    assert card["obs_dim_matches_probe_layout"] is True
    assert card["history_length"] == 1
    assert card["expected_obs_dim"] == 10
    card_s1 = derive_sanity_card(tmp_dataset_dir, expected_probe_layout="s1")
    assert card_s1["obs_dim_matches_probe_layout"] is False


def test_obs_dim_check_respects_history_length(tmp_path: Path) -> None:
    """obs_dim must equal base_dim × history_length to match probe_layout."""
    n = 8
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, 40), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "transitions.npz",
        obs=obs,
        actions=rng.standard_normal((n, 2), dtype=np.float32),
        rewards=rng.standard_normal(n, dtype=np.float32),
        next_obs=rng.standard_normal((n, 40), dtype=np.float32),
        dones=np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "obs_dim": 40,
                "history_length": 4,
                "num_transitions": n,
                "num_episodes": 2,
                "success_rate": 1.0,
                "mean_return": 0.0,
            }
        ),
        encoding="utf-8",
    )
    card = derive_sanity_card(tmp_path, expected_probe_layout="s0")
    assert card["history_length"] == 4
    assert card["expected_obs_dim"] == 40
    assert card["obs_dim_matches_probe_layout"] is True

    card_wrong = derive_sanity_card(tmp_path, expected_probe_layout="s1")
    assert card_wrong["expected_obs_dim"] == 48
    assert card_wrong["obs_dim_matches_probe_layout"] is False


def test_write_creates_file(tmp_dataset_dir: Path) -> None:
    out_path = write_sanity_card(tmp_dataset_dir, expected_probe_layout="s0")
    assert out_path.name == "sanity_card.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["obs_dim"] == 10
