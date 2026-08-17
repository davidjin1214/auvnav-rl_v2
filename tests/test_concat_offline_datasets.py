"""Unit tests for the dataset concat helper (A2 mix5050)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.concat_offline_datasets import concat_datasets


def _write_synthetic(
    out_dir: Path,
    *,
    policy: str,
    num_episodes: int,
    transitions_per_episode: int,
    seed: int,
    obs_dim: int = 10,
    privileged_dim: int = 2,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = num_episodes * transitions_per_episode
    rng = np.random.default_rng(seed)
    payload = {
        "obs": rng.standard_normal((n, obs_dim), dtype=np.float32),
        "actions": rng.standard_normal((n, 2), dtype=np.float32),
        "rewards": rng.standard_normal(n, dtype=np.float32),
        "next_obs": rng.standard_normal((n, obs_dim), dtype=np.float32),
        "dones": np.zeros(n, dtype=np.float32),
        "privileged_obs": rng.standard_normal((n, privileged_dim), dtype=np.float32),
        "next_privileged_obs": rng.standard_normal((n, privileged_dim), dtype=np.float32),
    }
    payload["dones"][transitions_per_episode - 1::transitions_per_episode] = 1.0
    np.savez_compressed(out_dir / "transitions.npz", **payload)

    metadata = {
        "policy": policy,
        "obs_dim": obs_dim,
        "action_dim": 2,
        "privileged_obs_dim": privileged_dim,
        "num_episodes": num_episodes,
        "num_transitions": n,
        "success_rate": 0.5,
        "mean_return": 10.0,
        "std_return": 2.0,
        "mean_episode_length": float(transitions_per_episode),
        "flow_path": "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
        "probe_layout": "s0",
        "history_length": 4,
        "task_geometry": "cross_stream",
        "target_speed": 1.5,
        "objective": "efficiency_v2",
        "seed": seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out_dir


def test_concat_preserves_episode_boundaries(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"

    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")

    with np.load(out_dir / "transitions.npz") as merged:
        dones = merged["dones"]
    # Total = 3 ep * 5 + 2 ep * 4 = 23 transitions, 5 dones boundaries.
    assert dones.shape == (23,)
    assert int(dones.sum()) == 5
    # First 15 transitions belong to dataset A (3 episodes of length 5).
    assert dones[4] == 1.0 and dones[9] == 1.0 and dones[14] == 1.0
    # Last 8 transitions belong to dataset B (2 episodes of length 4).
    assert dones[18] == 1.0 and dones[22] == 1.0


def test_concat_writes_extended_metadata(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"

    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    components = metadata["mix_components"]
    assert len(components) == 2
    assert components[0]["policy"] == "goalseek"
    assert components[0]["num_episodes"] == 3
    assert components[1]["policy"] == "crosscomp"
    assert components[1]["num_episodes"] == 2
    assert metadata["mix_strategy"] == "episode_level"
    assert metadata["num_episodes"] == 5
    assert metadata["num_transitions"] == 23
    assert metadata["task_sampler"] == "anchor_distribution"


def test_concat_rejects_dim_mismatch(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0, obs_dim=10)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1, obs_dim=12)
    with pytest.raises(ValueError, match="obs_dim"):
        concat_datasets([a, b], output_dir=tmp_path / "merged", mix_strategy="episode_level")


def test_concat_preserves_privileged_obs(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"
    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")
    with np.load(out_dir / "transitions.npz") as merged:
        assert "privileged_obs" in merged.files
        assert "next_privileged_obs" in merged.files
        assert merged["privileged_obs"].shape[1] == 2
