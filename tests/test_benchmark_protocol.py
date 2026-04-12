from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_utils import (
    BenchmarkEpisode,
    build_benchmark_manifest,
    load_benchmark_manifest,
    save_benchmark_manifest,
)
from scripts.train_utils import evaluate_agent, make_planar_env


class ZeroAgent:
    def reset_policy_state(self) -> None:
        return None

    def act(self, obs: np.ndarray, policy_state: None = None, deterministic: bool = True):
        _ = obs, policy_state, deterministic
        return np.zeros(2, dtype=np.float32), None


def _flow_path() -> Path:
    root = Path(__file__).parent.parent
    return root / "wake_data" / "wake_dummy_roi.npy"


def test_benchmark_manifest_roundtrip(tmp_path: Path):
    manifest = build_benchmark_manifest(
        flow_path=_flow_path(),
        probe_layout=None,
        history_length=None,
        base_reset_options={"task_geometry": "upstream"},
        episodes=[
            BenchmarkEpisode(
                episode_id="ep_0000",
                seed=123,
                reset_options={"flow_time": 1.5, "start_xy": [0.0, 0.0], "goal_xy": [1.0, 0.0]},
            )
        ],
        benchmark_id="single_u15_upstream_tgt15",
        factor_values={"task_geometry": "upstream", "target_speed_mps": 1.5},
        notes="roundtrip",
    )
    path = tmp_path / "manifest.json"
    save_benchmark_manifest(path, manifest)
    loaded = load_benchmark_manifest(path)
    assert loaded.flow_path == str(_flow_path())
    assert loaded.probe_layout is None
    assert loaded.history_length is None
    assert loaded.episodes[0].episode_id == "ep_0000"
    assert loaded.benchmark_id == "single_u15_upstream_tgt15"
    assert loaded.factor_values == {"task_geometry": "upstream", "target_speed_mps": 1.5}
    assert loaded.notes == "roundtrip"


def test_evaluate_agent_uses_fixed_manifest_episodes(tmp_path: Path):
    env = make_planar_env(_flow_path(), history_length=1, probe_layout="s0")
    base_reset_options = {
        "task_geometry": "upstream",
        "action_mode": "absolute_heading",
        "target_auv_max_speed_mps": 1.5,
        "initial_speed": 0.3,
    }
    sampled: list[BenchmarkEpisode] = []
    for idx in range(2):
        seed = 100 + idx
        _, info = env.reset(seed=seed, options=base_reset_options)
        sampled.append(
            BenchmarkEpisode(
                episode_id=f"ep_{idx:04d}",
                seed=seed,
                reset_options={
                    "flow_time": float(info["flow_time_s"]),
                    "start_xy": info["start_xy_m"].tolist(),
                    "goal_xy": info["goal_xy_m"].tolist(),
                    "initial_heading": float(info["psi_rad"]),
                    "initial_speed": 0.3,
                    "task_geometry": str(info["task_geometry"]),
                    "action_mode": str(info["action_mode"]),
                    "target_auv_max_speed_mps": float(info["target_auv_max_speed_mps"]),
                },
            )
        )

    manifest = build_benchmark_manifest(
        flow_path=_flow_path(),
        probe_layout=None,
        history_length=None,
        base_reset_options=base_reset_options,
        episodes=sampled,
    )
    manifest_path = tmp_path / "fixed_eval.json"
    save_benchmark_manifest(manifest_path, manifest)
    loaded = load_benchmark_manifest(manifest_path)

    metrics = evaluate_agent(
        env=env,
        agent=ZeroAgent(),
        reset_options={},
        seed=999,
        num_episodes=99,
        benchmark_manifest=loaded,
    )

    assert metrics["num_eval_episodes"] == 2.0
    assert len(metrics["eval_episode_results"]) == 2
    assert metrics["eval_episode_results"][0]["episode_id"] == "ep_0000"
    assert "eval_energy" in metrics
    assert "eval_path_length_m" in metrics
    assert "eval_progress_ratio" in metrics
    assert isinstance(metrics["eval_termination_counts"], dict)

    payload = json.loads(manifest_path.read_text())
    assert payload["history_length"] is None
