from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.collect_offline_data import (
    TERMINATION_REASON_TO_CODE,
    _collect_episode,
    _episode_chunks,
    collect,
)


def _flow_path() -> Path:
    root = Path(__file__).parent.parent
    return root / "wake_data" / "wake_dummy_roi.npy"


def _make_args(output_dir: Path, *, num_workers: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy="goalseek",
        policy_mixture=None,
        flow=_flow_path(),
        probe_layout="s0",
        history_length=1,
        difficulty="easy",
        task_geometry="downstream",
        target_speed=1.0,
        objective="arrival_v1",
        energy_cost_gain=None,
        safety_cost_gain=None,
        episodes=2,
        seed=7,
        action_noise_std=0.0,
        action_noise_clip=0.5,
        num_workers=num_workers,
        output_dir=str(output_dir),
    )


def test_episode_chunks_balances_ranges() -> None:
    assert _episode_chunks(0, 4) == []
    assert _episode_chunks(1, 4) == [(0, 1)]
    assert _episode_chunks(5, 2) == [(0, 3), (3, 5)]
    assert _episode_chunks(5, 8) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_parallel_collect_matches_serial(tmp_path: Path) -> None:
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    collect(_make_args(serial_dir, num_workers=1))
    collect(_make_args(parallel_dir, num_workers=2))

    with np.load(serial_dir / "transitions.npz") as serial_data:
        serial_payload = {key: serial_data[key].copy() for key in serial_data.files}
    with np.load(parallel_dir / "transitions.npz") as parallel_data:
        parallel_payload = {key: parallel_data[key].copy() for key in parallel_data.files}

    assert serial_payload.keys() == parallel_payload.keys()
    for key in serial_payload:
        assert np.array_equal(serial_payload[key], parallel_payload[key]), key

    serial_meta = json.loads((serial_dir / "metadata.json").read_text(encoding="utf-8"))
    parallel_meta = json.loads((parallel_dir / "metadata.json").read_text(encoding="utf-8"))
    assert serial_meta == parallel_meta


def test_parallel_collect_matches_serial_with_mixture_and_noise(tmp_path: Path) -> None:
    serial_dir = tmp_path / "serial_mix"
    parallel_dir = tmp_path / "parallel_mix"

    serial_args = _make_args(serial_dir, num_workers=1)
    serial_args.policy = "crosscomp"
    serial_args.policy_mixture = "crosscomp:0.5,goalseek:0.5"
    serial_args.action_noise_std = 0.1
    serial_args.episodes = 4

    parallel_args = _make_args(parallel_dir, num_workers=2)
    parallel_args.policy = serial_args.policy
    parallel_args.policy_mixture = serial_args.policy_mixture
    parallel_args.action_noise_std = serial_args.action_noise_std
    parallel_args.episodes = serial_args.episodes

    collect(serial_args)
    collect(parallel_args)

    with np.load(serial_dir / "transitions.npz") as serial_data:
        serial_payload = {key: serial_data[key].copy() for key in serial_data.files}
    with np.load(parallel_dir / "transitions.npz") as parallel_data:
        parallel_payload = {key: parallel_data[key].copy() for key in parallel_data.files}

    assert serial_payload.keys() == parallel_payload.keys()
    for key in serial_payload:
        assert np.array_equal(serial_payload[key], parallel_payload[key]), key

    serial_meta = json.loads((serial_dir / "metadata.json").read_text(encoding="utf-8"))
    parallel_meta = json.loads((parallel_dir / "metadata.json").read_text(encoding="utf-8"))
    assert serial_meta == parallel_meta
    assert "policy_episode_counts" in serial_meta
    assert "behavior_policy_vocab" in serial_meta


class _OneStepTimeoutEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
        )

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        return np.array([1.0, 2.0], dtype=np.float32), {"privileged_obs": None}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return (
            np.array([3.0, 4.0], dtype=np.float32),
            -1.0,
            False,
            True,
            {
                "step_safety_cost": 0.25,
                "success": False,
                "reason": "timeout",
                "privileged_obs": None,
            },
        )


class _ZeroPolicy:
    def act(self, base_env: object, obs: np.ndarray) -> np.ndarray:
        del base_env, obs
        return np.zeros(2, dtype=np.float32)


def test_collect_episode_marks_timeout_as_done() -> None:
    env = _OneStepTimeoutEnv()
    transitions, success, ep_return, ep_length, ep_reason = _collect_episode(
        env,
        env,
        _ZeroPolicy(),
        seed=0,
        reset_options={},
        action_noise_std=0.0,
        action_noise_clip=0.5,
        episode_rng=np.random.default_rng(0),
        behavior_policy_name="goalseek",
    )

    assert success is False
    assert ep_return == -1.0
    assert ep_length == 1
    assert ep_reason == "timeout"
    assert transitions["dones"] == [True]
    assert transitions["terminateds"] == [False]
    assert transitions["truncateds"] == [True]
    assert transitions["terminal_reason_codes"] == [TERMINATION_REASON_TO_CODE["timeout"]]
