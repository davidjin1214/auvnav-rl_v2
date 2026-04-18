from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.collect_offline_data import _episode_chunks, collect


def _flow_path() -> Path:
    root = Path(__file__).parent.parent
    return root / "wake_data" / "wake_dummy_roi.npy"


def _make_args(output_dir: Path, *, num_workers: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy="goalseek",
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
