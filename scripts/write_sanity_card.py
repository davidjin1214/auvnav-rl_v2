"""Derive a 7-field sanity card from an offline-data directory.

Spec §4.4: every new dataset under offline_data/ must have a sanity_card.json
recording the 7 fields below before being used for training.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


_PROBE_TO_BASE_OBS_DIM = {"s0": 10, "s1": 12, "s2": 16}


def _episode_lengths(dones: np.ndarray) -> list[int]:
    """Return list of episode lengths inferred from `dones` boundaries."""
    lengths: list[int] = []
    current = 0
    for flag in dones:
        current += 1
        if bool(flag):
            lengths.append(current)
            current = 0
    if current > 0:  # trailing partial episode (should not happen on closed datasets)
        lengths.append(current)
    return lengths


def derive_sanity_card(
    dataset_dir: Path,
    *,
    expected_probe_layout: str | None = None,
) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    transitions_path = dataset_dir / "transitions.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing {transitions_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(transitions_path, mmap_mode="r") as payload:
        dones = np.asarray(payload["dones"])
    episode_lengths = _episode_lengths(dones)
    if not episode_lengths:
        raise ValueError(f"transitions.npz at {transitions_path} has no episode boundaries")

    flow_path = metadata.get("flow_path", "")
    flow_file = Path(flow_path).name if flow_path else ""

    privileged_dim = int(metadata.get("privileged_obs_dim", 0) or 0)

    card: dict[str, Any] = {
        "collector_success_rate": float(metadata.get("success_rate", float("nan"))),
        "collector_mean_return": float(metadata.get("mean_return", float("nan"))),
        "episode_length_mean": statistics.fmean(episode_lengths),
        "episode_length_std": statistics.pstdev(episode_lengths) if len(episode_lengths) > 1 else 0.0,
        "obs_dim": int(metadata.get("obs_dim", -1)),
        "n_transitions": int(metadata.get("num_transitions", dones.shape[0])),
        "privileged_obs_present": privileged_dim > 0,
        "flow_file": flow_file,
        "n_episodes": len(episode_lengths),
    }

    if expected_probe_layout is not None:
        history_length = int(metadata.get("history_length", 1) or 1)
        base_dim = _PROBE_TO_BASE_OBS_DIM.get(expected_probe_layout)
        context_dim = 2 if bool(metadata.get("include_episode_context_obs", False)) else 0
        expected_dim = (base_dim + context_dim) * history_length if base_dim is not None else None
        card["expected_probe_layout"] = expected_probe_layout
        card["history_length"] = history_length
        card["include_episode_context_obs"] = bool(
            metadata.get("include_episode_context_obs", False)
        )
        card["expected_obs_dim"] = expected_dim
        card["obs_dim_matches_probe_layout"] = (
            expected_dim is not None and card["obs_dim"] == expected_dim
        )

    return card


def write_sanity_card(
    dataset_dir: Path,
    *,
    expected_probe_layout: str | None = None,
) -> Path:
    card = derive_sanity_card(
        dataset_dir, expected_probe_layout=expected_probe_layout
    )
    output_path = dataset_dir / "sanity_card.json"
    output_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a dataset sanity card.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-probe-layout",
        choices=["s0", "s1", "s2"],
        default=None,
    )
    args = parser.parse_args()
    out_path = write_sanity_card(
        args.dataset_dir,
        expected_probe_layout=args.expected_probe_layout,
    )
    print(f"[write] sanity card: {out_path}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
