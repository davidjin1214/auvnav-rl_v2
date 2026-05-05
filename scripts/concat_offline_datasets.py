"""Concat multiple offline datasets along the transition (= episode) dimension.

Used by the broad-validation A2 mix5050 dataset (spec §4.3): produces an
episode-level mixture by concatenating two separate same-shape datasets
collected with different behavior policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    transitions_path = dataset_dir / "transitions.npz"
    metadata_path = dataset_dir / "metadata.json"
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing {transitions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    with np.load(transitions_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return arrays, metadata


def _check_compatibility(metadatas: list[dict]) -> None:
    base = metadatas[0]
    for m in metadatas[1:]:
        if int(m.get("obs_dim", -1)) != int(base.get("obs_dim", -2)):
            raise ValueError(
                f"obs_dim mismatch: {base.get('obs_dim')} vs {m.get('obs_dim')}"
            )
        if int(m.get("action_dim", -1)) != int(base.get("action_dim", -2)):
            raise ValueError("action_dim mismatch across datasets")
        if int(m.get("privileged_obs_dim", 0) or 0) != int(
            base.get("privileged_obs_dim", 0) or 0
        ):
            raise ValueError("privileged_obs_dim mismatch across datasets")


def concat_datasets(
    dataset_dirs: list[Path],
    *,
    output_dir: Path,
    mix_strategy: str,
    task_sampler: str = "anchor_distribution",
) -> Path:
    if len(dataset_dirs) < 2:
        raise ValueError("Need at least two datasets to concat.")

    payloads: list[dict[str, np.ndarray]] = []
    metadatas: list[dict] = []
    for d in dataset_dirs:
        arrs, meta = _load_dataset(d)
        payloads.append(arrs)
        metadatas.append(meta)

    _check_compatibility(metadatas)

    keys = sorted({k for p in payloads for k in p})
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        # All payloads must contain the same keys; if any are missing, drop the key.
        if not all(key in p for p in payloads):
            continue
        merged[key] = np.concatenate([p[key] for p in payloads], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "transitions.npz", **merged)

    base_meta = metadatas[0]
    components = [
        {
            "policy": m.get("policy"),
            "num_episodes": int(m.get("num_episodes", 0)),
            "num_transitions": int(m.get("num_transitions", 0)),
            "success_rate": float(m.get("success_rate", float("nan"))),
            "mean_return": float(m.get("mean_return", float("nan"))),
            "seed": int(m.get("seed", -1)),
            "source_dir": str(d),
        }
        for m, d in zip(metadatas, dataset_dirs)
    ]

    merged_meta = dict(base_meta)
    merged_meta["mix_components"] = components
    merged_meta["mix_strategy"] = mix_strategy
    merged_meta["task_sampler"] = task_sampler
    merged_meta["num_episodes"] = sum(c["num_episodes"] for c in components)
    merged_meta["num_transitions"] = int(merged["dones"].shape[0])
    merged_meta["policy"] = "+".join(c["policy"] or "?" for c in components)

    # Recompute aggregate scalars.
    weighted_success = sum(
        c["success_rate"] * c["num_episodes"] for c in components
    ) / max(1, merged_meta["num_episodes"])
    weighted_return = sum(
        c["mean_return"] * c["num_episodes"] for c in components
    ) / max(1, merged_meta["num_episodes"])
    merged_meta["success_rate"] = float(weighted_success)
    merged_meta["mean_return"] = float(weighted_return)
    merged_meta.pop("seed", None)
    merged_meta.pop("std_return", None)
    merged_meta.pop("mean_episode_length", None)

    (output_dir / "metadata.json").write_text(
        json.dumps(merged_meta, indent=2), encoding="utf-8"
    )
    return output_dir / "transitions.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Concat offline datasets.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        type=Path,
        help="Source dataset directory. Pass --input-dir twice or more.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-strategy", default="episode_level")
    parser.add_argument("--task-sampler", default="anchor_distribution")
    args = parser.parse_args()

    out_path = concat_datasets(
        args.input_dir,
        output_dir=args.output_dir,
        mix_strategy=args.mix_strategy,
        task_sampler=args.task_sampler,
    )
    print(f"[write] merged dataset: {out_path}")


if __name__ == "__main__":
    main()
