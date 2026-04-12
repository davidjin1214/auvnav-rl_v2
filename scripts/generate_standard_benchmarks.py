from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_catalog import (
    BENCHMARK_GROUPS,
    BENCHMARK_SPECS,
    default_manifest_path,
    resolve_benchmark_specs,
)
from .benchmark_utils import BenchmarkEpisode, build_benchmark_manifest, save_benchmark_manifest
from .train_utils import make_planar_env


def _build_manifest_for_benchmark(
    benchmark_key: str,
    episodes: int,
    output_dir: Path,
    notes: str | None,
) -> Path:
    spec = BENCHMARK_SPECS[benchmark_key]
    env = make_planar_env(spec.flow_path, history_length=1, probe_layout="s0")
    base_reset_options = spec.reset_options()
    base_reset_options["initial_speed"] = 0.3

    frozen_episodes: list[BenchmarkEpisode] = []
    for idx in range(episodes):
        seed = spec.manifest_seed + idx
        _, info = env.reset(seed=seed, options=base_reset_options)
        frozen_episodes.append(
            BenchmarkEpisode(
                episode_id=f"{spec.key}_ep_{idx:04d}",
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
        flow_path=spec.flow_path,
        probe_layout=None,
        history_length=None,
        base_reset_options=base_reset_options,
        episodes=frozen_episodes,
        benchmark_id=spec.key,
        factor_values=spec.factor_values,
        notes=notes or spec.description,
    )
    output_path = output_dir / Path(default_manifest_path(spec, manifest_dir=".")).name
    save_benchmark_manifest(output_path, manifest)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the repository's standard benchmark manifests.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark keys.",
    )
    parser.add_argument(
        "--benchmark-group",
        type=str,
        choices=sorted(BENCHMARK_GROUPS.keys()),
        default=None,
        help="Named benchmark group to materialize.",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()

    specs = resolve_benchmark_specs(args.benchmarks, args.benchmark_group)
    if not specs:
        specs = list(BENCHMARK_SPECS.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        output_path = _build_manifest_for_benchmark(
            benchmark_key=spec.key,
            episodes=args.episodes,
            output_dir=args.output_dir,
            notes=args.notes,
        )
        print(f"[done] {spec.key} -> {output_path}")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
