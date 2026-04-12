from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_utils import BenchmarkEpisode, build_benchmark_manifest, save_benchmark_manifest
from .train_utils import discover_flow_path, make_planar_env, make_reset_options


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fixed benchmark manifest by sampling environment resets once.",
    )
    parser.add_argument("--flow", type=Path, default=None, help="Path to wake ROI .npy file.")
    parser.add_argument("--probe-layout", choices=["s0", "s1", "s2"], default="s0")
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument(
        "--task-geometry",
        choices=["downstream", "cross_stream", "upstream"],
        default=None,
    )
    parser.add_argument(
        "--action-mode",
        choices=["auto", "goal_relative_offset", "absolute_heading"],
        default=None,
    )
    parser.add_argument("--speed-ratio", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--initial-speed", type=float, default=0.3)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flow_path = args.flow or discover_flow_path()
    env = make_planar_env(
        flow_path,
        history_length=args.history_length,
        probe_layout=args.probe_layout,
    )
    base_reset_options = make_reset_options(args)
    base_reset_options["initial_speed"] = float(args.initial_speed)

    episodes: list[BenchmarkEpisode] = []
    for idx in range(args.episodes):
        seed = args.seed + idx
        _, info = env.reset(seed=seed, options=base_reset_options)
        reset_options = {
            "flow_time": float(info["flow_time_s"]),
            "start_xy": info["start_xy_m"].tolist(),
            "goal_xy": info["goal_xy_m"].tolist(),
            "initial_heading": float(info["psi_rad"]),
            "initial_speed": float(args.initial_speed),
            "task_geometry": str(info["task_geometry"]),
            "action_mode": str(info["action_mode"]),
            "target_auv_max_speed_mps": float(info["target_auv_max_speed_mps"]),
        }
        episodes.append(
            BenchmarkEpisode(
                episode_id=f"ep_{idx:04d}",
                seed=seed,
                reset_options=reset_options,
            )
        )

    manifest = build_benchmark_manifest(
        flow_path=flow_path,
        probe_layout=args.probe_layout,
        history_length=args.history_length,
        base_reset_options=base_reset_options,
        episodes=episodes,
        notes=args.notes,
    )
    save_benchmark_manifest(args.output, manifest)
    print(
        f"[done] wrote benchmark manifest with {len(episodes)} episodes to {args.output}\n"
        f"  flow={flow_path}\n"
        f"  probe_layout={args.probe_layout} history_length={args.history_length}"
    )


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
