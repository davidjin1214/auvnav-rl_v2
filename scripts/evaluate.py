"""Evaluate a saved SAC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_nav.reward import REWARD_OBJECTIVE_PRESETS
from auv_nav.sac import SACAgent, SACConfig
from .train_utils import (
    default_device,
    evaluate_agent,
    extract_env_config_overrides,
    load_trainer_state,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
    maybe_load_benchmark_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Run directory or trainer_state.json path.")
    parser.add_argument(
        "--agent-file",
        type=str,
        default=None,
        help="Optional agent filename relative to the run directory, for example agent_step_180000.pt.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Torch device. Defaults to cuda:0 when CUDA is available, otherwise cpu.",
    )
    parser.add_argument("--flow", type=str, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to a fixed benchmark manifest JSON.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to save evaluation metrics as JSON.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--task-geometry",
                        choices=["downstream", "cross_stream", "upstream"], default=None)
    parser.add_argument("--action-mode",
                        choices=["auto", "goal_relative_offset", "absolute_heading"],
                        default=None)
    parser.add_argument("--speed-ratio", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument(
        "--objective",
        choices=sorted(REWARD_OBJECTIVE_PRESETS.keys()),
        default=None,
        help="Override the checkpoint's reward objective preset.",
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument("--history-length", type=int, default=None,
                        help="Override stacked-observation history length.")
    args = parser.parse_args()

    trainer_state = load_trainer_state(args.checkpoint)
    agent_cfg_dict = trainer_state["agent_config"]

    reset_options = {**trainer_state.get("reset_options", {}), **make_reset_options(args)}
    benchmark_manifest = maybe_load_benchmark_manifest(args.manifest)
    env_config_overrides = make_env_config_overrides(
        args,
        base_overrides=extract_env_config_overrides(trainer_state),
    )

    flow_path = args.flow or trainer_state.get("flow_path", "wake_data/wake_test_roi.npy")
    history_length = (
        args.history_length
        if args.history_length is not None
        else int(trainer_state.get("history_length", 1))
    )
    probe_layout = trainer_state.get("probe_layout", "s0")
    if benchmark_manifest is not None:
        if (
            benchmark_manifest.probe_layout is not None
            and benchmark_manifest.probe_layout != probe_layout
        ):
            raise ValueError(
                f"Benchmark manifest probe_layout={benchmark_manifest.probe_layout} "
                f"!= checkpoint probe_layout={probe_layout}."
            )
        if (
            benchmark_manifest.history_length is not None
            and benchmark_manifest.history_length != history_length
        ):
            raise ValueError(
                f"Benchmark manifest history_length={benchmark_manifest.history_length} "
                f"!= evaluation history_length={history_length}."
            )
        flow_path = args.flow or benchmark_manifest.flow_path
    env = make_planar_env(
        flow_path,
        history_length=history_length,
        probe_layout=probe_layout,
        env_config_overrides=env_config_overrides,
    )
    agent = SACAgent(SACConfig(**agent_cfg_dict), device=args.device)

    checkpoint_root = Path(args.checkpoint)
    run_root = checkpoint_root if checkpoint_root.is_dir() else checkpoint_root.parent
    agent_relative_path = args.agent_file or trainer_state["agent_path"]
    agent_file = run_root / agent_relative_path
    agent.load(str(agent_file))

    metrics = evaluate_agent(
        env=env,
        agent=agent,
        reset_options=reset_options,
        seed=args.seed,
        num_episodes=args.episodes,
        benchmark_manifest=benchmark_manifest,
    )

    n_eps = int(metrics["num_eval_episodes"])
    print(f"reward_objective      : {metrics['reward_objective']}")
    print(f"energy_cost_gain      : {metrics['energy_cost_gain']:.6f}")
    print(f"safety_cost_gain      : {metrics['safety_cost_gain']:.6f}")
    print(f"episodes              : {n_eps}")
    print(f"success_rate          : {100.0 * metrics['eval_success_rate']:.1f}%")
    print(f"avg_return            : {metrics['eval_return']:.2f} +/- {metrics['eval_return_std']:.2f}")
    print(
        f"avg_safety_cost       : "
        f"{metrics['eval_safety_cost']:.3f} +/- {metrics['eval_safety_cost_std']:.3f}"
    )
    print(f"avg_time_s            : {metrics['eval_time_s']:.2f} +/- {metrics['eval_time_s_std']:.2f}")
    print(f"avg_time_s_success    : {metrics['eval_time_s_success']:.2f}")
    print(f"avg_energy            : {metrics['eval_energy']:.2f} +/- {metrics['eval_energy_std']:.2f}")
    print(
        f"avg_path_length_m     : "
        f"{metrics['eval_path_length_m']:.2f} +/- {metrics['eval_path_length_m_std']:.2f}"
    )
    print(
        f"avg_progress_ratio    : "
        f"{metrics['eval_progress_ratio']:.3f} +/- {metrics['eval_progress_ratio_std']:.3f}"
    )
    print(
        f"avg_path_efficiency   : "
        f"{metrics['eval_path_efficiency']:.3f} +/- {metrics['eval_path_efficiency_std']:.3f}"
    )
    print(f"termination           : {metrics['eval_termination_counts']}")
    if benchmark_manifest is not None:
        print(f"benchmark_manifest    : {args.manifest}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
