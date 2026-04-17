"""Evaluate a baseline policy on a fixed benchmark protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from auv_nav.baselines import (
    CrossCurrentCompensationPolicy,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    WorldFrameCurrentCompensationPolicy,
)
from auv_nav.env import ObservationHistoryWrapper
from auv_nav.reward import REWARD_OBJECTIVE_PRESETS
from .train_utils import (
    discover_flow_path,
    evaluate_agent,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
    maybe_load_benchmark_manifest,
)

POLICY_MAP = {
    "goalseek": GoalSeekPolicy,
    "crosscomp": CrossCurrentCompensationPolicy,
    "worldcomp": WorldFrameCurrentCompensationPolicy,
    "privileged": PrivilegedCorridorPolicy,
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class BaselineAgentAdapter:
    def __init__(self, env, policy_name: str) -> None:
        self.env = env
        self.policy = POLICY_MAP[policy_name]()

    def reset_policy_state(self) -> None:
        return None

    def act(self, obs, policy_state=None, deterministic: bool = True):
        _ = policy_state, deterministic
        env = self.env.env if isinstance(self.env, ObservationHistoryWrapper) else self.env
        if isinstance(self.env, ObservationHistoryWrapper):
            single_obs = self.env._history[-1]
        else:
            single_obs = obs
        return self.policy.act(env, single_obs), None


def _print_metrics(metrics: dict[str, float], manifest_path: Path | None) -> None:
    n_eps = int(metrics["num_eval_episodes"])
    print(f"reward_objective      : {metrics['reward_objective']}")
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
    if manifest_path is not None:
        print(f"benchmark_manifest    : {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline policy on a benchmark manifest.")
    parser.add_argument("--policy", choices=sorted(POLICY_MAP.keys()), required=True)
    parser.add_argument("--flow", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to a fixed benchmark manifest JSON.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", type=Path, default=None)
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
        default="arrival_v1",
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument(
        "--probe-layout",
        choices=["s0", "s1", "s2"],
        default="s0",
    )
    parser.add_argument("--history-length", type=int, default=1)
    args = parser.parse_args()

    benchmark_manifest = maybe_load_benchmark_manifest(args.manifest)
    if benchmark_manifest is not None:
        if (
            benchmark_manifest.probe_layout is not None
            and benchmark_manifest.probe_layout != args.probe_layout
        ):
            raise ValueError(
                f"Benchmark manifest probe_layout={benchmark_manifest.probe_layout} "
                f"!= requested probe_layout={args.probe_layout}."
            )
        if (
            benchmark_manifest.history_length is not None
            and benchmark_manifest.history_length != args.history_length
        ):
            raise ValueError(
                f"Benchmark manifest history_length={benchmark_manifest.history_length} "
                f"!= requested history_length={args.history_length}."
            )

    env_config_overrides = make_env_config_overrides(args)
    reset_options = make_reset_options(args)
    flow_path = (
        benchmark_manifest.flow_path
        if benchmark_manifest is not None
        else str(args.flow or discover_flow_path())
    )
    env = make_planar_env(
        flow_path,
        history_length=args.history_length,
        probe_layout=args.probe_layout,
        env_config_overrides=env_config_overrides,
    )
    agent = BaselineAgentAdapter(env, args.policy)

    metrics = evaluate_agent(
        env=env,
        agent=agent,
        reset_options=reset_options,
        seed=args.seed,
        num_episodes=args.episodes,
        benchmark_manifest=benchmark_manifest,
    )
    _print_metrics(metrics, args.manifest)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(_json_ready(metrics), fp, indent=2)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
