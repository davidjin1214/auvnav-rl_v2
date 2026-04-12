from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auv_nav.reward import REWARD_OBJECTIVE_PRESETS
from .benchmark_catalog import (
    BENCHMARK_GROUPS,
    BenchmarkSpec,
    default_manifest_path,
    resolve_benchmark_specs,
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    key: str
    train_module: str
    extra_args: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class SuitePreset:
    name: str
    description: str
    values: dict[str, Any]


METHOD_SPECS: dict[str, MethodSpec] = {
    "sac": MethodSpec(
        key="sac",
        train_module="scripts.train_sac",
        extra_args=("--history-length", "1"),
        description="Vanilla feedforward SAC.",
    ),
    "sac_stack4": MethodSpec(
        key="sac_stack4",
        train_module="scripts.train_sac",
        extra_args=("--history-length", "4"),
        description="Feedforward SAC with 4-step observation history.",
    ),
    "sac_ln_k16": MethodSpec(
        key="sac_ln_k16",
        train_module="scripts.train_sac",
        extra_args=("--use-layernorm", "--history-length", "16"),
        description="SAC + LayerNorm + history K=16",
    ),
    "sac_droq": MethodSpec(
        key="sac_droq",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--dropout-rate", "0.01",
            "--updates-per-step", "4", "--history-length", "16",
        ),
        description="DroQ: SAC + LayerNorm + Dropout(0.01) + UTD=4 + K=16",
    ),
    "sac_asym_k16": MethodSpec(
        key="sac_asym_k16",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--use-asymmetric-critic", "--history-length", "16",
        ),
        description="SAC + LayerNorm + Asymmetric Critic + K=16",
    ),
    "sac_full": MethodSpec(
        key="sac_full",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--dropout-rate", "0.01",
            "--updates-per-step", "4",
            "--use-asymmetric-critic", "--history-length", "16",
        ),
        description="SAC + all improvements: LN + DroQ + AsymCritic + UTD=4 + K=16",
    ),
    "rlpd_goalseek": MethodSpec(
        key="rlpd_goalseek",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--history-length", "1",
            "--offline-data", "offline_data/goalseek/transitions.npz",
            "--offline-ratio", "0.5",
        ),
        description="RLPD: SAC + GoalSeek offline data (50/50).",
    ),
    "rlpd_worldcomp": MethodSpec(
        key="rlpd_worldcomp",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--history-length", "1",
            "--offline-data", "offline_data/worldcomp/transitions.npz",
            "--offline-ratio", "0.5",
        ),
        description="RLPD: SAC + WorldFrameCompensation offline data (50/50).",
    ),
    "rlpd_privileged": MethodSpec(
        key="rlpd_privileged",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--history-length", "1",
            "--offline-data", "offline_data/privileged/transitions.npz",
            "--offline-ratio", "0.5",
        ),
        description="RLPD: SAC + PrivilegedCorridor offline data (50/50).",
    ),
}


SUITE_PRESETS: dict[str, SuitePreset] = {
    "medium_formal_v1": SuitePreset(
        name="medium_formal_v1",
        description=(
            "Legacy benchmark on medium difficulty "
            "(cross-stream) with 5 seeds and a 200k-step budget."
        ),
        values={
            "difficulty": "medium",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44,45,46",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "update_every": 1,
            "updates_per_step": 1,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
            "log_every_episodes": 10,
            "batch_size": 256,
            "replay_capacity": 1_000_000,
            "hidden_dim": 256,
        },
    ),
    "medium_pilot_v1": SuitePreset(
        name="medium_pilot_v1",
        description="Legacy pilot suite for quick checks before the formal medium benchmark.",
        values={
            "difficulty": "medium",
            "methods": "sac,sac_stack4",
            "seeds": "42,43",
            "total_steps": 50_000,
            "random_steps": 2_000,
            "update_after": 2_000,
            "update_every": 1,
            "updates_per_step": 1,
            "eval_every": 5_000,
            "eval_episodes": 10,
            "checkpoint_every": 5_000,
            "log_every_episodes": 5,
            "batch_size": 256,
            "replay_capacity": 1_000_000,
            "hidden_dim": 256,
        },
    ),
    "sac_ablation_v1": SuitePreset(
        name="sac_ablation_v1",
        description=(
            "5-variant SAC ablation study: baseline vs LN+K16 vs DroQ "
            "vs AsymCritic vs Full. 3 seeds each."
        ),
        values={
            "methods": "sac,sac_ln_k16,sac_droq,sac_asym_k16,sac_full",
            "seeds": "42,43,44",
            "total_steps": 200_000,
            "difficulty": "medium",
        },
    ),
    "geometry_factor_v1": SuitePreset(
        name="geometry_factor_v1",
        description="Factorized suite varying only task geometry on the single-cylinder matched-speed benchmark.",
        values={
            "benchmark_group": "geometry_factor_v1",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44",
            "objective": "efficiency_v1",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
    "flow_factor_v1": SuitePreset(
        name="flow_factor_v1",
        description="Factorized suite varying only the incoming free-stream speed.",
        values={
            "benchmark_group": "flow_factor_v1",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44",
            "objective": "efficiency_v1",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
    "topology_factor_v1": SuitePreset(
        name="topology_factor_v1",
        description="Factorized suite varying wake topology in the matched-speed upstream regime.",
        values={
            "benchmark_group": "topology_factor_v1",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44",
            "objective": "efficiency_v1",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
    "speed_factor_v1": SuitePreset(
        name="speed_factor_v1",
        description="Factorized suite varying only the target AUV max-speed setting.",
        values={
            "benchmark_group": "speed_factor_v1",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44",
            "objective": "efficiency_v1",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
    "study_core_v1": SuitePreset(
        name="study_core_v1",
        description="Recommended benchmark suite covering the project's primary transfer axes.",
        values={
            "benchmark_group": "study_core_v1",
            "methods": "sac,sac_stack4",
            "seeds": "42,43,44",
            "objective": "efficiency_v1",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
    "objective_ablation_v1": SuitePreset(
        name="objective_ablation_v1",
        description=(
            "Small objective ablation on the critical single-cylinder upstream benchmark: "
            "arrival_v1 vs efficiency_v1."
        ),
        values={
            "benchmarks": "single_u15_upstream_tgt15",
            "methods": "sac_stack4",
            "objectives": "arrival_v1,efficiency_v1",
            "seeds": "42,43,44",
            "total_steps": 200_000,
            "random_steps": 5_000,
            "update_after": 5_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
            "checkpoint_every": 10_000,
        },
    ),
}


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


def parse_objective_list(objective_text: str) -> list[str]:
    return [part.strip() for part in objective_text.split(",") if part.strip()]


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return
    if preset_name not in SUITE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    preset = SUITE_PRESETS[preset_name]
    for key, value in preset.values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def append_optional_arg(args: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def resolve_suite_benchmarks(args: argparse.Namespace) -> list[BenchmarkSpec]:
    return resolve_benchmark_specs(args.benchmarks, args.benchmark_group)


def validate_benchmark_mode(args: argparse.Namespace, benchmarks: list[BenchmarkSpec]) -> None:
    if not benchmarks:
        return
    conflicting = {
        "flow": args.flow,
        "difficulty": args.difficulty,
        "task_geometry": args.task_geometry,
        "action_mode": args.action_mode,
        "speed_ratio": args.speed_ratio,
        "target_speed": args.target_speed,
        "eval_manifest": args.eval_manifest,
    }
    active_conflicts = [key for key, value in conflicting.items() if value is not None]
    if active_conflicts:
        raise ValueError(
            "Direct task-defining flags are not allowed together with benchmark mode: "
            + ", ".join(active_conflicts)
        )


def benchmark_manifest_path(
    benchmark: BenchmarkSpec,
    manifest_dir: str,
) -> str:
    return default_manifest_path(benchmark, manifest_dir=manifest_dir)


def build_command(
    method: MethodSpec,
    seed: int,
    save_dir: Path,
    cli_args: argparse.Namespace,
    objective: str | None = None,
    benchmark: BenchmarkSpec | None = None,
) -> list[str]:
    cmd = [sys.executable, "-m", method.train_module, *method.extra_args]
    resolved_objective = objective if objective is not None else getattr(cli_args, "objective", None)
    if benchmark is None:
        append_optional_arg(cmd, "--flow", cli_args.flow)
        append_optional_arg(cmd, "--difficulty", cli_args.difficulty)
        append_optional_arg(cmd, "--task-geometry", cli_args.task_geometry)
        append_optional_arg(cmd, "--action-mode", cli_args.action_mode)
        append_optional_arg(cmd, "--speed-ratio", cli_args.speed_ratio)
        append_optional_arg(cmd, "--target-speed", cli_args.target_speed)
        append_optional_arg(cmd, "--eval-manifest", cli_args.eval_manifest)
    else:
        append_optional_arg(cmd, "--flow", benchmark.flow_path)
        append_optional_arg(cmd, "--task-geometry", benchmark.task_geometry)
        append_optional_arg(cmd, "--action-mode", benchmark.action_mode)
        append_optional_arg(cmd, "--target-speed", benchmark.target_speed)
        append_optional_arg(
            cmd,
            "--eval-manifest",
            benchmark_manifest_path(benchmark, cli_args.benchmark_manifest_dir),
        )
    append_optional_arg(cmd, "--objective", resolved_objective)
    append_optional_arg(cmd, "--energy-cost-gain", cli_args.energy_cost_gain)
    append_optional_arg(cmd, "--safety-cost-gain", cli_args.safety_cost_gain)
    append_optional_arg(cmd, "--total-steps", cli_args.total_steps)
    append_optional_arg(cmd, "--random-steps", cli_args.random_steps)
    append_optional_arg(cmd, "--update-after", cli_args.update_after)
    append_optional_arg(cmd, "--update-every", cli_args.update_every)
    append_optional_arg(cmd, "--updates-per-step", cli_args.updates_per_step)
    append_optional_arg(cmd, "--eval-every", cli_args.eval_every)
    append_optional_arg(cmd, "--eval-episodes", cli_args.eval_episodes)
    append_optional_arg(cmd, "--log-every-episodes", cli_args.log_every_episodes)
    append_optional_arg(cmd, "--device", cli_args.device)
    append_optional_arg(cmd, "--checkpoint-every", cli_args.checkpoint_every)
    append_optional_arg(cmd, "--seed", seed)
    append_optional_arg(cmd, "--save-dir", save_dir)
    append_optional_arg(cmd, "--batch-size", cli_args.batch_size)
    append_optional_arg(cmd, "--replay-capacity", cli_args.replay_capacity)
    append_optional_arg(cmd, "--hidden-dim", cli_args.hidden_dim)
    append_optional_arg(cmd, "--offline-data", getattr(cli_args, "offline_data", None))
    append_optional_arg(cmd, "--offline-ratio", getattr(cli_args, "offline_ratio", None))
    return cmd


def default_suite_root(args: argparse.Namespace, benchmarks: list[BenchmarkSpec]) -> str:
    if args.preset is not None:
        return f"experiments/{args.preset}"
    if args.benchmark_group is not None:
        return f"experiments/{args.benchmark_group}"
    if len(benchmarks) == 1:
        return f"experiments/{benchmarks[0].key}"
    return "experiments/ablation_suite"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run factorized RL suites over benchmark x method x seed.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(SUITE_PRESETS.keys()),
        default=None,
        help="Named experiment preset. Explicit CLI flags still override preset values.",
    )
    parser.add_argument(
        "--suite-root",
        type=str,
        default=None,
        help="Root directory for all benchmark/method/seed run folders.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method keys.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark keys from scripts.benchmark_catalog.",
    )
    parser.add_argument(
        "--benchmark-group",
        type=str,
        choices=sorted(BENCHMARK_GROUPS.keys()),
        default=None,
        help="Named benchmark group from scripts.benchmark_catalog.",
    )
    parser.add_argument(
        "--benchmark-manifest-dir",
        type=str,
        default="benchmarks",
        help="Directory containing standard benchmark manifests.",
    )
    parser.add_argument("--flow", type=str, default=None)
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
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument(
        "--objective",
        type=str,
        choices=sorted(REWARD_OBJECTIVE_PRESETS.keys()),
        default=None,
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default=None,
        help="Comma-separated reward objectives for objective-ablation suites.",
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--random-steps", type=int, default=None)
    parser.add_argument("--update-after", type=int, default=None)
    parser.add_argument("--update-every", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-every-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--offline-data", type=str, default=None)
    parser.add_argument("--offline-ratio", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    apply_preset_defaults(args)

    benchmarks = resolve_suite_benchmarks(args)
    validate_benchmark_mode(args, benchmarks)

    if args.suite_root is None:
        args.suite_root = default_suite_root(args, benchmarks)
    if args.methods is None:
        args.methods = "sac,sac_stack4"
    if args.seeds is None:
        args.seeds = "42,43,44"
    if args.total_steps is None:
        args.total_steps = 50_000
    if args.random_steps is None:
        args.random_steps = 2_000
    if args.update_after is None:
        args.update_after = 2_000
    if args.update_every is None:
        args.update_every = 1
    if args.updates_per_step is None:
        args.updates_per_step = 1
    if args.eval_every is None:
        args.eval_every = 5_000
    if args.eval_episodes is None:
        args.eval_episodes = 20
    if args.log_every_episodes is None:
        args.log_every_episodes = 5
    if args.checkpoint_every is None:
        args.checkpoint_every = 5_000
    if args.batch_size is None:
        args.batch_size = 256
    if args.replay_capacity is None:
        args.replay_capacity = 1_000_000
    if args.hidden_dim is None:
        args.hidden_dim = 256

    if args.objective is not None and args.objectives is not None:
        raise ValueError("Use either --objective or --objectives, not both.")

    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [key for key in method_keys if key not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown method keys: {unknown}")
    if args.objectives is not None:
        objectives = parse_objective_list(args.objectives)
    elif args.objective is not None:
        objectives = [args.objective]
    else:
        objectives = [None]
    unknown_objectives = [
        key for key in objectives if key is not None and key not in REWARD_OBJECTIVE_PRESETS
    ]
    if unknown_objectives:
        raise ValueError(f"Unknown objective keys: {unknown_objectives}")
    seeds = parse_seed_list(args.seeds)
    suite_root = Path(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)

    if benchmarks:
        for benchmark in benchmarks:
            manifest_path = Path(benchmark_manifest_path(benchmark, args.benchmark_manifest_dir))
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing benchmark manifest for {benchmark.key}: {manifest_path}"
                )
    else:
        benchmarks = []

    manifest = {
        "suite_root": str(suite_root),
        "methods": method_keys,
        "seeds": seeds,
        "benchmarks": [
            {
                "key": benchmark.key,
                "description": benchmark.description,
                "flow_path": benchmark.flow_path,
                "task_geometry": benchmark.task_geometry,
                "target_speed": benchmark.target_speed,
                "eval_manifest": benchmark_manifest_path(benchmark, args.benchmark_manifest_dir),
                "factor_values": benchmark.factor_values,
            }
            for benchmark in benchmarks
        ],
        "benchmark_group": args.benchmark_group,
        "task_geometry": args.task_geometry,
        "difficulty": args.difficulty,
        "action_mode": args.action_mode,
        "speed_ratio": args.speed_ratio,
        "target_speed": args.target_speed,
        "objective": args.objective,
        "objectives": objectives,
        "energy_cost_gain": args.energy_cost_gain,
        "safety_cost_gain": args.safety_cost_gain,
        "total_steps": args.total_steps,
        "device": args.device,
        "runs": [],
    }

    benchmark_items: list[BenchmarkSpec | None] = benchmarks if benchmarks else [None]
    for benchmark in benchmark_items:
        for objective in objectives:
            for method_key in method_keys:
                method = METHOD_SPECS[method_key]
                for seed in seeds:
                    if benchmark is None:
                        if objective is None:
                            run_dir = suite_root / method.key / f"seed_{seed}"
                        else:
                            run_dir = suite_root / objective / method.key / f"seed_{seed}"
                    else:
                        if objective is None:
                            run_dir = suite_root / benchmark.key / method.key / f"seed_{seed}"
                        else:
                            run_dir = suite_root / benchmark.key / objective / method.key / f"seed_{seed}"
                    cmd = build_command(
                        method,
                        seed,
                        run_dir,
                        args,
                        objective=objective,
                        benchmark=benchmark,
                    )
                    run_record = {
                        "benchmark": benchmark.key if benchmark is not None else None,
                        "benchmark_description": (
                            benchmark.description if benchmark is not None else None
                        ),
                        "factor_values": benchmark.factor_values if benchmark is not None else None,
                        "objective": objective,
                        "method": method.key,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "train_module": method.train_module,
                        "description": method.description,
                        "command": cmd,
                    }
                    if benchmark is not None:
                        run_record["eval_manifest"] = benchmark_manifest_path(
                            benchmark,
                            args.benchmark_manifest_dir,
                        )
                    manifest["runs"].append(run_record)

                    final_eval_path = run_dir / "final_eval.json"
                    if args.skip_existing and final_eval_path.exists():
                        label_parts = [
                            benchmark.key if benchmark is not None else None,
                            objective,
                            method.key,
                        ]
                        label = " / ".join(part for part in label_parts if part is not None)
                        print(f"[skip] {label} seed={seed} -> {run_dir}")
                        continue

                    label_parts = [
                        benchmark.key if benchmark is not None else None,
                        objective,
                        method.key,
                    ]
                    label = " / ".join(part for part in label_parts if part is not None)
                    print(f"[run] {label} seed={seed}")
                    print("      " + shlex.join(cmd))
                    if args.dry_run:
                        continue

                    run_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(cmd, check=True)

    with (suite_root / "suite_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
