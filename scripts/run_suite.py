from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
}


SUITE_PRESETS: dict[str, SuitePreset] = {
    "medium_formal_v1": SuitePreset(
        name="medium_formal_v1",
        description=(
            "Formal benchmark on medium difficulty "
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
        description="Smaller pilot version for quick checks before the formal suite.",
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
}


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


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


def build_command(
    method: MethodSpec,
    seed: int,
    save_dir: Path,
    cli_args: argparse.Namespace,
) -> list[str]:
    cmd = [sys.executable, "-m", method.train_module, *method.extra_args]
    append_optional_arg(cmd, "--flow", cli_args.flow)
    append_optional_arg(cmd, "--difficulty", cli_args.difficulty)
    append_optional_arg(cmd, "--task-geometry", cli_args.task_geometry)
    append_optional_arg(cmd, "--action-mode", cli_args.action_mode)
    append_optional_arg(cmd, "--speed-ratio", cli_args.speed_ratio)
    append_optional_arg(cmd, "--target-speed", cli_args.target_speed)
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
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAC ablation suite over multiple seeds.")
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
        help="Root directory for all method/seed run folders.",
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    apply_preset_defaults(args)
    if args.suite_root is None:
        if args.preset is not None:
            args.suite_root = f"experiments/{args.preset}"
        else:
            args.suite_root = "experiments/ablation_suite"
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

    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [key for key in method_keys if key not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown method keys: {unknown}")
    seeds = parse_seed_list(args.seeds)
    suite_root = Path(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_root": str(suite_root),
        "methods": method_keys,
        "seeds": seeds,
        "task_geometry": args.task_geometry,
        "difficulty": args.difficulty,
        "action_mode": args.action_mode,
        "speed_ratio": args.speed_ratio,
        "target_speed": args.target_speed,
        "total_steps": args.total_steps,
        "device": args.device,
        "runs": [],
    }

    for method_key in method_keys:
        method = METHOD_SPECS[method_key]
        for seed in seeds:
            run_dir = suite_root / method.key / f"seed_{seed}"
            cmd = build_command(method, seed, run_dir, args)
            run_record = {
                "method": method.key,
                "seed": seed,
                "run_dir": str(run_dir),
                "train_module": method.train_module,
                "description": method.description,
                "command": cmd,
            }
            manifest["runs"].append(run_record)

            final_eval_path = run_dir / "final_eval.json"
            if args.skip_existing and final_eval_path.exists():
                print(f"[skip] {method.key} seed={seed} -> {run_dir}")
                continue

            print(f"[run] {method.key} seed={seed}")
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
