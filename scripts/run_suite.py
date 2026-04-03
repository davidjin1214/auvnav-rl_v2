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
    "gru_sac": MethodSpec(
        key="gru_sac",
        train_module="scripts.train_gru_residual_sac",
        extra_args=("--disable-residual-prior", "--disable-safety-critic"),
        description="Recurrent SAC without residual prior or safety critic.",
    ),
    "gru_residual_sac": MethodSpec(
        key="gru_residual_sac",
        train_module="scripts.train_gru_residual_sac",
        extra_args=("--disable-safety-critic",),
        description="Recurrent residual SAC without safety critic.",
    ),
    "full": MethodSpec(
        key="full",
        train_module="scripts.train_gru_residual_sac",
        extra_args=(),
        description="Full GRU residual SAC with safety critic.",
    ),
}


SUITE_PRESETS: dict[str, SuitePreset] = {
    "medium_formal_v1": SuitePreset(
        name="medium_formal_v1",
        description=(
            "Recommended formal benchmark on medium difficulty "
            "(cross-stream) with 5 seeds and a common 200k-step budget."
        ),
        values={
            "difficulty": "medium",
            "methods": "sac,sac_stack4,gru_sac,gru_residual_sac,full",
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
            "sac_batch_size": 256,
            "sac_replay_capacity": 1_000_000,
            "sac_hidden_dim": 256,
            "gru_batch_size": 32,
            "burn_in": 8,
            "train_seq_len": 32,
            "replay_max_episodes": 5_000,
            "residual_l2_weight": 1e-3,
            "bc_weight": 0.0,
            "bc_decay_steps": 50_000,
            "offline_policy": "world_compensate",
            "offline_episodes": 0,
            "pretrain_updates": 0,
            "cost_limit": 0.02,
        },
    ),
    "medium_pilot_v1": SuitePreset(
        name="medium_pilot_v1",
        description=(
            "Smaller pilot version for quick checks before the formal suite."
        ),
        values={
            "difficulty": "medium",
            "methods": "sac,sac_stack4,gru_sac,gru_residual_sac,full",
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
            "sac_batch_size": 256,
            "sac_replay_capacity": 1_000_000,
            "sac_hidden_dim": 256,
            "gru_batch_size": 32,
            "burn_in": 8,
            "train_seq_len": 32,
            "replay_max_episodes": 5_000,
            "residual_l2_weight": 1e-3,
            "bc_weight": 0.0,
            "bc_decay_steps": 50_000,
            "offline_policy": "world_compensate",
            "offline_episodes": 0,
            "pretrain_updates": 0,
            "cost_limit": 0.02,
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

    if method.train_module == "scripts.train_sac":
        append_optional_arg(cmd, "--batch-size", cli_args.sac_batch_size)
        append_optional_arg(cmd, "--replay-capacity", cli_args.sac_replay_capacity)
        append_optional_arg(cmd, "--hidden-dim", cli_args.sac_hidden_dim)
    else:
        append_optional_arg(cmd, "--batch-size", cli_args.gru_batch_size)
        append_optional_arg(cmd, "--burn-in", cli_args.burn_in)
        append_optional_arg(cmd, "--train-seq-len", cli_args.train_seq_len)
        append_optional_arg(cmd, "--replay-max-episodes", cli_args.replay_max_episodes)
        append_optional_arg(cmd, "--residual-l2-weight", cli_args.residual_l2_weight)
        append_optional_arg(cmd, "--bc-weight", cli_args.bc_weight)
        append_optional_arg(cmd, "--bc-decay-steps", cli_args.bc_decay_steps)
        append_optional_arg(cmd, "--offline-policy", cli_args.offline_policy)
        append_optional_arg(cmd, "--offline-episodes", cli_args.offline_episodes)
        append_optional_arg(cmd, "--pretrain-updates", cli_args.pretrain_updates)
        append_optional_arg(cmd, "--cost-limit", cli_args.cost_limit)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RL ablation ladder over multiple seeds.")
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
    parser.add_argument("--sac-batch-size", type=int, default=None)
    parser.add_argument("--sac-replay-capacity", type=int, default=None)
    parser.add_argument("--sac-hidden-dim", type=int, default=None)
    parser.add_argument("--gru-batch-size", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--train-seq-len", type=int, default=None)
    parser.add_argument("--replay-max-episodes", type=int, default=None)
    parser.add_argument("--residual-l2-weight", type=float, default=None)
    parser.add_argument("--bc-weight", type=float, default=None)
    parser.add_argument("--bc-decay-steps", type=int, default=None)
    parser.add_argument("--offline-policy", type=str, default=None)
    parser.add_argument("--offline-episodes", type=int, default=None)
    parser.add_argument("--pretrain-updates", type=int, default=None)
    parser.add_argument("--cost-limit", type=float, default=None)
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
        args.methods = "sac,sac_stack4,gru_sac,gru_residual_sac,full"
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
    if args.sac_batch_size is None:
        args.sac_batch_size = 256
    if args.sac_replay_capacity is None:
        args.sac_replay_capacity = 1_000_000
    if args.sac_hidden_dim is None:
        args.sac_hidden_dim = 256
    if args.gru_batch_size is None:
        args.gru_batch_size = 32
    if args.burn_in is None:
        args.burn_in = 8
    if args.train_seq_len is None:
        args.train_seq_len = 32
    if args.replay_max_episodes is None:
        args.replay_max_episodes = 5_000
    if args.residual_l2_weight is None:
        args.residual_l2_weight = 1e-3
    if args.bc_weight is None:
        args.bc_weight = 0.0
    if args.bc_decay_steps is None:
        args.bc_decay_steps = 50_000
    if args.offline_policy is None:
        args.offline_policy = "world_compensate"
    if args.offline_episodes is None:
        args.offline_episodes = 0
    if args.pretrain_updates is None:
        args.pretrain_updates = 0
    if args.cost_limit is None:
        args.cost_limit = 0.02

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
