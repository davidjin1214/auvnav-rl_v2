"""Select and evaluate the best checkpoint from a training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _resolve_existing(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = run_dir / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of the expected paths exist under {run_dir}: {', '.join(relative_paths)}"
    )


def _resolve_saved_path(run_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return run_dir / path


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_eval_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def best_eval_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("Evaluation log is empty.")
    return max(
        rows,
        key=lambda row: (
            float(row["eval_success_rate"]),
            float(row["eval_return"]),
            -float(row["eval_safety_cost"]),
            -float(row["eval_time_s"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best periodic evaluation checkpoint and print the matching evaluate.py command.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing periodic evaluation logs and checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the best-checkpoint metadata.",
    )
    args = parser.parse_args()

    eval_log_path = _resolve_existing(
        args.run_dir,
        "results/eval_log.csv",
        "eval_log.csv",
    )

    row = best_eval_row(load_eval_rows(eval_log_path))
    env_step = int(row["env_step"])
    trainer_state_path = args.run_dir / "trainer_state.json"
    trainer_state: dict[str, Any] = {}
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as fp:
            trainer_state = json.load(fp)

    candidate_bases = [args.run_dir / "checkpoints", args.run_dir]
    checkpoint_dir = _resolve_saved_path(args.run_dir, trainer_state.get("checkpoint_dir"))
    if checkpoint_dir is not None:
        candidate_bases.append(checkpoint_dir)
    for key in ("latest_agent_path", "best_agent_path", "final_agent_path", "agent_path"):
        resolved = _resolve_saved_path(args.run_dir, trainer_state.get(key))
        if resolved is not None:
            candidate_bases.append(resolved.parent)
    candidate_bases = _unique_paths(candidate_bases)

    agent_path = None
    for base_dir in candidate_bases:
        for filename in (
            f"agent_step_{env_step}.pt",
            f"agent_step_{env_step:08d}.pt",
        ):
            candidate = base_dir / filename
            if candidate.exists():
                agent_path = candidate
                break
        if agent_path is not None:
            break
    if agent_path is None:
        searched = ", ".join(str(path) for path in candidate_bases)
        raise FileNotFoundError(
            f"Could not find periodic checkpoint for env_step={env_step} under: {searched}"
        )
    try:
        agent_file = str(agent_path.relative_to(args.run_dir))
    except ValueError:
        agent_file = str(agent_path)

    payload: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "best_env_step": env_step,
        "agent_file": agent_file,
        "selection_metric": "eval_success_rate -> eval_return -> -eval_safety_cost -> -eval_time_s",
        "eval_row": {
            key: (float(value) if key != "reward_objective" else value)
            for key, value in row.items()
        },
        "evaluate_command": [
            "python",
            "-m",
            "scripts.evaluate",
            "--checkpoint",
            str(args.run_dir),
            "--agent-file",
            agent_file,
        ],
    }

    print(f"run_dir              : {args.run_dir}")
    print(f"best_env_step        : {env_step}")
    print(f"agent_file           : {agent_file}")
    print(f"best_success_rate    : {float(row['eval_success_rate']):.4f}")
    print(f"best_eval_return     : {float(row['eval_return']):.4f}")
    print(f"best_eval_safety     : {float(row['eval_safety_cost']):.4f}")
    print(f"best_eval_time_s     : {float(row['eval_time_s']):.4f}")
    print("evaluate_command     :")
    print(
        "  python -m scripts.evaluate "
        f"--checkpoint {args.run_dir} --agent-file {agent_file}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
