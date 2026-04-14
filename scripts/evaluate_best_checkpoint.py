"""Select and evaluate the best checkpoint from a training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
        help="Training run directory containing eval_log.csv and agent_step_<N>.pt checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the best-checkpoint metadata.",
    )
    args = parser.parse_args()

    eval_log_path = args.run_dir / "eval_log.csv"
    if not eval_log_path.exists():
        raise FileNotFoundError(f"Missing eval_log.csv under {args.run_dir}")

    row = best_eval_row(load_eval_rows(eval_log_path))
    env_step = int(row["env_step"])
    agent_file = f"agent_step_{env_step}.pt"
    agent_path = args.run_dir / agent_file
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {agent_path}")

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
