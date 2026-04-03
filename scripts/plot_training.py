from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_train_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SAC training logs.")
    # TODO: Add multi-run aggregation with mean/std envelopes so repeated-seed
    # experiments can be visualized directly from a parent directory.
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Directory containing train_log.jsonl and eval_log.csv.")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional output image path. If omitted, show interactively.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib.") from exc

    run_dir = Path(args.run_dir)
    train_rows = load_train_log(run_dir / "train_log.jsonl")
    eval_rows = load_eval_log(run_dir / "eval_log.csv")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    if train_rows:
        env_steps = np.array([row["env_step"] for row in train_rows], dtype=float)
        returns = np.array([row["return"] for row in train_rows], dtype=float)
        episode_cost = np.array([row.get("episode_cost", np.nan) for row in train_rows], dtype=float)
        success = np.array([float(row["success"]) for row in train_rows], dtype=float)
        axes[0, 0].plot(env_steps, returns, lw=1.0, color="tab:blue")
        axes[0, 0].set_title("Episode Return")
        axes[0, 0].set_xlabel("Env Step")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)

        if len(success) >= 10:
            window = min(25, len(success))
            kernel = np.ones(window, dtype=float) / window
            success_smooth = np.convolve(success, kernel, mode="valid")
            axes[0, 1].plot(env_steps[window - 1:], success_smooth, lw=1.2, color="tab:green")
        else:
            axes[0, 1].plot(env_steps, success, lw=1.0, color="tab:green")
        axes[0, 1].set_title("Success Rate (smoothed)")
        axes[0, 1].set_xlabel("Env Step")
        axes[0, 1].set_ylabel("Success")
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].grid(True, alpha=0.3)

        if "alpha" in train_rows[-1]:
            alpha = np.array([row.get("alpha", np.nan) for row in train_rows], dtype=float)
            axes[1, 0].plot(env_steps, alpha, lw=1.0, color="tab:red", label="alpha")
            if "bc_weight" in train_rows[-1]:
                bc_weight = np.array([row.get("bc_weight", np.nan) for row in train_rows], dtype=float)
                axes[1, 0].plot(env_steps, bc_weight, lw=1.0, color="tab:purple", label="bc_weight")
            if "lagrange_lambda" in train_rows[-1]:
                lagrange = np.array([row.get("lagrange_lambda", np.nan) for row in train_rows], dtype=float)
                axes[1, 0].plot(env_steps, lagrange, lw=1.0, color="tab:brown", label="lambda")
            axes[1, 0].set_title("Temperature / BC Weight")
            axes[1, 0].set_xlabel("Env Step")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        if np.isfinite(episode_cost).any():
            axes[0, 2].plot(env_steps, episode_cost, lw=1.0, color="tab:red")
            axes[0, 2].set_title("Episode Cost")
            axes[0, 2].set_xlabel("Env Step")
            axes[0, 2].set_ylabel("Cost")
            axes[0, 2].grid(True, alpha=0.3)

    if eval_rows:
        eval_steps = np.array([row["env_step"] for row in eval_rows], dtype=float)
        eval_return = np.array([row["eval_return"] for row in eval_rows], dtype=float)
        eval_cost = np.array([row.get("eval_cost", np.nan) for row in eval_rows], dtype=float)
        eval_success = np.array([row["eval_success_rate"] for row in eval_rows], dtype=float)
        scale = max(1.0, float(np.max(np.abs(eval_return))))
        axes[1, 1].plot(eval_steps, eval_return, lw=1.2, color="tab:orange", label="eval_return")
        axes[1, 1].plot(eval_steps, eval_success * scale,
                        lw=1.0, color="tab:green", alpha=0.7, label="scaled eval_success")
        axes[1, 1].set_title("Evaluation")
        axes[1, 1].set_xlabel("Env Step")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        if np.isfinite(eval_cost).any():
            axes[1, 2].plot(eval_steps, eval_cost, lw=1.2, color="tab:red")
            axes[1, 2].set_title("Eval Cost")
            axes[1, 2].set_xlabel("Env Step")
            axes[1, 2].grid(True, alpha=0.3)

    # Hide unused panels when logs are sparse.
    for ax in (axes[0, 2], axes[1, 2]):
        if not ax.has_data():
            ax.axis("off")

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
