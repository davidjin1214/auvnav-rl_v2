from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_METHOD_ORDER = [
    "sac",
    "sac_stack4",
]

_METHOD_STYLE = {
    "sac": {
        "label": "SAC",
        "color": "#A6A6A6",
    },
    "sac_stack4": {
        "label": "SAC + Stack",
        "color": "#4C78A8",
    },
}


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def sort_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order_map = {name: idx for idx, name in enumerate(_METHOD_ORDER)}
    return sorted(rows, key=lambda row: order_map.get(str(row["method"]), 10_000))


def method_label(method: str) -> str:
    return _METHOD_STYLE.get(method, {}).get("label", method)


def method_color(method: str) -> str:
    return _METHOD_STYLE.get(method, {}).get("color", "#777777")


def collect_method_points(
    method_rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, np.ndarray]:
    points: dict[str, list[float]] = {}
    for row in method_rows:
        method = str(row["method"])
        value = row.get(metric)
        if isinstance(value, (int, float)) and np.isfinite(value):
            points.setdefault(method, []).append(float(value))
    return {
        method: np.asarray(values, dtype=float)
        for method, values in points.items()
    }


def panel_bar_with_points(
    ax: plt.Axes,
    *,
    summary_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    metric_mean_key: str,
    metric_std_key: str,
    metric_point_key: str,
    title: str,
    ylabel: str,
    higher_is_better: bool,
) -> None:
    summary_rows = sort_methods(summary_rows)
    point_groups = collect_method_points(run_rows, metric_point_key)
    x = np.arange(len(summary_rows), dtype=float)
    means = np.asarray([float(row[metric_mean_key]) for row in summary_rows], dtype=float)
    stds = np.asarray([float(row[metric_std_key]) for row in summary_rows], dtype=float)
    labels = [method_label(str(row["method"])) for row in summary_rows]
    colors = [method_color(str(row["method"])) for row in summary_rows]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=4.0,
        width=0.72,
        zorder=2,
    )

    rng = np.random.default_rng(12345)
    for idx, row in enumerate(summary_rows):
        method = str(row["method"])
        values = point_groups.get(method)
        if values is None or values.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(
            np.full(values.shape, x[idx]) + jitter,
            values,
            s=24,
            color="white",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.95,
            zorder=3,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Highlight the best bar by the metric direction to make the paper figure scan faster.
    if means.size > 0:
        best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
        bars[best_idx].set_linewidth(1.6)


def add_suite_annotation(fig: plt.Figure, suite_root: Path, run_rows: list[dict[str, Any]]) -> None:
    methods = sorted({str(row["method"]) for row in run_rows})
    seeds = sorted({int(row["seed"]) for row in run_rows if "seed" in row and row["seed"] != ""})
    annotation = (
        f"Suite: {suite_root.name} | Methods: {len(methods)} | Seeds: "
        f"{','.join(str(seed) for seed in seeds) if seeds else 'n/a'}"
    )
    fig.text(0.01, 0.01, annotation, fontsize=9, color="#444444")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style ablation summary figures.")
    parser.add_argument(
        "--suite-root",
        type=str,
        required=True,
        help="Directory containing ablation_summary.csv and ablation_runs.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output prefix. Defaults to <suite-root>/ablation_overview.",
    )
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    summary_rows = load_csv_rows(suite_root / "ablation_summary.csv")
    run_rows = load_csv_rows(suite_root / "ablation_runs.csv")
    if not summary_rows:
        raise FileNotFoundError(
            f"No summary rows found under {suite_root}. Run summarize_ablation_suite first."
        )

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else suite_root / "ablation_overview"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10.0,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    fig.suptitle("Ablation Study on Medium-Difficulty Wake Navigation", fontsize=14, fontweight="bold")

    panel_bar_with_points(
        axes[0, 0],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_success_rate_mean",
        metric_std_key="eval_success_rate_std",
        metric_point_key="eval_success_rate",
        title="Success Rate",
        ylabel="Success Rate",
        higher_is_better=True,
    )
    axes[0, 0].set_ylim(-0.02, 1.05)

    panel_bar_with_points(
        axes[0, 1],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_return_mean",
        metric_std_key="eval_return_std",
        metric_point_key="eval_return",
        title="Evaluation Return",
        ylabel="Return",
        higher_is_better=True,
    )

    panel_bar_with_points(
        axes[1, 0],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_cost_mean",
        metric_std_key="eval_cost_std",
        metric_point_key="eval_cost",
        title="Safety Cost",
        ylabel="Cost",
        higher_is_better=False,
    )

    panel_bar_with_points(
        axes[1, 1],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_time_s_mean",
        metric_std_key="eval_time_s_std",
        metric_point_key="eval_time_s",
        title="Time to Termination",
        ylabel="Seconds",
        higher_is_better=False,
    )

    add_suite_annotation(fig, suite_root, run_rows)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))

    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
