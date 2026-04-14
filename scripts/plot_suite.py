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
    "sac_ln_k16",
    "sac_droq",
    "sac_asym_k16",
    "sac_full",
]

_METHOD_STYLE = {
    "sac": {"label": "SAC", "color": "#A6A6A6"},
    "sac_stack4": {"label": "SAC + Stack", "color": "#4C78A8"},
    "sac_ln_k16": {"label": "SAC + LN", "color": "#72B7B2"},
    "sac_droq": {"label": "DroQ", "color": "#54A24B"},
    "sac_asym_k16": {"label": "SAC + Asym", "color": "#E45756"},
    "sac_full": {"label": "SAC + Full", "color": "#B279A2"},
}

_OBJECTIVE_ORDER = [
    "arrival_v1",
    "efficiency_v1",
]

_OBJECTIVE_STYLE = {
    "arrival_v1": {"label": "Arrival", "color": "#9E9E9E"},
    "efficiency_v1": {"label": "Efficiency", "color": "#2E8B57"},
}

_GAIN_PALETTE = [
    "#4C78A8",
    "#72B7B2",
    "#54A24B",
    "#E45756",
    "#F58518",
    "#B279A2",
    "#FF9DA6",
]


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


def _string_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    return str(value) if value is not None else ""


def infer_plot_mode(summary_rows: list[dict[str, Any]]) -> str:
    objectives = sorted({_string_value(row, "objective") for row in summary_rows if _string_value(row, "objective")})
    methods = sorted({_string_value(row, "method") for row in summary_rows if _string_value(row, "method")})
    benchmarks = sorted({_string_value(row, "benchmark") for row in summary_rows if _string_value(row, "benchmark")})
    gains = sorted({_string_value(row, "gain_label") for row in summary_rows if _string_value(row, "gain_label")})

    if len(objectives) > 1 and len(methods) == 1:
        return "objective"
    if len(gains) > 1 and len(methods) == 1 and len(objectives) <= 1:
        return "gain"
    if len(benchmarks) > 1 and len(methods) == 1 and len(objectives) <= 1:
        return "benchmark"
    return "method"


def _gain_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    safety = row.get("safety_cost_gain")
    energy = row.get("energy_cost_gain")
    gain_label = _string_value(row, "gain_label")
    safety_value = float(safety) if isinstance(safety, (int, float)) else np.inf
    energy_value = float(energy) if isinstance(energy, (int, float)) else np.inf
    return (safety_value, energy_value, gain_label)


def _gain_display_label(row: dict[str, Any]) -> str:
    energy = row.get("energy_cost_gain")
    safety = row.get("safety_cost_gain")
    if isinstance(energy, (int, float)) and isinstance(safety, (int, float)):
        return f"E={energy:.4g}\nS={safety:.4g}"
    return _string_value(row, "gain_label")


def _gain_color(gain_label: str, all_gain_labels: list[str]) -> str:
    if not all_gain_labels:
        return "#4C78A8"
    try:
        idx = all_gain_labels.index(gain_label)
    except ValueError:
        idx = 0
    return _GAIN_PALETTE[idx % len(_GAIN_PALETTE)]


def sort_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    method_order = {name: idx for idx, name in enumerate(_METHOD_ORDER)}
    objective_order = {name: idx for idx, name in enumerate(_OBJECTIVE_ORDER)}

    if mode == "objective":
        return sorted(
            rows,
            key=lambda row: (
                objective_order.get(_string_value(row, "objective"), 10_000),
                method_order.get(_string_value(row, "method"), 10_000),
            ),
        )
    if mode == "gain":
        return sorted(rows, key=_gain_sort_key)
    if mode == "benchmark":
        return sorted(rows, key=lambda row: _string_value(row, "benchmark"))
    return sorted(
        rows,
        key=lambda row: (
            method_order.get(_string_value(row, "method"), 10_000),
            objective_order.get(_string_value(row, "objective"), 10_000),
        ),
    )


def _style_for_row(
    row: dict[str, Any],
    *,
    mode: str,
    context_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    method = _string_value(row, "method")
    objective = _string_value(row, "objective")
    benchmark = _string_value(row, "benchmark")
    gain_label = _string_value(row, "gain_label")

    if mode == "objective":
        style = _OBJECTIVE_STYLE.get(objective, {})
        return style.get("label", objective or method), style.get("color", "#777777")
    if mode == "gain":
        gain_labels = sorted(
            {
                _string_value(candidate, "gain_label")
                for candidate in context_rows
                if _string_value(candidate, "gain_label")
            }
        )
        return _gain_display_label(row), _gain_color(gain_label, gain_labels)
    if mode == "benchmark":
        return benchmark or method, "#4C78A8"

    style = _METHOD_STYLE.get(method, {})
    label = style.get("label", method)
    if objective and len(_OBJECTIVE_ORDER) > 1:
        objective_label = _OBJECTIVE_STYLE.get(objective, {}).get("label", objective)
        label = f"{label} | {objective_label}"
    return label, style.get("color", "#777777")


def collect_variant_points(
    run_rows: list[dict[str, Any]],
    metric: str,
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    points: dict[str, list[float]] = {}
    for row in run_rows:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            continue
        key = variant_key(row, mode=mode)
        points.setdefault(key, []).append(float(value))
    return {
        key: np.asarray(values, dtype=float)
        for key, values in points.items()
    }


def variant_key(row: dict[str, Any], *, mode: str) -> str:
    if mode == "objective":
        return _string_value(row, "objective")
    if mode == "gain":
        return _string_value(row, "gain_label")
    if mode == "benchmark":
        return _string_value(row, "benchmark")
    objective = _string_value(row, "objective")
    method = _string_value(row, "method")
    return f"{objective}::{method}" if objective else method


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
    mode: str,
) -> None:
    summary_rows = sort_rows(summary_rows, mode=mode)
    point_groups = collect_variant_points(run_rows, metric_point_key, mode=mode)
    x = np.arange(len(summary_rows), dtype=float)
    means = np.asarray([float(row[metric_mean_key]) for row in summary_rows], dtype=float)
    stds = np.asarray([float(row[metric_std_key]) for row in summary_rows], dtype=float)
    labels: list[str] = []
    colors: list[str] = []
    for row in summary_rows:
        label, color = _style_for_row(row, mode=mode, context_rows=summary_rows)
        labels.append(label)
        colors.append(color)

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
        key = variant_key(row, mode=mode)
        values = point_groups.get(key)
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

    if means.size > 0:
        best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
        bars[best_idx].set_linewidth(1.6)


def figure_title(summary_rows: list[dict[str, Any]], *, mode: str) -> str:
    benchmarks = sorted({_string_value(row, "benchmark") for row in summary_rows if _string_value(row, "benchmark")})
    methods = sorted({_string_value(row, "method") for row in summary_rows if _string_value(row, "method")})
    objectives = sorted({_string_value(row, "objective") for row in summary_rows if _string_value(row, "objective")})

    if mode == "objective":
        benchmark = benchmarks[0] if len(benchmarks) == 1 else "Multiple Benchmarks"
        method = _METHOD_STYLE.get(methods[0], {}).get("label", methods[0]) if len(methods) == 1 else "Multiple Methods"
        return f"Objective Ablation on {benchmark} with {method}"
    if mode == "gain":
        benchmark = benchmarks[0] if len(benchmarks) == 1 else "Multiple Benchmarks"
        objective_label = (
            _OBJECTIVE_STYLE.get(objectives[0], {}).get("label", objectives[0])
            if len(objectives) == 1
            else "Multiple Objectives"
        )
        method = _METHOD_STYLE.get(methods[0], {}).get("label", methods[0]) if len(methods) == 1 else "Multiple Methods"
        return f"Gain Sweep on {benchmark} under {objective_label} with {method}"
    if mode == "benchmark":
        method = _METHOD_STYLE.get(methods[0], {}).get("label", methods[0]) if len(methods) == 1 else "Multiple Methods"
        return f"Benchmark Transfer Study with {method}"
    if len(objectives) == 1 and objectives:
        objective_label = _OBJECTIVE_STYLE.get(objectives[0], {}).get("label", objectives[0])
        return f"Method Ablation under {objective_label} Objective"
    return "Ablation Study"


def add_suite_annotation(fig: plt.Figure, suite_root: Path, run_rows: list[dict[str, Any]], *, mode: str) -> None:
    methods = sorted({_string_value(row, "method") for row in run_rows if _string_value(row, "method")})
    objectives = sorted({_string_value(row, "objective") for row in run_rows if _string_value(row, "objective")})
    gains = sorted({_string_value(row, "gain_label") for row in run_rows if _string_value(row, "gain_label")})
    seeds = sorted({int(row["seed"]) for row in run_rows if "seed" in row and row["seed"] != ""})
    annotation = (
        f"Suite: {suite_root.name} | Mode: {mode} | Methods: {','.join(methods) if methods else 'n/a'}"
    )
    if objectives:
        annotation += f" | Objectives: {','.join(objectives)}"
    if gains:
        annotation += f" | Gains: {','.join(gains)}"
    annotation += f" | Seeds: {','.join(str(seed) for seed in seeds) if seeds else 'n/a'}"
    fig.text(0.01, 0.01, annotation, fontsize=9, color="#444444")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style suite summary figures.")
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
            f"No summary rows found under {suite_root}. Run summarize_suite first."
        )

    mode = infer_plot_mode(summary_rows)
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
    fig.suptitle(figure_title(summary_rows, mode=mode), fontsize=14, fontweight="bold")

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
        mode=mode,
    )
    axes[0, 0].set_ylim(-0.02, 1.05)

    panel_bar_with_points(
        axes[0, 1],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_time_s_mean",
        metric_std_key="eval_time_s_std",
        metric_point_key="eval_time_s",
        title="Time to Termination",
        ylabel="Seconds",
        higher_is_better=False,
        mode=mode,
    )

    panel_bar_with_points(
        axes[1, 0],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_energy_mean",
        metric_std_key="eval_energy_std",
        metric_point_key="eval_energy",
        title="Energy Consumption",
        ylabel="Energy",
        higher_is_better=False,
        mode=mode,
    )

    panel_bar_with_points(
        axes[1, 1],
        summary_rows=summary_rows,
        run_rows=run_rows,
        metric_mean_key="eval_safety_cost_mean" if "eval_safety_cost_mean" in summary_rows[0] else "eval_cost_mean",
        metric_std_key="eval_safety_cost_std" if "eval_safety_cost_std" in summary_rows[0] else "eval_cost_std",
        metric_point_key="eval_safety_cost" if "eval_safety_cost" in run_rows[0] else "eval_cost",
        title="Safety Cost",
        ylabel="Cost",
        higher_is_better=False,
        mode=mode,
    )

    add_suite_annotation(fig, suite_root, run_rows, mode=mode)
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
