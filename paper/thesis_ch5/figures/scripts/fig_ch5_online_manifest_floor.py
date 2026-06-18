"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_manifest_floor.py

Figure (§5.5, Fig E): the manifest universal floor. For each of the 30
evaluation episodes, how many of the well-trained deployable runs fail it
(out of bounds / timeout). A small set of episodes fails in EVERY run,
independent of history length or seed -- these define a ceiling intrinsic to the
evaluation set rather than to the policy, so the deployable configuration that
saturates 27/30 has reached the manifest's own limit, not a training shortfall.

RESULTS figure (§5.5): bar height IS the count of failing runs.

Data (REAL final_eval.json eval_episode_results, cross-checked; do not edit from
memory): vanilla s0 runs k=12 {0,7,42} + k=8 {7,42}. Episodes 8, 16, 28 fail in
all five -> ceiling 27/30 = 0.900 (arrival_v2_experiment_report §7.9.7).

Design rules (shared _ch5_style + _ch5_data).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_manifest_floor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_data import episode_fail_matrix  # noqa: E402
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    apply_style,
    in_from_mm,
)

# Well-trained deployable runs whose common failures define the floor.
RUNS = [(12, 0), (12, 7), (12, 42), (8, 7), (8, 42)]


def _counts() -> tuple[np.ndarray, int, list[int]]:
    M = episode_fail_matrix(RUNS)
    n_runs = M.shape[0]
    fail = M.sum(axis=0)
    floor = [i for i in range(M.shape[1]) if fail[i] == n_runs]
    return fail, n_runs, floor


def draw(ax: plt.Axes, fail: np.ndarray, n_runs: int, floor: list[int]) -> None:
    x = np.arange(1, len(fail) + 1)
    colors = [COLORS["online_edge"] if (i in floor) else COLORS["ghost"]
              for i in range(len(fail))]
    ax.bar(x, fail, width=0.74, color=colors, edgecolor=COLORS["ink"],
           linewidth=0.4, zorder=3)
    ax.axhline(n_runs, color=COLORS["muted"], lw=0.8, linestyle=(0, (5, 3)),
               zorder=2)
    ax.text(0.7, n_runs + 0.12, f"all {n_runs} runs", ha="left", va="bottom",
            fontsize=7.0, color=COLORS["muted"], zorder=6)
    n_ep = len(fail)
    ax.text(0.98, 0.93,
            f"universal floor: {len(floor)}/{n_ep} episodes fail in every run\n"
            f"empirical ceiling = {n_ep - len(floor)}/{n_ep} = "
            f"{(n_ep - len(floor)) / n_ep:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
            color=COLORS["ink"], zorder=7, linespacing=1.3)


def style_axes(ax: plt.Axes, n_runs: int, n_ep: int) -> None:
    ax.set_xlim(0.3, n_ep + 0.7)
    ax.set_ylim(0.0, n_runs + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks(range(0, n_runs + 1))
    ax.set_xlabel("evaluation episode", fontsize=8.5)
    ax.set_ylabel("runs failing the episode", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    fail, n_runs, floor = _counts()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(62.0)))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.16, top=0.965)
    style_axes(ax, n_runs, len(fail))
    draw(ax, fail, n_runs, floor)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 manifest universal-floor figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_manifest_floor.pdf")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=MAX_WIDTH_MM)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved manifest universal-floor figure to {args.output}")


if __name__ == "__main__":
    main()
