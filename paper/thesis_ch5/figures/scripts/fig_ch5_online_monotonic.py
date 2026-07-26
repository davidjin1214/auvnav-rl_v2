"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_monotonic.py

Figure (§5.5, Fig C): the dose-response view of the temporal-access ablation.
Final success rate rises with history length k in the seed-aggregate trend, and
the spread across repeated runs contracts. ALL THREE per-seed trajectories are
drawn: two never fall back across k = 4 -> 8 -> 12, one sits flat at a high
level and then dips slightly. The discriminating evidence is the SAME-SEED
cross-k comparison (seed 0: stalled at 0.50 with k = 8, lifted to the 0.90
ceiling at k = 12 with the initialisation and every other condition held fixed)
-- an initialisation-induced local optimum is not undone by changing the history
length, and optimisation noise does not produce a jump of that form. The
contraction of spread is consistent with this picture but is confounded near
k = 12 by the manifest ceiling and by the evaluation's own sampling floor
(sqrt(0.9*0.1/30) = 0.055 > the reported 0.04), so it is reported as a
robustness side-observation, not as the discriminator.

Complements Fig A (learning curves over training steps): this is the summary
dose-response over the controlled variable.

RESULTS figure (§5.5): marker height IS the success number.

Data (REAL final_eval.json, the locked-table 'final' convention; cross-checked):
  aggregate  k=4/k=8/k=12 seeds {0,7,42}.
  per-seed    seed 0:  0.40 -> 0.50  -> 0.90  (strictly increasing)
              seed 7:  0.867 -> 0.867 -> 0.833 (flat, then a slight dip)
              seed 42: 0.10 -> 0.90  -> 0.90  (rises, then flat at the ceiling)
  ceiling 0.90 = manifest empirical ceiling (27/30).

rev (2026-07-24, chapter-review batch 4, finding H7): the earlier version drew a
single "representative run" -- which was seed 0, the ONLY strictly monotone seed
of the three -- while the prose leaned on "the climb recurs within a single run"
to exclude the competing explanations. That foregrounded a hand-picked best
case. All three seeds are now drawn and the argument in §5.5.4 rests on the
variance contraction (sigma 0.39 -> 0.22 -> 0.04) instead.

Design rules (shared _ch5_style + _ch5_data).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_monotonic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_data import final_eval  # noqa: E402
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    SENSING,
    apply_style,
    in_from_mm,
)

UPPER_BOUND = 0.900
DT_CTRL_S = 0.5
SEEDS = [0, 7, 42]
K_VALUES = [4, 8, 12]
# Critical 3-seed final_eval for the deployable k=4 baseline (seeds 0/7/42),
# kept as an explicit locked list so the k=4 aggregate matches the bottleneck
# table (0.46 +/- 0.39). Ground truth: docs/arrival_v2_experiment_report.md
# §7.10 (9/9 verified). k=8/k=12 read their three local final_eval.json files.
CRIT_K4_S0 = [0.40, 0.867, 0.10]


def _per_seed() -> tuple[np.ndarray, np.ndarray]:
    """(ks, matrix[n_seeds, n_ks]) of final success rate."""
    rows = []
    for i, s in enumerate(SEEDS):
        rows.append([
            CRIT_K4_S0[i] if k == 4
            else final_eval("sac_vanilla", k, s)["eval_success_rate"]
            for k in K_VALUES
        ])
    return np.array(K_VALUES, float), np.asarray(rows, float)


def _agg() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks, mat = _per_seed()
    return ks, mat.mean(axis=0), mat.std(axis=0, ddof=1)


def draw(ax: plt.Axes) -> None:
    ks, m, s = _agg()
    ax.fill_between(ks, m - s, m + s, color=SENSING["s0"], alpha=0.12, lw=0,
                    zorder=2)
    ax.plot(ks, m, color=SENSING["s0"], lw=1.7, marker="o", ms=5.2,
            mfc=SENSING["s0"], mec="white", mew=0.7, zorder=6,
            label="seed-aggregate")
    for x, y in zip(ks, m):
        ax.text(x, y + 0.045, f"{y:.2f}", ha="center", va="bottom",
                fontsize=7.0, color=COLORS["ink"], zorder=8)

    rk, mat = _per_seed()
    for j, row in enumerate(mat):
        ax.plot(rk, row, color=COLORS["muted"], lw=0.9, marker="o", ms=3.2,
                mfc="white", mec=COLORS["muted"], mew=0.8,
                linestyle=(0, (4, 2)), zorder=5,
                label="individual runs" if j == 0 else None)


def draw_ceiling(ax: plt.Axes) -> None:
    ax.axhline(UPPER_BOUND, color=COLORS["muted"], lw=0.9,
               linestyle=(0, (5, 3)), zorder=3)
    ax.text(4.15, UPPER_BOUND + 0.016, "empirical ceiling  0.90",
            ha="left", va="bottom", fontsize=7.0, color=COLORS["muted"], zorder=6)


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(3.4, 12.6)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([4, 8, 12])
    ax.set_xticklabels(["4", "8", "12"])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel("history length  k", fontsize=8.5)
    ax.set_ylabel("final success rate", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)
    sec = ax.secondary_xaxis("top", functions=(lambda k: k * DT_CTRL_S,
                                               lambda t: t / DT_CTRL_S))
    sec.set_xticks([4 * DT_CTRL_S, 8 * DT_CTRL_S, 12 * DT_CTRL_S])
    sec.set_xlabel("temporal window  [s]", fontsize=8.0, color=COLORS["muted"])
    sec.tick_params(colors=COLORS["muted"], direction="out", length=2.4,
                    width=0.7, labelsize=7.0)
    sec.spines["top"].set_color(COLORS["muted"])


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(70.0)))
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.14, top=0.86)
    style_axes(ax)
    draw_ceiling(ax)
    draw(ax)
    leg = ax.legend(loc="center right", frameon=False, fontsize=7.5,
                    handlelength=1.9, labelspacing=0.4, borderaxespad=0.8)
    leg._legend_box.align = "left"
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 monotonic dose-response figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_monotonic.pdf")
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
    print(f"Saved monotonic dose-response figure to {args.output}")


if __name__ == "__main__":
    main()
