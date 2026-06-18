"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_bottleneck.py

Figure (§5.5, Fig 5.5): the deployable-performance bottleneck is actor-side
temporal access, not the sensing hardware or the critic's estimate.

Single panel, critical regime (Re=250, U_inf=1.5 m/s, arrival reward, s0).
    - Main line: deployable single-point s0 success rate vs history length k
      (k = 4, 8, 12), 3-seed mean +/- std. The window grows ~2 -> 4 -> 6 s
      (Delta t_ctrl = 0.5 s; secondary top axis). Success climbs from the floor
      toward the upper bound; the spread is widest at k=8 and collapses by k=12
      (sigma 0.038) -> the gain is a systematic, robust trend, not seed noise.
    - Upper-bound line at 0.90: the spatial reference s1(k=4) AND the eval-set
      empirical ceiling (27/30) coincide here.
    - Reverse-axis contrast (warm): from the same k=4 baseline, injecting a
      privileged hull-integral flow estimate into the VALUE network pushes
      success DOWN to 0.066 (2-seed paired) -- the opposite direction to adding
      temporal access. More information on the critic side does not help; the
      bottleneck is the policy's access to single-point observations over time.

RESULTS figure (§5.5): line height / markers ARE the success numbers.

Data (cross-checked; do not edit from memory):
  k-sweep    k=4  PRELIMINARY new 3-seed sensor screen (== Fig 5.4 critical s0);
             k=8/k=12  arrival_v2_experiment_report.md §7.8/§7.9 (seeds 42/0/7).
  ceiling    27/30 = 0.900 (§7.9.7 manifest universal floor).
  privileged §7.7 2-seed paired (vanilla 0.220 -> asym 0.066; peak locked 0.267).

Design rules (shared _ch5_style): English-only minimal text, orthogonal
axes/grid, width <= 138 mm, vector PDF, muted palette; s0 keeps the chapter's
ink hue, s1 the cool reference hue, the critic-side reverse arm the warm
(online) hue.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_bottleneck.py

Output:
    paper/thesis_ch5/figures/fig_ch5_online_bottleneck.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    SENSING,
    apply_style,
    in_from_mm,
)

DT_CTRL_S = 0.5  # control period: history length k -> temporal window k * 0.5 s

# --- s0 success rate vs history length k (critical regime) -----------------
# k=4: PRELIMINARY new 3-seed sensor screen (same condition as Fig 5.4 s0_crit);
# k=8/k=12: §7.8/§7.9 history ablation (seeds 42 / 0 / 7). All n=3, sample std.
K_SEEDS = {
    4: [0.10, 0.35, 0.17],
    8: [0.900, 0.500, 0.867],
    12: [0.900, 0.900, 0.833],
}
KS = sorted(K_SEEDS)

UPPER_BOUND = 0.900       # s1(k=4) reference == empirical ceiling 27/30

# Critic-side reverse arm: privileged obs into the value net (§7.7, 2-seed).
PRIV_BASELINE = 0.220     # vanilla s0, k=4 (2-seed paired mean)
PRIV_VALUE = 0.066        # + privileged critic (2-seed paired mean)
PRIV_X = 3.15             # left margin lane, just before the k-sweep


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def _value_label(ax: plt.Axes, x: float, top: float, text: str,
                 *, dx: float = 0.0, color: str | None = None) -> None:
    ax.text(x + dx, top + 0.022, text, ha="center", va="bottom",
            fontsize=7.0, color=color or COLORS["muted"], zorder=8)


def draw_main_line(ax: plt.Axes) -> None:
    xs = np.array(KS, dtype=float)
    stats = [_mean_std(K_SEEDS[k]) for k in KS]
    means = np.array([m for m, _ in stats])
    stds = np.array([s for _, s in stats])
    ax.plot(xs, means, color=SENSING["s0"], linewidth=1.2, zorder=5,
            solid_capstyle="round")
    ax.errorbar(xs, means, yerr=stds, fmt="o", ms=4.6,
                mfc=SENSING["s0"], mec=SENSING["s0"], ecolor=SENSING["s0"],
                elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=6)
    for x, m, s in zip(xs, means, stds):
        _value_label(ax, x, m + s, f"{m:.2f}")


def draw_upper_bound(ax: plt.Axes) -> None:
    ax.axhline(UPPER_BOUND, color=SENSING["s1"], linewidth=0.9,
               linestyle=(0, (5, 3)), zorder=3)
    ax.text(2.65, UPPER_BOUND + 0.013,
            "s1 ref = ceiling  0.90", ha="left", va="bottom",
            fontsize=7.0, color=SENSING["s1"], zorder=7)


def draw_reverse_arm(ax: plt.Axes) -> None:
    warm = COLORS["online_edge"]
    # label above; vanilla baseline (shares the k=4 baseline ~0.21) -> down to
    # the + privileged-critic outcome.
    ax.text(PRIV_X, PRIV_BASELINE + 0.05, "+ priv.\ncritic", ha="center",
            va="bottom", fontsize=7.0, color=warm, zorder=7, linespacing=0.95)
    ax.scatter([PRIV_X], [PRIV_BASELINE], marker="o", s=26, facecolor="white",
               edgecolor=warm, linewidths=1.1, zorder=6)
    ax.annotate("", xy=(PRIV_X, PRIV_VALUE + 0.018),
                xytext=(PRIV_X, PRIV_BASELINE - 0.012),
                arrowprops=dict(arrowstyle="-|>", color=warm, lw=1.1,
                                mutation_scale=9), zorder=5)
    ax.scatter([PRIV_X], [PRIV_VALUE], marker="v", s=40, color=warm,
               edgecolor=warm, zorder=6)
    ax.text(PRIV_X - 0.16, PRIV_VALUE, "0.066", ha="right", va="center",
            fontsize=7.0, color=warm, zorder=7)


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(2.5, 12.9)
    ax.set_ylim(0.0, 1.16)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel("history length  k", fontsize=8.5)
    ax.set_ylabel("success rate", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)

    # secondary top axis: temporal window in seconds (k * Delta t_ctrl)
    sec = ax.secondary_xaxis("top", functions=(lambda k: k * DT_CTRL_S,
                                               lambda s: s / DT_CTRL_S))
    sec.set_xticks([k * DT_CTRL_S for k in KS])
    sec.set_xlabel("temporal window  [s]", fontsize=8.0, color=COLORS["muted"])
    sec.tick_params(colors=COLORS["muted"], direction="out", length=2.4,
                    width=0.7, labelsize=7.0)
    sec.spines["top"].set_color(COLORS["muted"])


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    width_in = in_from_mm(width_mm)
    height_in = in_from_mm(74.0)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.115, top=0.875)
    style_axes(ax)
    draw_upper_bound(ax)
    draw_reverse_arm(ax)
    draw_main_line(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 temporal-bottleneck figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_bottleneck.pdf")
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
    print(f"Saved temporal-bottleneck figure to {args.output}")


if __name__ == "__main__":
    main()
