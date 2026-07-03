"""
paper/thesis_ch5/figures/scripts/fig_ch5_td3bc_size.py

Figure (§5.6, bottleneck one): deployable offline success rate against offline
data scale, for the behaviour-constrained baseline TD3+BC and the same-budget
pure behaviour-cloning control, with the collector's own success rate as a
reference. The point of the figure is the DISSOCIATION between three trends on
one canvas:
  - the collector's trajectories get slightly CLEANER with scale (gentle rise),
  - yet both TD3+BC and pure BC peak at the middle scale and FALL at the largest
    scale (inverted-U).
"more (cleaner) data, worse deployable policy" -- the support-set-structure
bottleneck -- lands visually here in a way the table cannot.

RESULTS figure (§5.6): point height IS the success number.

Data (REAL, from docs/td3bc_phase0c_experiment_report.md; do not edit from
memory). Formal 5-seed, 100 test episodes/seed, terminal (final-checkpoint)
evaluation, s0 + k=4, history-efficiency reward, crosscomp collector:
  TD3+BC success  (report §5.3): 0.592+-0.119 / 0.672+-0.045 / 0.596+-0.036
  pure BC success (report §5.4): 0.524+-0.053 / 0.588+-0.070 / 0.534+-0.079
  collector succ. (report §3.2): 0.864 / 0.870 / 0.886  (dataset stat, no per-seed std)
  scales: 500 / 1000 / 2000 episodes.

Design rules (shared _ch5_style): English-only minimal in-figure text (semantics
in caption + body), width <= 138 mm, vector PDF, muted palette. Offline methods
use the cool branch and are separated by LIGHTNESS (greyscale-safe): TD3+BC dark
cool, pure BC light cool; the collector reference is a neutral grey dashed line.
No warm hue here -- warm stays reserved for the §5.5 privileged-critic figure.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_td3bc_size.py

Output:
    paper/thesis_ch5/figures/fig_ch5_td3bc_size.{pdf,png}
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
    SENSING,
    apply_style,
    in_from_mm,
)

SCALES = (500, 1000, 2000)

# Formal 5-seed terminal-evaluation success, mean and std (report §5.3/§5.4).
TD3BC_MEAN = (0.592, 0.672, 0.596)
TD3BC_STD = (0.119, 0.045, 0.036)
BC_MEAN = (0.524, 0.588, 0.534)
BC_STD = (0.053, 0.070, 0.079)
# Collector own-trajectory success = dataset statistic, no per-seed std (§3.2).
COLLECTOR = (0.864, 0.870, 0.886)

# Offline cool branch, separated by lightness so the two curves are also legible
# in greyscale; collector is a neutral grey dashed reference (not a data curve).
#   NB: mainline_edge (#2F5A6E) is reserved for the §5.7 ReBRAC-Q mainline
#   figures; the TD3+BC baseline here uses the plain offline-branch edge so the
#   two never share a hue across adjacent sections.
C_TD3BC = COLORS["offline_edge"]  # #5E8597 cool offline-branch edge (baseline)
C_BC = SENSING["s2"]      # light cool "#8AA4B2" (pure imitation control)
C_REF = COLORS["muted"]   # neutral grey (collector reference)


def draw(ax: plt.Axes) -> None:
    xs = np.arange(len(SCALES))

    # Collector reference line first (sits behind the policy curves).
    ax.plot(xs, COLLECTOR, color=C_REF, lw=1.0, linestyle=(0, (5, 3)),
            marker="^", markersize=4.0, markerfacecolor="white",
            markeredgecolor=C_REF, markeredgewidth=0.8, zorder=4,
            label="collector (reference)")

    # TD3+BC and pure BC with std error bars.
    ax.errorbar(xs, TD3BC_MEAN, yerr=TD3BC_STD, color=C_TD3BC, lw=1.7,
                marker="o", markersize=4.8, markerfacecolor=C_TD3BC,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.8, capthick=0.8, elinewidth=0.9, zorder=6,
                label="TD3+BC")
    ax.errorbar(xs, BC_MEAN, yerr=BC_STD, color=C_BC, lw=1.5,
                marker="s", markersize=4.4, markerfacecolor=C_BC,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.8, capthick=0.8, elinewidth=0.9, zorder=5,
                label="behaviour cloning")


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-0.35, len(SCALES) - 0.65)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(SCALES)))
    ax.set_xticklabels([str(s) for s in SCALES])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel("dataset size (episodes)", fontsize=8.5)
    ax.set_ylabel("success rate", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)


def add_legend(ax: plt.Axes) -> None:
    leg = ax.legend(loc="lower center", frameon=False, fontsize=7.5,
                    handlelength=1.8, labelspacing=0.32, borderaxespad=0.5,
                    ncol=1)
    leg._legend_box.align = "left"


def build_figure(width_mm: float = 90.0) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(66.0)))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.135, top=0.965)
    style_axes(ax)
    draw(ax)
    add_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.6 data-scale inverted-U figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_td3bc_size.pdf")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved §5.6 data-scale figure to {args.output}")


if __name__ == "__main__":
    main()
