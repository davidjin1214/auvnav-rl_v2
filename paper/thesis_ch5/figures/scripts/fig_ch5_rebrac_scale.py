"""
paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_scale.py

Figure (SS5.7.1, finding one): the data-scale degradation FLIP. Same axis as
fig_ch5_td3bc_size (SS5.6): deployable offline success rate against offline
data scale on the crosscomp collector. The single-sided baselines (TD3+BC and
same-budget pure BC) peak at the middle scale and FALL at 2000 episodes; the
dual-sided ReBRAC-Q mainline sits far above both and does NOT fall at 2000
(point estimate slightly higher). One canvas shows that the "more data hurts"
degradation is algorithm-relative, not intrinsic to the data.

RESULTS figure (SS5.7): point height IS the success number.

Data (REAL; do not edit from memory):
  ReBRAC-Q success (docs/rebrac_experiment_report.md rev.8 SS7.7, formal
      5-seed x test=100, terminal evaluation, (beta1, beta2) = (4.0, 2.0)):
      1000 eps: 0.902 +- 0.021 ; 2000 eps: 0.918 +- 0.030.
      ReBRAC-Q was NOT trained on the 500-episode set (screening started at
      1000, report SS6), so its series spans 1000-2000 only.
  TD3+BC success  (docs/td3bc_phase0c_experiment_report.md SS5.3, same protocol,
      per-scale posterior-selected alpha): 0.592+-0.119 / 0.672+-0.045 /
      0.596+-0.036 at 500/1000/2000.
  pure BC success (same report SS5.4): 0.524+-0.053 / 0.588+-0.070 /
      0.534+-0.079.
  collector succ. (same report SS3.2): 0.864 / 0.870 / 0.886 (dataset
      statistic, no per-seed std).
  The three baseline series are identical to fig_ch5_td3bc_size (SS5.6) so the
  two figures share one visual axis across sections.

Design rules (shared _ch5_style): English-only minimal in-figure text, width
<= 138 mm, vector PDF, muted palette. ReBRAC-Q uses the mainline_edge hue
(#2F5A6E) reserved for the SS5.7 mainline; TD3+BC keeps the plain offline
branch edge and pure BC the light cool, exactly as in fig_ch5_td3bc_size, so
a reader flipping between SS5.6 and SS5.7 tracks each method by hue.

NB: the 2000-vs-1000 difference within the ReBRAC-Q series (0.918 vs 0.902)
carries no significance test (report SS7.7); the figure's message is the flip
of the trend and the large level gap, both robust.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_scale.py

Output:
    paper/thesis_ch5/figures/fig_ch5_rebrac_scale.{pdf,png}
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

# ReBRAC-Q formal 5-seed terminal evaluation (report rev.8 SS7.7); no 500-ep
# cell exists, the series starts at index 1 (1000 episodes).
REBRAC_MEAN = (0.902, 0.918)
REBRAC_STD = (0.021, 0.030)
REBRAC_XIDX = (1, 2)

# Baselines identical to fig_ch5_td3bc_size (td3bc report SS5.3/SS5.4/SS3.2).
TD3BC_MEAN = (0.592, 0.672, 0.596)
TD3BC_STD = (0.119, 0.045, 0.036)
BC_MEAN = (0.524, 0.588, 0.534)
BC_STD = (0.053, 0.070, 0.079)
COLLECTOR = (0.864, 0.870, 0.886)

C_REBRAC = COLORS["mainline_edge"]  # #2F5A6E — reserved SS5.7 mainline hue
C_TD3BC = COLORS["offline_edge"]    # #5E8597 — same as fig_ch5_td3bc_size
C_BC = SENSING["s2"]                # #8AA4B2 — same as fig_ch5_td3bc_size
C_REF = COLORS["muted"]             # neutral grey collector reference


def draw(ax: plt.Axes) -> None:
    xs = np.arange(len(SCALES))

    # Collector reference line first (sits behind the policy curves).
    ax.plot(xs, COLLECTOR, color=C_REF, lw=1.0, linestyle=(0, (5, 3)),
            marker="^", markersize=4.0, markerfacecolor="white",
            markeredgecolor=C_REF, markeredgewidth=0.8, zorder=4,
            label="dataset collection succ.")

    ax.errorbar(xs, TD3BC_MEAN, yerr=TD3BC_STD, color=C_TD3BC, lw=1.5,
                marker="o", markersize=4.4, markerfacecolor=C_TD3BC,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.8, capthick=0.8, elinewidth=0.9, zorder=5,
                label="TD3+BC")
    ax.errorbar(xs, BC_MEAN, yerr=BC_STD, color=C_BC, lw=1.4,
                marker="s", markersize=4.2, markerfacecolor=C_BC,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.8, capthick=0.8, elinewidth=0.9, zorder=5,
                label="behaviour cloning")

    # ReBRAC-Q mainline on top, heavier weight, mainline hue; series spans
    # 1000-2000 only (no 500-episode training cell).
    rx = np.asarray(REBRAC_XIDX, dtype=float)
    ax.errorbar(rx, REBRAC_MEAN, yerr=REBRAC_STD, color=C_REBRAC, lw=2.0,
                marker="D", markersize=4.8, markerfacecolor=C_REBRAC,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.8, capthick=0.8, elinewidth=0.9, zorder=7,
                label="ReBRAC-Q (ours)")


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
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6,
                   width=0.75)


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
        description="Chapter 5 SS5.7.1 data-scale flip figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_rebrac_scale.pdf")
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
    print(f"Saved SS5.7.1 data-scale flip figure to {args.output}")


if __name__ == "__main__":
    main()
