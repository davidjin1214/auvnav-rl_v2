"""
paper/thesis_ch5/figures/scripts/fig_ch5_algo_interaction.py

Figure (§5.9, chapter-level headline): algorithm x data-quality interaction
across the SAC-checkpoint collector quality ladder, plus the direction-flip
contrast panel. The visual point:
  (a) On the four SAC-checkpoint tiers (collector success 0.002 -> 0.899),
      FQL pulls ahead of both ReBRAC-Q configs exactly in the informative
      mid-quality band (med-expert 0.922 vs 0.800), while all three converge
      near the collector's own level at the expert (saturated) tier and all
      floor at the degenerate random tier.
  (b) The FQL - ReBRAC-Q(beta1=1.0) contrast flips sign between the same-source
      mid-quality cells (CI strictly > 0) and the saturated mixed cell reached
      by a cross-source supplement (CI strictly < 0): algorithm ranking is
      regime-dependent, not global. Cross-source side supports DIRECTION only
      (different dataset/manifest); magnitudes across the divider are not
      commensurable (spec §0.4 red line 5) -- hence the visual group divider.

RESULTS figure (§5.9): point height IS the success number.

Data (REAL; do not edit from memory):
  - Panel (a): docs/arrival_v2_sac_collector_design.md §4.0.10 36-cell joint
    matrix (30-ep manifest, test_seed=456, 3 seeds [42,0,7], mean +- std):
      tier      collector  ReBRAC b1=1      ReBRAC b1=4      FQL
      random    0.002      0.000+-0.000     0.000+-0.000     0.000+-0.000
      medium    0.515      0.656+-0.069     0.633+-0.067     0.733+-0.100
      med-exp   0.751      0.800+-0.033     0.711+-0.038     0.922+-0.038
      expert    0.899      0.889+-0.019     0.889+-0.019     0.911+-0.019
  - Panel (b): same doc §4.0.10 stratified paired bootstrap 95% CI
    (N_boot=10000), FQL - ReBRAC-Q beta1=1.0:
      SAC med-expert (same source):        +0.123 [+0.022, +0.233]
      SAC med+mexp+exp aggregate (same):   +0.074 [+0.015, +0.137]
      mixed saturated (cross-source supp): -0.035 [-0.065, -0.010]
        (paired bootstrap, 100-ep manifest, 2 paired seeds [42, 0];
         reported in the doc as ReBRAC-FQL = +0.035 [+0.010, +0.065],
         sign-flipped here to keep one contrast direction on the axis)

Design rules (shared _ch5_style): English-only minimal in-figure text, width
<= 138 mm, vector PDF, muted palette. Hue map shared with the §5.9 noise-axis
figure: ReBRAC-Q beta1=1.0 = mainline dark cool (#2F5A6E); beta1=4.0 = light
cool dashed (#8AA4B2); FQL = muted plum (#7D6B8F); collector reference =
neutral grey dashed triangles (same vocabulary as the §5.6 scale figure).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_algo_interaction.py

Output:
    paper/thesis_ch5/figures/fig_ch5_algo_interaction.{pdf,png}
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
    apply_style,
    in_from_mm,
    panel_label,
)

TIERS = ("random", "medium", "med-expert", "expert")
COLLECTOR = (0.002, 0.515, 0.751, 0.899)

R1_MEAN = (0.000, 0.656, 0.800, 0.889)
R1_STD = (0.000, 0.069, 0.033, 0.019)
R4_MEAN = (0.000, 0.633, 0.711, 0.889)
R4_STD = (0.000, 0.067, 0.038, 0.019)
FQL_MEAN = (0.000, 0.733, 0.922, 0.911)
FQL_STD = (0.000, 0.100, 0.038, 0.019)

# Direction-flip contrasts: FQL - ReBRAC-Q beta1=1.0, mean and 95% CI.
# Order = top row first. The cross-source row is sign-flipped from the doc's
# ReBRAC-FQL orientation so one contrast direction spans the whole axis.
CONTRASTS = (
    ("SAC med-expert", 0.123, 0.022, 0.233, True),
    ("SAC aggregate", 0.074, 0.015, 0.137, True),
    ("saturated mix", -0.035, -0.065, -0.010, False),
)

C_R1 = COLORS["mainline_edge"]  # #2F5A6E
C_R4 = "#8AA4B2"
C_FQL = "#7D6B8F"
C_REF = COLORS["muted"]


def draw_ladder_panel(ax: plt.Axes) -> None:
    xs = np.arange(len(TIERS))

    ax.plot(xs, COLLECTOR, color=C_REF, lw=1.0, linestyle=(0, (5, 3)),
            marker="^", markersize=4.0, markerfacecolor="white",
            markeredgecolor=C_REF, markeredgewidth=0.8, zorder=4,
            label="collector (reference)")
    ax.errorbar(xs, R4_MEAN, yerr=R4_STD, color=C_R4, lw=1.4,
                linestyle=(0, (5, 3)), marker="s", markersize=4.2,
                markerfacecolor=C_R4, markeredgecolor="white",
                markeredgewidth=0.6, capsize=2.6, capthick=0.8,
                elinewidth=0.9, zorder=5, label="ReBRAC-Q $\\beta_1{=}4.0$")
    ax.errorbar(xs, R1_MEAN, yerr=R1_STD, color=C_R1, lw=1.7,
                marker="o", markersize=4.6, markerfacecolor=C_R1,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.6, capthick=0.8, elinewidth=0.9, zorder=6,
                label="ReBRAC-Q $\\beta_1{=}1.0$")
    ax.errorbar(xs, FQL_MEAN, yerr=FQL_STD, color=C_FQL, lw=1.7,
                marker="D", markersize=4.2, markerfacecolor=C_FQL,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2.6, capthick=0.8, elinewidth=0.9, zorder=7,
                label="FQL")

    ax.set_xlim(-0.3, len(TIERS) - 0.7)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{t}\n({c:.3f})" for t, c in zip(TIERS, COLLECTOR)], fontsize=7.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel("SAC-checkpoint tier (collector success)", fontsize=8.0)
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
    leg = ax.legend(loc="lower right", frameon=False, fontsize=6.8,
                    handlelength=1.8, labelspacing=0.30, borderaxespad=0.3)
    leg._legend_box.align = "left"


def draw_flip_panel(ax: plt.Axes) -> None:
    n = len(CONTRASTS)
    ys = np.arange(n)[::-1]  # first contrast on top

    ax.axvline(0.0, color=COLORS["muted"], lw=0.9, zorder=2)
    for (label, mean, lo, hi, fql_wins), y in zip(CONTRASTS, ys):
        color = C_FQL if fql_wins else C_R1
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round",
                zorder=4)
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.10, y + 0.10], color=color, lw=1.2,
                    zorder=4)
        ax.plot([mean], [y], marker="o", markersize=5.0, color=color,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.7, zorder=5)

    # Divider between the same-source pair and the cross-source row: the two
    # sides use different datasets/manifests -- direction is comparable,
    # magnitude is not (spec §0.4 red line 5).
    ax.axhline(ys[1] - 0.5, color=COLORS["ghost"], lw=0.8,
               linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.245, ys[0] + 0.36, "same source", ha="right", va="center",
            fontsize=6.6, color=COLORS["muted"], style="italic")
    ax.text(0.245, ys[2] + 0.36, "cross-source", ha="right", va="center",
            fontsize=6.6, color=COLORS["muted"], style="italic")

    ax.set_xlim(-0.10, 0.25)
    ax.set_ylim(-0.6, n - 0.25)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in CONTRASTS], fontsize=7.0)
    ax.set_xticks(np.arange(-0.1, 0.251, 0.1))
    ax.set_xlabel(
        "$\\Delta$ success (FQL $-$ ReBRAC-Q $\\beta_1{=}1.0$)",
        fontsize=8.0)
    ax.grid(True, axis="x", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6,
                   width=0.75)
    ax.tick_params(axis="y", length=0)


def build_figure(width_mm: float = 138.0) -> plt.Figure:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(in_from_mm(width_mm), in_from_mm(62.0)),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.46},
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.185, top=0.90)
    draw_ladder_panel(ax_a)
    draw_flip_panel(ax_b)
    panel_label(ax_a, "(a)", x=-0.135, y=1.10)
    panel_label(ax_b, "(b)", x=-0.30, y=1.10)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.9 algorithm x data-quality interaction "
                    "headline figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_algo_interaction.pdf")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=138.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved §5.9 interaction headline figure to {args.output}")


if __name__ == "__main__":
    main()
