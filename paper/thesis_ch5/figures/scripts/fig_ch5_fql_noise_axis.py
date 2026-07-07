"""
paper/thesis_ch5/figures/scripts/fig_ch5_fql_noise_axis.py

Figure (§5.9, mechanism subsection): the noise-axis grid that dissolves the
2x2 matrix's single positive cell. Three frozen-per-curve configurations --
ReBRAC-Q beta1=4.0 (screening default), ReBRAC-Q beta1=1.0 (canonical anchor),
FQL (distill weight alpha=1.0) -- evaluated on the clean unit (E-uni) and the
sigma=0.5 action-noise unit (M-uni-noise). The visual point:
  - beta1=4.0 collapses on noisy data (0.885 -> 0.705): raw-action anchoring
    chases injected noise;
  - beta1=1.0 improves on BOTH axes (0.910 / 0.940): no clean/noisy trade-off;
  - FQL sits in between (0.858 / 0.910): its flow-denoised reference needs no
    fix on noisy data but buys nothing on clean data.
Panel (b) shows worst-case-over-noise: 0.910 (beta1=1.0) > 0.858 (FQL) >
0.705 (beta1=4.0) -- the single fixed ReBRAC-Q beta1=1.0 dominates.

RESULTS figure (§5.9): point height IS the success number.

Data (REAL, from docs/fql_succession_p2_results.md §3 noise-axis grid; do not
edit from memory). Fixed 100-episode manifest, terminal-checkpoint evaluation,
n=2 seeds per cell except FQL clean n=4:
  ReBRAC-Q beta1=4.0: clean 0.885, noisy 0.705  -> worst 0.705
  ReBRAC-Q beta1=1.0: clean 0.910, noisy 0.940  -> worst 0.910
  FQL (alpha=1.0):    clean 0.858, noisy 0.910  -> worst 0.858

Design rules (shared _ch5_style): English-only minimal in-figure text, width
<= 138 mm, vector PDF, muted palette. Hue map (kept consistent with the §5.9
interaction figure): ReBRAC-Q beta1=1.0 = mainline dark cool (#2F5A6E);
ReBRAC-Q beta1=4.0 = light cool (#8AA4B2, dashed = de-emphasised default);
FQL = muted plum (#7D6B8F, the generative-prior family hue, used only in §5.9).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_fql_noise_axis.py

Output:
    paper/thesis_ch5/figures/fig_ch5_fql_noise_axis.{pdf,png}
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

# Noise-axis grid (fql_succession_p2_results.md §3): clean / noisy(sigma=0.5).
R4_CLEAN, R4_NOISY = 0.885, 0.705
R1_CLEAN, R1_NOISY = 0.910, 0.940
FQL_CLEAN, FQL_NOISY = 0.858, 0.910

# Worst-case-over-noise per configuration (min over the two axes).
WORST = {
    "ReBRAC-Q $\\beta_1{=}1.0$": min(R1_CLEAN, R1_NOISY),
    "FQL": min(FQL_CLEAN, FQL_NOISY),
    "ReBRAC-Q $\\beta_1{=}4.0$": min(R4_CLEAN, R4_NOISY),
}

C_R1 = COLORS["mainline_edge"]  # #2F5A6E dark cool -- winning fixed config
C_R4 = "#8AA4B2"                # light cool, dashed -- screening default
C_FQL = "#7D6B8F"               # muted plum -- generative-prior family (§5.9)


def draw_axis_panel(ax: plt.Axes) -> None:
    xs = np.array([0.0, 1.0])

    ax.plot(xs, [R4_CLEAN, R4_NOISY], color=C_R4, lw=1.5, linestyle=(0, (5, 3)),
            marker="s", markersize=4.4, markerfacecolor=C_R4,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
            label="ReBRAC-Q $\\beta_1{=}4.0$")
    ax.plot(xs, [FQL_CLEAN, FQL_NOISY], color=C_FQL, lw=1.6,
            marker="D", markersize=4.2, markerfacecolor=C_FQL,
            markeredgecolor="white", markeredgewidth=0.6, zorder=5,
            label="FQL ($\\alpha_{\\mathrm{FQL}}{=}1.0$)")
    ax.plot(xs, [R1_CLEAN, R1_NOISY], color=C_R1, lw=1.8,
            marker="o", markersize=4.8, markerfacecolor=C_R1,
            markeredgecolor="white", markeredgewidth=0.6, zorder=6,
            label="ReBRAC-Q $\\beta_1{=}1.0$")

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(0.65, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["clean ($\\sigma{=}0$)", "noisy ($\\sigma{=}0.5$)"])
    ax.set_yticks(np.arange(0.65, 1.001, 0.05))
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
    leg = ax.legend(loc="lower left", frameon=False, fontsize=7.0,
                    handlelength=1.8, labelspacing=0.32, borderaxespad=0.4)
    leg._legend_box.align = "left"


def draw_worst_panel(ax: plt.Axes) -> None:
    labels = list(WORST.keys())          # top row first
    values = [WORST[k] for k in labels]
    colors = [C_R1, C_FQL, C_R4]
    ys = np.arange(len(labels))[::-1]    # first label on top

    ax.barh(ys, values, height=0.55, color=colors, edgecolor="white",
            linewidth=0.6, zorder=4)
    for y, v in zip(ys, values):
        ax.text(v + 0.006, y, f"{v:.3f}", ha="left", va="center",
                fontsize=7.0, color=COLORS["ink"], zorder=5)

    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xticks(np.arange(0.6, 1.001, 0.1))
    ax.set_xlabel("worst-case success over noise axis", fontsize=8.0)
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
        1, 2, figsize=(in_from_mm(width_mm), in_from_mm(58.0)),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.52},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.90)
    draw_axis_panel(ax_a)
    draw_worst_panel(ax_b)
    panel_label(ax_a, "(a)", x=-0.16, y=1.12)
    panel_label(ax_b, "(b)", x=-0.42, y=1.12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.9 noise-axis mechanism figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_fql_noise_axis.pdf")
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
    print(f"Saved §5.9 noise-axis figure to {args.output}")


if __name__ == "__main__":
    main()
