"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_curves_subcrit.py

Figure (§5.5, Fig D): subcritical-regime learning curves for the three sensing
configurations. All three are learnable, but the information-richest s2 trains
LESS stably than s1 -- its seed band stays wide for the whole budget -- so more
observation channels do not buy more stable training. This is the training-
dynamics counterpart to the end-of-training bars in Fig 5.4(a).

RESULTS figure (§5.5): line height IS the success number.

Data (REAL eval_log.csv, A0 sensor screen, efficiency_v2; cross-checked):
  s0 / s1 / s2 at k=4, seeds {46, 47, 50}; 60 evaluation points over 600k steps.
  online_rl_line_summary.md §1.1.

Design rules (shared _ch5_style + _ch5_data): the s0/s1/s2 hues (SENSING) match
§5.3 probe-geometry and §5.5 Fig 5.4 so the reader tracks the configurations by
colour throughout the chapter.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_curves_subcrit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_data import a0_cell, seed_aggregate  # noqa: E402
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    SENSING,
    apply_style,
    in_from_mm,
)

STEP_SCALE = 1.0e5  # subcritical budget is 600k -> x in 10^5 steps (0..6)
SMOOTH_W = 3
SEEDS = [46, 47, 50]
LABELS = {"s0": "s0  (deployable)", "s1": "s1  (reference)",
          "s2": "s2  (reference)"}


def _smooth(y: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    if w <= 1:
        return y
    out = np.empty_like(y)
    half = w // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def draw(ax: plt.Axes) -> None:
    # s0 (protagonist) = heaviest solid ink; s1 solid and s2 dashed share the cool
    # SENSING ramp, so a redundant line style keeps the two reference curves
    # distinguishable where their close hues and seed bands overlap.
    styles = {"s0": ("-", 1.7), "s1": ("-", 1.4), "s2": ((0, (5, 2)), 1.4)}
    for cfg in ("s0", "s1", "s2"):
        d = seed_aggregate(a0_cell("efficiency_v2", cfg, SEEDS))
        x = d["steps"] / STEP_SCALE
        m, s = _smooth(d["mean"]), _smooth(d["std"])
        c = SENSING[cfg]
        ls, lw = styles[cfg]
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.10, lw=0, zorder=2)
        ax.plot(x, m, color=c, lw=lw, linestyle=ls, zorder=5,
                solid_capstyle="round", label=LABELS[cfg])


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(np.arange(0.0, 6.01, 1.0))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel(r"training steps  [$\times 10^{5}$]", fontsize=8.5)
    ax.set_ylabel("success rate", fontsize=8.5)
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
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(70.0)))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.14, top=0.965)
    style_axes(ax)
    draw(ax)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7.5,
                    handlelength=1.6, labelspacing=0.34, borderaxespad=0.6)
    leg._legend_box.align = "left"
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 subcritical learning curves.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_curves_subcrit.pdf")
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
    print(f"Saved subcritical learning-curve figure to {args.output}")


if __name__ == "__main__":
    main()
