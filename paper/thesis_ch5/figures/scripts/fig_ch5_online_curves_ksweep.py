"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_curves_ksweep.py

Figure (§5.5, Fig A): critical-regime learning curves for the deployable
single-point sensor s0 as the actor's temporal window grows (history length
k = 4 -> 8 -> 12). This is the convergence / sample-efficiency evidence the
section previously lacked: success rate vs training steps, seed mean +/- band.

Story (read off the curves):
  - k=4 sits near the floor for the whole budget;
  - k=8 climbs but plateaus below the ceiling on average (seed-noisy);
  - k=12 converges to the empirical ceiling (s1(k=4) == 27/30 = 0.90, dashed)
    and gets there EARLIEST (peak markers) -> temporal access is not just
    sufficient but sample-efficient.

RESULTS figure (§5.5): line height IS the success number.

Data (REAL eval_log.csv, cross-checked; do not edit from memory):
  k=8/k=12  arrival_v2_experiment_report.md §7.8/§7.9, seeds {0,7,42} (n=3).
  k=4       local prototype seeds {0,42} (n=2). The locked bottleneck table's
            k=4=0.21 comes from a NEW cloud 3-seed sensor screen not yet
            retrieved; this curve uses the available local n=2 and is labelled
            as such (re-verify / lift to n=3 once the cloud logs are back).
  ceiling   27/30 = 0.900 (§7.9.7 manifest universal floor) == s1(k=4) ref.

Design rules (shared _ch5_style + _ch5_data): English-only minimal text,
orthogonal axes/grid, width <= 138 mm, vector PDF, muted palette; the s0
history ramp (HISTORY) lightens->darkens with k, the cool s1/ceiling line is a
dashed reference.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_curves_ksweep.py

Output:
    paper/thesis_ch5/figures/fig_ch5_online_curves_ksweep.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_data import crit_cell, seed_aggregate  # noqa: E402
from _ch5_style import (  # noqa: E402
    COLORS,
    HISTORY,
    MAX_WIDTH_MM,
    apply_style,
    in_from_mm,
)

UPPER_BOUND = 0.900  # s1(k=4) reference == empirical ceiling 27/30
STEP_SCALE = 1.0e6   # x axis in 10^6 training steps
SMOOTH_W = 3         # light rolling-mean window on the periodic-eval curve

# (k, seeds available locally). k=8/k=12 = n=3; k=4 = n=2 (cloud n=3 pending).
K_SEEDS = {4: [0, 42], 8: [0, 7, 42], 12: [0, 7, 42]}


def _smooth(y: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    """Centred rolling mean (edge-shrinking window); mild eval-noise denoise."""
    if w <= 1:
        return y
    out = np.empty_like(y)
    half = w // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _load() -> dict[int, dict]:
    return {k: seed_aggregate(crit_cell("sac_vanilla", k, s))
            for k, s in K_SEEDS.items()}


def draw_curves(ax: plt.Axes, data: dict[int, dict]) -> None:
    for k in (4, 8, 12):
        d = data[k]
        x = d["steps"] / STEP_SCALE
        m, s = _smooth(d["mean"]), _smooth(d["std"])
        c = HISTORY[k]
        lw = 1.7 if k == 12 else 1.4
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.11, lw=0, zorder=2)
        ax.plot(x, m, color=c, lw=lw, zorder=5, solid_capstyle="round",
                label=f"k = {k}")
        # sample-efficiency marker: peak of the (smoothed) seed-mean curve.
        pk = int(np.argmax(m))
        ax.scatter([x[pk]], [m[pk]], s=16, color=c, edgecolor="white",
                   linewidths=0.6, zorder=7)


def draw_ceiling(ax: plt.Axes) -> None:
    # Neutral grey reference line so it reads as an annotation, not a fourth
    # data curve, and never collides in hue with the k = 4 steel-blue curve.
    ax.axhline(UPPER_BOUND, color=COLORS["muted"], lw=0.9,
               linestyle=(0, (5, 3)), zorder=3)
    ax.text(0.018, UPPER_BOUND + 0.018,
            "s1 reference = empirical ceiling  0.90",
            ha="left", va="bottom", fontsize=7.0, color=COLORS["muted"], zorder=7)


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(np.arange(0.0, 1.01, 0.25))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel(r"training steps  [$\times 10^{6}$]", fontsize=8.5)
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
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7.5,
                    handlelength=1.6, labelspacing=0.32, borderaxespad=0.4,
                    title="history length")
    leg.get_title().set_fontsize(7.5)
    leg._legend_box.align = "left"


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(74.0)))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.135, top=0.965)
    style_axes(ax)
    draw_ceiling(ax)
    draw_curves(ax, _load())
    add_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 critical k-sweep learning curves.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_curves_ksweep.pdf")
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
    print(f"Saved k-sweep learning-curve figure to {args.output}")


if __name__ == "__main__":
    main()
