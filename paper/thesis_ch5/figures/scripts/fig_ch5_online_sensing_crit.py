"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_sensing_crit.py

Figure (§5.5, Fig 5.5): the sensing gap in the critical regime. With only the
deployable single point s0 the task nearly fails; adding one forward probe (s1)
restores success, and the information-richest s2 sits between the two while
training less stably. This is the gap the rest of the section sets out to
attribute -- to actor-side temporal access rather than to the sensing hardware.

A bar figure rather than learning curves: s2 in the critical regime has no
trajectory logs (only end-of-training success), so the three configurations are
only comparable at their endpoints here.

RESULTS figure (§5.5): bar height IS the success number.

Data: PRELIMINARY -- the critical 3-seed sensing screen finished on the cluster
and the per-run logs are not yet retrieved locally; the values are entered from
the reported per-seed success rates and MUST be re-verified once the logs are
back (caption carries the PRELIMINARY mark).
  s0 = [0.10, 0.35, 0.17], s1 = [0.90, 0.87, 0.94], s2 = [0.80, 0.71, 0.92].

Design rules (shared _ch5_style): s0/s1/s2 hues (SENSING) match §5.3 and the
other §5.5 figures; English-only minimal text, width <= 138 mm, vector PDF.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_sensing_crit.py
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

CONFIGS = ("s0", "s1", "s2")
LABELS = {"s0": "s0\n(deployable)", "s1": "s1\n(reference)",
          "s2": "s2\n(reference)"}

# PRELIMINARY critical 3-seed sensing screen (re-verify once cloud logs return).
CRIT_SEEDS = {
    "s0": [0.10, 0.35, 0.17],
    "s1": [0.90, 0.87, 0.94],
    "s2": [0.80, 0.71, 0.92],
}
BAR_W = 0.6


def _mean_std(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, float)
    return float(a.mean()), float(a.std(ddof=1))


def draw(ax: plt.Axes) -> None:
    xs = np.arange(len(CONFIGS))
    means = [_mean_std(CRIT_SEEDS[c])[0] for c in CONFIGS]
    stds = [_mean_std(CRIT_SEEDS[c])[1] for c in CONFIGS]
    ax.bar(xs, means, width=BAR_W, color=[SENSING[c] for c in CONFIGS],
           edgecolor=COLORS["ink"], linewidth=0.6, zorder=3)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=COLORS["ink"],
                elinewidth=0.8, capsize=2.6, capthick=0.8, zorder=5)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.025, f"{m:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=COLORS["muted"], zorder=6)
    # gap annotation between s0 and s1.
    ax.annotate("", xy=(0.0, means[1]), xytext=(0.0, means[0]),
                arrowprops=dict(arrowstyle="<->", color=COLORS["muted"],
                                lw=0.8), zorder=4)
    ax.text(0.08, (means[0] + means[1]) / 2.0,
            f"gap  {means[1] - means[0]:.2f}", ha="left", va="center",
            fontsize=7.0, color=COLORS["muted"], zorder=6)


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_ylabel("success rate", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)


def build_figure(width_mm: float = 90.0) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(66.0)))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.135, top=0.965)
    style_axes(ax)
    draw(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 critical sensing-gap figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_sensing_crit.pdf")
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
    print(f"Saved critical sensing-gap figure to {args.output}")


if __name__ == "__main__":
    main()
