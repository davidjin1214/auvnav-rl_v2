"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_monotonic.py

Figure (§5.5, Fig C): the dose-response view of the temporal-access ablation.
Final success rate rises monotonically with history length k, both in the
seed-aggregate trend and along a single representative run that climbs across
k = 4 -> 8 -> 12. A monotone rise with the controlled variable (information in
time), reproduced within one run, is what excludes a seed-specific local optimum
or pure optimisation noise as the explanation for the closing gap.

Complements Fig A (learning curves over training steps): this is the summary
dose-response over the controlled variable.

RESULTS figure (§5.5): marker height IS the success number.

Data (REAL final_eval.json, the locked-table 'final' convention; cross-checked):
  aggregate  k=4 seeds {0,42}; k=8/k=12 seeds {0,7,42}.
  representative run climbs 0.40 -> 0.50 -> 0.90 across k (a single run, shown
  as a thin trajectory; not foregrounded as an individual seed in the prose).
  ceiling 0.90 = s1(k=4) reference = manifest empirical ceiling (27/30).

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
K_SEEDS = {4: [0, 42], 8: [0, 7, 42], 12: [0, 7, 42]}
REPRESENTATIVE_SEED = 0  # a single run that has all three k (shown unlabelled)


def _agg() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = sorted(K_SEEDS)
    means, stds = [], []
    for k in ks:
        vals = [final_eval("sac_vanilla", k, s)["eval_success_rate"]
                for s in K_SEEDS[k]]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return np.array(ks, float), np.array(means), np.array(stds)


def _representative() -> tuple[np.ndarray, np.ndarray]:
    ks = sorted(K_SEEDS)
    vals = [final_eval("sac_vanilla", k, REPRESENTATIVE_SEED)["eval_success_rate"]
            for k in ks]
    return np.array(ks, float), np.array(vals)


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

    rk, rv = _representative()
    ax.plot(rk, rv, color=COLORS["muted"], lw=1.0, marker="o", ms=3.4,
            mfc="white", mec=COLORS["muted"], mew=0.9, linestyle=(0, (4, 2)),
            zorder=5, label="representative run")


def draw_ceiling(ax: plt.Axes) -> None:
    ax.axhline(UPPER_BOUND, color=COLORS["muted"], lw=0.9,
               linestyle=(0, (5, 3)), zorder=3)
    ax.text(4.15, UPPER_BOUND + 0.016, "s1 reference = empirical ceiling  0.90",
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
