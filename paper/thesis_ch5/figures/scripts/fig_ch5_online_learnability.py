"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_learnability.py

Figure (§5.5, Fig 5.4): sensing-configuration learnability across regimes.

Two panels share the success-rate axis but NOT the protocol — they are placed
side by side precisely so the reader does not mistake them for a controlled A/B:
    (a) Subcritical (Re=150, U_inf=1.0 m/s), history-efficiency reward, 3 seeds:
        s0 / s1 / s2 are ALL learnable; the information-richest s2 is neither the
        best nor the most stable (its +/-std bar dwarfs s1's) -> more channels do
        not buy more stable training.
    (b) Critical (Re=250, U_inf=1.5 m/s), arrival reward, single seed:
        the deployable single-point s0 collapses (~0.10) while the spatial
        reference s1 holds (0.90); s2 was not measured in this regime (gap shown
        explicitly, not silently omitted).

This figure is a RESULTS figure (§5.5): bar heights ARE the success numbers
(spec §5.5 red line, the inverse of the setup/method figures). Each panel's
protocol (reward, seed count) is annotated so the cross-regime read stays
qualitative (all-learnable subcritically -> deployable s0 collapses critically).

Data (cross-checked, do not edit from memory):
  subcritical  online_rl_line_summary.md §1.1 (A0, efficiency_v2, n=3, +/-std)
  critical     arrival_v2_experiment_report.md §7.6 (single seed, no error bar)

Design rules (shared _ch5_style):
  - English-only, minimal in-figure text; semantics live in caption + body.
  - Orthogonal bars/axes/grid, width <= 138 mm, vector PDF, muted palette.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_learnability.py

Output:
    paper/thesis_ch5/figures/fig_ch5_online_learnability.{pdf,png}
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
LABELS = {"s0": "s0", "s1": "s1", "s2": "s2"}

# --- success rate (the plotted result number) -----------------------------
# Subcritical: A0 sensor screen, efficiency_v2, k=4, 3 seeds (mean, std).
# online_rl_line_summary.md §1.1.
SUBCRIT = {
    "s0": (0.789, 0.211),
    "s1": (0.967, 0.027),
    "s2": (0.856, 0.204),
}
# Critical: arrival_v2 sensor envelope, k=4, 3 seeds (per-seed success rates).
# PRELIMINARY — cloud run just finished, result files not yet retrieved locally;
# values entered by hand and MUST be re-verified against the retrieved logs.
# NOTE: these supersede the single-seed s0=0.10 / s1=0.90 / s2(not-run) figures
# in online.tex §5.5.2 / Table tab:ch5_online_bottleneck / §5.5.m — reconcile the
# prose, the bottleneck table baseline, and the impl note once the logs are back.
CRIT_SEEDS = {
    "s0": [0.10, 0.35, 0.17],
    "s1": [0.90, 0.87, 0.94],
    "s2": [0.80, 0.71, 0.92],
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    # sample std (ddof=1), the chapter's +/-std convention over repeated seeds.
    return float(arr.mean()), float(arr.std(ddof=1))


SUBCRIT_STATS = {c: SUBCRIT[c] for c in CONFIGS}
CRIT_STATS = {c: _mean_std(CRIT_SEEDS[c]) for c in CONFIGS}

BAR_W = 0.62


def _bar_colors() -> list[str]:
    return [SENSING[c] for c in CONFIGS]


def _style_panel(ax: plt.Axes, *, ylabel: bool) -> None:
    ax.set_xlim(-0.65, len(CONFIGS) - 0.35)
    ax.set_ylim(0.0, 1.16)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)
    if ylabel:
        ax.set_ylabel("success rate", fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)


def _value_label(ax: plt.Axes, x: float, top: float, text: str) -> None:
    ax.text(x, top + 0.022, text, ha="center", va="bottom",
            fontsize=7.0, color=COLORS["muted"], zorder=6)


def panel_bars(ax: plt.Axes, stats: dict[str, tuple[float, float]], *,
               tag: str, name: str, ylabel: bool) -> None:
    xs = np.arange(len(CONFIGS))
    means = [stats[c][0] for c in CONFIGS]
    stds = [stats[c][1] for c in CONFIGS]
    ax.bar(xs, means, width=BAR_W, color=_bar_colors(),
           edgecolor=COLORS["ink"], linewidth=0.6, zorder=3)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=COLORS["ink"],
                elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=5)
    for x, m, s in zip(xs, means, stds):
        _value_label(ax, x, m + s, f"{m:.2f}")
    _style_panel(ax, ylabel=ylabel)
    ax.text(0.0, 1.05, f"{tag}  {name}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.2, fontweight="bold",
            color=COLORS["ink"])


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    width_in = in_from_mm(width_mm)
    height_in = in_from_mm(58.0)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width_in, height_in),
                                     sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.135, top=0.86,
                        wspace=0.07)
    panel_bars(ax_a, SUBCRIT_STATS, tag="(a)", name="subcritical", ylabel=True)
    panel_bars(ax_b, CRIT_STATS, tag="(b)", name="critical", ylabel=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 sensing-learnability figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_learnability.pdf")
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
    print(f"Saved sensing-learnability figure to {args.output}")


if __name__ == "__main__":
    main()
