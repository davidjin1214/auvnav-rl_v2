"""
paper/thesis_ch5/figures/scripts/fig_ch5_online_priv_critic.py

Figure (§5.5, Fig B): the reverse-axis control. Injecting a privileged
hull-integral flow estimate into the VALUE network (critic) does not close the
critical-regime gap -- success stays near the floor and, decisively, its peak is
capped across seeds, well below the vanilla peak. This excludes the
"value-estimation is the bottleneck" hypothesis.

Only the robust signal is plotted (success-rate trajectories + the capped peak).
The behaviour-style metrics reported elsewhere (safety / progress / return) do
NOT replicate across the two seeds -- their pooled direction even reverses -- so
they are deliberately omitted rather than shown as a single-seed artefact.

RESULTS figure (§5.5): line height IS the success number.

Data (REAL eval_log.csv, cross-checked; do not edit from memory):
  vanilla s0_k4 + asym s0_k4, seeds {0, 42} (arrival_v2_experiment_report §7.7).
  asym peak is locked at 0.267 on BOTH seeds; vanilla peak in [0.37, 0.53].
  ceiling 0.90 shown for context (the gap neither arm closes).

Design rules (shared _ch5_style + _ch5_data): English-only minimal text,
orthogonal axes/grid, width <= 138 mm, vector PDF, muted palette. Vanilla keeps
the chapter's deployable-s0 ink hue; the privileged-critic arm uses the warm
(online reverse-arm) hue, matching its role across §5.5 figures.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_online_priv_critic.py
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
    MAX_WIDTH_MM,
    apply_style,
    in_from_mm,
)

UPPER_BOUND = 0.900
STEP_SCALE = 1.0e6
SMOOTH_W = 3
ASYM_PEAK = 0.267  # locked across seeds (information ceiling, not noise)

ARMS = {
    "vanilla": {"seeds": [0, 42], "color": COLORS["ink"],
                "label": "standard SAC", "algo": "sac_vanilla"},
    "asym": {"seeds": [0, 42], "color": COLORS["online_edge"],
             "label": "+ privileged critic", "algo": "sac_asym"},
}


def _smooth(y: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    if w <= 1:
        return y
    out = np.empty_like(y)
    half = w // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def draw_arms(ax: plt.Axes) -> None:
    for spec in ARMS.values():
        d = seed_aggregate(crit_cell(spec["algo"], 4, spec["seeds"]))
        x = d["steps"] / STEP_SCALE
        m, s = _smooth(d["mean"]), _smooth(d["std"])
        c = spec["color"]
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.11, lw=0, zorder=2)
        ax.plot(x, m, color=c, lw=1.6, zorder=5, solid_capstyle="round",
                label=spec["label"])


def draw_refs(ax: plt.Axes) -> None:
    ax.axhline(UPPER_BOUND, color=COLORS["muted"], lw=0.9,
               linestyle=(0, (5, 3)), zorder=3)
    ax.text(0.018, UPPER_BOUND + 0.018, "s1 reference = ceiling  0.90",
            ha="left", va="bottom", fontsize=7.0, color=COLORS["muted"], zorder=6)
    # capped privileged-critic peak: a hairline marker that the warm arm
    # never clears, on either seed.
    ax.axhline(ASYM_PEAK, color=COLORS["online_edge"], lw=0.7,
               linestyle=(0, (1, 1.6)), zorder=3, alpha=0.9)
    ax.text(0.985, ASYM_PEAK + 0.018, "privileged-critic peak (capped)  0.27",
            ha="right", va="bottom", fontsize=7.0,
            color=COLORS["online_edge"], zorder=6)


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
    leg = ax.legend(loc="center right", frameon=False, fontsize=7.5,
                    handlelength=1.6, labelspacing=0.4, borderaxespad=0.8)
    leg._legend_box.align = "left"


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(74.0)))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.135, top=0.965)
    style_axes(ax)
    draw_refs(ax)
    draw_arms(ax)
    add_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 §5.5 privileged-critic reverse-axis figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_online_priv_critic.pdf")
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
    print(f"Saved privileged-critic figure to {args.output}")


if __name__ == "__main__":
    main()
