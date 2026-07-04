"""
paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_qdrift.py

Figure (SS5.7.3, mechanism one): removing the critic-side support penalty
(beta_2 = 0) drifts the bootstrap-target Q in the SAME (less conservative)
direction on BOTH offline datasets, even though the two baselines sit on
OPPOSITE sides of zero. Grouped bars (beta_2 = 2.0 winner vs beta_2 = 0
ablation) per dataset, with an upward drift arrow between the bar tops
annotated by the absolute shift. The dataset-invariance of the drift
direction is the evidence the figure carries; success-rate deltas stay in
the table/text.

RESULTS figure (SS5.7): bar height IS the mean target-Q number.

Data (REAL; hand-filled from docs/rebrac_experiment_report.md, do not edit
from memory). mean_target_q = bootstrap target read before the critic
penalty, averaged over the late training window:
  crosscomp-1000: beta2=2.0 -> -8.25 (SS7.14.3, Stage C finalist)
                  beta2=0   -> -0.13 (SS7.14.3, 5 seeds)      drift +8.12 (+98%)
  worldcomp-1000: beta2=2.0 -> +15.22 (SS7.13.3, Phase 1, 5 seeds)
                  beta2=0   -> +22.30 (SS7.13.3, 2-seed probe) drift +7.08 (+46%)
The worldcomp beta2=0 cell is a 2-seed probe; the caveat lives in the caption
and SS5.7.m, not in the figure.

Design rules (shared _ch5_style): English-only minimal in-figure text, width
<= 138 mm, vector PDF, muted palette. ReBRAC-Q mainline hue #2F5A6E for the
winner bars; the ablated configuration is the same hue as an open bar.

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_qdrift.py

Output:
    paper/thesis_ch5/figures/fig_ch5_rebrac_qdrift.{pdf,png}
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
)

DATASETS = ("crosscomp-1000", "worldcomp-1000")
Q_WINNER = (-8.25, 15.22)    # beta2 = 2.0 (dual constraint, winner)
Q_ABLATED = (-0.13, 22.30)   # beta2 = 0   (critic-side penalty off)
DRIFT_LABELS = ("+8.1", "+7.1")

C_MAIN = COLORS["mainline_edge"]   # #2F5A6E winner bars
C_INK = COLORS["ink"]
C_MUTED = COLORS["muted"]

BAR_W = 0.30
GAP = 0.36  # centre-to-centre distance between the two bars of a group


def draw(ax: plt.Axes) -> None:
    xs = np.arange(len(DATASETS))
    x_win = xs - GAP / 2.0
    x_abl = xs + GAP / 2.0

    ax.axhline(0.0, color=C_MUTED, lw=0.75, zorder=2)

    ax.bar(x_win, Q_WINNER, width=BAR_W, facecolor=C_MAIN, edgecolor=C_MAIN,
           linewidth=0.9, zorder=3, label=r"$\beta_2 = 2.0$ (dual constraint)")
    ax.bar(x_abl, Q_ABLATED, width=BAR_W, facecolor="white", edgecolor=C_MAIN,
           linewidth=1.1, zorder=3, label=r"$\beta_2 = 0$ (critic side off)")

    # Upward drift arrow between the bar tops; both datasets drift in the
    # same (less conservative) direction despite opposite baseline signs.
    for xw, xa, qw, qa, lab in zip(x_win, x_abl, Q_WINNER, Q_ABLATED,
                                   DRIFT_LABELS):
        xm = (xw + xa) / 2.0
        ax.annotate(
            "", xy=(xm, qa), xytext=(xm, qw),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=9.0, lw=1.0,
                            color=C_INK, shrinkA=0.0, shrinkB=0.0),
            zorder=5,
        )
        ax.text(xm + 0.055, (qw + qa) / 2.0, lab, ha="left", va="center",
                fontsize=7.5, color=C_INK, zorder=5)

    # Direction-of-drift cue on the right edge (semantics in the caption).
    ax.annotate(
        "", xy=(1.62, 20.0), xytext=(1.62, 12.0),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=8.0, lw=0.8,
                        color=C_MUTED, shrinkA=0.0, shrinkB=0.0),
    )
    ax.text(1.575, 16.0, "less\nconservative", ha="right", va="center",
            fontsize=6.8, color=C_MUTED)


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-0.55, 1.7)
    ax.set_ylim(-13.0, 27.0)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(np.arange(-10.0, 26.0, 5.0))
    ax.set_ylabel("mean bootstrap-target Q", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)


def add_legend(ax: plt.Axes) -> None:
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5,
                    handlelength=1.4, labelspacing=0.32, borderaxespad=0.4)
    leg._legend_box.align = "left"


def build_figure(width_mm: float = 90.0) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(62.0)))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.115, top=0.97)
    style_axes(ax)
    draw(ax)
    add_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 SS5.7 beta2-ablation target-Q drift figure.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_rebrac_qdrift.pdf")
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
    print(f"Saved SS5.7 target-Q drift figure to {args.output}")


if __name__ == "__main__":
    main()
