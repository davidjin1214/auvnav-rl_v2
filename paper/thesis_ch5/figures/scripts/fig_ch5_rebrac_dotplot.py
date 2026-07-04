"""
paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_dotplot.py

Figure (SS5.7.2, strongest finding): per-seed test success on worldcomp-1000
across three protocols -- TD3+BC privileged-critic / ReBRAC-Q deployable /
ReBRAC-Q privileged-critic -- with thin connectors joining the same seed, the
collector's own success rate as a dashed reference, and a short mean tick per
column. Two things must land visually:
  - deployable-only ReBRAC-Q sits at the same level as privileged-critic
    TD3+BC (statistical parity carried by the text, not the figure);
  - the privileged critic no longer lifts the ReBRAC-Q mean; its residual
    value is rescuing ONE hard seed (lowest deployable point rises), while
    the other seeds stay put or drift slightly down.

RESULTS figure (SS5.7): point height IS the success number.

Data (REAL; hand-filled from ground-truth docs, do not edit from memory).
5 seeds x test=100 episodes/seed, terminal evaluation, worldcomp-1000
(history-efficiency reward, subcritical, s0 + k=4):
  TD3+BC privileged (alpha=0.1)   : 0.970 / 0.980 / 0.910 / 0.990 / 0.760
      (docs/rebrac_statistical_test_followup.md SS1; aggregate 0.922+-0.086)
  ReBRAC-Q deployable             : 0.990 / 0.930 / 0.780 / 0.980 / 0.960
      (docs/rebrac_experiment_report.md SS7.10.4; aggregate 0.928+-0.077)
  ReBRAC-Q privileged             : 0.960 / 0.920 / 0.900 / 0.930 / 0.960
      (docs/rebrac_experiment_report.md SS7.12.4; aggregate 0.934+-0.026)
  collector (reference)           : 0.990 (SS7.10.5)
Seed order in the tuples follows the fixed seed set of the formal protocol;
the hard seed (third entry) is anonymised in the figure ("hard seed" label,
no seed number) per the chapter-wide convention.

Design rules (shared _ch5_style): English-only minimal in-figure text, width
<= 138 mm, vector PDF, muted palette. The ReBRAC-Q mainline claims the
reserved mainline_edge hue (#2F5A6E); the TD3+BC baseline keeps the plain
offline-branch edge (#5E8597) as in the SS5.6 figure. Protocol is encoded by
fill: deployable = solid marker (the protagonist interface), privileged =
open marker (training-time reference).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_rebrac_dotplot.py

Output:
    paper/thesis_ch5/figures/fig_ch5_rebrac_dotplot.{pdf,png}
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

# Column order: the central parity comparison (TD3+BC privileged vs ReBRAC-Q
# deployable) sits adjacent; the ReBRAC-Q privileged column then shows the
# rescue of the hard seed.
PROTOCOLS = (
    "TD3+BC\n(privileged critic)",
    "ReBRAC-Q\n(deployable)",
    "ReBRAC-Q\n(privileged critic)",
)

# Per-seed test success, one tuple entry per seed of the fixed 5-seed set
# (same order in all three tuples so connectors join the same seed).
TD3BC_PRIV = (0.970, 0.980, 0.910, 0.990, 0.760)
REBRAC_DEP = (0.990, 0.930, 0.780, 0.980, 0.960)
REBRAC_PRIV = (0.960, 0.920, 0.900, 0.930, 0.960)
HARD_SEED_IDX = 2          # anonymised in-figure ("hard seed"), named only in SS5.7.m
COLLECTOR = 0.990          # collector's own trajectory success (reference level)

C_TD3BC = COLORS["offline_edge"]     # #5E8597 baseline (as in SS5.6 figure)
C_REBRAC = COLORS["mainline_edge"]   # #2F5A6E mainline (reserved for SS5.7)
C_CONNECT = COLORS["ghost"]          # light grey seed connectors
C_HARD = COLORS["muted"]             # darker connector for the hard seed
C_REF = COLORS["muted"]              # collector reference line

# Fixed small per-seed x offsets: keep connectors parallel and stop identical
# values (e.g. two 0.960 in the privileged column) from overlapping exactly.
SEED_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def draw(ax: plt.Axes) -> None:
    data = np.array([TD3BC_PRIV, REBRAC_DEP, REBRAC_PRIV])  # (3 protocols, 5 seeds)
    n_prot, n_seed = data.shape
    xs = np.arange(n_prot)

    # Collector reference first (sits behind everything).
    ax.axhline(COLLECTOR, color=C_REF, lw=0.9, linestyle=(0, (5, 3)), zorder=1)
    ax.text(n_prot - 0.52, COLLECTOR + 0.006, "collector (reference)",
            ha="right", va="bottom", fontsize=7.0, color=C_REF)

    # Same-seed connectors (hard seed darker so the rescue is traceable).
    for s in range(n_seed):
        xoff = xs + SEED_OFFSETS[s]
        if s == HARD_SEED_IDX:
            ax.plot(xoff, data[:, s], color=C_HARD, lw=1.1, zorder=3)
        else:
            ax.plot(xoff, data[:, s], color=C_CONNECT, lw=0.7, zorder=2)

    # Per-protocol mean tick (short horizontal bar in the column colour).
    means = data.mean(axis=1)
    tick_colors = (C_TD3BC, C_REBRAC, C_REBRAC)
    for x, m, c in zip(xs, means, tick_colors):
        ax.plot([x - 0.17, x + 0.17], [m, m], color=c, lw=1.8,
                solid_capstyle="butt", zorder=4)

    # Per-seed dots: colour = method, fill = protocol (solid deployable /
    # open privileged).
    styles = (
        dict(color=C_TD3BC, face="white"),      # TD3+BC privileged (open)
        dict(color=C_REBRAC, face=C_REBRAC),    # ReBRAC-Q deployable (solid)
        dict(color=C_REBRAC, face="white"),     # ReBRAC-Q privileged (open)
    )
    for p, st in enumerate(styles):
        xoff = xs[p] + np.asarray(SEED_OFFSETS)
        ax.scatter(xoff, data[p], s=26, facecolor=st["face"],
                   edgecolor=st["color"], linewidth=1.1, zorder=6)

    # Anonymised hard-seed annotation at its lowest (deployable) point.
    hx = xs[1] + SEED_OFFSETS[HARD_SEED_IDX]
    hy = data[1, HARD_SEED_IDX]
    ax.annotate("hard seed", xy=(hx, hy), xytext=(hx - 0.33, hy - 0.045),
                fontsize=7.0, color=C_HARD, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HARD,
                                shrinkA=1.0, shrinkB=2.5))


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-0.5, len(PROTOCOLS) - 0.5)
    ax.set_ylim(0.70, 1.015)
    ax.set_xticks(range(len(PROTOCOLS)))
    ax.set_xticklabels(PROTOCOLS)
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    ax.set_ylabel("test success rate", fontsize=8.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.5,
            color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.6, width=0.75)


def build_figure(width_mm: float = 96.0) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(in_from_mm(width_mm), in_from_mm(70.0)))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.97)
    style_axes(ax)
    draw(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 SS5.7 worldcomp three-protocol per-seed dotplot.")
    default_out = (Path(__file__).resolve().parents[1]
                   / "fig_ch5_rebrac_dotplot.pdf")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=96.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SS5.7 per-seed dotplot to {args.output}")


if __name__ == "__main__":
    main()
