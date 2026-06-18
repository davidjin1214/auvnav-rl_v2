"""
paper/thesis_ch5/figures/scripts/_ch5_style.py

Shared visual style for all Chapter 5 figures (方案 A: ch5 工程自足).

Design contract (与正文规划一致):
  - Nature-leaning publication style: Arial/Helvetica, 7-8.5 pt, thin lines,
    muted low-saturation palette, vector PDF (pdf.fonttype=42), no chart-junk.
  - English-only in-figure text, kept minimal; semantics live in caption + body.
  - Default max width 138 mm (height free); fits both letterpaper and A4 text blocks.
  - NO result numbers in setup/method figures (spec §0.4 red line).

This module is pure matplotlib/numpy (no auv_nav import) so method-relationship
figures render in any env; data-backed figures (flow field, probe layouts) add
their own auv_nav / npy dependencies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MM_PER_INCH = 25.4
MAX_WIDTH_MM = 138.0

# Muted, low-saturation palette. Hue carries meaning sparingly:
#   neutral slate = shared backbone / deployment;
#   warm  = online (SAC) branch; cool = offline (deterministic) branch.
# The offline mainline (ReBRAC-Q) is emphasised by weight, not a new hue.
COLORS = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "line": "#66788A",
    "grid": "#E5E7EB",
    "backbone_fill": "#E8EDF2",
    "backbone_edge": "#5B6B7B",
    "online_fill": "#FBEFE3",
    "online_edge": "#C68A4E",
    "offline_fill": "#E9F0F3",
    "offline_edge": "#5E8597",
    "mainline_fill": "#DCE6EC",
    "mainline_edge": "#2F5A6E",
    "deploy_fill": "#F4F6F8",
    "deploy_edge": "#66788A",
    "ghost": "#AEB8C2",
}

# Sensing-configuration colours — kept consistent across §5.3 Fig 5.1/5.2
# (probe geometry) and §5.5 Fig 5.4/5.5 (online learnability) so the reader
# tracks s0/s1/s2 by hue throughout the chapter. s0 = deployable single point
# (protagonist, darkest ink); s1/s2 = spatial reference upper bounds (cool ramp).
SENSING = {
    "s0": "#1F2933",  # ink — deployable single-point DVL (protagonist)
    "s1": "#5E8597",  # cool slate — + short-range ADCP (reference)
    "s2": "#8AA4B2",  # lighter cool — + long-range ADCP (reference)
}


def apply_style() -> None:
    """Apply the shared Nature-leaning rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def in_from_mm(mm: float) -> float:
    return mm / MM_PER_INCH


def rounded_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    fill: str,
    edge: str,
    lw: float = 1.0,
    zorder: float = 2.0,
) -> None:
    """Draw a centred rounded rectangle in axes data coordinates."""
    box = FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)


def box_text(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    title: str,
    sub: str | None = None,
    title_size: float = 8.5,
    sub_size: float = 7.0,
    title_color: str | None = None,
    sub_color: str | None = None,
    title_weight: str = "bold",
    zorder: float = 4.0,
) -> None:
    """Centred title (+ optional sub-label below) inside a box."""
    title_color = title_color or COLORS["ink"]
    sub_color = sub_color or COLORS["muted"]
    if sub is None:
        ax.text(
            cx, cy, title, ha="center", va="center",
            fontsize=title_size, fontweight=title_weight,
            color=title_color, zorder=zorder,
        )
    else:
        ax.text(
            cx, cy + 0.018, title, ha="center", va="center",
            fontsize=title_size, fontweight=title_weight,
            color=title_color, zorder=zorder,
        )
        ax.text(
            cx, cy - 0.026, sub, ha="center", va="center",
            fontsize=sub_size, color=sub_color, zorder=zorder,
        )


def arrow(
    ax: plt.Axes,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    *,
    color: str | None = None,
    lw: float = 0.9,
    style: str = "-|>",
    mutation_scale: float = 9.0,
    connectionstyle: str | None = None,
    linestyle: str = "-",
    shrinkA: float = 1.0,
    shrinkB: float = 2.0,
    zorder: float = 1.5,
) -> None:
    """Thin connector arrow in axes data coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            linestyle=linestyle,
            color=color or COLORS["line"],
            connectionstyle=connectionstyle or "arc3,rad=0.0",
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def panel_label(ax: plt.Axes, text: str, x: float = 0.0, y: float = 1.0) -> None:
    """Bold (a)/(b) panel tag in axes-fraction coordinates."""
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.0, fontweight="bold",
        color=COLORS["ink"], clip_on=False,
    )


def clean_axes(ax: plt.Axes, xlim=(0.0, 1.0), ylim=(0.0, 1.0)) -> None:
    """Blank canvas: equal-ish data box, no ticks, no spines."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
