"""
paper/thesis_ch5/figures/scripts/fig_ch5_method_framework.py

Figure (§5.4): Unified method framework — the chapter's single methodology
visual anchor. NO result numbers (spec §0.4 red line).

Panel (a) Method derivation:
    shared off-policy Actor-Critic backbone -> stochastic branch (SAC, online)
    | deterministic (TD3) backbone -> three behaviour-constrained offline
    methods (TD3+BC, ReBRAC-Q [mainline], FQL); all four collapse onto the
    SAME deployable single-point policy interface.

Panel (b) Train / deploy information asymmetry:
    the policy always uses the deployable observation; the value network may
    additionally read the privileged flow during training; at deployment the
    value network is discarded and the privileged flow is absent, so the
    deployed sensing is identical regardless of the protocol.

Design rules honoured (per user review of the first draft):
  - Orthogonal wiring only: every connector is a pure vertical/horizontal
    segment routed through a horizontal "bus"; no diagonal arrows.
  - In-figure text is English words/abbreviations, NOT math symbols
    (policy/value/observation, not pi_theta/Q_phi/o); kept minimal.
  - Same-row boxes share size and baseline for a grid-like look.
  - Loss-equation numbers are NOT printed in-figure (they auto-number in
    LaTeX); the caption cross-references them. LayerNorm (shared by ReBRAC-Q
    and FQL) is left to the caption, not tagged on a single box.

Usage (run under the mytorch1 conda env; pure matplotlib, no auv_nav needed):
    python paper/thesis_ch5/figures/scripts/fig_ch5_method_framework.py

Output:
    paper/thesis_ch5/figures/fig_ch5_method_framework.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    apply_style,
    arrow,
    box_text,
    clean_axes,
    in_from_mm,
    panel_label,
    rounded_box,
)


# ---------------------------------------------------------------------------
# Panel (a): method derivation — orthogonal "bus" wiring
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes) -> None:
    clean_axes(ax)
    line = COLORS["line"]
    ghost = COLORS["ghost"]

    # --- boxes: rows share size + baseline for a grid-like layout --------
    bb = dict(cx=0.500, cy=0.900, w=0.700, h=0.120)
    rounded_box(ax, **bb, fill=COLORS["backbone_fill"],
                edge=COLORS["backbone_edge"], lw=1.1)
    box_text(ax, bb["cx"], bb["cy"], title="Off-policy Actor–Critic",
             sub=r"shared policy + value networks $\cdot$ clipped double-Q")

    sac = dict(cx=0.150, cy=0.620, w=0.225, h=0.130)
    rounded_box(ax, **sac, fill=COLORS["online_fill"],
                edge=COLORS["online_edge"], lw=1.0)
    box_text(ax, sac["cx"], sac["cy"], title="SAC",
             sub=r"max-entropy $\cdot$ online")

    det = dict(cx=0.660, cy=0.620, w=0.480, h=0.130)
    rounded_box(ax, **det, fill=COLORS["offline_fill"],
                edge=COLORS["offline_edge"], lw=1.0)
    box_text(ax, det["cx"], det["cy"], title="Deterministic (TD3) backbone",
             sub=r"target smoothing $\cdot$ delayed update", title_size=8.2)

    methods = {
        "td3bc": dict(cx=0.400, cy=0.340, w=0.205, h=0.150),
        "rebrac": dict(cx=0.650, cy=0.340, w=0.205, h=0.150),
        "fql": dict(cx=0.880, cy=0.340, w=0.205, h=0.150),
    }
    rounded_box(ax, **methods["td3bc"], fill=COLORS["offline_fill"],
                edge=COLORS["offline_edge"], lw=1.0)
    box_text(ax, methods["td3bc"]["cx"], methods["td3bc"]["cy"],
             title="TD3+BC", sub="policy-side BC")
    rounded_box(ax, **methods["rebrac"], fill=COLORS["mainline_fill"],
                edge=COLORS["mainline_edge"], lw=1.7)
    box_text(ax, methods["rebrac"]["cx"], methods["rebrac"]["cy"],
             title="ReBRAC-Q", sub="two-sided BC",
             title_color=COLORS["mainline_edge"])
    rounded_box(ax, **methods["fql"], fill=COLORS["offline_fill"],
                edge=COLORS["offline_edge"], lw=1.0)
    box_text(ax, methods["fql"]["cx"], methods["fql"]["cy"],
             title="FQL", sub="flow-matching ref.")

    dep = dict(cx=0.500, cy=0.075, w=0.800, h=0.090)
    rounded_box(ax, **dep, fill=COLORS["deploy_fill"],
                edge=COLORS["deploy_edge"], lw=1.1)
    box_text(ax, dep["cx"], dep["cy"],
             title="Deployable single-point policy interface",
             title_size=8.2, sub=None)

    # --- orthogonal wiring: vertical stubs + horizontal buses ------------
    # backbone -> distribution bus -> SAC / deterministic backbone
    arrow(ax, (0.500, 0.840), (0.500, 0.745), style="-", color=line)
    arrow(ax, (0.150, 0.745), (0.660, 0.745), style="-", color=line)
    arrow(ax, (0.150, 0.745), (0.150, 0.685), color=line)
    arrow(ax, (0.660, 0.745), (0.660, 0.685), color=line)
    ax.text(0.118, 0.716, "stochastic", ha="right", va="center",
            fontsize=6.8, color=COLORS["online_edge"])
    ax.text(0.692, 0.716, "deterministic", ha="left", va="center",
            fontsize=6.8, color=COLORS["offline_edge"])

    # deterministic backbone -> distribution bus -> three offline methods
    arrow(ax, (0.660, 0.555), (0.660, 0.450), style="-", color=line)
    arrow(ax, (0.400, 0.450), (0.880, 0.450), style="-", color=line)
    for m in methods.values():
        arrow(ax, (m["cx"], 0.450), (m["cx"], 0.415), color=line)

    # SAC + three methods -> convergence bus -> deployment interface
    arrow(ax, (0.150, 0.555), (0.150, 0.180), style="-", color=ghost)
    for m in methods.values():
        arrow(ax, (m["cx"], 0.265), (m["cx"], 0.180), style="-", color=ghost)
    arrow(ax, (0.150, 0.180), (0.880, 0.180), style="-", color=ghost)
    arrow(ax, (0.515, 0.180), (0.515, 0.120), color=ghost)

    panel_label(ax, "(a)", x=-0.01, y=1.02)


# ---------------------------------------------------------------------------
# Panel (b): train / deploy information asymmetry
# ---------------------------------------------------------------------------

def _group_frame(ax: plt.Axes, x0: float, x1: float, label: str) -> None:
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(
        FancyBboxPatch(
            (x0, 0.085), x1 - x0, 0.715,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor="none", edgecolor=COLORS["grid"],
            linewidth=1.0, linestyle=(0, (4, 2)), zorder=0.5,
        )
    )
    ax.text((x0 + x1) / 2.0, 0.90, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=COLORS["ink"])


def _net_box(ax, cx, cy, title, sub, *, fill, edge, lw=1.0,
             title_color=None, sub_color=None, ghost=False) -> None:
    w, h = 0.37, 0.21
    if ghost:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2.0, cy - h / 2.0), w, h,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=fill, edgecolor=edge, linewidth=lw,
                linestyle=(0, (3, 2)), zorder=2,
            )
        )
    else:
        rounded_box(ax, cx, cy, w, h, fill=fill, edge=edge, lw=lw)
    # Manual title/sub placement with generous vertical separation; the short
    # panel (b) axes makes box_text's default offsets too tight.
    title_color = title_color or COLORS["ink"]
    sub_color = sub_color or COLORS["muted"]
    ax.text(cx, cy + 0.040, title, ha="center", va="center", fontsize=8.0,
            fontweight="bold", color=title_color, zorder=4)
    ax.text(cx, cy - 0.045, sub, ha="center", va="center", fontsize=6.5,
            color=sub_color, zorder=4)


def draw_panel_b(ax: plt.Axes) -> None:
    clean_axes(ax)

    _group_frame(ax, 0.045, 0.455, "Training")
    _group_frame(ax, 0.545, 0.955, "Deployment")

    # Training column
    _net_box(ax, 0.25, 0.60, "Policy network", "uses observation",
             fill=COLORS["online_fill"], edge=COLORS["online_edge"])
    _net_box(ax, 0.25, 0.26, "Value network", "observation + privileged flow",
             fill=COLORS["mainline_fill"], edge=COLORS["mainline_edge"], lw=1.3,
             sub_color=COLORS["mainline_edge"])

    # Deployment column
    _net_box(ax, 0.75, 0.60, "Policy network", "single-point observation",
             fill=COLORS["online_fill"], edge=COLORS["online_edge"])
    _net_box(ax, 0.75, 0.26, "Value network", "discarded",
             fill="white", edge=COLORS["ghost"], ghost=True,
             title_color=COLORS["ghost"], sub_color=COLORS["ghost"])

    # connectors train -> deploy (already horizontal)
    arrow(ax, (0.44, 0.60), (0.56, 0.60), lw=1.0, color=COLORS["line"])
    ax.text(0.50, 0.655, "kept", ha="center", va="bottom",
            fontsize=6.6, color=COLORS["muted"])
    arrow(ax, (0.44, 0.26), (0.56, 0.26), lw=0.9, color=COLORS["ghost"],
          linestyle=(0, (3, 2)), style="-|>")

    panel_label(ax, "(b)", x=-0.01, y=1.02)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    width_in = in_from_mm(width_mm)
    height_in = in_from_mm(133.0)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(width_in, height_in),
        gridspec_kw=dict(height_ratios=[2.4, 1.18]),
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.015, top=0.965, hspace=0.18)
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 5 method-framework figure.")
    default_out = Path(__file__).resolve().parents[1] / "fig_ch5_method_framework.pdf"
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=MAX_WIDTH_MM)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved method-framework figure to {args.output}")


if __name__ == "__main__":
    main()
