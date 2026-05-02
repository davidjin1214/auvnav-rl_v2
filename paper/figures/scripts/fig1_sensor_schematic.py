"""
paper/figures/scripts/fig1_sensor_schematic.py

Figure 1: Sensor schematic for §3 Problem Setup.

Layout: 2 rows × 1 column (vertical stack).

Top panel (a): D2Q9 TRT-LBM Kármán wake field 的一帧涡量场背景，
               叠加 REMUS-100 body schematic + heading 箭头，
               cross-stream task：goal 在 AUV 的 +y（横向）方向。
Bottom panel (b): 体坐标系下 sensor 抽样几何 —
                  s0 = 单点 DVL 水追踪（hull origin） vs
                  hull-integral 5-point sampling
                  (ξ ∈ {-0.4, -0.2, 0, +0.2, +0.4} × L body axis)。

Source data:
    wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy
    (T=1200, Nx=320, Ny=100, C=3 channels: u_mps, v_mps, omega_1ps)

ROI extent (from meta json):
    x ∈ [120, 312] m, y ∈ [60, 120] m, dx = 0.6 m
    U_ref = 1.0 m/s, D_ref = 12.0 m, Re = 150, Ti = 5%

REMUS-100 hull length L ≈ 1.6 m (per CLAUDE.md vehicle config).
Cross-stream task：free-stream U_∞ 沿 +x，goal 在 AUV 的 +y（垂直来流 ±20° 内）方向。

Usage:
    python paper/figures/scripts/fig1_sensor_schematic.py

Output:
    paper/figures/output/fig1_sensor_schematic.{pdf, png}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
WAKE_NPY = REPO_ROOT / "wake_data" / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
WAKE_META = REPO_ROOT / "wake_data" / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi_meta.json"
OUTPUT_DIR = REPO_ROOT / "paper" / "figures" / "output"
OUTPUT_PDF = OUTPUT_DIR / "fig1_sensor_schematic.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig1_sensor_schematic.png"


# ---------------------------------------------------------------------------
# Plotting params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotParams:
    frame_idx: int = 600
    auv_pos_world_m: tuple[float, float] = (210.0, 78.0)
    # heading: cross-stream → 朝 +y 偏 (≤20° from +y axis)；74° from +x ≡ 16° from +y。
    auv_heading_deg: float = 74.0
    goal_pos_world_m: tuple[float, float] = (220.0, 113.0)
    panel_a_xlim: tuple[float, float] = (170.0, 285.0)
    panel_a_ylim: tuple[float, float] = (62.0, 118.0)
    sample_xi: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4)


PARAMS = PlotParams()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_wake_frame(npy_path: Path, meta_path: Path, frame_idx: int):
    """Load a single (Nx, Ny, C) frame via memory-mapped read.

    Returns
    -------
    omega : np.ndarray
        Vorticity field [Nx, Ny] in 1/s.
    extent : tuple[float, float, float, float]
        (x_min, x_max, y_min, y_max) in metres for imshow extent.
    meta : dict
        Wake metadata.
    """
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    arr = np.load(npy_path, mmap_mode="r")
    frame = np.asarray(arr[frame_idx]).astype(np.float32)
    omega = frame[..., 2]
    extent = (
        float(meta["roi_x0_phys_m"]),
        float(meta["roi_x1_phys_m"]),
        float(meta["roi_y0_phys_m"]),
        float(meta["roi_y1_phys_m"]),
    )
    return omega, extent, meta


def draw_auv_body(ax, center_xy: tuple[float, float], length: float, diameter: float,
                   heading_deg: float, edge_color: str, face_color: str,
                   line_width: float = 1.5):
    cx, cy = center_xy
    body = Ellipse(
        xy=(cx, cy),
        width=length,
        height=diameter,
        angle=heading_deg,
        edgecolor=edge_color,
        facecolor=face_color,
        linewidth=line_width,
        zorder=5,
    )
    ax.add_patch(body)


def draw_heading_arrow(ax, center_xy: tuple[float, float], heading_deg: float,
                        length: float, color: str = "#cc3333", zorder: int = 6):
    cx, cy = center_xy
    rad = np.deg2rad(heading_deg)
    dx = length * np.cos(rad)
    dy = length * np.sin(rad)
    ax.annotate(
        "",
        xy=(cx + dx, cy + dy),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=14),
        zorder=zorder,
    )


def panel_a(ax, params: PlotParams) -> None:
    """Panel (a): wake field + AUV + cross-stream goal."""
    omega, extent, _meta = load_wake_frame(WAKE_NPY, WAKE_META, params.frame_idx)
    omega_T = omega.T
    vmax = float(np.percentile(np.abs(omega_T), 98))
    im = ax.imshow(
        omega_T,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        zorder=1,
    )

    cx, cy = params.auv_pos_world_m
    gx, gy = params.goal_pos_world_m

    # Cross-stream goal direction guide (dashed line AUV → goal)
    ax.annotate(
        "",
        xy=(gx, gy),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-", color="#1a5e1a", lw=1.0,
                         linestyle=(0, (4, 3))),
        zorder=4,
    )

    # AUV body (visually inflated for legibility — body-frame schematic in panel b is to scale)
    draw_auv_body(
        ax,
        center_xy=(cx, cy),
        length=14.0,
        diameter=4.5,
        heading_deg=params.auv_heading_deg,
        edge_color="black",
        face_color="#ffd966",
    )
    draw_heading_arrow(
        ax,
        center_xy=(cx, cy),
        heading_deg=params.auv_heading_deg,
        length=10.0,
    )

    # Goal marker
    ax.scatter([gx], [gy], marker="*", s=380, c="#2ca02c",
               edgecolor="black", linewidth=0.8, zorder=6)
    ax.annotate(
        "Goal",
        xy=(gx, gy),
        xytext=(gx + 4.0, gy - 1.0),
        fontsize=10,
        fontweight="bold",
        color="#1a5e1a",
        zorder=6,
    )

    # AUV label
    ax.annotate(
        "REMUS-100",
        xy=(cx, cy),
        xytext=(cx - 32.0, cy - 2.0),
        fontsize=9,
        ha="left",
        fontweight="bold",
        zorder=7,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    )

    # Cross-stream task annotation
    ax.text(
        gx + 2.0, (cy + gy) / 2.0,
        "cross-stream\n(goal $\\approx +y$)",
        fontsize=8.5,
        color="#1a5e1a",
        style="italic",
        zorder=7,
    )

    # Free-stream arrow at top-left
    x0_a, _x1_a = params.panel_a_xlim
    _y0_a, y1_a = params.panel_a_ylim
    ax.annotate(
        "$U_\\infty = 1.0$ m/s",
        xy=(x0_a + 5.0, y1_a - 4.5),
        fontsize=9.5,
        ha="left",
        color="#222222",
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(x0_a + 32.0, y1_a - 8.0),
        xytext=(x0_a + 5.0, y1_a - 8.0),
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.5, mutation_scale=13),
    )

    ax.set_xlim(*params.panel_a_xlim)
    ax.set_ylim(*params.panel_a_ylim)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("(a) Kármán wake field + REMUS-100 cross-stream task",
                  fontsize=10.5, loc="left", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015, aspect=20)
    cbar.set_label("vorticity $\\omega_z$ [1/s]", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)


def panel_b(ax, params: PlotParams) -> None:
    """Panel (b): sensor sampling geometry — s0 vs hull-integral 5-point."""
    L = 1.0
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-0.95, 0.65)
    # NOTE: panel (b) 是示意图，不强制 aspect=equal —— body 椭圆会被纵向略拉伸，
    # 但能换来文字 / 抽样点的纵向呼吸空间，不挤。

    # AUV body in body-frame
    body = Ellipse(
        xy=(0.0, 0.0),
        width=L,
        height=L * 0.12,
        angle=0.0,
        edgecolor="black",
        facecolor="#ffd966",
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(body)

    # Body-axis arrow
    ax.annotate("", xy=(L * 0.72, 0.0), xytext=(L * 0.45, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#cc3333", lw=2.0,
                                 mutation_scale=14), zorder=4)
    ax.text(L * 0.58, 0.06, "$\\hat{\\mathbf{e}}_{\\mathrm{body}}$",
            fontsize=11, color="#cc3333", zorder=5)

    # s0: single-point DVL probe ABOVE the body
    s0_y = 0.28
    ax.scatter([0.0], [s0_y], marker="o", s=160, c="#1f77b4",
                edgecolor="black", linewidth=1.0, zorder=6)
    ax.annotate("", xy=(0.0, 0.04), xytext=(0.0, s0_y - 0.03),
                arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=1.2,
                                 linestyle="--"), zorder=4)
    ax.text(0.07, s0_y, "$\\mathbf{s}_0$: single-point DVL water-track",
            fontsize=10, color="#1f77b4", zorder=7, fontweight="bold",
            va="center")
    ax.text(0.07, s0_y - 0.10,
            "(actor sees this in BOTH protocols)",
            fontsize=9, color="#1f77b4", zorder=7, va="center", style="italic")

    # Hull-integral 5 sample points BELOW the body
    samples_y = -0.35
    for xi in params.sample_xi:
        x_pt = xi * L
        ax.scatter([x_pt], [samples_y], marker="D", s=95,
                   c="#d62728", edgecolor="black", linewidth=0.8, zorder=6)
        ax.annotate("", xy=(x_pt, -0.04), xytext=(x_pt, samples_y + 0.03),
                    arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.0,
                                     linestyle=":"), zorder=4)

    # Numeric ξ labels just below diamonds
    labels_y = samples_y - 0.10
    for xi in params.sample_xi:
        ax.text(xi * L, labels_y, f"${xi:+.1f}$",
                fontsize=8.5, ha="center", color="#7a1a1a", zorder=7)

    # Single-line ξ_i legend just under labels
    legend_xi_y = labels_y - 0.10
    ax.text(0.0, legend_xi_y,
            "$\\xi_i$ in units of body length $L$",
            fontsize=8.5, ha="center", color="#7a1a1a", style="italic",
            zorder=7)

    # Bracket spanning the 5 points
    bracket_y = legend_xi_y - 0.13
    ax.annotate("", xy=(-0.45, bracket_y), xytext=(0.45, bracket_y),
                arrowprops=dict(arrowstyle="<->", color="#7a1a1a", lw=1.4),
                zorder=5)

    # Two-line caption below the bracket
    ax.text(0.0, bracket_y - 0.10,
            "5-pt hull-integral $\\to\\ \\mathbf{o}^{\\mathrm{priv}} = [u_{\\mathrm{eq}},\\ v_{\\mathrm{eq}}]$",
            fontsize=10, ha="center", va="top", color="#7a1a1a",
            fontweight="bold", zorder=7)
    ax.text(0.0, bracket_y - 0.20,
            "(critic sees this ONLY in privileged-critic protocol)",
            fontsize=9, ha="center", va="top", color="#7a1a1a",
            style="italic", zorder=7)

    # Body-frame caption (top-right corner, off body)
    ax.text(1.05, 0.58, "Body-frame (not to scale)",
            fontsize=8, color="#444444", style="italic", ha="right")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(b) Deployable vs. privileged sensor sampling",
                  fontsize=10.5, loc="left", fontweight="bold")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    panel_a(axes[0], PARAMS)
    panel_b(axes[1], PARAMS)

    fig.tight_layout(pad=1.4, h_pad=2.4)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
