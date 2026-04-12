"""
scripts/visualize_flow_field.py — Standalone wake field visualization.

Produces three publication-quality static figures directly from wake_data/*.npy
files, with no dependency on the RL environment or PyTorch.

Figures
-------
  fig_p1_snapshot_comparison.{pdf,png}
      3-row × 2-col vorticity snapshot grid for all 6 configurations.
      Rows: single cylinder / tandem (G/D=3.5) / side-by-side (G/D=3.5).
      Columns: Re=150 U=1.0 m/s  |  Re=250 U=1.5 m/s.
      Row heights are proportional to physical ROI height; each column
      shares a common colour scale.

  fig_p2_flow_statistics.{pdf,png}
      Time-averaged streamwise velocity (ū), time-averaged lateral velocity
      (v̄), and streamwise velocity fluctuation (u′_rms) for the hardest
      single-cylinder case (Re=250, U=1.5 m/s).  ū uses a two-slope colour
      norm centred at U_∞ to highlight the velocity deficit and acceleration
      regions.

  fig_p3_phase_evolution.{pdf,png}
      Six equally-spaced vorticity snapshots spanning one Kármán shedding
      period for the single-cylinder Re=250 case, showing the downstream
      advection of vortex cores.

TODO
----
  P4: Phase-averaged vorticity — use wake_data/*_phase.npy to compute
      ensemble-averaged fields at N_phase phase bins; reveal coherent
      structure without random fluctuation.

  P5: Probe layout overlay — superimpose s0 / s1 / s2 probe positions on
      a representative vorticity snapshot to visualise the AUV's local
      sensing footprint relative to the vortex street.

Usage
-----
  python -m scripts.visualize_flow_field                    # all figures, PDF
  python -m scripts.visualize_flow_field --format png       # PNG output
  python -m scripts.visualize_flow_field --figure p1        # single figure
  python -m scripts.visualize_flow_field --outdir figs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAKE_DIR = Path("wake_data")
D_REF    = 12.0   # cylinder diameter [m], identical for all cases

# File stem lookup: (config_key, condition) → stem (without directory / .npy)
STEMS: dict[tuple[str, str], str] = {
    ("single", "low"):  "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi",
    ("single", "high"): "wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi",
    ("tandem", "low"):  "wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi",
    ("tandem", "high"): "wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi",
    ("sbs",    "low"):  "wake_sbs_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi",
    ("sbs",    "high"): "wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi",
}

# Row order and display labels for P1
CONFIGS = [
    ("single", "Single Cylinder"),
    ("tandem", "Tandem  G/D = 3.5"),
    ("sbs",    "Side-by-Side  G/D = 3.5"),
]

SNAPSHOT_FRAME = 400   # representative frame index used for all P1 snapshots


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class WakeCase(NamedTuple):
    data:     np.ndarray   # memory-mapped (T, Nx, Ny, 3) float16
    meta:     dict
    x_phys:  np.ndarray   # (Nx,) physical x-coords [m]
    y_phys:  np.ndarray   # (Ny,) physical y-coords [m]
    dx:       float        # grid spacing [m]


def load_case(stem: str, wake_dir: Path = WAKE_DIR) -> WakeCase:
    npy  = wake_dir / f"{stem}.npy"
    jmeta = wake_dir / f"{stem}_meta.json"
    if not npy.exists():
        raise FileNotFoundError(npy)
    with open(jmeta) as f:
        meta = json.load(f)
    data = np.load(npy, mmap_mode="r")
    dx   = meta["dx_m"]
    _, Nx, Ny, _ = data.shape
    x0 = meta["roi_x0_phys_m"]
    y0 = meta["roi_y0_phys_m"]
    x_phys = x0 + np.arange(Nx) * dx
    y_phys = y0 + np.arange(Ny) * dx
    return WakeCase(data=data, meta=meta, x_phys=x_phys, y_phys=y_phys, dx=dx)


def omega_frame(case: WakeCase, frame: int) -> np.ndarray:
    """Return vorticity field (Ny, Nx) float32 at *frame*."""
    return case.data[frame, :, :, 2].T.astype(np.float32)


def imshow_extent(case: WakeCase) -> list[float]:
    """Half-cell-padded extent [x0, x1, y0, y1] for imshow origin='lower'."""
    h = case.dx / 2
    return [
        case.x_phys[0]  - h, case.x_phys[-1]  + h,
        case.y_phys[0]  - h, case.y_phys[-1]  + h,
    ]


def _flow_arrow(ax: plt.Axes, extent: list[float]) -> None:
    """Draw a small '→ flow' annotation in the upper-left corner of *ax*."""
    x0, x1, y0, y1 = extent
    ax.annotate(
        "flow",
        xy     =(x0 + (x1 - x0) * 0.13, y0 + (y1 - y0) * 0.87),
        xytext =(x0 + (x1 - x0) * 0.03, y0 + (y1 - y0) * 0.87),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.4),
        fontsize=6.5, color="k", va="center",
    )


# ---------------------------------------------------------------------------
# P1 — Six-case vorticity snapshot comparison
# ---------------------------------------------------------------------------

def plot_p1(outdir: Path, fmt: str, wake_dir: Path) -> None:
    """3-row × 2-col vorticity snapshot grid for all six configurations."""
    print("P1  loading wake cases …")

    cases: dict[tuple[str, str], WakeCase] = {}
    for cfg, _ in CONFIGS:
        for cond in ("low", "high"):
            try:
                cases[(cfg, cond)] = load_case(STEMS[(cfg, cond)], wake_dir)
                print(f"  ✓ {cfg:6s} {cond}")
            except FileNotFoundError as e:
                print(f"  ✗ {e}")

    # ── colour scale: one vmax per column (same U / Re) ─────────────────────
    def col_vmax(cond: str) -> float:
        vals = [
            float(np.percentile(np.abs(omega_frame(cases[(cfg, cond)], SNAPSHOT_FRAME)), 99))
            for cfg, _ in CONFIGS
            if (cfg, cond) in cases
        ]
        return max(vals) if vals else 0.3

    vmaxes = [col_vmax("low"), col_vmax("high")]

    # ── row heights proportional to physical ROI height ──────────────────────
    roi_heights = []
    for cfg, _ in CONFIGS:
        key = (cfg, "low") if (cfg, "low") in cases else (cfg, "high")
        if key in cases:
            c = cases[key]
            roi_heights.append(c.y_phys[-1] - c.y_phys[0])
        else:
            roi_heights.append(60.0)   # fallback

    min_h = min(roi_heights)
    height_ratios = [h / min_h for h in roi_heights]

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 5 + sum(height_ratios) * 0.9),
        gridspec_kw=dict(
            height_ratios=height_ratios,
            hspace=0.38, wspace=0.06,
            left=0.07, right=0.88, top=0.91, bottom=0.06,
        ),
    )

    col_titles = [
        "Re = 150,   U = 1.0 m/s",
        "Re = 250,   U = 1.5 m/s",
    ]

    # keep one image handle per column for the shared colorbar
    col_im = [None, None]

    for row, (cfg, cfg_label) in enumerate(CONFIGS):
        for col, (cond, vmax) in enumerate(zip(["low", "high"], vmaxes)):
            ax  = axes[row, col]
            key = (cfg, cond)

            if key not in cases:
                ax.text(0.5, 0.5, "data not found",
                        ha="center", va="center", transform=ax.transAxes, color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            c      = cases[key]
            omega  = omega_frame(c, SNAPSHOT_FRAME)
            extent = imshow_extent(c)

            im = ax.imshow(
                omega, origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                aspect="auto", interpolation="bilinear",
            )
            if col_im[col] is None:
                col_im[col] = im

            _flow_arrow(ax, extent)

            # Strouhal annotation
            St = c.meta.get("St_measured", 0.0)
            # For tandem Re=250, note that T_shed > recording window
            St_note = ""
            T_shed_s = 1.0 / (St * c.meta["U_ref"] / D_REF) if St > 0 else 0
            if T_shed_s > 360:
                St_note = " *"
            ax.text(
                0.98, 0.96, f"St = {St:.3f}{St_note}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=7, color="k",
                bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.5),
            )

            # Row label on left column only
            if col == 0:
                ax.set_ylabel(cfg_label, fontsize=9, labelpad=5)

            # Column title on top row only
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold", pad=5)

            # x-label on bottom row only
            if row == 2:
                ax.set_xlabel("x [m]", fontsize=7)

            ax.tick_params(labelsize=6.5)

    # ── per-column colourbars ─────────────────────────────────────────────────
    for col, im in enumerate(col_im):
        if im is None:
            continue
        cb = fig.colorbar(im, ax=axes[:, col], location="right",
                          fraction=0.035, pad=0.02, shrink=0.95)
        cb.set_label("ω [s⁻¹]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    t_snap = SNAPSHOT_FRAME * 0.3
    fig.suptitle(
        f"Vorticity Snapshots — All Six Wake Configurations"
        f"   (frame {SNAPSHOT_FRAME},  t = {t_snap:.0f} s)\n"
        "* St too low: shedding period exceeds recording window",
        fontsize=10.5, fontweight="bold",
    )

    out = outdir / f"fig_p1_snapshot_comparison.{fmt}"
    fig.savefig(out, dpi=150 if fmt == "png" else 100, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# P2 — Time-averaged flow statistics
# ---------------------------------------------------------------------------

def plot_p2(outdir: Path, fmt: str, wake_dir: Path) -> None:
    """Time-mean ū, v̄ and u′_rms for single-cylinder Re=250, U=1.5 m/s."""
    stem = STEMS[("single", "high")]
    print(f"P2  loading {stem} …")
    try:
        case = load_case(stem, wake_dir)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    T, Nx, Ny, _ = case.data.shape
    U_ref = case.meta["U_ref"]
    Re    = int(case.meta["Re"])
    extent = imshow_extent(case)

    print(f"  computing mean / std over {T} frames ({Nx}×{Ny} grid) …")
    # Cast to float32 in one go — single cylinder is ~230 MB float16 → ~460 MB float32
    data_f32 = case.data.astype(np.float32)   # (T, Nx, Ny, 3)
    u_mean_raw = data_f32[:, :, :, 0].mean(axis=0)   # (Nx, Ny)
    v_mean_raw = data_f32[:, :, :, 1].mean(axis=0)
    u_rms_raw  = data_f32[:, :, :, 0].std(axis=0)
    del data_f32

    # Transpose for imshow (Ny, Nx)
    u_mean = u_mean_raw.T
    v_mean = v_mean_raw.T
    u_rms  = u_rms_raw.T

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 7.5),
        gridspec_kw=dict(hspace=0.46, left=0.07, right=0.91, top=0.90, bottom=0.07),
    )

    # ── ū: TwoSlopeNorm centred at U_ref ─────────────────────────────────────
    u_lo  = float(u_mean.min())
    u_hi  = max(float(u_mean.max()), U_ref * 1.05)
    norm0 = TwoSlopeNorm(vmin=u_lo, vcenter=U_ref, vmax=u_hi)
    im0   = axes[0].imshow(
        u_mean, origin="lower", extent=extent,
        cmap="RdBu_r", norm=norm0,
        aspect="auto", interpolation="bilinear",
    )
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.01)
    cb0.set_label("ū [m/s]", fontsize=8)
    cb0.ax.tick_params(labelsize=7)
    cb0.ax.axhline(U_ref, color="k", lw=1.2, ls="--")
    cb0.ax.text(
        2.6, U_ref, f"U∞ = {U_ref} m/s",
        va="center", ha="left", fontsize=6, color="k",
        transform=cb0.ax.get_yaxis_transform(),
    )
    axes[0].set_title(
        r"Time-averaged streamwise velocity $\bar{u}$   "
        r"(blue = deficit $<U_\infty$,  red = acceleration $>U_\infty$)",
        fontsize=9.5,
    )

    # ── v̄: symmetric diverging ───────────────────────────────────────────────
    vmax_v = max(float(np.percentile(np.abs(v_mean), 99)), 0.02)
    im1    = axes[1].imshow(
        v_mean, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v,
        aspect="auto", interpolation="bilinear",
    )
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.01)
    cb1.set_label("v̄ [m/s]", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    axes[1].set_title(
        r"Time-averaged lateral velocity $\bar{v}$   "
        r"(symmetry breaking → non-zero mean indicates Coandă bias)",
        fontsize=9.5,
    )

    # ── u′_rms: sequential ───────────────────────────────────────────────────
    vmax_rms = float(np.percentile(u_rms, 99.5))
    im2      = axes[2].imshow(
        u_rms, origin="lower", extent=extent,
        cmap="YlOrRd", vmin=0, vmax=vmax_rms,
        aspect="auto", interpolation="bilinear",
    )
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)
    cb2.set_label("u′_rms [m/s]", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    axes[2].set_title(
        r"Streamwise velocity fluctuation $u'_{\rm rms}$   "
        r"(shear-layer cores = high turbulence intensity)",
        fontsize=9.5,
    )

    for ax in axes:
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=6.5)

    fig.suptitle(
        f"Time-Averaged Flow Statistics — Single Cylinder   "
        f"Re = {Re},  U = {U_ref} m/s   ({T} frames, {T * 0.3:.0f} s)",
        fontsize=11, fontweight="bold",
    )

    out = outdir / f"fig_p2_flow_statistics.{fmt}"
    fig.savefig(out, dpi=150 if fmt == "png" else 100, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# P3 — Single shedding-period phase evolution
# ---------------------------------------------------------------------------

def plot_p3(outdir: Path, fmt: str, wake_dir: Path) -> None:
    """Six vorticity frames spanning one Kármán period (single cyl Re=250)."""
    stem = STEMS[("single", "high")]
    print(f"P3  loading {stem} …")
    try:
        case = load_case(stem, wake_dir)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    meta   = case.meta
    St     = meta["St_measured"]
    U_ref  = meta["U_ref"]
    Re     = int(meta["Re"])
    dt_rec = meta["record_interval_actual_s"]   # seconds per stored frame

    f_shed   = St * U_ref / D_REF              # shedding frequency [Hz]
    T_shed_s = 1.0 / f_shed                    # shedding period [s]
    T_frames = T_shed_s / dt_rec               # period in stored frames

    print(
        f"  St = {St:.4f},  f = {f_shed:.5f} Hz,  "
        f"T = {T_shed_s:.1f} s = {T_frames:.1f} frames"
    )

    N_PHASES    = 6
    START_FRAME = 300    # well into the stationary periodic regime
    T_total     = case.data.shape[0]

    phase_frames = [
        min(int(round(START_FRAME + i * T_frames / N_PHASES)), T_total - 1)
        for i in range(N_PHASES)
    ]
    print(f"  frame indices: {phase_frames}")

    extent = imshow_extent(case)

    # Load only the required frames and find shared vmax
    omega_stack = np.stack([omega_frame(case, f) for f in phase_frames])  # (6, Ny, Nx)
    vmax = float(np.percentile(np.abs(omega_stack), 99.5))

    # ── figure: 2 rows × 3 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(17, 5.8),
        gridspec_kw=dict(
            hspace=0.32, wspace=0.06,
            left=0.05, right=0.92, top=0.88, bottom=0.08,
        ),
    )

    last_im = None
    for i, (ax, f_idx) in enumerate(zip(axes.flat, phase_frames)):
        omega = omega_stack[i]
        im = ax.imshow(
            omega, origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="bilinear",
        )
        last_im = im

        phase_frac = i / N_PHASES
        t_s = f_idx * dt_rec
        ax.set_title(
            f"φ = {phase_frac:.2f} T   (t = {t_s:.0f} s,  frame {f_idx})",
            fontsize=8.5,
        )
        if i % 3 == 0:
            ax.set_ylabel("y [m]", fontsize=7)
        if i >= 3:
            ax.set_xlabel("x [m]", fontsize=7)
        ax.tick_params(labelsize=6)
        _flow_arrow(ax, extent)

    # ── single shared colorbar on the far right ──────────────────────────────
    cbar_ax = fig.add_axes([0.935, 0.10, 0.013, 0.78])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("ω [s⁻¹]", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Kármán Vortex Street — One Shedding Period\n"
        f"Single Cylinder   Re = {Re},  U = {U_ref} m/s,  "
        f"St = {St:.4f},  T = {T_shed_s:.1f} s",
        fontsize=11, fontweight="bold",
    )

    out = outdir / f"fig_p3_phase_evolution.{fmt}"
    fig.savefig(out, dpi=150 if fmt == "png" else 100, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wake field visualization — P1 snapshot / P2 statistics / P3 phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir",   type=Path, default=Path("figs"),
                        help="Output directory.")
    parser.add_argument("--format",   choices=["pdf", "png"], default="pdf",
                        dest="fmt", help="Output file format.")
    parser.add_argument("--wake-dir", type=Path, default=WAKE_DIR,
                        help="Directory containing wake_*.npy files.")
    parser.add_argument("--figure",   choices=["p1", "p2", "p3", "all"],
                        default="all", help="Which figure(s) to generate.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    to_run = ["p1", "p2", "p3"] if args.figure == "all" else [args.figure]

    if "p1" in to_run:
        plot_p1(args.outdir, args.fmt, args.wake_dir)
    if "p2" in to_run:
        plot_p2(args.outdir, args.fmt, args.wake_dir)
    if "p3" in to_run:
        plot_p3(args.outdir, args.fmt, args.wake_dir)

    # TODO P4: Phase-averaged vorticity
    #   - Load *_phase.npy (shape: (T,) float, each entry = phase in [0, 2π))
    #   - Bin frames into N_phase bins; average omega per bin
    #   - Plot N_phase panels analogous to P3, but with ensemble-clean fields
    #
    # TODO P5: Probe layout overlay (s0 / s1 / s2)
    #   - Place a representative AUV at the ROI centre or a mid-episode position
    #   - Draw probe positions in body frame (rotated to heading ψ=0)
    #   - Overlay on a vorticity snapshot to show sensing footprint vs vortex scale


if __name__ == "__main__":
    main()
