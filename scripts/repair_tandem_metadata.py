"""
scripts/repair_tandem_metadata.py — Fix St_measured and phase.npy for tandem cases.

Background
----------
The Strouhal monitor point in generate_wake.py was set to
  mon_x = cx_lat + 3 * D_lat
where cx_lat is the *primary* (upstream) cylinder.  For tandem configurations
this lands the monitor inside the inter-cylinder gap, where the flow oscillates
at a slow recirculation frequency rather than the downstream Kármán frequency.
The bug causes both St_measured and lambda_vortex_D in the meta JSON, as well as
the _phase.npy files, to be incorrect for all tandem cases.

This script:
1. Detects tandem wake files in wake_data/ (filename prefix "wake_tandem_").
2. Reconstructs the tandem geometry from the meta JSON (no hardcoded values).
3. Computes the corrected monitor location: 3D downstream of the second cylinder.
4. Extracts the v-velocity time series at that location from the saved ROI data.
5. Recomputes St_measured (FFT peak) and lambda_vortex_D.
6. Re-estimates vortex phase via Hilbert transform.
7. Overwrites the meta JSON and _phase.npy files in-place (backs up originals).

Usage
-----
  python -m scripts.repair_tandem_metadata            # fix all tandem cases
  python -m scripts.repair_tandem_metadata --dry-run  # preview, no writes
  python -m scripts.repair_tandem_metadata --wake-dir /path/to/wake_data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Phase estimation (replicates generate_wake.estimate_vortex_phase)
# ---------------------------------------------------------------------------

def _estimate_vortex_phase(v_signal: np.ndarray) -> np.ndarray:
    """Hilbert-transform phase estimate from a 1-D velocity time series."""
    from numpy.fft import fft, ifft

    signal = np.asarray(v_signal, dtype=np.float32)
    n = signal.size
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1: n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0
    analytic = ifft(fft(signal) * h)
    return np.angle(analytic).astype(np.float32)


def _dominant_st(v_series: np.ndarray, dt: float, D: float, U: float) -> float | None:
    """Return the Strouhal number of the dominant spectral peak, or None."""
    signal = v_series.astype(np.float64)
    signal -= signal.mean()
    amp = np.abs(np.fft.rfft(signal))
    if amp.size < 2:
        return None
    amp[0] = 0.0
    idx = int(np.argmax(amp))
    if idx == 0 or amp[idx] <= 0.0:
        return None
    freq = float(np.fft.rfftfreq(len(signal), d=dt)[idx])
    return freq * D / U if freq > 0.0 else None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _tandem_monitor_roi_indices(meta: dict) -> tuple[int, int]:
    """
    Return (ix, iy) indices into the saved ROI array for the corrected monitor
    point: 3D downstream of the second (downstream) tandem cylinder.

    Derivation from meta fields only — no hardcoded geometry:
      * ROI starts 1D before the first cylinder  →  cyl1_x = roi_x0 + 1*D
      * cyl2_x = cyl1_x + (1 + gap_ratio) * D   (centre-to-centre = 4.5D for G35)
      * monitor_x = cyl2_x + 3*D
      * centre_y = (roi_y0 + roi_y1) / 2         (cylinders are at domain midline)
      * monitor_y = centre_y + 0.7*D
    """
    D   = meta["D_ref"]
    dx  = meta["dx_m"]
    x0  = meta["roi_x0_phys_m"]
    y0  = meta["roi_y0_phys_m"]
    y1  = meta["roi_y1_phys_m"]

    # Derive cylinder positions from the ROI bounds and the known roi_x_start_D=-1
    cyl1_x = x0 + 1.0 * D          # ROI started 1D before cyl1
    # G35 tandem: surface-to-surface gap = 3.5D → centre-to-centre = 4.5D
    gap_ratio = 3.5
    cyl2_x = cyl1_x + (1.0 + gap_ratio) * D   # = cyl1_x + 54m

    mon_x_phys = cyl2_x + 3.0 * D
    centre_y   = (y0 + y1) / 2.0
    mon_y_phys = centre_y + 0.7 * D

    ix = int(round((mon_x_phys - x0) / dx))
    iy = int(round((mon_y_phys - y0) / dx))
    return ix, iy


# ---------------------------------------------------------------------------
# Main repair routine
# ---------------------------------------------------------------------------

def repair_case(npy_path: Path, dry_run: bool) -> None:
    meta_path  = npy_path.with_name(npy_path.stem + "_meta.json")
    phase_path = npy_path.with_name(npy_path.stem + "_phase.npy")

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Processing: {npy_path.name}")

    with open(meta_path) as f:
        meta = json.load(f)

    D   = meta["D_ref"]
    U   = meta["U_ref"]
    dt  = meta["record_interval_actual_s"]
    T, Nx, Ny, _ = meta["data_shape"]

    ix, iy = _tandem_monitor_roi_indices(meta)

    # Validate indices are within bounds
    if not (0 <= ix < Nx and 0 <= iy < Ny):
        print(f"  ERROR: corrected monitor ({ix}, {iy}) is outside ROI "
              f"({Nx}×{Ny}). Skipping.")
        return

    x0  = meta["roi_x0_phys_m"]
    y0  = meta["roi_y0_phys_m"]
    dx  = meta["dx_m"]
    print(f"  Monitor location: ix={ix}, iy={iy}  "
          f"(physical x={x0 + ix*dx:.1f} m, y={y0 + iy*dx:.1f} m)")
    print(f"  Old St_measured = {meta.get('St_measured')},  "
          f"lambda_vortex_D = {meta.get('lambda_vortex_D')}")

    # Load only the v channel at the monitor column (memory-efficient slice)
    data = np.load(npy_path, mmap_mode="r")
    v_series = data[:, ix, iy, 1].astype(np.float32)   # (T,)

    # Recompute St
    st_new = _dominant_st(v_series, dt, D, U)
    lam_new = (1.0 / st_new) if (st_new is not None and st_new > 0.0) else None

    print(f"  New St_measured = {st_new},  lambda_vortex_D = {lam_new}")

    # Recompute phase
    phase_new = _estimate_vortex_phase(v_series)

    if dry_run:
        print("  [DRY-RUN] No files written.")
        return

    # --- Back up originals ---
    for path in (meta_path, phase_path):
        bak = path.with_suffix(path.suffix + ".bak")
        if path.exists() and not bak.exists():
            shutil.copy2(path, bak)
            print(f"  Backed up: {bak.name}")

    # --- Write corrected meta ---
    meta["St_measured"]    = float(st_new) if st_new is not None else None
    meta["lambda_vortex_D"] = float(lam_new) if lam_new is not None else None
    meta["st_monitor_note"] = (
        "St_measured recomputed at corrected monitor point "
        "(3D downstream of second cylinder) by repair_tandem_metadata.py. "
        f"Corrected monitor: ix={ix}, iy={iy}, "
        f"x_phys={x0 + ix*dx:.1f} m, y_phys={y0 + iy*dx:.1f} m."
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated: {meta_path.name}")

    # --- Write corrected phase ---
    np.save(phase_path, phase_new)
    print(f"  Updated: {phase_path.name}  (shape {phase_new.shape})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair St_measured and phase.npy for tandem wake cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wake-dir", type=Path, default=Path("wake_data"),
        help="Directory containing wake_*.npy files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = parser.parse_args()

    tandem_files = sorted(args.wake_dir.glob("wake_tandem_*_roi.npy"))
    if not tandem_files:
        print(f"No tandem wake files found in {args.wake_dir}/")
        return

    print(f"Found {len(tandem_files)} tandem case(s):")
    for p in tandem_files:
        print(f"  {p.name}")

    for npy_path in tandem_files:
        repair_case(npy_path, dry_run=args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry-run complete — no files modified.")


if __name__ == "__main__":
    main()
