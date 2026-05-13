"""Compute speed magnitude statistics for all wake_data .npy files.

Usage:
    python experiments/auvhamnode_spike/_wake_stats.py

Output: tab-separated stats table to stdout (printed with flush=True).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WAKE_FILES = [
    "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    "wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    "wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    "wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    "wake_sbs_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    "wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
]


def main() -> None:
    here = Path(__file__).resolve().parent.parent.parent  # repo root
    wake_dir = here / "wake_data"
    print(f"wake_dir = {wake_dir}", flush=True)
    print(
        f"{'file_short':<48} {'U_ref':>6} {'shape':>22} "
        f"{'speed_med':>10} {'speed_p50':>10} {'speed_p95':>10} "
        f"{'speed_p99':>10} {'speed_max':>10} {'|u|_med':>10} {'|v|_p99':>10}",
        flush=True,
    )
    print("-" * 160, flush=True)
    for fname in WAKE_FILES:
        path = wake_dir / fname
        if not path.exists():
            print(f"{fname:<48} MISSING", flush=True)
            continue
        try:
            data = np.load(path, mmap_mode="r")  # don't load all of it
        except Exception as exc:
            print(f"{fname:<48} ERROR {exc}", flush=True)
            continue
        # Convert to float32 progressively
        u_chan = np.asarray(data[..., 0], dtype=np.float32)
        v_chan = np.asarray(data[..., 1], dtype=np.float32)
        speed = np.sqrt(u_chan**2 + v_chan**2)
        if "U1p00" in fname:
            U_ref = 1.0
        elif "U1p50" in fname:
            U_ref = 1.5
        else:
            U_ref = float("nan")
        short = fname.replace("wake_", "").replace(
            "_dx0p60_Ti5pct_1200f_roi.npy", ""
        )
        print(
            f"{short:<48} {U_ref:>6.2f} {str(data.shape):>22} "
            f"{np.median(speed):>10.4f} {np.percentile(speed, 50):>10.4f} "
            f"{np.percentile(speed, 95):>10.4f} {np.percentile(speed, 99):>10.4f} "
            f"{speed.max():>10.4f} {np.median(np.abs(u_chan)):>10.4f} "
            f"{np.percentile(np.abs(v_chan), 99):>10.4f}",
            flush=True,
        )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
