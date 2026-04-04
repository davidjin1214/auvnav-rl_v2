"""
TRT-LBM wake generator v8.0 — cylinder wake dataset for RL training.

Simulates 2-D incompressible flow past a circular cylinder using the
Two-Relaxation-Time Lattice Boltzmann Method (D2Q9 TRT-LBM).

Key design choices:
  - GPU acceleration via CuPy with automatic NumPy/CPU fallback.
  - Zero-allocation streaming: pre-allocated buffers and slice assignments
    avoid transient array creation in the inner loop.
  - Direct D2Q9 momentum accumulation in macroscopic() to skip (9,Nx,Ny)
    broadcast temporaries.
  - Batched turbulent inlet generation to amortise Python/scipy overhead
    and CPU→GPU transfer cost.
  - ROI spatial downsampling: simulate at fine dx, output at coarser dx_out.
    Vorticity is computed at full resolution before downsampling.

Output: float16 arrays of shape (frames, roi_nx, roi_ny, 3) with channels
  [u (m/s), v (m/s), ω (1/s)], plus per-frame vortex phase and a JSON
  metadata file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

try:
    import numexpr as ne

    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False


# ---------------------------------------------------------------------------
# Stability thresholds
# ---------------------------------------------------------------------------
MIN_TAU_S = 0.52
MAX_MA = 0.15

# Number of inlet profiles generated per batch (amortises CPU→GPU transfer).
INLET_BATCH_SIZE = 2000


# ---------------------------------------------------------------------------
# D2Q9 lattice constants  (module-level, always NumPy)
# ---------------------------------------------------------------------------
# Velocity vectors: rest, E, N, W, S, NE, NW, SW, SE
D2Q9_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int32,
)
D2Q9_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)
# Opposite-direction map for half-way bounce-back.
D2Q9_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Broadcast-ready weight array (9, 1, 1).
_W_BC = D2Q9_W[:, np.newaxis, np.newaxis]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PhysicsConfig:
    """Physical and numerical parameters for one simulation case."""

    Re: float
    U_phys: float = 1.0
    D_phys: float = 20.0
    Lx_phys: float = 1000.0
    Ly_phys: float = 400.0
    dx: float = 0.5
    dt: float = 0.025
    cyl_x_phys: float = 200.0
    cyl_y_center: float = 200.0
    cyl_y_jitter: float = 0.0
    turbulence_intensity: float = 0.05
    turbulence_length_scale: int = 20
    T_spinup_phys: float = 4000.0
    T_record_phys: float = 2500.0
    record_interval: float = 0.2
    roi_x_start_D: float = 2.0
    roi_x_end_D: float = 30.0
    roi_y_half_D: float = 3.5
    # Spatial downsample factor applied to ROI output (1 = full resolution).
    roi_downsample: int = 1
    # Extra cylinders beyond the primary: list of (x_phys, y_phys, D_phys) tuples.
    # Default empty list → single-cylinder behaviour unchanged.
    extra_cylinders: list = field(default_factory=list)
    # Override the ROI y-centre (metres). None → use cyl_y_center.
    # Set to domain midpoint for side-by-side configs.
    roi_y_center_override: float | None = None
    # Short prefix tag embedded in output filename, e.g. "tandem_G35_".
    case_tag: str = ""


@dataclass(frozen=True)
class NavigationProfile:
    """Parameter template for a family of navigation-oriented wake cases."""

    name: str
    U_phys_values: tuple[float, ...]
    Re_values: tuple[float, ...]
    D_phys: float
    Lx_phys: float
    Ly_phys: float
    dx: float
    base_dt: float
    cyl_x_phys: float
    cyl_y_center: float
    T_spinup_phys: float
    T_record_phys: float
    record_interval: float
    roi_x_end_D: float
    roi_y_half_D: float
    turbulence_intensity: float
    turbulence_length_scale: int
    roi_downsample: int = 1

    def to_physics_config(self, Re: float, U_phys: float) -> PhysicsConfig:
        """Create a PhysicsConfig for a specific (Re, U_phys) combination."""
        dt = stable_dt_for_case(
            Re=Re, U_phys=U_phys, D_phys=self.D_phys,
            dx=self.dx, base_dt=self.base_dt,
        )
        return PhysicsConfig(
            Re=Re, U_phys=U_phys, D_phys=self.D_phys,
            Lx_phys=self.Lx_phys, Ly_phys=self.Ly_phys,
            dx=self.dx, dt=dt,
            cyl_x_phys=self.cyl_x_phys, cyl_y_center=self.cyl_y_center,
            cyl_y_jitter=0.0,
            turbulence_intensity=self.turbulence_intensity,
            turbulence_length_scale=self.turbulence_length_scale,
            T_spinup_phys=self.T_spinup_phys, T_record_phys=self.T_record_phys,
            record_interval=self.record_interval,
            roi_x_end_D=self.roi_x_end_D, roi_y_half_D=self.roi_y_half_D,
            roi_downsample=self.roi_downsample,
        )


@dataclass(frozen=True)
class LatticeConfig:
    """Derived lattice-Boltzmann parameters computed from PhysicsConfig."""

    Nx: int
    Ny: int
    D_lat: int
    U_lat: float
    nu_lat: float
    tau_s: float
    tau_a: float
    omega_s: float
    omega_a: float
    cx_lat: int
    cy_lat: int
    steps_spinup: int
    steps_record: int
    steps_per_frame: int
    total_frames: int
    actual_record_interval: float
    roi_x0: int
    roi_x1: int
    roi_y0: int
    roi_y1: int
    roi_downsample: int
    output_roi_nx: int
    output_roi_ny: int
    dx_out: float
    cylinders_lat: tuple = ()   # NEW: all cylinders as (cx_lat, cy_lat, r_lat)

    @property
    def roi_nx(self) -> int:
        return self.roi_x1 - self.roi_x0

    @property
    def roi_ny(self) -> int:
        return self.roi_y1 - self.roi_y0

    @classmethod
    def from_physics(cls, p: PhysicsConfig) -> "LatticeConfig":
        if p.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if p.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if p.record_interval <= 0.0:
            raise ValueError("record_interval must be positive.")
        if p.Re <= 0.0:
            raise ValueError("Re must be positive.")
        if p.U_phys <= 0.0:
            raise ValueError("U_phys must be positive.")
        if p.D_phys <= 0.0:
            raise ValueError("D_phys must be positive.")
        if p.turbulence_length_scale < 1:
            raise ValueError("turbulence_length_scale must be >= 1.")

        Nx = int(round(p.Lx_phys / p.dx))
        Ny = int(round(p.Ly_phys / p.dx))
        D_lat = int(round(p.D_phys / p.dx))
        U_lat = p.U_phys * (p.dt / p.dx)
        nu_lat = U_lat * D_lat / p.Re
        tau_s = 3.0 * nu_lat + 0.5
        tau_a = 0.5 + 0.25 / (tau_s - 0.5)
        omega_s = 1.0 / tau_s
        omega_a = 1.0 / tau_a
        cx_lat = int(round(p.cyl_x_phys / p.dx))
        cy_lat = int(round((p.cyl_y_center + p.cyl_y_jitter) / p.dx))
        steps_spinup = max(1, int(round(p.T_spinup_phys / p.dt)))
        steps_record = max(1, int(round(p.T_record_phys / p.dt)))
        steps_per_frame = max(1, int(round(p.record_interval / p.dt)))
        total_frames = (steps_record + steps_per_frame - 1) // steps_per_frame
        actual_record_interval = steps_per_frame * p.dt
        roi_x0 = max(0, cx_lat + int(round(p.roi_x_start_D * D_lat)))
        roi_x1 = min(Nx, cx_lat + int(round(p.roi_x_end_D * D_lat)))
        roi_cy_lat = (
            int(round(p.roi_y_center_override / p.dx))
            if p.roi_y_center_override is not None
            else cy_lat
        )
        roi_y0 = max(0, roi_cy_lat - int(round(p.roi_y_half_D * D_lat)))
        roi_y1 = min(Ny, roi_cy_lat + int(round(p.roi_y_half_D * D_lat)))

        if Nx < 2 or Ny < 2:
            raise ValueError("Domain resolution is too small.")
        if D_lat < 4:
            raise ValueError("Cylinder diameter is too small on the lattice.")
        if not (0 <= cx_lat < Nx and 0 <= cy_lat < Ny):
            raise ValueError("Cylinder center is outside the domain.")
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            raise ValueError("ROI is empty.")

        ds = p.roi_downsample
        roi_nx = roi_x1 - roi_x0
        roi_ny = roi_y1 - roi_y0
        if ds < 1:
            raise ValueError("roi_downsample must be >= 1.")
        if roi_nx % ds != 0:
            raise ValueError(
                f"ROI x-size {roi_nx} is not divisible by downsample factor {ds}."
            )
        if roi_ny % ds != 0:
            raise ValueError(
                f"ROI y-size {roi_ny} is not divisible by downsample factor {ds}."
            )
        output_roi_nx = roi_nx // ds
        output_roi_ny = roi_ny // ds
        dx_out = p.dx * ds

        # Build cylinders_lat: primary cylinder + any extra cylinders.
        r_lat = int(round(D_lat / 2))
        _cyls = [(cx_lat, cy_lat, r_lat)]
        for x_e, y_e, d_e in p.extra_cylinders:
            _cyls.append((
                int(round(x_e / p.dx)),
                int(round(y_e / p.dx)),
                int(round(d_e / (2.0 * p.dx))),
            ))
        cylinders_lat = tuple(_cyls)

        # Stability checks.
        cs = 1.0 / np.sqrt(3.0)
        ma = U_lat / cs
        if ma >= MAX_MA:
            raise ValueError(f"Lattice Mach number {ma:.4f} exceeds {MAX_MA:.3f}.")
        if tau_s < MIN_TAU_S:
            raise ValueError(
                f"tau_s={tau_s:.4f} is below the stability threshold {MIN_TAU_S:.3f}."
            )

        return cls(
            Nx=Nx, Ny=Ny, D_lat=D_lat, U_lat=U_lat, nu_lat=nu_lat,
            tau_s=tau_s, tau_a=tau_a, omega_s=omega_s, omega_a=omega_a,
            cx_lat=cx_lat, cy_lat=cy_lat,
            steps_spinup=steps_spinup, steps_record=steps_record,
            steps_per_frame=steps_per_frame, total_frames=total_frames,
            actual_record_interval=actual_record_interval,
            roi_x0=roi_x0, roi_x1=roi_x1, roi_y0=roi_y0, roi_y1=roi_y1,
            roi_downsample=ds, output_roi_nx=output_roi_nx,
            output_roi_ny=output_roi_ny, dx_out=dx_out,
            cylinders_lat=cylinders_lat,
        )


# ---------------------------------------------------------------------------
# TRT-LBM solver
# ---------------------------------------------------------------------------
class TRTSolver:
    """D2Q9 TRT-LBM solver with NumPy (CPU) or CuPy (GPU) backend.

    All heavy buffers are pre-allocated once; the main loop produces zero
    transient allocations for stream / collide / macroscopic.
    """

    def __init__(self, lc: LatticeConfig, use_gpu: bool = False) -> None:
        if use_gpu and not _HAS_CUPY:
            raise RuntimeError("CuPy is required for GPU acceleration but not installed.")
        self._on_gpu = use_gpu and _HAS_CUPY
        self.xp = cp if self._on_gpu else np

        xp = self.xp
        f32 = xp.float32
        shape9 = (9, lc.Nx, lc.Ny)

        # D2Q9 constants on the target device.
        self._w = xp.asarray(_W_BC, dtype=f32)
        self._cx = xp.asarray(D2Q9_C[:, 0, np.newaxis, np.newaxis], dtype=f32)
        self._cy = xp.asarray(D2Q9_C[:, 1, np.newaxis, np.newaxis], dtype=f32)
        self._opp = xp.asarray(D2Q9_OPP)

        # Pre-allocated work buffers (persist across all steps).
        self._stream_buf = xp.empty(shape9, dtype=f32)
        self._collide_buf = xp.empty(shape9, dtype=f32)
        self._feq_buf = xp.empty(shape9, dtype=f32)
        self._f_opp_buf = xp.empty(shape9, dtype=f32)
        self._feq_opp_buf = xp.empty(shape9, dtype=f32)

        # Cylinder obstacle mask (OR-combined for all cylinders).
        x = xp.arange(lc.Nx, dtype=f32)[:, None]
        y = xp.arange(lc.Ny, dtype=f32)[None, :]
        self._cylinder = xp.zeros((lc.Nx, lc.Ny), dtype=bool)
        for _cx, _cy, _r in lc.cylinders_lat:
            self._cylinder |= (x - _cx) ** 2 + (y - _cy) ** 2 <= xp.float32(_r) ** 2

        # Relaxation rates.
        self._omega_s = lc.omega_s
        self._omega_a = lc.omega_a

    @property
    def on_gpu(self) -> bool:
        """Whether the solver is running on GPU."""
        return self._on_gpu

    # -- Equilibrium --------------------------------------------------------

    def equilibrium(self, rho, u, v, out=None):
        """Compute f_eq.  If *out* is None a new array is allocated."""
        xp = self.xp
        if out is None:
            out = xp.empty((9, rho.shape[0], rho.shape[1]), dtype=xp.float32)
        cu = self._cx * u + self._cy * v
        u2 = u * u + v * v
        out[:] = self._w * rho * (xp.float32(1.0) + xp.float32(3.0) * cu
                                  + xp.float32(4.5) * cu * cu - xp.float32(1.5) * u2)
        return out

    # -- Streaming ----------------------------------------------------------

    def stream(self, f):
        """Zero-allocation streaming via slice assignments.

        Equivalent to np.roll per direction but avoids 9 temporary array
        allocations per step.  Periodic wrap-around values at domain edges
        are overwritten by apply_bcs(), so physical correctness is preserved.
        """
        out = self._stream_buf
        f_pre_cyl = f[:, self._cylinder].copy()

        # i=0: rest (0, 0)
        out[0] = f[0]
        # i=1: east (+1, 0)
        out[1, 1:, :] = f[1, :-1, :]
        out[1, 0, :] = f[1, -1, :]
        # i=2: north (0, +1)
        out[2, :, 1:] = f[2, :, :-1]
        out[2, :, 0] = f[2, :, -1]
        # i=3: west (-1, 0)
        out[3, :-1, :] = f[3, 1:, :]
        out[3, -1, :] = f[3, 0, :]
        # i=4: south (0, -1)
        out[4, :, :-1] = f[4, :, 1:]
        out[4, :, -1] = f[4, :, 0]
        # i=5: north-east (+1, +1)
        out[5, 1:, 1:] = f[5, :-1, :-1]
        out[5, 0, 1:] = f[5, -1, :-1]
        out[5, 1:, 0] = f[5, :-1, -1]
        out[5, 0, 0] = f[5, -1, -1]
        # i=6: north-west (-1, +1)
        out[6, :-1, 1:] = f[6, 1:, :-1]
        out[6, -1, 1:] = f[6, 0, :-1]
        out[6, :-1, 0] = f[6, 1:, -1]
        out[6, -1, 0] = f[6, 0, -1]
        # i=7: south-west (-1, -1)
        out[7, :-1, :-1] = f[7, 1:, 1:]
        out[7, -1, :-1] = f[7, 0, 1:]
        out[7, :-1, -1] = f[7, 1:, 0]
        out[7, -1, -1] = f[7, 0, 0]
        # i=8: south-east (+1, -1)
        out[8, 1:, :-1] = f[8, :-1, 1:]
        out[8, 0, :-1] = f[8, -1, 1:]
        out[8, 1:, -1] = f[8, :-1, 0]
        out[8, 0, -1] = f[8, -1, 0]

        return out, f_pre_cyl

    # -- Boundary conditions ------------------------------------------------

    def apply_bcs(self, f, f_pre_cyl, u_inlet):
        """Apply boundary conditions in-place on *f*.

        Order: cylinder bounce-back → outlet extrapolation → Zou-He inlet
        → top/bottom wall bounce-back.
        """
        # Cylinder: half-way bounce-back.
        f[:, self._cylinder] = f_pre_cyl[self._opp, :]

        # Outlet (x = Nx-1): first-order extrapolation for west-bound populations.
        f[[3, 6, 7], -1, :] = f[[3, 6, 7], -2, :]

        # Inlet (x = 0): Zou-He velocity boundary condition (assumes v_inlet = 0).
        rho_in = (1.0 / (1.0 - u_inlet)) * (
            f[0, 0, :] + f[2, 0, :] + f[4, 0, :]
            + 2.0 * (f[3, 0, :] + f[6, 0, :] + f[7, 0, :])
        )
        ru = rho_in * u_inlet
        df24 = f[2, 0, :] - f[4, 0, :]
        f[1, 0, :] = f[3, 0, :] + (2.0 / 3.0) * ru
        f[5, 0, :] = f[7, 0, :] - 0.5 * df24 + (1.0 / 6.0) * ru
        f[8, 0, :] = f[6, 0, :] + 0.5 * df24 + (1.0 / 6.0) * ru

        # Top wall (y = Ny-1): bounce-back.
        f[4, :, -1], f[7, :, -1], f[8, :, -1] = f[2, :, -1], f[5, :, -1], f[6, :, -1]
        # Bottom wall (y = 0): bounce-back.
        f[2, :, 0], f[5, :, 0], f[6, :, 0] = f[4, :, 0], f[7, :, 0], f[8, :, 0]

        return f

    # -- Macroscopic quantities ---------------------------------------------

    def macroscopic(self, f):
        """Compute density and velocity from distribution.

        Uses direct D2Q9 accumulation:
          Cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
          Cy = [0, 0, 1,  0,-1, 1,  1, -1,-1]
        Each f[i] is a zero-copy (Nx, Ny) view; no (9, Nx, Ny) temporaries
        are created.
        """
        xp = self.xp
        rho = xp.sum(f, axis=0)
        inv_rho = xp.float32(1.0) / xp.maximum(rho, xp.float32(1e-12))
        u = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) * inv_rho
        v = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) * inv_rho
        u[self._cylinder] = 0.0
        v[self._cylinder] = 0.0
        return rho, u, v

    # -- TRT collision ------------------------------------------------------

    def collide(self, f, feq):
        """Two-relaxation-time collision.

        Uses numexpr on CPU (if available) for fused evaluation; CuPy's
        element-wise backend handles fusion automatically on GPU.
        Pre-allocated _f_opp_buf / _feq_opp_buf avoid transient allocations
        that fancy indexing (f[opp]) would otherwise cause.
        """
        xp = self.xp
        f_opp = xp.take(f, self._opp, axis=0, out=self._f_opp_buf)
        feq_opp = xp.take(feq, self._opp, axis=0, out=self._feq_opp_buf)
        os_h = np.float32(0.5 * self._omega_s)
        oa_h = np.float32(0.5 * self._omega_a)
        out = self._collide_buf

        if not self._on_gpu and _HAS_NUMEXPR:
            ne.evaluate(
                "f - os * (f + fo - fq - fqo) - oa * (f - fo - fq + fqo)",
                local_dict={
                    "f": f, "fo": f_opp, "fq": feq, "fqo": feq_opp,
                    "os": os_h, "oa": oa_h,
                },
                out=out,
            )
        else:
            out[:] = (f - os_h * (f + f_opp - feq - feq_opp)
                      - oa_h * (f - f_opp - feq + feq_opp))
        return out

    # -- Full step ----------------------------------------------------------

    def step(self, f, u_inlet):
        """Execute one LBM step: stream → BCs → macroscopic → feq → collide.

        Returns ``(f_new, u, v)`` where *f_new* is the post-collision
        distribution and *u*, *v* are the macroscopic velocity fields.
        """
        f_s, f_pre_cyl = self.stream(f)
        self.apply_bcs(f_s, f_pre_cyl, u_inlet)
        rho, u, v = self.macroscopic(f_s)
        feq = self.equilibrium(rho, u, v, out=self._feq_buf)
        f_new = self.collide(f_s, feq)
        return f_new, u, v


# ---------------------------------------------------------------------------
# Turbulent inlet generation (always CPU, batched)
# ---------------------------------------------------------------------------
def generate_inlet_batch(
    Ny: int,
    batch_size: int,
    rng: np.random.Generator,
    intensity: float,
    length_scale: int,
    U_lat: float,
) -> np.ndarray:
    """Generate *batch_size* inlet velocity profiles at once on CPU.

    Returns an array of shape ``(batch_size, Ny)`` in float32.
    Spatial correlations are introduced via a 1-D uniform filter, then
    profiles are normalised to the target turbulence intensity.
    """
    if intensity <= 0.0:
        return np.full((batch_size, Ny), U_lat, dtype=np.float32)

    pad = length_scale
    raw = rng.standard_normal((batch_size, Ny + 2 * pad)).astype(np.float32)
    filtered = uniform_filter1d(raw, size=length_scale, axis=1, mode="reflect")
    filtered = filtered[:, pad: pad + Ny]

    std = filtered.std(axis=1, keepdims=True)
    # Avoid division by zero when the filtered noise is negligible.
    std = np.where(std < 1e-12, np.float32(1.0), std)
    return np.float32(U_lat) + filtered / std * np.float32(intensity * U_lat)


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------
def validate_strouhal(
    v_monitor: List[float],
    sample_dt: float,
    D_phys: float,
    U_phys: float,
) -> float | None:
    """Estimate Strouhal number from a velocity time series."""
    if len(v_monitor) < 2 or sample_dt <= 0.0:
        return None

    signal = np.asarray(v_monitor, dtype=np.float64)
    signal -= signal.mean()
    fft_amp = np.abs(np.fft.rfft(signal))
    if fft_amp.size < 2:
        return None

    fft_amp[0] = 0.0
    dominant_idx = int(np.argmax(fft_amp))
    if dominant_idx == 0 or fft_amp[dominant_idx] <= 0.0:
        return None

    fft_freq = np.fft.rfftfreq(len(signal), d=sample_dt)
    dominant_freq = float(fft_freq[dominant_idx])
    if dominant_freq <= 0.0:
        return None
    return dominant_freq * D_phys / U_phys


def estimate_vortex_phase(v_signal: np.ndarray) -> np.ndarray:
    """Hilbert-transform based vortex-phase estimation."""
    signal = np.asarray(v_signal, dtype=np.float32)
    if signal.size == 0:
        return np.empty(0, dtype=np.float32)
    if signal.size == 1:
        return np.zeros(1, dtype=np.float32)

    from numpy.fft import fft, ifft

    n = signal.size
    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1: n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0
    analytic = ifft(fft(signal) * h)
    return (np.angle(analytic) % (2.0 * np.pi)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _to_numpy(arr) -> np.ndarray:
    """Transfer to CPU numpy array.  No-op if already numpy."""
    if hasattr(arr, "get"):  # CuPy arrays expose .get()
        return arr.get()
    return np.asarray(arr)


def _tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def stable_dt_for_case(
    Re: float,
    U_phys: float,
    D_phys: float,
    dx: float,
    base_dt: float = 0.025,
    min_tau_s: float = MIN_TAU_S,
    max_ma: float = MAX_MA,
) -> float:
    """Find a feasible dt satisfying both tau_s and Mach constraints."""
    min_dt = ((min_tau_s - 0.5) * dx * dx * Re) / (3.0 * U_phys * D_phys)
    dt = max(base_dt, min_dt * 1.01)
    max_dt = max_ma * dx / (U_phys * np.sqrt(3.0))
    if dt >= max_dt:
        raise ValueError(
            f"No feasible dt for Re={Re:.0f}: dt={dt:.5f} violates Mach limit {max_dt:.5f}."
        )
    return float(dt)


def make_tandem_physics_config(
    Re: float = 150.0,
    U_phys: float = 1.0,
    gap_ratio: float = 3.5,
) -> PhysicsConfig:
    """Return a PhysicsConfig for tandem (串联) dual-cylinder flow.

    Two identical cylinders of diameter D are aligned along the stream-wise
    direction with a surface-to-surface gap of gap_ratio * D.  The ROI starts
    1D upstream of the first cylinder so the agent can learn gap-traversal
    strategies (Option A from the design spec).

    Parameters
    ----------
    Re : float
        Reynolds number (default 150).
    U_phys : float
        Free-stream velocity in m/s (default 1.0).
    gap_ratio : float
        Surface-to-surface gap normalised by D (default 3.5 — critical spacing).
    """
    D = 12.0          # cylinder diameter [m], consistent with navigation profile
    dx = 0.3          # grid spacing [m]
    cyl1_x = 96.0     # upstream cylinder x-position [m]
    center_y = 90.0   # domain y-midline [m]
    gap_phys = gap_ratio * D
    cyl2_x = cyl1_x + D + gap_phys  # centre-to-centre = (1 + gap_ratio) * D

    dt = stable_dt_for_case(Re=Re, U_phys=U_phys, D_phys=D, dx=dx, base_dt=0.015)

    return PhysicsConfig(
        Re=Re,
        U_phys=U_phys,
        D_phys=D,
        Lx_phys=540.0,        # longer domain to accommodate two cylinders
        Ly_phys=180.0,
        dx=dx,
        dt=dt,
        cyl_x_phys=cyl1_x,
        cyl_y_center=center_y,
        extra_cylinders=[(cyl2_x, center_y, D)],
        turbulence_intensity=0.05,
        turbulence_length_scale=20,
        T_spinup_phys=720.0,
        T_record_phys=360.0,
        record_interval=0.3,
        roi_x_start_D=-1.0,   # 1D before upstream cylinder (covers gap region)
        roi_x_end_D=24.5,     # 20D past downstream cylinder from upstream origin
        roi_y_half_D=3.0,
        roi_downsample=2,
        case_tag="tandem_G35_",
    )


def make_side_by_side_physics_config(
    Re: float = 150.0,
    U_phys: float = 1.0,
    gap_ratio: float = 3.5,
) -> PhysicsConfig:
    """Return a PhysicsConfig for side-by-side (并排) dual-cylinder flow.

    Two identical cylinders of diameter D are placed at the same x-position,
    separated by a surface-to-surface gap of gap_ratio * D in the y-direction.
    The domain is taller than the single-cylinder case to avoid wall interference.
    The ROI y-centre is overridden to the domain midline so both wakes are
    captured symmetrically.

    Parameters
    ----------
    Re : float
        Reynolds number (default 150).
    U_phys : float
        Free-stream velocity in m/s (default 1.0).
    gap_ratio : float
        Surface-to-surface gap normalised by D (default 3.5).
    """
    D = 12.0
    dx = 0.3
    cyl_x = 96.0
    domain_center_y = 120.0   # taller domain (Ly=240m), centre at 120m
    gap_phys = gap_ratio * D
    half_cc = (D + gap_phys) / 2.0   # half of centre-to-centre distance
    cyl1_y = domain_center_y + half_cc   # upper cylinder
    cyl2_y = domain_center_y - half_cc   # lower cylinder

    dt = stable_dt_for_case(Re=Re, U_phys=U_phys, D_phys=D, dx=dx, base_dt=0.015)

    return PhysicsConfig(
        Re=Re,
        U_phys=U_phys,
        D_phys=D,
        Lx_phys=480.0,
        Ly_phys=240.0,         # taller domain: ±6D clearance from each cylinder
        dx=dx,
        dt=dt,
        cyl_x_phys=cyl_x,
        cyl_y_center=cyl1_y,  # primary cylinder (upper)
        extra_cylinders=[(cyl_x, cyl2_y, D)],
        turbulence_intensity=0.05,
        turbulence_length_scale=20,
        T_spinup_phys=720.0,
        T_record_phys=360.0,
        record_interval=0.3,
        roi_x_start_D=2.0,
        roi_x_end_D=25.0,
        roi_y_half_D=6.0,      # ±6D about override centre covers both wakes
        roi_y_center_override=domain_center_y,
        roi_downsample=2,
        case_tag="sbs_G35_",
    )


def estimate_case_storage_bytes(pc: PhysicsConfig) -> int:
    """Estimate output file size (float16, 3 channels, after downsampling)."""
    lc = LatticeConfig.from_physics(pc)
    return lc.total_frames * lc.output_roi_nx * lc.output_roi_ny * 3 * np.dtype(np.float16).itemsize


def estimate_dataset_storage_bytes(configs: List[PhysicsConfig]) -> int:
    return sum(estimate_case_storage_bytes(pc) for pc in configs)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_simulation(
    pc: PhysicsConfig,
    output_dir: str = ".",
    rng_seed: int = 42,
    use_gpu: bool = False,
) -> str:
    """Run one TRT-LBM case and stream output to disk."""

    lc = LatticeConfig.from_physics(pc)
    rng = np.random.default_rng(rng_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    solver = TRTSolver(lc, use_gpu=use_gpu)
    xp = solver.xp

    backend = "CuPy (GPU)" if solver.on_gpu else "NumPy (CPU)"
    print(f"\nPreparing case: Re={pc.Re:.0f}, dt={pc.dt:.5f}, tau_s={lc.tau_s:.4f}  [{backend}]")

    # --- Initialise distribution ---
    rho_init = xp.ones((lc.Nx, lc.Ny), dtype=xp.float32)
    u_init = xp.full((lc.Nx, lc.Ny), lc.U_lat, dtype=xp.float32)
    v_init = xp.zeros((lc.Nx, lc.Ny), dtype=xp.float32)

    # Small asymmetric perturbation downstream of cylinder to trigger shedding.
    px0, px1 = lc.cx_lat, min(lc.cx_lat + lc.D_lat, lc.Nx)
    py0, py1 = max(0, lc.cy_lat - lc.D_lat // 4), lc.cy_lat
    u_init[px0:px1, py0:py1] *= xp.float32(1.002)

    f = solver.equilibrium(rho_init, u_init, v_init)

    # --- Output files ---
    ds = lc.roi_downsample
    tag = (
        f"{pc.case_tag}"          # empty string for single-cylinder cases
        f"v8_U{_tag_float(pc.U_phys)}_"
        f"Re{pc.Re:.0f}_"
        f"D{_tag_float(pc.D_phys)}_"
        f"dx{_tag_float(lc.dx_out)}_"
        f"Ti{pc.turbulence_intensity * 100:.0f}pct_"
        f"{lc.total_frames}f_roi"
    )
    flow_path = output_path / f"wake_{tag}.npy"
    phase_path = output_path / f"wake_{tag}_phase.npy"
    meta_path = output_path / f"wake_{tag}_meta.json"

    flow_data = np.empty(
        (lc.total_frames, lc.output_roi_nx, lc.output_roi_ny, 3),
        dtype=np.float16,
    )

    # --- Monitor setup ---
    total_steps = lc.steps_spinup + lc.steps_record
    mon_x = min(lc.cx_lat + 3 * lc.D_lat, lc.Nx - 1)
    mon_y = min(lc.cy_lat + int(round(0.7 * lc.D_lat)), lc.Ny - 1)
    monitor_samples = 2000
    monitor_start = max(0, lc.steps_spinup - monitor_samples * lc.steps_per_frame)

    v_spinup_monitor: List[float] = []
    v_record_monitor: List[float] = []
    st_measured: float | None = None
    frame_idx = 0
    velocity_scale = np.float32(pc.dx / pc.dt)
    t_start = time.time()

    # --- Pre-generate first inlet batch ---
    batch_size = min(INLET_BATCH_SIZE, total_steps)
    inlet_batch_cpu = generate_inlet_batch(
        lc.Ny, batch_size, rng,
        pc.turbulence_intensity, pc.turbulence_length_scale, lc.U_lat,
    )
    inlet_batch = xp.asarray(inlet_batch_cpu) if solver.on_gpu else inlet_batch_cpu
    inlet_idx = 0

    # --- Main loop ---
    for step in tqdm(range(total_steps), desc=f"Re={pc.Re:.0f}", ncols=80):
        # Refill inlet batch when exhausted.
        if inlet_idx >= batch_size:
            inlet_batch_cpu = generate_inlet_batch(
                lc.Ny, batch_size, rng,
                pc.turbulence_intensity, pc.turbulence_length_scale, lc.U_lat,
            )
            inlet_batch = xp.asarray(inlet_batch_cpu) if solver.on_gpu else inlet_batch_cpu
            inlet_idx = 0

        u_inlet = inlet_batch[inlet_idx]
        inlet_idx += 1

        f, u, v = solver.step(f, u_inlet)

        # Spinup monitoring for Strouhal validation.
        if monitor_start <= step < lc.steps_spinup and (step - monitor_start) % lc.steps_per_frame == 0:
            v_spinup_monitor.append(float(v[mon_x, mon_y]))

        if step == lc.steps_spinup - 1:
            st_measured = validate_strouhal(
                v_spinup_monitor, lc.actual_record_interval, pc.D_phys, pc.U_phys,
            )

        # Frame recording.
        if step >= lc.steps_spinup:
            local_step = step - lc.steps_spinup
            if local_step % lc.steps_per_frame == 0:
                if frame_idx >= lc.total_frames:
                    raise RuntimeError("Frame index exceeded allocated output shape.")

                u_roi = u[lc.roi_x0: lc.roi_x1, lc.roi_y0: lc.roi_y1]
                v_roi = v[lc.roi_x0: lc.roi_x1, lc.roi_y0: lc.roi_y1]

                # Vorticity at full simulation resolution, then spatial downsample.
                dv_dx = xp.gradient(v_roi, pc.dx, axis=0)
                du_dy = xp.gradient(u_roi, pc.dx, axis=1)
                omega_roi = dv_dx - du_dy

                # Scale to physical units and transfer to CPU.
                frame = xp.stack([
                    u_roi[::ds, ::ds] * velocity_scale,
                    v_roi[::ds, ::ds] * velocity_scale,
                    omega_roi[::ds, ::ds] * velocity_scale,
                ], axis=-1)
                flow_data[frame_idx] = _to_numpy(frame).astype(np.float16)

                v_record_monitor.append(float(v[mon_x, mon_y]))
                frame_idx += 1

    if frame_idx != lc.total_frames:
        raise RuntimeError(f"Recorded {frame_idx} frames, expected {lc.total_frames}.")

    np.save(flow_path, flow_data)
    del flow_data

    # --- Post-processing ---
    phase_arr = estimate_vortex_phase(np.asarray(v_record_monitor, dtype=np.float32))
    np.save(phase_path, phase_arr)

    lambda_vortex_D = None
    if st_measured is not None and st_measured > 0.0:
        lambda_vortex_D = 1.0 / st_measured

    elapsed_s = time.time() - t_start
    meta = {
        "generator_version": "v8.0",
        "Re": float(pc.Re),
        "St_measured": None if st_measured is None else float(st_measured),
        "lambda_vortex_D": None if lambda_vortex_D is None else float(lambda_vortex_D),
        "D_ref": float(pc.D_phys),
        "U_ref": float(pc.U_phys),
        # dx_m is the OUTPUT grid spacing used by downstream WakeFieldMetadata
        # to build coordinate arrays.  dx_sim_m records the finer simulation grid.
        "dx_m": float(lc.dx_out),
        "dx_sim_m": float(pc.dx),
        "dt_s": float(pc.dt),
        "record_interval_target_s": float(pc.record_interval),
        "record_interval_actual_s": float(lc.actual_record_interval),
        "spinup_duration_actual_s": float(lc.steps_spinup * pc.dt),
        "record_duration_actual_s": float(lc.steps_record * pc.dt),
        "turbulence_intensity": float(pc.turbulence_intensity),
        "Nx_full": int(lc.Nx),
        "Ny_full": int(lc.Ny),
        "roi_x0_phys_m": float(lc.roi_x0 * pc.dx),
        "roi_x1_phys_m": float(lc.roi_x1 * pc.dx),
        "roi_y0_phys_m": float(lc.roi_y0 * pc.dx),
        "roi_y1_phys_m": float(lc.roi_y1 * pc.dx),
        "roi_downsample": int(lc.roi_downsample),
        "spinup_steps": int(lc.steps_spinup),
        "record_steps": int(lc.steps_record),
        "total_frames": int(lc.total_frames),
        "data_shape": [int(lc.total_frames), int(lc.output_roi_nx), int(lc.output_roi_ny), 3],
        "channels": ["u_mps", "v_mps", "omega_1ps"],
        "dtype": "float16",
        "backend": backend,
        "rng_seed": int(rng_seed),
        "elapsed_s": float(elapsed_s),
    }

    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, sort_keys=True)

    print(f"Saved flow data to {flow_path.name}")
    return str(flow_path)


# ---------------------------------------------------------------------------
# Training configuration builder
# ---------------------------------------------------------------------------
def make_training_configs(
    profile: str = "navigation",
    re_values: Optional[Sequence[float]] = None,
    u_values: Optional[Sequence[float]] = None,
) -> List[PhysicsConfig]:
    """Build a list of PhysicsConfig for the requested training profile.

    Parameters
    ----------
    profile : str
        Name of the predefined navigation profile.
    re_values : sequence of float, optional
        Override the profile's default Re values.
    u_values : sequence of float, optional
        Override the profile's default U_phys values.
    """

    profiles = {
        "navigation": NavigationProfile(
            name="navigation",
            # Cover the REMUS-100 operating speed range (1.0–1.5 m/s).
            U_phys_values=(1.0, 1.25, 1.5),
            # Two Re values for different wake complexity; chosen for LBM
            # stability (tau_s ∈ [0.524, 0.56]) and mechanism diversity.
            Re_values=(150.0, 250.0),
            # Cylinder diameter: 12 m keeps scale ratio L_AUV/D ≈ 0.13,
            # within the "equivalent current" approximation regime.
            D_phys=12.0,
            Lx_phys=480.0,
            Ly_phys=180.0,
            # dx=0.3 m gives D_lat=40, which is sufficient LBM resolution.
            dx=0.3,
            # base_dt=0.015 s is the largest dt that keeps all cases below
            # the Mach limit (max_dt ≈ 0.0173 s at U=1.5 m/s, dx=0.3 m).
            base_dt=0.015,
            cyl_x_phys=96.0,
            cyl_y_center=90.0,
            # 720 s ≈ 60 convective times (D/U) for the slowest case —
            # sufficient to establish periodic vortex shedding at Re 150–250.
            T_spinup_phys=720.0,
            # 360 s record time with margin for randomised episode start times
            # (max episode 240 s) and time-looping.
            T_record_phys=360.0,
            # 0.3 s record interval: fastest vortex period ~40 s gives
            # Nyquist ratio > 66×.  Env sim_dt=0.1 s interpolates between
            # ~1.7 frames per 0.5 s control decision step.
            record_interval=0.3,
            roi_x_end_D=18.0,
            roi_y_half_D=2.5,
            turbulence_intensity=0.05,
            turbulence_length_scale=20,
            # 2× spatial downsample: output dx_out=0.6 m.  Downstream
            # FlowSampler performs bilinear point queries; probe offsets ±2 m
            # span ~3.3 output cells, providing independent spatial information.
            roi_downsample=2,
        ),
    }
    # --- multi-cylinder profiles (built from preset functions) ---
    _MULTI_CYL_RE = (150.0, 200.0)
    _MULTI_CYL_U  = (0.8, 1.0, 1.2)

    if profile == "tandem_G35_nav":
        actual_re = tuple(re_values) if re_values is not None else _MULTI_CYL_RE
        actual_u  = tuple(u_values)  if u_values  is not None else _MULTI_CYL_U
        return [
            make_tandem_physics_config(Re=Re, U_phys=U)
            for U in actual_u
            for Re in actual_re
        ]

    if profile == "side_by_side_G35_nav":
        actual_re = tuple(re_values) if re_values is not None else _MULTI_CYL_RE
        actual_u  = tuple(u_values)  if u_values  is not None else _MULTI_CYL_U
        return [
            make_side_by_side_physics_config(Re=Re, U_phys=U)
            for U in actual_u
            for Re in actual_re
        ]

    if profile not in profiles:
        raise ValueError(f"Unknown training profile: {profile}")

    profile_cfg = profiles[profile]

    actual_u = tuple(u_values) if u_values is not None else profile_cfg.U_phys_values
    actual_re = tuple(re_values) if re_values is not None else profile_cfg.Re_values

    return [
        profile_cfg.to_physics_config(Re=Re, U_phys=U_phys)
        for U_phys in actual_u
        for Re in actual_re
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    profile: str = "navigation",
    use_gpu: bool = False,
    re_values: Optional[Sequence[float]] = None,
    u_values: Optional[Sequence[float]] = None,
) -> None:
    output_dir = Path("./wake_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  TRT-LBM wake generator v8.0")
    print("=" * 65)

    configs = make_training_configs(profile=profile, re_values=re_values, u_values=u_values)
    estimated_gib = estimate_dataset_storage_bytes(configs) / 1024**3
    print(f"\n  Profile: '{profile}', {len(configs)} cases.")
    print(f"  Estimated output footprint: {estimated_gib:.2f} GiB.")
    if use_gpu:
        if _HAS_CUPY:
            dev = cp.cuda.Device()
            print(f"  GPU: {dev.id} ({cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()})")
        else:
            print("  WARNING: --gpu requested but CuPy not available; falling back to CPU.")
            use_gpu = False
    print()

    for idx, pc in enumerate(configs, start=1):
        print(f"\n{'=' * 65}")
        print(f"  [{idx}/{len(configs)}] Re={pc.Re:.0f}, U={pc.U_phys:.2f} m/s")
        print(f"{'=' * 65}")
        try:
            run_simulation(pc, output_dir=str(output_dir), rng_seed=41 + idx, use_gpu=use_gpu)
        except Exception as exc:
            print(f"  Skipping case ({type(exc).__name__}: {exc})")


def cli() -> None:
    parser = argparse.ArgumentParser(description="TRT-LBM wake generator v8.0")
    parser.add_argument(
        "--profile",
        default="navigation",
        help=(
            "Training profile name. Available: navigation, "
            "tandem_G35_nav, side_by_side_G35_nav (default: navigation)"
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration via CuPy")
    parser.add_argument("--re", type=float, nargs="*", default=None, metavar="RE",
                        help="Reynolds number(s), e.g. --re 150 250. Defaults to profile values.")
    parser.add_argument("--u", type=float, nargs="*", default=None, metavar="U",
                        help="Free-stream velocity(s) in m/s, e.g. --u 1.0 1.5. Defaults to profile values.")
    args = parser.parse_args()
    main(profile=args.profile, use_gpu=args.gpu, re_values=args.re, u_values=args.u)


if __name__ == "__main__":
    cli()
