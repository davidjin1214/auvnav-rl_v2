# Multi-Cylinder Wake Field Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `scripts/generate_wake.py` to support tandem (串联) and side-by-side (并排) dual-cylinder TRT-LBM simulations via new `PhysicsConfig` fields and high-level preset functions, with zero changes to existing single-cylinder behavior.

**Architecture:** Add `extra_cylinders` and `roi_y_center_override` fields to `PhysicsConfig`; add `cylinders_lat` to `LatticeConfig`; update `TRTSolver` to OR-combine multiple circle masks; expose new profiles through `make_training_configs()` and the existing `--profile` CLI flag.

**Tech Stack:** Python 3.10+, NumPy, (optional) CuPy, dataclasses.

---

## File Map

| File | Change |
|------|--------|
| `scripts/generate_wake.py` | All changes for this sub-project |
| `tests/test_multi_cylinder_lbm.py` | New test file (create) |

---

### Task 1: Extend PhysicsConfig + LatticeConfig + TRTSolver mask

**Files:**
- Modify: `scripts/generate_wake.py` (PhysicsConfig ~L84, LatticeConfig ~L155, TRTSolver ~L311)
- Create: `tests/test_multi_cylinder_lbm.py`

- [ ] **Step 1.1: Create test file and write failing tests**

Create `tests/__init__.py` (empty) and `tests/test_multi_cylinder_lbm.py`:

```python
"""Tests for multi-cylinder LBM extensions — no simulation is run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scripts.generate_wake import PhysicsConfig, LatticeConfig, TRTSolver


def _minimal_single_cyl() -> PhysicsConfig:
    """Minimal valid single-cylinder config for fast tests."""
    return PhysicsConfig(
        Re=150.0, U_phys=1.0, D_phys=12.0,
        Lx_phys=240.0, Ly_phys=120.0, dx=0.3, dt=0.015,
        cyl_x_phys=48.0, cyl_y_center=60.0,
        T_spinup_phys=1.0, T_record_phys=1.0, record_interval=0.3,
        roi_x_end_D=10.0, roi_y_half_D=2.5,
    )


def test_physics_config_extra_cylinders_default_empty():
    """extra_cylinders defaults to empty list."""
    pc = _minimal_single_cyl()
    assert pc.extra_cylinders == []


def test_physics_config_roi_y_center_override_default_none():
    """roi_y_center_override defaults to None."""
    pc = _minimal_single_cyl()
    assert pc.roi_y_center_override is None


def test_lattice_config_cylinders_lat_single():
    """Single cylinder → cylinders_lat has exactly one entry."""
    pc = _minimal_single_cyl()
    lc = LatticeConfig.from_physics(pc)
    assert len(lc.cylinders_lat) == 1
    cx, cy, r = lc.cylinders_lat[0]
    assert cx == int(round(48.0 / 0.3))   # 160
    assert cy == int(round(60.0 / 0.3))   # 200
    assert r == int(round(12.0 / (2 * 0.3)))  # 20


def test_lattice_config_cylinders_lat_two():
    """Two cylinders → cylinders_lat has exactly two entries."""
    pc = _minimal_single_cyl()
    pc = PhysicsConfig(
        **{**pc.__dict__,
           "extra_cylinders": [(120.0, 60.0, 12.0)]}
    )
    lc = LatticeConfig.from_physics(pc)
    assert len(lc.cylinders_lat) == 2
    cx2, cy2, r2 = lc.cylinders_lat[1]
    assert cx2 == int(round(120.0 / 0.3))  # 400
    assert cy2 == int(round(60.0 / 0.3))   # 200
    assert r2 == int(round(12.0 / (2 * 0.3)))  # 20


def test_roi_y_center_override_shifts_roi():
    """roi_y_center_override repositions ROI y-bounds."""
    pc = _minimal_single_cyl()
    # Domain center 60m, but override to 45m
    pc_override = PhysicsConfig(
        **{**pc.__dict__, "roi_y_center_override": 45.0}
    )
    lc_default = LatticeConfig.from_physics(pc)
    lc_override = LatticeConfig.from_physics(pc_override)
    # ROI y-center should differ
    default_center = (lc_default.roi_y0 + lc_default.roi_y1) // 2
    override_center = (lc_override.roi_y0 + lc_override.roi_y1) // 2
    assert abs(override_center - int(round(45.0 / 0.3))) <= 1


def test_trt_solver_single_cyl_mask_shape():
    """Single cylinder mask has correct shape and contains cylinder region."""
    pc = _minimal_single_cyl()
    lc = LatticeConfig.from_physics(pc)
    solver = TRTSolver(lc, use_gpu=False)
    mask = solver._cylinder
    assert mask.shape == (lc.Nx, lc.Ny)
    # Centre pixel must be solid
    assert bool(mask[lc.cx_lat, lc.cy_lat])


def test_trt_solver_two_cyl_mask_both_solid():
    """Two-cylinder mask marks both cylinder centres as solid."""
    pc = PhysicsConfig(
        Re=150.0, U_phys=1.0, D_phys=12.0,
        Lx_phys=240.0, Ly_phys=120.0, dx=0.3, dt=0.015,
        cyl_x_phys=48.0, cyl_y_center=60.0,
        T_spinup_phys=1.0, T_record_phys=1.0, record_interval=0.3,
        roi_x_end_D=10.0, roi_y_half_D=2.5,
        extra_cylinders=[(120.0, 60.0, 12.0)],
    )
    lc = LatticeConfig.from_physics(pc)
    solver = TRTSolver(lc, use_gpu=False)
    mask = solver._cylinder
    # Primary cylinder centre must be solid
    assert bool(mask[lc.cx_lat, lc.cy_lat])
    # Secondary cylinder centre must also be solid
    cx2 = int(round(120.0 / 0.3))
    cy2 = int(round(60.0 / 0.3))
    assert bool(mask[cx2, cy2])
    # A point between the two cylinders (not overlapping either) must be fluid
    mid_x = (lc.cx_lat + cx2) // 2
    assert not bool(mask[mid_x, lc.cy_lat])
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/new_off_rl/rl_v2
python -m pytest tests/test_multi_cylinder_lbm.py -v 2>&1 | head -40
```

Expected: multiple ERRORS about `extra_cylinders` / `cylinders_lat` not existing.

- [ ] **Step 1.3: Add fields to PhysicsConfig**

In `scripts/generate_wake.py`, the `PhysicsConfig` dataclass ends with `roi_downsample: int = 1` (around line 107). Add two new fields immediately after:

```python
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
```

Also add `from dataclasses import dataclass, field` if `field` is not yet imported. Check the existing import at the top — it currently reads `from dataclasses import dataclass`. Change it to:

```python
from dataclasses import dataclass, field
```

- [ ] **Step 1.4: Add cylinders_lat field to LatticeConfig**

`LatticeConfig` is a `@dataclass(frozen=True)`. All its current fields have no defaults (they are all `int` or `float` without `=`). Add `cylinders_lat` at the end with a default so existing callsites don't break:

```python
    dx_out: float
    cylinders_lat: tuple = ()   # NEW: all cylinders as (cx_lat, cy_lat, r_lat)
```

- [ ] **Step 1.5: Populate cylinders_lat in LatticeConfig.from_physics()**

In `LatticeConfig.from_physics()`, just before the `return cls(...)` call (around line 265), add the `cylinders_lat` computation and ROI y-centre override. Replace the existing ROI y lines and the final `return cls(...)` call:

Find these lines (around L224–L227):
```python
        roi_x0 = max(0, cx_lat + int(round(p.roi_x_start_D * D_lat)))
        roi_x1 = min(Nx, cx_lat + int(round(p.roi_x_end_D * D_lat)))
        roi_y0 = max(0, cy_lat - int(round(p.roi_y_half_D * D_lat)))
        roi_y1 = min(Ny, cy_lat + int(round(p.roi_y_half_D * D_lat)))
```

Replace with:
```python
        roi_x0 = max(0, cx_lat + int(round(p.roi_x_start_D * D_lat)))
        roi_x1 = min(Nx, cx_lat + int(round(p.roi_x_end_D * D_lat)))
        roi_cy_lat = (
            int(round(p.roi_y_center_override / p.dx))
            if p.roi_y_center_override is not None
            else cy_lat
        )
        roi_y0 = max(0, roi_cy_lat - int(round(p.roi_y_half_D * D_lat)))
        roi_y1 = min(Ny, roi_cy_lat + int(round(p.roi_y_half_D * D_lat)))
```

Then, just before the stability checks (around L255, before `cs = 1.0 / np.sqrt(3.0)`), add:
```python
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
```

Finally, in the `return cls(...)` call (around L265), add `cylinders_lat=cylinders_lat` as a new keyword argument:
```python
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
```

- [ ] **Step 1.6: Update TRTSolver to build combined cylinder mask**

In `TRTSolver.__init__()`, find this single line (around L314):
```python
        self._cylinder = (x - lc.cx_lat) ** 2 + (y - lc.cy_lat) ** 2 <= (lc.D_lat / 2.0) ** 2
```

Replace with:
```python
        self._cylinder = xp.zeros((lc.Nx, lc.Ny), dtype=bool)
        for _cx, _cy, _r in lc.cylinders_lat:
            self._cylinder |= (x - _cx) ** 2 + (y - _cy) ** 2 <= xp.float32(_r) ** 2
```

- [ ] **Step 1.7: Run tests — expect pass**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -v
```

Expected output:
```
PASSED tests/test_multi_cylinder_lbm.py::test_physics_config_extra_cylinders_default_empty
PASSED tests/test_multi_cylinder_lbm.py::test_physics_config_roi_y_center_override_default_none
PASSED tests/test_multi_cylinder_lbm.py::test_lattice_config_cylinders_lat_single
PASSED tests/test_multi_cylinder_lbm.py::test_lattice_config_cylinders_lat_two
PASSED tests/test_multi_cylinder_lbm.py::test_roi_y_center_override_shifts_roi
PASSED tests/test_multi_cylinder_lbm.py::test_trt_solver_single_cyl_mask_shape
PASSED tests/test_multi_cylinder_lbm.py::test_trt_solver_two_cyl_mask_both_solid
7 passed
```

- [ ] **Step 1.8: Commit**

```bash
git add scripts/generate_wake.py tests/__init__.py tests/test_multi_cylinder_lbm.py
git commit -m "feat: extend PhysicsConfig/LatticeConfig/TRTSolver for multi-cylinder LBM"
```

---

### Task 2: Add make_tandem_physics_config() preset

**Files:**
- Modify: `scripts/generate_wake.py` (add function after `stable_dt_for_case`, around L602)
- Modify: `tests/test_multi_cylinder_lbm.py` (add tests)

- [ ] **Step 2.1: Write failing test for tandem preset**

Append to `tests/test_multi_cylinder_lbm.py`:

```python
from scripts.generate_wake import make_tandem_physics_config


def test_make_tandem_physics_config_domain():
    """Tandem config has correct domain size."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0)
    assert pc.Lx_phys == pytest.approx(540.0)
    assert pc.Ly_phys == pytest.approx(180.0)


def test_make_tandem_physics_config_two_cylinders():
    """Tandem config produces exactly two cylinders in LatticeConfig."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    assert len(lc.cylinders_lat) == 2


def test_make_tandem_physics_config_cyl2_downstream():
    """Second cylinder is downstream (larger x) of first."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    cx1, _, _ = lc.cylinders_lat[0]
    cx2, _, _ = lc.cylinders_lat[1]
    assert cx2 > cx1


def test_make_tandem_physics_config_gap_ratio():
    """Gap between cylinder surfaces equals 3.5 * D."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0, gap_ratio=3.5)
    lc = LatticeConfig.from_physics(pc)
    cx1, _, r1 = lc.cylinders_lat[0]
    cx2, _, r2 = lc.cylinders_lat[1]
    # surface-to-surface gap in lattice units
    gap_lat = cx2 - r2 - (cx1 + r1)
    expected_gap_lat = int(round(3.5 * pc.D_phys / pc.dx))
    assert abs(gap_lat - expected_gap_lat) <= 1


def test_make_tandem_physics_config_roi_includes_gap():
    """ROI x-start is upstream of the first cylinder (Option A)."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    # roi_x0 should be less than cx_lat (primary cylinder)
    assert lc.roi_x0 < lc.cx_lat


def test_make_tandem_physics_config_stable():
    """LatticeConfig.from_physics() succeeds without raising."""
    pc = make_tandem_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)   # must not raise
    assert lc.total_frames > 0
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py::test_make_tandem_physics_config_domain -v
```

Expected: `ImportError: cannot import name 'make_tandem_physics_config'`

- [ ] **Step 2.3: Implement make_tandem_physics_config()**

Add the following function to `scripts/generate_wake.py` immediately after `stable_dt_for_case()` (around L602):

```python
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
```

- [ ] **Step 2.4: Run tandem tests**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -k "tandem" -v
```

Expected: 6 PASSED.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/generate_wake.py tests/test_multi_cylinder_lbm.py
git commit -m "feat: add make_tandem_physics_config() for G/D=3.5 tandem cylinders"
```

---

### Task 3: Add make_side_by_side_physics_config() preset

**Files:**
- Modify: `scripts/generate_wake.py`
- Modify: `tests/test_multi_cylinder_lbm.py`

- [ ] **Step 3.1: Write failing tests for side-by-side preset**

Append to `tests/test_multi_cylinder_lbm.py`:

```python
from scripts.generate_wake import make_side_by_side_physics_config


def test_make_side_by_side_physics_config_domain():
    """Side-by-side domain is taller than single-cylinder."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    assert pc.Ly_phys == pytest.approx(240.0)


def test_make_side_by_side_physics_config_two_cylinders():
    """Side-by-side config produces exactly two cylinders."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    assert len(lc.cylinders_lat) == 2


def test_make_side_by_side_physics_config_same_x():
    """Both cylinders have the same x-position."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    cx1, _, _ = lc.cylinders_lat[0]
    cx2, _, _ = lc.cylinders_lat[1]
    assert cx1 == cx2


def test_make_side_by_side_physics_config_symmetric_y():
    """Two cylinders are symmetric about the domain y-midline."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    _, cy1, _ = lc.cylinders_lat[0]
    _, cy2, _ = lc.cylinders_lat[1]
    domain_cy = int(round(120.0 / pc.dx))  # domain centre at 120m
    assert abs((cy1 + cy2) // 2 - domain_cy) <= 1


def test_make_side_by_side_physics_config_roi_centered():
    """ROI y-bounds are centred between the two cylinders."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    roi_y_center = (lc.roi_y0 + lc.roi_y1) // 2
    expected = int(round(120.0 / pc.dx))  # roi_y_center_override=120m
    assert abs(roi_y_center - expected) <= 1


def test_make_side_by_side_physics_config_stable():
    """LatticeConfig.from_physics() succeeds without raising."""
    pc = make_side_by_side_physics_config(Re=150.0, U_phys=1.0)
    lc = LatticeConfig.from_physics(pc)
    assert lc.total_frames > 0
```

- [ ] **Step 3.2: Confirm failure**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -k "side_by_side" -v
```

Expected: `ImportError: cannot import name 'make_side_by_side_physics_config'`

- [ ] **Step 3.3: Implement make_side_by_side_physics_config()**

Add immediately after `make_tandem_physics_config()`:

```python
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
```

- [ ] **Step 3.4: Run side-by-side tests**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -k "side_by_side" -v
```

Expected: 6 PASSED.

- [ ] **Step 3.5: Run full test suite**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -v
```

Expected: all 19 tests PASSED.

- [ ] **Step 3.6: Commit**

```bash
git add scripts/generate_wake.py tests/test_multi_cylinder_lbm.py
git commit -m "feat: add make_side_by_side_physics_config() for G/D=3.5 parallel cylinders"
```

---

### Task 4: Update make_training_configs() + filename tagging + CLI

**Files:**
- Modify: `scripts/generate_wake.py` (make_training_configs ~L802, run_simulation ~L618, cli ~L913)
- Modify: `tests/test_multi_cylinder_lbm.py`

- [ ] **Step 4.1: Write failing tests**

Append to `tests/test_multi_cylinder_lbm.py`:

```python
from scripts.generate_wake import make_training_configs


def test_make_training_configs_tandem_returns_configs():
    """'tandem_G35_nav' profile returns non-empty list of PhysicsConfig."""
    configs = make_training_configs(profile="tandem_G35_nav")
    assert len(configs) > 0


def test_make_training_configs_tandem_each_has_two_cylinders():
    """Every config in tandem profile has exactly two cylinders."""
    for pc in make_training_configs(profile="tandem_G35_nav"):
        assert len(pc.extra_cylinders) == 1   # primary + 1 extra = 2 total


def test_make_training_configs_side_by_side_returns_configs():
    """'side_by_side_G35_nav' profile returns non-empty list."""
    configs = make_training_configs(profile="side_by_side_G35_nav")
    assert len(configs) > 0


def test_make_training_configs_unknown_profile_raises():
    """Unknown profile name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown training profile"):
        make_training_configs(profile="nonexistent")


def test_case_tag_in_filename_prefix():
    """case_tag is non-empty for multi-cylinder configs."""
    for pc in make_training_configs(profile="tandem_G35_nav"):
        assert pc.case_tag.startswith("tandem")
    for pc in make_training_configs(profile="side_by_side_G35_nav"):
        assert pc.case_tag.startswith("sbs")
```

- [ ] **Step 4.2: Confirm failure**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -k "training_configs" -v
```

Expected: FAILED — `ValueError: Unknown training profile: tandem_G35_nav`.

- [ ] **Step 4.3: Update make_training_configs() to include new profiles**

In `make_training_configs()` (around L819), the `profiles` dict currently has one entry `"navigation"`. Extend it:

```python
def make_training_configs(
    profile: str = "navigation",
    re_values: Optional[Sequence[float]] = None,
    u_values: Optional[Sequence[float]] = None,
) -> List[PhysicsConfig]:
    ...
    profiles = {
        "navigation": NavigationProfile(...),   # unchanged
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
    ...  # rest unchanged
```

- [ ] **Step 4.4: Update run_simulation() filename to prepend case_tag**

In `run_simulation()` (around L651), find:
```python
    tag = (
        f"v8_U{_tag_float(pc.U_phys)}_"
        ...
    )
```

Replace with:
```python
    tag = (
        f"{pc.case_tag}"          # empty string for single-cylinder cases
        f"v8_U{_tag_float(pc.U_phys)}_"
        f"Re{pc.Re:.0f}_"
        f"D{_tag_float(pc.D_phys)}_"
        f"dx{_tag_float(lc.dx_out)}_"
        f"Ti{pc.turbulence_intensity * 100:.0f}pct_"
        f"{lc.total_frames}f_roi"
    )
```

- [ ] **Step 4.5: Update CLI --profile choices**

In `cli()` (around L914), update the `--profile` argument help text (no hard-coded choices needed since validation is done in `make_training_configs`):

```python
    parser.add_argument(
        "--profile",
        default="navigation",
        help=(
            "Training profile name. Available: navigation, "
            "tandem_G35_nav, side_by_side_G35_nav (default: navigation)"
        ),
    )
```

- [ ] **Step 4.6: Run all tests**

```bash
python -m pytest tests/test_multi_cylinder_lbm.py -v
```

Expected: all 24 tests PASSED.

- [ ] **Step 4.7: Smoke-test config creation (no simulation)**

```bash
python -c "
from scripts.generate_wake import make_training_configs, estimate_dataset_storage_bytes
configs = make_training_configs('tandem_G35_nav')
print(f'Tandem: {len(configs)} cases, ~{estimate_dataset_storage_bytes(configs)/1024**3:.2f} GiB')
configs = make_training_configs('side_by_side_G35_nav')
print(f'Side-by-side: {len(configs)} cases, ~{estimate_dataset_storage_bytes(configs)/1024**3:.2f} GiB')
"
```

Expected (approximate):
```
Tandem: 6 cases, ~X.XX GiB
Side-by-side: 6 cases, ~X.XX GiB
```
No errors.

- [ ] **Step 4.8: Commit**

```bash
git add scripts/generate_wake.py tests/test_multi_cylinder_lbm.py
git commit -m "feat: add tandem_G35_nav and side_by_side_G35_nav training profiles"
```

---

## How to Run the New Simulations

```bash
# Tandem dual-cylinder (串联)
python -m scripts.generate_wake --profile tandem_G35_nav

# Side-by-side dual-cylinder (并排)
python -m scripts.generate_wake --profile side_by_side_G35_nav

# With GPU acceleration
python -m scripts.generate_wake --profile tandem_G35_nav --gpu

# Single Re/U override
python -m scripts.generate_wake --profile tandem_G35_nav --re 150 --u 1.0
```
