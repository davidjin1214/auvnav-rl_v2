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
