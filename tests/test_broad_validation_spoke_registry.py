"""Unit tests for the broad-validation spoke registry."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.broad_validation_spoke_registry import REGISTRY, SpokeConfig, get_spoke


def test_all_eight_spokes_present() -> None:
    expected = {"A1", "A2", "A2-td3bc", "A3", "B1", "B2", "C1", "C3"}
    assert set(REGISTRY.keys()) == expected


def test_anchor_invariants() -> None:
    """Spec §3.5: spokes change exactly one axis vs anchor."""
    anchor = SpokeConfig(
        spoke_id="anchor",
        dataset_name="crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="crosscomp",
        policy_mixture=None,
        probe_layout="s0",
        task_geometry="cross_stream",
        flow_path="wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
        benchmark_key="single_u10_cross_tgt15",
        target_speed=1.5,
        algo="rebrac",
        actor_penalty_coef=4.0,
        critic_penalty_coef=2.0,
        td3bc_alpha=None,
    )
    a_axis = {"A1", "A2", "A3"}
    b_axis = {"B1", "B2"}
    c_axis = {"C1", "C3"}
    for sid in a_axis:
        cfg = get_spoke(sid)
        assert cfg.probe_layout == anchor.probe_layout
        assert cfg.task_geometry == anchor.task_geometry
        assert cfg.flow_path == anchor.flow_path
    for sid in b_axis:
        cfg = get_spoke(sid)
        assert cfg.collector_policy == anchor.collector_policy
        assert cfg.task_geometry == anchor.task_geometry
        assert cfg.flow_path == anchor.flow_path
    for sid in c_axis:
        cfg = get_spoke(sid)
        assert cfg.collector_policy == anchor.collector_policy
        assert cfg.probe_layout == anchor.probe_layout


def test_a2_uses_episode_level_mixture() -> None:
    cfg = get_spoke("A2")
    assert cfg.policy_mixture == "goalseek:1.0,crosscomp:1.0"


def test_a2_td3bc_shares_dataset_with_a2() -> None:
    assert get_spoke("A2").dataset_name == get_spoke("A2-td3bc").dataset_name


def test_a2_td3bc_alpha_matches_phase0c_winner() -> None:
    cfg = get_spoke("A2-td3bc")
    assert cfg.algo == "td3bc"
    assert cfg.td3bc_alpha == 0.25


def test_b_axis_obs_dim_consistent_with_probe() -> None:
    assert get_spoke("B1").probe_layout == "s1"
    assert get_spoke("B2").probe_layout == "s2"


def test_c_axis_flow_paths_exist_in_repo() -> None:
    """C3 must use the tandem wake; C1 must keep single Re150 wake."""
    assert "tandem" in get_spoke("C3").flow_path
    assert "tandem" not in get_spoke("C1").flow_path
    assert get_spoke("C1").task_geometry == "upstream"
    assert get_spoke("C1").benchmark_key == "single_u10_upstream_tgt15"


def test_cli_get_field() -> None:
    """Bash driver must be able to query fields via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.broad_validation_spoke_registry",
         "--get-field", "A1", "dataset_name"],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == (
        "goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"
    )


def test_cli_get_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.broad_validation_spoke_registry",
         "--get-json", "A1"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["spoke_id"] == "A1"
    assert payload["collector_policy"] == "goalseek"
