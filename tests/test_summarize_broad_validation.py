"""Unit tests for the broad-validation summary + trigger gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_broad_validation import (
    ANCHOR_MEAN,
    ANCHOR_STD,
    apply_trigger_gate,
    collect_p1_results,
    summarize_p1,
)


def _write_test_json(path: Path, success: float, rtn: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "eval_success_rate": success,
        "eval_return": rtn,
        "eval_safety_cost": 0.0,
        "eval_time_s": 1.0,
    }), encoding="utf-8")


def test_anchor_constants_match_spec() -> None:
    assert ANCHOR_MEAN == pytest.approx(0.902)
    assert ANCHOR_STD == pytest.approx(0.021)


def test_collect_p1_walks_directory(tmp_path: Path) -> None:
    base = tmp_path / "results" / "offline" / "rebrac" / "broad_validation"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.6)
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.95)
    rows = collect_p1_results(base)
    spokes = {r["spoke_id"] for r in rows}
    assert spokes == {"A1", "B2"}


def test_summarize_aggregates_seeds(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.7)
    rows = summarize_p1(base)
    target = next(r for r in rows if r["spoke_id"] == "A1")
    assert target["num_seeds"] == 2
    assert target["mean_test_success_rate"] == pytest.approx(0.6)
    assert target["std_test_success_rate"] > 0


def test_trigger_mean_shift_negative(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.5)
    decisions = apply_trigger_gate(summarize_p1(base))
    a1 = next(d for d in decisions if d["spoke_id"] == "A1")
    assert a1["triggered"] is True
    assert "mean_shift" in a1["reasons"]
    assert a1["delta_pp"] < 0


def test_trigger_mean_shift_positive(tmp_path: Path) -> None:
    """Spec §6.1: positive 5pp shift also triggers (bidirectional)."""
    base = tmp_path / "broad"
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.97)
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.98)
    decisions = apply_trigger_gate(summarize_p1(base))
    b2 = next(d for d in decisions if d["spoke_id"] == "B2")
    assert b2["triggered"] is True


def test_trigger_a2_rebrac_vs_td3bc_gap_collapse(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.85)
    _write_test_json(base / "A2" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.85)
    _write_test_json(base / "A2-td3bc" / "alpha_0p25" / "test" / "seed_42.json", 0.83)
    _write_test_json(base / "A2-td3bc" / "alpha_0p25" / "test" / "seed_44.json", 0.83)
    decisions = apply_trigger_gate(summarize_p1(base))
    a2 = next(d for d in decisions if d["spoke_id"] == "A2")
    assert a2["triggered"] is True
    assert "td3bc_gap_collapse" in a2["reasons"]


def test_no_trigger_within_tolerance(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "B1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.91)
    _write_test_json(base / "B1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.92)
    decisions = apply_trigger_gate(summarize_p1(base))
    b1 = next(d for d in decisions if d["spoke_id"] == "B1")
    assert b1["triggered"] is False
