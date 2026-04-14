from __future__ import annotations

from pathlib import Path

from scripts.summarize_suite import infer_runs_from_layout


def test_infer_runs_supports_objective_dimension(tmp_path: Path) -> None:
    run_dir = tmp_path / "single_u15_upstream_tgt15" / "efficiency_v1" / "sac_stack4" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "trainer_state.json").write_text("{}", encoding="utf-8")

    runs = infer_runs_from_layout(tmp_path)
    assert len(runs) == 1
    assert runs[0]["benchmark"] == "single_u15_upstream_tgt15"
    assert runs[0]["objective"] == "efficiency_v1"
    assert runs[0]["method"] == "sac_stack4"
    assert runs[0]["seed"] == 42


def test_infer_runs_supports_gain_dimension(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "single_u15_upstream_tgt15"
        / "efficiency_v1"
        / "e0_s0p5"
        / "sac_stack4"
        / "seed_42"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "trainer_state.json").write_text("{}", encoding="utf-8")

    runs = infer_runs_from_layout(tmp_path)
    assert len(runs) == 1
    assert runs[0]["benchmark"] == "single_u15_upstream_tgt15"
    assert runs[0]["objective"] == "efficiency_v1"
    assert runs[0]["gain_label"] == "e0_s0p5"
    assert runs[0]["method"] == "sac_stack4"
    assert runs[0]["seed"] == 42
