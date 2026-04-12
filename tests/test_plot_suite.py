from __future__ import annotations

from scripts.plot_suite import infer_plot_mode, variant_key


def test_infer_plot_mode_detects_objective_ablation() -> None:
    summary_rows = [
        {"benchmark": "single_u15_upstream_tgt15", "objective": "arrival_v1", "method": "sac_stack4"},
        {"benchmark": "single_u15_upstream_tgt15", "objective": "efficiency_v1", "method": "sac_stack4"},
    ]
    assert infer_plot_mode(summary_rows) == "objective"


def test_variant_key_uses_objective_in_objective_mode() -> None:
    row = {"benchmark": "single_u15_upstream_tgt15", "objective": "efficiency_v1", "method": "sac_stack4"}
    assert variant_key(row, mode="objective") == "efficiency_v1"
    assert variant_key(row, mode="method") == "efficiency_v1::sac_stack4"
