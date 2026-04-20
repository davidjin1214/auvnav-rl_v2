from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_offline_td3bc_phase0b_v2 import analyze_phase0b_v2


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trainer_state(*, alpha: float, episodes: int, total_steps: int) -> dict:
    return {
        "train_step": total_steps,
        "train_config": {"total_steps": total_steps},
        "agent_config": {"alpha": alpha},
        "offline_metadata": {
            "policy": "crosscomp",
            "num_episodes": episodes,
            "num_transitions": episodes * 100,
            "success_rate": 0.85,
        },
    }


def _eval_payload(success: float, ret: float) -> dict:
    return {
        "eval_success_rate": success,
        "eval_return": ret,
    }


def test_analyze_phase0b_v2_reports_expected_diagnostics(tmp_path: Path) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    results_root = tmp_path / "results"

    for dataset_name, episodes, best_alpha, max_alpha, bc_val, best_val, td3bc_test, bc_test, chosen_step in [
        ("dataset_ep500", 500, 0.1, 0.5, 0.60, 0.66, 0.64, 0.58, 700),
        ("dataset_ep1000", 1000, 0.5, 0.5, 0.52, 0.61, 0.55, 0.48, 920),
    ]:
        best_alpha_tag = f"alpha_{str(best_alpha).replace('.', 'p')}"
        bc_alpha_tag = "alpha_0p0"
        best_seed = "seed_42"

        _write_json(
            checkpoints_root / dataset_name / best_alpha_tag / best_seed / "trainer_state.json",
            _trainer_state(alpha=best_alpha, episodes=episodes, total_steps=1000),
        )
        _write_json(
            checkpoints_root / dataset_name / bc_alpha_tag / best_seed / "trainer_state.json",
            _trainer_state(alpha=0.0, episodes=episodes, total_steps=1000),
        )
        _write_json(
            results_root / dataset_name / "selection" / best_alpha_tag / best_seed / "selected_checkpoint.json",
            {"best": {"agent_file": "agent_step_000700.pt", "train_step": chosen_step}},
        )
        _write_json(
            results_root / dataset_name / "selection" / bc_alpha_tag / best_seed / "selected_checkpoint.json",
            {"best": {"agent_file": "agent_step_000650.pt", "train_step": 650}},
        )
        _write_json(
            results_root / dataset_name / "selection" / "best_alpha.json",
            {
                "best_alpha_tag": best_alpha_tag,
                "best_alpha": best_alpha,
                "per_alpha": [
                    {
                        "alpha_tag": bc_alpha_tag,
                        "alpha": 0.0,
                        "num_seeds": 1,
                        "mean_val_success_rate": bc_val,
                        "std_val_success_rate": 0.0,
                        "mean_val_return": -10.0,
                        "mean_val_safety_cost": 1.0,
                        "mean_val_time_s": 80.0,
                    },
                    {
                        "alpha_tag": best_alpha_tag,
                        "alpha": best_alpha,
                        "num_seeds": 1,
                        "mean_val_success_rate": best_val,
                        "std_val_success_rate": 0.0,
                        "mean_val_return": -8.0,
                        "mean_val_safety_cost": 0.8,
                        "mean_val_time_s": 78.0,
                    },
                    {
                        "alpha_tag": "alpha_0p5",
                        "alpha": max_alpha,
                        "num_seeds": 1,
                        "mean_val_success_rate": best_val,
                        "std_val_success_rate": 0.0,
                        "mean_val_return": -8.0,
                        "mean_val_safety_cost": 0.8,
                        "mean_val_time_s": 78.0,
                    },
                ],
                "selected_runs": [
                    {
                        "seed": best_seed,
                        "agent_file": "agent_step_000700.pt",
                        "train_step": chosen_step,
                        "val_success_rate": best_val,
                    }
                ],
            },
        )
        _write_json(
            results_root / dataset_name / "test_selected" / best_alpha_tag / f"{best_seed}.json",
            _eval_payload(td3bc_test, -50.0),
        )
        _write_json(
            results_root / dataset_name / "test_bc_selected" / bc_alpha_tag / f"{best_seed}.json",
            _eval_payload(bc_test, -55.0),
        )
        _write_json(
            results_root / dataset_name / "baselines" / "crosscomp_test_eval.json",
            _eval_payload(0.90 if episodes == 500 else 0.92, -40.0),
        )

    _write_json(
        results_root / "dataset_ep1000" / "validation" / "alpha_0p0" / "seed_42" / "agent_step_000500.json",
        _eval_payload(0.45, -60.0),
    )
    _write_json(
        results_root / "dataset_ep1000" / "validation" / "alpha_0p5" / "seed_42" / "agent_step_000900.json",
        _eval_payload(0.61, -48.0),
    )

    analysis = analyze_phase0b_v2(
        results_root=results_root,
        checkpoints_root=checkpoints_root,
    )

    assert analysis["num_datasets"] == 2
    assert analysis["largest_dataset"] == "dataset_ep1000"
    assert len(analysis["largest_dataset_validation_curve"]) == 2

    rows_by_episode = {
        int(row["episodes"]): row for row in analysis["dataset_diagnostics"]
    }
    assert rows_by_episode[500]["best_alpha_hits_boundary"] is False
    assert rows_by_episode[1000]["best_alpha_hits_boundary"] is True
    assert abs(float(rows_by_episode[1000]["mean_selected_ckpt_frac"]) - 0.92) < 1e-9

    compare_by_episode = {
        int(row["episodes"]): row for row in analysis["td3bc_vs_bc_test"]
    }
    assert abs(float(compare_by_episode[500]["td3bc_minus_bc_success"]) - 0.06) < 1e-9
    assert abs(float(compare_by_episode[1000]["td3bc_minus_bc_success"]) - 0.07) < 1e-9

    conclusions = "\n".join(analysis["conclusions"])
    assert "BC validation success decreases monotonically" in conclusions
    assert "BC test success also decreases monotonically" in conclusions
    assert "hits the current sweep boundary" in conclusions
    assert "Increase TRAIN_EPOCHS" in conclusions
    assert "TD3BC consistently improves over BC" in conclusions
