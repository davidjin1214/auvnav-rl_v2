from __future__ import annotations

from scripts.evaluate_best_checkpoint import best_eval_row
from auv_nav.reward import reward_objective_config


def test_best_eval_row_prioritizes_success_then_return() -> None:
    rows = [
        {
            "env_step": "30000",
            "eval_success_rate": "0.0",
            "eval_return": "-100.0",
            "eval_safety_cost": "10.0",
            "eval_time_s": "50.0",
        },
        {
            "env_step": "60000",
            "eval_success_rate": "0.1",
            "eval_return": "-300.0",
            "eval_safety_cost": "30.0",
            "eval_time_s": "150.0",
        },
        {
            "env_step": "90000",
            "eval_success_rate": "0.1",
            "eval_return": "-200.0",
            "eval_safety_cost": "20.0",
            "eval_time_s": "120.0",
        },
    ]

    best = best_eval_row(rows)
    assert best["env_step"] == "90000"


def test_efficiency_v2_reward_preset_matches_gain_sweep_choice() -> None:
    cfg = reward_objective_config("efficiency_v2")
    assert cfg["energy_cost_gain"] == 0.0
    assert cfg["safety_cost_gain"] == 0.25
