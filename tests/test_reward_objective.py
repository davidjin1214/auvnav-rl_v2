from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from auv_nav.reward import reward_objective_config
from scripts.train_utils import make_env_config_overrides, make_planar_env


def _flow_path() -> Path:
    root = Path(__file__).parent.parent
    return root / "wake_data" / "wake_dummy_roi.npy"


def test_reward_objective_config_efficiency_has_nonzero_cost_gains() -> None:
    config = reward_objective_config("efficiency_v1")
    assert config["energy_cost_gain"] > 0.0
    assert config["safety_cost_gain"] > 0.0


def test_make_env_config_overrides_applies_objective_and_cli_overrides() -> None:
    args = SimpleNamespace(
        objective="efficiency_v1",
        step_penalty=None,
        time_penalty_per_second=None,
        reward_progress_gain=None,
        success_reward=None,
        failure_penalty=None,
        timeout_penalty=None,
        energy_cost_gain=0.001,
        safety_cost_gain=None,
    )
    overrides = make_env_config_overrides(args)
    assert overrides["reward_objective"] == "efficiency_v1"
    assert overrides["energy_cost_gain"] == 0.001
    assert overrides["safety_cost_gain"] == reward_objective_config("efficiency_v1")["safety_cost_gain"]


def test_make_planar_env_applies_reward_objective() -> None:
    overrides = make_env_config_overrides(
        SimpleNamespace(
            objective="efficiency_v1",
            step_penalty=None,
            time_penalty_per_second=None,
            reward_progress_gain=None,
            success_reward=None,
            failure_penalty=None,
            timeout_penalty=None,
            energy_cost_gain=None,
            safety_cost_gain=None,
        )
    )
    env = make_planar_env(
        _flow_path(),
        history_length=1,
        probe_layout="s0",
        env_config_overrides=overrides,
    )
    assert env.unwrapped.config.reward_objective == "efficiency_v1"
    assert env.unwrapped.config.energy_cost_gain > 0.0
    assert env.unwrapped.config.safety_cost_gain > 0.0
