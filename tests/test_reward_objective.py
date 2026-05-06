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


def test_arrival_v2_simple_field_lockdown() -> None:
    """Lock arrival_v2_simple values to prevent silent drift.

    Adding any new fields (early-failure-penalty / normalized-progress / etc.)
    requires a deliberate update to this assertion plus the dominance test below.
    """
    cfg = reward_objective_config("arrival_v2_simple")
    assert cfg == {
        "step_penalty": -0.2,
        "time_penalty_per_second": None,
        "reward_progress_gain": 1.0,
        "success_reward": 200.0,
        "failure_penalty": -200.0,
        "timeout_penalty": -50.0,
        "energy_cost_gain": 0.0,
        "safety_cost_gain": 0.5,
    }


def test_arrival_v2_simple_terminal_dominance() -> None:
    """Required ordering for arrival-first offline ReBRAC training:
    fast_success > slow_success > timeout_near > timeout_far > slow_OOB.

    Note: fast_OOB > slow_OOB inversion remains (no early-failure-penalty in
    arrival_v2_simple); documented limitation of this minimal preset, not asserted.
    See docs/online_sac_reward_redesign.md §4.4 for the full arrival_v2 ordering.
    """
    cfg = reward_objective_config("arrival_v2_simple")
    sp = cfg["step_penalty"]
    pg = cfg["reward_progress_gain"]
    sr = cfg["success_reward"]
    tp = cfg["timeout_penalty"]
    fp = cfg["failure_penalty"]

    def episode_return(steps: int, total_progress: float, terminal: float) -> float:
        # task_reward per step = step_penalty (time_penalty_per_second auto-derived to cancel dt)
        return sp * steps + pg * total_progress + terminal

    # Synthetic episodes representative of upstream u10 dynamics (D_init ~ 10 m, max 480 steps).
    fast_success = episode_return(steps=200, total_progress=10.0, terminal=sr)
    slow_success = episode_return(steps=480, total_progress=10.0, terminal=sr)
    timeout_near = episode_return(steps=480, total_progress=8.0, terminal=tp)
    timeout_far = episode_return(steps=480, total_progress=2.0, terminal=tp)
    slow_oob = episode_return(steps=480, total_progress=2.0, terminal=fp)

    assert fast_success > slow_success, (fast_success, slow_success)
    assert slow_success > timeout_near, (slow_success, timeout_near)
    assert timeout_near > timeout_far, (timeout_near, timeout_far)
    assert timeout_far > slow_oob, (timeout_far, slow_oob)
