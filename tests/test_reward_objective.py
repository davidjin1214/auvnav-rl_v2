from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auv_nav.reward import (
    SAFETY_TERMINAL_VIOLATION_REASONS,
    RewardModel,
    RewardModelConfig,
    reward_objective_config,
)
from scripts.train_sac import replay_done_for_objective
from scripts.train_utils import make_env_config_overrides, make_planar_env

CONTROL_DT = 0.5
MAX_EPISODE_TIME_S = 240.0


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


def test_make_planar_env_adds_arrival_v2_episode_context_observation() -> None:
    overrides = make_env_config_overrides(
        SimpleNamespace(
            objective="arrival_v2",
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
    obs, _ = env.reset(seed=0)
    decoded = env.unwrapped.decode_observation(obs)

    assert env.unwrapped.config.reward_objective == "arrival_v2"
    assert env.unwrapped.config.include_episode_context_obs is True
    assert env.observation_space.shape == (12,)
    assert "episode_context" in decoded
    assert decoded["episode_context"].shape == (2,)
    assert decoded["episode_context"][0] == 0.0


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
    model = RewardModel(
        RewardModelConfig(
            control_dt=CONTROL_DT,
            reward_objective="arrival_v2_simple",
            **cfg,
        )
    )

    def episode_return(steps: int, total_progress: float, terminal_reason: str) -> float:
        total = 0.0
        step_progress = total_progress / steps
        for idx in range(steps):
            reason = terminal_reason if idx == steps - 1 else "running"
            total += model.compute(
                progress=step_progress,
                safety_cost=0.0,
                reason=reason,
                terminated=reason not in {"running", "timeout"},
                truncated=reason == "timeout",
                actuator_rpm=0.0,
            ).reward
        return total

    # Synthetic episodes representative of upstream u10 dynamics (D_init ~ 10 m, max 480 steps).
    fast_success = episode_return(steps=200, total_progress=10.0, terminal_reason="goal")
    slow_success = episode_return(steps=480, total_progress=10.0, terminal_reason="goal")
    timeout_near = episode_return(steps=480, total_progress=8.0, terminal_reason="timeout")
    timeout_far = episode_return(steps=480, total_progress=2.0, terminal_reason="timeout")
    slow_oob = episode_return(steps=480, total_progress=2.0, terminal_reason="out_of_bounds")

    assert fast_success > slow_success, (fast_success, slow_success)
    assert slow_success > timeout_near, (slow_success, timeout_near)
    assert timeout_near > timeout_far, (timeout_near, timeout_far)
    assert timeout_far > slow_oob, (timeout_far, slow_oob)


@dataclass(frozen=True)
class EpisodeFixture:
    name: str
    n_control_steps: int
    reason: str
    initial_distance_m: float
    final_distance_m: float
    avg_per_step_safety: float


FIXTURES = [
    EpisodeFixture("fast_success", 120, "goal", 65.0, 4.0, 0.005),
    EpisodeFixture("slow_success", 240, "goal", 65.0, 4.0, 0.005),
    EpisodeFixture("unsafe_success", 180, "goal", 65.0, 4.0, 0.150),
    EpisodeFixture("timeout_near", 480, "timeout", 65.0, 13.0, 0.130),
    EpisodeFixture("timeout_far", 480, "timeout", 65.0, 65.0, 0.130),
    EpisodeFixture("late_oob", 240, "out_of_bounds", 65.0, 65.0, 0.150),
    EpisodeFixture("mid_oob", 120, "out_of_bounds", 65.0, 97.5, 0.180),
    EpisodeFixture("fast_oob", 60, "out_of_bounds", 65.0, 78.0, 0.185),
]


def _make_arrival_v2_reward_model(
    objective: str = "arrival_v2",
    *,
    w_safety_override: float | None = None,
) -> RewardModel:
    cfg = reward_objective_config(objective)
    if w_safety_override is not None:
        cfg["w_safety"] = w_safety_override
    return RewardModel(
        RewardModelConfig(
            control_dt=CONTROL_DT,
            max_episode_time_s=MAX_EPISODE_TIME_S,
            reward_objective=objective,
            **cfg,
        )
    )


def _compute_arrival_v2_return(
    fixture: EpisodeFixture,
    *,
    objective: str = "arrival_v2",
    gamma: float | None = None,
    w_safety_override: float | None = None,
) -> float:
    reward_model = _make_arrival_v2_reward_model(
        objective,
        w_safety_override=w_safety_override,
    )
    distances = np.linspace(
        fixture.initial_distance_m,
        fixture.final_distance_m,
        fixture.n_control_steps + 1,
    )
    total = 0.0
    discount = 1.0
    for step_idx in range(fixture.n_control_steps):
        previous_distance = float(distances[step_idx])
        current_distance = float(distances[step_idx + 1])
        reason = fixture.reason if step_idx == fixture.n_control_steps - 1 else "running"
        terminated = reason not in {"running", "timeout"}
        truncated = reason == "timeout"
        reward = reward_model.compute(
            progress=previous_distance - current_distance,
            previous_distance_to_goal_m=previous_distance,
            current_distance_to_goal_m=current_distance,
            initial_distance_to_goal_m=fixture.initial_distance_m,
            elapsed_time_s=(step_idx + 1) * CONTROL_DT,
            max_episode_time_s=MAX_EPISODE_TIME_S,
            dt=CONTROL_DT,
            safety_cost=fixture.avg_per_step_safety,
            reason=reason,
            terminated=terminated,
            truncated=truncated,
            actuator_rpm=0.0,
        ).reward
        total += discount * reward
        if gamma is not None:
            discount *= gamma
    return float(total)


def _fixture_returns(
    *,
    gamma: float | None = None,
    w_safety_override: float | None = None,
) -> dict[str, float]:
    return {
        fixture.name: _compute_arrival_v2_return(
            fixture,
            gamma=gamma,
            w_safety_override=w_safety_override,
        )
        for fixture in FIXTURES
    }


def test_arrival_v2_field_lockdown() -> None:
    cfg = reward_objective_config("arrival_v2")
    assert cfg["w_progress"] == 50.0
    assert cfg["w_time"] == 5.0
    assert cfg["w_safety"] == 2.0
    assert cfg["r_success"] == 100.0
    assert cfg["r_fast_success"] == 0.0
    assert cfg["r_failure"] == 100.0
    assert cfg["r_early_failure"] == 100.0
    assert cfg["r_timeout"] == 50.0
    assert cfg["r_final_distance"] == 50.0
    assert cfg["d_init_min_m"] == 10.0


def test_arrival_v2_undiscounted_terminal_dominance() -> None:
    returns = _fixture_returns()
    expected_order = [
        "fast_success",
        "slow_success",
        "unsafe_success",
        "timeout_near",
        "timeout_far",
        "late_oob",
        "fast_oob",
        "mid_oob",
    ]
    assert sorted(returns, key=returns.get, reverse=True) == expected_order


def test_arrival_v2_discounted_terminal_dominance_gamma_0995() -> None:
    returns = _fixture_returns(gamma=0.995)
    success_min = min(v for k, v in returns.items() if "success" in k)
    timeout_max = max(v for k, v in returns.items() if "timeout" in k)
    hard_failure_max = max(v for k, v in returns.items() if "oob" in k)

    assert success_min > timeout_max + 50.0, returns
    assert returns["timeout_far"] > returns["late_oob"], returns
    assert returns["fast_oob"] < timeout_max - 100.0, returns
    assert timeout_max > hard_failure_max, returns
    assert returns["slow_success"] > returns["unsafe_success"] + 10.0, returns


def test_arrival_v2_blocks_discounted_unsafe_shortcut() -> None:
    safe_route = EpisodeFixture("safe_succ", 180, "goal", 65.0, 4.0, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, "goal", 65.0, 4.0, 0.150)
    for gamma in [None, 0.995]:
        safe_return = _compute_arrival_v2_return(safe_route, gamma=gamma)
        risky_return = _compute_arrival_v2_return(risky_route, gamma=gamma)
        assert safe_return > risky_return, (gamma, safe_return, risky_return)


def test_arrival_v2_v5_weak_safety_would_fail_discounted_shortcut_gate() -> None:
    safe_route = EpisodeFixture("safe_succ", 180, "goal", 65.0, 4.0, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, "goal", 65.0, 4.0, 0.150)
    safe_return = _compute_arrival_v2_return(
        safe_route,
        gamma=0.995,
        w_safety_override=0.5,
    )
    risky_return = _compute_arrival_v2_return(
        risky_route,
        gamma=0.995,
        w_safety_override=0.5,
    )
    assert safe_return < risky_return, (safe_return, risky_return)


def test_arrival_v2_w_safety_stress_preserves_core_dominance() -> None:
    for w_safety in [1.5, 2.0, 2.5, 3.0]:
        undiscounted = _fixture_returns(w_safety_override=w_safety)
        discounted = _fixture_returns(gamma=0.995, w_safety_override=w_safety)
        timeout_max = max(v for k, v in discounted.items() if "timeout" in k)
        hard_failure_max = max(v for k, v in discounted.items() if "oob" in k)

        assert undiscounted["timeout_far"] > undiscounted["late_oob"], w_safety
        assert timeout_max > hard_failure_max, w_safety
        assert discounted["fast_oob"] < timeout_max - 100.0, w_safety


def test_arrival_v2_d_init_clamp_prevents_reward_blowup() -> None:
    edge = EpisodeFixture("near_goal_start", 20, "goal", 4.0, 0.5, 0.001)
    reward = _compute_arrival_v2_return(edge)
    assert -1e3 < reward < 1e3


def test_arrival_v2_timeout_is_not_safety_terminal_violation() -> None:
    assert "timeout" not in SAFETY_TERMINAL_VIOLATION_REASONS
    assert "out_of_bounds" in SAFETY_TERMINAL_VIOLATION_REASONS


def test_arrival_v2_timeout_replay_done_semantics() -> None:
    assert replay_done_for_objective("arrival_v2", terminated=False, truncated=True)
    assert replay_done_for_objective("arrival_v2_fast", terminated=False, truncated=True)
    assert not replay_done_for_objective("efficiency_v2", terminated=False, truncated=True)
    assert replay_done_for_objective("efficiency_v2", terminated=True, truncated=False)
