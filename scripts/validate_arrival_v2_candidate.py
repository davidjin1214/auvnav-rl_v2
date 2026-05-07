"""Independent pre-integration validator for the arrival_v2 reward candidate.

This script intentionally does not import auv_nav.reward. It is a design gate:
the candidate reward must pass these formula-level checks before the production
RewardModel/env/training paths are changed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace


CONTROL_DT = 0.5
MAX_EPISODE_TIME_S = 240.0
DEFAULT_GAMMA = 0.995


@dataclass(frozen=True, slots=True)
class RewardParams:
    w_progress: float = 50.0
    w_time: float = 5.0
    w_safety: float = 2.0
    r_success: float = 100.0
    r_fast_success: float = 0.0
    r_failure: float = 100.0
    r_early_failure: float = 100.0
    r_timeout: float = 50.0
    r_final_distance: float = 50.0
    d_init_min_m: float = 10.0


@dataclass(frozen=True, slots=True)
class EpisodeFixture:
    name: str
    n_control_steps: int
    reason: str
    initial_distance_m: float
    final_distance_m: float
    avg_safety: float


FIXTURES = (
    EpisodeFixture("fast_success", 120, "goal", 65.0, 4.0, 0.005),
    EpisodeFixture("slow_success", 240, "goal", 65.0, 4.0, 0.005),
    EpisodeFixture("unsafe_success", 180, "goal", 65.0, 4.0, 0.150),
    EpisodeFixture("timeout_near", 480, "timeout", 65.0, 13.0, 0.130),
    EpisodeFixture("timeout_far", 480, "timeout", 65.0, 65.0, 0.130),
    EpisodeFixture("late_oob", 240, "out_of_bounds", 65.0, 65.0, 0.150),
    EpisodeFixture("fast_oob", 60, "out_of_bounds", 65.0, 78.0, 0.185),
    EpisodeFixture("mid_oob", 120, "out_of_bounds", 65.0, 97.5, 0.180),
)

SAFE_SHORTCUT_REFERENCE = EpisodeFixture("safe_success", 180, "goal", 65.0, 4.0, 0.005)
RISKY_SHORTCUT_REFERENCE = EpisodeFixture(
    "risky_shortcut_success", 120, "goal", 65.0, 4.0, 0.150
)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _distance_at(fixture: EpisodeFixture, step_idx: int) -> float:
    frac = step_idx / fixture.n_control_steps
    return fixture.initial_distance_m + (
        fixture.final_distance_m - fixture.initial_distance_m
    ) * frac


def arrival_v2_step_reward(
    *,
    params: RewardParams,
    previous_distance_m: float,
    current_distance_m: float,
    initial_distance_m: float,
    elapsed_time_s: float,
    avg_safety: float,
    terminal_reason: str,
) -> float:
    d_init_safe = max(initial_distance_m, params.d_init_min_m)
    elapsed_frac = _clip(elapsed_time_s / MAX_EPISODE_TIME_S, 0.0, 1.0)
    distance_ratio = _clip(current_distance_m / d_init_safe, 0.0, 2.0)

    reward = params.w_progress * ((previous_distance_m - current_distance_m) / d_init_safe)
    reward -= params.w_time * (CONTROL_DT / MAX_EPISODE_TIME_S)
    reward -= params.w_safety * avg_safety

    if terminal_reason == "goal":
        reward += params.r_success
        reward += params.r_fast_success * (1.0 - elapsed_frac)
    elif terminal_reason == "timeout":
        reward -= params.r_timeout
        reward -= params.r_final_distance * distance_ratio
    elif terminal_reason != "running":
        reward -= params.r_failure
        reward -= params.r_early_failure * (1.0 - elapsed_frac)
        reward -= params.r_final_distance * distance_ratio

    return float(reward)


def arrival_v2_return(
    fixture: EpisodeFixture,
    *,
    params: RewardParams,
    gamma: float | None = None,
) -> float:
    total = 0.0
    discount = 1.0
    for step_idx in range(fixture.n_control_steps):
        previous_distance = _distance_at(fixture, step_idx)
        current_distance = _distance_at(fixture, step_idx + 1)
        terminal_reason = (
            fixture.reason if step_idx == fixture.n_control_steps - 1 else "running"
        )
        reward = arrival_v2_step_reward(
            params=params,
            previous_distance_m=previous_distance,
            current_distance_m=current_distance,
            initial_distance_m=fixture.initial_distance_m,
            elapsed_time_s=(step_idx + 1) * CONTROL_DT,
            avg_safety=fixture.avg_safety,
            terminal_reason=terminal_reason,
        )
        total += discount * reward
        if gamma is not None:
            discount *= gamma
    return float(total)


def efficiency_v2_summary_return(fixture: EpisodeFixture) -> float:
    """Approximate current efficiency_v2 return on the same episode summaries."""
    progress_m = fixture.initial_distance_m - fixture.final_distance_m
    terminal = 100.0 if fixture.reason == "goal" else -20.0
    return (
        -1.0 * fixture.n_control_steps
        + progress_m
        + terminal
        - 0.25 * fixture.avg_safety * fixture.n_control_steps
    )


def _returns(
    *,
    params: RewardParams,
    gamma: float | None = None,
) -> dict[str, float]:
    return {
        fixture.name: arrival_v2_return(fixture, params=params, gamma=gamma)
        for fixture in FIXTURES
    }


def _assert_order(values: dict[str, float], expected_order: list[str]) -> None:
    actual_order = sorted(values, key=values.get, reverse=True)
    if actual_order != expected_order:
        raise AssertionError(f"order mismatch: actual={actual_order}, values={values}")


def _assert_greater(lhs: float, rhs: float, *, label: str, margin: float = 0.0) -> None:
    if not lhs > rhs + margin:
        raise AssertionError(f"{label}: expected {lhs:.3f} > {rhs:.3f} + {margin:.3f}")


def _assert_finite(values: dict[str, float]) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise AssertionError(f"non-finite return for {name}: {value}")
        if abs(value) > 1_000.0:
            raise AssertionError(f"return blow-up for {name}: {value}")


def validate_efficiency_v2_pathology() -> None:
    legacy = {fixture.name: efficiency_v2_summary_return(fixture) for fixture in FIXTURES}
    _assert_greater(
        legacy["fast_oob"],
        legacy["timeout_near"],
        label="efficiency_v2 fast_oob should outrank near timeout pathology",
        margin=100.0,
    )


def validate_arrival_v2_candidate(params: RewardParams, gamma: float) -> None:
    undiscounted = _returns(params=params)
    discounted = _returns(params=params, gamma=gamma)
    _assert_finite(undiscounted)
    _assert_finite(discounted)

    _assert_order(
        undiscounted,
        [
            "fast_success",
            "slow_success",
            "unsafe_success",
            "timeout_near",
            "timeout_far",
            "late_oob",
            "fast_oob",
            "mid_oob",
        ],
    )

    success_min = min(value for name, value in discounted.items() if "success" in name)
    timeout_max = max(value for name, value in discounted.items() if "timeout" in name)
    hard_failure_max = max(value for name, value in discounted.items() if "oob" in name)
    _assert_greater(
        success_min,
        timeout_max,
        label="discounted success must dominate timeout",
        margin=50.0,
    )
    _assert_greater(
        discounted["timeout_far"],
        discounted["late_oob"],
        label="discounted timeout_far must dominate late_oob",
    )
    _assert_greater(
        timeout_max,
        hard_failure_max,
        label="discounted timeout must dominate hard failure",
    )
    _assert_greater(
        timeout_max - 100.0,
        discounted["fast_oob"],
        label="discounted fast_oob must be much worse than timeout",
    )


def validate_shortcut_gate(params: RewardParams, gamma: float) -> None:
    safe_undiscounted = arrival_v2_return(SAFE_SHORTCUT_REFERENCE, params=params)
    risky_undiscounted = arrival_v2_return(RISKY_SHORTCUT_REFERENCE, params=params)
    safe_discounted = arrival_v2_return(SAFE_SHORTCUT_REFERENCE, params=params, gamma=gamma)
    risky_discounted = arrival_v2_return(RISKY_SHORTCUT_REFERENCE, params=params, gamma=gamma)

    _assert_greater(
        safe_undiscounted,
        risky_undiscounted,
        label="undiscounted safe route must dominate risky shortcut",
    )
    _assert_greater(
        safe_discounted,
        risky_discounted,
        label="discounted safe route must dominate risky shortcut",
    )


def validate_v5_weak_safety_is_caught(gamma: float) -> None:
    weak = RewardParams(w_safety=0.5)
    safe = arrival_v2_return(SAFE_SHORTCUT_REFERENCE, params=weak, gamma=gamma)
    risky = arrival_v2_return(RISKY_SHORTCUT_REFERENCE, params=weak, gamma=gamma)
    if safe >= risky:
        raise AssertionError(
            "w_safety=0.5 should fail the discounted shortcut gate; "
            f"safe={safe:.3f}, risky={risky:.3f}"
        )


def validate_w_safety_stress(gamma: float) -> None:
    for w_safety in (1.5, 2.0, 2.5, 3.0):
        params = RewardParams(w_safety=w_safety)
        undiscounted = _returns(params=params)
        discounted = _returns(params=params, gamma=gamma)
        _assert_finite(undiscounted)
        _assert_finite(discounted)
        _assert_greater(
            undiscounted["timeout_far"],
            undiscounted["late_oob"],
            label=f"timeout_far must dominate late_oob at w_safety={w_safety}",
        )
        timeout_max = max(value for name, value in discounted.items() if "timeout" in name)
        hard_failure_max = max(value for name, value in discounted.items() if "oob" in name)
        _assert_greater(
            timeout_max,
            hard_failure_max,
            label=f"discounted timeout must dominate hard failure at w_safety={w_safety}",
        )
        _assert_greater(
            timeout_max - 100.0,
            discounted["fast_oob"],
            label=f"discounted fast_oob must remain much worse at w_safety={w_safety}",
        )
        validate_shortcut_gate(params, gamma)

    beyond_limit = _returns(params=RewardParams(w_safety=4.0))
    if beyond_limit["timeout_far"] >= beyond_limit["late_oob"]:
        raise AssertionError("w_safety=4.0 should flip timeout_far vs late_oob")


def print_return_table(params: RewardParams, gamma: float) -> None:
    for label, values in (
        ("undiscounted", _returns(params=params)),
        (f"discounted_gamma_{gamma}", _returns(params=params, gamma=gamma)),
    ):
        print(f"\n[{label}]")
        for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True):
            print(f"{name:16s} {value:9.3f}")

    safe = arrival_v2_return(SAFE_SHORTCUT_REFERENCE, params=params, gamma=gamma)
    risky = arrival_v2_return(RISKY_SHORTCUT_REFERENCE, params=params, gamma=gamma)
    print(f"\n[discounted_shortcut] safe={safe:.3f} risky={risky:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--w-safety", type=float, default=2.0)
    parser.add_argument("--fast-success", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = replace(
        RewardParams(),
        w_safety=args.w_safety,
        r_fast_success=args.fast_success,
    )

    validate_efficiency_v2_pathology()
    validate_arrival_v2_candidate(params, args.gamma)
    validate_shortcut_gate(params, args.gamma)
    validate_v5_weak_safety_is_caught(args.gamma)
    validate_w_safety_stress(args.gamma)

    if not args.quiet:
        print_return_table(params, args.gamma)
        print("\nPASS: arrival_v2 candidate pre-integration gates passed.")


if __name__ == "__main__":
    main()
