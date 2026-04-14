"""Reward model and safety cost model for the planar REMUS environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .vehicle import S

if TYPE_CHECKING:
    from .env import PlanarRemusEnv


SAFETY_FAILURE_REASONS = frozenset(
    {
        "out_of_bounds",
        "depth_hold_failure",
        "timeout",
        "speed_limit",
        "relative_speed_limit",
        "nonfinite_state",
        "nonfinite_derivative",
        "derivative_limit",
        "angular_rate_limit",
        "attitude_limit",
    }
)


REWARD_CONFIG_FIELDS = (
    "step_penalty",
    "time_penalty_per_second",
    "reward_progress_gain",
    "success_reward",
    "failure_penalty",
    "timeout_penalty",
    "energy_cost_gain",
    "safety_cost_gain",
)


@dataclass(frozen=True, slots=True)
class RewardObjectivePreset:
    key: str
    description: str
    reward_config: dict[str, float | None]


REWARD_OBJECTIVE_PRESETS: dict[str, RewardObjectivePreset] = {
    "arrival_v1": RewardObjectivePreset(
        key="arrival_v1",
        description="Legacy task-completion objective: optimize time-to-goal with progress shaping.",
        reward_config={
            "step_penalty": -1.0,
            "time_penalty_per_second": None,
            "reward_progress_gain": 1.0,
            "success_reward": 100.0,
            "failure_penalty": -20.0,
            "timeout_penalty": None,
            "energy_cost_gain": 0.0,
            "safety_cost_gain": 0.0,
        },
    ),
    "efficiency_v1": RewardObjectivePreset(
        key="efficiency_v1",
        description=(
            "Efficiency-aware objective: retain time/progress shaping while penalizing "
            "control effort and soft safety-risk accumulation."
        ),
        reward_config={
            "step_penalty": -1.0,
            "time_penalty_per_second": None,
            "reward_progress_gain": 1.0,
            "success_reward": 100.0,
            "failure_penalty": -20.0,
            "timeout_penalty": None,
            "energy_cost_gain": 5e-4,
            "safety_cost_gain": 2.0,
        },
    ),
    "efficiency_v2": RewardObjectivePreset(
        key="efficiency_v2",
        description=(
            "Weak safety-shaped objective selected from the gain sweep: preserve arrival "
            "learning while adding only a light safety-risk penalty."
        ),
        reward_config={
            "step_penalty": -1.0,
            "time_penalty_per_second": None,
            "reward_progress_gain": 1.0,
            "success_reward": 100.0,
            "failure_penalty": -20.0,
            "timeout_penalty": None,
            "energy_cost_gain": 0.0,
            "safety_cost_gain": 0.25,
        },
    ),
}

REWARD_OBJECTIVE_ALIASES = {
    "arrival": "arrival_v1",
    "time_optimal": "arrival_v1",
    "efficiency": "efficiency_v1",
    "efficient": "efficiency_v1",
    "efficiency_aware": "efficiency_v1",
    "efficiency_weak_safety": "efficiency_v2",
}


def canonical_reward_objective(key: str) -> str:
    canonical = REWARD_OBJECTIVE_ALIASES.get(key, key)
    if canonical not in REWARD_OBJECTIVE_PRESETS:
        known = ", ".join(sorted(REWARD_OBJECTIVE_PRESETS))
        raise ValueError(f"Unknown reward objective: {key}. Known objectives: {known}")
    return canonical


def reward_objective_config(key: str) -> dict[str, float | None]:
    canonical = canonical_reward_objective(key)
    return dict(REWARD_OBJECTIVE_PRESETS[canonical].reward_config)


@dataclass(slots=True)
class RewardBreakdown:
    reward: float
    task_reward: float
    progress_reward: float
    terminal_reward: float
    safety_cost: float
    energy_cost: float


@dataclass(slots=True)
class RewardModelConfig:
    control_dt: float
    step_penalty: float
    reward_progress_gain: float
    success_reward: float
    failure_penalty: float
    time_penalty_per_second: float | None = None
    timeout_penalty: float | None = None
    energy_cost_gain: float = 0.0
    safety_cost_gain: float = 0.0


class RewardModel:
    """Time-consistent reward and auxiliary cost accounting."""

    def __init__(self, config: RewardModelConfig) -> None:
        self.config = config

    def compute(
        self,
        *,
        progress: float,
        safety_cost: float,
        reason: str,
        terminated: bool,
        truncated: bool,
        actuator_rpm: float,
    ) -> RewardBreakdown:
        cfg = self.config
        time_penalty_per_second = cfg.time_penalty_per_second
        if time_penalty_per_second is None:
            time_penalty_per_second = max(0.0, -cfg.step_penalty / max(cfg.control_dt, 1e-8))

        task_reward = -time_penalty_per_second * cfg.control_dt
        progress_reward = cfg.reward_progress_gain * progress
        terminal_reward = 0.0
        if reason == "goal":
            terminal_reward += cfg.success_reward
        elif truncated and reason == "timeout":
            terminal_reward += (
                cfg.timeout_penalty if cfg.timeout_penalty is not None else cfg.failure_penalty
            )
        elif terminated:
            terminal_reward += cfg.failure_penalty

        energy_cost = abs(float(actuator_rpm)) * cfg.control_dt

        reward = task_reward + progress_reward + terminal_reward
        reward -= cfg.safety_cost_gain * safety_cost
        reward -= cfg.energy_cost_gain * energy_cost
        return RewardBreakdown(
            reward=float(reward),
            task_reward=float(task_reward),
            progress_reward=float(progress_reward),
            terminal_reward=float(terminal_reward),
            safety_cost=float(safety_cost),
            energy_cost=float(energy_cost),
        )


@dataclass(slots=True)
class SafetyCostBreakdown:
    total: float
    boundary_risk: float
    depth_risk: float
    roll_risk: float
    pitch_risk: float
    speed_risk: float
    relative_speed_risk: float
    angular_rate_risk: float
    terminal_violation: float


@dataclass(slots=True)
class SafetyCostModelConfig:
    boundary_margin_m: float
    boundary_risk_buffer_m: float
    depth_fail_tol_m: float
    roll_fail_tol_rad: float
    pitch_fail_tol_rad: float
    max_body_speed_mps: float
    max_relative_speed_mps: float
    max_angular_rate_radps: float
    risk_activation_ratio: float = 0.7


class SafetyCostModel:
    """Unified step cost for risk shaping and terminal constraint violations."""

    def __init__(self, config: SafetyCostModelConfig) -> None:
        self.config = config

    @staticmethod
    def _soft_margin_risk(value: float, limit: float, activation_ratio: float) -> float:
        if limit <= 0.0:
            return 0.0
        activation = max(0.0, min(float(activation_ratio), 0.999)) * limit
        if value <= activation:
            return 0.0
        if value >= limit:
            return 1.0
        scaled = (value - activation) / max(limit - activation, 1e-8)
        return float(np.clip(scaled * scaled, 0.0, 1.0))

    def compute(
        self,
        *,
        env: "PlanarRemusEnv",
        reason: str,
    ) -> SafetyCostBreakdown:
        cfg = self.config
        state = env.state
        x = float(state[S.XN])
        y = float(state[S.YE])
        depth_error = abs(float(state[S.ZD] - env.config.z_ref_m))
        roll_abs = abs(float(state[S.PHI]))
        pitch_abs = abs(float(state[S.THETA]))
        body_speed = float(np.hypot(state[S.U], state[S.V]))
        relative_speed = float(
            np.hypot(
                state[S.U] - env.last_equivalent_current_body[0],
                state[S.V] - env.last_equivalent_current_body[1],
            )
        )
        angular_rate = float(np.max(np.abs(state[[S.P, S.Q, S.R]])))

        x_lo = env.wake_field.x_min + cfg.boundary_margin_m
        x_hi = env.wake_field.x_max - cfg.boundary_margin_m
        y_lo = env.wake_field.y_min + cfg.boundary_margin_m
        y_hi = env.wake_field.y_max - cfg.boundary_margin_m
        clearance = min(x - x_lo, x_hi - x, y - y_lo, y_hi - y)
        boundary_buffer = max(cfg.boundary_risk_buffer_m, 1e-6)
        if clearance <= 0.0:
            boundary_risk = 1.0
        elif clearance >= boundary_buffer:
            boundary_risk = 0.0
        else:
            scaled = 1.0 - (clearance / boundary_buffer)
            boundary_risk = float(np.clip(scaled * scaled, 0.0, 1.0))

        depth_risk = self._soft_margin_risk(
            depth_error,
            cfg.depth_fail_tol_m,
            cfg.risk_activation_ratio,
        )
        roll_risk = self._soft_margin_risk(
            roll_abs,
            cfg.roll_fail_tol_rad,
            cfg.risk_activation_ratio,
        )
        pitch_risk = self._soft_margin_risk(
            pitch_abs,
            cfg.pitch_fail_tol_rad,
            cfg.risk_activation_ratio,
        )
        speed_risk = self._soft_margin_risk(
            body_speed,
            cfg.max_body_speed_mps,
            cfg.risk_activation_ratio,
        )
        relative_speed_risk = self._soft_margin_risk(
            relative_speed,
            cfg.max_relative_speed_mps,
            cfg.risk_activation_ratio,
        )
        angular_rate_risk = self._soft_margin_risk(
            angular_rate,
            cfg.max_angular_rate_radps,
            cfg.risk_activation_ratio,
        )

        terminal_violation = 1.0 if reason in SAFETY_FAILURE_REASONS else 0.0
        total = max(
            boundary_risk,
            depth_risk,
            roll_risk,
            pitch_risk,
            speed_risk,
            relative_speed_risk,
            angular_rate_risk,
            terminal_violation,
        )
        return SafetyCostBreakdown(
            total=float(total),
            boundary_risk=float(boundary_risk),
            depth_risk=float(depth_risk),
            roll_risk=float(roll_risk),
            pitch_risk=float(pitch_risk),
            speed_risk=float(speed_risk),
            relative_speed_risk=float(relative_speed_risk),
            angular_rate_risk=float(angular_rate_risk),
            terminal_violation=float(terminal_violation),
        )
