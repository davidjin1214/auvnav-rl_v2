"""Planar REMUS-100 wake-navigation Gymnasium environment and related types."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .flow import (
    FlowSampler,
    ReferenceFlowConfig,
    ReferenceFlowEstimator,
    WakeField,
    rotation_world_to_body,
)
from .vehicle import ActuatorState, Remus100, S, ssa
from .autopilot import (
    DepthHold6DOFBackend,
    DepthHold6DOFBackendConfig,
    DepthHoldAutopilotConfig,
    EquivalentCurrentModel,
    HeadingAutopilotConfig,
)
from .reward import (
    RewardBreakdown,
    RewardModel,
    RewardModelConfig,
    SafetyCostBreakdown,
    SafetyCostModel,
    SafetyCostModelConfig,
)


# ---------------------------------------------------------------------------
# From envs/observation.py
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ObsNormScales:
    """Hand-tuned normalisation denominators so every obs channel is O(1).

    These are *not* learned*; they are physically-motivated constants that
    keep the observation vector well-scaled for neural-network inputs.
    """
    speed: float = 2.0          # m/s  — max expected AUV speed
    yaw_rate: float = 1.0       # rad/s — max expected yaw rate
    distance: float = 200.0     # m    — typical start-goal distance
    goal_xy: float = 200.0      # m    — same scale as distance
    flow_vel: float = 2.0       # m/s  — max expected flow velocity
    flow_omega: float = 2.0     # 1/s  — max expected vorticity


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    heading_cos_index: int
    heading_sin_index: int
    goal_x_index: int
    goal_y_index: int
    first_probe_index: int
    probe_stride: int
    num_probes: int

    @property
    def center_probe_v_index(self) -> int:
        return self.first_probe_index + 1


# ---------------------------------------------------------------------------
# From envs/task_sampler.py
# ---------------------------------------------------------------------------

def rotate_2d(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return rot @ np.asarray(vec, dtype=float)


@dataclass(slots=True)
class TaskSamplerConfig:
    boundary_margin_m: float
    benchmark_start_goal_distance_m: tuple[float, float]
    benchmark_direction_tolerance_rad: float
    benchmark_center_margin_m: float
    benchmark_max_sampling_tries: int
    task_geometry: str
    task_difficulty: str | None
    action_mode: str


class TaskSampler:
    """Resolve task geometry/action mode and sample valid start-goal pairs."""

    def __init__(self, wake_field: WakeField, config: TaskSamplerConfig) -> None:
        self.wake_field = wake_field
        self.config = config

    def resolve_task_geometry(self, options: dict[str, Any]) -> str:
        difficulty = options.get("task_difficulty", self.config.task_difficulty)
        if difficulty is not None:
            mapping = {
                "easy": "downstream",
                "medium": "cross_stream",
                "hard": "upstream",
            }
            return mapping[difficulty]

        geometry = str(options.get("task_geometry", self.config.task_geometry))
        return geometry

    def resolve_action_mode(self, options: dict[str, Any], task_geometry: str) -> str:
        _ = task_geometry
        mode = str(options.get("action_mode", self.config.action_mode))
        if mode == "auto":
            return "absolute_heading"
        return mode

    def sample_start_goal(
        self,
        *,
        rng: np.random.Generator,
        options: dict[str, Any],
        task_geometry: str,
        reference_flow_heading_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if "start_xy" in options and "goal_xy" in options:
            return (
                np.asarray(options["start_xy"], dtype=float),
                np.asarray(options["goal_xy"], dtype=float),
            )

        return self._sample_start_goal_benchmark(
            rng=rng,
            geometry=task_geometry,
            reference_flow_heading_rad=reference_flow_heading_rad,
        )

    def _sample_start_goal_benchmark(
        self,
        *,
        rng: np.random.Generator,
        geometry: str,
        reference_flow_heading_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_min = self.wake_field.x_min + self.config.boundary_margin_m
        x_max = self.wake_field.x_max - self.config.boundary_margin_m
        y_min = self.wake_field.y_min + self.config.boundary_margin_m
        y_max = self.wake_field.y_max - self.config.boundary_margin_m
        d_min, d_max = self.config.benchmark_start_goal_distance_m
        extra_margin = self.config.benchmark_center_margin_m

        for _ in range(self.config.benchmark_max_sampling_tries):
            direction = self._sample_task_direction(rng, geometry, reference_flow_heading_rad)
            feasible_distance = self._max_feasible_distance_for_direction(
                direction, x_min, x_max, y_min, y_max, extra_margin
            )
            max_distance = min(d_max, feasible_distance)
            if max_distance < d_min:
                continue

            distance = float(rng.uniform(d_min, max_distance))
            half_span = 0.5 * distance * direction
            x_lo = x_min + extra_margin + abs(half_span[0])
            x_hi = x_max - extra_margin - abs(half_span[0])
            y_lo = y_min + extra_margin + abs(half_span[1])
            y_hi = y_max - extra_margin - abs(half_span[1])
            if x_lo > x_hi or y_lo > y_hi:
                continue

            center = np.array(
                [float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))],
                dtype=float,
            )
            start_xy = center - half_span
            goal_xy = center + half_span
            if self._point_within_bounds(start_xy) and self._point_within_bounds(goal_xy):
                return start_xy, goal_xy

        raise RuntimeError(
            f"Failed to sample a valid {geometry} benchmark start-goal pair from the wake bounds."
        )

    def _sample_task_direction(
        self,
        rng: np.random.Generator,
        geometry: str,
        reference_flow_heading_rad: float,
    ) -> np.ndarray:
        if geometry == "downstream":
            heading = reference_flow_heading_rad
        elif geometry == "upstream":
            heading = reference_flow_heading_rad + np.pi
        elif geometry == "cross_stream":
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            heading = reference_flow_heading_rad + sign * (0.5 * np.pi)
        else:
            raise ValueError(f"Unsupported task geometry: {geometry}")

        jitter = float(
            rng.uniform(
                -self.config.benchmark_direction_tolerance_rad,
                self.config.benchmark_direction_tolerance_rad,
            )
        )
        unit = rotate_2d(np.array([np.cos(heading), np.sin(heading)], dtype=float), jitter)
        norm = float(np.linalg.norm(unit))
        if norm <= 1e-8:
            return np.array([1.0, 0.0], dtype=float)
        return unit / norm

    @staticmethod
    def _max_feasible_distance_for_direction(
        direction: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        extra_margin: float,
    ) -> float:
        usable_x = max(0.0, (x_max - x_min) - 2.0 * extra_margin)
        usable_y = max(0.0, (y_max - y_min) - 2.0 * extra_margin)
        dx = abs(float(direction[0]))
        dy = abs(float(direction[1]))
        caps = []
        if dx > 1e-8:
            caps.append(usable_x / dx)
        if dy > 1e-8:
            caps.append(usable_y / dy)
        if not caps:
            return 0.0
        return float(min(caps))

    def _point_within_bounds(self, xy: np.ndarray) -> bool:
        x = float(xy[0])
        y = float(xy[1])
        m = self.config.boundary_margin_m
        return (
            self.wake_field.x_min + m <= x <= self.wake_field.x_max - m
            and self.wake_field.y_min + m <= y <= self.wake_field.y_max - m
        )


# ---------------------------------------------------------------------------
# From envs/core.py
# ---------------------------------------------------------------------------

def _default_center_probe() -> np.ndarray:
    """Single center probe — the minimal observation scheme (Scheme A)."""
    return np.array([[0.0, 0.0]], dtype=float)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PlanarRemusEnvConfig:
    """All tuneable knobs for the planar wake-navigation environment."""

    # -- data paths --
    flow_path: str | Path = ""
    meta_path: str | Path | None = None

    # -- timing --
    sim_dt: float = 0.1            # inner-loop physics step  [s]
    control_dt: float = 0.5        # outer-loop RL decision step  [s]

    # -- episode geometry --
    goal_radius_m: float = 4.0
    max_episode_time_s: float = 240.0
    boundary_margin_m: float = 2.0
    benchmark_start_goal_distance_m: tuple[float, float] = (40.0, 90.0)
    benchmark_direction_tolerance_rad: float = np.deg2rad(20.0)
    benchmark_center_margin_m: float = 5.0
    benchmark_max_sampling_tries: int = 200
    task_geometry: str = "downstream"      # {"downstream", "cross_stream", "upstream"}
    task_difficulty: str | None = None     # {"easy", "medium", "hard"} → geometry alias

    # -- low-level heading PD controller --
    yaw_kp: float = 1.4
    yaw_kd: float = 0.4

    # -- hidden depth-hold autopilot --
    z_ref_m: float = 0.0
    depth_kp_z: float = 0.08
    depth_t_z: float = 25.0
    depth_k_grad: float = 0.02
    depth_kp_theta: float = 5.0
    depth_kd_theta: float = 2.5
    depth_ki_theta: float = 0.3
    depth_fail_tol_m: float = 1.5
    roll_fail_tol_rad: float = np.deg2rad(45.0)
    pitch_fail_tol_rad: float = np.deg2rad(35.0)

    # -- action mapping --
    backend_mode: str = "depth_hold_6dof"          # currently supported: depth_hold_6dof
    action_mode: str = "auto"                      # {"auto", "goal_relative_offset", "absolute_heading"}
    heading_offset_limit_rad: float = np.deg2rad(60.0)   # ±60°
    max_rpm_command: float = 1200.0

    # -- reward --
    step_penalty: float = -1.0          # per control step (not per sim step)
    time_penalty_per_second: float | None = None
    reward_progress_gain: float = 1.0   # scale on distance-progress shaping
    success_reward: float = 100.0
    failure_penalty: float = -20.0
    timeout_penalty: float | None = None
    energy_cost_gain: float = 0.0
    safety_cost_gain: float = 0.0

    # -- safety cost --
    safety_risk_activation_ratio: float = 0.7
    safety_boundary_risk_buffer_m: float = 12.0
    max_body_speed_mps: float = 3.5
    max_relative_speed_mps: float = 4.5
    max_angular_rate_radps: float = 5.0
    max_state_derivative: float = 250.0

    # -- flow time --
    randomize_start_time: bool = True
    reference_flow_grid_points: int = 9
    reference_flow_time_points: int = 3
    reference_flow_speed_quantile: float = 0.5

    # -- observation probes (agent perception) --
    probe_offsets_body: np.ndarray = field(default_factory=_default_center_probe)
    # "velocity" → (u, v) per probe (physically realistic, DVL/ADCP).
    # "full"     → (u, v, ω) per probe (includes vorticity, simulation-only).
    probe_channels: str = "velocity"

    # -- hull-averaged current for dynamics --
    hull_flow_sample_fractions: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4)
    hull_flow_sample_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    # -- observation normalisation --
    obs_norm: ObsNormScales = field(default_factory=ObsNormScales)
    normalize_obs: bool = True

    # -- benchmark speed scaling --
    target_speed_ratio: float | None = None
    target_auv_max_speed_mps: float | None = None
    nominal_max_speed_mps_at_max_rpm: float = 2.0

    def __post_init__(self) -> None:
        if self.sim_dt <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("sim_dt and control_dt must be positive.")
        if self.control_dt < self.sim_dt:
            raise ValueError("control_dt must be >= sim_dt.")
        if self.goal_radius_m <= 0.0:
            raise ValueError("goal_radius_m must be positive.")
        if self.backend_mode not in {"depth_hold_6dof"}:
            raise ValueError("backend_mode must be 'depth_hold_6dof'.")
        if self.action_mode not in {"auto", "goal_relative_offset", "absolute_heading"}:
            raise ValueError(
                "action_mode must be 'auto', 'goal_relative_offset', or 'absolute_heading'."
            )
        if len(self.hull_flow_sample_fractions) == 0:
            raise ValueError("At least one hull flow sample fraction is required.")
        if len(self.hull_flow_sample_fractions) != len(self.hull_flow_sample_weights):
            raise ValueError("Hull flow sample fractions and weights must have the same length.")
        if self.probe_channels not in {"velocity", "full"}:
            raise ValueError("probe_channels must be 'velocity' or 'full'.")
        if self.task_geometry not in {"downstream", "cross_stream", "upstream"}:
            raise ValueError(
                "task_geometry must be one of: downstream, cross_stream, upstream."
            )
        if self.task_difficulty not in {None, "easy", "medium", "hard"}:
            raise ValueError("task_difficulty must be None, 'easy', 'medium', or 'hard'.")
        d_min, d_max = self.benchmark_start_goal_distance_m
        if d_min <= 0.0 or d_max < d_min:
            raise ValueError("benchmark_start_goal_distance_m must satisfy 0 < min <= max.")
        if self.benchmark_direction_tolerance_rad < 0.0:
            raise ValueError("benchmark_direction_tolerance_rad must be nonnegative.")
        if self.benchmark_center_margin_m < 0.0:
            raise ValueError("benchmark_center_margin_m must be nonnegative.")
        if self.benchmark_max_sampling_tries <= 0:
            raise ValueError("benchmark_max_sampling_tries must be positive.")
        if self.reference_flow_grid_points < 2:
            raise ValueError("reference_flow_grid_points must be at least 2.")
        if self.reference_flow_time_points < 1:
            raise ValueError("reference_flow_time_points must be at least 1.")
        if not 0.0 <= self.reference_flow_speed_quantile <= 1.0:
            raise ValueError("reference_flow_speed_quantile must be in [0, 1].")
        if self.target_speed_ratio is not None and self.target_speed_ratio <= 0.0:
            raise ValueError("target_speed_ratio must be positive when provided.")
        if self.target_auv_max_speed_mps is not None and self.target_auv_max_speed_mps <= 0.0:
            raise ValueError("target_auv_max_speed_mps must be positive when provided.")
        if self.nominal_max_speed_mps_at_max_rpm <= 0.0:
            raise ValueError("nominal_max_speed_mps_at_max_rpm must be positive.")
        if not 0.0 <= self.safety_risk_activation_ratio < 1.0:
            raise ValueError("safety_risk_activation_ratio must be in [0, 1).")
        if self.safety_boundary_risk_buffer_m <= 0.0:
            raise ValueError("safety_boundary_risk_buffer_m must be positive.")
        if self.max_body_speed_mps <= 0.0:
            raise ValueError("max_body_speed_mps must be positive.")
        if self.max_relative_speed_mps <= 0.0:
            raise ValueError("max_relative_speed_mps must be positive.")
        if self.max_angular_rate_radps <= 0.0:
            raise ValueError("max_angular_rate_radps must be positive.")
        if self.max_state_derivative <= 0.0:
            raise ValueError("max_state_derivative must be positive.")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PlanarRemusEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for planar AUV navigation in a dynamic wake field.

    Observation layout (when normalize_obs=True, every channel is O(1)):
        [0]   u / speed_scale           surge velocity
        [1]   v / speed_scale           sway velocity
        [2]   r / yaw_rate_scale        yaw rate
        [3]   cos(psi)                  heading (x)
        [4]   sin(psi)                  heading (y)
        [5]   goal_body_x / goal_scale  goal in body frame
        [6]   goal_body_y / goal_scale
        [7]   distance / dist_scale
        [8..] probe flow velocities     (n_probes × 3) flattened
    """

    metadata = {"render_modes": []}

    def __init__(self, config: PlanarRemusEnvConfig) -> None:
        super().__init__()
        self.config = config
        self.vehicle = Remus100()
        self.wake_field = WakeField.from_files(config.flow_path, config.meta_path)
        self.flow_sampler = FlowSampler(self.wake_field, loop_time=True)
        self.probe_offsets_body = np.asarray(config.probe_offsets_body, dtype=float)
        self.equivalent_current_model = EquivalentCurrentModel(
            flow_sampler=self.flow_sampler,
            vehicle_length_m=self.vehicle.params.length,
            sample_fractions=config.hull_flow_sample_fractions,
            sample_weights=config.hull_flow_sample_weights,
        )
        self.task_sampler = TaskSampler(
            wake_field=self.wake_field,
            config=TaskSamplerConfig(
                boundary_margin_m=config.boundary_margin_m,
                benchmark_start_goal_distance_m=config.benchmark_start_goal_distance_m,
                benchmark_direction_tolerance_rad=config.benchmark_direction_tolerance_rad,
                benchmark_center_margin_m=config.benchmark_center_margin_m,
                benchmark_max_sampling_tries=config.benchmark_max_sampling_tries,
                task_geometry=config.task_geometry,
                task_difficulty=config.task_difficulty,
                action_mode=config.action_mode,
            ),
        )
        self.reference_flow_estimator = ReferenceFlowEstimator(
            wake_field=self.wake_field,
            flow_sampler=self.flow_sampler,
            config=ReferenceFlowConfig(
                grid_points=config.reference_flow_grid_points,
                time_points=config.reference_flow_time_points,
                speed_quantile=config.reference_flow_speed_quantile,
            ),
        )
        self.reward_model = RewardModel(
            RewardModelConfig(
                control_dt=config.control_dt,
                step_penalty=config.step_penalty,
                time_penalty_per_second=config.time_penalty_per_second,
                reward_progress_gain=config.reward_progress_gain,
                success_reward=config.success_reward,
                failure_penalty=config.failure_penalty,
                timeout_penalty=config.timeout_penalty,
                energy_cost_gain=config.energy_cost_gain,
                safety_cost_gain=config.safety_cost_gain,
            )
        )
        self.safety_cost_model = SafetyCostModel(
            SafetyCostModelConfig(
                boundary_margin_m=config.boundary_margin_m,
                boundary_risk_buffer_m=config.safety_boundary_risk_buffer_m,
                depth_fail_tol_m=config.depth_fail_tol_m,
                roll_fail_tol_rad=config.roll_fail_tol_rad,
                pitch_fail_tol_rad=config.pitch_fail_tol_rad,
                max_body_speed_mps=config.max_body_speed_mps,
                max_relative_speed_mps=config.max_relative_speed_mps,
                max_angular_rate_radps=config.max_angular_rate_radps,
                risk_activation_ratio=config.safety_risk_activation_ratio,
            )
        )
        self.backend = DepthHold6DOFBackend(
            vehicle=self.vehicle,
            equivalent_current_model=self.equivalent_current_model,
            heading_config=HeadingAutopilotConfig(
                yaw_kp=config.yaw_kp,
                yaw_kd=config.yaw_kd,
                rudder_limit_rad=self.vehicle.params.delta_r_max,
            ),
            depth_config=DepthHoldAutopilotConfig(
                z_ref_m=config.z_ref_m,
                kp_z=config.depth_kp_z,
                t_z=config.depth_t_z,
                k_grad=config.depth_k_grad,
                kp_theta=config.depth_kp_theta,
                kd_theta=config.depth_kd_theta,
                ki_theta=config.depth_ki_theta,
                elevator_limit_rad=self.vehicle.params.delta_s_max,
            ),
            backend_config=DepthHold6DOFBackendConfig(
                sim_dt=config.sim_dt,
                z_ref_m=config.z_ref_m,
                depth_fail_tol_m=config.depth_fail_tol_m,
                roll_fail_tol_rad=config.roll_fail_tol_rad,
                pitch_fail_tol_rad=config.pitch_fail_tol_rad,
                max_body_speed_mps=config.max_body_speed_mps,
                max_relative_speed_mps=config.max_relative_speed_mps,
                max_angular_rate_radps=config.max_angular_rate_radps,
                max_state_derivative=config.max_state_derivative,
            ),
        )
        self.n_substeps = int(round(config.control_dt / config.sim_dt))
        if not np.isclose(self.n_substeps * config.sim_dt, config.control_dt):
            raise ValueError("control_dt must be an integer multiple of sim_dt.")

        # --- spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        n_probes = self.probe_offsets_body.shape[0]
        self.channels_per_probe = 2 if config.probe_channels == "velocity" else 3
        obs_dim = 8 + n_probes * self.channels_per_probe
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        # Named slices for decode_observation
        self._obs_own = slice(0, 5)
        self._obs_goal = slice(5, 8)
        self._obs_probe = slice(8, obs_dim)
        self.observation_layout = ObservationLayout(
            heading_cos_index=3,
            heading_sin_index=4,
            goal_x_index=5,
            goal_y_index=6,
            first_probe_index=8,
            probe_stride=self.channels_per_probe,
            num_probes=n_probes,
        )

        # Precompute per-step probe normalization scales (immutable after init).
        ns0 = config.obs_norm
        if self.channels_per_probe == 2:
            _channel_scales = [ns0.flow_vel, ns0.flow_vel]
        else:
            _channel_scales = [ns0.flow_vel, ns0.flow_vel, ns0.flow_omega]
        self._probe_flow_scales = np.tile(_channel_scales, n_probes).astype(np.float32)

        # --- mutable episode state ---
        self.state = np.zeros(S.N, dtype=float)
        self.actuator_state = ActuatorState()
        self.goal_xy = np.zeros(2, dtype=float)
        self.start_xy = np.zeros(2, dtype=float)
        self.elapsed_time = 0.0
        self.flow_time = 0.0
        self.last_distance = np.inf
        self.initial_distance = np.inf
        self.cumulative_rpm = 0.0
        self.step_count = 0
        self.last_equivalent_current_world = np.zeros(3, dtype=np.float32)
        self.last_equivalent_current_body = np.zeros(3, dtype=np.float32)
        self.last_center_current_world = np.zeros(3, dtype=np.float32)
        self.last_reward_breakdown = RewardBreakdown(
            reward=0.0,
            task_reward=0.0,
            progress_reward=0.0,
            terminal_reward=0.0,
            safety_cost=0.0,
            energy_cost=0.0,
        )
        self.last_safety_cost_breakdown = SafetyCostBreakdown(
            total=0.0,
            boundary_risk=0.0,
            depth_risk=0.0,
            roll_risk=0.0,
            pitch_risk=0.0,
            speed_risk=0.0,
            relative_speed_risk=0.0,
            angular_rate_risk=0.0,
            terminal_violation=0.0,
        )
        self.reference_flow_world = np.zeros(2, dtype=np.float32)
        self.reference_flow_speed_mps = 0.0
        self.reference_flow_heading_rad = 0.0
        self.current_task_geometry = config.task_geometry
        self.current_action_mode = self.task_sampler.resolve_action_mode({}, self.current_task_geometry)
        self.current_target_auv_max_speed_mps = config.nominal_max_speed_mps_at_max_rpm
        self.current_max_rpm_command = float(config.max_rpm_command)
        self._refresh_reference_flow()

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "flow_time" in options:
            self.flow_time = float(options["flow_time"])
        elif self.config.randomize_start_time and self.wake_field.time_horizon > 0.0:
            self.flow_time = float(
                self.np_random.uniform(0.0, self.wake_field.time_horizon)
            )
        else:
            self.flow_time = 0.0

        self.current_task_geometry = self.task_sampler.resolve_task_geometry(options)
        self.current_action_mode = self.task_sampler.resolve_action_mode(
            options, self.current_task_geometry
        )
        self._refresh_reference_flow(center_time=self.flow_time)
        self.current_target_auv_max_speed_mps = self._resolve_target_auv_max_speed(options)
        self.current_max_rpm_command = self._resolve_max_rpm_command(
            self.current_target_auv_max_speed_mps
        )

        start_xy, goal_xy = self.task_sampler.sample_start_goal(
            rng=self.np_random,
            options=options,
            task_geometry=self.current_task_geometry,
            reference_flow_heading_rad=self.reference_flow_heading_rad,
        )
        self.start_xy = start_xy.copy()
        self.goal_xy = goal_xy.copy()

        initial_heading = options.get("initial_heading")
        if initial_heading is None:
            goal_bearing = np.arctan2(
                goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]
            )
            initial_heading = float(goal_bearing + self.np_random.uniform(-0.25, 0.25))

        self.backend.reset(
            initial_xy=start_xy,
            initial_heading=ssa(float(initial_heading)),
            initial_speed=float(options.get("initial_speed", 0.3)),
        )
        self._sync_backend_state()
        current = self.equivalent_current_model.sample(self.state, self.flow_time)
        self.backend.last_current = current
        self.last_equivalent_current_world = current.world.copy()
        self.last_equivalent_current_body = current.body.copy()
        self.last_center_current_world = current.center_world.copy()

        self.elapsed_time = 0.0
        self.cumulative_rpm = 0.0
        self.step_count = 0
        self.last_reward_breakdown = RewardBreakdown(
            reward=0.0,
            task_reward=0.0,
            progress_reward=0.0,
            terminal_reward=0.0,
            safety_cost=0.0,
            energy_cost=0.0,
        )
        self.last_safety_cost_breakdown = SafetyCostBreakdown(
            total=0.0,
            boundary_risk=0.0,
            depth_risk=0.0,
            roll_risk=0.0,
            pitch_risk=0.0,
            speed_risk=0.0,
            relative_speed_risk=0.0,
            angular_rate_risk=0.0,
            terminal_violation=0.0,
        )

        self.last_distance = self._distance_to_goal()
        self.initial_distance = self.last_distance

        obs = self._build_observation()
        info = self._build_info(
            success=False, terminated=False, truncated=False, reason="reset"
        )
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(2)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        terminated = False
        truncated = False
        reason = "running"

        # --- decode action ---
        psi_ref = self._decode_heading_action(action)
        rpm_cmd = float(0.5 * (action[1] + 1.0) * self.current_max_rpm_command)

        # --- inner loop: n_substeps of physics ---
        for _ in range(self.n_substeps):
            self.backend.substep(
                psi_ref=psi_ref,
                rpm_cmd=rpm_cmd,
                flow_time=self.flow_time,
            )
            self._sync_backend_state()
            backend_info = self.backend.info_state()
            self.last_equivalent_current_world = backend_info["equivalent_current_world"]
            self.last_equivalent_current_body = backend_info["equivalent_current_body"]
            self.last_center_current_world = backend_info["center_current_world"]

            self.elapsed_time += self.config.sim_dt
            self.flow_time += self.config.sim_dt
            self.cumulative_rpm += abs(self.actuator_state.n_rpm) * self.config.sim_dt

            # --- termination checks ---
            validity = self.backend.validity()
            if not validity.ok:
                terminated = True
                reason = validity.reason
                break

            if self._distance_to_goal() <= self.config.goal_radius_m:
                terminated = True
                reason = "goal"
                break

            if not self._within_bounds():
                terminated = True
                reason = "out_of_bounds"
                break

            if self.elapsed_time >= self.config.max_episode_time_s:
                truncated = True
                reason = "timeout"
                break

        # --- reward ---
        distance = self._distance_to_goal()
        progress = self.last_distance - distance
        safety_breakdown = self.safety_cost_model.compute(env=self, reason=reason)
        reward_breakdown = self.reward_model.compute(
            progress=progress,
            safety_cost=safety_breakdown.total,
            reason=reason,
            terminated=terminated,
            truncated=truncated,
            actuator_rpm=float(self.actuator_state.n_rpm),
        )
        reward = reward_breakdown.reward
        self.last_safety_cost_breakdown = safety_breakdown
        self.last_reward_breakdown = reward_breakdown
        self.last_distance = distance
        self.step_count += 1

        obs = self._build_observation()
        info = self._build_info(
            success=(reason == "goal"),
            terminated=terminated,
            truncated=truncated,
            reason=reason,
        )
        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Observation helpers
    # -----------------------------------------------------------------------

    def decode_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat observation vector into named sub-arrays."""
        obs = np.asarray(obs, dtype=np.float32)
        n_probes = self.probe_offsets_body.shape[0]
        return {
            "own": obs[self._obs_own],
            "goal": obs[self._obs_goal],
            "probes": obs[self._obs_probe].reshape(n_probes, self.channels_per_probe),
        }

    def _build_observation(self) -> np.ndarray:
        psi = float(self.state[S.PSI])
        pos_xy = np.array([self.state[S.XN], self.state[S.YE]], dtype=float)
        goal_world = self.goal_xy - pos_xy
        rot = rotation_world_to_body(psi)
        goal_body = rot @ goal_world
        distance = float(np.linalg.norm(goal_world))

        # Probe flow sensing — sample all 3 channels, then keep channels_per_probe.
        probes_all = self.flow_sampler.sample_probes_body(
            float(pos_xy[0]),
            float(pos_xy[1]),
            psi,
            self.flow_time,
            self.probe_offsets_body,
        )
        probes_raw = np.ascontiguousarray(
            probes_all[:, : self.channels_per_probe]
        ).reshape(-1)

        ns = self.config.obs_norm if self.config.normalize_obs else None

        own = np.array([
            self.state[S.U]  / (ns.speed if ns else 1.0),
            self.state[S.V]  / (ns.speed if ns else 1.0),
            self.state[S.R]  / (ns.yaw_rate if ns else 1.0),
            np.cos(psi),
            np.sin(psi),
        ], dtype=np.float32)

        goal = np.array([
            goal_body[0] / (ns.goal_xy if ns else 1.0),
            goal_body[1] / (ns.goal_xy if ns else 1.0),
            distance     / (ns.distance if ns else 1.0),
        ], dtype=np.float32)

        if ns:
            probes_obs = probes_raw / self._probe_flow_scales
        else:
            probes_obs = probes_raw.astype(np.float32)

        return np.concatenate([own, goal, probes_obs])

    # -----------------------------------------------------------------------
    # Info dict
    # -----------------------------------------------------------------------

    def _build_info(
        self,
        *,
        success: bool,
        terminated: bool,
        truncated: bool,
        reason: str,
    ) -> dict[str, Any]:
        equivalent_world = self.last_equivalent_current_world
        equivalent_body = self.last_equivalent_current_body
        depth_hold_ok = (
            abs(float(self.state[S.ZD] - self.config.z_ref_m)) <= self.config.depth_fail_tol_m
            and abs(float(self.state[S.PHI])) <= self.config.roll_fail_tol_rad
            and abs(float(self.state[S.THETA])) <= self.config.pitch_fail_tol_rad
        )
        failure = bool((terminated or truncated) and not success)
        return {
            # episode status
            "success": success,
            "failure": failure,
            "terminated": terminated,
            "truncated": truncated,
            "reason": reason,
            # timing
            "elapsed_time_s": float(self.elapsed_time),
            "flow_time_s": float(self.flow_time),
            "step_count": self.step_count,
            # navigation
            "distance_to_goal_m": float(self._distance_to_goal()),
            "initial_distance_m": float(self.initial_distance),
            "position_xy_m": np.array(
                [self.state[S.XN], self.state[S.YE]], dtype=np.float32
            ),
            "start_xy_m": self.start_xy.astype(np.float32),
            "goal_xy_m": self.goal_xy.astype(np.float32),
            "psi_rad": float(self.state[S.PSI]),
            # body velocity
            "body_velocity": self.state[[S.U, S.V, S.R]].astype(np.float32),
            "ground_speed_mps": float(np.hypot(self.state[S.U], self.state[S.V])),
            "water_relative_speed_mps": float(
                np.hypot(
                    self.state[S.U] - equivalent_body[0],
                    self.state[S.V] - equivalent_body[1],
                )
            ),
            # flow field
            "center_current_world": self.last_center_current_world.astype(np.float32),
            "equivalent_current_world": equivalent_world.astype(np.float32),
            "equivalent_current_body": equivalent_body.astype(np.float32),
            # Privileged observation for asymmetric critic training (not used by actor).
            # Contains body-frame equivalent flow [u_eq, v_eq] in m/s.
            "privileged_obs": equivalent_body[:2].astype(np.float32),
            "reference_flow_world": self.reference_flow_world.copy(),
            "reference_flow_speed_mps": float(self.reference_flow_speed_mps),
            "reference_flow_heading_rad": float(self.reference_flow_heading_rad),
            # control effort
            "cumulative_rpm": float(self.cumulative_rpm),
            "actuator_rpm": float(self.actuator_state.n_rpm),
            "actuator_rudder_rad": float(self.actuator_state.delta_r),
            "actuator_elevator_rad": float(self.actuator_state.delta_s),
            "task_geometry": self.current_task_geometry,
            "action_mode": self.current_action_mode,
            "target_auv_max_speed_mps": float(self.current_target_auv_max_speed_mps),
            "max_rpm_command": float(self.current_max_rpm_command),
            "reward_task": float(self.last_reward_breakdown.task_reward),
            "reward_progress": float(self.last_reward_breakdown.progress_reward),
            "reward_terminal": float(self.last_reward_breakdown.terminal_reward),
            "step_safety_cost": float(self.last_safety_cost_breakdown.total),
            "safety_cost": float(self.last_reward_breakdown.safety_cost),
            "boundary_risk": float(self.last_safety_cost_breakdown.boundary_risk),
            "depth_risk": float(self.last_safety_cost_breakdown.depth_risk),
            "roll_risk": float(self.last_safety_cost_breakdown.roll_risk),
            "pitch_risk": float(self.last_safety_cost_breakdown.pitch_risk),
            "speed_risk": float(self.last_safety_cost_breakdown.speed_risk),
            "relative_speed_risk": float(self.last_safety_cost_breakdown.relative_speed_risk),
            "angular_rate_risk": float(self.last_safety_cost_breakdown.angular_rate_risk),
            "terminal_violation_cost": float(self.last_safety_cost_breakdown.terminal_violation),
            "energy_cost": float(self.last_reward_breakdown.energy_cost),
            "depth_m": float(self.state[S.ZD]),
            "depth_error_m": float(self.state[S.ZD] - self.config.z_ref_m),
            "roll_rad": float(self.state[S.PHI]),
            "pitch_rad": float(self.state[S.THETA]),
            "autopilot_delta_r_cmd": float(self.backend.last_delta_r_cmd),
            "autopilot_delta_s_cmd": float(self.backend.last_delta_s_cmd),
            "depth_hold_ok": bool(depth_hold_ok),
        }

    def _refresh_reference_flow(self, center_time: float | None = None) -> None:
        ref = self.reference_flow_estimator.estimate(center_time=center_time)
        self.reference_flow_world = ref.world.copy()
        self.reference_flow_speed_mps = float(ref.speed_mps)
        self.reference_flow_heading_rad = float(ref.heading_rad)

    def get_observation_layout(self) -> ObservationLayout:
        return self.observation_layout

    def _resolve_target_auv_max_speed(self, options: dict[str, Any]) -> float:
        if "target_auv_max_speed_mps" in options:
            return float(options["target_auv_max_speed_mps"])
        if self.config.target_auv_max_speed_mps is not None:
            return float(self.config.target_auv_max_speed_mps)

        target_speed_ratio = options.get("target_speed_ratio", self.config.target_speed_ratio)
        if target_speed_ratio is not None:
            return float(target_speed_ratio) * self.reference_flow_speed_mps

        return float(self.config.nominal_max_speed_mps_at_max_rpm)

    def _resolve_max_rpm_command(self, target_auv_max_speed_mps: float) -> float:
        nominal_speed = self.config.nominal_max_speed_mps_at_max_rpm
        rpm = self.config.max_rpm_command * (target_auv_max_speed_mps / nominal_speed)
        return float(np.clip(rpm, 0.0, self.vehicle.params.n_rpm_max))

    # -----------------------------------------------------------------------
    # Low-level heading controller
    # -----------------------------------------------------------------------

    def _decode_heading_action(self, action: np.ndarray) -> float:
        if self.current_action_mode == "absolute_heading":
            return ssa(float(action[0]) * np.pi)

        goal_heading = self.goal_heading()
        heading_offset = float(action[0]) * self.config.heading_offset_limit_rad
        return ssa(goal_heading + heading_offset)

    def goal_heading(self) -> float:
        return float(np.arctan2(
            self.goal_xy[1] - self.state[S.YE],
            self.goal_xy[0] - self.state[S.XN],
        ))

    def encode_heading_action(self, desired_heading: float) -> float:
        desired_heading = ssa(float(desired_heading))
        if self.current_action_mode == "absolute_heading":
            return float(np.clip(desired_heading / np.pi, -1.0, 1.0))
        offset = ssa(desired_heading - self.goal_heading())
        limit = max(float(self.config.heading_offset_limit_rad), 1e-6)
        return float(np.clip(offset / limit, -1.0, 1.0))

    def _sync_backend_state(self) -> None:
        self.state = self.backend.state
        self.actuator_state = self.backend.actuator_state

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    def _distance_to_goal(self) -> float:
        return float(
            np.hypot(
                self.goal_xy[0] - self.state[S.XN],
                self.goal_xy[1] - self.state[S.YE],
            )
        )

    def _within_bounds(self) -> bool:
        x = float(self.state[S.XN])
        y = float(self.state[S.YE])
        m = self.config.boundary_margin_m
        return (
            self.wake_field.x_min + m <= x <= self.wake_field.x_max - m
            and self.wake_field.y_min + m <= y <= self.wake_field.y_max - m
        )


# ---------------------------------------------------------------------------
# From envs/wrappers.py
# ---------------------------------------------------------------------------

class ObservationHistoryWrapper(gym.ObservationWrapper):
    """Stack the latest K observations for feedforward history baselines."""

    def __init__(self, env: gym.Env, history_length: int) -> None:
        super().__init__(env)
        self.history_length = max(1, int(history_length))
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("ObservationHistoryWrapper requires a Box observation space.")
        self._single_shape = obs_space.shape
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)

        low = np.tile(obs_space.low, self.history_length)
        high = np.tile(obs_space.high, self.history_length)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(obs.copy())
        return self.observation(obs), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=self.observation_space.dtype)
        if not self._history:
            for _ in range(self.history_length):
                self._history.append(obs.copy())
        else:
            self._history.append(obs.copy())
        return np.concatenate(tuple(self._history), axis=0)
