"""
Baseline navigation policies for the planar REMUS wake environment.

These serve two purposes:
  1. Sanity-check that the environment is working correctly.
  2. Provide performance references for RL agents.

Policies:
  - GoalSeekPolicy:              always point at goal, fixed throttle.
  - CrossCurrentCompensationPolicy: offset heading against lateral flow.
  - StillWaterStraightLine:      heading = straight to goal, no flow awareness.
  - WorldFrameCurrentCompensationPolicy: local current compensation in world frame.
  - PrivilegedCorridorPolicy:    route selection with privileged flow-field access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PlanarRemusEnv
from .vehicle import S


def _wrap_to_pi(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _throttle_to_action(throttle: float) -> float:
    return float(np.clip(2.0 * throttle - 1.0, -1.0, 1.0))


def _effective_vehicle_speed(env: PlanarRemusEnv, speed_scale: float = 0.9) -> float:
    target = float(getattr(env, "current_target_auv_max_speed_mps", 1.5))
    return max(0.2, speed_scale * target)


def _current_compensated_heading(
    direction_world: np.ndarray,
    current_world: np.ndarray,
    vehicle_speed_mps: float,
) -> float:
    """Return a water-relative heading that compensates the local current."""
    direction = np.asarray(direction_world, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    direction /= norm

    current = np.asarray(current_world[:2], dtype=float)
    required_water_velocity = vehicle_speed_mps * direction - current
    req_norm = float(np.linalg.norm(required_water_velocity))
    if req_norm <= 1e-8:
        required_water_velocity = direction
    else:
        required_water_velocity /= req_norm

    return float(np.arctan2(required_water_velocity[1], required_water_velocity[0]))


def _heading_to_action_offset(env: PlanarRemusEnv, desired_heading: float) -> float:
    return env.encode_heading_action(desired_heading)


def _position_xy(env: PlanarRemusEnv) -> np.ndarray:
    return np.array([env.state[S.XN], env.state[S.YE]], dtype=float)


@dataclass(slots=True)
class GoalSeekPolicy:
    """Always point directly at the goal with a fixed throttle.

    action[0] = 0  →  heading_ref = goal_bearing  (no offset)
    action[1]      →  mapped from self.throttle ∈ [0, 1] to [-1, 1]
    """

    throttle: float = 0.7

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        _ = obs
        throttle_action = _throttle_to_action(self.throttle)
        return np.array([env.encode_heading_action(env.goal_heading()), throttle_action], dtype=np.float32)


@dataclass(slots=True)
class CrossCurrentCompensationPolicy:
    """Offset heading against the lateral (body-frame v) flow component.

    Uses the first probe in the observation as the center-point flow estimate.
    Heading offset ∝ -v_flow_body, clamped to [-1, 1].
    """

    throttle: float = 0.7
    lateral_gain: float = 0.8

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        decoded = env.decode_observation(obs)
        center_probe = decoded["probes"][0]   # (3,): [u_body, v_body, omega]

        # The probe value may be normalised; lateral_gain absorbs the scale.
        v_body = float(center_probe[1])
        heading_offset = float(np.clip(-self.lateral_gain * v_body, -1.0, 1.0))
        desired_heading = env.goal_heading() + heading_offset * float(env.config.heading_offset_limit_rad)

        throttle_action = _throttle_to_action(self.throttle)
        return np.array([env.encode_heading_action(desired_heading), throttle_action], dtype=np.float32)


@dataclass(slots=True)
class StillWaterStraightLine:
    """Head straight toward the goal — identical to GoalSeek but semantically
    labelled as the 'still-water baseline' for reporting purposes.
    """

    throttle: float = 0.7

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        _ = obs
        throttle_action = _throttle_to_action(self.throttle)
        return np.array([env.encode_heading_action(env.goal_heading()), throttle_action], dtype=np.float32)


@dataclass(slots=True)
class WorldFrameCurrentCompensationPolicy:
    """Compensate the local world-frame current against desired ground-track.

    Unlike the simple lateral-flow baseline, this policy computes a desired
    ground-track in world coordinates and chooses a water-relative heading that
    approximately cancels the local current vector.
    """

    throttle: float = 0.8
    speed_scale: float = 0.9

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        _ = obs
        pos_xy = _position_xy(env)
        goal_vec = env.goal_xy - pos_xy
        current_world = env.last_equivalent_current_world[:2]
        desired_heading = _current_compensated_heading(
            direction_world=goal_vec,
            current_world=current_world,
            vehicle_speed_mps=_effective_vehicle_speed(env, self.speed_scale),
        )
        heading_offset = _heading_to_action_offset(env, desired_heading)
        throttle_action = _throttle_to_action(self.throttle)
        return np.array([heading_offset, throttle_action], dtype=np.float32)


@dataclass(slots=True)
class PrivilegedCorridorPolicy:
    """A stronger privileged benchmark with simple route selection.

    This policy is intentionally stronger than deployable baselines: it uses
    the simulator's wake field to score a small set of candidate corridors
    (direct / upper / lower) and then tracks the best route with local current
    compensation. It is useful as a benchmark upper bound for the new task
    geometries, especially cross-stream and upstream cases.
    """

    throttle: float = 0.85
    speed_scale: float = 0.9
    waypoint_tolerance_m: float = 8.0
    lane_margin_m: float = 6.0
    segment_samples: int = 12
    direct_bias: float = 0.98
    weak_flow_bias: float = 0.25
    boundary_penalty_gain: float = 0.15
    min_effective_speed_mps: float = 0.08
    upstream_throttle: float = 0.95
    cross_stream_throttle: float = 0.88

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        _ = obs
        pos_xy = _position_xy(env)
        target_xy = self._select_target(env, pos_xy)
        geometry = getattr(env, "current_task_geometry", "downstream")
        current_world = env.last_equivalent_current_world[:2]
        desired_heading = _current_compensated_heading(
            direction_world=(target_xy - pos_xy),
            current_world=current_world,
            vehicle_speed_mps=_effective_vehicle_speed(env, self.speed_scale),
        )
        heading_offset = _heading_to_action_offset(env, desired_heading)
        throttle_action = _throttle_to_action(self._throttle_for_geometry(geometry))
        return np.array([heading_offset, throttle_action], dtype=np.float32)

    def _throttle_for_geometry(self, geometry: str) -> float:
        if geometry == "upstream":
            return self.upstream_throttle
        if geometry == "cross_stream":
            return self.cross_stream_throttle
        return self.throttle

    def _select_target(self, env: PlanarRemusEnv, pos_xy: np.ndarray) -> np.ndarray:
        goal_xy = env.goal_xy.astype(float)
        geometry = getattr(env, "current_task_geometry", "downstream")
        if geometry == "downstream":
            return goal_xy

        routes = self._candidate_routes(env, pos_xy, goal_xy)
        best_route = min(routes, key=lambda route: self._route_cost(env, route))
        if len(best_route) == 1:
            return best_route[0]

        for waypoint in best_route[:-1]:
            if np.linalg.norm(waypoint - pos_xy) > self.waypoint_tolerance_m:
                return waypoint
        return best_route[-1]

    def _candidate_routes(
        self,
        env: PlanarRemusEnv,
        pos_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> list[list[np.ndarray]]:
        y_top = env.wake_field.y_max - env.config.boundary_margin_m - self.lane_margin_m
        y_bot = env.wake_field.y_min + env.config.boundary_margin_m + self.lane_margin_m
        y_mid = 0.5 * (y_top + y_bot)
        y_hi = 0.5 * (y_top + y_mid)
        y_lo = 0.5 * (y_bot + y_mid)

        geometry = getattr(env, "current_task_geometry", "downstream")
        direct = [goal_xy]

        if geometry == "cross_stream":
            x_mid = 0.5 * (pos_xy[0] + goal_xy[0])
            return [
                direct,
                [np.array([x_mid, y_top], dtype=float), goal_xy],
                [np.array([x_mid, y_hi], dtype=float), goal_xy],
                [np.array([x_mid, y_lo], dtype=float), goal_xy],
                [np.array([x_mid, y_bot], dtype=float), goal_xy],
                [
                    np.array([pos_xy[0], y_top], dtype=float),
                    np.array([goal_xy[0], y_top], dtype=float),
                    goal_xy,
                ],
                [
                    np.array([pos_xy[0], y_bot], dtype=float),
                    np.array([goal_xy[0], y_bot], dtype=float),
                    goal_xy,
                ],
            ]

        if geometry == "upstream":
            x_mid = 0.5 * (pos_xy[0] + goal_xy[0])
            return [
                direct,
                [np.array([x_mid, y_top], dtype=float), goal_xy],
                [np.array([x_mid, y_hi], dtype=float), goal_xy],
                [np.array([x_mid, y_lo], dtype=float), goal_xy],
                [np.array([x_mid, y_bot], dtype=float), goal_xy],
                [
                    np.array([pos_xy[0], y_top], dtype=float),
                    np.array([goal_xy[0], y_top], dtype=float),
                    goal_xy,
                ],
                [
                    np.array([pos_xy[0], y_bot], dtype=float),
                    np.array([goal_xy[0], y_bot], dtype=float),
                    goal_xy,
                ],
            ]

        return [direct]

    def _route_cost(self, env: PlanarRemusEnv, route: list[np.ndarray]) -> float:
        own_speed = _effective_vehicle_speed(env, self.speed_scale)
        points = [_position_xy(env)] + [np.asarray(p, dtype=float) for p in route]
        cost = 0.0
        elapsed_t = float(env.flow_time)
        y_top = env.wake_field.y_max - env.config.boundary_margin_m
        y_bot = env.wake_field.y_min + env.config.boundary_margin_m
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            segment = p1 - p0
            length = float(np.linalg.norm(segment))
            if length <= 1e-8:
                continue
            direction = segment / length
            along_flows = []
            speed_magnitudes = []
            for alpha in np.linspace(0.0, 1.0, self.segment_samples):
                xy = p0 + alpha * segment
                sample_t = elapsed_t + alpha * (length / max(own_speed, self.min_effective_speed_mps))
                sample = env.flow_sampler.sample_world(float(xy[0]), float(xy[1]), sample_t)
                along_flows.append(float(np.dot(sample[:2], direction)))
                speed_magnitudes.append(float(np.linalg.norm(sample[:2])))
            mean_along = float(np.mean(along_flows))
            mean_speed = float(np.mean(speed_magnitudes))
            effective_speed = max(self.min_effective_speed_mps, own_speed + mean_along)
            segment_cost = length / effective_speed
            weak_flow_bonus = self.weak_flow_bias * mean_speed * (length / max(1.0, own_speed))
            boundary_clearance = min(abs(p1[1] - y_bot), abs(y_top - p1[1]))
            boundary_penalty = self.boundary_penalty_gain / max(boundary_clearance, 1.0)
            segment_cost += weak_flow_bonus + boundary_penalty
            cost += segment_cost
            elapsed_t += segment_cost

        if len(route) == 1:
            cost *= self.direct_bias
        return float(cost)


# ---------------------------------------------------------------------------
# From policies/residual_prior.py
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None


@dataclass(slots=True)
class ResidualPriorConfig:
    throttle: float = 0.7
    lateral_gain: float = 0.8
    heading_cos_index: int | None = None
    heading_sin_index: int | None = None
    goal_x_index: int | None = None
    goal_y_index: int | None = None
    center_probe_v_index: int | None = None
    action_mode: str = "absolute_heading"
    heading_offset_limit_rad: float = np.deg2rad(60.0)


class CrossCurrentResidualPrior:
    """Analytic heading/throttle prior used by the residual policy."""

    def __init__(self, config: ResidualPriorConfig | None = None) -> None:
        self.config = config or ResidualPriorConfig()

    def _get_index(self, name: str) -> int:
        value = getattr(self.config, name)
        if value is None:
            raise ValueError(f"Residual prior requires `{name}` to be configured explicitly.")
        return int(value)

    def action_np(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        v_body = 0.0
        center_probe_v_index = self._get_index("center_probe_v_index")
        if obs.shape[-1] > center_probe_v_index:
            v_body = float(obs[center_probe_v_index])
        heading_offset = float(np.clip(-self.config.lateral_gain * v_body, -1.0, 1.0))
        heading_action = heading_offset
        if self.config.action_mode == "absolute_heading":
            heading_sin_index = self._get_index("heading_sin_index")
            heading_cos_index = self._get_index("heading_cos_index")
            goal_y_index = self._get_index("goal_y_index")
            goal_x_index = self._get_index("goal_x_index")
            psi = float(np.arctan2(obs[heading_sin_index], obs[heading_cos_index]))
            goal_bearing_body = float(np.arctan2(obs[goal_y_index], obs[goal_x_index]))
            desired_heading = psi + goal_bearing_body
            desired_heading += heading_offset * self.config.heading_offset_limit_rad
            desired_heading = float(np.arctan2(np.sin(desired_heading), np.cos(desired_heading)))
            heading_action = float(np.clip(desired_heading / np.pi, -1.0, 1.0))
        throttle_action = float(np.clip(2.0 * self.config.throttle - 1.0, -1.0, 1.0))
        return np.array([heading_action, throttle_action], dtype=np.float32)

    def action_tensor(self, obs: "torch.Tensor") -> "torch.Tensor":
        v_body = torch.zeros_like(obs[..., 0])
        center_probe_v_index = self._get_index("center_probe_v_index")
        if obs.shape[-1] > center_probe_v_index:
            v_body = obs[..., center_probe_v_index]
        heading_offset = torch.clamp(-self.config.lateral_gain * v_body, -1.0, 1.0)
        heading_action = heading_offset
        if self.config.action_mode == "absolute_heading":
            heading_sin_index = self._get_index("heading_sin_index")
            heading_cos_index = self._get_index("heading_cos_index")
            goal_y_index = self._get_index("goal_y_index")
            goal_x_index = self._get_index("goal_x_index")
            psi = torch.atan2(obs[..., heading_sin_index], obs[..., heading_cos_index])
            goal_bearing_body = torch.atan2(obs[..., goal_y_index], obs[..., goal_x_index])
            desired_heading = psi + goal_bearing_body
            desired_heading = desired_heading + heading_offset * self.config.heading_offset_limit_rad
            desired_heading = torch.atan2(torch.sin(desired_heading), torch.cos(desired_heading))
            heading_action = torch.clamp(desired_heading / np.pi, -1.0, 1.0)
        throttle_action = torch.full_like(heading_offset, 2.0 * self.config.throttle - 1.0)
        return torch.stack([heading_action, throttle_action], dim=-1)
