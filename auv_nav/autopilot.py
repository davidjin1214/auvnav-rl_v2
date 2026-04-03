from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .flow import FlowSampler, rotation_world_to_body
from .vehicle import (
    ActuatorState,
    DepthController,
    Remus100,
    S,
    ValidityReport,
    rk4_step,
    sat,
    ssa,
)


@dataclass(slots=True)
class HeadingAutopilotConfig:
    yaw_kp: float = 1.4
    yaw_kd: float = 0.4
    rudder_limit_rad: float = 0.0


class HeadingAutopilot:
    """Simple heading-hold inner loop for high-level navigation commands."""

    def __init__(self, config: HeadingAutopilotConfig) -> None:
        self.config = config

    def command(self, psi_ref: float, psi: float, yaw_rate: float) -> float:
        heading_error = ssa(psi_ref - psi)
        delta_r = self.config.yaw_kp * heading_error - self.config.yaw_kd * yaw_rate
        return sat(delta_r, self.config.rudder_limit_rad)


@dataclass(slots=True)
class DepthHoldAutopilotConfig:
    z_ref_m: float = 0.0
    kp_z: float = 0.08
    t_z: float = 25.0
    k_grad: float = 0.02
    kp_theta: float = 5.0
    kd_theta: float = 2.5
    ki_theta: float = 0.3
    elevator_limit_rad: float = 0.0


class DepthHoldAutopilot:
    """Wrap the REMUS depth controller as a hidden inner-loop autopilot."""

    def __init__(self, dt: float, config: DepthHoldAutopilotConfig) -> None:
        self.dt = float(dt)
        self.config = config
        self.controller = self._build_controller()
        self.last_theta_ref = 0.0

    def _build_controller(self) -> DepthController:
        return DepthController(
            h=self.dt,
            Kp_z=self.config.kp_z,
            T_z=self.config.t_z,
            k_grad=self.config.k_grad,
            Kp_theta=self.config.kp_theta,
            Kd_theta=self.config.kd_theta,
            Ki_theta=self.config.ki_theta,
        )

    def reset(self) -> None:
        self.controller = self._build_controller()
        self.last_theta_ref = 0.0

    def command(
        self,
        *,
        z_now: float,
        theta: float,
        q: float,
        u: float,
        w: float,
    ) -> float:
        delta_s, theta_ref = self.controller.update(
            z_ref=self.config.z_ref_m,
            z_now=z_now,
            theta=theta,
            q=q,
            u=u,
            w=w,
        )
        self.last_theta_ref = float(theta_ref)
        # REMUS stern-plane sign is opposite to the controller convention used
        # by the generic depth controller, so flip it here in one place.
        return sat(float(-delta_s), self.config.elevator_limit_rad)

    def diagnostics(self, *, z_now: float, phi: float, theta: float) -> dict[str, float | bool]:
        return {
            "depth_m": float(z_now),
            "depth_error_m": float(z_now - self.config.z_ref_m),
            "roll_rad": float(phi),
            "pitch_rad": float(theta),
            "theta_ref_rad": float(self.last_theta_ref),
            "depth_hold_ok": bool(np.isfinite(z_now) and np.isfinite(phi) and np.isfinite(theta)),
        }


@dataclass(slots=True)
class EquivalentCurrentSample:
    world: np.ndarray
    body: np.ndarray
    center_world: np.ndarray


class EquivalentCurrentModel:
    """Estimate a dynamics current from weighted multi-point hull samples."""

    def __init__(
        self,
        flow_sampler: FlowSampler,
        vehicle_length_m: float,
        sample_fractions: tuple[float, ...],
        sample_weights: tuple[float, ...],
    ) -> None:
        self.flow_sampler = flow_sampler
        self.offsets_body = self._build_offsets(vehicle_length_m, sample_fractions)
        self.weights = self._build_weights(sample_weights)

    @staticmethod
    def _build_offsets(vehicle_length_m: float, sample_fractions: tuple[float, ...]) -> np.ndarray:
        fractions = np.asarray(sample_fractions, dtype=float)
        if fractions.size == 0:
            raise ValueError("At least one hull flow sample fraction is required.")
        x_offsets = fractions * float(vehicle_length_m)
        return np.column_stack([x_offsets, np.zeros_like(x_offsets)])

    @staticmethod
    def _build_weights(sample_weights: tuple[float, ...]) -> np.ndarray:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.size == 0:
            raise ValueError("At least one hull flow sample weight is required.")
        if np.any(weights < 0.0):
            raise ValueError("Hull flow sample weights must be nonnegative.")
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("Hull flow sample weights must sum to a positive value.")
        return weights / weight_sum

    def sample(self, state: np.ndarray, t: float) -> EquivalentCurrentSample:
        x = float(state[S.XN])
        y = float(state[S.YE])
        psi = float(state[S.PSI])
        hull_world_samples = self.flow_sampler.sample_points_world(
            x=x,
            y=y,
            psi=psi,
            t=t,
            offsets_body=self.offsets_body,
        )
        equivalent_world = np.sum(
            self.weights[:, np.newaxis] * hull_world_samples,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        center_world = self.flow_sampler.sample_world(x, y, t).astype(np.float32)
        rot_wb = rotation_world_to_body(psi)
        equivalent_body_uv = rot_wb @ equivalent_world[:2]
        equivalent_body = np.array(
            [equivalent_body_uv[0], equivalent_body_uv[1], equivalent_world[2]],
            dtype=np.float32,
        )
        return EquivalentCurrentSample(
            world=equivalent_world,
            body=equivalent_body,
            center_world=center_world,
        )


@dataclass(slots=True)
class DepthHold6DOFBackendConfig:
    sim_dt: float
    z_ref_m: float = 0.0
    depth_fail_tol_m: float = 1.5
    roll_fail_tol_rad: float = np.deg2rad(45.0)
    pitch_fail_tol_rad: float = np.deg2rad(35.0)
    max_body_speed_mps: float = 3.5
    max_relative_speed_mps: float = 4.5
    max_angular_rate_radps: float = 5.0
    max_state_derivative: float = 250.0


class DepthHold6DOFBackend:
    """Closed-loop 6DOF backend with hidden heading/depth autopilots."""

    def __init__(
        self,
        vehicle: Remus100,
        equivalent_current_model: EquivalentCurrentModel,
        heading_config: HeadingAutopilotConfig,
        depth_config: DepthHoldAutopilotConfig,
        backend_config: DepthHold6DOFBackendConfig,
    ) -> None:
        self.vehicle = vehicle
        self.current_model = equivalent_current_model
        self.heading_autopilot = HeadingAutopilot(heading_config)
        self.depth_autopilot = DepthHoldAutopilot(backend_config.sim_dt, depth_config)
        self.config = backend_config
        self.state = np.zeros(S.N, dtype=float)
        self.actuator_state = ActuatorState()
        self.last_current = EquivalentCurrentSample(
            world=np.zeros(3, dtype=np.float32),
            body=np.zeros(3, dtype=np.float32),
            center_world=np.zeros(3, dtype=np.float32),
        )
        self.last_delta_r_cmd = 0.0
        self.last_delta_s_cmd = 0.0

    def reset(self, *, initial_xy: np.ndarray, initial_heading: float, initial_speed: float) -> None:
        self.state = np.zeros(S.N, dtype=float)
        self.state[S.XN] = float(initial_xy[0])
        self.state[S.YE] = float(initial_xy[1])
        self.state[S.ZD] = float(self.config.z_ref_m)
        self.state[S.PSI] = float(initial_heading)
        self.state[S.U] = float(initial_speed)
        self.actuator_state = ActuatorState()
        self.depth_autopilot.reset()
        self.last_delta_r_cmd = 0.0
        self.last_delta_s_cmd = 0.0

    def substep(self, *, psi_ref: float, rpm_cmd: float, flow_time: float) -> None:
        current = self.current_model.sample(self.state, flow_time)
        self.last_current = current
        delta_r_cmd = self.heading_autopilot.command(
            psi_ref=psi_ref,
            psi=float(self.state[S.PSI]),
            yaw_rate=float(self.state[S.R]),
        )
        delta_s_cmd = self.depth_autopilot.command(
            z_now=float(self.state[S.ZD]),
            theta=float(self.state[S.THETA]),
            q=float(self.state[S.Q]),
            u=float(self.state[S.U]),
            w=float(self.state[S.W]),
        )
        self.last_delta_r_cmd = float(delta_r_cmd)
        self.last_delta_s_cmd = float(delta_s_cmd)
        cmd = np.array([delta_r_cmd, delta_s_cmd, rpm_cmd], dtype=float)
        self.actuator_state = self.vehicle.step_actuators(
            cmd,
            self.actuator_state,
            self.config.sim_dt,
        )

        current_world = current.world
        v_c = float(np.hypot(current_world[0], current_world[1]))
        beta_c = float(np.arctan2(current_world[1], current_world[0]))
        self.state = rk4_step(
            self.vehicle,
            self.state,
            self.actuator_state.as_array(),
            self.config.sim_dt,
            Vc=v_c,
            beta_Vc=beta_c,
            w_c=0.0,
        )

    def validity(self) -> ValidityReport:
        flow = self.vehicle.compute_relative_flow(
            self.state,
            Vc=float(np.hypot(self.last_current.world[0], self.last_current.world[1])),
            beta_Vc=float(np.arctan2(self.last_current.world[1], self.last_current.world[0])),
            w_c=0.0,
        )
        validity = self.vehicle.state_validity_check(
            self.state,
            flow=flow,
            max_speed=self.config.max_body_speed_mps,
            max_rate=self.config.max_angular_rate_radps,
            max_derivative=self.config.max_state_derivative,
        )
        if not validity.ok:
            return validity
        if flow.U_r > self.config.max_relative_speed_mps:
            return ValidityReport(False, "relative_speed_limit")

        depth_error = abs(float(self.state[S.ZD] - self.config.z_ref_m))
        if depth_error > self.config.depth_fail_tol_m:
            return ValidityReport(False, "depth_hold_failure")
        if abs(float(self.state[S.PHI])) > self.config.roll_fail_tol_rad:
            return ValidityReport(False, "depth_hold_failure")
        if abs(float(self.state[S.THETA])) > self.config.pitch_fail_tol_rad:
            return ValidityReport(False, "depth_hold_failure")
        return ValidityReport(True, "ok")

    def planar_navigation_state(self) -> tuple[float, float, float]:
        return (
            float(self.state[S.XN]),
            float(self.state[S.YE]),
            float(self.state[S.PSI]),
        )

    def info_state(self) -> dict[str, Any]:
        depth_diag = self.depth_autopilot.diagnostics(
            z_now=float(self.state[S.ZD]),
            phi=float(self.state[S.PHI]),
            theta=float(self.state[S.THETA]),
        )
        return {
            "state": self.state.copy(),
            "actuator_state": self.actuator_state,
            "equivalent_current_world": self.last_current.world.copy(),
            "equivalent_current_body": self.last_current.body.copy(),
            "center_current_world": self.last_current.center_world.copy(),
            "autopilot_delta_r_cmd": float(self.last_delta_r_cmd),
            "autopilot_delta_s_cmd": float(self.last_delta_s_cmd),
            **depth_diag,
        }
