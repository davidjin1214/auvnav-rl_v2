from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class WakeFieldMetadata:
    channels: tuple[str, ...]
    data_shape: tuple[int, int, int, int]
    dx_m: float
    record_interval_s: float
    roi_x0_phys_m: float
    roi_x1_phys_m: float
    roi_y0_phys_m: float
    roi_y1_phys_m: float
    total_frames: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WakeFieldMetadata":
        return cls(
            channels=tuple(data["channels"]),
            data_shape=tuple(int(v) for v in data["data_shape"]),
            dx_m=float(data["dx_m"]),
            record_interval_s=float(data["record_interval_actual_s"]),
            roi_x0_phys_m=float(data["roi_x0_phys_m"]),
            roi_x1_phys_m=float(data["roi_x1_phys_m"]),
            roi_y0_phys_m=float(data["roi_y0_phys_m"]),
            roi_y1_phys_m=float(data["roi_y1_phys_m"]),
            total_frames=int(data["total_frames"]),
        )


class WakeField:
    """Memory-mapped wake-field dataset with physical-coordinate helpers."""

    def __init__(self, flow: np.ndarray, metadata: WakeFieldMetadata) -> None:
        if flow.ndim != 4:
            raise ValueError("Wake-field array must have shape (T, Nx, Ny, C).")
        if tuple(flow.shape) != metadata.data_shape:
            raise ValueError(
                f"Flow shape {tuple(flow.shape)} does not match metadata {metadata.data_shape}."
            )
        self.flow = flow
        self.meta = metadata
        self.channels = metadata.channels
        self.dt = metadata.record_interval_s
        self.dx = metadata.dx_m
        self.total_frames = metadata.total_frames
        self.x_coords = metadata.roi_x0_phys_m + self.dx * np.arange(flow.shape[1], dtype=float)
        self.y_coords = metadata.roi_y0_phys_m + self.dx * np.arange(flow.shape[2], dtype=float)

    @classmethod
    def from_files(
        cls,
        flow_path: str | Path,
        meta_path: str | Path | None = None,
        mmap_mode: str = "r",
    ) -> "WakeField":
        flow_path = Path(flow_path)
        if meta_path is None:
            meta_path = flow_path.with_name(flow_path.stem + "_meta.json")
        meta_path = Path(meta_path)

        flow = np.load(flow_path, mmap_mode=mmap_mode)
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = WakeFieldMetadata.from_dict(json.load(fp))
        return cls(flow=flow, metadata=meta)

    @property
    def x_min(self) -> float:
        return float(self.x_coords[0])

    @property
    def x_max(self) -> float:
        return float(self.x_coords[-1])

    @property
    def y_min(self) -> float:
        return float(self.y_coords[0])

    @property
    def y_max(self) -> float:
        return float(self.y_coords[-1])

    @property
    def time_horizon(self) -> float:
        return float(max(0, self.total_frames - 1) * self.dt)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def clamp_position(self, x: float, y: float) -> tuple[float, float]:
        x_clamped = float(np.clip(x, self.x_min, self.x_max))
        y_clamped = float(np.clip(y, self.y_min, self.y_max))
        return x_clamped, y_clamped


def rotation_world_to_body(psi: float) -> np.ndarray:
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    return np.array([[cpsi, spsi], [-spsi, cpsi]], dtype=float)


def rotation_body_to_world(psi: float) -> np.ndarray:
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    return np.array([[cpsi, -spsi], [spsi, cpsi]], dtype=float)


def default_probe_offsets() -> np.ndarray:
    """Sparse, body-frame probe layout; center probe is always first."""
    return np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
            [2.0, 2.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )


def make_probe_offsets(layout: str) -> np.ndarray:
    """Return body-frame probe offsets (m) for the named sensing scheme.

    All coordinates are in the AUV body frame: x points forward (bow),
    y points to port.  Only velocity channels (u, v) are assumed; no
    vorticity is required.

    Physical basis
    --------------
    All three layouts are derived from sensors physically realizable on a
    REMUS-100 class AUV in the horizontal navigation plane.

    A standard downward-looking DVL (1200 kHz, Janus 4-beam) in water-tracking
    mode samples the water layer directly below the AUV; in 2-D horizontal
    navigation all four beams converge to the same horizontal position, giving
    a single equivalent (u, v) measurement at the vehicle centre (S0).

    A compact forward-looking ADCP (e.g. Nortek Aquadopp 1 or 2 MHz) mounted
    in the nose payload bay extends sensing ahead of the vehicle.  Each beam
    measures only its radial (along-beam) velocity component; at least three
    non-parallel beams at a given range cell are combined to reconstruct the
    full (u, v) vector at that cell.  The probe positions below represent those
    reconstructed vector locations, not individual beam hit-points.

    Schemes
    -------
    s0 — 1 probe  — DVL water-track (standard REMUS-100 baseline)
         Single (u, v) at vehicle centre.  Purely reactive sensing.
         d_obs = 8 + 1 × 2 = 10

    s1 — 2 probes — short-range forward ADCP  (≈ 2 MHz, ~5 m range)
         DVL centre + one near-field reconstructed vector at 4.5 m ahead.
         Provides ~3 control steps of advance warning at 1.5 m/s.
         No lateral beam spread at this short range (< 2 m lateral offset).
         d_obs = 8 + 2 × 2 = 12

    s2 — 4 probes — long-range forward ADCP  (1 MHz, ~9 m range)
         DVL centre + near forward cell at 5 m + two lateral beam cells at
         [8 m, ±4 m] (≈ 27° half-angle, consistent with Nortek Aquadopp
         1 MHz beam geometry at 9 m range).
         Near cell gives axial preview; lateral pair gives cross-stream
         gradient for locating the AUV relative to the vortex core.
         Provides ~3–7 control steps of advance warning.
         d_obs = 8 + 4 × 2 = 16
    """
    if layout == "s0":
        return np.array([[0.0, 0.0]], dtype=float)
    if layout == "s1":
        return np.array(
            [
                [0.0, 0.0],   # centre — DVL water-track
                [4.5, 0.0],   # near forward cell — short-range ADCP (~2 MHz)
            ],
            dtype=float,
        )
    if layout == "s2":
        return np.array(
            [
                [0.0, 0.0],   # centre — DVL water-track
                [5.0, 0.0],   # near forward cell (~1.7 s preview at 1.5 m/s)
                [8.0,  4.0],  # port  lateral beam, 27° half-angle at ~9 m range
                [8.0, -4.0],  # stbd  lateral beam
            ],
            dtype=float,
        )
    raise ValueError(
        f"Unknown probe layout {layout!r}. Choose from 's0', 's1', 's2'."
    )


@dataclass(slots=True)
class FlowSample:
    world: np.ndarray
    body: np.ndarray


class FlowSampler:
    """Space-time interpolation of the wake field at arbitrary agent poses."""

    def __init__(self, wake_field: WakeField, loop_time: bool = True) -> None:
        self.field = wake_field
        self.loop_time = loop_time

    def _time_indices(self, t: float) -> tuple[int, int, float]:
        if self.field.total_frames == 1:
            return 0, 0, 0.0

        tau = max(0.0, float(t)) / self.field.dt
        if self.loop_time:
            tau %= self.field.total_frames
        else:
            tau = min(tau, self.field.total_frames - 1)

        i0 = int(np.floor(tau))
        alpha_t = float(tau - i0)
        if self.loop_time:
            i1 = (i0 + 1) % self.field.total_frames
        else:
            i1 = min(i0 + 1, self.field.total_frames - 1)
        return i0, i1, alpha_t

    def _spatial_indices(self, x: float, y: float) -> tuple[int, int, float, float]:
        x, y = self.field.clamp_position(x, y)

        if self.field.flow.shape[1] == 1:
            ix = 0
            alpha_x = 0.0
        else:
            fx = (x - self.field.x_min) / self.field.dx
            ix = int(np.clip(np.floor(fx), 0, self.field.flow.shape[1] - 2))
            alpha_x = float(fx - ix)

        if self.field.flow.shape[2] == 1:
            iy = 0
            alpha_y = 0.0
        else:
            fy = (y - self.field.y_min) / self.field.dx
            iy = int(np.clip(np.floor(fy), 0, self.field.flow.shape[2] - 2))
            alpha_y = float(fy - iy)

        return ix, iy, alpha_x, alpha_y

    @staticmethod
    def _bilinear(frame: np.ndarray, ix: int, iy: int, ax: float, ay: float) -> np.ndarray:
        c00 = frame[ix, iy]
        c10 = frame[ix + 1, iy]
        c01 = frame[ix, iy + 1]
        c11 = frame[ix + 1, iy + 1]
        return (
            (1.0 - ax) * (1.0 - ay) * c00
            + ax * (1.0 - ay) * c10
            + (1.0 - ax) * ay * c01
            + ax * ay * c11
        )

    def sample_world(self, x: float, y: float, t: float) -> np.ndarray:
        i0, i1, alpha_t = self._time_indices(t)
        ix, iy, alpha_x, alpha_y = self._spatial_indices(x, y)

        frame0 = self.field.flow[i0]
        sample0 = self._bilinear(frame0, ix, iy, alpha_x, alpha_y)
        if i0 == i1:
            return np.asarray(sample0, dtype=np.float32)

        frame1 = self.field.flow[i1]
        sample1 = self._bilinear(frame1, ix, iy, alpha_x, alpha_y)
        return np.asarray((1.0 - alpha_t) * sample0 + alpha_t * sample1, dtype=np.float32)

    def sample_points_world(
        self,
        x: float,
        y: float,
        psi: float,
        t: float,
        offsets_body: Array,
    ) -> np.ndarray:
        offsets = np.asarray(offsets_body, dtype=float)
        rot_bw = rotation_body_to_world(psi)
        samples = np.zeros((offsets.shape[0], 3), dtype=np.float32)
        for idx, offset in enumerate(offsets):
            world_xy = np.array([x, y], dtype=float) + rot_bw @ offset
            samples[idx] = self.sample_world(float(world_xy[0]), float(world_xy[1]), t)
        return samples

    def sample_body(self, x: float, y: float, psi: float, t: float) -> FlowSample:
        world = self.sample_world(x, y, t)
        rot = rotation_world_to_body(psi)
        body_uv = rot @ world[:2]
        body = np.array([body_uv[0], body_uv[1], world[2]], dtype=np.float32)
        return FlowSample(world=np.asarray(world, dtype=np.float32), body=body)

    def sample_probes_body(
        self,
        x: float,
        y: float,
        psi: float,
        t: float,
        offsets_body: Array,
    ) -> np.ndarray:
        world_samples = self.sample_points_world(x, y, psi, t, offsets_body)
        rot_wb = rotation_world_to_body(psi)
        samples = np.zeros_like(world_samples, dtype=np.float32)
        for idx, sample in enumerate(world_samples):
            body_uv = rot_wb @ sample[:2]
            samples[idx] = np.array([body_uv[0], body_uv[1], sample[2]], dtype=np.float32)
        return samples


@dataclass(slots=True)
class ReferenceFlowConfig:
    grid_points: int
    time_points: int
    speed_quantile: float


@dataclass(slots=True)
class ReferenceFlowSample:
    world: np.ndarray
    speed_mps: float
    heading_rad: float


class ReferenceFlowEstimator:
    """Estimate a task-level reference flow from sparse space-time averaging."""

    def __init__(
        self,
        wake_field: WakeField,
        flow_sampler: FlowSampler,
        config: ReferenceFlowConfig,
    ) -> None:
        self.wake_field = wake_field
        self.flow_sampler = flow_sampler
        self.config = config

    def estimate(self, center_time: float | None = None) -> ReferenceFlowSample:
        wf = self.wake_field
        cfg = self.config
        xs = np.linspace(wf.x_min, wf.x_max, cfg.grid_points, dtype=float)
        ys = np.linspace(wf.y_min, wf.y_max, cfg.grid_points, dtype=float)
        ts = self._sample_times(center_time)

        mean_uv = np.zeros(2, dtype=np.float64)
        speeds: list[float] = []
        count = 0
        for t in ts:
            for x in xs:
                for y in ys:
                    sample = self.flow_sampler.sample_world(float(x), float(y), float(t))
                    uv = np.asarray(sample[:2], dtype=np.float64)
                    mean_uv += uv
                    speeds.append(float(np.linalg.norm(uv)))
                    count += 1

        if count == 0:
            return ReferenceFlowSample(
                world=np.array([1.0, 0.0], dtype=np.float32),
                speed_mps=1.0,
                heading_rad=0.0,
            )

        mean_uv /= float(count)
        if float(np.linalg.norm(mean_uv)) <= 1e-8:
            mean_uv = np.array([1.0, 0.0], dtype=np.float64)
        return ReferenceFlowSample(
            world=mean_uv.astype(np.float32),
            speed_mps=float(
                np.quantile(np.asarray(speeds, dtype=np.float64), cfg.speed_quantile)
            ),
            heading_rad=float(np.arctan2(mean_uv[1], mean_uv[0])),
        )

    def _sample_times(self, center_time: float | None) -> np.ndarray:
        if center_time is not None:
            return np.full(max(1, self.config.time_points), float(center_time), dtype=float)
        if self.wake_field.time_horizon > 0.0:
            return np.linspace(
                0.0,
                self.wake_field.time_horizon,
                self.config.time_points,
                dtype=float,
            )
        return np.zeros(1, dtype=float)
