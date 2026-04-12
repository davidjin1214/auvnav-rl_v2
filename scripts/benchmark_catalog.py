from __future__ import annotations

from dataclasses import dataclass


FLOW_SINGLE_U10 = "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
FLOW_SINGLE_U15 = "wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
FLOW_SBS_U15 = "wake_data/wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
FLOW_TANDEM_U15 = "wake_data/wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    key: str
    description: str
    flow_path: str
    task_geometry: str
    target_speed: float
    manifest_seed: int
    action_mode: str = "auto"
    tags: tuple[str, ...] = ()
    factor_values: dict[str, object] | None = None

    def reset_options(self) -> dict[str, object]:
        return {
            "task_geometry": self.task_geometry,
            "action_mode": self.action_mode,
            "target_auv_max_speed_mps": float(self.target_speed),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkGroup:
    key: str
    description: str
    benchmarks: tuple[str, ...]


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "single_u15_downstream_tgt15": BenchmarkSpec(
        key="single_u15_downstream_tgt15",
        description="Single-cylinder benchmark with matched flow/vehicle speed, downstream task geometry.",
        flow_path=FLOW_SINGLE_U15,
        task_geometry="downstream",
        target_speed=1.5,
        manifest_seed=1100,
        tags=("geometry", "single", "u15", "target15"),
        factor_values={
            "topology": "single",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "downstream",
            "target_speed_mps": 1.5,
        },
    ),
    "single_u15_cross_tgt15": BenchmarkSpec(
        key="single_u15_cross_tgt15",
        description="Single-cylinder benchmark with matched flow/vehicle speed, cross-stream task geometry.",
        flow_path=FLOW_SINGLE_U15,
        task_geometry="cross_stream",
        target_speed=1.5,
        manifest_seed=1200,
        tags=("geometry", "single", "u15", "target15"),
        factor_values={
            "topology": "single",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "cross_stream",
            "target_speed_mps": 1.5,
        },
    ),
    "single_u15_upstream_tgt15": BenchmarkSpec(
        key="single_u15_upstream_tgt15",
        description="Single-cylinder benchmark in the critical matched-speed upstream regime.",
        flow_path=FLOW_SINGLE_U15,
        task_geometry="upstream",
        target_speed=1.5,
        manifest_seed=1300,
        tags=("core", "geometry", "flow", "topology", "speed", "single", "u15", "target15"),
        factor_values={
            "topology": "single",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "upstream",
            "target_speed_mps": 1.5,
        },
    ),
    "single_u10_upstream_tgt15": BenchmarkSpec(
        key="single_u10_upstream_tgt15",
        description="Single-cylinder upstream benchmark with weaker incoming flow and higher actuation margin.",
        flow_path=FLOW_SINGLE_U10,
        task_geometry="upstream",
        target_speed=1.5,
        manifest_seed=1400,
        tags=("core", "flow", "single", "u10", "target15"),
        factor_values={
            "topology": "single",
            "flow_speed_mps": 1.0,
            "reynolds": 150,
            "task_geometry": "upstream",
            "target_speed_mps": 1.5,
        },
    ),
    "sbs_u15_upstream_tgt15": BenchmarkSpec(
        key="sbs_u15_upstream_tgt15",
        description="Side-by-side double-cylinder upstream benchmark at matched speed.",
        flow_path=FLOW_SBS_U15,
        task_geometry="upstream",
        target_speed=1.5,
        manifest_seed=1500,
        tags=("core", "topology", "sbs", "u15", "target15"),
        factor_values={
            "topology": "side_by_side",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "upstream",
            "target_speed_mps": 1.5,
        },
    ),
    "tandem_u15_upstream_tgt15": BenchmarkSpec(
        key="tandem_u15_upstream_tgt15",
        description="Tandem double-cylinder upstream benchmark at matched speed.",
        flow_path=FLOW_TANDEM_U15,
        task_geometry="upstream",
        target_speed=1.5,
        manifest_seed=1600,
        tags=("core", "topology", "tandem", "u15", "target15"),
        factor_values={
            "topology": "tandem",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "upstream",
            "target_speed_mps": 1.5,
        },
    ),
    "single_u15_upstream_tgt20": BenchmarkSpec(
        key="single_u15_upstream_tgt20",
        description="Single-cylinder upstream benchmark with stronger propulsion than free-stream velocity.",
        flow_path=FLOW_SINGLE_U15,
        task_geometry="upstream",
        target_speed=2.0,
        manifest_seed=1700,
        tags=("speed", "single", "u15", "target20"),
        factor_values={
            "topology": "single",
            "flow_speed_mps": 1.5,
            "reynolds": 250,
            "task_geometry": "upstream",
            "target_speed_mps": 2.0,
        },
    ),
}


BENCHMARK_GROUPS: dict[str, BenchmarkGroup] = {
    "geometry_factor_v1": BenchmarkGroup(
        key="geometry_factor_v1",
        description="Single-cylinder matched-speed study with task geometry as the only varying factor.",
        benchmarks=(
            "single_u15_downstream_tgt15",
            "single_u15_cross_tgt15",
            "single_u15_upstream_tgt15",
        ),
    ),
    "flow_factor_v1": BenchmarkGroup(
        key="flow_factor_v1",
        description="Single-cylinder upstream study varying only incoming free-stream speed.",
        benchmarks=(
            "single_u10_upstream_tgt15",
            "single_u15_upstream_tgt15",
        ),
    ),
    "topology_factor_v1": BenchmarkGroup(
        key="topology_factor_v1",
        description="Matched-speed upstream study varying wake topology across single, side-by-side, and tandem cylinders.",
        benchmarks=(
            "single_u15_upstream_tgt15",
            "sbs_u15_upstream_tgt15",
            "tandem_u15_upstream_tgt15",
        ),
    ),
    "speed_factor_v1": BenchmarkGroup(
        key="speed_factor_v1",
        description="Single-cylinder upstream study varying only the vehicle max-speed setting.",
        benchmarks=(
            "single_u15_upstream_tgt15",
            "single_u15_upstream_tgt20",
        ),
    ),
    "study_core_v1": BenchmarkGroup(
        key="study_core_v1",
        description="Recommended core suite covering geometry difficulty, flow strength, and topology transfer.",
        benchmarks=(
            "single_u15_upstream_tgt15",
            "single_u10_upstream_tgt15",
            "sbs_u15_upstream_tgt15",
            "tandem_u15_upstream_tgt15",
        ),
    ),
}


def parse_benchmark_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_benchmark_specs(
    benchmarks: str | None = None,
    benchmark_group: str | None = None,
) -> list[BenchmarkSpec]:
    resolved: list[str] = []
    if benchmark_group is not None:
        if benchmark_group not in BENCHMARK_GROUPS:
            raise ValueError(f"Unknown benchmark group: {benchmark_group}")
        resolved.extend(BENCHMARK_GROUPS[benchmark_group].benchmarks)
    if benchmarks is not None:
        resolved.extend(parse_benchmark_list(benchmarks))
    if not resolved:
        return []
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for key in resolved:
        if key not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown benchmark key: {key}")
        if key in seen:
            continue
        ordered_unique.append(key)
        seen.add(key)
    return [BENCHMARK_SPECS[key] for key in ordered_unique]


def default_manifest_path(
    benchmark: BenchmarkSpec,
    manifest_dir: str = "benchmarks",
) -> str:
    return f"{manifest_dir.rstrip('/')}/{benchmark.key}.json"
