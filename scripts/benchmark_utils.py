from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class BenchmarkEpisode:
    episode_id: str
    seed: int
    reset_options: dict[str, Any]


@dataclass(slots=True)
class BenchmarkManifest:
    schema_version: int
    created_at: str
    flow_path: str
    probe_layout: str | None
    history_length: int | None
    base_reset_options: dict[str, Any]
    episodes: list[BenchmarkEpisode]
    benchmark_id: str | None = None
    benchmark_group: str | None = None
    factor_values: dict[str, Any] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["episodes"] = [asdict(ep) for ep in self.episodes]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkManifest":
        episodes = [BenchmarkEpisode(**item) for item in data["episodes"]]
        return cls(
            schema_version=int(data["schema_version"]),
            created_at=str(data["created_at"]),
            flow_path=str(data["flow_path"]),
            probe_layout=(
                str(data["probe_layout"])
                if data.get("probe_layout") is not None
                else None
            ),
            history_length=(
                int(data["history_length"])
                if data.get("history_length") is not None
                else None
            ),
            base_reset_options=dict(data.get("base_reset_options", {})),
            episodes=episodes,
            benchmark_id=(
                str(data["benchmark_id"])
                if data.get("benchmark_id") is not None
                else None
            ),
            benchmark_group=(
                str(data["benchmark_group"])
                if data.get("benchmark_group") is not None
                else None
            ),
            factor_values=(
                dict(data["factor_values"])
                if data.get("factor_values") is not None
                else None
            ),
            notes=data.get("notes"),
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_benchmark_manifest(path: str | Path, manifest: BenchmarkManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_json_ready(manifest.to_dict()), fp, indent=2)


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    with Path(path).open("r", encoding="utf-8") as fp:
        return BenchmarkManifest.from_dict(json.load(fp))


def build_benchmark_manifest(
    *,
    flow_path: str | Path,
    probe_layout: str | None,
    history_length: int | None,
    base_reset_options: dict[str, Any],
    episodes: list[BenchmarkEpisode],
    benchmark_id: str | None = None,
    benchmark_group: str | None = None,
    factor_values: dict[str, Any] | None = None,
    notes: str | None = None,
) -> BenchmarkManifest:
    return BenchmarkManifest(
        schema_version=2,
        created_at=datetime.now().isoformat(timespec="seconds"),
        flow_path=str(flow_path),
        probe_layout=probe_layout,
        history_length=(int(history_length) if history_length is not None else None),
        base_reset_options=dict(base_reset_options),
        episodes=episodes,
        benchmark_id=benchmark_id,
        benchmark_group=benchmark_group,
        factor_values=(dict(factor_values) if factor_values is not None else None),
        notes=notes,
    )
