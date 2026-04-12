from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark_catalog import BENCHMARK_GROUPS, BENCHMARK_SPECS, resolve_benchmark_specs
from scripts.run_suite import METHOD_SPECS, build_command


def test_resolve_benchmark_group_preserves_order() -> None:
    specs = resolve_benchmark_specs(benchmark_group="geometry_factor_v1")
    assert [spec.key for spec in specs] == list(BENCHMARK_GROUPS["geometry_factor_v1"].benchmarks)


def test_build_command_in_benchmark_mode_uses_manifest_and_explicit_factors(tmp_path: Path) -> None:
    args = argparse.Namespace(
        flow=None,
        difficulty=None,
        task_geometry=None,
        action_mode=None,
        speed_ratio=None,
        target_speed=None,
        eval_manifest=None,
        total_steps=1000,
        random_steps=100,
        update_after=100,
        update_every=1,
        updates_per_step=1,
        eval_every=500,
        eval_episodes=10,
        log_every_episodes=5,
        device="cpu",
        checkpoint_every=500,
        batch_size=256,
        replay_capacity=100000,
        hidden_dim=128,
        offline_data=None,
        offline_ratio=None,
        benchmark_manifest_dir="benchmarks",
    )
    benchmark = BENCHMARK_SPECS["single_u15_upstream_tgt15"]
    cmd = build_command(
        method=METHOD_SPECS["sac"],
        seed=7,
        save_dir=tmp_path / "run",
        cli_args=args,
        benchmark=benchmark,
    )

    assert "--flow" in cmd
    assert benchmark.flow_path in cmd
    assert "--task-geometry" in cmd
    assert benchmark.task_geometry in cmd
    assert "--target-speed" in cmd
    assert str(benchmark.target_speed) in cmd
    assert "--eval-manifest" in cmd
    assert f"benchmarks/{benchmark.key}.json" in cmd
    assert "--difficulty" not in cmd
