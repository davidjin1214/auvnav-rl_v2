from __future__ import annotations

from scripts.train_sac import recommended_num_envs


def test_recommended_num_envs_prefers_physical_core_count() -> None:
    assert recommended_num_envs(logical_cpus=12, physical_cpus=6) == 6


def test_recommended_num_envs_caps_large_machines() -> None:
    assert recommended_num_envs(logical_cpus=32, physical_cpus=16) == 8


def test_recommended_num_envs_falls_back_to_half_logical_threads() -> None:
    assert recommended_num_envs(logical_cpus=12, physical_cpus=None) == 6
