# Standard Benchmarks

This directory stores the repository's fixed evaluation manifests.

Each JSON file freezes one benchmark as a list of concrete episodes:

- `flow_time`
- `start_xy`
- `goal_xy`
- `initial_heading`
- `initial_speed`
- `task_geometry`
- `action_mode`
- `target_auv_max_speed_mps`

The manifests intentionally do not lock `probe_layout` or `history_length`, so the same task set can be reused across `s0`, `s1`, `s2`, and different temporal-context settings.

Recommended usage:

```bash
conda run -n mytorch1 python -m scripts.generate_standard_benchmarks --episodes 30
conda run -n mytorch1 python -m scripts.run_suite --preset geometry_factor_v1
conda run -n mytorch1 python -m scripts.evaluate \
    --checkpoint <checkpoint_dir> \
    --manifest benchmarks/single_u15_upstream_tgt15.json
```

The benchmark catalog itself is defined in `scripts/benchmark_catalog.py`.
