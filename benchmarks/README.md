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

## `clean_probe/`

Manifests for the item-(1) supplementary final evaluation (`docs/data_integrity_open_items.md`),
reported in Chapter 5 §5.7.1 / §5.7.m.

- `single_u10_cross_tgt15_ep100_s3000.json` — 100 episodes, manifest seeds 3000..3099. Disjoint
  from both the training seed range 0..1999 and the published test set 1250..1349, with every
  other generation parameter matching the main evaluation manifest.
- `_repro_check_s1250.json` — the evidence for that last clause. Regenerated with the *same*
  generator and parameters but manifest seed 1250, it reproduces
  `single_u10_cross_tgt15_ep100.json` episode for episode, so the s3000 manifest differs from the
  main one only in the seed. Not an evaluation target; keep it out of result protocols.

Note for `python -m scripts.audit_seed_overlap`: its range pass treats every JSON with an
`episodes` key as a manifest, so `_repro_check_s1250.json` shows up as a third overlap line
against the 2000-episode dataset. That line is the same instance family as
`single_u10_cross_tgt15_ep100.json`, not an additional contaminated evaluation set.
