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

## Per-study split manifests (`offline_rebrac_*/`, `c1_reward_ablation/`)

The val/test splits the offline studies actually evaluated on. They were generated on Colab and
lived only on the Drive copy until 2026-08-21, when the repo-wide contamination enumeration was
closed against them — a closure that could not be reproduced from a clone while they were
missing, since `python -m scripts.audit_seed_overlap` would have swept 11 manifests instead of
24 and said nothing about the difference. Copied off Drive rather than regenerated: `auv_nav/env.py`
changed after the benchmark protocol was frozen in `9b96a7d`, so rerunning the generator is not
guaranteed to reproduce them, and these are evaluation records. Their content is untouched; line
endings follow the repository's convention like every other manifest (stored LF, `core.autocrlf`
renders them CRLF in a Windows working tree), so compare against Drive on a Linux checkout.

Seed ranges, verified against that audit's own printout:

| Directory | Episodes | Manifest seeds |
|---|---|---|
| `offline_rebrac_{broad,screen,worldcomp_final}/test_100/single_u10_cross_tgt15.json` | 100 | 1250..1349 |
| `offline_rebrac_{broad,screen,worldcomp_final}/val_40/single_u10_cross_tgt15.json` | 40 | 1250..1289 |
| `offline_rebrac_worldcomp_epoch_probe/{test_40,val_40}/single_u10_cross_tgt15.json` | 40 | 1250..1289 |
| `offline_rebrac_broad/{test_100,val_40}/single_u10_upstream_tgt15.json` | 100 / 40 | 1400..1499 / 1400..1439 |
| `c1_reward_ablation/{test_100,val_40}/single_u10_upstream_tgt15.json` | 100 / 40 | 1400..1499 / 1400..1439 |
| `single_u15_cross_tgt15_ep100.json` | 100 | 1200..1299 |

Two things these files make checkable that were previously only assertions:

- **Every `val_40` is a prefix of its sibling `test_100`** — the checkpoint-selection episodes are
  a subset of the reported test episodes. This is item (3) of `docs/data_integrity_open_items.md`,
  disclosed in Chapter 5 §5.3.6.
- **`offline_rebrac_worldcomp_epoch_probe/test_40` and `val_40` are the same 40 episodes**, the two
  files differing only in a `created_at` six seconds apart. For that unit "test" and "validation"
  are one set, not merely overlapping ones.

Which table drew on which of these files is not recorded in `results/offline/**`; the readouts
there do not name their manifest.
