---
name: rl-v2-commands
description: CLI invocation reference for rl_v2 — training, evaluation, offline data collection, RLPD, and sweep-launcher commands with the project's exact flag conventions (probe-layout, task-geometry, target-speed, eval-manifest paths, num_envs protocol). Use when running, writing, or debugging a `scripts.*` command or a sweep launcher in this repo.
---

# rl_v2 command reference

All scripts are run as modules from the repo root (`python -m scripts.<name>`).

```bash
# Basic SAC training (uses synthetic wake data if no flow file found)
python -m scripts.train_sac

# Training with key options (A0 sensor-screen / SAC collector defaults)
python -m scripts.train_sac \
  --total-steps 600000 \
  --batch-size 256 \
  --device cuda \
  --seed 46 \
  --task-geometry upstream \
  --target-speed 1.5 \
  --probe-layout s0 \
  --history-length 4 \
  --num-envs 6 \
  --eval-every 10000 \
  --eval-manifest benchmarks/single_u15_upstream_tgt15.json \
  --eval-episodes 30 \
  --objective efficiency_v2 \
  --save-dir experiments/<study>/<cell>/seed_46

# Asymmetric Critic + LayerNorm + UTD=4 (ablation lever; still used by offline line)
python -m scripts.train_sac \
  --use-asymmetric-critic \
  --use-layernorm \
  --updates-per-step 4 \
  ... # other args as above

# Resume from checkpoint
python -m scripts.train_sac --resume <save-dir>

# Evaluate a saved checkpoint against a benchmark manifest
python -m scripts.evaluate --checkpoint <save-dir> \
  --eval-manifest benchmarks/tandem_u15_upstream_tgt15.json \
  --episodes 30

# Run baseline policies (goal-seek, current-compensation, etc.)
python -m scripts.demo --policy all --episodes 10

# Generate synthetic wake field data
python -m scripts.generate_wake

# Generate (or refresh) a fixed evaluation manifest
python -m scripts.generate_standard_benchmarks \
  --benchmarks single_u15_upstream_tgt15 --episodes 30

# Collect offline data from baseline policies (for RLPD / offline RL)
python -m scripts.collect_offline_data \
  --policy worldcomp \
  --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective efficiency_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000

# RLPD training (SAC + offline data; reserved for offline line, not the cancelled online thesis)
python -m scripts.train_sac \
  --offline-data offline_data/<dataset>/transitions.npz \
  --offline-ratio 0.5 \
  --use-asymmetric-critic   # critic gets privileged_obs from offline + online streams
  ... # other args

# Visualize trajectories and plot training curves
python -m scripts.visualize --checkpoint <save-dir>
python -m scripts.plot_training --log-dir <save-dir>
```

## Stage scripts (sweep launchers)

`[skip]` resume logic in each launcher: re-running after a Colab session restart picks up where it left off (skip is keyed on `agent_final.pt`, not `trainer_state.json`).

| Stage | Status | Run script | Summarize script |
|---|---|---|---|
| A0 (cross_u10 sensor screen) | ✅ thesis-grade result, citable | [`scripts/run_stage_a0_layout_screen.sh`](../../../scripts/run_stage_a0_layout_screen.sh) | [`scripts/summarize_stage_a0_layout_screen.sh`](../../../scripts/summarize_stage_a0_layout_screen.sh) |
| A1 (layout main) | ⚠ archived; thesis matrix cancelled 2026-05-06 | [`scripts/run_stage_a1_layout_main.sh`](../../../scripts/run_stage_a1_layout_main.sh) | [`scripts/summarize_stage_a1_layout_main.sh`](../../../scripts/summarize_stage_a1_layout_main.sh) |

[`scripts/run_protocol_stage_common.sh`](../../../scripts/run_protocol_stage_common.sh) is the shared sweep launcher; current online sprints (sanity + SAC collector) drive it via env-var overrides from each notebook (no per-stage wrapper needed). The previously planned `s2`–`s7` thesis stage wrappers were never created (matrix cancelled).
