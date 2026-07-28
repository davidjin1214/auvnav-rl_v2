---
name: rl-v2-commands
description: CLI invocation reference for rl_v2 — online SAC training, offline training (train_offline; td3bc / rebrac / fql), evaluation, offline data collection, RLPD, and sweep-launcher commands with the project's exact flag conventions (probe-layout, task-geometry, target-speed, penalty coefficients, eval-manifest paths, num_envs protocol). Use when running, writing, or debugging a `scripts.*` command or a sweep launcher in this repo.
---

# rl_v2 command reference

All scripts are run as modules from the repo root (`python -m scripts.<name>`).

```bash
# SAC training (A0 sensor-screen / SAC collector defaults)
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
```

## Offline training (`scripts.train_offline`)

The offline line's single entry point; `--algo` selects `td3bc` (default) / `rebrac` / `fql`. The
protocol block (`--probe-layout` … `--objective`) must match the dataset the `.npz` was collected
under — a mismatch trains on silently misaligned observations.

ReBRAC mainline, at the finalist penalty pair (β1 = actor 4.0, β2 = critic 2.0). Values below are
the defaults in [`scripts/run_offline_rebrac_broad.sh`](../../../scripts/run_offline_rebrac_broad.sh);
that launcher is the authority if the two ever drift:

```bash
python -m scripts.train_offline \
  --algo rebrac \
  --offline-data offline_data/<dataset>/transitions.npz \
  --flow wake_data/<field>.npy \
  --manifest benchmarks/single_u10_cross_tgt15.json \
  --probe-layout s0 --history-length 4 \
  --task-geometry cross_stream --target-speed 1.5 --objective efficiency_v2 \
  --sampling-mode shuffle_no_replacement \
  --num-epochs 64 --batch-size 256 \
  --hidden-dim 256 --num-hidden-layers 3 \
  --actor-lr 3e-4 --critic-lr 3e-4 --gamma 0.99 --tau 0.005 \
  --actor-penalty-coef 4.0 --critic-penalty-coef 2.0 \
  --policy-noise 0.2 --noise-clip 0.5 --policy-freq 2 \
  --grad-clip-norm 10.0 --normalizer-eps 1e-3 \
  --critic-layernorm --no-actor-layernorm \
  --log-every 1000 --seed 42 --device cuda \
  --save-dir checkpoints/<study>/<cell>/seed_42
```

Per-algorithm deltas — everything else in the block above is shared:

```bash
# TD3+BC — single BC coefficient replaces the two ReBRAC penalties; no critic-side BC
--algo td3bc --alpha 0.25          # sweeps in phase0b used 0.0 0.05 0.1 0.25 0.5 1.0

# FQL — flow-matching teacher + distilled student; no LayerNorm flags
--algo fql --flow-steps 10 --distill-alpha-bc 1.0 \
  --teacher-lr 3e-4 --flow-time-embed-dim 32

# Asymmetric critic (any algo) — actor still zero-pads the privileged channels
--use-asymmetric-critic --privileged-actor-update-mode zeros   # 'batch' feeds dataset privileged_obs
```

`--critic-layernorm` is **not** optional for ReBRAC-Q — it is critic-side representation
infrastructure, not an ablation knob. `--eval-every 0 --skip-final-eval` is the sweep pattern
(checkpoints are scored afterwards by `scripts.evaluate_best_checkpoint`), so a standalone run
that wants curves must set `--eval-every` and `--eval-episodes` explicitly.

## Stage scripts (sweep launchers)

`[skip]` resume logic in each launcher: re-running after a Colab session restart picks up where it left off (skip is keyed on `agent_final.pt`, not `trainer_state.json`).

| Stage | Status | Run script | Summarize script |
|---|---|---|---|
| A0 (cross_u10 sensor screen) | ✅ thesis-grade result, citable | [`scripts/run_stage_a0_layout_screen.sh`](../../../scripts/run_stage_a0_layout_screen.sh) | [`scripts/summarize_stage_a0_layout_screen.sh`](../../../scripts/summarize_stage_a0_layout_screen.sh) |
| A1 (layout main) | ⚠ archived; thesis matrix cancelled 2026-05-06 | [`scripts/run_stage_a1_layout_main.sh`](../../../scripts/run_stage_a1_layout_main.sh) | [`scripts/summarize_stage_a1_layout_main.sh`](../../../scripts/summarize_stage_a1_layout_main.sh) |

[`scripts/run_protocol_stage_common.sh`](../../../scripts/run_protocol_stage_common.sh) is the shared sweep launcher; current online sprints (sanity + SAC collector) drive it via env-var overrides from each notebook (no per-stage wrapper needed).

Offline sweeps have their own launchers rather than a stage table — [`run_offline_rebrac_screen.sh`](../../../scripts/run_offline_rebrac_screen.sh) (β grid screen), [`run_offline_rebrac_broad.sh`](../../../scripts/run_offline_rebrac_broad.sh) (broad validation spokes), [`run_offline_td3bc_phase0b_v2.sh`](../../../scripts/run_offline_td3bc_phase0b_v2.sh) (α × dataset-size ablation). FQL sprints run from notebook builders (`scripts/_build_fql_succession_*`), not shell launchers.
