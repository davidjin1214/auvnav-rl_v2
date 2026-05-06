# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training autonomous underwater vehicles (AUVs) — specifically REMUS-100 class — to navigate wake fields using reinforcement learning. The project is organised into **two parallel research lines**, both targeting a thesis-grade systematic study:

1. **Online RL line** — primary algorithm Soft Actor-Critic (SAC) with deployment-realistic sensor (`s0` = DVL water-track only) as the main axis. Methodological contribution centres on **Asymmetric Critic with privileged hull-integral flow**, exposing the integrated effective flow that drives the dynamics to the critic during training while the actor remains restricted to the deployable single-point sensor at evaluation. Plan: [`docs/online_rl_thesis_plan.md`](docs/online_rl_thesis_plan.md).
2. **Offline RL line** — TD3+BC was the first finalist; **ReBRAC** is the current main track. Plan: [`docs/rebrac_experiment_plan.md`](docs/rebrac_experiment_plan.md). Report: [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md). Bridges to the online line via shared `auv_nav` env, probe layouts, and offline data formats.

The project explicitly **does not** chase generic algorithm-paper improvements; algorithmic content is justified by the sim2real story (privileged training) and by offline RL deployment realism.

## Workflow & Compute Environment

Training runs on **Google Colab Pro / L4 GPU** with the codebase mounted from Google Drive. Local machine is for editing, doc work, and running smoke tests. The standard workflow:

1. Author / edit code locally; commit to git.
2. Sync the project directory to Google Drive (`drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5/`).
3. Open a notebook (one per experimental sprint) in Colab; mount Drive; `cd` into the project; set env overrides; invoke shell scripts via `!bash`.
4. Results land in `experiments/<study>/...` (small) and `checkpoints/<study>/...` (large) on Drive; sync back to git for analysis docs only.

L4 wallclock for a 600k-step SAC run with `num_envs=6` is ~1.5h. Notebook-driven sprints are sized so each sprint fits in 1-2 Colab sessions.

## Common Commands

```bash
# Basic SAC training (uses synthetic wake data if no flow file found)
python -m scripts.train_sac

# Training with key options (current online thesis-plan defaults)
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
  --save-dir experiments/online_thesis_v1/.../seed_46

# Asymmetric Critic + LayerNorm + UTD=4 (online thesis §3 treatment)
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

# RLPD training (SAC + offline data; reserved for offline chapter, not online thesis)
python -m scripts.train_sac \
  --offline-data offline_data/<dataset>/transitions.npz \
  --offline-ratio 0.5 \
  --use-asymmetric-critic   # critic gets privileged_obs from offline + online streams
  ... # other args

# Visualize trajectories and plot training curves
python -m scripts.visualize --checkpoint <save-dir>
python -m scripts.plot_training --log-dir <save-dir>
```

All scripts are run as modules from the repo root (`python -m scripts.<name>`).

### Stage scripts (sweep launchers)

Each thesis stage has a thin sweep wrapper that calls `train_sac` per `(probe, seed)` pair, with `[skip]` resume logic so re-running the script after a Colab session restart picks up where it left off.

| Stage | Run script | Summarize script |
|---|---|---|
| A0 (cross_u10, completed) | [`scripts/run_stage_a0_layout_screen.sh`](scripts/run_stage_a0_layout_screen.sh) | [`scripts/summarize_stage_a0_layout_screen.sh`](scripts/summarize_stage_a0_layout_screen.sh) |
| A1 (deprecated, kept for reference) | [`scripts/run_stage_a1_layout_main.sh`](scripts/run_stage_a1_layout_main.sh) | [`scripts/summarize_stage_a1_layout_main.sh`](scripts/summarize_stage_a1_layout_main.sh) |

The new `online_rl_thesis_plan` sprints reuse [`scripts/run_protocol_stage_common.sh`](scripts/run_protocol_stage_common.sh) directly via env-var overrides from each notebook (no per-stage wrapper needed).

## Architecture

### Package: `auv_nav/`

The core library. Components are loosely coupled; non-ML parts work without PyTorch.

| Module | Role |
|--------|------|
| `vehicle.py` | REMUS-100 6-DOF nonlinear dynamics (RK4 integrator, ~850 lines) |
| `flow.py` | Memory-mapped wake field dataset; flow sampling at body-relative positions |
| `autopilot.py` | Inner-loop PID controllers + `EquivalentCurrentModel` (hull-integral flow estimator) |
| `env.py` | `PlanarRemusEnv` — Gymnasium environment (obs: 8-base + n_probes×2; emits `privileged_obs` in info) |
| `sac.py` | SAC agent: `SquashedGaussianActor`, `QNetwork`, **`AsymmetricQNetwork` (privileged-input critic)**, auto-tuned temperature |
| `networks.py` | MLP building blocks used by SAC |
| `replay.py` | `TransitionReplay` off-policy buffer (with optional `privileged_obs` / `next_privileged_obs`); `DualBufferSampler` for RLPD symmetric sampling |
| `reward.py` | `RewardModel` (progress, success, timeout) + `SafetyCostModel` |
| `baselines.py` | Non-learning policies (goal-seek, crosscomp, worldcomp, privileged) used for offline data collection |
| `rebrac.py` | ReBRAC agent (TD3+BC variant with critic-side BC penalty); used by the offline RL line |
| `offline_registry.py` | Registry of offline dataset configurations (probe / objective / policy / episodes) |

### Scripts: `scripts/`

| Script | Role |
|--------|------|
| `train_sac.py` | Main online training entry point; supports vanilla SAC, LayerNorm/Dropout/UTD, asymmetric critic, and RLPD via `--offline-data` |
| `train_utils.py` | Shared helpers: env creation, checkpointing, evaluation loop, CSV/JSONL logging |
| `train_offline_rebrac.py` | Offline ReBRAC training entry point |
| `run_suite.py` | Coordinates multi-seed experiment sweeps; defines `METHOD_SPECS` and `SUITE_PRESETS` |
| `collect_offline_data.py` | Collects transition data from baseline policies (records `privileged_obs` for AsymCritic) |
| `evaluate.py` | Loads a checkpoint and runs deterministic evaluation against a manifest |
| `generate_standard_benchmarks.py` | Builds fixed evaluation manifests (`benchmarks/<key>.json`) for reproducible `--eval-manifest` |
| `run_protocol_stage_common.sh` | Shared sweep launcher used by all online stage scripts |

### Key Architectural Patterns

**Dataclass configs everywhere.** `TrainConfig`, `SACConfig`, `PlanarRemusEnvConfig`, `TaskSamplerConfig` are all `@dataclass`. They serialize to JSON for reproducibility and map directly to CLI args.

**Policy API.** All agents (SAC, baselines, ReBRAC) implement `act(obs, policy_state, deterministic) → (action, policy_state)` and `reset_policy_state()`. Supports both stateless and recurrent policies uniformly.

**Checkpointing.** `save_training_state()` in `train_utils.py` saves agent weights, replay buffer, Python RNG state, and a JSON metadata file. `maybe_resume()` restores all of it to continue training exactly. Checkpoints can be externalized via `--checkpoint-dir <PATH>` (kept separate from result `experiments/` tree).

**Parallel environments.** Training supports `gymnasium.vector.AsyncVectorEnv` for wall-clock speedup. The `--num-envs` flag controls parallelism. **`num_envs` is part of the experimental protocol** (mixing different `num_envs` values across runs in the same study breaks comparability — see `docs/online_rl_thesis_plan.md` §1.2).

**Asymmetric Critic with privileged hull-integral flow.** When `--use-asymmetric-critic` is set:
- `AsymmetricQNetwork` extends critic input with `privileged_obs` (dim=2: body-frame `[u_eq, v_eq]` from `EquivalentCurrentModel`, the integrated effective flow that drives the AUV dynamics).
- During TD target and critic-loss steps, `privileged_obs` is passed in.
- During actor improvement (`Q(s, π_θ(s))`), `privileged_obs=None` → zero-padded → mimics deployment, where actor only has the s0 single-point sample.
- Works in both pure-online mode and RLPD mode (offline data must include `privileged_obs` and `next_privileged_obs` columns).

**RLPD (offline-to-online).** When `--offline-data` is provided, `train_sac.py` loads a pre-collected `.npz` dataset into a read-only `TransitionReplay` via `from_npz()`, then uses `DualBufferSampler` to draw each mini-batch with configurable split (via `--offline-ratio`) between offline and online buffers. Without `--offline-data`, training behaves as standard SAC with no regression.

**Conditional PyTorch.** `vehicle.py`, `flow.py`, and `env.py` import PyTorch via a `require_torch()` guard, so they can be used for analysis without a GPU install.

### Environment Details

- **Observation:** 8 base channels + n_probes × 2 (velocity) channels
  - Base 8: surge `u`, sway `v`, yaw rate `r`, `cos(ψ)`, `sin(ψ)`, goal body-frame x/y, distance-to-goal
  - Probe channels: (u, v) per probe in body frame (single-point samples)
  - `s0` (1 probe): **10-D**; `s1` (2 probes): **12-D**; `s2` (4 probes): **16-D**
- **`privileged_obs`** (emitted in `info`, used only by `AsymmetricQNetwork`): body-frame `[u_eq, v_eq]` (dim=2) — the **hull-integral** equivalent flow computed by `EquivalentCurrentModel` (weighted sum over multi-point hull samples). This is the true effective flow driving the dynamics, distinct from the actor's single-point probe samples.
- **Action (2-D):** continuous heading command, speed command
- **Probe layouts (`--probe-layout`):** all physically grounded in real REMUS-100 sensors
  - `s0` — 1 probe at (0,0); DVL water-track, **deployment-realistic baseline** (the online thesis main axis)
  - `s1` — 2 probes at (0,0)+(4.5,0); DVL + 2 MHz short-range forward ADCP, ~3 steps advance warning (reference upper bound)
  - `s2` — 4 probes at (0,0)+(5,0)+(8,±4); DVL + 1 MHz long-range ADCP, ~7 steps warning + lateral gradient (reference upper bound)
- **Task geometry:** `downstream`, `cross_stream`, `upstream`
- **Benchmarks (fixed evaluation manifests):** `benchmarks/<key>.json`, e.g. `single_u10_cross_tgt15`, `single_u10_upstream_tgt15`, `single_u15_upstream_tgt15`, `tandem_u15_upstream_tgt15`, `sbs_u15_upstream_tgt15`. Difficulty in this study is parameterised by the benchmark key (flow speed, geometry, target speed) rather than the legacy `--difficulty {easy,medium,hard}` flag.
- **Observation history:** `--history-length N` wraps env with `ObservationHistoryWrapper` for stacking N recent observations. Default for the online thesis line is `k=4`.

### Data

Wake field data lives in `wake_data/` (gitignored). `scripts/generate_wake.py` creates synthetic fields. `flow.py`'s `WakeField` class memory-maps `.npy` files with shape `(T, Nx, Ny, C)`. Each `.npy` file requires a co-located `<stem>_meta.json` companion file (auto-generated by `generate_wake.py`).

Offline transition data lives in `offline_data/` (gitignored). Each subdirectory contains `transitions.npz` (`obs`, `actions`, `rewards`, `costs`, `next_obs`, `dones`, optionally `privileged_obs` / `next_privileged_obs`) and `metadata.json`. Generated by `scripts/collect_offline_data.py`. Supported baseline policies: `goalseek`, `crosscomp`, `worldcomp`, `privileged`.

### Default Hyperparameters (SACConfig)

`hidden_dim=256`, `gamma=0.995`, `tau=0.005`, `actor/critic/alpha_lr=3e-4`, `init_alpha=0.2`, `grad_clip_norm=10.0`. Training defaults: `batch_size=256`, `random_steps=2000`, `update_after=2000`, `updates_per_step=1`. Online thesis line uses `random_steps=5000`, `update_after=5000` (per `run_protocol_stage_common.sh`).

## Documentation Index

| Doc | Role |
|---|---|
| [`docs/online_rl_thesis_plan.md`](docs/online_rl_thesis_plan.md) | **Active** plan for the online RL chapter (47 training run, 6 sections) |
| [`docs/rebrac_experiment_plan.md`](docs/rebrac_experiment_plan.md) | Active plan for the offline RL chapter (ReBRAC main track) |
| [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) | ReBRAC accumulated results |
| [`docs/SAC_improvements_survey.md`](docs/SAC_improvements_survey.md) | 2020-2026 survey of SAC improvements (REDQ / DroQ / CrossQ / RLPD / SimBa etc.) — design context |
| [`docs/world_model_and_offline_rl_survey.md`](docs/world_model_and_offline_rl_survey.md) | Survey for offline RL line |
| [`docs/environment_design.md`](docs/environment_design.md) | Environment, sensor, reward, and benchmark specification |
| [`docs/rlpd_design.md`](docs/rlpd_design.md) | RLPD implementation notes (used by both lines) |
| [`docs/systematic_improved_sac_experiment_plan.md`](docs/systematic_improved_sac_experiment_plan.md) | **DEPRECATED** (2026-04-26) — replaced by `online_rl_thesis_plan.md` |
| [`docs/systematic_improved_sac_experiment_report.md`](docs/systematic_improved_sac_experiment_report.md) | **DEPRECATED** — A0 results retained as historical record |
| [`docs/td3bc_*.md`](docs/) | Historical TD3+BC closure documents (offline RL line, before ReBRAC took over) |

## Notebooks

The Colab workflow drives experiments from notebooks (one per sprint). Notebooks are committed under `notebooks/`. Each notebook follows the pattern: drive mount → `cd` into project → env-var overrides → `!bash` invoke of stage script → analysis cells reading the resulting summary CSVs.

Active / planned notebooks:

| Notebook | Sprint | Status |
|---|---|---|
| `notebooks/sac_thesis_s0_preflight.ipynb` | Online thesis Sprint 0 (P1+P2) | scaffolded |
| `notebooks/sac_thesis_s0b_profiling.ipynb` | Online thesis Sprint 0 addendum (P0 — cProfile + num_envs benchmark) | scaffolded |
| `notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb` | Online thesis Sprint 1 | **deprecated 2026-05-06** — thesis 矩阵撤销，见 [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) |
| `notebooks/sac_thesis_s3_privileged_critic.ipynb` | Online thesis Sprint 2 (§3) | **cancelled 2026-05-06** (never created) |
| `notebooks/sac_thesis_s4_cross_sensor.ipynb` | Online thesis Sprint 2 (§4) | **cancelled 2026-05-06** (never created) |
| `notebooks/sac_thesis_s5_topology_eval.ipynb` | Online thesis Sprint 3 | **cancelled 2026-05-06** (never created) |
| `notebooks/sac_thesis_s6_confirmation.ipynb` | Online thesis Sprint 4 | **cancelled 2026-05-06** (never created) |
| `notebooks/sac_thesis_s7_history.ipynb` | Online thesis Sprint 5 | **cancelled 2026-05-06** (never created) |
| `notebooks/rebrac_*.ipynb` | Offline RL line (ReBRAC) | active |
