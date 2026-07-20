# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training autonomous underwater vehicles (AUVs) — specifically REMUS-100 class — to navigate wake fields using reinforcement learning. Two research lines, **both closed**:

1. **Offline RL line (primary, paper-driving)** — all phases closed: TD3+BC baseline → ReBRAC-Q mainline → broad-validation v2 (PASS) → FQL succession (honest-negative) → AUVHamNODE offline ⏸ PAUSED. Entry + full dated timeline: [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md). Authoritative numbers: [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md).
2. **Online RL line (support)** — thesis-grade SAC matrix cancelled. Surviving roles: (a) `A0 sensor screen` on `cross_u10` (the only citable thesis-grade multi-seed online result) and (b) the **SAC collector** (`arrival_v2`) building the offline datasets that feed ReBRAC / FQL. Entry: [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md). The `AsymmetricQNetwork` (privileged hull-integral critic) survives as an offline-line ablation lever.

Bridges between the two lines: shared `auv_nav` env, probe layouts, offline data format, and the `AsymmetricQNetwork` used by ReBRAC's `--use-asymmetric-critic`. The project explicitly **does not** chase generic algorithm-paper improvements; algorithmic content is justified by deployment realism (single-point `s0` actor, hull-integral critic) and by mechanism-discriminating negative findings.

**Current focus: PhD dissertation Chapter 5 writing** — all standalone papers cancelled, material folded into the single chapter; central thesis = *under deployment constraints, the lever is using existing information/data better, not adding capability*. Live writing status ledger: [`paper/thesis_ch5/status.md`](paper/thesis_ch5/status.md) — **single source of truth; do not duplicate writing status here or in memory**; per-round task entry: [`paper/thesis_ch5/next_session_prompt.md`](paper/thesis_ch5/next_session_prompt.md) (rewritten each round, current-round scope only). Spec: [`paper/thesis_chapter_outline.md`](paper/thesis_chapter_outline.md). **Never write thesis numbers from memory** — trace every figure to the ground-truth docs below.

**Out of scope (do not propose without explicit user request):** AUVHamNODE / MBRL work (paused, user-confirmed); thesis-grade Online RL matrix expansion (cancelled); new standalone papers.

## Workflow & Compute Environment

Training runs on **Google Colab Pro / L4 GPU** with the codebase mounted from Google Drive. Local machine is for editing, doc work, and running smoke tests. The standard workflow:

1. Author / edit code locally; commit to git.
2. Sync the project directory to Google Drive (`drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5/`).
3. Open a notebook (one per experimental sprint) in Colab; mount Drive; `cd` into the project; set env overrides; invoke training via IPython shell magic (`!python -m scripts.train_*`, **not** `subprocess.run` — needed for realtime stdout in Colab).
4. Results land in `experiments/<study>/...` (small) and `checkpoints/<study>/...` (large) on Drive; sync back to git for analysis docs only.

L4 wallclock for a 600k-step SAC run with `num_envs=6` is ~1.5h. Notebook-driven sprints are sized so each sprint fits in 1-2 Colab sessions.

**Local Python env.** Use the `mytorch1` conda environment for local smoke tests and analysis scripts (already on `PATH`); Colab supplies its own runtime.

## Common Commands

Full CLI reference (training/eval/offline-collection invocations, sweep launcher table, `[skip]`-resume semantics) moved to the `rl-v2-commands` skill — see [`.claude/skills/rl-v2-commands/SKILL.md`](.claude/skills/rl-v2-commands/SKILL.md). All scripts are run as modules from the repo root (`python -m scripts.<name>`).

## Architecture

Module/script layout is derivable by reading `auv_nav/` and `scripts/` directly; not repeated here.

### Key Architectural Patterns

**Dataclass configs everywhere.** `TrainConfig`, `SACConfig`, `PlanarRemusEnvConfig`, `TaskSamplerConfig` are all `@dataclass`. They serialize to JSON for reproducibility and map directly to CLI args.

**Policy API.** All agents (SAC, baselines, ReBRAC) implement `act(obs, policy_state, deterministic) → (action, policy_state)` and `reset_policy_state()`. Supports both stateless and recurrent policies uniformly.

**Checkpointing.** `save_training_state()` in `train_utils.py` saves agent weights, replay buffer, Python RNG state, and a JSON metadata file. `maybe_resume()` restores all of it to continue training exactly. Checkpoints can be externalized via `--checkpoint-dir <PATH>` (kept separate from result `experiments/` tree).

**Parallel environments.** Training supports `gymnasium.vector.AsyncVectorEnv` for wall-clock speedup. The `--num-envs` flag controls parallelism. **`num_envs` is part of the experimental protocol** — mixing different `num_envs` values across runs in the same study breaks comparability; pick a value and hold it constant for any cell that will be compared.

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
  - `s0` — 1 probe at (0,0); DVL water-track, **deployment-realistic baseline** (the actor sensor for both SAC collector and ReBRAC deployable cells)
  - `s1` — 2 probes at (0,0)+(4.5,0); DVL + 2 MHz short-range forward ADCP, ~3 steps advance warning (reference upper bound)
  - `s2` — 4 probes at (0,0)+(5,0)+(8,±4); DVL + 1 MHz long-range ADCP, ~7 steps warning + lateral gradient (reference upper bound)
- **Task geometry:** `downstream`, `cross_stream`, `upstream`
- **Benchmarks (fixed evaluation manifests):** `benchmarks/<key>.json`, e.g. `single_u10_cross_tgt15`, `single_u10_upstream_tgt15`, `single_u15_upstream_tgt15`, `tandem_u15_upstream_tgt15`, `sbs_u15_upstream_tgt15`. Difficulty in this study is parameterised by the benchmark key (flow speed, geometry, target speed) rather than the legacy `--difficulty {easy,medium,hard}` flag.
- **Observation history:** `--history-length N` wraps env with `ObservationHistoryWrapper` for stacking N recent observations. Default for the SAC collector / A0 sensor screen is `k=4`.

### Data

Wake field data lives in `wake_data/` (gitignored). `scripts/generate_wake.py` creates synthetic fields. `flow.py`'s `WakeField` class memory-maps `.npy` files with shape `(T, Nx, Ny, C)`. Each `.npy` file requires a co-located `<stem>_meta.json` companion file (auto-generated by `generate_wake.py`).

Offline transition data lives in `offline_data/` (gitignored). Each subdirectory contains `transitions.npz` (`obs`, `actions`, `rewards`, `costs`, `next_obs`, `dones`, optionally `privileged_obs` / `next_privileged_obs`) and `metadata.json`. Generated by `scripts/collect_offline_data.py`. Supported baseline policies: `goalseek`, `crosscomp`, `worldcomp`, `privileged`.

### Default Hyperparameters (SACConfig)

Defaults are the `SACConfig` dataclass fields in `auv_nav/sac.py`. A0 sensor screen / SAC collector runs override `random_steps=5000`, `update_after=5000` (per `run_protocol_stage_common.sh`) — not the dataclass default.

## Documentation Index

**Two line-summary docs are the canonical index** — phase timelines, full doc routing, archive status, retrofit triggers all live there. Read them first; only fall back to the short list below for the highest-traffic entry points.

| Doc | Role |
|---|---|
| [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) | Offline RL line entry (primary) — phases, citable results, full doc index |
| [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) | Online RL line entry (support) — A0 + SAC collector roles, thesis-matrix closure |
| [`paper/thesis_chapter_outline.md`](paper/thesis_chapter_outline.md) | **Dissertation Ch5 writing spec (rev.12)** — central thesis, §0.4 red lines, register/terminology conventions, 10-section skeleton, reuse matrix |
| [`paper/thesis_ch5/status.md`](paper/thesis_ch5/status.md) | Ch5 writing status ledger — per-section state, pending decisions, locked-decision pointers |
| [`paper/thesis_ch5/next_session_prompt.md`](paper/thesis_ch5/next_session_prompt.md) | Per-round thesis-writing entry — current round's task, section-specific red lines (rewritten each round) |
| [`docs/environment_design.md`](docs/environment_design.md) / [`docs/rlpd_design.md`](docs/rlpd_design.md) | Env / RLPD design spec (cross-line) |
| [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) | ReBRAC ground truth (rev.8) — only authoritative source for numbers |
| [`docs/rebrac_paper_writing_index.md`](docs/rebrac_paper_writing_index.md) | "Which doc to open, which paragraph to copy" map for ReBRAC paper writing |
| [`docs/fql_succession_p2_results.md`](docs/fql_succession_p2_results.md) | FQL Succession main report (NEGATIVE closed 2026-05-23) |
| [`docs/auvhamnode_mbrl_line_pause_memo.md`](docs/auvhamnode_mbrl_line_pause_memo.md) | AUVHamNODE pause memo (⏸ PAUSED 2026-05-13) — read first if line is resumed |

## Notebooks

The Colab workflow drives experiments from notebooks under `notebooks/` (~117 files). Every notebook follows the same pattern: drive mount → `cd` into project → env-var overrides → `!python -m scripts.train_*` (IPython shell magic, **not** `subprocess.run` — needed for realtime stdout in Colab) → analysis cells reading the resulting summary CSVs.

Family conventions (use `ls notebooks/<prefix>*` to enumerate; line summary docs index every closed family):

- `sac_thesis_*` — online thesis preflight / profiling (✅ closed; thesis matrix cancelled 2026-05-06)
- `sac_arrival_v2_*` — `arrival_v2` reward + SAC collector R&D (✅ closed, feeds Offline collector)
- `sac_collector_*` — D4RL tier audit / Plan A 4-tier / h2h ReBRAC vs FQL sweeps (✅ closed)
- `rebrac_*` — ReBRAC mainline screen/formal + Stage D/E + broad-validation v1 (⚠ SUPERSEDED 2026-05-18) / v2 + C1 deep-dive (✅ closed; Findings (i)–(iv) + v2 N2′ ACTOR_FUNDAMENTAL_CONFIRMED 2026-05-27)
- `fql_succession_*` — Gate B + P2 2×2 modality×noise matrix + Q1/Q1b/Q1c mechanism + xbench FLOOR probe + verdict (✅ NEGATIVE closed 2026-05-23)
- `profile_train_*` — env / update wallclock profile (✅ closed)

Naming: bare `*.ipynb` = template; sibling `*_completed.ipynb` = materialised outputs. `_seedN` suffix = per-seed split for 3-way parallel Colab sessions.
