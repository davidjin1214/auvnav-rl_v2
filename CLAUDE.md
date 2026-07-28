# CLAUDE.md

## Project Overview

Research codebase for training autonomous underwater vehicles (AUVs) — specifically REMUS-100 class — to navigate wake fields using reinforcement learning. Two research lines, **both closed**:

1. **Offline RL line (primary, paper-driving)** — all phases closed: TD3+BC baseline → ReBRAC-Q mainline → broad-validation v2 (PASS) → FQL succession (honest-negative) → AUVHamNODE offline ⏸ PAUSED. Entry + full dated timeline: [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md). Authoritative numbers: [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md).
2. **Online RL line (support)** — thesis-grade SAC matrix cancelled. Surviving roles: (a) `A0 sensor screen` on `cross_u10` (the only citable thesis-grade multi-seed online result) and (b) the **SAC collector** (`arrival_v2`) building the offline datasets that feed ReBRAC / FQL. Entry: [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md). The `AsymmetricQNetwork` (privileged hull-integral critic) survives as an offline-line ablation lever.

Bridges between the two lines: shared `auv_nav` env, probe layouts, offline data format, and the `AsymmetricQNetwork` used by ReBRAC's `--use-asymmetric-critic`. The project explicitly **does not** chase generic algorithm-paper improvements; algorithmic content is justified by deployment realism (single-point `s0` actor, hull-integral critic) and by mechanism-discriminating negative findings.

**Current focus: PhD dissertation Chapter 5 writing** — all standalone papers cancelled, material folded into the single chapter; central thesis = *under deployment constraints, the lever is using existing information/data better, not adding capability*. Live writing status ledger: [`paper/thesis_ch5/status.md`](paper/thesis_ch5/status.md) — **single source of truth; do not duplicate writing status here or in memory**; per-round task entry: [`paper/thesis_ch5/next_session_prompt.md`](paper/thesis_ch5/next_session_prompt.md) (rewritten each round, current-round scope only). Spec: [`paper/thesis_chapter_outline.md`](paper/thesis_chapter_outline.md). **Never write thesis numbers from memory** — trace every figure to the ground-truth docs below.

**Language convention for the thesis audit chain (user-confirmed 2026-07-28):** the `.tex` rev blocks under `paper/thesis_ch5/sections/` and the `docs:` commit messages are written in **Chinese** and stay that way — `status.md` indexes commits by hash and描述, findings追注 quote the rev blocks, so anglicizing new entries breaks a cross-referenced ledger spanning five整改 batches. Deliberate exception to the global "comments and commit messages in English" rule; that rule still holds for **Python** (`paper/thesis_ch5/tools/`, `paper/thesis_ch5/figures/scripts/`).

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

Full CLI reference (training/eval/offline-collection invocations, sweep launcher table, `[skip]`-resume semantics) lives in the `rl-v2-commands` skill — see [`.claude/skills/rl-v2-commands/SKILL.md`](.claude/skills/rl-v2-commands/SKILL.md).

## Environment & Sensing

Probe coordinates, per-layout observation dimensions, and the sensor physics behind `s0`/`s1`/`s2` are specified in the `get_probe_positions()` docstring in [`auv_nav/flow.py`](auv_nav/flow.py) (authoritative, alongside the code) and derived in full in [`docs/environment_design.md`](docs/environment_design.md) (beam geometry, advance-warning budget, footprint vs. vortex wavelength). Read those rather than a summary.

## Non-obvious Conventions

**`num_envs` is part of the experimental protocol.** Mixing different `--num-envs` values across runs in the same study breaks comparability; pick a value and hold it constant for any cell that will be compared.

**Asymmetric critic — actor-side zero-padding is deliberate.** With `--use-asymmetric-critic`, `privileged_obs` is passed during TD-target and critic-loss steps but **not** during actor improvement (`Q(s, π_θ(s))` gets `privileged_obs=None` → zero-padded). This mimics deployment, where the actor only has the `s0` single-point sample. The code shows the `None` branch but not the reason.

**Benchmark keys, not `--difficulty`.** Difficulty is parameterised by the benchmark manifest key (flow speed, geometry, target speed) — `benchmarks/<key>.json`. The legacy `--difficulty {easy,medium,hard}` flag still exists in the CLI but is not used by this study.

**SACConfig overrides.** A0 sensor screen / SAC collector runs use `random_steps=5000`, `update_after=5000` (per `run_protocol_stage_common.sh`) — not the dataclass defaults in `auv_nav/sac.py`.

## Data

Wake field data lives in `wake_data/` (gitignored). `scripts/generate_wake.py` creates synthetic fields. `flow.py`'s `WakeField` class memory-maps `.npy` files with shape `(T, Nx, Ny, C)`. Each `.npy` file requires a co-located `<stem>_meta.json` companion file (auto-generated by `generate_wake.py`).

Offline transition data lives in `offline_data/` (gitignored). Each subdirectory contains `transitions.npz` (`obs`, `actions`, `rewards`, `costs`, `next_obs`, `dones`, optionally `privileged_obs` / `next_privileged_obs`) and `metadata.json`. Generated by `scripts/collect_offline_data.py`. Supported baseline policies: `goalseek`, `crosscomp`, `worldcomp`, `privileged`.

## Documentation Index

**Two line-summary docs are the canonical index** — phase timelines, full doc routing, archive status, retrofit triggers all live there. Read them first; only fall back to the short list below for the highest-traffic entry points.

**Pointer rot is the standing failure mode of this index.** Version numbers, section numbers, and deprecation status embedded in prose go stale silently and then get quoted as fact (2026-07-28 sweep found the spec rev pointer two revisions behind, six docs still citing the chapter's superseded 8-section `§N.k` numbering, and the authoritative ReBRAC report routing readers to broad-validation v1 with no mention that v2 exists). Run `python -m scripts.check_doc_pointers` after any doc reshuffle — it resolves every markdown link and bare `docs/foo.md` reference repo-wide, `.claude/skills/` included (only `.claude/agents/` is exempt; those cite invented filenames by design). It only proves targets *exist*; whether a banner, a rev number, or a dated claim is still *true* stays human work. **Never write a live spec's rev number into another doc** — including this table; each doc carries its own version in its header.

| Doc | Role |
|---|---|
| [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) | Offline RL line entry (primary) — phases, citable results, full doc index |
| [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) | Online RL line entry (support) — A0 + SAC collector roles, thesis-matrix closure |
| [`paper/thesis_chapter_outline.md`](paper/thesis_chapter_outline.md) | **Dissertation Ch5 writing spec** — central thesis, §0.4 red lines, register/terminology conventions, 10-section skeleton, reuse matrix |
| [`paper/thesis_ch5/status.md`](paper/thesis_ch5/status.md) | Ch5 writing status ledger — per-section state, pending decisions, locked-decision pointers |
| [`paper/thesis_ch5/next_session_prompt.md`](paper/thesis_ch5/next_session_prompt.md) | Per-round thesis-writing entry — current round's task, section-specific red lines (rewritten each round) |
| [`docs/environment_design.md`](docs/environment_design.md) / [`docs/rlpd_design.md`](docs/rlpd_design.md) | Env / RLPD design spec (cross-line) |
| [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) | ReBRAC ground truth — only authoritative source for numbers |
| [`docs/rebrac_paper_writing_index.md`](docs/rebrac_paper_writing_index.md) | "Which doc to open, which paragraph to copy" map for ReBRAC paper writing |
| [`docs/fql_succession_p2_results.md`](docs/fql_succession_p2_results.md) | FQL Succession main report (NEGATIVE closed 2026-05-23) |
| [`docs/auvhamnode_mbrl_line_pause_memo.md`](docs/auvhamnode_mbrl_line_pause_memo.md) | AUVHamNODE pause memo (⏸ PAUSED 2026-05-13) — read first if line is resumed |

## Notebooks

Experiments are driven from notebooks under `notebooks/` (~117 files, all sprint families closed — the line summary docs index them). Every notebook follows the same pattern: drive mount → `cd` into project → env-var overrides → `!python -m scripts.train_*` → analysis cells reading the resulting summary CSVs.

Naming: bare `*.ipynb` = template; sibling `*_completed.ipynb` = materialised outputs. `_seedN` suffix = per-seed split for 3-way parallel Colab sessions.
