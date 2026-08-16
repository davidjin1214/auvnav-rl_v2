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

Training runs on **Google Colab Pro / L4 GPU** with the codebase mounted from Google Drive; the local machine is for editing, doc work, and smoke tests only. The four-step sync-and-launch procedure, the Drive path, and L4 wallclock budgeting live in the `rl-v2-commands` skill — see [`.claude/skills/rl-v2-commands/SKILL.md`](.claude/skills/rl-v2-commands/SKILL.md).

**Local Python env.** Use the `mytorch1` conda environment for local smoke tests and analysis scripts; Colab supplies its own runtime. Whether `conda` is on `PATH` is machine-dependent — in non-interactive shells it often is not, so activate it first or call the interpreter by full path.

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

Offline transition data lives in `offline_data/` (gitignored). Each subdirectory contains a `transitions.npz` plus a `metadata.json`, generated by `scripts/collect_offline_data.py` — that script is authoritative for the array keys and the supported baseline policies.

**Readouts live under `results/`, not `experiments/`.** The offline line's per-run readouts (`test_result.json`, `eval_log.csv`, `summaries/`) sit in `results/offline/{rebrac,td3bc,fql_succession,sac_collector_h2h}/`; `experiments/` holds the online line's run trees plus `arrival_v2_prototype/`. A 2026-08-16 audit found the offline line summary's data table pointing every row at `experiments/offline/...`, where almost none of it exists — a reader following that table concludes the data is gone. Check `results/` first.

**The five gitignored data directories may be junctions.** `results/ experiments/ checkpoints/ wake_data/ offline_data/` total ~40 GB, of which ~0.5 GB is what rebuilding the thesis figures and rerunning the audits actually needs. A machine may keep the bodies on another volume and mount them back with a directory junction (`mklink /J`) so repo-relative paths stay unchanged — figure scripts, audit tools, notebooks and `scripts/*.py` need no edits, and git cannot see them. Two consequences: (a) they do **not** arrive with a `git clone`, so a fresh working copy has them empty; (b) `benchmarks/` is **tracked** — never junction that one over the checkout.

## Documentation Index

**Two line-summary docs are the canonical index** — phase timelines, full doc routing, archive status, retrofit triggers all live there. Read them first; only fall back to the short list below for the highest-traffic entry points.

**They index the research narrative, not the file tree.** An audit on 2026-08-16 found 11 of the 52 top-level `docs/` files reachable from no entry point at all, and 20 whose validity could not be told without opening them — markdown lives in 21 directories here. [`docs/DOC_INDEX.md`](docs/DOC_INDEX.md) is the generated file map that closes that gap: every markdown in the repo, its H1, and its own status banner. Rebuild it with `python -m scripts.build_doc_index` after adding or moving docs; `--check` fails when it has drifted. Its status column reads `—` for "this doc does not label itself", **which is not the same as "still current"** — that call stays human work.

That orphan set closed on 2026-08-17. Ten of the eleven were FQL-line files, and the boundary that split them was **P2 closure artifacts vs pre-P2 construction records** — not "FQL vs the rest": FQL is §5.9 of the chapter, a closed honest-negative result, not a dropped line. Seven pre-P2 records moved to [`docs/archive/fql_succession/`](docs/archive/fql_succession/README.md); the other three stayed at `docs/` top level because inbound-link counting cannot tell a historical record from a **live contract that only code cites** — `fql_pytorch_port_design.md` and `fql_audit_multimodality_design.md` are the design contracts for `auv_nav/fql.py` and `scripts/audit_multimodality.py` (both still tested), and `fql_e_uni_anchor_dataset_card.md` cards a dataset that still exists. **Archiving here means relocation plus role labelling and nothing else** — no conclusion in those files was withdrawn.

**Pointer rot is the standing failure mode of this index.** Version numbers, section numbers, and deprecation status embedded in prose go stale silently and then get quoted as fact (2026-07-28 sweep found the spec rev pointer two revisions behind, six docs still citing the chapter's superseded 8-section `§N.k` numbering, and the authoritative ReBRAC report routing readers to broad-validation v1 with no mention that v2 exists). Run `python -m scripts.check_doc_pointers` after any doc reshuffle — it resolves every markdown link and bare `docs/foo.md` reference repo-wide, with no directory exempt — `.claude/` included. It only proves targets *exist*; whether a banner, a rev number, or a dated claim is still *true* stays human work. **Never write a live spec's rev number into another doc** — including this table; each doc carries its own version in its header.

**A path that legitimately does not exist is declared in the citing doc's own prose, not deleted.** The 2026-08-17 sweep cleared 67 `真失效` misses to 0, and 45 of them were cancelled or paused plan docs pointing at modules nobody ever wrote — deleting those links would have erased what the line planned, which is the only reason those plan docs are kept. So the doc says why, in the same sentence as the path, and `check_doc_pointers` reads that sentence and buckets the miss as `自述缺席`. Four accepted reasons, closed list: **从未产出 / 从未创建**（planned, never built）, **用后即删**（built then deliberately deleted）, **原计划文件名**（shipped under another name — give the real one）, **未入库 / 不入 git**（generated at run time, deliberately untracked). Establish which one applies with `git log --all --diff-filter=A -- <path>` before writing it; a false declaration silences the alarm permanently.

| Doc | Role |
|---|---|
| [`docs/doc_cleanup_status.md`](docs/doc_cleanup_status.md) | 🔧 **In-flight** — the repo-wide documentation cleanup: what the three layers are, which are done, the priority rule for the remaining one, and the traps. Sole status ledger for that work; delete when it closes |
| [`docs/DOC_INDEX.md`](docs/DOC_INDEX.md) | **Generated file map** — every markdown in the repo with its H1 and self-declared status. Use it to locate a doc; use the two line summaries below to understand a research line |
| [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) | Offline RL line entry (primary) — phases, citable results, full doc index |
| [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) | Online RL line entry (support) — A0 + SAC collector roles, thesis-matrix closure |
| [`paper/thesis_chapter_outline.md`](paper/thesis_chapter_outline.md) | **Dissertation Ch5 writing spec** — central thesis, §0.4 red lines, register/terminology conventions, 10-section skeleton, reuse matrix |
| [`paper/thesis_ch5/status.md`](paper/thesis_ch5/status.md) | Ch5 writing status ledger — per-section state, pending decisions, locked-decision pointers |
| [`paper/thesis_ch5/next_session_prompt.md`](paper/thesis_ch5/next_session_prompt.md) | Per-round thesis-writing entry — current round's task, section-specific red lines (rewritten each round) |
| [`docs/environment_design.md`](docs/environment_design.md) / [`docs/rlpd_design.md`](docs/rlpd_design.md) | Env / RLPD design spec (cross-line) |
| [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) | ReBRAC ground truth — only authoritative source for numbers |
| [`docs/data_integrity_open_items.md`](docs/data_integrity_open_items.md) | ⚠ Bookkeeping issues against those numbers — **items ①, ③ and ⑤ all verified to hold, disposition undecided**: the `cross-2000` dataset trains on the eval tasks (①), the checkpoint-selection set is a prefix of the reported test set (③), and the noisy-2000 dataset repeats ① (⑤ — nailed down Drive-side 2026-08-16, 100/100 identical task instances). All three land on the same §5.6.2 sentence. The contamination surface is **still not closed**: the Drive-side enumeration was itself shown to be incomplete. Read alongside the ground-truth report. `python -m scripts.audit_seed_overlap` scans **local** `offline_data/` + `benchmarks/` only — run it Drive-side before claiming repo-wide closure |
| [`paper/thesis_ch5/data_integrity_impact_assessment_review.md`](paper/thesis_ch5/data_integrity_impact_assessment_review.md) | Independent review of the impact assessment (2026-08-16) — graded findings, the four disposition options plus a missed fifth, and the ③ quantification. Tool: `paper/thesis_ch5/tools/ch5_holdout_split_audit.py` splits published 100-episode readouts into the 40 that fed checkpoint selection and the 60 that did not |
| [`docs/rebrac_paper_writing_index.md`](docs/rebrac_paper_writing_index.md) | "Which doc to open, which paragraph to copy" map for ReBRAC paper writing |
| [`docs/fql_succession_p2_results.md`](docs/fql_succession_p2_results.md) | FQL Succession main report (NEGATIVE closed 2026-05-23) |
| [`docs/auvhamnode_mbrl_line_pause_memo.md`](docs/auvhamnode_mbrl_line_pause_memo.md) | AUVHamNODE pause memo (⏸ PAUSED 2026-05-13) — read first if line is resumed |

## Notebooks

Experiments are driven from notebooks under `notebooks/` — all sprint families closed; the line summary docs index them.

Naming: bare `*.ipynb` = template; sibling `*_completed.ipynb` = materialised outputs. `_seedN` suffix = per-seed split for 3-way parallel Colab sessions.
