---
name: experiment-summary
description: Aggregate a finished rl_v2 sweep — read summary.csv / ablation_summary.csv / eval_log.csv, group by (probe, seed) or (algo, seed), produce a docs report section with metric tables and Gate-pass verdict
---

# experiment-summary

Turn a completed sweep into a `docs/<study>_report.md` section. The project follows a tight pattern: every sprint commits Gate thresholds in the notebook §0, then closes with a markdown table + Gate verdict (pass / partial / fail) and a one-paragraph next-step trigger.

## When to invoke

- User says "summarize sprint X" / "close out the sweep" / "write the report section"
- A `summary.csv`, `ablation_summary.csv`, or `gate_*_overview.csv` has just been produced
- The user asks for Gate verdict on a completed run

## CSV shape registry

The repo uses three CSV layouts; identify which one before parsing.

| Pattern | Columns of interest | Found under |
|---|---|---|
| `ablation_summary.csv` | `benchmark, objective, method, num_runs, eval_return_mean, eval_return_std, eval_success_rate_mean, eval_success_rate_std, eval_path_efficiency_mean, ...` | `experiments/<study>/<sprint>/` |
| `ablation_runs.csv` | per-seed rows: `benchmark, objective, method, seed, eval_return, eval_success_rate, eval_path_efficiency, ...` | `experiments/<study>/<sprint>/` |
| `gate_*_overview.csv` | `algo, seed, last3_eval_success_mean, last30pct_eval_slope, loss_flow_last_5pct_mean, actor_loss_last_5pct_mean, n_eval_points` | `results/offline/<line>/<gate>/summaries/` |
| `eval_log.csv` (per-run) | `step, eval_return, eval_success_rate, ...` | `results/offline/<line>/<gate>/<run>/` |

Always check column headers with `head -1` before computing.

## What the report table carries

Report `method`, `num_runs`, then mean/std pairs for `eval_return`, `eval_success_rate`, and
`eval_path_efficiency` — drop the rest. From `ablation_runs.csv` (per-seed rows) the same table is
reached by grouping on `method` with `nunique` on `seed`; from `gate_*_overview.csv` group on `algo`
and carry `last3_eval_success_mean` and `last30pct_eval_slope`.

## Report section template

Append (do not overwrite) the section to the appropriate report doc:

- Online line → `docs/online_rl_line_summary.md`. (Both former destinations are dead: `docs/online_rl_thesis_plan.md` is DEPRECATED 2026-05-06 — thesis matrix cancelled — and `docs/systematic_improved_sac_experiment_report.md` is DEPRECATED 2026-04-26, kept only as the A0 single source. Do not append new results to either.)
- ReBRAC offline → `docs/rebrac_experiment_report.md`
- FQL / new offline algos → `docs/fql_succession_gate_<x>_interim_report.md` or close out into `docs/offline_rl_line_summary.md`

Section skeleton:

```markdown
## §<n> <sprint-slug> — closure (<YYYY-MM-DD>)

**Gate** (committed in notebook §0):
- pass: <metric ≥ threshold>
- partial: <metric in range>
- fail: <metric < threshold>

**Result table**:

<paste markdown table from pandas .to_markdown()>

**Verdict**: <pass / partial / fail> — <one sentence why>.

**Next-step trigger**: <what unlocks; what this closes>.

**Artifacts**:
- summary CSV: `experiments/<study>/<sprint>/ablation_summary.csv`
- per-seed runs: `experiments/<study>/<sprint>/ablation_runs.csv`
- notebook: `notebooks/<sprint-slug>_completed.ipynb`
- commit: `<short sha>`
```

## Conventions to enforce

1. **Rounding and ±std notation follow the rows already in the target report** — open it and match. `docs/rebrac_experiment_report.md` C1 P1 is the anchor row for the offline line.
2. **Commit short SHA** in Artifacts — readers chase reproducibility through git, not Drive paths.
3. **Verdict noun must be one of**: pass / partial / fail. No invented categories.

## Cross-check before posting the section

- Does the per-seed std reported match `groupby('method')['eval_*'].std()`? If `num_runs=1`, std is undefined — write `n=1, std n/a`.
- If `last3_eval_success_mean` is computed on a slope-positive run (`last30pct_eval_slope > 0`), flag in the verdict line.
- If any seed JSON is missing from the sweep dir, surface it — do not silently average over fewer seeds.

## When NOT to use this skill

- The sweep is still running (no closing snapshot CSV yet).
- A bug-rerun is in progress (Gate is contaminated; close only after a clean sweep).
- The user wants exploratory analysis without committing to a report section.
