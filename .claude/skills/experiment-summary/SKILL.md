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

## Standard summary pipeline

```python
import pandas as pd
from pathlib import Path

csv = Path("experiments/<study>/<sprint>/ablation_summary.csv")
df = pd.read_csv(csv)

# Drop columns you don't need for the report table
keep = [
    "method", "num_runs",
    "eval_return_mean", "eval_return_std",
    "eval_success_rate_mean", "eval_success_rate_std",
    "eval_path_efficiency_mean", "eval_path_efficiency_std",
]
print(df[keep].round(3).to_markdown(index=False))
```

For per-seed (`ablation_runs.csv` or per-run JSON):

```python
g = df.groupby(["method"]).agg(
    n_seeds=("seed", "nunique"),
    success_mean=("eval_success_rate", "mean"),
    success_std=("eval_success_rate", "std"),
    return_mean=("eval_return", "mean"),
    return_std=("eval_return", "std"),
).round(3)
print(g.to_markdown())
```

For Gate B / offline RL (`gate_*_overview.csv`):

```python
df = pd.read_csv("results/offline/fql_succession/gate_b/summaries/gate_b_overview.csv")
df.groupby("algo")[["last3_eval_success_mean", "last30pct_eval_slope"]].agg(["mean", "std"]).round(3)
```

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

1. **Round to 3 decimals** for success rate / efficiency; **2 decimals** for return; match existing report style.
2. **±std notation** in prose (`0.225 ± 0.005`) matches `docs/rebrac_experiment_report.md` C1 P1 anchor row.
3. **Commit short SHA** in Artifacts — readers chase reproducibility through git, not Drive paths.
4. **Never edit `*_completed.ipynb`** — the PreToolUse hook blocks this. If the notebook needs corrections, edit the builder under `scripts/_build_*_notebook.py` and regenerate.
5. **Verdict noun must be one of**: pass / partial / fail. No invented categories.
6. **Date is absolute** (`2026-05-20`), not relative — memory rule for this project.

## Cross-check before posting the section

- Does the per-seed std reported match `groupby('method')['eval_*'].std()`? If `num_runs=1`, std is undefined — write `n=1, std n/a`.
- If `last3_eval_success_mean` is computed on a slope-positive run (`last30pct_eval_slope > 0`), flag in the verdict line.
- If any seed JSON is missing from the sweep dir, surface it — do not silently average over fewer seeds.

## When NOT to use this skill

- The sweep is still running (no closing snapshot CSV yet).
- A bug-rerun is in progress (Gate is contaminated; close only after a clean sweep).
- The user wants exploratory analysis without committing to a report section.
