---
name: notebook-from-template
description: Scaffold a Colab sprint notebook following the rl_v2 scripts/_build_*_notebook.py convention — drive mount, project cd, env-var override, !bash sweep, summary-CSV analysis
disable-model-invocation: true
---

# notebook-from-template

Generate a new Colab sprint notebook by writing a `scripts/_build_<sprint>_notebook.py` builder, then running it. The resulting `.ipynb` lands under `notebooks/<sprint>.ipynb` and follows the five-cell convention used across the project (rebrac, fql-succession, sac thesis, etc.).

## When to invoke

User says `/notebook-from-template <sprint-slug>` or asks to "scaffold a new sprint notebook". Use the slug to derive:

- Builder path: `scripts/_build_<sprint-slug>_notebook.py`
- Notebook path: `notebooks/<sprint-slug>.ipynb`

## Why we use a builder (not direct .ipynb authoring)

`.ipynb` cell IDs and JSON quoting are fragile (commits `e6e23ec`, `9685b0f`, `6a3a88d` were notebook-edit bug fixes). The `_build_*` Python script is the source of truth; the notebook is a build artifact. The PreToolUse hook will block edits to `*_completed.ipynb`, so always regenerate from the builder.

## Builder skeleton

```python
"""One-shot builder for notebooks/<sprint-slug>.ipynb.

<one-paragraph hypothesis + gate>

Run from repo root:
    python -m scripts._build_<sprint-slug>_notebook
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _join(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(*lines: str) -> dict:
    return {"id": _new_id(), "cell_type": "markdown", "metadata": {}, "source": _join(lines)}


def code(*lines: str) -> dict:
    return {
        "id": _new_id(),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _join(lines),
    }


CELLS: list[dict] = []

# §0 — Hypothesis + gate (markdown)
CELLS.append(md(
    "# <Sprint title>",
    "",
    "## 假设",
    "<one sentence>",
    "",
    "## Gate (committed before run)",
    "- pass:  <metric ≥ threshold>",
    "- partial: <metric in range>",
    "- fail:  <metric < threshold>",
))

# §1 — Drive mount
CELLS.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
))

# §2 — cd into project
CELLS.append(code(
    "%cd /content/drive/MyDrive/Colab\\ Notebooks/new_offRL/rl_v2_5",
    "!pwd && git rev-parse --short HEAD",
))

# §3 — Env overrides (one per protocol knob)
CELLS.append(code(
    "import os",
    "os.environ['STAGE'] = '<stage>'",
    "os.environ['PROBE'] = 's0'",
    "os.environ['SEEDS'] = '42 43 44 45 46'",
    "os.environ['TOTAL_STEPS'] = '600000'",
    "os.environ['NUM_ENVS'] = '6'",
    "os.environ['EVAL_MANIFEST'] = 'benchmarks/single_u15_upstream_tgt15.json'",
    "os.environ['SAVE_ROOT'] = 'experiments/<study>/<sprint-slug>'",
))

# §4 — Sweep launch (real-time streaming; do not pipe through tee unless required)
CELLS.append(code(
    "!bash scripts/run_<stage>_<sprint-slug>.sh",
))

# §5 — Summary CSV + analysis
CELLS.append(code(
    "import pandas as pd",
    "from pathlib import Path",
    "csv = Path('experiments/<study>/<sprint-slug>/summary.csv')",
    "df = pd.read_csv(csv)",
    "df.groupby(['probe', 'seed'])[['success_rate', 'mean_return']].mean().round(3)",
))


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/<sprint-slug>.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
```

## Conventions to preserve

1. **`md()` / `code()` / `_join()` helpers are fixed signatures** — copy them verbatim. Other builders in `scripts/` rely on the same shapes.
2. **Cell IDs are 8-char hex from `uuid.uuid4().hex[:8]`** — never reuse, never hand-author.
3. **Markdown sections use 中文标题 + 英文 metric names** to match existing notebook voice.
4. **§4 sweep cell uses `!bash`, not `%%bash`** — `%%bash` buffers output and breaks real-time streaming (fix in commit `9685b0f`).
5. **Paths with spaces (`Colab Notebooks`) need backslash escaping in `!`/`%cd`**, not quoting (fix in commit `e6e23ec`).
6. **When a cell invokes `python` directly instead of a sweep script, use `!python -m scripts.<name>`, never `subprocess.run`/`Popen`** — piped stdout is block-buffered (4 KB), so per-1000-step `actor_loss`/`critic_loss` lines only surface in late bursts and stalls become invisible. Interpolate Python values with `{var}`, wrap path arguments in single quotes (`'{path_str}'`) to survive shell-split, continue long commands with a trailing backslash, and bracket the call with a start banner + `[done] ... {(time.time()-t0)/60:.1f} min` for wallclock feedback.
7. **Gate thresholds committed in §0 before the run starts** — non-negotiable; documented in `docs/rebrac_experiment_plan.md` (the online-line precedent lived in `docs/online_rl_thesis_plan.md`, DEPRECATED 2026-05-06 — read it as a historical example only).

## Build + verify

```bash
python -m scripts._build_<sprint-slug>_notebook
ls -la notebooks/<sprint-slug>.ipynb
jupyter nbconvert --to script notebooks/<sprint-slug>.ipynb --stdout | head -40  # sanity scan
```

If the sweep script does not exist yet, scaffold it from `scripts/run_protocol_stage_common.sh` first, then the notebook's §4 cell will resolve.

## When NOT to use this skill

- Editing an existing notebook in-place (PreToolUse hook will block `_completed.ipynb`; use the builder for non-completed ones too).
- Authoring `*_completed.ipynb` directly — those are Colab outputs, not source.
- Ad-hoc exploratory notebooks where the five-cell discipline does not apply.
