"""Build the Q1 critic-penalty ablation notebook for FQL Succession P2.

Q1 mechanism lemma:
    H0 (null) : ReBRAC critic BC penalty (rebrac.py:284) is NOT the dominant
                cause of the +20.5 pp FQL > ReBRAC gap on M-uni-noise.
    H1 (alt)  : Setting `--critic-penalty-coef 0.0` lifts ReBRAC SR from
                ~0.70 → ~0.85-0.91 (matching FQL within 5 pp).

If H1 confirmed, the diagnostic's mechanistic claim closes in closed form:
    critic_penalty inflation IS the bottleneck on noisy data.
If H1 rejected (SR stays low), the actor BC noise floor or some other path
dominates; we'd then need a 2nd ablation (--actor-penalty-coef 0.0).

Design:
- Dataset: REUSE existing M-uni-noise (no new collection).
- Algo: ReBRAC only, with `--critic-penalty-coef 0.0` (actor β1=4.0 unchanged).
- Seeds: [42, 0] (matches primary spec n=2).
- Compares against existing results/.../m_uni_noise/test/{rebrac,fql}_seed{42,0}.json
  produced by sprint 0.

Output: notebooks/fql_succession_p2_q1_critic_penalty_ablation.ipynb
Regenerate: python -m scripts._build_fql_succession_p2_q1_ablation_notebook
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

# Reuse helpers from the main builder via import (no copy-paste drift).
from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    _new_id,
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_q1_critic_penalty_ablation.ipynb")

CELL_ID_BASE = "m_uni_noise"
DATASET = "offline_data/fql_succession/m_uni_noise_eps0p5_1000"
SUFFIX = "_q1_critic_pen0"  # checkpoint/results suffix so it doesn't collide with sprint-0

# Literal Python expression embedded into the generated notebook (3 sections use it).
BASELINE_DIR_LITERAL = f"Path('results/fql_succession/p2/{CELL_ID_BASE}/test')"


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **Q1 critic-penalty ablation** (M-uni-noise)",
        "",
        "**Purpose**: Closed-form mechanism lemma for the diagnostic in "
        "`docs/fql_succession_p2_mechanism_diagnostic.md` §3.1 / §8 Q1.",
        "",
        "## Hypothesis",
        "",
        "Diagnostic shows ReBRAC's `critic_penalty` (rebrac.py:284) on M-uni-noise "
        "doubles vs E-uni (0.21 vs 0.09) due to dataset action noise σ=0.5. This "
        "inflation enters every TD target as additional pessimism and compounds "
        "through γ=0.99 → mean_q gap −126 (ReBRAC vs FQL).",
        "",
        "- **H0 (null)**: Setting `--critic-penalty-coef 0.0` does *not* recover "
        "ReBRAC's SR on M-uni-noise — actor BC noise floor or some other path "
        "dominates.",
        "- **H1 (alt)**: `--critic-penalty-coef 0.0` lifts ReBRAC SR from "
        "0.68/0.73 → ≥ 0.85, matching FQL (0.91/0.91) within 5 pp.",
        "",
        "## Why this matters",
        "",
        "If H1 confirmed, the noise-axis story closes in closed form:",
        "- The Δ = +20.5 pp FQL win on M-uni-noise is mechanistically attributable "
        "to **critic BC penalty inflation**, not to actor BC, not to capacity, not "
        "to FQL teacher diversity.",
        "- Paper revision can cite this 2-run ablation as the smoking gun.",
        "",
        "If H1 rejected, follow up with Q1b: `--critic-penalty-coef 0.0 "
        "--actor-penalty-coef 0.0` (pure TD3 offline). But Q1b risks pathological "
        "extrapolation; we'd then have to argue why FQL avoids it.",
        "",
        "## Design",
        "",
        "| Knob | Sprint-0 ReBRAC (baseline) | This Q1 ablation |",
        "|---|---|---|",
        "| `--actor-penalty-coef` | 4.0 | **4.0** (unchanged) |",
        "| `--critic-penalty-coef` | 2.0 | **0.0** ⚠️ |",
        "| `--critic-layernorm` | on | on |",
        "| `--no-actor-layernorm` | on | on |",
        "| dataset | `m_uni_noise_eps0p5_1000` | **same** |",
        "| seeds | [42, 0] | [42, 0] |",
        "",
        "**2 runs total**. Wallclock ≈ 1h on Colab L4.",
        "",
        "## Out of scope",
        "",
        "- FQL re-run (its baseline already at 0.91 — no headroom).",
        "- New collection (reuse existing M-uni-noise).",
        "- Actor-penalty-coef ablation (Q1b, follow-up if Q1 rejects H1).",
        "- M-multi-mix / E-uni Q1 ablation (only M-uni-noise has the inflation we want to suppress).",
    )


def section_1() -> list[dict]:
    return [
        md(
            "## 1. Environment sanity",
            "",
            "确认 cwd 是 repo root + GPU 可用 + `auv_nav` import 通过。",
        ),
        code(
            "import os, sys, subprocess",
            "from pathlib import Path",
            "",
            "if Path.cwd().name != 'rl_v2':",
            "    candidates = [",
            "        Path('/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5'),",
            "        Path('/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2'),",
            "    ]",
            "    for c in candidates:",
            "        if c.exists():",
            "            os.chdir(c); break",
            "    else:",
            "        raise RuntimeError('repo root not found')",
            "",
            "print('cwd =', Path.cwd())",
            "import torch",
            "print('torch =', torch.__version__, '  cuda =', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print('  device =', torch.cuda.get_device_name(0))",
            "import auv_nav",
            "print('auv_nav OK')",
        ),
    ]


def section_2() -> list[dict]:
    return [
        md(
            "## 2. Config",
            "",
            "复用 sprint-0 的所有共享参数,仅改 ReBRAC flags:",
            "`--critic-penalty-coef 2.0` → **0.0**(其他不变)。",
            "",
            "checkpoint / results 用 `_q1_critic_pen0` 后缀,避免覆盖 sprint-0 baseline。",
        ),
        code(
            "import os",
            "import shutil",
            "from pathlib import Path",
            "",
            f"CELL_ID  = '{CELL_ID_BASE}{SUFFIX}'",
            f"DATASET  = '{DATASET}'",
            "",
            "FLOW     = 'wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy'",
            "MANIFEST = 'benchmarks/single_u10_cross_tgt15_ep100.json'",
            "",
            "PROBE_LAYOUT   = 's0'",
            "HISTORY_LENGTH = 4",
            "TARGET_SPEED   = 1.5",
            "TASK_GEOMETRY  = 'cross_stream'",
            "OBJECTIVE      = 'arrival_v2'",
            "",
            "TOTAL_STEPS    = 200_000",
            "BATCH_SIZE     = 256",
            "EVAL_EVERY     = 10_000",
            "EVAL_EPISODES  = 100",
            "TEST_EPISODES  = 100",
            "EVAL_NUM_WORKERS = 6",
            "",
            "# Q1 ablation: critic_penalty_coef 2.0 → 0.0 (actor β1=4.0 unchanged)",
            "REBRAC_Q1_FLAGS = (",
            "    '--actor-penalty-coef 4.0 --critic-penalty-coef 0.0 '",
            "    '--critic-layernorm --no-actor-layernorm'",
            ")",
            "",
            "TRAIN_SEEDS = [42, 0]",
            "ALGOS = ['rebrac']  # FQL not re-run (baseline already at 0.91)",
            "",
            "CHECKPOINT_ROOT = Path(f'checkpoints/fql_succession/p2/{CELL_ID}')",
            "RESULTS_ROOT    = Path(f'results/fql_succession/p2/{CELL_ID}')",
            "TEST_DIR        = RESULTS_ROOT / 'test'",
            "CURVES_DIR      = RESULTS_ROOT / 'training_curves'",
            "",
            "MIRROR_FILES = ('train_log.jsonl', 'eval_log.csv', 'trainer_state.json', 'train_config.txt')",
            "",
            "def save_dir(algo, seed):     return CHECKPOINT_ROOT / f'{algo}_seed{seed}'",
            "def test_json(algo, seed):    return TEST_DIR / f'{algo}_seed{seed}.json'",
            "def mirror_dir(algo, seed):   return CURVES_DIR / f'{algo}_seed{seed}'",
            "",
            "print('CELL_ID         =', CELL_ID)",
            "print('DATASET         =', DATASET)",
            "print('REBRAC_Q1_FLAGS =', REBRAC_Q1_FLAGS)",
            "print('TRAIN_SEEDS     =', TRAIN_SEEDS, '  ALGOS =', ALGOS)",
            "print('CHECKPOINT_ROOT =', CHECKPOINT_ROOT)",
            "print('RESULTS_ROOT    =', RESULTS_ROOT)",
        ),
    ]


def section_3() -> list[dict]:
    return [
        md(
            "## 3. Pre-flight",
            "",
            "确认:",
            "1. Dataset directory + `transitions.npz` + `metadata.json` exist",
            "2. metadata 与 sprint-0 M-uni-noise 一致(σ=0.5, priv 1.0, 1000 ep, ~120525 transitions)",
            "3. manifest + flow exist",
            "4. baseline sprint-0 test json 存在(供 §6 对比) — 不强制(若 missing 则 §6 用 None placeholder)",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "",
            "ds = Path(DATASET)",
            "assert ds.is_dir(), f'dataset dir missing: {ds}'",
            "assert (ds / 'transitions.npz').is_file(), f'transitions.npz missing under {ds}'",
            "meta_path = ds / 'metadata.json'",
            "assert meta_path.is_file(), f'metadata.json missing under {ds}'",
            "meta = json.loads(meta_path.read_text())",
            "print('policy_mixture       =', meta.get('policy_mixture'))",
            "print('action_noise_std     =', meta.get('action_noise_std'))",
            "print('num_transitions      =', meta.get('num_transitions'))",
            "print('source_dataset       =', ds.name)",
            "",
            "# Soft expectations (printed, not asserted — fail fast only on file missing)",
            "exp_noise = 0.5",
            "exp_n_tx_approx = 120525",
            "noise = float(meta.get('action_noise_std', float('nan')))",
            "n_tx  = int(meta.get('num_transitions', 0))",
            "assert abs(noise - exp_noise) < 1e-6, f'noise {noise} != expected {exp_noise}'",
            "assert abs(n_tx - exp_n_tx_approx) < 5000, f'tx count {n_tx} far from expected {exp_n_tx_approx}'",
            "",
            "assert Path(FLOW).is_file(), f'flow missing: {FLOW}'",
            "assert Path(MANIFEST).is_file(), f'manifest missing: {MANIFEST}'",
            "",
            "# Baseline comparison (sprint-0); printed if available",
            f"BASELINE_DIR = {BASELINE_DIR_LITERAL}",
            "for algo in ('rebrac', 'fql'):",
            "    for s in TRAIN_SEEDS:",
            "        p = BASELINE_DIR / f'{algo}_seed{s}.json'",
            "        if p.exists():",
            "            d = json.loads(p.read_text())",
            "            print(f'baseline {algo} seed={s}  SR = {d[\"eval_success_rate\"]:.3f}')",
            "        else:",
            "            print(f'baseline {algo} seed={s}  (no file — §6 will skip)')",
            "",
            "print('\\nPre-flight PASS — ready to train.')",
        ),
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. Train (2 runs)",
            "",
            "ReBRAC × seeds [42, 0],`--critic-penalty-coef 0.0`,skip-resume on "
            "`agent_final.pt`(与 sprint-0 一致的 3 阶段恢复)。",
            "",
            "**`--skip-final-eval`**:在线 eval 跑 50 ep 即可,test eval 100 ep 走 §5。",
        ),
        code(
            "import time",
            "import shutil",
            "from pathlib import Path",
            "",
            "for algo in ALGOS:",
            "    ALGO_FLAGS = REBRAC_Q1_FLAGS  # only ReBRAC in this notebook",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        sd_str = str(sd)",
            "        if (sd / 'agent_final.pt').exists():",
            "            print(f'[skip-train] {algo} seed={seed} (agent_final.pt exists at {sd})')",
            "            continue",
            "        sd.mkdir(parents=True, exist_ok=True)",
            "        t0 = time.time()",
            "        print(f'>>> train {algo} (Q1 critic_pen=0) seed={seed} → {sd}')",
            "        !python -m scripts.train_offline \\",
            "            --algo {algo} \\",
            "            --offline-data '{DATASET}/transitions.npz' \\",
            "            --flow '{FLOW}' \\",
            "            --manifest '{MANIFEST}' \\",
            "            --probe-layout {PROBE_LAYOUT} \\",
            "            --history-length {HISTORY_LENGTH} \\",
            "            --target-speed {TARGET_SPEED} \\",
            "            --task-geometry {TASK_GEOMETRY} \\",
            "            --objective {OBJECTIVE} \\",
            "            --total-steps {TOTAL_STEPS} \\",
            "            --batch-size {BATCH_SIZE} \\",
            "            --eval-every {EVAL_EVERY} \\",
            "            --eval-episodes 50 \\",
            "            {ALGO_FLAGS} \\",
            "            --skip-final-eval \\",
            "            --seed {seed} \\",
            "            --save-dir '{sd_str}' \\",
            "            --device cuda",
            "        print(f'<<< done {algo} seed={seed} in {time.time()-t0:.0f}s')",
            "",
            "print('All training done.')",
        ),
    ]


def section_5() -> list[dict]:
    return [
        md(
            "## 5. Test eval (100 ep / seed) + mirror small files",
            "",
            "对每个 `agent_final.pt` 跑 `evaluate_offline` 100 ep(独立 manifest re-eval),"
            "结果写 `results/.../test/<algo>_seed<S>.json`;同时把 4 个 small files 从 "
            "`checkpoints/` mirror 到 `results/training_curves/<algo>_seed<S>/`。",
            "",
            "CLI 与 sprint-0 §5 完全一致(`--episodes`, `--worker-device cpu`, `--device cuda`,"
            "skip-test on existing json)。",
        ),
        code(
            "import time",
            "import shutil",
            "from pathlib import Path",
            "",
            "TEST_DIR.mkdir(parents=True, exist_ok=True)",
            "CURVES_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "t_start_eval = time.time()",
            "",
            "for algo in ALGOS:",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        sd_str = str(sd)",
            "        out_path = test_json(algo, seed)",
            "        out_path_str = str(out_path)",
            "",
            "        if out_path.exists():",
            "            print(f'[skip-eval] {algo} seed={seed}: {out_path}')",
            "            continue",
            "",
            "        if not (sd / 'agent_final.pt').exists():",
            "            print(f'[warn] missing agent_final.pt at {sd} — training not done?')",
            "            continue",
            "",
            "        print(f'\\n{\"=\" * 72}')",
            "        print(f'  eval {algo} seed={seed} → {out_path}')",
            "        print(f'{\"=\" * 72}')",
            "        t0 = time.time()",
            "",
            "        !python -m scripts.evaluate_offline \\",
            "            --checkpoint '{sd_str}' \\",
            "            --manifest '{MANIFEST}' \\",
            "            --episodes {TEST_EPISODES} \\",
            "            --num-workers {EVAL_NUM_WORKERS} \\",
            "            --worker-device cpu \\",
            "            --device cuda \\",
            "            --output-json '{out_path_str}'",
            "",
            "        print(f'[eval done] {algo} seed={seed} in {(time.time() - t0) / 60:.1f} min')",
            "",
            "print(f'\\n[all eval] total = {(time.time() - t_start_eval) / 60:.1f} min')",
            "",
            "# Mirror 4 small files for downstream plotting (always — outside the skip path)",
            "for algo in ALGOS:",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        mdir = mirror_dir(algo, seed)",
            "        mdir.mkdir(parents=True, exist_ok=True)",
            "        for fname in MIRROR_FILES:",
            "            src = sd / fname",
            "            if src.exists():",
            "                shutil.copy2(src, mdir / fname)",
            "        print(f'mirrored small files for {algo} seed={seed} → {mdir}')",
        ),
    ]


def section_6() -> list[dict]:
    return [
        md(
            "## 6. Three-way comparison",
            "",
            "Q1 ablation 与 sprint-0 baseline 同一 dataset 同一 (seed, algo) 对照:",
            "",
            "| variant | actor_β1 | critic_β2 | flag |",
            "|---|---:|---:|---|",
            "| sprint-0 ReBRAC | 4.0 | 2.0 | baseline (M-uni-noise) |",
            "| sprint-0 FQL    | (teacher distill, no λ-style BC) | — | upper bound |",
            "| **Q1 ReBRAC**   | **4.0** | **0.0** | this notebook |",
            "",
            "Primary metric = `eval_success_rate` (100 ep).",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "",
            f"BASELINE_DIR = {BASELINE_DIR_LITERAL}",
            "",
            "def load_sr(path):",
            "    if not path.exists():",
            "        return None",
            "    return float(json.loads(path.read_text())['eval_success_rate'])",
            "",
            "rows = []",
            "for seed in TRAIN_SEEDS:",
            "    b_re = load_sr(BASELINE_DIR / f'rebrac_seed{seed}.json')",
            "    b_fq = load_sr(BASELINE_DIR / f'fql_seed{seed}.json')",
            "    q1   = load_sr(test_json('rebrac', seed))",
            "    rows.append((seed, b_re, b_fq, q1))",
            "",
            "print(f'{\"seed\":<6}{\"baseline rebrac\":<18}{\"baseline fql\":<14}{\"Q1 rebrac (β2=0)\":<18}{\"Δ Q1−baseline\":<14}')",
            "for seed, b_re, b_fq, q1 in rows:",
            "    if q1 is not None and b_re is not None:",
            "        d = q1 - b_re",
            "        print(f'{seed:<6}{b_re:<18.3f}{(b_fq if b_fq else float(\"nan\")):<14.3f}{q1:<18.3f}{d:+.3f}')",
            "    else:",
            "        print(f'{seed:<6}MISSING')",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Lemma verdict (auto)",
            "",
            "Decision rule:",
            "- **H1 PASS**: mean(Q1_rebrac) ≥ mean(baseline_fql) − 0.05  AND  "
            "mean(Q1_rebrac) − mean(baseline_rebrac) ≥ +0.10",
            "- **H1 PARTIAL**: gain ≥ +0.05 but not closing the gap → critic_penalty contributes but not solely",
            "- **H1 FAIL**: gain < +0.05 → other path dominates,run Q1b",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            f"BASELINE_DIR = {BASELINE_DIR_LITERAL}",
            "",
            "def load_sr(path):",
            "    return float(json.loads(path.read_text())['eval_success_rate']) if path.exists() else None",
            "",
            "b_re = [load_sr(BASELINE_DIR / f'rebrac_seed{s}.json') for s in TRAIN_SEEDS]",
            "b_fq = [load_sr(BASELINE_DIR / f'fql_seed{s}.json')    for s in TRAIN_SEEDS]",
            "q1   = [load_sr(test_json('rebrac', s)) for s in TRAIN_SEEDS]",
            "",
            "assert all(x is not None for x in b_re), 'baseline rebrac missing'",
            "assert all(x is not None for x in b_fq), 'baseline fql missing'",
            "assert all(x is not None for x in q1),   'Q1 rebrac missing'",
            "",
            "m_re = mean(b_re); m_fq = mean(b_fq); m_q1 = mean(q1)",
            "gain    = m_q1 - m_re",
            "gap_close = m_q1 - m_fq  # negative if still below FQL",
            "",
            "print(f'baseline ReBRAC  μ = {m_re:.3f}')",
            "print(f'baseline FQL     μ = {m_fq:.3f}')",
            "print(f'Q1 ReBRAC (β2=0) μ = {m_q1:.3f}')",
            "print(f'gain  (Q1 - base) = {gain:+.3f}')",
            "print(f'remaining gap to FQL = {gap_close:+.3f}')",
            "",
            "if gain >= 0.10 and gap_close >= -0.05:",
            "    print('\\nVERDICT: H1 PASS — critic_penalty inflation IS the dominant mechanism.')",
            "elif gain >= 0.05:",
            "    print('\\nVERDICT: H1 PARTIAL — critic_penalty contributes but another path also matters.')",
            "    print('  → consider running Q1b (--actor-penalty-coef 0.0 --critic-penalty-coef 0.0).')",
            "else:",
            "    print('\\nVERDICT: H1 FAIL — critic_penalty is NOT the dominant mechanism.')",
            "    print('  → run Q1b or revise diagnostic hypothesis.')",
        ),
    ]


def section_8() -> dict:
    return md(
        "## 8. Report checklist",
        "",
        "完成后回写主诊断文档:",
        "- [ ] 写入 `docs/fql_succession_p2_mechanism_diagnostic.md` §8 Q1 子段 — "
        "记录 verdict 字符串 + 4 个数字(b_re μ, b_fq μ, q1 μ, gain)",
        "- [ ] 把 results/fql_succession/p2/m_uni_noise_q1_critic_pen0/test/*.json 4 个 "
        "(2 seeds × 1 algo,actually 2 files) sync 回 local repo",
        "- [ ] (可选)在 `docs/assets/fql_succession_p2/` 加一张 4-bar 对比图:"
        "baseline_rebrac / baseline_fql / Q1_rebrac × 2 seeds",
        "",
        "如果 verdict = PASS,paper 改写后 §3.3 'mechanism' 一节可以直接引用 "
        "「Q1 ablation: setting critic_bc_coef=0 lifts ReBRAC from 0.71 to X.XX on "
        "M-uni-noise, closing 95%+ of the FQL gap」作为 mechanism causality 的 "
        "closed-form 证据。",
    )


def build() -> dict:
    cells = [section_0()]
    cells.extend(section_1())
    cells.extend(section_2())
    cells.extend(section_3())
    cells.extend(section_4())
    cells.extend(section_5())
    cells.extend(section_6())
    cells.extend(section_7())
    cells.append(section_8())
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    # quick stat
    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"  cells: {len(nb['cells'])} total ({md_cells} md, {code_cells} code)")


if __name__ == "__main__":
    main()
