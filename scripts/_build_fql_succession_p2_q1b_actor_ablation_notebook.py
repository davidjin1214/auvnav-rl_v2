"""Build the Q1b actor-penalty ablation notebook for FQL Succession P2.

Q1b mechanism lemma (the decisive follow-up after Q1 FAILED):

    Q1 ruled out the critic BC penalty: setting critic_penalty_coef=0 gave
    +0.01 SR (no recovery). The corrected hypothesis (diagnostic §9) is that the
    **actor BC regression to raw noisy actions** (β1=4.0) is the bottleneck.

    H0 (null): weakening the actor BC anchor (β1 4.0 → 1.0) does NOT recover
               ReBRAC on M-uni-noise → the issue is BC *target quality*
               (denoising), which only FQL provides.
    H1 (alt) : β1 1.0 lifts ReBRAC SR from ~0.71 → toward FQL (0.91) → the
               *strength* of BC to a noisy target is the binding constraint.

Either outcome sharpens the paper claim:
- H1 PASS  → "strong BC to a noisy target binds"; FQL wins partly by having a
             milder effective anchor on clean (denoised) targets.
- H0 (FAIL)→ "BC target quality binds"; FQL's flow-denoised target is the
             essential mechanism, not anchor strength. (Stronger FQL story.)

Design:
- Dataset: REUSE existing M-uni-noise (no new collection).
- Algo: ReBRAC only, `--actor-penalty-coef 1.0` (β1 4.0→1.0), critic β2=2.0 unchanged.
- Seeds: [42, 0]. 2 runs. ~1h Colab L4.
- β1=0.0 (pure-TD3) deliberately NOT included — risks extrapolation collapse and
  would confound the test. If Q1b is ambiguous, add β1=0.0 as a separate run.

Output: notebooks/fql_succession_p2_q1b_actor_penalty_ablation.ipynb
Regenerate: python -m scripts._build_fql_succession_p2_q1b_actor_ablation_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_q1b_actor_penalty_ablation.ipynb")

CELL_ID_BASE = "m_uni_noise"
DATASET = "offline_data/fql_succession/m_uni_noise_eps0p5_1000"
SUFFIX = "_q1b_actor_pen1"  # checkpoint/results suffix, no collision with sprint-0 / Q1

BASELINE_DIR_LITERAL = f"Path('results/fql_succession/p2/{CELL_ID_BASE}/test')"


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **Q1b actor-penalty ablation** (M-uni-noise)",
        "",
        "**Purpose**: Decisive mechanism test after **Q1 FAILED**. See "
        "`docs/fql_succession_p2_mechanism_diagnostic.md` §9.",
        "",
        "## Background — what Q1 settled",
        "",
        "Q1 set `critic_penalty_coef = 0.0` and got **+0.01 SR** (no recovery): "
        "the critic BC penalty raised `mean_q` (−150→−123, confirming the coef "
        "took effect) but did **not** improve the policy. So the critic-side "
        "penalty is *not* the bottleneck.",
        "",
        "The corrected hypothesis: the **actor BC term** "
        "`actor_loss = −λ·Q(s,π(s)) + β1·‖π(s) − a‖²` with **β1=4.0** strongly drags "
        "the deterministic actor toward the **raw noisy** behavior actions "
        "`a = π*(s) + ε` (bc_loss stuck at the noise floor ~0.134). FQL avoids this "
        "by regressing its student to the **flow-denoised** `a_teacher = integrate(s)`.",
        "",
        "## Hypothesis (this notebook)",
        "",
        "- **H1 (alt)**: weakening the anchor (β1 4.0 → 1.0) lifts ReBRAC SR from "
        "~0.71 toward FQL (0.91) → *strength* of BC to a noisy target binds.",
        "- **H0 (null)**: β1=1.0 does **not** recover ReBRAC → the issue is BC "
        "*target quality* (denoising), which only FQL provides. (Stronger FQL story.)",
        "",
        "Both outcomes sharpen the paper:",
        "- H1 PASS → FQL wins partly via a milder effective anchor on clean targets.",
        "- H0/FAIL → FQL's flow-denoised target is the essential mechanism, not "
        "anchor strength.",
        "",
        "## Design",
        "",
        "| Knob | Sprint-0 ReBRAC (baseline) | Q1 (done) | **Q1b (this)** |",
        "|---|---|---|---|",
        "| `--actor-penalty-coef` | 4.0 | 4.0 | **1.0** ⚠️ |",
        "| `--critic-penalty-coef` | 2.0 | **0.0** | 2.0 |",
        "| `--critic-layernorm` | on | on | on |",
        "| `--no-actor-layernorm` | on | on | on |",
        "| dataset | `m_uni_noise_eps0p5_1000` | same | **same** |",
        "| seeds | [42, 0] | [42, 0] | [42, 0] |",
        "",
        "**2 runs total**. Wallclock ≈ 1 h on Colab L4.",
        "",
        "## Out of scope",
        "",
        "- β1=0.0 (pure-TD3) — risks extrapolation collapse; add separately only if "
        "Q1b is ambiguous.",
        "- FQL re-run (baseline already at 0.91).",
        "- New collection (reuse existing M-uni-noise).",
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
            "`--actor-penalty-coef 4.0` → **1.0**(critic β2=2.0 不变)。",
            "",
            "checkpoint / results 用 `_q1b_actor_pen1` 后缀,避免覆盖 sprint-0 / Q1。",
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
            "# Q1b ablation: actor_penalty_coef 4.0 → 1.0 (critic β2=2.0 unchanged)",
            "REBRAC_Q1B_FLAGS = (",
            "    '--actor-penalty-coef 1.0 --critic-penalty-coef 2.0 '",
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
            "print('CELL_ID          =', CELL_ID)",
            "print('DATASET          =', DATASET)",
            "print('REBRAC_Q1B_FLAGS =', REBRAC_Q1B_FLAGS)",
            "print('TRAIN_SEEDS      =', TRAIN_SEEDS, '  ALGOS =', ALGOS)",
            "print('CHECKPOINT_ROOT  =', CHECKPOINT_ROOT)",
            "print('RESULTS_ROOT     =', RESULTS_ROOT)",
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
            "4. baseline sprint-0 + Q1 test json 存在(供 §6 对比) — 不强制",
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
            "Q1_DIR = Path('results/fql_succession/p2/m_uni_noise_q1_critic_pen0/test')",
            "for algo in ('rebrac', 'fql'):",
            "    for s in TRAIN_SEEDS:",
            "        p = BASELINE_DIR / f'{algo}_seed{s}.json'",
            "        if p.exists():",
            "            d = json.loads(p.read_text())",
            "            print(f'baseline {algo} seed={s}  SR = {d[\"eval_success_rate\"]:.3f}')",
            "        else:",
            "            print(f'baseline {algo} seed={s}  (no file)')",
            "for s in TRAIN_SEEDS:",
            "    p = Q1_DIR / f'rebrac_seed{s}.json'",
            "    if p.exists():",
            "        d = json.loads(p.read_text())",
            "        print(f'Q1 rebrac seed={s}      SR = {d[\"eval_success_rate\"]:.3f}')",
            "",
            "print('\\nPre-flight PASS — ready to train.')",
        ),
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. Train (2 runs)",
            "",
            "ReBRAC × seeds [42, 0],`--actor-penalty-coef 1.0`,skip-resume on "
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
            "    ALGO_FLAGS = REBRAC_Q1B_FLAGS  # only ReBRAC in this notebook",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        sd_str = str(sd)",
            "        if (sd / 'agent_final.pt').exists():",
            "            print(f'[skip-train] {algo} seed={seed} (agent_final.pt exists at {sd})')",
            "            continue",
            "        sd.mkdir(parents=True, exist_ok=True)",
            "        t0 = time.time()",
            "        print(f'>>> train {algo} (Q1b actor_pen=1.0) seed={seed} → {sd}')",
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
            "结果写 `results/.../test/<algo>_seed<S>.json`;同时 mirror 4 个 small files。",
            "",
            "CLI 与 sprint-0 §5 完全一致(`--episodes`, `--worker-device cpu`, `--device cuda`)。",
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
            "## 6. Four-way comparison",
            "",
            "Q1b 与 sprint-0 baseline / FQL / Q1 同一 dataset 对照:",
            "",
            "| variant | actor β1 | critic β2 | role |",
            "|---|---:|---:|---|",
            "| sprint-0 ReBRAC | 4.0 | 2.0 | baseline (0.705) |",
            "| sprint-0 FQL | — | — | upper bound (0.910) |",
            "| Q1 ReBRAC | 4.0 | 0.0 | critic-side (0.715, FAIL) |",
            "| **Q1b ReBRAC** | **1.0** | 2.0 | actor-side (this notebook) |",
            "",
            "Primary metric = `eval_success_rate` (100 ep).",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "",
            f"BASELINE_DIR = {BASELINE_DIR_LITERAL}",
            "Q1_DIR = Path('results/fql_succession/p2/m_uni_noise_q1_critic_pen0/test')",
            "",
            "def load_sr(path):",
            "    if not path.exists():",
            "        return None",
            "    return float(json.loads(path.read_text())['eval_success_rate'])",
            "",
            "print(f'{\"seed\":<6}{\"base rebrac\":<14}{\"base fql\":<12}{\"Q1 (β2=0)\":<12}{\"Q1b (β1=1)\":<12}{\"Δ Q1b−base\":<12}')",
            "for seed in TRAIN_SEEDS:",
            "    b_re = load_sr(BASELINE_DIR / f'rebrac_seed{seed}.json')",
            "    b_fq = load_sr(BASELINE_DIR / f'fql_seed{seed}.json')",
            "    q1   = load_sr(Q1_DIR / f'rebrac_seed{seed}.json')",
            "    q1b  = load_sr(test_json('rebrac', seed))",
            "    def f(x): return f'{x:.3f}' if x is not None else '  --  '",
            "    d = (q1b - b_re) if (q1b is not None and b_re is not None) else float('nan')",
            "    print(f'{seed:<6}{f(b_re):<14}{f(b_fq):<12}{f(q1):<12}{f(q1b):<12}{d:+.3f}')",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Lemma verdict (auto)",
            "",
            "Decision rule:",
            "- **H1 PASS** (anchor strength binds): mean(Q1b) − mean(baseline) ≥ +0.10 "
            "AND mean(Q1b) ≥ mean(FQL) − 0.05.",
            "- **H1 PARTIAL**: gain ≥ +0.05 但没闭合到 FQL → 强度部分相关,target quality 也重要。",
            "- **H0 / FAIL** (target quality binds): gain < +0.05 → 削弱 anchor 没用,"
            "FQL 的去噪 target 才是本质 → **更强的 FQL 故事**。",
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
            "q1b  = [load_sr(test_json('rebrac', s)) for s in TRAIN_SEEDS]",
            "",
            "assert all(x is not None for x in b_re), 'baseline rebrac missing'",
            "assert all(x is not None for x in b_fq), 'baseline fql missing'",
            "assert all(x is not None for x in q1b),  'Q1b rebrac missing'",
            "",
            "m_re = mean(b_re); m_fq = mean(b_fq); m_q1b = mean(q1b)",
            "gain      = m_q1b - m_re",
            "gap_close = m_q1b - m_fq",
            "",
            "print(f'baseline ReBRAC (β1=4.0) μ = {m_re:.3f}')",
            "print(f'baseline FQL            μ = {m_fq:.3f}')",
            "print(f'Q1b ReBRAC (β1=1.0)     μ = {m_q1b:.3f}')",
            "print(f'gain (Q1b − baseline)     = {gain:+.3f}')",
            "print(f'remaining gap to FQL      = {gap_close:+.3f}')",
            "",
            "if gain >= 0.10 and gap_close >= -0.05:",
            "    print('\\nVERDICT: H1 PASS — actor BC anchor *strength* to noisy target binds.')",
            "    print('  paper: FQL wins partly via milder effective anchor on denoised targets.')",
            "elif gain >= 0.05:",
            "    print('\\nVERDICT: H1 PARTIAL — anchor strength contributes but target quality also matters.')",
            "else:",
            "    print('\\nVERDICT: H0 / FAIL — weakening anchor did NOT help.')",
            "    print('  → BC *target quality* (denoising) is the essential mechanism — stronger FQL story.')",
            "    print('  → optionally add β1=0.0 pure-TD3 run to confirm extrapolation floor.')",
        ),
    ]


def section_8() -> dict:
    return md(
        "## 8. Report checklist",
        "",
        "完成后回写主诊断文档:",
        "- [ ] 写入 `docs/fql_succession_p2_mechanism_diagnostic.md` §9.4 — "
        "记录 Q1b verdict + 4 个数字 (b_re μ, b_fq μ, q1b μ, gain)",
        "- [ ] sync `results/fql_succession/p2/m_uni_noise_q1b_actor_pen1/test/*.json` (2 个) 回 local",
        "- [ ] 在 train_log 抓 Q1b 的 bc_loss + mean_q,看 β1=1.0 时 bc_loss 是否仍卡噪声地板",
        "",
        "Paper 影响:",
        "- **H1 PASS**: §3 mechanism 一节写「actor BC anchor strength to noisy "
        "behavior actions is the binding constraint; FQL's distillation to a "
        "denoised target acts as a milder, cleaner anchor」。",
        "- **H0/FAIL**: §3 写「even at matched anchor strength, ReBRAC cannot recover "
        "— the essential mechanism is BC *target quality*: FQL regresses to a "
        "flow-denoised reconstruction E[a|s], ReBRAC to raw noisy a」。这是最强的 "
        "FQL claim,直接把优势归到 flow-matching 的去噪能力。",
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
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")
    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"  cells: {len(nb['cells'])} total ({md_cells} md, {code_cells} code)")


if __name__ == "__main__":
    main()
