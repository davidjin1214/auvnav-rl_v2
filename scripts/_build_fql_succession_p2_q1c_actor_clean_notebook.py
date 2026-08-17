"""Build the Q1c actor-penalty-on-CLEAN ablation notebook for FQL Succession P2.

Q1c is the *fairness/robustness* follow-up forced by Q1b's surprising result.

    Q1b set actor β1 4.0 → 1.0 on the NOISY M-uni-noise data and ReBRAC jumped
    0.705 → 0.940 — it not only recovered but **overtook FQL (0.910)**. So the
    +20.5 pp "FQL > ReBRAC on noisy data" advantage is largely a ReBRAC
    *mis-tuning* artifact (β1=4.0 is too strong an anchor to a noisy target),
    NOT a structural FQL advantage.

    That reframes the whole paper claim. The deciding question is now:
    **does β1=1.0 cost anything on CLEAN data?**

    H_robust (FQL survives, modestly): β1=1.0 DEGRADES clean performance
        (drops below the β1=4.0 baseline 0.885 and below FQL clean 0.855).
        → No single ReBRAC β1 spans the noise axis: it needs β1=4.0 on clean,
          β1=1.0 on noisy. FQL's single FROZEN config is decent on both.
        → Defensible claim: FQL provides hyperparameter robustness across the
          noise axis without per-dataset tuning.

    H_collapse (FQL claim dies): β1=1.0 is FINE on clean (≈ 0.885, drop < 0.03).
        → β1=1.0 dominates β1=4.0 everywhere (≥ clean, ≫ noisy) AND matches/beats
          FQL on both axes → ReBRAC merely had a bad default. FQL offers no
          robustness advantage, only a tuning convenience. Superiority claim
          collapses to "less tuning needed".

Design:
- Dataset: REUSE existing E-uni clean (privileged σ=0, no new collection).
- Algo: ReBRAC only, `--actor-penalty-coef 1.0` (β1 4.0→1.0), critic β2=2.0.
- Seeds: [42, 0]. 2 runs. ~1h Colab L4.
- The §6 noise-axis table loads ALL live numbers (E-uni β1=4.0, Q1c β1=1.0,
  FQL clean from e_uni; M-uni-noise β1=4.0, Q1b β1=1.0, FQL noisy) to compute a
  worst-case-over-noise robustness comparison.

Output: notebooks/fql_succession_p2_q1c_actor_pen1_clean.ipynb
Regenerate: python -m scripts._build_fql_succession_p2_q1c_actor_clean_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_q1c_actor_pen1_clean.ipynb")

CELL_ID_BASE = "e_uni"
DATASET = "offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000"
SUFFIX = "_q1c_actor_pen1_clean"  # no collision with sprint-0 / Q1 / Q1b

# Reference result dirs loaded by §3 / §6 / §7
EU_DIR = "results/fql_succession/p2/e_uni/test"            # β1=4.0 clean + FQL clean
MUN_DIR = "results/fql_succession/p2/m_uni_noise/test"     # β1=4.0 noisy + FQL noisy
Q1B_DIR = "results/fql_succession/p2/m_uni_noise_q1b_actor_pen1/test"  # β1=1.0 noisy


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **Q1c actor-penalty on CLEAN data** (E-uni)",
        "",
        "**Purpose**: the *fairness/robustness* test forced by **Q1b's result**. "
        "See `docs/fql_succession_p2_mechanism_diagnostic.md` §9.6.",
        "",
        "## Background — what Q1b just showed",
        "",
        "Q1b set actor `β1 4.0 → 1.0` on the **noisy** M-uni-noise data: ReBRAC went "
        "**0.705 → 0.940**, not only recovering but **overtaking FQL (0.910)**. "
        "So the +20.5 pp \"FQL > ReBRAC on noisy data\" gap is largely a ReBRAC "
        "**mis-tuning artifact** (β1=4.0 is too strong an anchor to a noisy target), "
        "**not** a structural FQL advantage.",
        "",
        "That reframes the paper claim. The deciding question:",
        "",
        "> **Does β1=1.0 cost anything on CLEAN data?**",
        "",
        "The baseline β1=4.0 was (presumably) tuned for clean data, where ReBRAC is "
        "strong (E-uni 0.885). If β1=1.0 *degrades* clean performance, then no single "
        "β1 spans the noise axis and FQL's frozen config has real robustness value. "
        "If β1=1.0 is *fine* on clean, then β1=4.0 was simply a bad default and a "
        "properly-tuned ReBRAC dominates FQL everywhere.",
        "",
        "## Hypothesis (this notebook)",
        "",
        "- **H_robust** (FQL survives, modestly): β1=1.0 **degrades** clean "
        "(drop ≥ 5 pp below β1=4.0's 0.885) → no single ReBRAC β1 works on both "
        "noise levels; FQL's single frozen config does → **robustness claim holds**.",
        "- **H_collapse** (FQL claim dies): β1=1.0 is **fine** on clean (drop < 3 pp) "
        "→ β1=1.0 dominates β1=4.0 across the axis and matches/beats FQL → ReBRAC "
        "just needed a better default → **superiority claim collapses**.",
        "",
        "## Design",
        "",
        "| Knob | E-uni baseline | M-uni-noise Q1b | **Q1c (this)** |",
        "|---|---|---|---|",
        "| `--actor-penalty-coef` | 4.0 | 1.0 | **1.0** ⚠️ |",
        "| `--critic-penalty-coef` | 2.0 | 2.0 | 2.0 |",
        "| `--critic-layernorm` | on | on | on |",
        "| `--no-actor-layernorm` | on | on | on |",
        "| dataset | E-uni **clean** (σ=0) | M-uni-noise (σ=0.5) | E-uni **clean** (σ=0) |",
        "| seeds | [42, 0] | [42, 0] | [42, 0] |",
        "",
        "**2 runs total**. Wallclock ≈ 1 h on Colab L4.",
        "",
        "## Out of scope",
        "",
        "- Intermediate β1 (e.g. 2.0) sweep — only if Q1c lands in the gray band.",
        "- FQL re-run (E-uni FQL baseline already at 0.855).",
        "- New collection (reuse existing E-uni clean dataset).",
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
            "`--actor-penalty-coef 4.0` → **1.0**(critic β2=2.0 不变),数据集换成 "
            "**E-uni clean**(σ=0)。",
            "",
            "checkpoint / results 用 `_q1c_actor_pen1_clean` 后缀,避免覆盖 sprint-0 / Q1 / Q1b。",
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
            "# Q1c ablation: actor_penalty_coef 4.0 → 1.0, on CLEAN data (critic β2=2.0 unchanged)",
            "REBRAC_Q1C_FLAGS = (",
            "    '--actor-penalty-coef 1.0 --critic-penalty-coef 2.0 '",
            "    '--critic-layernorm --no-actor-layernorm'",
            ")",
            "",
            "TRAIN_SEEDS = [42, 0]",
            "ALGOS = ['rebrac']  # FQL not re-run (E-uni FQL baseline already at 0.855)",
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
            "print('REBRAC_Q1C_FLAGS =', REBRAC_Q1C_FLAGS)",
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
            "2. metadata 是 **CLEAN**(σ=0.0)+ unimodal(single privileged)+ ~86685 transitions",
            "3. manifest + flow exist",
            "4. baseline E-uni / M-uni-noise / Q1b test json 存在(供 §6 robustness 表) — 不强制",
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
            "# E-uni MUST be clean (σ=0) and unimodal to make Q1c a clean-data fairness test.",
            "noise = float(meta.get('action_noise_std', float('nan')))",
            "assert abs(noise) < 1e-6, (",
            "    f'expected action_noise_std==0.0 (clean) but got {noise} '",
            "    '— Q1c must run on CLEAN data to test the β1=1.0 trade-off.'",
            ")",
            "exp_n_tx_approx = 86685",
            "n_tx = int(meta.get('num_transitions', 0))",
            "assert abs(n_tx - exp_n_tx_approx) < 5000, f'tx count {n_tx} far from expected {exp_n_tx_approx}'",
            "",
            "assert Path(FLOW).is_file(), f'flow missing: {FLOW}'",
            "assert Path(MANIFEST).is_file(), f'manifest missing: {MANIFEST}'",
            "",
            "# Reference baselines (printed if available)",
            f"EU_DIR  = Path('{EU_DIR}')",
            f"MUN_DIR = Path('{MUN_DIR}')",
            f"Q1B_DIR = Path('{Q1B_DIR}')",
            "def _sr(p): return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "for s in TRAIN_SEEDS:",
            "    print(f'E-uni  β1=4.0 rebrac seed={s}  SR =', _sr(EU_DIR / f'rebrac_seed{s}.json'))",
            "    print(f'E-uni        fql    seed={s}  SR =', _sr(EU_DIR / f'fql_seed{s}.json'))",
            "    print(f'Q1b    β1=1.0 rebrac seed={s}  SR =', _sr(Q1B_DIR / f'rebrac_seed{s}.json'), '(noisy)')",
            "",
            "print('\\nPre-flight PASS — ready to train.')",
        ),
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. Train (2 runs)",
            "",
            "ReBRAC × seeds [42, 0],`--actor-penalty-coef 1.0` on **clean** E-uni,"
            "skip-resume on `agent_final.pt`(与 sprint-0 一致的 3 阶段恢复)。",
            "",
            "**`--skip-final-eval`**:在线 eval 跑 50 ep 即可,test eval 100 ep 走 §5。",
        ),
        code(
            "import time",
            "import shutil",
            "from pathlib import Path",
            "",
            "for algo in ALGOS:",
            "    ALGO_FLAGS = REBRAC_Q1C_FLAGS  # only ReBRAC in this notebook",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        sd_str = str(sd)",
            "        if (sd / 'agent_final.pt').exists():",
            "            print(f'[skip-train] {algo} seed={seed} (agent_final.pt exists at {sd})')",
            "            continue",
            "        sd.mkdir(parents=True, exist_ok=True)",
            "        t0 = time.time()",
            "        print(f'>>> train {algo} (Q1c actor_pen=1.0, clean) seed={seed} → {sd}')",
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
            "## 6. Noise-axis robustness table",
            "",
            "把 Q1c(β1=1.0 clean)放进完整的 **3 config × 2 noise** 网格,每个 config 计算 "
            "**worst-case-over-noise**(min over clean/noisy)。这是 robustness claim 的核心量:",
            "config 在两个噪声水平里最差的那个表现。",
            "",
            "| config | clean (E-uni) | noisy (M-uni-noise) | worst-case |",
            "|---|---|---|---|",
            "| ReBRAC β1=4.0 | 0.885 | 0.705 | 0.705 |",
            "| ReBRAC β1=1.0 | **Q1c (this)** | 0.940 (Q1b) | min(Q1c, 0.940) |",
            "| FQL (frozen) | 0.855 | 0.910 | 0.855 |",
            "",
            "若 FQL 的 worst-case(0.855)> 两个 ReBRAC config 的 worst-case → FQL 是单配置下"
            "最 noise-robust 的算法(论文能站住的卖点)。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            f"EU_DIR  = Path('{EU_DIR}')",
            f"MUN_DIR = Path('{MUN_DIR}')",
            f"Q1B_DIR = Path('{Q1B_DIR}')",
            "",
            "def mu(d, algo):",
            "    vals = []",
            "    for s in TRAIN_SEEDS:",
            "        p = d / f'{algo}_seed{s}.json'",
            "        if not p.exists():",
            "            return None",
            "        vals.append(float(json.loads(p.read_text())['eval_success_rate']))",
            "    return mean(vals)",
            "",
            "re40_clean = mu(EU_DIR, 'rebrac')   # β1=4.0 clean",
            "fql_clean  = mu(EU_DIR, 'fql')      # FQL clean",
            "re40_noisy = mu(MUN_DIR, 'rebrac')  # β1=4.0 noisy",
            "fql_noisy  = mu(MUN_DIR, 'fql')     # FQL noisy",
            "re10_noisy = mu(Q1B_DIR, 'rebrac')  # β1=1.0 noisy (Q1b)",
            "re10_clean = mu(TEST_DIR, 'rebrac') # β1=1.0 clean (Q1c — this)",
            "",
            "def f(x): return f'{x:.3f}' if x is not None else '  --  '",
            "def wc(a, b):",
            "    xs = [v for v in (a, b) if v is not None]",
            "    return min(xs) if xs else None",
            "",
            "rows = [",
            "    ('ReBRAC β1=4.0', re40_clean, re40_noisy),",
            "    ('ReBRAC β1=1.0', re10_clean, re10_noisy),",
            "    ('FQL (frozen)',  fql_clean,  fql_noisy),",
            "]",
            "print(f'{\"config\":<16}{\"clean\":<10}{\"noisy\":<10}{\"worst-case\":<12}')",
            "print('-' * 48)",
            "for name, c, n in rows:",
            "    print(f'{name:<16}{f(c):<10}{f(n):<10}{f(wc(c, n)):<12}')",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Verdict (auto)",
            "",
            "Decision rule(以 Q1c clean vs E-uni β1=4.0 baseline 0.885 的 drop 为主):",
            "- **H_robust PASS**: drop ≥ +0.05(β1=1.0 在 clean 上掉 ≥5pp)→ 没有单一 β1 "
            "同时吃下两种噪声;FQL 单配置 worst-case 更高 → **robustness claim 成立**。",
            "- **GRAY**: 0.03 ≤ drop < 0.05 → 部分 trade-off;考虑扫中间 β1(=2.0)或扩 n=4。",
            "- **H_collapse / FAIL**: drop < 0.03(β1=1.0 在 clean 上几乎不掉)→ β1=1.0 "
            "处处 dominate β1=4.0 且匹配/超过 FQL → **superiority claim 崩,只剩调参便利性**。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            f"EU_DIR  = Path('{EU_DIR}')",
            f"MUN_DIR = Path('{MUN_DIR}')",
            f"Q1B_DIR = Path('{Q1B_DIR}')",
            "",
            "def mu(d, algo):",
            "    vals = []",
            "    for s in TRAIN_SEEDS:",
            "        p = d / f'{algo}_seed{s}.json'",
            "        if not p.exists():",
            "            return None",
            "        vals.append(float(json.loads(p.read_text())['eval_success_rate']))",
            "    return mean(vals)",
            "",
            "re40_clean = mu(EU_DIR, 'rebrac')",
            "fql_clean  = mu(EU_DIR, 'fql')",
            "re40_noisy = mu(MUN_DIR, 'rebrac')",
            "fql_noisy  = mu(MUN_DIR, 'fql')",
            "re10_noisy = mu(Q1B_DIR, 'rebrac')",
            "re10_clean = mu(TEST_DIR, 'rebrac')",
            "",
            "assert re10_clean is not None, 'Q1c rebrac results missing'",
            "assert re40_clean is not None, 'E-uni baseline rebrac missing'",
            "",
            "drop = re40_clean - re10_clean   # positive = β1=1.0 degrades clean",
            "print(f'E-uni ReBRAC β1=4.0 (clean) μ = {re40_clean:.3f}')",
            "print(f'Q1c   ReBRAC β1=1.0 (clean) μ = {re10_clean:.3f}')",
            "print(f'clean drop (β1=4.0 − β1=1.0)  = {drop:+.3f}')",
            "if fql_clean is not None:",
            "    print(f'(ref) FQL clean μ             = {fql_clean:.3f}')",
            "",
            "# worst-case-over-noise per config (robustness summary)",
            "def wc(a, b):",
            "    xs = [v for v in (a, b) if v is not None]",
            "    return min(xs) if xs else None",
            "wc_b40 = wc(re40_clean, re40_noisy)",
            "wc_b10 = wc(re10_clean, re10_noisy)",
            "wc_fql = wc(fql_clean, fql_noisy)",
            "print(f'\\nworst-case-over-noise:  β1=4.0 = {wc_b40}, β1=1.0 = {wc_b10}, FQL = {wc_fql}')",
            "",
            "if drop >= 0.05:",
            "    print('\\nVERDICT: H_robust PASS — β1=1.0 trades off clean performance.')",
            "    print('  No single ReBRAC β1 spans the noise axis (needs 4.0 clean / 1.0 noisy).')",
            "    print('  paper: FQL provides hyperparameter robustness across the noise axis')",
            "    print('         (single frozen config, no per-dataset tuning).')",
            "    if wc_fql is not None and wc_b10 is not None and wc_b40 is not None and wc_fql > max(wc_b10, wc_b40):",
            "        print('  STRONG: FQL has the highest worst-case-over-noise of all three configs.')",
            "elif drop >= 0.03:",
            "    print('\\nVERDICT: GRAY — partial clean trade-off (3–5 pp).')",
            "    print('  Consider an intermediate β1=2.0 sweep or expand to n=4 seeds before concluding.')",
            "else:",
            "    print('\\nVERDICT: H_collapse / FAIL — β1=1.0 costs ~nothing on clean.')",
            "    print('  β1=1.0 dominates β1=4.0 across the noise axis AND matches/beats FQL.')",
            "    print('  → ReBRAC merely had a bad default; FQL superiority claim collapses to')",
            "    print('    a tuning-convenience argument. Reframe the paper away from \"FQL wins\".')",
        ),
    ]


def section_8() -> dict:
    return md(
        "## 8. Report checklist",
        "",
        "完成后回写主诊断文档:",
        "- [ ] 写入 `docs/fql_succession_p2_mechanism_diagnostic.md` §9.6 — "
        "记录 Q1c verdict + clean drop + worst-case-over-noise 三元组",
        "- [ ] sync `results/fql_succession/p2/e_uni_q1c_actor_pen1_clean/test/*.json` (2 个) 回 local",
        "- [ ] 据 verdict 决定 N3/N4 的 claim 框定(robustness vs collapse)",
        "",
        "Paper 影响:",
        "- **H_robust PASS**: 论文 claim 框定为 **hyperparameter robustness across "
        "the noise axis** — FQL 单配置在 clean/noisy 都 OK,ReBRAC 要 noise-aware 调 β1。"
        "这是诚实且可发表的温和 claim。",
        "- **H_collapse / FAIL**: \"FQL > ReBRAC\" superiority 站不住,只剩 \"FQL 少调参\"。"
        "需要严肃重审 P2 是否还值得作为 paper 主线,或转为 negative/null result 报告。",
        "- **GRAY**: 加 β1=2.0 中间点或 n=4,再定。",
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
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"  cells: {len(nb['cells'])} total ({md_cells} md, {code_cells} code)")


if __name__ == "__main__":
    main()
