"""Build the C-1 FQL `distill_alpha_bc` rescue-sweep notebook (FQL Succession P2).

C-1 is the **fairness rematch** mandated by the mechanism diagnostic §9.10.

    Q1c (§9.7) showed ReBRAC β1=1.0 *dominates* FQL on every cell
    (worst-case-over-noise 0.910 > FQL 0.855), killing the "FQL wins" claim.
    BUT Gate B tuned ReBRAC's β1 while FREEZING FQL's `distill_alpha_bc=1.0` —
    an asymmetric-tuning gap (§9.7 fairness caveat). C gives FQL the same
    courtesy: sweep its own BC-anchor knob, one honest chance to overturn the
    verdict before we finalize a negative/mechanism result (B+A fallback).

    FQL's knob: actor_loss = −λ·Q̄ + distill_alpha_bc·‖a_student − a_teacher‖²
    (fql.py:472–474). It is the direct analog of ReBRAC's β1, except it anchors
    to the flow-DENOISED teacher action rather than raw data. Default frozen 1.0.

    The bar FQL must clear: worst-case-over-noise > 0.910 (ReBRAC β1=1.0).
    FQL's worst-case is its CLEAN cell (0.855), so clean is the binding axis to
    improve → this notebook sweeps alpha on CLEAN E-uni only (staged: noisy axis
    = Phase C-2 follow-up, built only if a candidate emerges — user's choice).

    Direction is genuinely uncertain: ReBRAC's lesson was "weaker anchor helps"
    (β1 4.0→1.0), but FQL anchors to a near-expert DENOISED teacher on clean
    data, where STRONGER anchoring (higher alpha) could help. So we sweep
    log-spaced both sides of 1.0.

Two job groups (8 FQL train+eval runs total):
  1. SWEEP   — distill_alpha_bc ∈ {0.3, 3.0, 10.0} × seeds [42, 0] = 6 runs,
               written to a fresh `e_uni_c1_fql_alpha_sweep` tree.
  2. TOP-UP  — distill_alpha_bc = 1.0 × seeds [1, 2] = 2 runs, EXTENDING the
               existing `e_uni` cell in-place to n=4 (§9.9: FQL-clean is the only
               statistically under-determined cell, swings 0.80↔0.91 at n=2).
               Config matches the original E-uni FQL runs EXACTLY (only seed
               differs) so it is a valid extension of that baseline.

Output: notebooks/fql_succession_p2_c1_fql_alpha_sweep.ipynb
Regenerate: python -m scripts._build_fql_succession_p2_c1_fql_alpha_sweep_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_c1_fql_alpha_sweep.ipynb")

# Clean E-uni dataset (same privileged σ=0 unimodal source as the FQL baseline).
DATASET = "offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000"

# Reference dirs loaded by §3 / §6 / §7.
EUNI_TEST = "results/fql_succession/p2/e_uni/test"                       # FQL α=1.0 baseline (42,0) + top-up (1,2)
Q1C_DIR = "results/fql_succession/p2/e_uni_q1c_actor_pen1_clean/test"    # ReBRAC β1=1.0 clean bar (0.910)


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **C-1: FQL `distill_alpha_bc` rescue sweep** (clean E-uni)",
        "",
        "**Purpose**: the *fairness rematch* (direction **C**) from "
        "`docs/fql_succession_p2_mechanism_diagnostic.md` **§9.10** — give FQL its own "
        "BC-anchor tuning, **one honest chance to overturn Q1c's verdict** before we "
        "finalize a negative/mechanism result.",
        "",
        "## Background — why FQL needs a rematch",
        "",
        "Q1c (§9.7) showed **ReBRAC β1=1.0 dominates FQL on every cell** "
        "(worst-case-over-noise **0.910 > FQL 0.855**), killing the \"FQL wins\" claim. "
        "**But** Gate B tuned ReBRAC's β1 while **freezing FQL's `distill_alpha_bc=1.0`** "
        "and never sweeping it — an **asymmetric-tuning gap** (§9.7 fairness caveat). "
        "C closes that gap.",
        "",
        "FQL's knob is the BC-distillation weight in",
        "",
        "> `actor_loss = −λ·Q̄ + distill_alpha_bc · ‖a_student − a_teacher‖²`  (fql.py:472–474)",
        "",
        "the direct analog of ReBRAC's β1 — except it anchors to the flow-**denoised** "
        "teacher action rather than raw data. CLI `--distill-alpha-bc`, default **1.0**.",
        "",
        "## The bar FQL must clear",
        "",
        "Worst-case-over-noise must exceed **0.910** (ReBRAC β1=1.0). FQL's worst-case is "
        "its **clean** cell (0.855), so **clean is the binding axis** to improve. This "
        "notebook sweeps alpha on **clean E-uni only** (staged — the noisy axis is "
        "**Phase C-2**, a follow-up notebook built *only if* a candidate emerges).",
        "",
        "Direction is genuinely uncertain: ReBRAC's lesson was *weaker* anchor helps "
        "(β1 4.0→1.0), but FQL anchors to a near-expert **denoised** teacher on clean "
        "data, where *stronger* anchoring could help → sweep **log-spaced both sides** of 1.0.",
        "",
        "## Hypothesis (this notebook, clean axis)",
        "",
        "- **RESCUE-PROMISING**: some α lifts FQL clean to **≥ 0.90** (within ~1 pp of the "
        "0.910 bar) → carry that α into **Phase C-2** (noisy) to confirm worst-case.",
        "- **RESCUE-WEAK**: best α clean in **[0.86, 0.90)** → improved but short; C-2 optional.",
        "- **RESCUE-FAIL**: no α beats FQL's own α=1.0 clean (~0.855) → FQL cannot clear the "
        "binding axis → **fall back to B+A** (honest mechanism/negative result).",
        "",
        "## Design — 8 FQL train+eval runs",
        "",
        "| group | `--distill-alpha-bc` | seeds | dataset | tree | runs |",
        "|---|---|---|---|---|---|",
        "| SWEEP | **0.3, 3.0, 10.0** | [42, 0] | E-uni **clean** (σ=0) | `e_uni_c1_fql_alpha_sweep` | 6 |",
        "| TOP-UP | 1.0 (frozen) | **[1, 2]** new | E-uni **clean** (σ=0) | extends `e_uni` → **n=4** | 2 |",
        "",
        "TOP-UP matches the original E-uni FQL config **exactly** (only seed differs) so it "
        "is a valid extension of that baseline (§9.9: FQL-clean is the sole under-determined "
        "cell, 0.80↔0.91 at n=2). Wallclock ≈ 30–45 min/run on L4 → ~4–6 h, 1–2 sessions.",
        "",
        "## Out of scope",
        "",
        "- **Noisy axis** (M-uni-noise) — Phase C-2 follow-up, only if C-1 finds a candidate.",
        "- ReBRAC re-run — the β1=1.0 clean bar (0.910) is already in "
        "`e_uni_q1c_actor_pen1_clean/test`.",
        "- No edits to Gate-B-frozen `auv_nav/{fql,rebrac}.py` (only the CLI value changes).",
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
            "共享参数与原 E-uni run **完全一致**(probe s0 / h4 / target 1.5 / cross_stream / "
            "arrival_v2 / 200k steps / batch 256 / eval-every 10k / eval-episodes 100)。"
            "FQL flags 也冻结,**唯一变动是 `--distill-alpha-bc`**。",
            "",
            "- SWEEP:α ∈ {0.3, 3.0, 10.0} × seeds [42, 0] → `e_uni_c1_fql_alpha_sweep` 树。",
            "- TOP-UP:α=1.0 × seeds **[1, 2]**(新)→ **就地扩** 现有 `e_uni` cell 到 n=4。",
        ),
        code(
            "from pathlib import Path",
            "",
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
            "# Matches the original E-uni FQL runs EXACTLY (so the α=1.0 top-up extends that baseline).",
            "TOTAL_STEPS    = 200_000",
            "BATCH_SIZE     = 256",
            "EVAL_EVERY     = 10_000",
            "EVAL_EPISODES  = 100   # in-training monitoring; test eval (§5) is the comparison metric",
            "TEST_EPISODES  = 100",
            "EVAL_NUM_WORKERS = 6",
            "",
            "# FQL Gate-B-frozen flags; ONLY --distill-alpha-bc varies in this notebook.",
            "def fql_flags(alpha):",
            "    return (",
            "        f'--flow-steps 10 --distill-alpha-bc {alpha} '",
            "        f'--teacher-lr 3e-4 --flow-time-embed-dim 32'",
            "    )",
            "",
            "def atag(alpha):",
            "    # 0.3->'0p3', 1.0->'1p0', 3.0->'3p0', 10.0->'10p0'",
            "    return str(alpha).replace('.', 'p')",
            "",
            "ALPHAS_SWEEP   = [0.3, 3.0, 10.0]   # new alphas, clean E-uni",
            "ALPHA_BASELINE = 1.0                # frozen value, for the seed top-up",
            "TRAIN_SEEDS    = [42, 0]            # baseline seeds (existing for α=1.0; reused for sweep)",
            "TOPUP_SEEDS    = [1, 2]             # NEW seeds: bump α=1.0 clean to n=4 (§9.9)",
            "",
            "# Two output trees:",
            "SWEEP_CKPT = Path('checkpoints/fql_succession/p2/e_uni_c1_fql_alpha_sweep')",
            "SWEEP_RES  = Path('results/fql_succession/p2/e_uni_c1_fql_alpha_sweep')",
            "EUNI_CKPT  = Path('checkpoints/fql_succession/p2/e_uni')   # existing baseline cell",
            "EUNI_RES   = Path('results/fql_succession/p2/e_uni')",
            "",
            "MIRROR_FILES = ('train_log.jsonl', 'eval_log.csv', 'trainer_state.json', 'train_config.txt')",
            "",
            "def make_job(alpha, seed, ckpt_root, res_root, name):",
            "    return {",
            "        'alpha': alpha, 'seed': seed, 'name': name,",
            "        'ckpt': ckpt_root / name,",
            "        'test_json': res_root / 'test' / f'{name}.json',",
            "        'mirror': res_root / 'training_curves' / name,",
            "    }",
            "",
            "JOBS = []",
            "# SWEEP: new alphas at baseline seeds → sweep tree, name fql_a<tag>_seed<S>",
            "for alpha in ALPHAS_SWEEP:",
            "    for seed in TRAIN_SEEDS:",
            "        JOBS.append(make_job(alpha, seed, SWEEP_CKPT, SWEEP_RES, f'fql_a{atag(alpha)}_seed{seed}'))",
            "# TOP-UP: α=1.0 at NEW seeds → extend existing e_uni cell, name fql_seed<S> (matches baseline)",
            "for seed in TOPUP_SEEDS:",
            "    JOBS.append(make_job(ALPHA_BASELINE, seed, EUNI_CKPT, EUNI_RES, f'fql_seed{seed}'))",
            "",
            "print(f'{len(JOBS)} jobs:')",
            "for j in JOBS:",
            "    print(f\"  α={j['alpha']:<5} seed={j['seed']:<3} → {j['ckpt']}\")",
        ),
    ]


def section_3() -> list[dict]:
    return [
        md(
            "## 3. Pre-flight",
            "",
            "确认:",
            "1. Dataset dir + `transitions.npz` + `metadata.json` exist",
            "2. metadata 是 **CLEAN**(σ=0.0)+ unimodal + ~86685 transitions(与 baseline 同源)",
            "3. manifest + flow exist",
            "4. 打印 FQL α=1.0 baseline(seeds 42,0)+ ReBRAC β1=1.0 clean bar(0.910)供对照 — 不强制",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            "ds = Path(DATASET)",
            "assert ds.is_dir(), f'dataset dir missing: {ds}'",
            "assert (ds / 'transitions.npz').is_file(), f'transitions.npz missing under {ds}'",
            "meta_path = ds / 'metadata.json'",
            "assert meta_path.is_file(), f'metadata.json missing under {ds}'",
            "meta = json.loads(meta_path.read_text())",
            "print('policy_mixture   =', meta.get('policy_mixture'))",
            "print('action_noise_std =', meta.get('action_noise_std'))",
            "print('num_transitions  =', meta.get('num_transitions'))",
            "print('source_dataset   =', ds.name)",
            "",
            "# MUST be clean (σ=0) — the rescue sweep is on the CLEAN binding axis.",
            "noise = float(meta.get('action_noise_std', float('nan')))",
            "assert abs(noise) < 1e-6, (",
            "    f'expected action_noise_std==0.0 (clean) but got {noise} '",
            "    '— C-1 sweeps FQL on CLEAN E-uni (its worst-case axis).'",
            ")",
            "exp_n_tx_approx = 86685",
            "n_tx = int(meta.get('num_transitions', 0))",
            "assert abs(n_tx - exp_n_tx_approx) < 5000, f'tx count {n_tx} far from expected {exp_n_tx_approx}'",
            "",
            "assert Path(FLOW).is_file(), f'flow missing: {FLOW}'",
            "assert Path(MANIFEST).is_file(), f'manifest missing: {MANIFEST}'",
            "",
            f"EUNI_TEST = Path('{EUNI_TEST}')",
            f"Q1C_DIR   = Path('{Q1C_DIR}')",
            "def _sr(p): return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "print()",
            "for s in TRAIN_SEEDS:",
            "    print(f'FQL  α=1.0 clean  seed={s}  SR =', _sr(EUNI_TEST / f'fql_seed{s}.json'))",
            "rb = [_sr(Q1C_DIR / f'rebrac_seed{s}.json') for s in TRAIN_SEEDS]",
            "rb = [v for v in rb if v is not None]",
            "print('ReBRAC β1=1.0 clean BAR (mean) =', round(mean(rb), 3) if rb else None, '  <- FQL worst-case must exceed this')",
            "",
            "print('\\nPre-flight PASS — ready to train.')",
        ),
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. Train (8 runs: 6 sweep + 2 top-up)",
            "",
            "FQL only,逐 job 跑;skip-resume on `agent_final.pt`(re-run 安全)。"
            "唯一变动是 `--distill-alpha-bc`(由 `fql_flags(alpha)` 注入)。",
            "",
            "**`--skip-final-eval`**:在线 eval 跑 100 ep 监控,test eval 100 ep 走 §5。",
        ),
        code(
            "import time",
            "from pathlib import Path",
            "",
            "t_start_train = time.time()",
            "",
            "for job in JOBS:",
            "    alpha = job['alpha']; seed = job['seed']",
            "    sd = job['ckpt']; sd_str = str(sd)",
            "    FQLF = fql_flags(alpha)",
            "    if (sd / 'agent_final.pt').exists():",
            "        print(f'[skip-train] α={alpha} seed={seed} (agent_final.pt exists at {sd})')",
            "        continue",
            "    sd.mkdir(parents=True, exist_ok=True)",
            "    print(f'\\n{\"=\" * 72}')",
            "    print(f'  train FQL α={alpha} seed={seed} → {sd}')",
            "    print(f'{\"=\" * 72}')",
            "    t0 = time.time()",
            "    !python -m scripts.train_offline \\",
            "        --algo fql \\",
            "        --offline-data '{DATASET}/transitions.npz' \\",
            "        --flow '{FLOW}' \\",
            "        --manifest '{MANIFEST}' \\",
            "        --probe-layout {PROBE_LAYOUT} \\",
            "        --history-length {HISTORY_LENGTH} \\",
            "        --target-speed {TARGET_SPEED} \\",
            "        --task-geometry {TASK_GEOMETRY} \\",
            "        --objective {OBJECTIVE} \\",
            "        --total-steps {TOTAL_STEPS} \\",
            "        --batch-size {BATCH_SIZE} \\",
            "        --eval-every {EVAL_EVERY} \\",
            "        --eval-episodes {EVAL_EPISODES} \\",
            "        {FQLF} \\",
            "        --skip-final-eval \\",
            "        --seed {seed} \\",
            "        --save-dir '{sd_str}' \\",
            "        --device cuda",
            "    print(f'[train done] α={alpha} seed={seed} in {(time.time() - t0) / 60:.1f} min')",
            "",
            "print(f'\\n[all train] total = {(time.time() - t_start_train) / 60:.1f} min')",
        ),
    ]


def section_5() -> list[dict]:
    return [
        md(
            "## 5. Test eval (100 ep / seed) + mirror small files",
            "",
            "对每个 `agent_final.pt` 跑 `evaluate_offline` 100 ep(固定 manifest re-eval),"
            "写 `<test_json>`;同时 mirror 4 个 small files。CLI 与 sprint-0 §5 一致。",
        ),
        code(
            "import time",
            "import shutil",
            "from pathlib import Path",
            "",
            "t_start_eval = time.time()",
            "",
            "for job in JOBS:",
            "    alpha = job['alpha']; seed = job['seed']",
            "    sd = job['ckpt']; sd_str = str(sd)",
            "    out_path = job['test_json']; out_path_str = str(out_path)",
            "    out_path.parent.mkdir(parents=True, exist_ok=True)",
            "",
            "    if out_path.exists():",
            "        print(f'[skip-eval] α={alpha} seed={seed}: {out_path}')",
            "        continue",
            "    if not (sd / 'agent_final.pt').exists():",
            "        print(f'[warn] missing agent_final.pt at {sd} — training not done?')",
            "        continue",
            "",
            "    print(f'\\n{\"=\" * 72}')",
            "    print(f'  eval FQL α={alpha} seed={seed} → {out_path}')",
            "    print(f'{\"=\" * 72}')",
            "    t0 = time.time()",
            "    !python -m scripts.evaluate_offline \\",
            "        --checkpoint '{sd_str}' \\",
            "        --manifest '{MANIFEST}' \\",
            "        --episodes {TEST_EPISODES} \\",
            "        --num-workers {EVAL_NUM_WORKERS} \\",
            "        --worker-device cpu \\",
            "        --device cuda \\",
            "        --output-json '{out_path_str}'",
            "    print(f'[eval done] α={alpha} seed={seed} in {(time.time() - t0) / 60:.1f} min')",
            "",
            "print(f'\\n[all eval] total = {(time.time() - t_start_eval) / 60:.1f} min')",
            "",
            "# Mirror 4 small files (always — outside the skip path)",
            "for job in JOBS:",
            "    sd = job['ckpt']; mdir = job['mirror']",
            "    mdir.mkdir(parents=True, exist_ok=True)",
            "    for fname in MIRROR_FILES:",
            "        src = sd / fname",
            "        if src.exists():",
            "            shutil.copy2(src, mdir / fname)",
            "    print(f\"mirrored small files for α={job['alpha']} seed={job['seed']} → {mdir}\")",
        ),
    ]


def section_6() -> list[dict]:
    return [
        md(
            "## 6. FQL clean success-rate vs `distill_alpha_bc`",
            "",
            "把 4 个 α(0.3 / 1.0[n=4] / 3.0 / 10.0)的 **clean** test SR 排成一列,"
            "对照两条线:**FQL α=1.0 旧值 0.855** 和 **ReBRAC β1=1.0 clean bar 0.910**。",
            "",
            "α=1.0 读 `e_uni/test`(seeds 42,0 旧 + 1,2 top-up = **n=4**);其余 α 读 sweep 树(seeds 42,0)。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            f"EUNI_TEST  = Path('{EUNI_TEST}')",
            f"Q1C_DIR    = Path('{Q1C_DIR}')",
            "SWEEP_TEST = SWEEP_RES / 'test'",
            "",
            "def sr(p): return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "",
            "def alpha_cells(alpha):",
            "    # returns (mean_or_None, n, dict{seed: sr})",
            "    if alpha == ALPHA_BASELINE:",
            "        seeds = TRAIN_SEEDS + TOPUP_SEEDS          # n=4",
            "        paths = {s: EUNI_TEST / f'fql_seed{s}.json' for s in seeds}",
            "    else:",
            "        paths = {s: SWEEP_TEST / f'fql_a{atag(alpha)}_seed{s}.json' for s in TRAIN_SEEDS}",
            "    vals = {s: sr(p) for s, p in paths.items()}",
            "    have = [v for v in vals.values() if v is not None]",
            "    return (mean(have) if have else None), len(have), vals",
            "",
            "# ReBRAC β1=1.0 clean bar",
            "rb = [sr(Q1C_DIR / f'rebrac_seed{s}.json') for s in TRAIN_SEEDS]",
            "rb = [v for v in rb if v is not None]",
            "BAR = mean(rb) if rb else None",
            "FQL_OLD = 0.855  # FQL α=1.0 clean at n=2 (reference)",
            "",
            "all_alphas = sorted(set(ALPHAS_SWEEP) | {ALPHA_BASELINE})",
            "print(f'{\"alpha\":>7} | {\"per-seed\":<34} | {\"mean\":>6} | n')",
            "print('-' * 64)",
            "summary = {}",
            "for a in all_alphas:",
            "    m, n, vals = alpha_cells(a)",
            "    summary[a] = (m, n)",
            "    cells = '  '.join(f'{s}:{(\"%.2f\" % v) if v is not None else \"--\"}' for s, v in vals.items())",
            "    tag = '  <- frozen (n=4)' if a == ALPHA_BASELINE else ''",
            "    print(f'{a:>7} | {cells:<34} | {(\"%.3f\" % m) if m is not None else \"  --  \":>6} | {n}{tag}')",
            "",
            "print('-' * 64)",
            "print(f'reference  FQL α=1.0 (old n=2)      = {FQL_OLD:.3f}')",
            "print(f'BAR        ReBRAC β1=1.0 clean      = {BAR:.3f}' if BAR is not None else 'BAR        (missing)')",
            "print('           FQL must reach ~this on clean to have a shot at worst-case > 0.910')",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Verdict (auto) — does any α rescue FQL on the clean axis?",
            "",
            "Decision gate(以最佳 α 的 clean mean vs 0.910 bar 为准):",
            "- **RESCUE-PROMISING**: best α clean ≥ **0.90** → 带该 α 进 **Phase C-2**(noisy)确认 worst-case。",
            "- **RESCUE-WEAK**: best α clean ∈ **[0.86, 0.90)** → 有改善但不够;C-2 可选。",
            "- **RESCUE-FAIL**: best α clean < **0.86**(≈ 没超过 α=1.0 自身)→ FQL 过不了 binding axis "
            "→ **回落 B+A**(诚实机制/负面结果)。",
            "",
            "**统计提醒(§9.9)**:sweep α 是 n=2,SE_δ≈4pp;固定 manifest 下对比是配对的,但小差距"
            "(<5pp)仍属 NULL 分辨率 — PROMISING 的 α 必须在 C-2 / n=4 复核后才算数。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            f"EUNI_TEST  = Path('{EUNI_TEST}')",
            f"Q1C_DIR    = Path('{Q1C_DIR}')",
            "SWEEP_TEST = SWEEP_RES / 'test'",
            "",
            "def sr(p): return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "def amean(alpha):",
            "    if alpha == ALPHA_BASELINE:",
            "        paths = [EUNI_TEST / f'fql_seed{s}.json' for s in TRAIN_SEEDS + TOPUP_SEEDS]",
            "    else:",
            "        paths = [SWEEP_TEST / f'fql_a{atag(alpha)}_seed{s}.json' for s in TRAIN_SEEDS]",
            "    vals = [sr(p) for p in paths]",
            "    vals = [v for v in vals if v is not None]",
            "    return mean(vals) if vals else None",
            "",
            "rb = [sr(Q1C_DIR / f'rebrac_seed{s}.json') for s in TRAIN_SEEDS]",
            "rb = [v for v in rb if v is not None]",
            "BAR = mean(rb) if rb else 0.910",
            "",
            "scored = {a: amean(a) for a in (ALPHAS_SWEEP + [ALPHA_BASELINE])}",
            "have = {a: v for a, v in scored.items() if v is not None}",
            "assert have, 'no FQL clean results found yet — run §4/§5 first'",
            "best_alpha = max(have, key=have.get)",
            "best_clean = have[best_alpha]",
            "a1 = scored.get(ALPHA_BASELINE)",
            "",
            "print('FQL clean mean by α:', {a: round(v, 3) for a, v in scored.items() if v is not None})",
            "print(f'best α            = {best_alpha}  (clean μ = {best_clean:.3f})')",
            "print(f'α=1.0 (n=4)       = {a1:.3f}' if a1 is not None else 'α=1.0 (n=4)       = (incomplete)')",
            "print(f'ReBRAC β1=1.0 BAR = {BAR:.3f}')",
            "",
            "if best_clean >= 0.90:",
            "    print('\\nVERDICT: RESCUE-PROMISING — α={} lifts FQL clean to {:.3f} (≥0.90).'.format(best_alpha, best_clean))",
            "    print('  → Build Phase C-2: run FQL --distill-alpha-bc {} on NOISY M-uni-noise (2 seeds),'.format(best_alpha))",
            "    print('    confirm noisy stays ≥0.91 so worst-case-over-noise > 0.910 (beats ReBRAC β1=1.0).')",
            "    print('    Then take the winning α to n=4 on both axes before any claim.')",
            "elif best_clean >= 0.86:",
            "    print('\\nVERDICT: RESCUE-WEAK — best α={} clean {:.3f} in [0.86,0.90).'.format(best_alpha, best_clean))",
            "    print('  Improved over α=1.0 but short of the 0.910 bar. C-2 optional / marginal.')",
            "    print('  Likely still ends at B+A unless C-2 noisy is unexpectedly strong.')",
            "else:",
            "    print('\\nVERDICT: RESCUE-FAIL — no α beats FQL α=1.0 clean (~0.855).')",
            "    print('  FQL cannot clear its binding (clean) axis → fall back to B+A:')",
            "    print('  the mechanism trilogy stands, and \"FQL got a fair tuned shot and still')",
            "    print('  did not win\" is a STRONGER honest-negative than the asymmetric-tuning version.')",
        ),
    ]


def section_8() -> dict:
    return md(
        "## 8. Report checklist",
        "",
        "完成后回写主诊断文档:",
        "- [ ] 写入 `docs/fql_succession_p2_mechanism_diagnostic.md` §9.10 — C-1 verdict "
        "(per-α clean table + best α + α=1.0 n=4 修正值 + RESCUE 判定)",
        "- [ ] sync `results/fql_succession/p2/e_uni_c1_fql_alpha_sweep/test/*.json`(6 个)"
        "+ 新的 `results/fql_succession/p2/e_uni/test/fql_seed{1,2}.json`(2 个)回 local",
        "- [ ] 据 verdict 决定:**PROMISING** → 建 C-2 noisy notebook;"
        "**WEAK/FAIL** → 回落 B+A,解冻 N3/N4",
        "",
        "Decision impact:",
        "- **RESCUE-PROMISING**: FQL 在公平调参后于 clean 追平 → C-2 验 worst-case;若 worst-case > "
        "0.910,\"FQL wins\" 主线**复活**(tuned-vs-tuned),N3/N4 按 superiority 框定。",
        "- **RESCUE-WEAK/FAIL**: FQL 公平调参仍不赢 → 锁定 **B+A**。这反而是更强的诚实负面结果"
        "(FQL 拿到对称调参机会仍输),且机制三连(§9.4/9.7/9.9)统计显著、可发表。",
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
