"""Build the cross-benchmark confirmation notebook (FQL Succession P2 — `single_u15_cross`).

Minimal generalization check of P2's two load-bearing claims on a HARDER benchmark
(U=1.5 / Re250, vs P2's U=1.0 / Re150), per `docs/fql_succession_p2_xbench_spec.md`:

    C1  noise is the discriminator (clean vs noisy).
    C2  fixed ReBRAC β1=1.0 ≥ FQL on worst-case-over-noise — i.e. the apparent
        "FQL wins" is a β1 mis-tuning artifact — AND the β1 4→1 noise recovery
        reproduces.

Scope (deliberately NOT the full P2 expansion):
  - noise axis only (drop modality — E-multi was NULL → modality already exonerated).
  - 3 configs {FQL frozen, ReBRAC β1=4.0, ReBRAC β1=1.0} × 2 cells {clean, noisy}
    × 2 seeds [42, 0] = 12 runs.
  - a FLOOR-EFFECT GATE (1 run) runs first: only commit the other 11 if the
    benchmark is learnable AND has headroom.
  - no C-1 FQL-knob rematch, no seed-补, no algo re-tune, no auv_nav edits.

Output: notebooks/fql_succession_p2_xbench.ipynb
Regenerate: python -m scripts._build_fql_succession_p2_xbench_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_xbench.ipynb")

# u15_cross flow + datasets (collected fresh on the U1p50_Re250 flow; see spec §3).
FLOW = "wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
MANIFEST = "benchmarks/single_u15_cross_tgt15_ep100.json"
DATASET_CLEAN = "offline_data/fql_succession/xbench_u15cross/e_uni_clean_1000"
DATASET_NOISY = "offline_data/fql_succession/xbench_u15cross/m_uni_noise_eps0p5_1000"


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **Cross-Benchmark Confirmation** (`single_u15_cross`, U=1.5/Re250)",
        "",
        "**Purpose**: minimally re-test P2's two load-bearing claims on a **harder** "
        "benchmark, to harden the mechanism finding's generalization before publication. "
        "Spec: `docs/fql_succession_p2_xbench_spec.md`.",
        "",
        "P2 closed as **B+A (mechanism finding + honest negative, NOT \"FQL wins\")** on "
        "`single_u10_cross`. This notebook checks the same conclusions reproduce at "
        "U=1.5/Re250:",
        "",
        "- **C1** — noise (not modality) is the discriminator → test clean vs noisy.",
        "- **C2** — fixed **ReBRAC β1=1.0 ≥ FQL on worst-case-over-noise** (the apparent "
        "\"FQL wins\" is a β1 mis-tuning artifact), AND the **β1 4→1 noise recovery** reproduces.",
        "",
        "## Design — 12 runs (noise axis only)",
        "",
        "| cell | noise | configs | seeds | runs |",
        "|---|---|---|---|---|",
        "| `e_uni_clean` | privileged σ=0 | FQL / ReBRAC β1=4.0 / ReBRAC β1=1.0 | 42, 0 | 6 |",
        "| `m_uni_noise` | privileged σ=0.5 | FQL / ReBRAC β1=4.0 / ReBRAC β1=1.0 | 42, 0 | 6 |",
        "",
        "**modality axis dropped** (E-multi was NULL in P2 → modality already exonerated). "
        "β1=4.0 kept — it is needed to reproduce the headline mechanism (β1=4 collapses on "
        "noise, β1=1 recovers).",
        "",
        "## Order of operations",
        "",
        "1. **§3** prerequisites — generate ep100 manifest + collect 2 datasets (idempotent).",
        "2. **§4 FLOOR GATE** — train+eval **1** run (ReBRAC β1=1.0 / clean / seed 42); only "
        "proceed if SR ∈ (0.50, 0.97) (learnable AND has headroom). **STOP otherwise.**",
        "3. **§5/§6** — the remaining 11 runs + eval (skip-resume skips the gate run).",
        "4. **§7/§8** — worst-case-over-noise table + auto verdict.",
        "",
        "## Out of scope",
        "",
        "- C-1 FQL `distill_alpha_bc` rematch (only if §8 hits NEW POSITIVE).",
        "- seed-补 (>2), algo re-tune, edits to Gate-B-frozen `auv_nav/{fql,rebrac}.py`.",
        "- GMM audit re-run (advisory per D22).",
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
            "共享参数与 P2 **完全一致**(probe s0 / h4 / target 1.5 / cross_stream / arrival_v2 / "
            "200k steps / batch 256 / eval-every 10k / eval-episodes 100)。唯一变化是 "
            "**flow + manifest + dataset** 换到 `re250_u15cross`。",
            "",
            "3 个 config:`fql`(冻结)、`rebrac_b1p4`(默认 β1=4.0)、`rebrac_b1p1`(β1=1.0)。",
        ),
        code(
            "from pathlib import Path",
            "",
            f"FLOW          = '{FLOW}'",
            f"MANIFEST      = '{MANIFEST}'",
            f"DATASET_CLEAN = '{DATASET_CLEAN}'",
            f"DATASET_NOISY = '{DATASET_NOISY}'",
            "",
            "PROBE_LAYOUT   = 's0'",
            "HISTORY_LENGTH = 4",
            "TARGET_SPEED   = 1.5",
            "TASK_GEOMETRY  = 'cross_stream'",
            "OBJECTIVE      = 'arrival_v2'",
            "",
            "# Matches P2 EXACTLY (only flow/manifest/dataset differ).",
            "TOTAL_STEPS    = 200_000",
            "BATCH_SIZE     = 256",
            "EVAL_EVERY     = 10_000",
            "EVAL_EPISODES  = 100   # in-training monitoring; test eval (§6) is the comparison metric",
            "TEST_EPISODES  = 100",
            "EVAL_NUM_WORKERS = 6",
            "",
            "# Algo flags. FQL Gate-B-frozen; ReBRAC β2 frozen at 2.0, only β1 ∈ {4.0, 1.0}.",
            "FQL_FLAGS = '--flow-steps 10 --distill-alpha-bc 1.0 --teacher-lr 3e-4 --flow-time-embed-dim 32'",
            "",
            "# config: (name, algo, algo-flags)",
            "CONFIGS = [",
            "    ('fql',         'fql',    FQL_FLAGS),",
            "    ('rebrac_b1p4', 'rebrac', '--actor-penalty-coef 4.0 --critic-penalty-coef 2.0'),",
            "    ('rebrac_b1p1', 'rebrac', '--actor-penalty-coef 1.0 --critic-penalty-coef 2.0'),",
            "]",
            "",
            "# cell: (cell_name, dataset)",
            "CELLS = [",
            "    ('e_uni_clean', DATASET_CLEAN),",
            "    ('m_uni_noise', DATASET_NOISY),",
            "]",
            "SEEDS = [42, 0]",
            "",
            "CKPT_ROOT = Path('checkpoints/fql_succession/p2_xbench')",
            "RES_ROOT  = Path('results/fql_succession/p2_xbench')",
            "MIRROR_FILES = ('train_log.jsonl', 'eval_log.csv', 'trainer_state.json', 'train_config.txt')",
            "",
            "def make_job(cell, dataset, cfg_name, algo, flags, seed):",
            "    name = f'{cfg_name}_seed{seed}'",
            "    return {",
            "        'cell': cell, 'dataset': dataset, 'cfg': cfg_name, 'algo': algo,",
            "        'flags': flags, 'seed': seed, 'name': name,",
            "        'ckpt': CKPT_ROOT / cell / name,",
            "        'test_json': RES_ROOT / cell / 'test' / f'{name}.json',",
            "        'mirror': RES_ROOT / cell / 'training_curves' / name,",
            "    }",
            "",
            "JOBS = []",
            "for cell, dataset in CELLS:",
            "    for cfg_name, algo, flags in CONFIGS:",
            "        for seed in SEEDS:",
            "            JOBS.append(make_job(cell, dataset, cfg_name, algo, flags, seed))",
            "",
            "# FLOOR GATE job = ReBRAC β1=1.0 / clean / seed 42 (run FIRST, see §4).",
            "GATE_JOB = next(j for j in JOBS if j['cell'] == 'e_uni_clean'",
            "                and j['cfg'] == 'rebrac_b1p1' and j['seed'] == 42)",
            "",
            "print(f'{len(JOBS)} jobs:')",
            "for j in JOBS:",
            "    gate = '  <-- FLOOR GATE' if j is GATE_JOB else ''",
            "    print(f\"  {j['cell']:<12} {j['cfg']:<12} seed={j['seed']:<3} → {j['ckpt']}{gate}\")",
        ),
    ]


def section_3() -> list[dict]:
    return [
        md(
            "## 3. Prerequisites — ep100 manifest + 2 datasets (idempotent)",
            "",
            "全部 skip-if-exists。manifest 用现有 key + `--output-name` 生成 100-ep 变体;"
            "两个 dataset 复用 P2 配方,仅换 flow(本机 6-worker CPU,各 ~3-4 min)。",
        ),
        code(
            "from pathlib import Path",
            "",
            "# 3.1 ep100 manifest",
            "if not Path(MANIFEST).is_file():",
            "    print('generating ep100 manifest ...')",
            "    !python -m scripts.generate_standard_benchmarks \\",
            "        --benchmarks single_u15_cross_tgt15 \\",
            "        --episodes 100 \\",
            "        --output-name single_u15_cross_tgt15_ep100",
            "else:",
            "    print(f'[skip] manifest exists: {MANIFEST}')",
            "assert Path(MANIFEST).is_file(), f'manifest still missing: {MANIFEST}'",
            "",
            "# 3.2 clean-uni dataset (privileged σ=0)",
            "if not (Path(DATASET_CLEAN) / 'transitions.npz').is_file():",
            "    print('collecting clean-uni dataset ...')",
            "    !python -m scripts.collect_offline_data --policy privileged \\",
            "        --flow '{FLOW}' \\",
            "        --probe-layout {PROBE_LAYOUT} --task-geometry {TASK_GEOMETRY} --target-speed {TARGET_SPEED} \\",
            "        --history-length {HISTORY_LENGTH} --objective {OBJECTIVE} \\",
            "        --episodes 1000 --seed 0 --num-workers 6 --action-noise-std 0.0 \\",
            "        --output-dir '{DATASET_CLEAN}'",
            "else:",
            "    print(f'[skip] clean dataset exists: {DATASET_CLEAN}')",
            "",
            "# 3.3 noisy-uni dataset (privileged σ=0.5)",
            "if not (Path(DATASET_NOISY) / 'transitions.npz').is_file():",
            "    print('collecting noisy-uni dataset ...')",
            "    !python -m scripts.collect_offline_data --policy privileged \\",
            "        --flow '{FLOW}' \\",
            "        --probe-layout {PROBE_LAYOUT} --task-geometry {TASK_GEOMETRY} --target-speed {TARGET_SPEED} \\",
            "        --history-length {HISTORY_LENGTH} --objective {OBJECTIVE} \\",
            "        --episodes 1000 --seed 1 --num-workers 6 --action-noise-std 0.5 \\",
            "        --output-dir '{DATASET_NOISY}'",
            "else:",
            "    print(f'[skip] noisy dataset exists: {DATASET_NOISY}')",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "",
            "# Pre-flight: datasets + flow + manifest exist; print collection success rates.",
            "for tag, ds, exp_noise in [('clean', DATASET_CLEAN, 0.0), ('noisy', DATASET_NOISY, 0.5)]:",
            "    d = Path(ds)",
            "    assert (d / 'transitions.npz').is_file(), f'transitions.npz missing under {d}'",
            "    meta = json.loads((d / 'metadata.json').read_text())",
            "    noise = float(meta.get('action_noise_std', float('nan')))",
            "    assert abs(noise - exp_noise) < 1e-6, f'{tag}: expected σ={exp_noise} got {noise}'",
            "    print(f'{tag:<6} σ={noise}  success_rate={meta.get(\"success_rate\")}  '",
            "          f'n_tx={meta.get(\"num_transitions\")}  policy={meta.get(\"policy_mixture\")}')",
            "",
            "assert Path(FLOW).is_file(), f'flow missing: {FLOW}'",
            "assert Path(MANIFEST).is_file(), f'manifest missing: {MANIFEST}'",
            "print('\\nPre-flight PASS.')",
            "print('NOTE (spec §4): clean privileged SR should be >>0.5; noisy >~0.4. '",
            "      'If noisy collection SR is very low, the benchmark may floor — watch §4.')",
        ),
    ]


def _train_cell_lines(job_iter_expr: str) -> list[str]:
    """Shared train loop body; `job_iter_expr` is the python expr yielding jobs."""
    return [
        "import time",
        "from pathlib import Path",
        "",
        "t_start = time.time()",
        f"for job in {job_iter_expr}:",
        "    sd = job['ckpt']; sd_str = str(sd)",
        "    ALGO = job['algo']; FLAGS = job['flags']",
        "    if (sd / 'agent_final.pt').exists():",
        "        print(f\"[skip-train] {job['cell']} {job['cfg']} seed={job['seed']} (exists)\")",
        "        continue",
        "    sd.mkdir(parents=True, exist_ok=True)",
        "    print(f\"\\n{'=' * 72}\")",
        "    print(f\"  train {job['cell']} {job['cfg']} ({ALGO}) seed={job['seed']} → {sd}\")",
        "    print(f\"{'=' * 72}\")",
        "    t0 = time.time()",
        "    DATASET = job['dataset']",
        "    seed = job['seed']",
        "    !python -m scripts.train_offline \\",
        "        --algo {ALGO} \\",
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
        "        {FLAGS} \\",
        "        --skip-final-eval \\",
        "        --seed {seed} \\",
        "        --save-dir '{sd_str}' \\",
        "        --device cuda",
        "    print(f\"[train done] {job['cfg']} seed={job['seed']} in {(time.time() - t0) / 60:.1f} min\")",
        "print(f\"\\n[train total] {(time.time() - t_start) / 60:.1f} min\")",
    ]


def _eval_one_lines(job_iter_expr: str) -> list[str]:
    """Shared eval loop body; `job_iter_expr` yields jobs."""
    return [
        "import time",
        "from pathlib import Path",
        "",
        "t_start = time.time()",
        f"for job in {job_iter_expr}:",
        "    sd = job['ckpt']; sd_str = str(sd)",
        "    out_path = job['test_json']; out_path_str = str(out_path)",
        "    out_path.parent.mkdir(parents=True, exist_ok=True)",
        "    if out_path.exists():",
        "        print(f\"[skip-eval] {job['cell']} {job['cfg']} seed={job['seed']}: {out_path}\")",
        "        continue",
        "    if not (sd / 'agent_final.pt').exists():",
        "        print(f\"[warn] missing agent_final.pt at {sd} — train not done?\")",
        "        continue",
        "    print(f\"\\n  eval {job['cell']} {job['cfg']} seed={job['seed']} → {out_path}\")",
        "    t0 = time.time()",
        "    !python -m scripts.evaluate_offline \\",
        "        --checkpoint '{sd_str}' \\",
        "        --manifest '{MANIFEST}' \\",
        "        --episodes {TEST_EPISODES} \\",
        "        --num-workers {EVAL_NUM_WORKERS} \\",
        "        --worker-device cpu \\",
        "        --device cuda \\",
        "        --output-json '{out_path_str}'",
        "    print(f\"[eval done] {job['cfg']} seed={job['seed']} in {(time.time() - t0) / 60:.1f} min\")",
        "print(f\"\\n[eval total] {(time.time() - t_start) / 60:.1f} min\")",
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. FLOOR-EFFECT GATE (run FIRST — 1 run)",
            "",
            "u15_cross 更难。先只跑 **1 个** run(ReBRAC β1=1.0 / clean / seed 42),确认 benchmark "
            "**可学且有 headroom**,再解锁 §5 的剩余 11 runs。**判据**:",
            "",
            "- **PASS** — test SR ∈ **(0.50, 0.97)** → 跑 §5/§6。",
            "- **FLOOR** — SR ≤ 0.50 → benchmark 对 offline-from-privileged 太难 → **STOP**,回报后再议。",
            "- **CEILING** — SR ≥ 0.97 → 无区分 headroom → **STOP**,回报后再议。",
        ),
        code(*_train_cell_lines("[GATE_JOB]")),
        code(*_eval_one_lines("[GATE_JOB]")),
        code(
            "import json",
            "from pathlib import Path",
            "",
            "p = GATE_JOB['test_json']",
            "assert p.exists(), f'gate eval json missing: {p} — run the two cells above first'",
            "sr = float(json.loads(p.read_text())['eval_success_rate'])",
            "print(f'FLOOR GATE — ReBRAC β1=1.0 / clean / seed 42  SR = {sr:.3f}')",
            "if sr <= 0.50:",
            "    print('\\nVERDICT: FLOOR — benchmark too hard for offline-from-privileged. STOP, report back.')",
            "elif sr >= 0.97:",
            "    print('\\nVERDICT: CEILING — no discriminating headroom. STOP, report back.')",
            "else:",
            "    print('\\nVERDICT: PASS — learnable AND has headroom. Proceed to §5/§6 (remaining 11 runs).')",
        ),
    ]


def section_5() -> list[dict]:
    return [
        md(
            "## 5. Train full matrix (12 runs; gate run is skipped via skip-resume)",
            "",
            "只在 §4 GATE = **PASS** 后运行。逐 job 跑,skip-resume on `agent_final.pt`。",
            "`--skip-final-eval`:在线 eval 100 ep 监控,test eval 走 §6。",
        ),
        code(*_train_cell_lines("JOBS")),
    ]


def section_6() -> list[dict]:
    return [
        md(
            "## 6. Test eval (100 ep / run) + mirror small files",
            "",
            "对每个 `agent_final.pt` 跑 `evaluate_offline` 100 ep(固定 manifest re-eval),写 "
            "`<test_json>`;再 mirror 4 个 small files。CLI 与 P2 §5 一致。",
        ),
        code(*_eval_one_lines("JOBS")),
        code(
            "import shutil",
            "from pathlib import Path",
            "",
            "for job in JOBS:",
            "    sd = job['ckpt']; mdir = job['mirror']",
            "    mdir.mkdir(parents=True, exist_ok=True)",
            "    for fname in MIRROR_FILES:",
            "        src = sd / fname",
            "        if src.exists():",
            "            shutil.copy2(src, mdir / fname)",
            "    print(f\"mirrored small files for {job['cell']} {job['cfg']} seed={job['seed']} → {mdir}\")",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Worst-case-over-noise table",
            "",
            "每个 config 取 **min(clean SR, noisy SR)** = worst-case-over-noise。三个 config 排成一列对照。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            "def sr(cell, cfg, seed):",
            "    p = RES_ROOT / cell / 'test' / f'{cfg}_seed{seed}.json'",
            "    return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "",
            "def cell_mean(cell, cfg):",
            "    vals = [sr(cell, cfg, s) for s in SEEDS]",
            "    vals = [v for v in vals if v is not None]",
            "    return mean(vals) if vals else None",
            "",
            "CFG_NAMES = [c[0] for c in CONFIGS]",
            "rows = {}",
            "print(f'{\"config\":<14} | {\"clean\":>6} | {\"noisy\":>6} | {\"worst-case\":>10}')",
            "print('-' * 48)",
            "for cfg in CFG_NAMES:",
            "    cl = cell_mean('e_uni_clean', cfg)",
            "    no = cell_mean('m_uni_noise', cfg)",
            "    wc = min(cl, no) if (cl is not None and no is not None) else None",
            "    rows[cfg] = {'clean': cl, 'noisy': no, 'worst': wc}",
            "    f = lambda v: f'{v:.3f}' if v is not None else '  --  '",
            "    print(f'{cfg:<14} | {f(cl):>6} | {f(no):>6} | {f(wc):>10}')",
        ),
    ]


def section_8() -> list[dict]:
    return [
        md(
            "## 8. Verdict (auto)",
            "",
            "预注册判据(spec §5):",
            "- ✅ **CONFIRM** — ReBRAC β1=1.0 worst-case **≥** FQL worst-case(FQL 不赢)**AND** "
            "β1=4.0 noisy < β1=1.0 noisy(4→1 噪声恢复复现)→ P2 在更难 benchmark 上泛化。",
            "- ⚠ **NEW POSITIVE** — FQL worst-case > ReBRAC β1=1.0 worst-case **> 5pp** → FQL 意外赢,"
            "**重开**(非协议失败)。",
            "- **NULL/TIE** — 差距 < 5pp / 方向不一 → 与 P2 相容(FQL 无 edge),verdict 性质不变。",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            "def sr(cell, cfg, seed):",
            "    p = RES_ROOT / cell / 'test' / f'{cfg}_seed{seed}.json'",
            "    return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "def cm(cell, cfg):",
            "    vals = [v for v in (sr(cell, cfg, s) for s in SEEDS) if v is not None]",
            "    return mean(vals) if vals else None",
            "def wc(cfg):",
            "    cl, no = cm('e_uni_clean', cfg), cm('m_uni_noise', cfg)",
            "    return (min(cl, no), cl, no) if (cl is not None and no is not None) else (None, cl, no)",
            "",
            "fql_wc, _, _ = wc('fql')",
            "b1_wc,  _, b1_no = wc('rebrac_b1p1')",
            "b4_wc,  _, b4_no = wc('rebrac_b1p4')",
            "assert None not in (fql_wc, b1_wc, b4_no, b1_no), 'incomplete results — run §5/§6 first'",
            "",
            "print(f'FQL          worst-case = {fql_wc:.3f}')",
            "print(f'ReBRAC β1=1.0 worst-case = {b1_wc:.3f}')",
            "print(f'ReBRAC β1=4.0 worst-case = {b4_wc:.3f}')",
            "print(f'noise recovery: β1=4.0 noisy {b4_no:.3f}  vs  β1=1.0 noisy {b1_no:.3f}')",
            "",
            "fql_wins_by = fql_wc - b1_wc",
            "recovery_reproduces = b4_no < b1_no",
            "if fql_wins_by > 0.05:",
            "    print('\\nVERDICT: NEW POSITIVE — FQL beats ReBRAC β1=1.0 on worst-case by '",
            "          f'{fql_wins_by*100:.1f}pp (>5pp). Re-open: investigate before any claim.')",
            "elif b1_wc >= fql_wc and recovery_reproduces:",
            "    print('\\nVERDICT: CONFIRM — ReBRAC β1=1.0 ≥ FQL on worst-case AND β1 4→1 noise '",
            "          'recovery reproduces. P2 generalizes to U=1.5/Re250; mechanism claim hardened.')",
            "elif b1_wc >= fql_wc and not recovery_reproduces:",
            "    print('\\nVERDICT: PARTIAL — FQL does not win (good), but the β1 4→1 noise recovery '",
            "          'did NOT reproduce. Report both; C2 partially generalizes.')",
            "else:",
            "    print('\\nVERDICT: NULL/TIE — FQL within 5pp of ReBRAC β1=1.0 on worst-case. '",
            "          'Consistent with P2 (FQL no edge); verdict nature unchanged.')",
        ),
    ]


def section_9() -> dict:
    return md(
        "## 9. Report checklist",
        "",
        "跑完后:",
        "- [ ] sync `results/fql_succession/p2_xbench/**/test/*.json`(12 个)回 local",
        "- [ ] 把 worst-case-over-noise 表 + verdict **追加**进 "
        "`docs/fql_succession_p2_results.md` 的 generalization 小节(**不新开**报告)",
        "- [ ] verdict notebook `notebooks/fql_succession_p2_verdict.ipynb` 增一个 xbench 行",
        "- [ ] 若 **NEW POSITIVE** → 才考虑建 C-1 u15 rematch;否则 B+A 闭环不变",
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
    cells.extend(section_8())
    cells.append(section_9())
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
