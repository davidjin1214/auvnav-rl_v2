"""One-shot builder for the 2026-07-08 supplementary-verification sprint notebooks.

Generates four Colab driver notebooks (approved 2026-07-08, see
docs/rebrac_broad_validation_v2_seed43_supplement_plan.md):

1. notebooks/sac_arrival_v2_sensing_crit_rescue_seed0.ipynb   (online: s1_k4 + s2_k4, seed 0)
2. notebooks/sac_arrival_v2_sensing_crit_rescue_seed7.ipynb   (online: s0/s1/s2_k4, seed 7)
3. notebooks/sac_arrival_v2_sensing_crit_rescue_seed42.ipynb  (online: s2_k4, seed 42)
4. notebooks/rebrac_broad_validation_v2_seed43_supplement.ipynb (offline: N0/N2p/N2p_asym, seed 43)

The online trio re-runs, under the exact §7.1/§7.6/§7.7 protocol, the six
critical-sensing cells whose cloud final_eval.json files are unreachable from
the local machine (arrival_v2 report §7.10 evidence gap / thesis finding E·M1).
The offline notebook adds pre-registered seed 43 to the three §5.8 units
(broad_validation_v2 N0 / N2' / N2'-asym), cloned verbatim from
rebrac_broad_validation_v2_core.ipynb + rebrac_broad_validation_v2_n2p_asym_critic.ipynb.

Run from repo root:
    python -m scripts._build_supplementary_verification_notebooks
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def md(*lines: str) -> dict:
    return {
        "id": _new_id(),
        "cell_type": "markdown",
        "metadata": {},
        "source": _join(lines),
    }


def code(*lines: str) -> dict:
    return {
        "id": _new_id(),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _join(lines),
    }


def _join(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _envelope(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared facts (single source inside this builder; keep in sync with
# docs/arrival_v2_experiment_report.md §7.10 and boundary-plan doc)
# ---------------------------------------------------------------------------

# Transcribed per-seed success rates from the 2026-06-18 cloud confirmation
# (online.tex rev.5 header + fig_ch5_online_sensing_crit.py CRIT_SEEDS +
# commit 46e4ca0). ✅-verified anchors carry verified=True.
SENSING_MATRIX = {
    ("s0", 0): {"expected": 0.400, "verified": True},
    ("s0", 7): {"expected": 0.267, "verified": False},
    ("s0", 42): {"expected": 0.100, "verified": True},
    ("s1", 0): {"expected": 0.900, "verified": False},
    ("s1", 7): {"expected": 0.800, "verified": False},
    ("s1", 42): {"expected": 0.900, "verified": True},
    ("s2", 0): {"expected": 0.800, "verified": False},
    ("s2", 7): {"expected": 0.733, "verified": False},
    ("s2", 42): {"expected": 0.733, "verified": False},
}

# Expected stacked obs dims per layout under arrival_v2 (+2 context) × k=4.
OBS_DIM = {"s0": 48, "s1": 56, "s2": 72}

RESCUE_SEED_PLAN = {
    0: ["s1", "s2"],
    7: ["s0", "s1", "s2"],
    42: ["s2"],
}

RUN_TREE = "experiments/arrival_v2_prototype/single_u15_cross_tgt15/arrival_v2/sac_vanilla"
CKPT_TREE = "checkpoints/arrival_v2_prototype/single_u15_cross_tgt15/arrival_v2/sac_vanilla"
RESCUE_SUMMARY_DIR = "experiments/arrival_v2_prototype/sensing_crit_rescue_summary"


def _matrix_table_md() -> list[str]:
    rows = [
        "| 配置 | seed 0 | seed 7 | seed 42 |",
        "|---|---|---|---|",
    ]
    for layout in ("s0", "s1", "s2"):
        cells = []
        for seed in (0, 7, 42):
            m = SENSING_MATRIX[(layout, seed)]
            mark = "✅ 已实核" if m["verified"] else "⚠ 待补跑"
            cells.append(f"{m['expected']:.3f} {mark}")
        rows.append(f"| {layout}_k4 | {cells[0]} | {cells[1]} | {cells[2]} |")
    return rows


def build_rescue_notebook(seed: int) -> dict:
    layouts = RESCUE_SEED_PLAN[seed]
    runs_desc = " + ".join(f"{p}_k4" for p in layouts)
    n_runs = len(layouts)
    cells: list[dict] = []

    cells.append(md(
        f"# sensing-crit rescue — 临界传感 6-cell 同协议补跑（seed={seed} 道：{runs_desc}）",
        "",
        "> 文档锚点：[`docs/arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) **§7.10**（取证缺口与缺失清单）"
        " · [`docs/rebrac_broad_validation_v2_seed43_supplement_plan.md`](../docs/rebrac_broad_validation_v2_seed43_supplement_plan.md) 附录 A"
        " · 论文章级验收 finding **E·M1**。",
        "> 用户批准（2026-07-08）：六份云端 final_eval 本机不可达（Mac 不在身边），走**同协议补跑重取证**，与 §5.8 补种子并为一个执行轮。",
        "",
        "本 notebook 是 3 道并行中的 **seed=" + str(seed) + " 道**"
        f"（{n_runs} 个 1M-step 训练单元，L4 约 {n_runs * 2.5:.1f} h；`[skip]`/resume 支持跨 session 续跑）。",
        "另两道：`sac_arrival_v2_sensing_crit_rescue_seed{0,7,42}.ipynb`。",
        "",
        "## 九宫格现状（转录值 = 2026-06-18 云端确认；✅ = 本机 final_eval 已实核）",
        "",
        *_matrix_table_md(),
        "",
        "## 预登记判读语义（跑前锁定）",
        "",
        "- 补跑产出的 `final_eval.json` 即该 cell 的**新 ground truth**（可追溯证据）。",
        "- 补跑值与转录值**逐位一致** → §7.10 该读数 ⚠→✅；六格全一致后 E·M1 勾销。",
        "- **任何偏差（含 1 个 episode 之差）→ 先呈报**，`.tex` 刊值与 §7.10 一字不动，由用户裁决后统一处置。",
        "- 本 notebook 只产出事实（per-cell 补跑值 + EXACT_MATCH 布尔），不做任何论文侧改动。",
        "",
        "## 协议（与 §7.1 / §7.6 / §7.7 逐项一致，唯一变量 = probe layout × seed）",
        "",
        "`vanilla SAC / arrival_v2 / k=4 / 1M steps / num_envs=6 / random_steps=update_after=5000 /`",
        "`batch=256 / hidden=256 / eval_every=25k×30ep / 终检 final_eval 30 deterministic ep /`",
        "`flow=wake_v8_U1p50_Re250 / manifest=benchmarks/single_u15_cross_tgt15.json（固定，禁止再生成）`",
    ))

    cells.append(md("## 0. GPU sanity"))
    cells.append(code("!nvidia-smi | head -10"))

    cells.append(md("## 1. Mount Drive + cwd"))
    cells.append(code(
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=True)",
        "",
        "REPO_DIR = '/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5'",
        "%cd $REPO_DIR",
    ))

    cells.append(md("## 2. Config — 本道 RUNS 矩阵（路径一律相对仓库根，回避 Drive 路径含空格的 `!python {var}` 拆断 bug）"))

    run_lines = []
    for p in layouts:
        m = SENSING_MATRIX[(p, seed)]
        run_lines.append(
            "    {"
            f"'probe': '{p}', 'expected': {m['expected']}, 'obs_dim': {OBS_DIM[p]},"
            f" 'run_root': '{RUN_TREE}/{p}_k4/seed_{seed}',"
            f" 'ckpt_root': '{CKPT_TREE}/{p}_k4/seed_{seed}'"
            "},"
        )
    cells.append(code(
        "import json",
        "import os",
        "from pathlib import Path",
        "",
        "import pandas as pd",
        "",
        "# ==== 与 §7.1/§7.6/§7.7 严格一致（除 probe layout / seed 外零新变量）====",
        "OBJECTIVE = 'arrival_v2'",
        "HISTORY_LENGTH = 4",
        "TARGET_SPEED = 1.5",
        f"SEED = {seed}",
        "",
        "RANDOM_STEPS = 5_000",
        "UPDATE_AFTER = 5_000",
        "BATCH_SIZE = 256",
        "HIDDEN_DIM = 256",
        "NUM_ENVS = 6",
        "EVAL_EVERY = 25_000",
        "EVAL_EPISODES = 30",
        "CHECKPOINT_EVERY = 100_000",
        "TOTAL_STEPS = 1_000_000",
        "DEVICE = 'cuda'",
        "",
        "# 显式拒绝所有 SAC 改进项 — vanilla（与被补的 6 cell 原协议一致）",
        "USE_ASYMMETRIC_CRITIC = False",
        "USE_LAYERNORM = False",
        "UPDATES_PER_STEP = 1",
        "DROPOUT_RATE = 0.0",
        "",
        "SINGLE_FLOW = 'wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy'",
        "BENCHMARK_KEY = 'single_u15_cross_tgt15'",
        "TASK_GEOMETRY = 'cross_stream'",
        "MANIFEST_PATH = Path(f'benchmarks/{BENCHMARK_KEY}.json')",
        "",
        "RUNS = [",
        *run_lines,
        "]",
        "",
        f"SUMMARY_DIR = Path('{RESCUE_SUMMARY_DIR}')",
        f"VERDICT_JSON = SUMMARY_DIR / 'rescue_verdict_seed{seed}.json'",
        "",
        "os.environ['PYTHONUNBUFFERED'] = '1'",
        "",
        "print(f'SEED = {SEED}   runs = ' + ', '.join(r['probe'] + '_k4' for r in RUNS))",
        "for r in RUNS:",
        "    print(f\"  {r['probe']}_k4  expected(transcribed)={r['expected']:.3f}  \"",
        "          f\"expected obs_dim={r['obs_dim']}  -> {r['run_root']}\")",
    ))

    cells.append(md("## 3. Preflight — flow / 固定 manifest / 已实核锚点 / 目标位状态"))
    cells.append(code(
        "fp = Path(SINGLE_FLOW)",
        "if not fp.exists():",
        "    raise FileNotFoundError(f'missing flow file: {fp}')",
        "print(f'[OK] flow file: {fp}  ({fp.stat().st_size / 1e6:.1f} MB)')",
        "",
        "# manifest 是固定评估集（git 内）；缺失说明 Drive 同步不完整——禁止在此再生成",
        "if not MANIFEST_PATH.exists():",
        "    raise FileNotFoundError(",
        "        f'fixed manifest missing: {MANIFEST_PATH} — sync repo to Drive first; DO NOT regenerate')",
        "print(f'[OK] fixed manifest: {MANIFEST_PATH}')",
    ))
    cells.append(code(
        "# 已实核 ✅ 锚点（§7.10）——就位即打印，用于 sanity（不参与本道训练）",
        "ANCHORS = [",
        f"    ('s0_k4/seed_0',  '{RUN_TREE}/s0_k4/seed_0',  0.400),",
        f"    ('s0_k4/seed_42', '{RUN_TREE}/s0_k4/seed_42', 0.100),",
        f"    ('s1_k4/seed_42', '{RUN_TREE}/s1_k4/seed_42', 0.900),",
        "]",
        "for label, root, ref in ANCHORS:",
        "    f = Path(root) / 'results' / 'final_eval.json'",
        "    if f.exists():",
        "        v = float(json.loads(f.read_text(encoding='utf-8'))['eval_success_rate'])",
        "        mark = '✓' if abs(v - ref) < 1e-9 else f'✗ MISMATCH vs {ref}'",
        "        print(f'[OK] anchor {label}: final={v:.3f} {mark}')",
        "    else:",
        "        print(f'[WARN] anchor {label} missing on Drive: {f}')",
        "",
        "# 目标位：六缺格的本道子集应为空（若 Mac 文件已先行同步，[skip] 会自动跳过）",
        "for r in RUNS:",
        "    f = Path(r['run_root']) / 'results' / 'final_eval.json'",
        "    print(('[PRESENT — 将被 [skip] 保护，不覆盖] ' if f.exists() else '[EMPTY — 待补跑] ') + str(f))",
    ))

    cells.append(md(
        f"## 4. Train — {runs_desc} × seed={seed}（各 1M steps，`[skip]`/resume 跨 session 安全）",
        "",
        "`[skip]` 判定：`trainer_state.json` 的 `env_step >= 1M`；中断后重跑本 cell 自动 `--resume`。",
    ))
    cells.append(code(
        "for r in RUNS:",
        "    run_root = Path(r['run_root'])",
        "    run_root_s = str(run_root)",
        "    ckpt_root_s = r['ckpt_root']",
        "    probe = r['probe']",
        "    manifest_s = str(MANIFEST_PATH)",
        "",
        "    state_path = run_root / 'trainer_state.json'",
        "    current_step = 0",
        "    if state_path.exists():",
        "        current_step = int(json.loads(state_path.read_text(encoding='utf-8')).get('env_step', 0))",
        "    print(f'\\n========== {probe}_k4 seed={SEED}: env_step={current_step:,} / {TOTAL_STEPS:,} ==========')",
        "",
        "    if current_step >= TOTAL_STEPS:",
        "        print(f'[skip] already trained to {current_step:,}')",
        "        continue",
        "    elif current_step > 0:",
        "        print(f'[resume] continuing from {current_step:,}')",
        "        !python -u -m scripts.train_sac \\",
        "            --resume {run_root_s} \\",
        "            --total-steps {TOTAL_STEPS} \\",
        "            --eval-every {EVAL_EVERY} \\",
        "            --eval-episodes {EVAL_EPISODES} \\",
        "            --checkpoint-every {CHECKPOINT_EVERY} \\",
        "            --eval-manifest {manifest_s} \\",
        "            --device {DEVICE}",
        "    else:",
        "        print('[train] fresh start')",
        "        !python -u -m scripts.train_sac \\",
        "            --flow {SINGLE_FLOW} \\",
        "            --task-geometry {TASK_GEOMETRY} \\",
        "            --target-speed {TARGET_SPEED} \\",
        "            --objective {OBJECTIVE} \\",
        "            --probe-layout {probe} \\",
        "            --history-length {HISTORY_LENGTH} \\",
        "            --total-steps {TOTAL_STEPS} \\",
        "            --random-steps {RANDOM_STEPS} \\",
        "            --update-after {UPDATE_AFTER} \\",
        "            --batch-size {BATCH_SIZE} \\",
        "            --hidden-dim {HIDDEN_DIM} \\",
        "            --num-envs {NUM_ENVS} \\",
        "            --eval-every {EVAL_EVERY} \\",
        "            --eval-episodes {EVAL_EPISODES} \\",
        "            --checkpoint-every {CHECKPOINT_EVERY} \\",
        "            --eval-manifest {manifest_s} \\",
        "            --seed {SEED} \\",
        "            --device {DEVICE} \\",
        "            --save-dir {run_root_s} \\",
        "            --checkpoint-dir {ckpt_root_s}",
    ))

    cells.append(md("## 5. Verdict — 补跑值 vs 转录值（纯事实输出，不判读、不改论文侧）"))
    cells.append(code(
        "records = []",
        "for r in RUNS:",
        "    run_root = Path(r['run_root'])",
        "    fe_path = run_root / 'results' / 'final_eval.json'",
        "    tc_path = run_root / 'results' / 'train_config.txt'",
        "    if not fe_path.exists():",
        "        print(f\"[WARN] {r['probe']}_k4: final_eval.json 缺失（训练未完成？）\")",
        "        continue",
        "    d = json.loads(fe_path.read_text(encoding='utf-8'))",
        "    succ = float(d['eval_success_rate'])",
        "    n_ep = int(float(d.get('num_eval_episodes', 0)))",
        "    counts = d.get('eval_termination_counts', {})",
        "",
        "    obs_dim_cfg = None",
        "    if tc_path.exists():",
        "        for ln in tc_path.read_text(encoding='utf-8').splitlines():",
        "            if ln.strip().startswith('obs_dim='):",
        "                obs_dim_cfg = int(ln.strip().split('=', 1)[1])",
        "    obs_ok = (obs_dim_cfg == r['obs_dim'])",
        "",
        "    # 转录值是三位小数舍入（如 8/30→0.267），final_eval 是全精度；",
        "    # 「逐位一致」按成功回合数比较，不做浮点直等。",
        "    n_succ = round(succ * n_ep) if n_ep else -1",
        "    expected_n_succ = round(r['expected'] * 30)",
        "    exact = (n_ep == 30) and (n_succ == expected_n_succ)",
        "    records.append({",
        "        'probe': r['probe'], 'seed': SEED,",
        "        'rescue_success': succ, 'rescue_n_succ': n_succ,",
        "        'transcribed': r['expected'], 'transcribed_n_succ': expected_n_succ,",
        "        'exact_match': exact, 'num_eval_episodes': n_ep,",
        "        'termination_counts': counts,",
        "        'obs_dim': obs_dim_cfg, 'obs_dim_expected': r['obs_dim'], 'obs_dim_ok': obs_ok,",
        "        'final_eval_path': str(fe_path),",
        "    })",
        "    mark = ('EXACT_MATCH ✓' if exact",
        "            else f'MISMATCH ✗ ({n_succ}/{n_ep} vs {expected_n_succ}/30) → 呈报')",
        "    print(f\"{r['probe']}_k4 seed={SEED}: rescue={succ:.3f} ({n_succ}/{n_ep})  \"",
        "          f\"transcribed={r['expected']:.3f} ({expected_n_succ}/30)\"",
        "          f\"  [{mark}]  counts={counts}  obs_dim={obs_dim_cfg} ({'ok' if obs_ok else 'UNEXPECTED'})\")",
        "",
        "SUMMARY_DIR.mkdir(parents=True, exist_ok=True)",
        "VERDICT_JSON.write_text(json.dumps({",
        "    'notebook': 'sac_arrival_v2_sensing_crit_rescue_seed" + str(seed) + "',",
        "    'protocol': 'vanilla SAC / arrival_v2 / k4 / 1M / num_envs=6 / final_eval 30ep / '",
        "                'flow wake_v8_U1p50_Re250 / manifest single_u15_cross_tgt15',",
        "    'semantics': 'rescue final_eval is the new ground truth; any deviation from the '",
        "                 'transcribed 2026-06-18 value must be escalated before touching thesis numbers',",
        "    'runs': records,",
        "}, indent=1, ensure_ascii=False), encoding='utf-8')",
        "print(f'\\n[saved] {VERDICT_JSON}')",
        "print('all EXACT_MATCH:', bool(records) and all(x['exact_match'] for x in records),",
        "      f'({len(records)}/{len(RUNS)} runs evaluated)')",
    ))

    cells.append(md(
        "## 6. 跑完后清单（回到 local）",
        "",
        "三道全部收齐后（`rescue_verdict_seed0/7/42.json` × 3）：",
        "",
        "1. 只取回轻量结果（**不要**取回 `state/replay_latest.pkl`，每个 ~GB 级）：",
        "",
        "```bash",
        "# 在本地 repo root（对本道涉及的每个 run dir）：",
        "rsync -av --exclude 'state/' --exclude 'logs/' \\",
        f"  '<drive>/{RUN_TREE}/<layout>_k4/seed_{seed}/' \\",
        f"  {RUN_TREE}/<layout>_k4/seed_{seed}/",
        f"rsync -av '<drive>/{RESCUE_SUMMARY_DIR}/' {RESCUE_SUMMARY_DIR}/",
        "```",
        "",
        "2. 回读判读入口：`paper/thesis_ch5/next_session_prompt.md`（执行轮回读节）。",
        "   六格全 EXACT_MATCH → §7.10 ⚠→✅ + E·M1 勾销；任一 MISMATCH → 呈报硬停。",
    ))

    return _envelope(cells)


# ---------------------------------------------------------------------------
# Offline seed-43 supplement notebook
# ---------------------------------------------------------------------------

def build_seed43_notebook() -> dict:
    cells: list[dict] = []

    cells.append(md(
        "# ReBRAC Broad Validation v2 — seed 43 supplement（N0 / N2' / N2'-asym，§5.8 补种子）",
        "",
        "> 文档锚点：[`docs/rebrac_broad_validation_v2_seed43_supplement_plan.md`](../docs/rebrac_broad_validation_v2_seed43_supplement_plan.md)"
        "（**用户批准 2026-07-08，最小矩阵 +seed 43 × 3 单元**）"
        " · [`docs/rebrac_broad_validation_v2_plan.md`](../docs/rebrac_broad_validation_v2_plan.md) §6.2（预登记 3 seed）"
        " · [`docs/rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md)（2-seed 基线 + §6.3 backlog）。",
        "",
        "命令逐字克隆自 `rebrac_broad_validation_v2_core.ipynb`（N0/N2p）与"
        " `rebrac_broad_validation_v2_n2p_asym_critic.ipynb`（asym），唯一新变量 = `--seed 43`。",
        "",
        "## 既有 2-seed 基线（report §2 / §4.5）",
        "",
        "| 单元 | seed 42 | seed 0 | verdict |",
        "|---|---|---|---|",
        "| N0（crosscomp / sub-critical） | 0.867 | 0.833 | HOLDS（≥0.70） |",
        "| N2'（privileged / critical） | 0.000 | 0.000 | STRONG_NEGATIVE |",
        "| N2'-asym（+asym critic） | 0.000 | 0.000 | ACTOR_FUNDAMENTAL_CONFIRMED |",
        "",
        "## 预登记判读门槛（plan §2，跑前锁定、不允许 post-hoc 调整）",
        "",
        "- **N0**：3-seed 均值 ≥ 0.70 且 < 0.902（vs anchor 同向退化）→ 一致；否则呈报。",
        "- **N2'**：seed 43 = **0/30** → 一致；任何 >0 成功 → 呈报（「零成功完全一致」表述失效）。",
        "- **N2'-asym**：seed 43 = **0/30** 且终止以越界为主（OOB 占多数）→ 一致；否则呈报。",
        "- 三者全一致 → 仅做 plan §4 零论证定点微修；任一不一致 → 微修冻结，先呈报。",
        "",
        "预算：3 × 64-epoch ReBRAC + 3 × 终检 ≈ **≤2h L4 单 session**。",
    ))

    cells.append(md("## 0. 环境 sanity"))
    cells.append(code("!nvidia-smi"))

    cells.append(md("## 1. Mount Drive + cwd"))
    cells.append(code(
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=True)",
        "%cd /content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5",
    ))

    cells.append(md(
        "## 2. Run matrix — 3 runs × seed 43",
        "",
        "路径一律相对仓库根（Drive 路径含空格，`!python {var}` 插值不加引号会在空格处拆断）。",
    ))
    cells.append(code(
        "import json",
        "import os",
        "from pathlib import Path",
        "",
        "SEED = 43",
        "",
        "# ---- 与首轮完全一致的两个 cell 定义（core notebook cell 1 逐字）----",
        "N0_DATASET  = 'offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/transitions.npz'",
        "N0_MANIFEST = 'benchmarks/single_u10_cross_tgt15.json'",
        "N2P_DATASET  = 'offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000/transitions.npz'",
        "N2P_MANIFEST = 'benchmarks/single_u15_cross_tgt15.json'",
        "",
        "RUNS = [",
        "    {'cell_id': 'N0',       'asym': False, 'dataset': N0_DATASET,  'manifest': N0_MANIFEST,",
        "     'ckpt_dir': f'checkpoints/offline/rebrac/broad_validation_v2/N0/seed_{SEED}',",
        "     'result_dir': f'results/offline/rebrac/broad_validation_v2/N0/seed_{SEED}',",
        "     'baseline_dir': 'results/offline/rebrac/broad_validation_v2/N0',",
        "     'regime_note': 'sub-critical (U=1.0/Re=150)'},",
        "    {'cell_id': 'N2p',      'asym': False, 'dataset': N2P_DATASET, 'manifest': N2P_MANIFEST,",
        "     'ckpt_dir': f'checkpoints/offline/rebrac/broad_validation_v2/N2p/seed_{SEED}',",
        "     'result_dir': f'results/offline/rebrac/broad_validation_v2/N2p/seed_{SEED}',",
        "     'baseline_dir': 'results/offline/rebrac/broad_validation_v2/N2p',",
        "     'regime_note': 'critical (U=1.5/Re=250)'},",
        "    {'cell_id': 'N2p_asym', 'asym': True,  'dataset': N2P_DATASET, 'manifest': N2P_MANIFEST,",
        "     'ckpt_dir': f'checkpoints/offline/rebrac/broad_validation_v2_n2p_asym/seed_{SEED}',",
        "     'result_dir': f'results/offline/rebrac/broad_validation_v2_n2p_asym/seed_{SEED}',",
        "     'baseline_dir': 'results/offline/rebrac/broad_validation_v2_n2p_asym',",
        "     'regime_note': 'critical (U=1.5/Re=250), asym critic'},",
        "]",
        "",
        "SUMMARIES_DIR = Path('results/offline/rebrac/broad_validation_v2/summaries')",
        "VERDICT_JSON = SUMMARIES_DIR / 'seed43_supplement_verdict.json'",
        "",
        "# eval 参数与首轮逐字一致（--episodes 100 受 manifest 30-ep 约束实跑 30）",
        "EVAL_EPISODES    = 100",
        "EVAL_SEED        = 123",
        "EVAL_NUM_WORKERS = 4",
        "",
        "os.environ['PYTHONUNBUFFERED'] = '1'",
        "",
        "for i, r in enumerate(RUNS, 1):",
        "    print(f\"{i} {r['cell_id']:<9} asym={r['asym']!s:<5} ({r['regime_note']})\")",
        "    print(f\"   ckpt:   {r['ckpt_dir']}\")",
        "    print(f\"   result: {r['result_dir']}\")",
    ))

    cells.append(md("## 3. Preflight — dataset / manifest / 2-seed 基线就位"))
    cells.append(code(
        "for r in RUNS:",
        "    for p in (r['dataset'], r['manifest']):",
        "        if not Path(p).exists():",
        "            raise FileNotFoundError(f\"missing: {p} — sync repo/offline_data to Drive first\")",
        "print('[OK] datasets + manifests in place')",
        "",
        "for r in RUNS:",
        "    for s in (42, 0):",
        "        f = Path(r['baseline_dir']) / f'seed_{s}' / 'test_result.json'",
        "        if f.exists():",
        "            v = float(json.loads(f.read_text(encoding='utf-8'))['eval_success_rate'])",
        "            print(f\"[OK] baseline {r['cell_id']} seed_{s}: success={v:.3f}\")",
        "        else:",
        "            print(f\"[WARN] baseline missing: {f}（§5 汇总将回退 report 转载值）\")",
    ))

    cells.append(md(
        "## 4. Train — 3 runs × 64 epochs（`[skip]` 用 `agent_final.pt`）",
        "",
        "vanilla 两单元与 asym 单元共用一条命令，`{EXTRA}` 只在 asym 追加"
        " `--use-asymmetric-critic --privileged-actor-update-mode zeros`（与首轮 asym notebook 逐字一致）。",
    ))
    cells.append(code(
        "import time",
        "",
        "for r in RUNS:",
        "    cdir = Path(r['ckpt_dir'])",
        "    if (cdir / 'agent_final.pt').exists():",
        "        print(f\"[skip] {r['cell_id']} seed {SEED} done: {cdir}\")",
        "        continue",
        "    cdir.mkdir(parents=True, exist_ok=True)",
        "",
        "    dataset = r['dataset']",
        "    manifest = r['manifest']",
        "    save_dir = str(cdir)",
        "    EXTRA = ('--use-asymmetric-critic --privileged-actor-update-mode zeros'",
        "             if r['asym'] else '')",
        "",
        "    print(f\"\\n========== train {r['cell_id']} seed={SEED} ({r['regime_note']}) ==========\")",
        "    t0 = time.time()",
        "    !python -u -m scripts.train_offline \\",
        "        --algo rebrac \\",
        "        --offline-data {dataset} \\",
        "        --manifest {manifest} \\",
        "        --probe-layout s0 \\",
        "        --history-length 4 \\",
        "        --task-geometry cross_stream \\",
        "        --target-speed 1.5 \\",
        "        --objective arrival_v2 \\",
        "        --sampling-mode shuffle_no_replacement \\",
        "        --num-epochs 64 \\",
        "        --batch-size 256 \\",
        "        --hidden-dim 256 \\",
        "        --num-hidden-layers 3 \\",
        "        --actor-lr 3e-4 \\",
        "        --critic-lr 3e-4 \\",
        "        --gamma 0.99 \\",
        "        --tau 0.005 \\",
        "        --actor-penalty-coef 4.0 \\",
        "        --critic-penalty-coef 2.0 \\",
        "        --policy-noise 0.2 \\",
        "        --noise-clip 0.5 \\",
        "        --policy-freq 2 \\",
        "        --grad-clip-norm 10.0 \\",
        "        --normalizer-eps 1e-3 \\",
        "        --critic-layernorm \\",
        "        --no-actor-layernorm \\",
        "        --eval-every 0 \\",
        "        --skip-final-eval \\",
        "        --log-every 1000 \\",
        "        --seed {SEED} \\",
        "        --device cuda \\",
        "        --save-dir {save_dir} \\",
        "        {EXTRA}",
        "    print(f\"[done] {r['cell_id']} seed {SEED} ({(time.time()-t0)/60:.1f} min)\")",
    ))

    cells.append(md("## 5. Eval — 3 runs（命令与首轮逐字一致；`[skip]` 用 `test_result.json`）"))
    cells.append(code(
        "for r in RUNS:",
        "    cdir = Path(r['ckpt_dir'])",
        "    rdir = Path(r['result_dir'])",
        "    test_json = rdir / 'test_result.json'",
        "    if test_json.exists():",
        "        print(f'[skip] eval done: {test_json}')",
        "        continue",
        "    if not (cdir / 'agent_final.pt').exists():",
        "        print(f'[warn] missing agent_final.pt: {cdir}（训练未完成？）')",
        "        continue",
        "    rdir.mkdir(parents=True, exist_ok=True)",
        "",
        "    ckpt = str(cdir)",
        "    manifest = r['manifest']",
        "    out_json = str(test_json)",
        "    print(f\"\\n========== eval {r['cell_id']} seed={SEED} → {test_json} ==========\")",
        "    !python -u -m scripts.evaluate_offline \\",
        "        --checkpoint {ckpt} \\",
        "        --agent-file agent_final.pt \\",
        "        --manifest {manifest} \\",
        "        --episodes {EVAL_EPISODES} \\",
        "        --seed {EVAL_SEED} \\",
        "        --device cuda \\",
        "        --num-workers {EVAL_NUM_WORKERS} \\",
        "        --worker-device cpu \\",
        "        --output-json {out_json}",
    ))

    cells.append(md("## 6. 汇总 + 预登记判读（plan §2 三门槛；输出 verdict JSON）"))
    cells.append(code(
        "import statistics",
        "",
        "ANCHOR_EFF_V2_MEAN = 0.902  # main-line efficiency_v2 5-seed anchor",
        "",
        "def read_success(path):",
        "    d = json.loads(Path(path).read_text(encoding='utf-8'))",
        "    n = int(float(d.get('num_eval_episodes', 0)))",
        "    return float(d['eval_success_rate']), n, d.get('eval_termination_counts', {})",
        "",
        "cells_out = {}",
        "all_consistent = True",
        "for r in RUNS:",
        "    per_seed = {}",
        "    for s in (42, 0, SEED):",
        "        f = Path(r['baseline_dir']) / f'seed_{s}' / 'test_result.json'",
        "        if f.exists():",
        "            succ, n, counts = read_success(f)",
        "            per_seed[str(s)] = {'success': succ, 'n': n, 'termination_counts': counts}",
        "        else:",
        "            per_seed[str(s)] = None",
        "    vals = [v['success'] for v in per_seed.values() if v is not None]",
        "    mean3 = statistics.mean(vals) if vals else float('nan')",
        "    std3 = statistics.stdev(vals) if len(vals) >= 2 else float('nan')",
        "",
        "    new = per_seed.get(str(SEED))",
        "    if new is None:",
        "        verdict = 'INCOMPLETE (seed 43 missing)'",
        "        consistent = False",
        "    elif r['cell_id'] == 'N0':",
        "        consistent = (len(vals) == 3) and (0.70 <= mean3 < ANCHOR_EFF_V2_MEAN)",
        "        verdict = 'CONSISTENT (HOLDS retained)' if consistent else 'ESCALATE'",
        "    else:",
        "        zero = (new['success'] == 0.0)",
        "        if r['cell_id'] == 'N2p_asym':",
        "            c = new['termination_counts']",
        "            total = sum(c.values()) if c else 0",
        "            oob_major = total > 0 and c.get('out_of_bounds', 0) > total / 2",
        "            consistent = zero and oob_major",
        "        else:",
        "            consistent = zero",
        "        verdict = 'CONSISTENT (zero-success intact)' if consistent else 'ESCALATE'",
        "    all_consistent = all_consistent and consistent",
        "",
        "    cells_out[r['cell_id']] = {'per_seed': per_seed, 'mean': mean3, 'std': std3,",
        "                               'n_seeds': len(vals), 'verdict': verdict}",
        "    print(f\"{r['cell_id']:<9} per-seed \" +",
        "          ' '.join(f'{s}={v[\"success\"]:.3f}' if v else f'{s}=NA' for s, v in per_seed.items()) +",
        "          f'  mean={mean3:.3f} std={std3:.3f}  → {verdict}')",
        "",
        "print('\\nALL CONSISTENT:', all_consistent,",
        "      '→ ' + ('可按 plan §4 零论证微修' if all_consistent else '微修冻结，先呈报'))",
        "",
        "SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)",
        "VERDICT_JSON.write_text(json.dumps({",
        "    'notebook': 'rebrac_broad_validation_v2_seed43_supplement',",
        "    'seed_added': SEED,",
        "    'preregistered_gates': 'plan §2: N0 mean3 in [0.70, 0.902); N2p seed43==0/30; '",
        "                           'N2p_asym seed43==0/30 & OOB-majority',",
        "    'cells': cells_out,",
        "    'all_consistent': all_consistent,",
        "}, indent=1, ensure_ascii=False), encoding='utf-8')",
        "print(f'[saved] {VERDICT_JSON}')",
    ))

    cells.append(md(
        "## 7. 跑完后清单（回到 local）",
        "",
        "```bash",
        "# 在本地 repo root：只取回 results/（checkpoints 留 Drive）",
        "rsync -av '<drive>/results/offline/rebrac/broad_validation_v2/' \\",
        "  results/offline/rebrac/broad_validation_v2/",
        "rsync -av '<drive>/results/offline/rebrac/broad_validation_v2_n2p_asym/' \\",
        "  results/offline/rebrac/broad_validation_v2_n2p_asym/",
        "```",
        "",
        "回读判读入口：`paper/thesis_ch5/next_session_prompt.md`。",
        "ALL CONSISTENT → plan §4 零论证微修（ground truth 先行，再 `boundary.tex` 定点 Edit + latexmk）；",
        "任一 ESCALATE → 呈报硬停，§5.8 一字不动。",
    ))

    return _envelope(cells)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "notebooks"
    outputs = {
        "sac_arrival_v2_sensing_crit_rescue_seed0.ipynb": build_rescue_notebook(0),
        "sac_arrival_v2_sensing_crit_rescue_seed7.ipynb": build_rescue_notebook(7),
        "sac_arrival_v2_sensing_crit_rescue_seed42.ipynb": build_rescue_notebook(42),
        "rebrac_broad_validation_v2_seed43_supplement.ipynb": build_seed43_notebook(),
    }
    for name, nb in outputs.items():
        path = out_dir / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
