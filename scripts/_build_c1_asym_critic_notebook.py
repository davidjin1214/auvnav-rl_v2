"""One-shot builder for notebooks/rebrac_c1_asym_critic_ablation.ipynb.

Reuses the arrival_v2_simple dataset (privileged_obs_present=True) from the
preceding reward ablation, and trains ReBRAC anchor (β1=4.0, β2=2.0) × 2 seeds
with --use-asymmetric-critic. Compares against the symmetric-critic anchor
(0.215 ± 0.015) and the original efficiency_v2 + sym anchor (0.225 ± 0.005).

Run from repo root:
    python -m scripts._build_c1_asym_critic_notebook
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


CELLS: list[dict] = []

# ---------------------------------------------------------------------------
# §0 Hypothesis & gate
# ---------------------------------------------------------------------------
CELLS.append(md(
    "# ReBRAC C1 — Asymmetric Critic Ablation",
    "",
    "> 文档锚点：[CLAUDE.md §3 Asymmetric Critic](../CLAUDE.md) · "
    "[docs/online_rl_thesis_plan.md](../docs/online_rl_thesis_plan.md) · "
    "[notebooks/rebrac_c1_reward_ablation_completed.ipynb](rebrac_c1_reward_ablation_completed.ipynb)",
    "",
    "## 前传 — 已知事实",
    "",
    "| run | dataset | reward | critic | seeds | test success | mean_return |",
    "|---|---|---|---|---:|---:|---:|",
    "| broad C1 anchor | crosscomp_s0_h4_eff_v2_u10upstream | efficiency_v2 | sym (10-D) | 5 | **0.225 ± 0.005** | −371 ± 0.41 |",
    "| reward ablation | crosscomp_s0_h4_arr_v2_s_u10upstream | arrival_v2_simple | sym (10-D) | 2 | **0.215 ± 0.015** | −98 ± 4 |",
    "",
    "Reward 干预把 dataset mean_R 从 −108 翻到 +205，"
    "**actor 摆脱 deterministic-collapse**（termination 从 ~all timeout → 51% timeout / 26% OOB / 21% goal），"
    "但 test success 几乎不动（−1pp）。Verdict：**β 主导，C1 是 s0 sensor info 不够**。",
    "",
    "## 假设 — Asymmetric critic 能否打开 s0 actor 的天花板？",
    "",
    "CLAUDE.md 主方法论：**actor 受 deployment-realistic s0 约束（10-D obs），"
    "critic 拿 hull-integral 流场（privileged_obs, dim=2）做 TD target**。",
    "deployment 时 actor 仍 s0，但训练时 critic 给出更好的 supervisory signal。",
    "",
    "数据通路（已就绪）：",
    "- arrival_v2_simple dataset: sanity card `privileged_obs_present: True` ✓",
    "- `train_offline.py --use-asymmetric-critic` flag",
    "- `--privileged-actor-update-mode zeros` 默认值：actor improvement 时 zero-pad "
    "privileged channels，mimic deployment（与 online line 一致）",
    "",
    "## Verdict 逻辑（事先 commit）",
    "",
    "| arrival_v2_simple + asym test success | verdict | 含义 |",
    "|---:|---|---|",
    "| **≥ 0.6** | **asym 主导** | privileged critic 解锁 s0 actor — paper headline；CLAUDE.md §3 在 offline 同样有效 |",
    "| **0.4–0.6** | **asym 部分有效** | privileged signal 帮助但 s0 actor 仍受限；考虑加 seed 或调 actor_update_mode=batch |",
    "| **≤ 0.4** | **β floor** | s0 actor 物理上不能在 upstream u10 跑通 — C1 重定义为 sensor floor spoke，paper 写「需要 s1+ 才能 deploy」 |",
    "",
    "## 与前两次实验的隔离",
    "",
    "| 维度 | broad C1 | reward ablation | **本 notebook** |",
    "|---|---|---|---|",
    "| dataset | eff_v2 | arr_v2_s | **arr_v2_s（复用）** |",
    "| reward | efficiency_v2 | arrival_v2_simple | **arrival_v2_simple** |",
    "| critic | sym (10-D) | sym (10-D) | **asym (10-D + 2-D priv)** |",
    "| β1, β2 | 4.0, 2.0 | 4.0, 2.0 | **4.0, 2.0**（anchor 不动） |",
    "| seeds | 42–46 (5) | 42, 44 (2) | **42, 44 (2)** |",
    "",
    "唯一变量：`--use-asymmetric-critic`。其余完全一致以隔离信号。",
))

# ---------------------------------------------------------------------------
# §1 Env sanity check
# ---------------------------------------------------------------------------
CELLS.append(md("## 1. 环境 sanity check"))
CELLS.append(code(
    "!lscpu | head -10",
    "print()",
    "!nvidia-smi",
))
CELLS.append(code(
    "import torch",
    "print(f'PyTorch: {torch.__version__}')",
    "print(f'CUDA available: {torch.cuda.is_available()}')",
))
CELLS.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive', force_remount=True)",
    "%cd /content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5",
))

# ---------------------------------------------------------------------------
# §2 Common config (reuses arrival_v2_simple dataset + manifests)
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 2. 通用配置",
    "",
    "**复用** reward ablation 的 dataset / manifest，仅切换 critic 类型。"
    "`CHECKPOINT_ROOT` / `RESULTS_ROOT` 单独命名以避免与 sym critic 结果碰撞。",
))
CELLS.append(code(
    "import os",
    "from pathlib import Path",
    "",
    "# ---- 与 reward ablation 完全一致的 dataset / 任务 / 评估配置 ----",
    "OBJECTIVE = 'arrival_v2_simple'",
    "PROBE_LAYOUT = 's0'",
    "TASK_GEOMETRY = 'upstream'",
    "TARGET_SPEED = 1.5",
    "HISTORY_LENGTH = 4",
    "FLOW_PATH = 'wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy'",
    "BENCHMARK_KEY = 'single_u10_upstream_tgt15'",
    "DATASET_EPISODES = 1000",
    "VAL_EPISODES = 40",
    "TEST_EPISODES = 100",
    "",
    "# ---- 训练 anchor（保持与 sym critic anchor 完全一致）----",
    "ACTOR_PENALTY_COEF = 4.0",
    "CRITIC_PENALTY_COEF = 2.0",
    "TRAIN_SEEDS = [42, 44]",
    "TRAIN_EPOCHS = 64",
    "BATCH_SIZE = 256",
    "HIDDEN_DIM = 256",
    "NUM_HIDDEN_LAYERS = 3",
    "GAMMA = 0.99",
    "TAU = 0.005",
    "ACTOR_LR = 3e-4",
    "CRITIC_LR = 3e-4",
    "POLICY_NOISE = 0.2",
    "NOISE_CLIP = 0.5",
    "POLICY_FREQ = 2",
    "GRAD_CLIP_NORM = 10.0",
    "NORMALIZER_EPS = 1e-3",
    "",
    "# ---- Asymmetric critic 配置（CLAUDE.md §3 默认）----",
    "PRIVILEGED_ACTOR_UPDATE_MODE = 'zeros'  # actor improvement zero-pads priv channels (deployment-realistic)",
    "",
    "# ---- 路径 ----",
    "DATASET_NAME = (",
    "    f'crosscomp_{PROBE_LAYOUT}_h{HISTORY_LENGTH}_{OBJECTIVE}'",
    "    f'_re150_u10upstream_fixdone_ep{DATASET_EPISODES}'",
    ")",
    "DATASET_DIR = Path('offline_data') / DATASET_NAME",
    "TRANSITIONS_NPZ = str(DATASET_DIR / 'transitions.npz')",
    "DATASET_DIR_STR = str(DATASET_DIR)",
    "",
    "# 单独的 checkpoint / result 树（asym critic 不与 sym 混）",
    "CHECKPOINT_ROOT = Path('checkpoints/offline/rebrac/c1_asym_critic_ablation')",
    "RESULTS_ROOT = Path('results/offline/rebrac/c1_asym_critic_ablation')",
    "",
    "# 复用 reward ablation 的 manifests（geometry / probe / target_speed 完全一致）",
    "VAL_MANIFEST_PATH  = Path(f'benchmarks/c1_reward_ablation/val_{VAL_EPISODES}/{BENCHMARK_KEY}.json')",
    "TEST_MANIFEST_PATH = Path(f'benchmarks/c1_reward_ablation/test_{TEST_EPISODES}/{BENCHMARK_KEY}.json')",
    "VAL_MANIFEST_STR  = str(VAL_MANIFEST_PATH)",
    "TEST_MANIFEST_STR = str(TEST_MANIFEST_PATH)",
    "",
    "print(f'DATASET_DIR     = {DATASET_DIR}')",
    "print(f'VAL_MANIFEST    = {VAL_MANIFEST_PATH}  (exists={VAL_MANIFEST_PATH.exists()})')",
    "print(f'TEST_MANIFEST   = {TEST_MANIFEST_PATH}  (exists={TEST_MANIFEST_PATH.exists()})')",
    "print(f'CHECKPOINT_ROOT = {CHECKPOINT_ROOT}')",
    "print(f'RESULTS_ROOT    = {RESULTS_ROOT}')",
))

# ---------------------------------------------------------------------------
# §3 Verify dataset has privileged_obs columns
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 3. Pre-flight — 验证 dataset 含 `privileged_obs` / `next_privileged_obs`",
    "",
    "Asymmetric critic 在 TD target 时取 `privileged_obs` 和 `next_privileged_obs`；"
    "这两列必须在 npz 里。reward ablation 的 sanity card 已说 `privileged_obs_present: True`，"
    "这里再做一次硬检查 + dim assert (=2)。",
))
CELLS.append(code(
    "import numpy as np",
    "",
    "assert (DATASET_DIR / 'transitions.npz').exists(), f'missing dataset: {DATASET_DIR}'",
    "data = np.load(DATASET_DIR / 'transitions.npz')",
    "print('npz keys:', list(data.keys()))",
    "",
    "for key in ('privileged_obs', 'next_privileged_obs'):",
    "    assert key in data.files, f'missing column: {key} (asym critic requires it)'",
    "    arr = data[key]",
    "    assert arr.ndim == 2, f'{key} ndim={arr.ndim} (expect 2)'",
    "    assert arr.shape[1] == 2, f'{key} dim={arr.shape[1]} (expect 2: body-frame [u_eq, v_eq])'",
    "    print(f'  {key:22s} shape={arr.shape}  '",
    "          f'mean=[{arr[:, 0].mean():+.3f}, {arr[:, 1].mean():+.3f}]  '",
    "          f'std=[{arr[:, 0].std():.3f}, {arr[:, 1].std():.3f}]')",
    "",
    "print()",
    "print('[OK] dataset is asym-critic ready')",
))

# ---------------------------------------------------------------------------
# §4 Train ReBRAC + asym critic × 2 seeds
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 4. 训练 — ReBRAC anchor (β1=4.0, β2=2.0) + asym critic × 2 seeds",
    "",
    "唯一新增 flag：",
    "- `--use-asymmetric-critic` — critic 用 (s, a, priv_obs) 计算 Q",
    "- `--privileged-actor-update-mode zeros` — actor 改进时 zero-pad priv 通道（mimic deployment）",
    "",
    "其余超参与 reward ablation 完全一致。预期 ~30 min/seed × 2 = ~1h L4。",
))
CELLS.append(code(
    "PAIR_TAG = f'actorb_{ACTOR_PENALTY_COEF:.1f}__criticb_{CRITIC_PENALTY_COEF:.1f}'.replace('.', 'p')",
    "",
    "def seed_dir(seed: int) -> Path:",
    "    return CHECKPOINT_ROOT / DATASET_NAME / PAIR_TAG / f'seed_{seed}'",
    "",
    "for seed in TRAIN_SEEDS:",
    "    save_dir = seed_dir(seed)",
    "    save_dir_str = str(save_dir)",
    "    if (save_dir / 'agent_final.pt').exists():",
    "        print(f'[skip] seed {seed} done: {save_dir}')",
    "        continue",
    "    print(f'\\n========== train seed {seed} (asym critic) → {save_dir} ==========')",
    "    !python -m scripts.train_offline \\",
    "        --algo rebrac \\",
    "        --offline-data {TRANSITIONS_NPZ} \\",
    "        --flow {FLOW_PATH} \\",
    "        --manifest {VAL_MANIFEST_STR} \\",
    "        --probe-layout {PROBE_LAYOUT} \\",
    "        --history-length {HISTORY_LENGTH} \\",
    "        --task-geometry {TASK_GEOMETRY} \\",
    "        --target-speed {TARGET_SPEED} \\",
    "        --objective {OBJECTIVE} \\",
    "        --sampling-mode shuffle_no_replacement \\",
    "        --num-epochs {TRAIN_EPOCHS} \\",
    "        --batch-size {BATCH_SIZE} \\",
    "        --hidden-dim {HIDDEN_DIM} \\",
    "        --num-hidden-layers {NUM_HIDDEN_LAYERS} \\",
    "        --gamma {GAMMA} \\",
    "        --tau {TAU} \\",
    "        --actor-lr {ACTOR_LR} \\",
    "        --critic-lr {CRITIC_LR} \\",
    "        --actor-penalty-coef {ACTOR_PENALTY_COEF} \\",
    "        --critic-penalty-coef {CRITIC_PENALTY_COEF} \\",
    "        --policy-noise {POLICY_NOISE} \\",
    "        --noise-clip {NOISE_CLIP} \\",
    "        --policy-freq {POLICY_FREQ} \\",
    "        --grad-clip-norm {GRAD_CLIP_NORM} \\",
    "        --normalizer-eps {NORMALIZER_EPS} \\",
    "        --use-asymmetric-critic \\",
    "        --privileged-actor-update-mode {PRIVILEGED_ACTOR_UPDATE_MODE} \\",
    "        --eval-every 0 \\",
    "        --skip-final-eval \\",
    "        --seed {seed} \\",
    "        --device cuda \\",
    "        --save-dir {save_dir_str}",
))

# ---------------------------------------------------------------------------
# §5 Test eval
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 5. Test eval — 100-episode test manifest",
    "",
    "evaluate_offline 自动从 `trainer_state.json` 读出 `use_asymmetric_critic=True`，"
    "重建 `AsymmetricQNetwork`。**actor 在 eval 时仍只看 s0**（不传 priv_obs），"
    "与 deployment 一致。",
))
CELLS.append(code(
    "import json",
    "",
    "TEST_DIR = RESULTS_ROOT / DATASET_NAME / PAIR_TAG / 'test'",
    "TEST_DIR.mkdir(parents=True, exist_ok=True)",
    "EVAL_NUM_WORKERS = 6",
    "",
    "for seed in TRAIN_SEEDS:",
    "    out_path = TEST_DIR / f'seed_{seed}.json'",
    "    out_path_str = str(out_path)",
    "    if out_path.exists():",
    "        print(f'[skip] eval done: {out_path}')",
    "        continue",
    "    save_dir = seed_dir(seed)",
    "    save_dir_str = str(save_dir)",
    "    if not (save_dir / 'agent_final.pt').exists():",
    "        print(f'[warn] missing checkpoint: {save_dir}')",
    "        continue",
    "    print(f'\\n========== eval seed {seed} (asym) → {out_path} ==========')",
    "    !python -m scripts.evaluate_offline \\",
    "        --checkpoint {save_dir_str} \\",
    "        --manifest {TEST_MANIFEST_STR} \\",
    "        --episodes {TEST_EPISODES} \\",
    "        --num-workers {EVAL_NUM_WORKERS} \\",
    "        --worker-device cpu \\",
    "        --device cuda \\",
    "        --output-json {out_path_str}",
))

# ---------------------------------------------------------------------------
# §6 Compare table — three-way
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 6. 三向对比：eff_v2 + sym  /  arr_v2_s + sym  /  arr_v2_s + asym",
    "",
    "前两行是已知锚点（hard-coded），第三行是本 notebook 输出。",
    "也读取 termination 分布（goal / timeout / out_of_bounds），观察 collapse / explore / succeed 状态。",
))
CELLS.append(code(
    "import json",
    "import pandas as pd",
    "",
    "asym_rows = []",
    "for seed in TRAIN_SEEDS:",
    "    p = TEST_DIR / f'seed_{seed}.json'",
    "    if not p.exists():",
    "        print(f'[warn] missing: {p}')",
    "        continue",
    "    d = json.loads(p.read_text(encoding='utf-8'))",
    "    term = d.get('eval_termination_counts', {})",
    "    n = max(int(d.get('eval_episodes', 100)), 1)",
    "    asym_rows.append({",
    "        'seed': seed,",
    "        'success_rate': d['eval_success_rate'],",
    "        'mean_return':  d['eval_return'],",
    "        'safety_cost':  d['eval_safety_cost'],",
    "        'goal_pct':     term.get('goal', 0)         / n,",
    "        'timeout_pct':  term.get('timeout', 0)      / n,",
    "        'oob_pct':      term.get('out_of_bounds', 0) / n,",
    "    })",
    "asym_df = pd.DataFrame(asym_rows)",
    "print('[arrival_v2_simple + ASYM critic — per-seed test]')",
    "print(asym_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))",
    "",
    "# Hard-coded anchors from completed runs",
    "EFF_V2_SYM = {",
    "    'objective': 'efficiency_v2',",
    "    'critic': 'sym (10-D)',",
    "    'success_rate_mean': 0.225, 'success_rate_std': 0.005,",
    "    'mean_return_mean':  -371.0, 'mean_return_std':  0.41,",
    "    'goal_pct': 0.225, 'timeout_pct': 0.775, 'oob_pct': 0.000,",
    "    'n_seeds': 5,",
    "}",
    "ARR_V2_S_SYM = {",
    "    'objective': 'arrival_v2_simple',",
    "    'critic': 'sym (10-D)',",
    "    'success_rate_mean': 0.215, 'success_rate_std': 0.015,",
    "    'mean_return_mean':  -98.23, 'mean_return_std':  4.03,",
    "    'goal_pct': 0.215, 'timeout_pct': 0.525, 'oob_pct': 0.260,",
    "    'n_seeds': 2,",
    "}",
    "",
    "if not asym_df.empty:",
    "    asym_summary = {",
    "        'objective': 'arrival_v2_simple',",
    "        'critic': 'asym (10-D + 2-D priv)',",
    "        'success_rate_mean': asym_df['success_rate'].mean(),",
    "        'success_rate_std':  asym_df['success_rate'].std(ddof=0) if len(asym_df) > 1 else 0.0,",
    "        'mean_return_mean':  asym_df['mean_return'].mean(),",
    "        'mean_return_std':   asym_df['mean_return'].std(ddof=0) if len(asym_df) > 1 else 0.0,",
    "        'goal_pct':    asym_df['goal_pct'].mean(),",
    "        'timeout_pct': asym_df['timeout_pct'].mean(),",
    "        'oob_pct':     asym_df['oob_pct'].mean(),",
    "        'n_seeds':     len(asym_df),",
    "    }",
    "    cmp = pd.DataFrame([EFF_V2_SYM, ARR_V2_S_SYM, asym_summary])",
    "    print()",
    "    print('[three-way summary]')",
    "    cols = ['objective', 'critic', 'n_seeds',",
    "            'success_rate_mean', 'success_rate_std',",
    "            'mean_return_mean',  'mean_return_std',",
    "            'goal_pct', 'timeout_pct', 'oob_pct']",
    "    print(cmp[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))",
))

# ---------------------------------------------------------------------------
# §7 Verdict
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 7. 自动 verdict — asym critic 能否打开 s0 actor 在 upstream 的天花板？",
    "",
    "事先 commit 的判定规则（§0 回顾）：",
    "",
    "| arrival_v2_simple + asym test success | verdict |",
    "|---:|---|",
    "| ≥ 0.6 | **asym 主导** — paper headline result |",
    "| 0.4–0.6 | **asym 部分有效** — 加 seed 或调 actor_update_mode |",
    "| ≤ 0.4 | **β floor** — s0 actor 物理上限 — C1 重定义为 sensor-floor spoke |",
))
CELLS.append(code(
    "if asym_df.empty:",
    "    print('[abort] no asym seeds completed; cannot verdict')",
    "else:",
    "    asym_succ = asym_df['success_rate'].mean()",
    "    sym_succ_arr = ARR_V2_S_SYM['success_rate_mean']",
    "    sym_succ_eff = EFF_V2_SYM['success_rate_mean']",
    "    delta_vs_arr_sym = (asym_succ - sym_succ_arr) * 100",
    "    delta_vs_eff_sym = (asym_succ - sym_succ_eff) * 100",
    "",
    "    print('=' * 72)",
    "    print(f'C1 asym critic ablation — verdict (n_seeds={len(asym_df)})')",
    "    print('-' * 72)",
    "    print(f'  efficiency_v2 + sym       success = {sym_succ_eff:.4f}  (5-seed)')",
    "    print(f'  arrival_v2_simple + sym   success = {sym_succ_arr:.4f}  (2-seed)')",
    "    print(f'  arrival_v2_simple + ASYM  success = {asym_succ:.4f}  ({len(asym_df)}-seed)')",
    "    print(f'  Δ vs arr_v2_s+sym = {delta_vs_arr_sym:+.2f} pp   '",
    "          f'Δ vs eff_v2+sym = {delta_vs_eff_sym:+.2f} pp')",
    "    print('-' * 72)",
    "",
    "    if asym_succ >= 0.60:",
    "        verdict = 'asym 主导：privileged critic 解锁 s0 actor (paper headline)'",
    "        next_step = (",
    "            'Plan: (1) 升 5-seed 严格统计 + bootstrap CI；'",
    "            '(2) 在其他 upstream spokes (e.g. sbs_u15) 复现；'",
    "            '(3) 对比 priv-actor-update-mode zeros vs batch；'",
    "            '(4) paper 主结果章节落地：CLAUDE.md §3 在 offline 同样有效'",
    "        )",
    "    elif asym_succ >= 0.40:",
    "        verdict = 'asym 部分有效：privileged signal 帮但 s0 actor 仍受限'",
    "        next_step = (",
    "            'Plan: (1) 加 seed 收紧 std；(2) 试 --privileged-actor-update-mode batch '",
    "            '(actor 也用 priv_obs 训练但 eval 时仍 s0)；(3) 升 critic_bc_coef 看是否 BC 抑制了 priv signal'",
    "        )",
    "    else:",
    "        verdict = 'β floor：s0 actor 物理上限 — C1 重定义为 sensor-floor spoke'",
    "        next_step = (",
    "            'Plan: (1) C1 redefine：success target 从 90% 降到 ~30%，定位为「s0 deployment floor」；'",
    "            '(2) 加新 spoke C1-s1 (probe_layout=s1) 看 sensor 升级有多大提升；'",
    "            '(3) paper 写「s0 在 upstream 上不可 deploy，需要 s1+」'",
    "        )",
    "",
    "    print(f'  verdict   : {verdict}')",
    "    print(f'  next-step : {next_step}')",
    "    print('=' * 72)",
))

# ---------------------------------------------------------------------------
# §8 Report writing checklist
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 8. 报告写入清单",
    "",
    "**任何 verdict（必填）**",
    "- `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md` "
    "新增 §「C1 asym critic ablation」段：anchor 配置 + n_seeds + Δ + termination 分布",
    "- `docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md` C1 行：本 ablation 结果",
    "- `docs/rebrac_experiment_report.md`（如果存在）：把 reward + asym 两次 ablation 串成一段叙事",
    "",
    "**verdict = asym 主导（≥ 0.6）**",
    "- 这是 **paper headline result**：CLAUDE.md §3 (Asymmetric Critic with privileged hull-integral flow) "
    "在 offline RL 上同样有效",
    "- 立即任务：5-seed × bootstrap CI；在 sbs_u15_upstream 等其他 upstream spokes 复现",
    "- paper 主章节升级：online + offline 双线都用 asym critic 作为方法论亮点",
    "",
    "**verdict = asym 部分有效（0.4–0.6）**",
    "- 需要进一步 ablation：actor_update_mode zeros vs batch；critic_bc_coef sweep",
    "- spec 加 backlog：「Asym critic on offline — full hyperparameter sweep」",
    "",
    "**verdict = β floor（≤ 0.4）**",
    "- C1 的两条修复路径（reward / asym critic）都验证为无效 → s0 在 upstream 是 fundamental limitation",
    "- 提议 C1 重定义为「sensor floor spoke」，broad_validation §3 的 success target 分级",
    "- backlog：「C1-s1 / C1-s2 验证 sensor 升级提升幅度」",
))

# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "rebrac_c1_asym_critic_ablation.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
