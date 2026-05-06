"""One-shot builder for notebooks/rebrac_c1_s1_sensor_upgrade.ipynb.

C1 sensor-floor ablation chain (reward / asym critic / 4× epochs) confirmed
sensor-info bottleneck. This follow-up upgrades the probe from s0 (DVL only,
10-D obs) to s1 (DVL + short-range ADCP, 12-D obs) while keeping every other
variable identical to C1 P1 anchor — collector policy, task geometry, target
speed, history length, ReBRAC hyperparameters, evaluation manifest.

Verdict thresholds (committed before run):
    success ≥ 0.50  -> sensor upgrade unlocks upstream u10 (paper headline)
    0.30–0.50       -> partial; trigger C1-s2 follow-up
    success < 0.30  -> upstream u10 is task-fundamental limitation, not sensor

Run from repo root:
    python -m scripts._build_c1_s1_sensor_upgrade_notebook
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
    "# ReBRAC C1-s1 — Sensor Upgrade Follow-up",
    "",
    "> 文档锚点：[spec §13.6](../docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) · "
    "[plan Task 11A Step 8](../docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md) · "
    "[report §10A.4](../docs/rebrac_experiment_report.md)",
    "",
    "## 前传 — C1 sensor-floor 已确认",
    "",
    "| run | dataset / critic / budget | seeds | success | mean_R |",
    "|---|---|---:|---:|---:|",
    "| C1 P1 anchor (s0, eff_v2, sym, 64 ep) | crosscomp_s0_h4_eff_v2_u10upstream | 5 | **0.225 ± 0.005** | −371 ± 0.41 |",
    "| Ablation A: reward swap (arr_v2_s) | crosscomp_s0_h4_arr_v2_s_u10upstream | 2 | 0.215 ± 0.015 | −98.23 ± 4.03 |",
    "| Ablation B: asym critic | (same dataset, asym critic) | 2 | 0.195 ± 0.015 | −114.87 ± 8.02 |",
    "| Ablation C: 4× epochs (256 ep) | (arr_v2_s, sym, 256 ep) | 1 | 0.220 (ep256) | −96.16 |",
    "",
    "**结论**：C1 (s0 / upstream / u10 / crosscomp / Re150) 是 sensor floor spoke。"
    "三个独立 root-cause 假设（reward / privileged critic / training budget）全部排除。"
    "所有干预钉在 0.19–0.23 区间（4-pp 区间，远小于 ablation 的 ±3pp 噪声半径）。",
    "",
    "## 假设 — sensor 升级 (s0 → s1) 能否解锁 upstream u10？",
    "",
    "**s1 layout**（[CLAUDE.md §Environment Details](../CLAUDE.md)）：",
    "- 2 probes at (0, 0) + (4.5, 0) 米；DVL water-track + 2 MHz 短程 ADCP",
    "- 提供 ~3 步前向流场 advance warning",
    "- obs_dim = 12（8 base + 2 probes × 2）",
    "",
    "其他维度全部与 C1 P1 anchor 一致：",
    "",
    "| 维度 | C1 P1 anchor | **本 notebook (C1-s1)** |",
    "|---|---|---|",
    "| probe layout | s0 (10-D) | **s1 (12-D)** |",
    "| collector | crosscomp | crosscomp |",
    "| objective | efficiency_v2 | **efficiency_v2**（不混 reward 干预，纯 sensor 维度对比）|",
    "| task geometry | upstream | upstream |",
    "| target speed | 1.5 | 1.5 |",
    "| history length | 4 | 4 |",
    "| flow | wake_v8_U1p00_Re150 | 同 |",
    "| critic | sym | **sym**（先排除 sensor 单变量；asym 留作后续）|",
    "| β1, β2 | 4.0, 2.0 | 4.0, 2.0 |",
    "| epochs | 64 | 64 |",
    "| seeds | 42–46 | 42, 44（2-seed probe）|",
    "",
    "唯一变量：`--probe-layout s0 → s1`。隔离 sensor 维度信号。",
    "",
    "## Verdict 逻辑（事先 commit）",
    "",
    "| C1-s1 test success (100 ep, 2-seed mean) | verdict | 含义 |",
    "|---:|---|---|",
    "| **≥ 0.50** | **sensor 升级解锁** | s1 提供的短程 ADCP 信号足以让 actor 在 upstream u10 deploy；paper headline + sim2real narrative 强证据 |",
    "| **0.30–0.50** | **部分有效** | s1 帮但仍受限；触发 C1-s2 follow-up（s2 = DVL + 长程 ADCP + 横向梯度，4 probes, 16-D）|",
    "| **< 0.30** | **task-fundamental floor** | upstream u10 在所有 deployable sensor 上都接近 floor；§6 limitations 写「u10 流速 + upstream geometry 的物理上限」|",
    "",
    "## Cost",
    "",
    "- 数据收集 (1000 ep × s1) ~30 min L4",
    "- 训练 (2 seeds × 64 ep × s1) ~1.5h L4",
    "- Test eval (2 seeds × 100 ep × 6 worker) ~5 min",
    "- **总计 ~2h L4**",
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
# §2 Common config
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 2. 通用配置",
    "",
    "唯一变量：`PROBE_LAYOUT = s1`。其余复用 C1 P1 anchor 配置。",
    "manifest 沿用 `c1_reward_ablation/` 的 `test_100`（与所有 C1 ablation 数据集合可比）。",
))
CELLS.append(code(
    "import os",
    "from pathlib import Path",
    "",
    "# ---- sensor 维度 (唯一变量) ----",
    "PROBE_LAYOUT = 's1'  # ← C1 P1 用 s0；本 notebook 升级到 s1",
    "",
    "# ---- 与 C1 P1 anchor 完全一致 ----",
    "OBJECTIVE = 'efficiency_v2'",
    "TASK_GEOMETRY = 'upstream'",
    "TARGET_SPEED = 1.5",
    "HISTORY_LENGTH = 4",
    "FLOW_PATH = 'wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy'",
    "BENCHMARK_KEY = 'single_u10_upstream_tgt15'",
    "COLLECTOR_POLICY = 'crosscomp'",
    "DATASET_EPISODES = 1000",
    "DATASET_SEED = 0",
    "COLLECT_WORKERS = 8",
    "VAL_EPISODES = 40",
    "TEST_EPISODES = 100",
    "",
    "# ---- 训练 anchor (与 C1 P1 完全一致) ----",
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
    "# ---- 输出树（与 ablation 隔离）----",
    "DATASET_NAME = (",
    "    f'{COLLECTOR_POLICY}_{PROBE_LAYOUT}_h{HISTORY_LENGTH}_{OBJECTIVE}'",
    "    f'_re150_u10upstream_fixdone_ep{DATASET_EPISODES}'",
    ")",
    "DATASET_DIR = Path('offline_data') / DATASET_NAME",
    "DATASET_DIR_STR = str(DATASET_DIR)",
    "TRANSITIONS_NPZ = str(DATASET_DIR / 'transitions.npz')",
    "",
    "CHECKPOINT_ROOT = Path('checkpoints/offline/rebrac/c1_s1_sensor_upgrade')",
    "RESULTS_ROOT = Path('results/offline/rebrac/c1_s1_sensor_upgrade')",
    "",
    "# 复用 c1_reward_ablation 的 manifest（与所有 ablation 数据可比）",
    "VAL_MANIFEST_PATH  = Path(f'benchmarks/c1_reward_ablation/val_{VAL_EPISODES}/{BENCHMARK_KEY}.json')",
    "TEST_MANIFEST_PATH = Path(f'benchmarks/c1_reward_ablation/test_{TEST_EPISODES}/{BENCHMARK_KEY}.json')",
    "VAL_MANIFEST_DIR_STR  = str(VAL_MANIFEST_PATH.parent)",
    "TEST_MANIFEST_DIR_STR = str(TEST_MANIFEST_PATH.parent)",
    "VAL_MANIFEST_STR  = str(VAL_MANIFEST_PATH)",
    "TEST_MANIFEST_STR = str(TEST_MANIFEST_PATH)",
    "",
    "print(f'PROBE_LAYOUT    = {PROBE_LAYOUT}  (← unique vs C1 P1 anchor s0)')",
    "print(f'DATASET_DIR     = {DATASET_DIR}')",
    "print(f'VAL_MANIFEST    = {VAL_MANIFEST_PATH}  (exists={VAL_MANIFEST_PATH.exists()})')",
    "print(f'TEST_MANIFEST   = {TEST_MANIFEST_PATH}  (exists={TEST_MANIFEST_PATH.exists()})')",
    "print(f'CHECKPOINT_ROOT = {CHECKPOINT_ROOT}')",
    "print(f'RESULTS_ROOT    = {RESULTS_ROOT}')",
))

# ---------------------------------------------------------------------------
# §3 Generate manifests if missing
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 3. Manifest sanity",
    "",
    "C1 ablation 已经生成过 `val_40` / `test_100` manifest（reused here）。"
    "如果 Drive 同步前路径不存在，自动生成。",
))
CELLS.append(code(
    "VAL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)",
    "TEST_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)",
    "",
    "if not VAL_MANIFEST_PATH.exists():",
    "    !python -m scripts.generate_standard_benchmarks \\",
    "        --benchmarks {BENCHMARK_KEY} \\",
    "        --episodes {VAL_EPISODES} \\",
    "        --output-dir {VAL_MANIFEST_DIR_STR}",
    "else:",
    "    print(f'[skip] val manifest exists: {VAL_MANIFEST_PATH}')",
    "",
    "if not TEST_MANIFEST_PATH.exists():",
    "    !python -m scripts.generate_standard_benchmarks \\",
    "        --benchmarks {BENCHMARK_KEY} \\",
    "        --episodes {TEST_EPISODES} \\",
    "        --output-dir {TEST_MANIFEST_DIR_STR}",
    "else:",
    "    print(f'[skip] test manifest exists: {TEST_MANIFEST_PATH}')",
))

# ---------------------------------------------------------------------------
# §4 Collect dataset (s1, upstream, efficiency_v2)
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 4. 任务 1 — Collect crosscomp dataset (s1, upstream, efficiency_v2)",
    "",
    "1000 episodes × upstream / u10 / **s1** / h4 / efficiency_v2。"
    "complexity 与 C1 efficiency_v2 dataset 完全一致，唯一变量是 probe layout。",
    "",
    "预期 collector_success_rate ≥ 0.95 (s1 比 s0 多 3 步 advance warning，crosscomp 在 s1 上更容易成功)。",
))
CELLS.append(code(
    "if (DATASET_DIR / 'transitions.npz').exists():",
    "    print(f'[skip] dataset exists: {DATASET_DIR}')",
    "else:",
    "    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)",
    "    !python -m scripts.collect_offline_data \\",
    "        --policy {COLLECTOR_POLICY} \\",
    "        --flow {FLOW_PATH} \\",
    "        --probe-layout {PROBE_LAYOUT} \\",
    "        --task-geometry {TASK_GEOMETRY} \\",
    "        --target-speed {TARGET_SPEED} \\",
    "        --history-length {HISTORY_LENGTH} \\",
    "        --objective {OBJECTIVE} \\",
    "        --episodes {DATASET_EPISODES} \\",
    "        --seed {DATASET_SEED} \\",
    "        --num-workers {COLLECT_WORKERS} \\",
    "        --output-dir {DATASET_DIR_STR}",
))

# ---------------------------------------------------------------------------
# §5 Sanity card + obs_dim assertion
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 5. Sanity card + obs_dim hard-assert",
    "",
    "必须验证 `obs_dim_matches_probe_layout=true` 且 `obs_dim=12`（s1 = 8 base + 2 probes × 2）。",
    "如果 obs_dim 与 s1 不匹配（例如收集时 layout 配置错），数据集不可用，停下。",
))
CELLS.append(code(
    "!python -m scripts.write_sanity_card \\",
    "    --dataset-dir {DATASET_DIR_STR} \\",
    "    --expected-probe-layout {PROBE_LAYOUT}",
))
CELLS.append(code(
    "import json",
    "import numpy as np",
    "",
    "card = json.loads((DATASET_DIR / 'sanity_card.json').read_text())",
    "print('[sanity card]')",
    "for k, v in card.items():",
    "    print(f'  {k:35s} = {v}')",
    "",
    "# Hard-assert: obs_dim must match s1 (12-D)",
    "EXPECTED_OBS_DIM = 12",
    "assert card['obs_dim'] == EXPECTED_OBS_DIM, (",
    "    f'obs_dim mismatch: got {card[\"obs_dim\"]}, expected {EXPECTED_OBS_DIM} (s1)'",
    ")",
    "assert card.get('obs_dim_matches_probe_layout', False), (",
    "    'obs_dim_matches_probe_layout=False — collector / probe_layout misaligned'",
    ")",
    "print()",
    "print(f'[OK] obs_dim={card[\"obs_dim\"]} matches s1 layout')",
    "print(f'     collector_success_rate={card.get(\"collector_success_rate\", \"N/A\")}')",
    "print(f'     n_transitions={card.get(\"n_transitions\", \"N/A\")}')",
))

# ---------------------------------------------------------------------------
# §6 Train ReBRAC × 2 seeds (s1, sym critic, 64 ep)
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 6. 任务 2 — Train ReBRAC anchor (β1=4.0, β2=2.0) × 2 seeds (s1, sym critic, 64 ep)",
    "",
    "强制 collect/train objective 一致（[`scripts/train_offline.py:190-194`](../scripts/train_offline.py)）。"
    "本节训练命令与 C1 P1 anchor 完全一致，仅 `--probe-layout s0 → s1`。",
    "",
    "预期 ~30 min/seed × 2 = ~1h L4 (相比 s0 略慢，因为 obs_dim 12 > 10)。",
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
    "    print(f'\\n========== train seed {seed} (s1) → {save_dir} ==========')",
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
    "        --eval-every 0 \\",
    "        --skip-final-eval \\",
    "        --seed {seed} \\",
    "        --device cuda \\",
    "        --save-dir {save_dir_str}",
))

# ---------------------------------------------------------------------------
# §7 Test eval
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 7. Test eval — 100-episode test manifest",
    "",
    "evaluate_offline 自动从 `trainer_state.json` 读出 `probe_layout=s1`，重建 12-D 观测。"
    "Eval 时 actor 在 deployment 状态下使用 s1 sensor（与训练一致；s1 本身就是 deploy-realistic）。",
))
CELLS.append(code(
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
    "    print(f'\\n========== eval seed {seed} (s1) → {out_path} ==========')",
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
# §8 Five-way comparison: C1 P1 / Ablation A / Ablation B / Ablation C / C1-s1
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 8. 五向对比 — C1 P1 / Ablation A/B/C / 本 C1-s1",
    "",
    "前 4 行是已知锚点（hardcoded from completed ablations）；第 5 行是本 notebook 输出。"
    "包含 termination 分布以观察 sensor 维度对 actor 探索行为的影响。",
))
CELLS.append(code(
    "import json",
    "import pandas as pd",
    "",
    "s1_rows = []",
    "for seed in TRAIN_SEEDS:",
    "    p = TEST_DIR / f'seed_{seed}.json'",
    "    if not p.exists():",
    "        print(f'[warn] missing: {p}')",
    "        continue",
    "    d = json.loads(p.read_text(encoding='utf-8'))",
    "    term = d.get('eval_termination_counts', {})",
    "    n = max(int(d.get('eval_episodes', 100)), 1)",
    "    s1_rows.append({",
    "        'seed': seed,",
    "        'success_rate': d['eval_success_rate'],",
    "        'mean_return':  d['eval_return'],",
    "        'safety_cost':  d.get('eval_safety_cost', 0.0),",
    "        'goal_pct':     term.get('goal', 0)         / n,",
    "        'timeout_pct':  term.get('timeout', 0)      / n,",
    "        'oob_pct':      term.get('out_of_bounds', 0) / n,",
    "    })",
    "s1_df = pd.DataFrame(s1_rows)",
    "print('[C1-s1 (s1 sensor + eff_v2 + sym critic) — per-seed test]')",
    "print(s1_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))",
    "",
    "# Hard-coded anchors from completed runs (see spec §13.3 / report §10A.2)",
    "C1_P1_ANCHOR = {",
    "    'config': 's0 + eff_v2 + sym + 64ep',",
    "    'success_rate_mean': 0.225, 'success_rate_std': 0.005,",
    "    'mean_return_mean': -371.0, 'mean_return_std': 0.41,",
    "    'goal_pct': 0.225, 'timeout_pct': 0.775, 'oob_pct': 0.000,",
    "    'n_seeds': 5,",
    "}",
    "ABLATION_A = {",
    "    'config': 's0 + arr_v2_s + sym + 64ep',",
    "    'success_rate_mean': 0.215, 'success_rate_std': 0.015,",
    "    'mean_return_mean': -98.23, 'mean_return_std': 4.03,",
    "    'goal_pct': 0.215, 'timeout_pct': 0.525, 'oob_pct': 0.260,",
    "    'n_seeds': 2,",
    "}",
    "ABLATION_B = {",
    "    'config': 's0 + arr_v2_s + asym + 64ep',",
    "    'success_rate_mean': 0.195, 'success_rate_std': 0.015,",
    "    'mean_return_mean': -114.87, 'mean_return_std': 8.02,",
    "    'goal_pct': 0.195, 'timeout_pct': 0.485, 'oob_pct': 0.320,",
    "    'n_seeds': 2,",
    "}",
    "ABLATION_C = {",
    "    'config': 's0 + arr_v2_s + sym + 256ep (single seed)',",
    "    'success_rate_mean': 0.220, 'success_rate_std': 0.0,",
    "    'mean_return_mean': -96.16, 'mean_return_std': 0.0,",
    "    'goal_pct': 0.22, 'timeout_pct': 0.52, 'oob_pct': 0.26,",
    "    'n_seeds': 1,",
    "}",
    "",
    "if not s1_df.empty:",
    "    s1_summary = {",
    "        'config': 's1 + eff_v2 + sym + 64ep',",
    "        'success_rate_mean': s1_df['success_rate'].mean(),",
    "        'success_rate_std':  s1_df['success_rate'].std(ddof=0) if len(s1_df) > 1 else 0.0,",
    "        'mean_return_mean':  s1_df['mean_return'].mean(),",
    "        'mean_return_std':   s1_df['mean_return'].std(ddof=0) if len(s1_df) > 1 else 0.0,",
    "        'goal_pct':    s1_df['goal_pct'].mean(),",
    "        'timeout_pct': s1_df['timeout_pct'].mean(),",
    "        'oob_pct':     s1_df['oob_pct'].mean(),",
    "        'n_seeds':     len(s1_df),",
    "    }",
    "    cmp = pd.DataFrame([C1_P1_ANCHOR, ABLATION_A, ABLATION_B, ABLATION_C, s1_summary])",
    "    print()",
    "    print('[five-way summary]')",
    "    cols = ['config', 'n_seeds',",
    "            'success_rate_mean', 'success_rate_std',",
    "            'mean_return_mean',  'mean_return_std',",
    "            'goal_pct', 'timeout_pct', 'oob_pct']",
    "    print(cmp[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))",
))

# ---------------------------------------------------------------------------
# §9 Verdict
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 9. 自动 verdict — sensor 升级是否解锁 upstream u10？",
    "",
    "事先 commit 的判定规则（§0 回顾）：",
    "",
    "| C1-s1 test success | verdict | next step |",
    "|---:|---|---|",
    "| ≥ 0.50 | **sensor 升级解锁** | paper headline + sim2real narrative；考虑 5-seed bootstrap CI |",
    "| 0.30–0.50 | **部分有效** | 触发 C1-s2 follow-up（s2 = 4 probes, 16-D） |",
    "| < 0.30 | **task-fundamental floor** | upstream u10 在所有 deployable sensor 上接近 floor；§6 limitations |",
))
CELLS.append(code(
    "if s1_df.empty:",
    "    print('[abort] no C1-s1 seeds completed; cannot verdict')",
    "else:",
    "    s1_succ = s1_df['success_rate'].mean()",
    "    delta_vs_p1 = (s1_succ - C1_P1_ANCHOR['success_rate_mean']) * 100",
    "    delta_vs_abl_a = (s1_succ - ABLATION_A['success_rate_mean']) * 100",
    "    delta_vs_abl_b = (s1_succ - ABLATION_B['success_rate_mean']) * 100",
    "    delta_vs_abl_c = (s1_succ - ABLATION_C['success_rate_mean']) * 100",
    "",
    "    print('=' * 78)",
    "    print(f'C1-s1 sensor upgrade — verdict (n_seeds={len(s1_df)})')",
    "    print('-' * 78)",
    "    print(f'  C1 P1 anchor (s0 baseline)    success = {C1_P1_ANCHOR[\"success_rate_mean\"]:.4f}  '",
    "          f'(5-seed)')",
    "    print(f'  Ablation A (reward swap)      success = {ABLATION_A[\"success_rate_mean\"]:.4f}  '",
    "          f'(2-seed)')",
    "    print(f'  Ablation B (asym critic)      success = {ABLATION_B[\"success_rate_mean\"]:.4f}  '",
    "          f'(2-seed)')",
    "    print(f'  Ablation C (4× epochs)        success = {ABLATION_C[\"success_rate_mean\"]:.4f}  '",
    "          f'(1-seed, ep256)')",
    "    print(f'  C1-s1 (sensor upgrade s0→s1)  success = {s1_succ:.4f}  '",
    "          f'({len(s1_df)}-seed)')",
    "    print(f'  Δ vs C1 P1 anchor   = {delta_vs_p1:+.2f} pp')",
    "    print(f'  Δ vs Ablation A     = {delta_vs_abl_a:+.2f} pp')",
    "    print(f'  Δ vs Ablation B     = {delta_vs_abl_b:+.2f} pp')",
    "    print(f'  Δ vs Ablation C     = {delta_vs_abl_c:+.2f} pp')",
    "    print('-' * 78)",
    "",
    "    if s1_succ >= 0.50:",
    "        verdict = 'sensor 升级解锁：s1 提供的短程 ADCP 信号让 actor 在 upstream u10 deploy'",
    "        next_step = (",
    "            'Plan: (1) 升 5-seed × bootstrap CI 严格收紧；'",
    "            '(2) 在其他 upstream spokes (e.g. sbs_u15) 复现 sensor-upgrade 信号；'",
    "            '(3) paper §experiments 加 \"sensor floor + sensor upgrade\" 双向证据；'",
    "            '(4) §discussion 把 sensor 维度写为 paper-level controllable axis'",
    "        )",
    "    elif s1_succ >= 0.30:",
    "        verdict = '部分有效：s1 抬升 success 但未到 deploy-grade'",
    "        next_step = (",
    "            'Plan: (1) 触发 C1-s2 follow-up（s2 = DVL + 长程 ADCP + 横向梯度，4 probes 16-D）；'",
    "            '(2) 如 C1-s2 也仅部分有效，§discussion 论证 \"sensor 升级单调贡献但 u10 流速本身限制 ceiling\"'",
    "        )",
    "    else:",
    "        verdict = 'task-fundamental floor：upstream u10 在所有 deployable sensor 上都接近 floor'",
    "        next_step = (",
    "            'Plan: (1) §6 limitations 显式声明 \"u10 流速 + upstream geometry 是 task-fundamental 物理上限\"；'",
    "            '(2) C1 spoke 在 paper 中作为 \"deployment-impossible boundary\" 例子；'",
    "            '(3) 考虑加 follow-up：把 target_speed 提升到 2.0（顺流方向）看是否变 deploy-grade'",
    "        )",
    "",    "    print(f'  verdict   : {verdict}')",
    "    print(f'  next-step : {next_step}')",
    "    print('=' * 78)",
))

# ---------------------------------------------------------------------------
# §10 Report writing checklist
# ---------------------------------------------------------------------------
CELLS.append(md(
    "## 10. 报告写入清单",
    "",
    "**任何 verdict（必填）**",
    "- [`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](../docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) §13.6：把 C1-s1 follow-up 实测填入；按 verdict 分支更新 paper claim 框架",
    "- [`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](../docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md) Task 11A Step 8：mark complete",
    "- [`docs/rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md) §10A.4 + §10A.5：填实测；§10A.5 limitations 中 \"C1-s1 未跑\" 项删除或转写为残余 limitation",
    "",
    "**verdict = sensor 升级解锁 (≥ 0.50)**",
    "- 这是 paper-level 强 finding：sensor floor 既存在又可控（s0 floor = 0.225；s1 ≥ 0.50 解锁）",
    "- paper §experiments：加 \"sensor floor demonstration + sensor upgrade unlocks deployment\" 双向表",
    "- paper §discussion：sim2real 论证升级——deployable s0 不是终点，能把 sensor 维度作为 controllable axis",
    "- 立即任务：考虑 5-seed × bootstrap CI 收紧 effect size",
    "",
    "**verdict = 部分有效 (0.30–0.50)**",
    "- 触发 C1-s2 follow-up（s2 layout = 4 probes 16-D）",
    "- 完成 C1-s2 后，sensor 维度变成 ordered triple {s0=0.225, s1=??, s2=??}，paper 写 sensor monotonicity",
    "",
    "**verdict = task-fundamental floor (< 0.30)**",
    "- C1 在所有 deployable sensor 上失败 → C1 spoke 的论文角色变为 \"deployment-impossible boundary\"",
    "- §6 limitations 显式声明 \"u10 流速 + upstream geometry 是任务-传感器共同决定的物理上限\"",
    "- 可能 follow-up：尝试 target_speed=2.0（顺流方向）看 task 是否变 deploy-grade",
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

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "rebrac_c1_s1_sensor_upgrade.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
