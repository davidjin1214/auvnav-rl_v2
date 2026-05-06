"""One-shot builder for notebooks/rebrac_c1_train_convergence_check.ipynb.

Pure log-reading notebook (no GPU). Loads train_log.jsonl from the existing
arrival_v2_simple + sym critic seed_42 / seed_44 runs, plots the 6 ReBRAC
metrics over training, and runs an automatic convergence diagnostic that
compares the late-window mean vs the mid-window mean to test the hypothesis
"upstream needs more epochs".

Run from repo root:
    python -m scripts._build_c1_convergence_check_notebook
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

CELLS.append(md(
    "# ReBRAC C1 — Train Convergence Check (no GPU)",
    "",
    "## 假设（用户提出）",
    "",
    "> upstream 难度比 cross_stream 高，C1 在 64 epochs 可能没收敛。"
    "现有 broad_validation 全 spokes 用同样 64 epochs，对 cross_stream 够用，"
    "但 upstream 需要更多。",
    "",
    "## 测试方法（cheap，~30s，无 GPU）",
    "",
    "ReBRAC 训练每 1000 步写一行到 `<save_dir>/train_log.jsonl`，含：",
    "",
    "| metric | 含义 |",
    "|---|---|",
    "| `critic_loss` | (q1_loss + q2_loss) / 2 |",
    "| `actor_loss` | actor 总损失 = −lambda·Q + β1·BC |",
    "| `bc_loss` | actor 端 BC 项 (在 actor_penalty_coef 之前) |",
    "| `mean_q` | Q(s, π(s)) 均值 |",
    "| `target_q` | TD target 均值 |",
    "| `td_abs_error` | mean abs(q1_pred − q_target) |",
    "| `critic_penalty` | critic 端 BC 惩罚项 (在 critic_penalty_coef 之前) |",
    "| `lambda` | Q-normalization 系数 |",
    "",
    "如果 64 epochs 已 plateau → H1（epochs 不够）失败 → 跑 [asym critic notebook](rebrac_c1_asym_critic_ablation.ipynb)。",
    "如果末段仍在显著下降 → H1 成立 → 跑 [epoch sensitivity ablation notebook](rebrac_c1_epoch_sensitivity_ablation.ipynb)。",
    "",
    "## 自动 verdict 规则",
    "",
    "对每个 metric `m`，定义：",
    "- mid-window: epoch 16–32（训练中段）",
    "- late-window: epoch 56–64（训练末段）",
    "- relative_change = `(mean_late − mean_mid) / max(|mean_mid|, eps)`",
    "",
    "| metric | 期望「已收敛」signal |",
    "|---|---|",
    "| critic_loss / td_abs_error / bc_loss | `|relative_change| < 5%`（损失 plateau） |",
    "| mean_q / target_q | `|relative_change| < 3%`（Q-landscape 稳定） |",
    "",
    "≥ 4/5 metrics 满足条件 → **plateaued**，H1 失败。",
    "否则 → **still moving**，H1 可能成立，建议跑 epoch sensitivity。",
))

CELLS.append(md("## 1. Drive mount + cd"))
CELLS.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive', force_remount=True)",
    "%cd /content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5",
))

CELLS.append(md(
    "## 2. Config — 现有 reward ablation 训练日志路径",
    "",
    "复用 reward ablation completed run 的 train_log.jsonl。",
))
CELLS.append(code(
    "from pathlib import Path",
    "",
    "DATASET_NAME = 'crosscomp_s0_h4_arrival_v2_simple_re150_u10upstream_fixdone_ep1000'",
    "PAIR_TAG = 'actorb_4p0__criticb_2p0'",
    "CHECKPOINT_ROOT = Path('checkpoints/offline/rebrac/c1_reward_ablation')",
    "SEEDS = [42, 44]",
    "",
    "import math",
    "TRAIN_EPOCHS = 64  # the budget used by reward ablation",
    "DATASET_TRANSITIONS = 268_329  # from sanity card",
    "BATCH_SIZE = 256",
    "STEPS_PER_EPOCH = math.ceil(DATASET_TRANSITIONS / BATCH_SIZE)  # 1049 (drop_last=False)",
    "TOTAL_STEPS = STEPS_PER_EPOCH * TRAIN_EPOCHS  # ~67k",
    "",
    "log_paths = {",
    "    seed: CHECKPOINT_ROOT / DATASET_NAME / PAIR_TAG / f'seed_{seed}' / 'train_log.jsonl'",
    "    for seed in SEEDS",
    "}",
    "",
    "for seed, p in log_paths.items():",
    "    print(f'seed {seed}: {p}  (exists={p.exists()})')",
    "print()",
    "print(f'steps_per_epoch = {STEPS_PER_EPOCH}, total_steps = {TOTAL_STEPS}')",
))

CELLS.append(md(
    "## 3. 读 train_log.jsonl + 加 epoch 索引",
))
CELLS.append(code(
    "import json",
    "import pandas as pd",
    "",
    "def load_train_log(path: Path) -> pd.DataFrame:",
    "    if not path.exists():",
    "        return pd.DataFrame()",
    "    rows = []",
    "    for line in path.read_text(encoding='utf-8').splitlines():",
    "        line = line.strip()",
    "        if not line:",
    "            continue",
    "        try:",
    "            rows.append(json.loads(line))",
    "        except json.JSONDecodeError as e:",
    "            print(f'[warn] bad line: {e}')",
    "    df = pd.DataFrame(rows)",
    "    # train_offline writes the column as 'train_step' (not 'step').",
    "    if 'train_step' in df.columns:",
    "        df['epoch_frac'] = df['train_step'] / STEPS_PER_EPOCH",
    "    elif 'step' in df.columns:  # legacy fallback",
    "        df['epoch_frac'] = df['step'] / STEPS_PER_EPOCH",
    "    return df",
    "",
    "logs = {seed: load_train_log(p) for seed, p in log_paths.items()}",
    "",
    "for seed, df in logs.items():",
    "    if df.empty:",
    "        print(f'[warn] seed {seed}: empty / missing log')",
    "        continue",
    "    step_col = 'train_step' if 'train_step' in df.columns else 'step'",
    "    print(f'seed {seed}: {len(df)} log rows  '",
    "          f'({step_col} {df[step_col].min()} → {df[step_col].max()})')",
    "    print(f'  columns: {list(df.columns)}')",
    "    break  # show columns once",
))

CELLS.append(md(
    "## 4. Plot 6-panel — 关键 ReBRAC metrics over training",
    "",
    "X 轴用 epoch（不是 step），方便对照 64-epoch budget。",
))
CELLS.append(code(
    "import matplotlib.pyplot as plt",
    "",
    "METRICS = [",
    "    ('critic_loss',     'critic_loss',    'log'),",
    "    ('actor_loss',      'actor_loss',     'linear'),",
    "    ('bc_loss',         'bc_loss',        'log'),",
    "    ('mean_q',          'mean_q (Q at policy actions)', 'linear'),",
    "    ('target_q',        'target_q (TD target)',         'linear'),",
    "    ('td_abs_error',    'td_abs_error',   'log'),",
    "]",
    "",
    "fig, axes = plt.subplots(2, 3, figsize=(16, 8))",
    "for ax, (key, title, scale) in zip(axes.flat, METRICS):",
    "    for seed, df in logs.items():",
    "        if df.empty or key not in df.columns:",
    "            continue",
    "        ax.plot(df['epoch_frac'], df[key], label=f'seed {seed}', alpha=0.85)",
    "    ax.set_title(title)",
    "    ax.set_xlabel('epoch')",
    "    ax.set_yscale(scale)",
    "    ax.axvspan(16, 32, color='gray', alpha=0.10, label='mid-window')",
    "    ax.axvspan(56, 64, color='C2',   alpha=0.15, label='late-window')",
    "    ax.legend(fontsize=8)",
    "plt.tight_layout()",
    "plt.show()",
))

CELLS.append(md(
    "## 5. 自动 convergence diagnostic",
    "",
    "Per metric: `mean_late / mean_mid` 相对变化。"
    "Per seed 计算，再 seed 间平均。",
))
CELLS.append(code(
    "import numpy as np",
    "",
    "EPS = 1e-8",
    "MID_EPOCHS = (16, 32)",
    "LATE_EPOCHS = (56, 64)",
    "",
    "DIAGNOSTIC_KEYS = [",
    "    ('critic_loss',  0.05, 'plateau if |Δ| < 5%'),",
    "    ('actor_loss',   0.05, 'plateau if |Δ| < 5%'),",
    "    ('bc_loss',      0.05, 'plateau if |Δ| < 5%'),",
    "    ('td_abs_error', 0.05, 'plateau if |Δ| < 5%'),",
    "    ('mean_q',       0.03, 'plateau if |Δ| < 3%'),",
    "    ('target_q',     0.03, 'plateau if |Δ| < 3%'),",
    "]",
    "",
    "def window_mean(df: pd.DataFrame, key: str, lo: float, hi: float) -> float:",
    "    mask = (df['epoch_frac'] >= lo) & (df['epoch_frac'] <= hi)",
    "    sub = df.loc[mask, key]",
    "    return float(sub.mean()) if len(sub) > 0 else float('nan')",
    "",
    "rows = []",
    "for key, threshold, _ in DIAGNOSTIC_KEYS:",
    "    seed_changes = []",
    "    for seed, df in logs.items():",
    "        if df.empty or key not in df.columns:",
    "            continue",
    "        m_mid  = window_mean(df, key, *MID_EPOCHS)",
    "        m_late = window_mean(df, key, *LATE_EPOCHS)",
    "        rel = (m_late - m_mid) / max(abs(m_mid), EPS)",
    "        seed_changes.append({'seed': seed, 'mid': m_mid, 'late': m_late, 'rel_change': rel})",
    "    if not seed_changes:",
    "        continue",
    "    avg_rel = float(np.mean([s['rel_change'] for s in seed_changes]))",
    "    plateau = abs(avg_rel) < threshold",
    "    rows.append({",
    "        'metric':       key,",
    "        'mean_mid':     float(np.mean([s['mid']  for s in seed_changes])),",
    "        'mean_late':    float(np.mean([s['late'] for s in seed_changes])),",
    "        'rel_change':   avg_rel,",
    "        'threshold':    threshold,",
    "        'plateau':      plateau,",
    "    })",
    "",
    "diag_df = pd.DataFrame(rows)",
    "if not diag_df.empty:",
    "    print('[per-metric convergence diagnostic]')",
    "    print(diag_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}' if isinstance(x, float) else str(x)))",
    "",
    "    n_plateau = int(diag_df['plateau'].sum())",
    "    n_total = len(diag_df)",
    "    print()",
    "    print(f'plateau hits: {n_plateau}/{n_total}')",
))

CELLS.append(md(
    "## 6. Verdict — H1 (epochs 不够) 是否成立？",
))
CELLS.append(code(
    "if diag_df.empty:",
    "    print('[abort] no metrics — cannot verdict')",
    "else:",
    "    n_plateau = int(diag_df['plateau'].sum())",
    "    n_total = len(diag_df)",
    "    n_total_ge1 = max(n_total, 1)",
    "    plateau_ratio = n_plateau / n_total_ge1",
    "",
    "    print('=' * 72)",
    "    print(f'C1 train convergence verdict — n_metrics={n_total}, n_plateau={n_plateau}')",
    "    print('-' * 72)",
    "",
    "    if plateau_ratio >= 4 / 6:",
    "        verdict = 'H1 失败：64 epochs 已基本 plateau'",
    "        next_step = (",
    "            'epochs 不是 C1 失败主因。继续跑 [rebrac_c1_asym_critic_ablation.ipynb] '",
    "            '验证 asym critic 假设 (H2)'",
    "        )",
    "    elif plateau_ratio >= 2 / 6:",
    "        verdict = 'H1 弱信号：部分 metric 还在动'",
    "        next_step = (",
    "            'mixed signal — 建议先跑 epoch sensitivity (单 seed × 256 epochs) 廉价确认；'",
    "            '若 success 显著上升再扩 broad_validation upstream budget'",
    "        )",
    "    else:",
    "        verdict = 'H1 成立：64 epochs 大部分 metric 仍在显著移动'",
    "        next_step = (",
    "            '强烈建议跑 [rebrac_c1_epoch_sensitivity_ablation.ipynb] '",
    "            '把 epochs 升到 256 看 success 曲线；可能需要重做 broad_validation upstream spokes'",
    "        )",
    "",
    "    print(f'  verdict   : {verdict}')",
    "    print(f'  next-step : {next_step}')",
    "    print('=' * 72)",
))

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "rebrac_c1_train_convergence_check.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
