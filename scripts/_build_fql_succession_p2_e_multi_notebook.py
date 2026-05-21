"""Build the E-multi notebook for FQL Succession P2 (post-diagnostic N1 cell).

E-multi = clean privileged 50% + clean goalseek 50% (σ=0), the 4th cell that
completes the 2×2 (noise × modality) matrix. Per the revised iff in
``docs/fql_succession_p2_mechanism_diagnostic.md`` (2026-05-22):

    FQL > ReBRAC iff dataset's BC anchor is corrupted by action-level noise.
    Multi-modality alone (σ=0) does NOT trigger FQL advantage.

So E-multi predicted verdict = **NULL**. If observed, the noise-axis claim is
confirmed in causal-isolation form (M-uni-noise = noise without modality →
positive; E-multi = modality without noise → null).

Run order:
1. Collect 2 clean components per ``docs/fql_succession_p2_collection_log.md §7``.
2. Concat into ``offline_data/fql_succession/e_multi_50priv_50goal_clean_1000``.
3. (Optional) audit.
4. Regenerate this notebook: ``python -m scripts._build_fql_succession_p2_e_multi_notebook``
5. Run on Colab L4 (~2 h, skip-resume ready).
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._build_fql_succession_p2_run_notebooks import (  # type: ignore
    code,
    md,
)

OUT = Path("notebooks/fql_succession_p2_run_cell_e_multi.ipynb")
CELL_ID = "e_multi"
DATASET = "offline_data/fql_succession/e_multi_50priv_50goal_clean_1000"


def section_0() -> dict:
    return md(
        "# FQL Succession P2 — **E-multi** (clean priv 50% + clean goalseek 50%, σ=0)",
        "",
        "> The N1 cell from `docs/fql_succession_p2_mechanism_diagnostic.md` §7.",
        "> Completes the 2×2 (noise × modality) matrix to causally separate",
        "> *modality* from *noise* in the FQL vs ReBRAC comparison.",
        "",
        "## Revised iff (post-diagnostic, 2026-05-22)",
        "",
        "Original spec v1.3 §1: *FQL > ReBRAC iff sub-optimal AND multi-modal*.",
        "Sprint-0 12 runs **falsified** this (M-uni-noise +20.5 pp, M-multi-mix −3.5 pp).",
        "",
        "**Revised**: *FQL > ReBRAC iff dataset's BC anchor signal is corrupted by",
        "action-level noise.* Modality alone does NOT trigger the advantage.",
        "",
        "## 2×2 matrix (modality × noise)",
        "",
        "| | σ ≈ 0 (clean) | σ ≥ 0.5 (noisy) |",
        "|---|---|---|",
        "| **uni-modal** (single policy) | E-uni  (sprint-0) → NULL ✓ | M-uni-noise (sprint-0) → **POSITIVE** ✓ |",
        "| **multi-modal** (mixture) | **E-multi (this notebook)** → predicts NULL | M-multi-mix (sprint-0) → ~NULL (σ=0.1 small) |",
        "",
        "If E-multi observes **NULL**, the noise-axis story closes:",
        "- Row 1 (clean): both cells null → no FQL advantage on clean data.",
        "- Row 2 (noisy): M-uni-noise positive, M-multi-mix near-ceiling so no headroom.",
        "- The discriminator IS the column (noise), NOT the row (modality).",
        "",
        "If E-multi observes **POSITIVE**, then modality is also a partial driver "
        "(weaker conclusion but still publishable). If E-multi observes some "
        "other unexpected direction, expand to n=4 seeds before drawing conclusions.",
        "",
        "## Scope",
        "",
        "- **2 algo × 2 seed [42, 0] = 4 runs** (matches sprint-0 cells).",
        "- Dataset reuse pattern: components in `_components/*_clean_seed*`, concat",
        "  into `e_multi_50priv_50goal_clean_1000`.",
        "- Wallclock: ~2 h on Colab L4 (4 × ~30 min each).",
        "- skip-resume on `agent_final.pt` (3-stage like sprint-0).",
    )


def section_1() -> list[dict]:
    return [
        md(
            "## 1. Environment sanity",
            "",
            "cwd = repo root + CUDA + auv_nav import OK.",
        ),
        code(
            "import os",
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
            "print('cwd =', Path.cwd())",
            "",
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
            "Sprint-0 共享所有 training/eval hyperparams,唯一差异是 `DATASET` 指向 E-multi。"
            "ReBRAC/FQL flags 与 sprint-0 完全一致(Gate B frozen 参数:β1=4.0, β2=2.0;"
            "FQL flow_steps=10, distill_alpha_bc=1.0)。",
        ),
        code(
            "import os",
            "import shutil",
            "from pathlib import Path",
            "",
            f"CELL_ID  = '{CELL_ID}'",
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
            "EVAL_EPISODES  = 50    # in-line eval per step",
            "TEST_EPISODES  = 100   # held-out test eval (§5)",
            "EVAL_NUM_WORKERS = 6",
            "",
            "# Gate B frozen flags (identical to sprint-0)",
            "REBRAC_FLAGS = (",
            "    '--actor-penalty-coef 4.0 --critic-penalty-coef 2.0 '",
            "    '--critic-layernorm --no-actor-layernorm'",
            ")",
            "FQL_FLAGS = (",
            "    '--flow-steps 10 --distill-alpha-bc 1.0 '",
            "    '--teacher-lr 3e-4 --flow-time-embed-dim 32'",
            ")",
            "",
            "TRAIN_SEEDS = [42, 0]",
            "ALGOS = ['rebrac', 'fql']",
            "",
            "CHECKPOINT_ROOT = Path(f'checkpoints/fql_succession/p2/{CELL_ID}')",
            "RESULTS_ROOT    = Path(f'results/fql_succession/p2/{CELL_ID}')",
            "TEST_DIR        = RESULTS_ROOT / 'test'",
            "CURVES_DIR      = RESULTS_ROOT / 'training_curves'",
            "",
            "MIRROR_FILES = ('train_log.jsonl', 'eval_log.csv', 'trainer_state.json', 'train_config.txt')",
            "",
            "def save_dir(algo, seed):   return CHECKPOINT_ROOT / f'{algo}_seed{seed}'",
            "def test_json(algo, seed):  return TEST_DIR / f'{algo}_seed{seed}.json'",
            "def mirror_dir(algo, seed): return CURVES_DIR / f'{algo}_seed{seed}'",
            "",
            "print('CELL_ID         =', CELL_ID)",
            "print('DATASET         =', DATASET)",
            "print('TRAIN_SEEDS     =', TRAIN_SEEDS, '  ALGOS =', ALGOS, '  total =', len(ALGOS)*len(TRAIN_SEEDS))",
            "print('CHECKPOINT_ROOT =', CHECKPOINT_ROOT)",
            "print('RESULTS_ROOT    =', RESULTS_ROOT)",
        ),
    ]


def section_3() -> list[dict]:
    return [
        md(
            "## 3. Pre-flight",
            "",
            "Hard sanity (assert) — fail fast on any missing artifact:",
            "1. Dataset dir + `transitions.npz` + `metadata.json` exist",
            "2. metadata: action_noise_std == 0.0,num_episodes == 1000,"
            "mix_components 含 privileged + goalseek 两 component",
            "3. flow + manifest exist",
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
            "",
            "for k in ['policy', 'action_noise_std', 'num_episodes', 'num_transitions',",
            "          'mix_components', 'mix_strategy']:",
            "    v = meta.get(k)",
            "    print(f'  {k:<22} = {v}')",
            "",
            "# 期望: σ=0.0, 1000 ep total, 2 mix_components (priv + goalseek)",
            "assert abs(float(meta.get('action_noise_std', 1.0))) < 1e-6, (",
            "    f'expected action_noise_std==0.0 but got {meta.get(\"action_noise_std\")} '",
            "    '— E-multi must be CLEAN to isolate modality from noise.'",
            ")",
            "assert int(meta.get('num_episodes', 0)) == 1000, (",
            "    f'expected 1000 episodes, got {meta.get(\"num_episodes\")}'",
            ")",
            "components = meta.get('mix_components', [])",
            "assert len(components) == 2, f'expected 2 mix_components, got {len(components)}'",
            "policies = {c.get('policy') for c in components}",
            "assert policies == {'privileged', 'goalseek'}, (",
            "    f'expected components = {{priv, goalseek}}, got {policies}'",
            ")",
            "",
            "assert Path(FLOW).is_file(), f'flow missing: {FLOW}'",
            "assert Path(MANIFEST).is_file(), f'manifest missing: {MANIFEST}'",
            "",
            "print('\\nPre-flight PASS — ready to train.')",
        ),
    ]


def section_4() -> list[dict]:
    return [
        md(
            "## 4. Train (4 runs: 2 algo × 2 seed)",
            "",
            "Train order: rebrac×{42,0},然后 fql×{42,0}。skip-resume on `agent_final.pt`。",
        ),
        code(
            "import time",
            "from pathlib import Path",
            "",
            "for algo in ALGOS:",
            "    ALGO_FLAGS = REBRAC_FLAGS if algo == 'rebrac' else FQL_FLAGS",
            "    for seed in TRAIN_SEEDS:",
            "        sd = save_dir(algo, seed)",
            "        sd_str = str(sd)",
            "        if (sd / 'agent_final.pt').exists():",
            "            print(f'[skip-train] {algo} seed={seed} (agent_final.pt exists at {sd})')",
            "            continue",
            "        sd.mkdir(parents=True, exist_ok=True)",
            "        t0 = time.time()",
            "        print(f'>>> train {algo} seed={seed} → {sd}')",
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
            "            --eval-episodes {EVAL_EPISODES} \\",
            "            {ALGO_FLAGS} \\",
            "            --skip-final-eval \\",
            "            --seed {seed} \\",
            "            --save-dir '{sd_str}' \\",
            "            --device cuda",
            "        print(f'<<< done {algo} seed={seed} in {(time.time()-t0)/60:.1f} min')",
            "",
            "print('All training done.')",
        ),
    ]


def section_5() -> list[dict]:
    return [
        md(
            "## 5. Test eval (100 ep / seed) + mirror small files",
            "",
            "skip-eval on existing test json;`agent_final.pt` 不在则 warn 跳过。",
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
            "# Mirror small files for downstream plotting (always)",
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
            "## 6. Cell summary + 2×2 matrix verdict",
            "",
            "对照 sprint-0 三 cell test json + 本 E-multi 的 4 个 test json,"
            "render 2×2 matrix:每格填 (rebrac μ, fql μ, δ, verdict)。",
            "",
            "Primary verdict rule (per spec §5.6):",
            "- `|δ| ≤ 0.03` → NULL",
            "- `|δ| ≥ 0.05 AND 方向一致(两 seed 同号)` → POSITIVE/NEGATIVE",
            "- 否则 → GRAY",
        ),
        code(
            "import json",
            "from pathlib import Path",
            "from statistics import mean",
            "",
            "def load_sr(cell, algo, seed):",
            "    p = Path(f'results/fql_succession/p2/{cell}/test/{algo}_seed{seed}.json')",
            "    return float(json.loads(p.read_text())['eval_success_rate']) if p.exists() else None",
            "",
            "CELLS_2x2 = [",
            "    # (label, cell_dir, row='uni'|'multi', col='clean'|'noisy')",
            "    ('E-uni',       'e_uni',       'uni',   'clean'),",
            "    ('M-uni-noise', 'm_uni_noise', 'uni',   'noisy'),",
            "    ('E-multi',     CELL_ID,       'multi', 'clean'),",
            "    ('M-multi-mix', 'm_multi_mix', 'multi', 'noisy'),",
            "]",
            "",
            "print(f'{\"cell\":<14}{\"rebrac μ\":<12}{\"fql μ\":<12}{\"δ=F-R\":<10}{\"verdict\":<10}')",
            "print('-' * 60)",
            "results = {}",
            "for label, cdir, row, col in CELLS_2x2:",
            "    re_vals = [load_sr(cdir, 'rebrac', s) for s in TRAIN_SEEDS]",
            "    fq_vals = [load_sr(cdir, 'fql',    s) for s in TRAIN_SEEDS]",
            "    if any(v is None for v in re_vals + fq_vals):",
            "        print(f'{label:<14}MISSING')",
            "        continue",
            "    rmu = mean(re_vals); fmu = mean(fq_vals); d = fmu - rmu",
            "    per_seed = [fq_vals[i] - re_vals[i] for i in range(2)]",
            "    consistent = (per_seed[0] > 0) == (per_seed[1] > 0)",
            "    ad = abs(d)",
            "    if ad <= 0.03: verdict = 'NULL'",
            "    elif ad >= 0.05 and consistent and d > 0: verdict = 'POSITIVE'",
            "    elif ad >= 0.05 and consistent and d < 0: verdict = 'NEGATIVE'",
            "    elif 0.03 < ad < 0.05: verdict = 'GRAY'",
            "    else: verdict = 'INCONSIST'",
            "    print(f'{label:<14}{rmu:<12.3f}{fmu:<12.3f}{d:<+10.3f}{verdict:<10}')",
            "    results[label] = (rmu, fmu, d, verdict, row, col)",
            "",
            "print()",
            "print('=== 2×2 matrix (rows = modality, cols = noise) ===')",
            "print()",
            "print(f'{\"\":<10}{\"clean (σ≈0)\":<22}{\"noisy (σ≥0.5)\":<22}')",
            "for row in ('uni', 'multi'):",
            "    cells = {col: None for col in ('clean', 'noisy')}",
            "    for label, vals in results.items():",
            "        _, _, _, _, r, c = vals",
            "        if r == row:",
            "            cells[c] = (label, vals[3])  # (label, verdict)",
            "    def fmt(t):",
            "        return f'{t[0]} → {t[1]}' if t else 'MISSING'",
            "    print(f'{row:<10}{fmt(cells[\"clean\"]):<22}{fmt(cells[\"noisy\"]):<22}')",
        ),
    ]


def section_7() -> list[dict]:
    return [
        md(
            "## 7. Final iff verdict (revised, post-diagnostic)",
            "",
            "Revised iff: **FQL > ReBRAC iff dataset BC anchor noise-corrupted**.",
            "",
            "Decision rule:",
            "- `E-multi == NULL` AND `M-uni-noise == POSITIVE` → **REVISED IFF CONFIRMED**",
            "  (noise IS the discriminator, modality alone is not).",
            "- `E-multi == POSITIVE` → modality also contributes (weaker conclusion).",
            "- `E-multi` unexpected direction → expand to n=4 seeds before drawing conclusions.",
        ),
        code(
            "# Re-use `results` dict from §6",
            "v_eu  = results.get('E-uni',       (None,)*4)[3] if 'E-uni'       in results else 'MISSING'",
            "v_mun = results.get('M-uni-noise', (None,)*4)[3] if 'M-uni-noise' in results else 'MISSING'",
            "v_em  = results.get('E-multi',     (None,)*4)[3] if 'E-multi'     in results else 'MISSING'",
            "v_mmm = results.get('M-multi-mix', (None,)*4)[3] if 'M-multi-mix' in results else 'MISSING'",
            "",
            "print(f'E-uni        verdict = {v_eu}')",
            "print(f'M-uni-noise  verdict = {v_mun}')",
            "print(f'E-multi      verdict = {v_em}     ← N1 this notebook')",
            "print(f'M-multi-mix  verdict = {v_mmm}')",
            "print()",
            "",
            "if v_em == 'NULL' and v_mun == 'POSITIVE':",
            "    print('VERDICT: REVISED IFF CONFIRMED — noise IS the discriminator.')",
            "    print('  paper §results can state the noise-axis claim in causal-isolation form.')",
            "elif v_em == 'POSITIVE' and v_mun == 'POSITIVE':",
            "    print('VERDICT: PARTIAL — modality also contributes; iff weakens to a disjunction.')",
            "    print('  Consider expanding n_seeds to 4 to nail down which condition dominates.')",
            "elif v_em != 'NULL' and v_em != 'POSITIVE':",
            "    print(f'VERDICT: UNCLEAR — E-multi verdict = {v_em}.')",
            "    print('  Run additional seeds [1, 7] before drawing conclusions.')",
            "else:",
            "    print('VERDICT: unexpected pattern; review per-seed numbers.')",
        ),
    ]


def section_8() -> dict:
    return md(
        "## 8. Report checklist",
        "",
        "回到本地 sync 步骤:",
        "- [ ] sync `results/fql_succession/p2/e_multi/test/*.json` (4 个)回 local repo",
        "- [ ] sync `results/fql_succession/p2/e_multi/training_curves/*/` (4 个 mirror 目录)",
        "- [ ] 重跑 `python -m scripts._plot_fql_succession_p2_diagnostic` "
        "  (需先扩它支持 4 cell;或在 verdict notebook 里独立画 2×2 panel)",
        "- [ ] 回写 `docs/fql_succession_p2_mechanism_diagnostic.md` §7 N1 完成状态",
        "",
        "Verdict path-dependent next:",
        "- **REVISED IFF CONFIRMED**(E-multi=NULL):写 `docs/fql_succession_p2_results.md` "
        "主报告 + spec v1.3 → v1.4 改写;启动 N2(n=4 seeds)。",
        "- **PARTIAL**(E-multi=POSITIVE):iff 弱化为 OR;考虑加 N3 = `M-uni-noise + reduced σ` "
        "扫描 noise threshold。",
        "- **UNCLEAR**:先加 seed [1, 7],n=4 后再判。",
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
