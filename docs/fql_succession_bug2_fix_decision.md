# Bug 2 修复方向决策 — `--episodes` vs manifest size

> **状态**：DECIDED — **方案 (a) 大 manifest** 落地;(b) CLI honor 作为 follow-up cleanup,**本 session 不做**
> **背景**：[`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) §6.2、[`fql_succession_gate_b_interim_report.md`](fql_succession_gate_b_interim_report.md) §5.2
> **目的**：为 P2 main comparison 解锁充足 eval episode 数,降低 c4 slope 噪声底
> **作者**：Claude Code session,2026-05-20
> **Commit gate**: P2 pre-requisite 1/2

---

## 1. Bug 复述

CLI `--episodes 100` (`evaluate_offline.py`) 和 `--eval-episodes 100` (`train_offline.py`) 在 `--manifest` 给定时**被静默 override 至 manifest size**。

**根因定位**:[`scripts/train_utils.py:185-213`](../scripts/train_utils.py:185) `_resolved_eval_episodes()`

```python
def _resolved_eval_episodes(*, reset_options, seed, num_episodes, benchmark_manifest):
    if benchmark_manifest is None:
        return [BenchmarkEpisode(...) for idx in range(max(0, int(num_episodes)))]
    episodes = []
    for spec in benchmark_manifest.episodes:   # ← 完全忽略 num_episodes
        episodes.append(BenchmarkEpisode(...))
    return episodes
```

`single_u10_cross_tgt15.json` 当前是 30 ep → 所有 in-training eval + final test 都是 30 ep,与 spec / CLI 声明的 100 ep 不符,造成 c4 slope SE 比预期大 1.8×(§3 §5)。

---

## 2. 选项对比

| 选项 | 操作 | 代码改动 | 影响面 | 噪声减少 |
|---|---|---|---|---|
| **(a) 大 manifest** | 生成 `single_u10_cross_tgt15_ep100.json` | `generate_standard_benchmarks.py` 加 `--output-name` flag (~5 行) | 0 现存 manifest 调用方 | per-eval SE 0.0837 → 0.0459(halves) |
| **(b) CLI honor `--episodes`** | 改 `_resolved_eval_episodes()` 让 CLI 显式给定时覆盖 manifest size | `train_utils.py` + audit `train_offline.py` / `evaluate_offline.py` 优先级 | 全部 30+ 现存调用方都要 audit | 同上 |

### 2.1 (a) 的优势(选定理由)

1. **不动现有调用方** — broad val v2 / arrival_v2 / paper 1 ReBRAC mainline / td3bc phase0c / online sac 等 30+ 处对 30-ep manifest 的引用全部保持不变,paper 1 历史可重现性零冲击
2. **manifest 语义干净** — manifest 的核心契约 = 固定 N 个固定 seed 的 episode。`_ep100` 是新 manifest 文件,语义清晰
3. **唯一代码改动** ≈ 5 行 — `generate_standard_benchmarks.py` 加 `--output-name` 可选 flag(default None → 保持原 `{key}.json` 行为)
4. **byte-identical extension** — `manifest_seed=1250 + idx` 保证前 30 个 episode 与现 30-ep manifest **完全一致**;`_ep100` 严格扩展现版本
5. **side benefit** — 100-ep manifest 也供 P3 mix ratio ablation / paper 2 后续 cell 复用

### 2.2 (b) 的劣势

1. **manifest 语义模糊化** — 同一份 manifest 在 N=30 / N=100 / N=200 下被截断,「用同一 benchmark」的可比性下降。manifest 的 N 不再是 manifest 固有属性
2. **代码改动面大,audit 负担重** — `_resolved_eval_episodes` 的语义改变,影响 `train_offline.py` 的 in-training eval、`evaluate_offline.py` 的 final test、`train_utils.py` 的 parallel eval workers,共 3 处需要重新走 truncate 逻辑
3. **`num_episodes > manifest_size` 的语义未定** — 报错?重复 episode?用 manifest_seed + episodes_so_far procedurally extend?都是需要新设计的选项,与 manifest 「固定 N seed」的可重现性契约冲突
4. **现存 30+ 调用方都要回归测试** — 任何依赖 train_offline.py default `eval_episodes=30` 同时传 manifest 的脚本,在 (b) 下默认行为可能改变(取决于实现选择)
5. **不是 P2 阻塞** — c4 噪声底问题用 (a) 已能完全解决;(b) 是「让 CLI 更通用」的工程清洁化,可作为 follow-up

### 2.3 (b) 暂缓但未来仍可能需要

如果未来出现需要「同一 manifest 在不同 cell 用不同 N」的研究场景,(b) 的子集 **(b') truncate-only** 是更小风险的实现:

- 仅在 `num_episodes < manifest_size` 时 truncate to 前 num_episodes
- 在 `num_episodes >= manifest_size` 时仍用全部 manifest 内容(向后兼容 default 行为)

(b') 改动比 (b) 小 ~一半;作为 follow-up 跟踪 issue,**本 session 不做**。

---

## 3. 决定:落地方案 (a)

### 3.1 代码改动 — `scripts/generate_standard_benchmarks.py`

加 `--output-name` 可选参数(默认 None → 保持现行 `{key}.json` 行为):

```python
parser.add_argument(
    "--output-name",
    type=str,
    default=None,
    help="Optional custom basename (without .json) for the output manifest. "
         "Defaults to '{benchmark_key}.json'. Use to generate variant manifests "
         "(e.g. '--output-name single_u10_cross_tgt15_ep100').",
)
```

`_build_manifest_for_benchmark()` 接收 `output_name` 并按 `f"{output_name}.json"` 或 default 行为决定输出路径。

### 3.2 Manifest 生成

```bash
python -m scripts.generate_standard_benchmarks \
    --benchmarks single_u10_cross_tgt15 \
    --episodes 100 \
    --output-name single_u10_cross_tgt15_ep100
```

预期产物:`benchmarks/single_u10_cross_tgt15_ep100.json`(100 episodes,seed=1250..1349)。

### 3.3 验证步骤

1. 加载 ep100 manifest 验证 `len(episodes) == 100`
2. 前 30 个 episode 与 `single_u10_cross_tgt15.json` byte-identical(reset_options / seed 字段)
3. 加载到 `BenchmarkManifest` 对象成功
4. (可选) 调 `_resolved_eval_episodes` 验证返回 100 个 episode

### 3.4 噪声降低预期

| 量 | 30-ep manifest | 100-ep manifest | 改善 |
|---|---:|---:|---:|
| per-eval SE @ p=0.7 | 0.0837 | 0.0458 | 1.83× |
| slope SE per-seed (n_pts=6) | 0.0200 | 0.0110 | 1.83× |
| slope SE aggregated n_seeds=5 | 0.00894 | 0.00489 | 1.83× |

在 n_seeds=5 + 100-ep eval 下,c4 slope SE 降至 ~0.005,即使保留「slope ≥ 0」严格阈值,信噪比也比 Gate B Option B (n_seeds=2 + 30 ep) 提升 ~2.9×。但 c4 阈值仍建议改革(见 c4 阈值 design memo)。

---

## 4. 不在本 commit 做的

- ❌ 修改 `_resolved_eval_episodes()` 优先级逻辑(选项 b/b')
- ❌ 更新 `fql_succession_p0p1_spec.md` §5 引用的 manifest 名(留 Session A 在 P2 spec 时统一)
- ❌ 在 Colab / Drive 上重跑 Gate B(verdict 已定)
- ❌ Audit 现有 30-ep manifest 调用方(本方案保持原 manifest 不变 → 不需要)
- ❌ 删除现存 `single_u10_cross_tgt15.json`

---

## 5. Acceptance

- [x] Bug 2 根因定位到 `train_utils.py:185` `_resolved_eval_episodes()`
- [x] (a) vs (b) 决策完成,(a) 选定
- [x] `generate_standard_benchmarks.py` 加 `--output-name` 落地
- [x] `benchmarks/single_u10_cross_tgt15_ep100.json` 生成 + 验证
  - 100 episodes,前 30 与 `single_u10_cross_tgt15.json` byte-identical
  - metadata (flow_path / benchmark_id / factor_values / base_reset_options) 完全一致
  - `_resolved_eval_episodes(num_episodes=100, manifest=ep100)` 返回 100 episodes ✓
  - 30-ep manifest 行为未变(向后兼容) ✓
- [ ] Commit "P2 pre-requisite 1/2: Bug 2 fix via larger manifest"

---

*Decision recorded: 2026-05-20. (b)/(b') 不动 train_utils.py 优先级逻辑,留作 follow-up cleanup,未来若需可独立 commit。*
