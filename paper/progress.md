# Paper Writing Progress & Plan

> Title: **Deployable-only Offline RL Closes the Privileged-Critic Gap on Underwater Wake Navigation**
> Algorithm: ReBRAC-Q (Q-normalized dual-penalty TD3+BC variant)
> Layout: 单栏 11pt（thesis / arXiv preprint 风格）
> 主算法别名：ReBRAC-Q（actor 端保留 TD3+BC Q-normalization；critic 端追加 dual-penalty + LayerNorm）
>
> **Last updated:** 2026-05-02（Phase 4 Appendix 收口）
> **Last commit:** `2830589` — Phase 4.0+4.1 完成（Fig 1 + 占位 cite 全替换 + 进度追踪文档）；**Phase 4 (Appendix A–G) 待 commit**
> **Branch:** main, +26 ahead of `origin/main`（Appendix 改动尚未 commit）
> **Build status:** `latexmk -pdf -xelatex`，exit=0，0 undefined refs，0 cite warnings，0 missing chars，**12 cites parsed**（无 placeholder），bibtex `warning$ -- 0`
> **Compiled output:** `main.pdf`，**29 页**，588 KB（含 Fig 1 + Appendix A–G）

---

## 1. 总览（Phase 进度条）

| Phase | 范围 | 状态 | Commit |
|---|---|---|---|
| Phase 0 | outline + writing index 起草 | ✅ done | `cf775c8` 之前 |
| Phase 1 | §4 Method + §5 Experiments rev.2 严审收口 | ✅ done | `4572f03` |
| Phase 2 | Figure 2 (seed dotplot) + Figure 3 (Q drift) + refs.bib 雏形 | ✅ done | `4572f03` |
| Phase 3 | 全 8 节 + Abstract rev.1，单栏改造，编译干净 | ✅ done | `eca5af9` |
| Phase 4.0 | Figure 1 sensor schematic（matplotlib 脚本 + setup.tex callout） | ✅ done | `2830589` |
| Phase 4.1 | 占位 cite 替换：`td3bc_phase0c_report` → inline supplementary；`underwater_rl_placeholder` → 4 篇 concrete refs（Carlucho2018, Yu2017, Verma2018pnas, Gunnarson2021ncomm） | ✅ done | `2830589` |
| Phase 4 (Appendix) | Appendix A–G（reproducibility / hyperparams / full tables / stats / derivations / curves / obs spec） | ✅ done | uncommitted |
| **Phase 5** | **reviewer stress-test（outline §6 D7）+ 全文 polish** | ⏳ **TODO（next）** | — |
| Phase 6 | 投稿格式打包（CoRL / RA-L 双栏切换 + supplementary 拆分） | ⏳ TODO | — |

---

## 2. 章节级完成度

| § | 文件 | 行数 | rev | 状态 | 关键 anchor |
|---|---|---:|---|---|---|
| Abstract | `main.tex` | — | rev.1 | ✅ | — |
| §1 Introduction | `sections/intro.tex` | 40 | rev.1 | ✅ | `sec:intro` + C1–C4 contributions |
| §2 Related Work | `sections/related_work.tex` | 25 | rev.1 | ✅ | `sec:related_work` |
| §3 Problem Setup | `sections/setup.tex` | 99 | rev.2 | ✅ | `sec:setup`, `subsec:setup_task`, `subsec:setup_obs_deployable`, `subsec:setup_obs_privileged`, `subsec:setup_reward`, `subsec:setup_datasets`, `tab:dataset_matrix`, `fig:sensor_schematic`, `eq:setup_priv_obs`, `eq:setup_reward` |
| §4 Method | `sections/method.tex` | 134 | rev.2 | ✅ | `sec:method`, `subsec:method_actor_loss`, `subsec:method_critic_loss`, `subsec:method_diff` |
| §5 Experiments | `sections/experiments.tex` | 233 | rev.2 | ✅ | `sec:experiments`, `subsec:main_results`, `subsec:ablations`, `subsubsec:finding_1..4`, `tab:main_table`, `tab:beta2_ablation`, `tab:ln_ablation`, `fig:seed_dotplot`, `fig:q_drift` |
| §6 Mechanistic Discussion | `sections/discussion.tex` | 41 | rev.1 | ✅ | `sec:discussion` + 5 subsec |
| §7 Limitations | `sections/limitations.tex` | 38 | rev.1 | ✅ | `sec:limitations`（L1–L8） |
| §8 Conclusion | `sections/conclusion.tex` | 15 | rev.1 | ✅ | `sec:conclusion` |

**Body 总行数：616**（不含 main.tex 框架、refs.bib、appendix.tex）

**Appendix（rev.1，2026-05-02）：**

| App | 文件 | 范围 | 关键 anchor |
|---|---|---|---|
| A | `sections/appendix.tex` §A | Reproducibility（notebooks / data manifests / seeds / commit hashes / compute env） | `app:repro`, `tab:repro_notebooks`, `tab:repro_data` |
| B | `sections/appendix.tex` §B | Hyperparameters（ReBRAC-Q winner config + TD3+BC α-sweep + ReBRAC-Q sweep grid） | `app:hyperparams`, `tab:hyperparams_rebrac` |
| C | `sections/appendix.tex` §C | Full per-seed results（main matrix + β₂=0 + LN-off） | `app:full_tables`, `tab:full_main`, `tab:full_beta2`, `tab:full_ln` |
| D | `sections/appendix.tex` §D | Statistical tests（Welch's $t$ + paired bootstrap pseudocode + Cohen's $d$ + multiple-comparisons） | `app:stats`, `eq:welch_t`, `eq:welch_df`, `eq:cohens_d` |
| E | `sections/appendix.tex` §E | Method derivations（β₁↔α 等价 + Q-norm 数值稳定性 + LN placement） | `app:derivations` |
| F | `sections/appendix.tex` §F | Training curves（critic/actor loss + $\hat{Q}_{\text{target}}$ trajectory + LN-off failure mode 描述） | `app:curves` |
| G | `sections/appendix.tex` §G | Sensor / observation full spec（s0/s1/s2 + 10-D obs + history stacking + privileged $\mathbf{o}^{\text{priv}}$） | `app:obs_spec`, `tab:obs_probe_layout`, `tab:obs_channels_full` |

---

## 3. 4 个 paper-level findings（核心数字锚点）

> 这些数字是全文的"硬骨头"。任何 abstract / intro / experiments / discussion 中的数字必须与 `tab:main_table / tab:beta2_ablation / tab:ln_ablation` 一致；改动一处必须全文同步。

| ID | Finding | 核心数字 | 主表 | 引用位置 |
|---|---|---|---|---|
| **C1** | Anti-scaling reversal & cross-dataset uplift | TD3+BC `0.672 → 0.596` (-7.6pp on cross 1000→2000)；ReBRAC-Q `+23.0 / +32.2pp` over TD3+BC on cross-1000 / cross-2000 | `tab:main_table` | abstract, intro C1, exp §5.4.1, disc §6.5 |
| **C2** | Deployable matches privileged-critic | dep ReBRAC-Q `0.928 ± 0.086` vs priv TD3+BC `0.922 ± 0.096`；Welch's `t=0.104, p=0.9195`；paired-bootstrap 95% CI `[-3.0pp, +4.2pp]`；seed 44 `0.78 → 0.90 (+12pp)` 救援 | `tab:main_table` + `fig:seed_dotplot` | abstract, intro C2, exp §5.4.2, disc §6.4 |
| **C3** | Dual penalty dataset-invariant | β₂=0 让 mean 仅退化 `-1.8pp / -2.4pp`，但 $\hat{Q}$ 漂移方向\textbf{相同} `+46% / +98%`（worldcomp Q≈+15、crosscomp Q≈-8 起点反号） | `tab:beta2_ablation` + `fig:q_drift` | abstract, intro C3, exp §5.4.3, disc §6.1, §6.2 |
| **C4** | LN ⊥ dual penalty | LN-off 退化 `-16.2pp` + std blow-up `~11×` + Q 朝\textbf{相反}方向漂移；β₂=0 朝更正方向 | `tab:ln_ablation` | abstract, intro C4, exp §5.4.4, disc §6.3 |

**统一统计约定：**
- `std`：sample std (`ddof=1`，Bessel correction)
- `Welch's t-test`：unequal variance, `t=0.104, p=0.9195`
- `paired bootstrap`：episode-level, B=10000, 95% CI

---

## 4. Figures 进度

| ID | 标题 | 脚本 | 输出 | 状态 |
|---|---|---|---|---|
| Fig 1 | Sensor schematic — (a) Kármán wake + REMUS-100 cross-stream task；(b) body-frame s0 单点 vs hull-integral 5 点抽样几何（2×1 vertical layout） | `figures/scripts/fig1_sensor_schematic.py` | ✅ `figures/output/fig1_sensor_schematic.{pdf,png}` | ✅ done |
| Fig 2 | Per-seed dot plot for finding C2（dep vs priv） | `figures/scripts/fig2_seed_dotplot.py` | ✅ `figures/output/fig2_seed_dotplot.{pdf,png}` | ✅ done |
| Fig 3 | $\hat{Q}$ drift bar chart for C3（β₂=0 vs β₂=2，两 dataset） | `figures/scripts/fig3_q_drift.py` | ✅ `figures/output/fig3_q_drift.{pdf,png}` | ✅ done |

**Fig 1 实现要点（已完成）：**
- 上 panel (a)：D2Q9 TRT-LBM Kármán 涡街尾迹场涡量场底图（取 frame 600，`wake_v8_U1p00_Re150_*.npy`）+ REMUS-100 body schematic（heading 74° 朝 cross-stream goal）+ 绿色五角星 goal 在 +y 方向；左上角标注 `U_∞ = 1.0 m/s` + 自由来流箭头
- 下 panel (b)：体坐标系 sensor 抽样几何 — body 椭圆 + 蓝色 s0 单点（actor 在两种协议下都读）+ 红色菱形 5 点抽样（`ξᵢ ∈ {-0.4,-0.2,0,+0.2,+0.4}·L`，仅 privileged-critic 协议下 critic 读取）+ bracket + `o^priv = [u_eq, v_eq]` caption
- Layout：2×1 vertical stack，`figsize=(8.0, 8.0)`，`height_ratios=[1.0, 1.25]`；panel (a) 用 `aspect=equal` 保持物理比例，panel (b) 不锁 aspect 让示意图垂直呼吸

---

## 5. References 进度

`paper/refs.bib`：**12 entries，0 placeholder**

| Cite key | Type | Status |
|---|---|---|
| `fossen2011handbook` | book | ✅ |
| `pinto2017asymmetric` | inproc (RSS 2018) | ✅ |
| `fujimoto2021td3bc` | inproc (NeurIPS 2021) | ✅ |
| `tarasov2023rebrac` | inproc (NeurIPS 2023) | ✅ |
| `fujimoto2019bcq` | inproc (ICML 2019) | ✅ |
| `kumar2020cql` | inproc (NeurIPS 2020) | ✅ |
| `kostrikov2022iql` | inproc (ICLR 2022) | ✅ |
| `nair2020awac` | misc (arXiv) | ✅ |
| `carlucho2018ras` | journal (RAS 2018) | ✅ added rev.4 — fluid-aware AUV control |
| `yu2017auv` | inproc (CCC 2017) | ✅ added rev.4 — AUV trajectory tracking |
| `verma2018pnas` | journal (PNAS 2018) | ✅ added rev.4 — vortex collective swimming |
| `gunnarson2021ncomm` | journal (Nature Comm 2021) | ✅ added rev.4 — vortical flow navigation |

**已移除 (rev.3 → rev.4)：**
- `td3bc_phase0c_report`：experiments.tex:26 改为 inline "supplementary materials" 引用（不再走 cite key）
- `underwater_rl_placeholder`：related_work.tex 改为 4 篇 concrete refs，按 fluid-aware vs wake-aware 子方向精确分配

---

## 6. 下一步计划（Phase 4，priority-ordered）

### ~~P0 — Figure 1 sensor schematic~~（✅ done）
- ✅ 写好 `paper/figures/scripts/fig1_sensor_schematic.py`（rev.2，2×1 vertical layout）
- ✅ 从 `wake_data/wake_v8_U1p00_Re150_*.npy` 取 frame 600 涡量场底图
- ✅ 输出 `fig1_sensor_schematic.{pdf,png}`
- ✅ `setup.tex` 在 §3.3 末尾插入 figure 块 `\label{fig:sensor_schematic}`，正文 forward-ref `图~\ref{fig:sensor_schematic}`
- ✅ main.pdf 21 页编译干净

### ~~P1 — 替换 2 个 placeholder cite~~（✅ done）
- ✅ `td3bc_phase0c_report` → experiments.tex:26 inline 改为 "来自 privileged-critic 协议下的 α-sweep；reproducibility artifacts 与完整 sweep 表见 supplementary materials"
- ✅ `underwater_rl_placeholder` → related_work.tex 改为 4 篇 concrete refs：
  - **fluid-aware control**：Carlucho et al. 2018 RAS, Yu et al. 2017 CCC
  - **wake-aware navigation**：Verma et al. 2018 PNAS, Gunnarson et al. 2021 Nature Comm
- ✅ refs.bib 删除 2 placeholder + 添加 4 entries（rev.3 → rev.4）
- ✅ main.pdf 21 页编译干净（12 cites parsed，0 placeholder mention）

### ~~P2 — Appendix A1–A7~~（✅ done，rev.1，2026-05-02）
- ✅ A1 Reproducibility（notebook 列表 + offline data manifest + seed 协议 + commit hash 表 + compute env）
- ✅ A2 Hyperparameters（ReBRAC-Q winner config + TD3+BC α-sweep + ReBRAC-Q sweep grid）
- ✅ A3 Full results tables（main matrix per-seed + β₂=0 ablation per-seed + LN-off probe per-seed）
- ✅ A4 Statistical test details（Welch's $t$-test \eqref{eq:welch_t}–\eqref{eq:welch_df} + paired bootstrap pseudocode + Cohen's $d$ \eqref{eq:cohens_d} + multiple comparisons）
- ✅ A5 Method derivations（$\beta_1 \leftrightarrow \alpha_{\text{TD3+BC}}$ 等价推导 + Q-norm $\varepsilon$ 数值稳定性 + critic LN forward graph 位置）
- ✅ A6 Training curves（critic loss / actor loss two-term split / $\hat{Q}_{\text{target}}$ trajectory cross-dataset / LN-off seed-42 spike 描述；原始 CSV forward 至 supplementary）
- ✅ A7 Sensor / observation full spec（s0/s1/s2 probe layout 真实 sensor 映射 + 10-D obs channel 完整定义 + history stacking + privileged $\mathbf{o}^{\text{priv}}$ 实现注）
- ✅ main.tex 接入 `\input{sections/appendix}`，编译干净 29 页 588 KB
- ✅ 中间文件已 latexmk -c 清理

### P3 — Reviewer stress-test（outline §6 D7）
- 用 `docs/rebrac_mainline_review.md` §2.2 / §4 反向压力测试全文
- 检查每个 finding 是否有 ≥1 处可被审稿人 imagined 但未做的对照实验
- 任何缺口要么补（升级到 Phase 4.5），要么写进 §7 Limitations 明示边界

### P4 — 投稿格式打包
- `\documentclass[10pt, twocolumn]{article}` + `[margin=0.85in]`
- `figure*` / `table*` 切换（如有跨栏需要）
- Supplementary 拆分（Appendix → 独立 `supplementary.tex`）
- Anonymization 检查（`Anonymous Authors` 占位已 OK）

---

## 7. 一致性 checklist（防止数字 / 命名漂移）

每次改动后 run-through：

- [ ] **数字一致性**：abstract / intro / experiments / discussion 的所有 success rate / std / pp 数字与 `tab:main_table` 等表格一致
- [ ] **算法命名**：全文统一使用 `ReBRAC-Q`（Q-normalized dual-penalty TD3+BC variant），不混用 "ReBRAC" / "TD3+BC w/ critic penalty" / "our method" 等同义词
- [ ] **dataset 命名**：`worldcomp-1000` / `crosscomp-1000` / `crosscomp-2000`（不是 `world-1000` / `cross-1000`）
- [ ] **协议命名**：`deployable-only` / `privileged-critic`（不是 `dep` / `priv` 在正文中）；表 / 图标签可用缩写
- [ ] **probe layout**：`s0` / `s1` / `s2` 全文一致；`s0` = "single-point DVL water-track"（不是 "1-probe"）
- [ ] **统计约定**：`sample std (ddof=1)` 全文统一；`Welch's t-test, p=0.92` 不要写成 `p=0.9195`（abstract 简化）or 反之
- [ ] **forward-ref 全闭合**：`grep -c 'Reference .* on input line' main.log` 应为 0
- [ ] **missing chars**：`grep 'Missing character' main.log` 应为空（圆圈数字 ①②③④ 已消解）
- [ ] **bibtex warnings**：`main.blg` 末尾应为 `(There were 0 ... warnings)` 或仅 placeholder cite 警告

---

## 8. 编译指令

```bash
# 标准编译流程（cwd=repo root）
cd paper
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

# 或用 latexmk
cd paper && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex

# 清理 build artifacts（已被 .gitignore 排除）
cd paper && latexmk -C
```

**当前 build artifacts**（gitignored）：
- `main.aux / main.bbl / main.blg / main.log / main.out / main.toc / main.fls / main.fdb_latexmk`
- `figures/output/.DS_Store`

---

## 9. 写作纪律（出自 outline §8 R*）

每次新增 / 修订段落前 self-check：

- **R1**：不写抽象 future work 套话；每条 limitation / claim 锚定具体 ref / 表
- **R2**：每段 3–4 句封顶（discussion / related_work），3 段封顶（intro 各部分）
- **R3**：不重复 §5 已报数字到 §6 / §1；§6 写 implication，§1 写 contribution
- **R4**：mechanism > storytelling — 任何 "we believe / it suggests" 必有数字支撑
- **R5**：Qnorm / dual-penalty / LN 三件套不能混淆角色（actor-side / critic-side / representation-level）
- **R6**：seed 44 是 cross-(protocol, β₂, dataset) 闭环证据，不是 outlier 噪音
- **R7**：anti-scaling reversal 仅在本环境 / 本 sensor 协议 scope 内成立，不包装为 algorithm-paper claim
- **R8**：sim2real implication 严格限定在 `s0 + Kármán wake + REMUS-100` 三条边界内
- **R9**：每节有 forward-ref（`\ref{sec:X}`）必须在对应文件 `\label` 闭合
- **R10**：投稿前 run-through §7 一致性 checklist 全表

---

## 10. 文件树快照

```
paper/
├── main.tex                       # rev.4 — 单栏 11pt 框架 + Abstract + \input{appendix}
├── refs.bib                       # 12 cites（0 placeholder）
├── outline.md                     # Phase 0 outline 主文档
├── progress.md                    # ← 本文档
├── main.pdf                       # 29pp 588KB（最新 build artifact，含 Appendix A–G）
├── sections/
│   ├── intro.tex                  # §1, rev.1
│   ├── related_work.tex           # §2, rev.1
│   ├── setup.tex                  # §3, rev.2（含 Fig 1 callout）
│   ├── method.tex                 # §4, rev.2
│   ├── experiments.tex            # §5, rev.2
│   ├── discussion.tex             # §6, rev.1
│   ├── limitations.tex            # §7, rev.1
│   ├── conclusion.tex             # §8, rev.1
│   └── appendix.tex               # Appendix A–G, rev.1（NEW 2026-05-02）
└── figures/
    ├── output/
    │   ├── fig1_sensor_schematic.{pdf,png}  ✅
    │   ├── fig2_seed_dotplot.{pdf,png}      ✅
    │   └── fig3_q_drift.{pdf,png}            ✅
    └── scripts/
        ├── fig1_sensor_schematic.py          ✅
        ├── fig2_seed_dotplot.py              ✅
        └── fig3_q_drift.py                    ✅
```
