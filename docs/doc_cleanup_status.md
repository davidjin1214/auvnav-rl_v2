# 文档整理 — 状态与交接

> 🔧 **进行中（2026-08-17）** — 本文件是「全仓文档整理」这条线的**唯一状态账本**，
> 供跨会话接手。三层做完即可删除本文件。
> 状态只写在这里，不要复制进 `CLAUDE.md`、线总览或 memory。

## 0. 动手之前：先确认你在哪个副本

**本机（Windows 副机）绝对路径，仅本文件此处出现：**

- 工作副本 = `C:\Code\rl_v2` ← **一切改动都在这里**
- 退役中的旧副本 = `C:\Users\jinxiang\OneDrive\我的\Code\new_off_rl\rl_v2`

三个必踩的坑：

1. **会话默认 cwd 可能仍指向 OneDrive 旧副本。** 旧副本的 `CLAUDE.md` 是迁移前的版本，
   `experiments/` 已删、10 个提交也不在那里。在旧副本上跑校验会得到假结果——
   `check_doc_pointers` 在旧副本报**真失效 150**，在工作副本报 **67**，差的 83 条全是
   指向已删 `experiments/` 的路径。**每条命令都显式带上工作副本路径。**
2. **git 用 `git -C "<工作副本>" …`。** `cd <路径> && git …` 这种复合形式会被权限层拒绝
   （推送时连拒两次，换成 `git -C` 一次就过）。
3. **五个数据目录是 junction**（`results/ experiments/ checkpoints/ wake_data/ offline_data/`
   实体在 D 盘）。它们不随 `git clone` 到位；`benchmarks/` 相反，是被 git 追踪的，**绝不能** junction。

## 1. 这条线分三层

| 层 | 内容 | 状态 |
|---|---|---|
| ② | 两份 line summary 的**真值核对**——逐条对照源文档、`results/`、CSV、`metadata.json` | ✅ 已完成 |
| ③ | 文档**归位与角色标注** | ✅ 已完成 |
| ① | 25 篇**论文承重文档**深核 | ⬜ **下一步** |
| 附 | 67 处**真失效指针** | ⬜ 未做 |

层号是当初提出的顺序，执行顺序是 ② → ③ → ①：先确认索引本身可信，再动它索引的东西。

## 2. 已完成的部分（提交号）

| 提交 | 内容 |
|---|---|
| `a68b4c2` | `experiments/auvhamnode_spike/` → `docs/auvhamnode_spike/` |
| `167fa5a` | 第 5 章 21 个过程文档 → `paper/thesis_ch5/notes/` |
| `e6b729d` | 已撤销的 standalone ReBRAC 论文 → `paper/archive/rebrac_standalone/` |
| `81ddc4b` | 新增 [`DOC_INDEX.md`](DOC_INDEX.md)（自动生成的全仓文件地图） |
| `a6decd2` | ② 层订正：两份 line summary 七处失效断言 |
| `bd60aca` | 把 2026-07-12 的三种子撤销传播回 `docs/`——另有五份仍在引用作废数字 |
| `dc64520` | 数据外置约定 + 「读数在 `results/` 不在 `experiments/`」 |
| `68ee9b9` | `build_doc_index.py` 不再走进数据目录与工具缓存 |
| `5d2f727` | `CLAUDE.md` 数据完整性行订正（第 ⑤ 条已核实成立，不再是 open） |
| `480d834` | ③ 层：FQL 线 7 份 P2 之前的施工记录归位 `docs/archive/fql_succession/` |

全部已推送到 `origin/codex-arrival-v2-prototype`。

## 3. ② 层的方法学结论 —— ① 层直接拿来用

**最关键的一条：有一类缺陷能通过全部机械检查。** 链接解析得开、rev 号对得上、§ 号存在、
commit 都在——但文档在复述一个**源文档已经撤销的结论**。只有把总览和源文档并排读才看得出来。
② 层最严重的一处（A2）就是这样：总览把广验 v2 的 Δ 标成「paired same-direction」，
而 report 明写 per-seed 方向已不一致（−3.5 / −6.9 / **+3.1** pp）、首轮读法**已撤销**。

**腐烂模式高度集中：源文档后来做了 supplement，引用方没跟。** 落点全在 2026-05 之后
被追加过的三个条目：

- **v2 broad validation supplement，2026-07-12**（2 seed → 3 seed {42, 0, 43}）
- **asym-critic addendum，2026-05-27**（`ACTOR_FUNDAMENTAL_CONFIRMED`）
- **data_integrity 第 ⑤ 条，2026-08-16**

**→ ① 层的优先级规则：不要 25 篇从头读到尾，先查引用了上述三个源文档的那些。**
② 层追查 A1/A2 时发现，那次撤销**只传播到了论文侧**（`boundary.tex`、
`thesis_chapter_outline.md`、`status.md` 都在 2026-07-12 改了），`docs/` 侧一份未改，
全仓扫描又揪出五份仍在引用作废数字。同一个模式很可能还在别处。

**数字基座本身经得起核**，所以是订正不是重建：A0 六个数字逐位吻合 `ablation_summary.csv`；
SAC 四档八个数字逐位吻合 `metadata.json`；36 个 h2h `test_result.json` 实测吻合；
19 个 commit hash 全部存在且在当前分支。

## 4. ① 层怎么做

对象是 `paper/` 侧有引用的 25 篇文档（论文承重）。做法沿用 ② 层：

1. 先列出每篇的**源文档**（它的数字/结论从哪来）
2. 查源文档在该篇最后更新之后是否被追加过（git log + 头注日期）
3. 被追加过的，并排读，核数字与**解读**——解读被撤销是机械检查抓不到的那一类
4. 论文侧同步核一遍：`paper/thesis_ch5/sections/*.tex` 引的是不是同一个值

**改法的两条硬规矩**（② 层已经踩定）：

- **预登记的阈值、SUPERSEDED 文档里的当时对照值：加追注，不改值。** 改值等于篡改判据。
  已按此处理的两处：`fql_e_uni_anchor_dataset_card.md` §4（0.85 是 Gate B 门槛依据）、
  `docs/archive/fql_succession/fql_succession_gate_b_interim_report.md` §7。
- **`*_completed.ipynb` 是执行记录**：路径可以同步改，数字一律不动。

## 5. 67 处真失效指针

分布极集中，前四份占 45 条：

| 文档 | 条数 | 性质 |
|---|---:|---|
| `auvhamnode_offline_mbrl_plan.md` | 18 | 已撤销的 plan，指向从未创建的文件 |
| `superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md` | 11 | SUPERSEDED v1 plan |
| `online_rl_thesis_plan.md` | 10 | thesis-grade 矩阵已取消 |
| `offline_mbrl_plan/NODE_IQL_FQL_SORL_revised_roadmap_v3.md` | 6 | 已 deprecated；其中 6 条是参考文献 refdef 误判 |
| 其余 | 22 | 散在 15 份文档 |

多数是「计划里预告了但从未产出的文件」，**不是断链，是计划没执行**。清理的正确动作多半是
在 plan 里标注「未产出」而不是补文件。动手前先分类，别一律当断链修。

## 6. 收尾必跑的三件事

```
python -m scripts.check_doc_pointers      # 真失效基线 = 67（在工作副本里跑）
python -m scripts.build_doc_index --check # 索引是否落后
python -m pytest tests/ -q --tb=no        # 需 --basetemp 指到 scratchpad，否则沙箱挡系统临时目录
```

`check_doc_pointers` 只验目标存在；横幅、rev 号、日期声明是否仍为真，机器验不了。

**两个自己踩过的坑：**

- **插横幅会顶掉行号引用。** 给 7 份文档各插两行归档横幅后，
  `data_integrity_open_items.md` 引的 `bug2_fix_decision.md:27` 得改成 `:29`。
  全仓目前只有这一处 `文件.md:行号` 形式，加横幅前先扫一遍。
- **把文档移进 `docs/archive/` 会让 `build_doc_index.py` 误判。** 横幅里出现字面量
  `archive` 时，紧邻的 `**Spec**:` 标签满足了强调判据，**活着的文档被标成 ARCHIVE**。
  `480d834` 已修（路径上下文跳过 + 逐次匹配），但同类误报未必只此一种：
  移完一定重建索引并**逐行看状态列的 diff**，不要只看总数。

## 7. 不属于这条线（用户 2026-08-17 确认延后）

**迁移收尾**（见 memory `repo-migration-to-local`）：

- 删 OneDrive 侧剩余四个数据目录（`checkpoints/ wake_data/ offline_data/ results/`，约 5.8 GB）——
  观察一两天后再动，**需用户再次确认**
- 删整个旧副本——最后一步
- **Mac 侧读数恢复**：OneDrive 双向同步，`experiments/` 的删除已传到 Mac，
  那边重建第 5 章插图会断；数据未丢（D 盘完整副本 + Google Drive 的 Colab 副本 + 回收站 30 天），
  补回只需**读数层约 125 MB**

**研究侧**（用户明确「等仓库整理完再做」）：数据完整性 ①③⑤ 的处置未决，
三条同时落在 §5.6.2 那一句上。入口 [`data_integrity_open_items.md`](data_integrity_open_items.md)
+ [`../paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../paper/thesis_ch5/data_integrity_impact_assessment_review.md)。
污染面至今未封闭——Drive 侧枚举本身被证残缺。
