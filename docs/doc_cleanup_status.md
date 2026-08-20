# 文档整理 — 状态与交接

> 🔧 **进行中（2026-08-17）** — 本文件是「全仓文档整理」这条线的**唯一状态账本**，
> 供跨会话接手。三层做完即可删除本文件。
> 状态只写在这里，不要复制进 `CLAUDE.md`、线总览或 memory。

## 0. 动手之前：先确认你在哪个副本

**本机（Windows 副机）绝对路径，仅本文件此处出现：**

- 工作副本 = `D:\Codes\rl_v2` ← **一切改动都在这里**（2026-08-18 由 `C:\Code\rl_v2` 迁入，
  代码与数据从此同处一盘、**不再有 junction**）
- 退役中的旧副本 = `C:\Users\jinxiang\OneDrive\我的\Code\new_off_rl\rl_v2`（停在 `3d1f83b`，
  落后 30+ 个提交；用户裁决暂不处置）

三个必踩的坑：

1. **会话默认 cwd 可能仍指向 OneDrive 旧副本。** 旧副本的 `CLAUDE.md` 是迁移前的版本，
   `experiments/` 已删、10 个提交也不在那里。在旧副本上跑校验会得到假结果——
   `check_doc_pointers` 在旧副本报**真失效 150**，在工作副本报 **67**，差的 83 条全是
   指向已删 `experiments/` 的路径。**每条命令都显式带上工作副本路径。**
   ⚠ **2026-08-18 复测：工作副本上 `check_doc_pointers` 现报真失效 `0`**（未解析 138 全部落在
   自述缺席／计划表预告／产物路径／示例占位／绝对路径五类豁免里）。上面的 `67` 与下表「附」层的
   `67 处 ⬜ 未做` 都已过期——**归因已核完，见 §5**。
2. **git 用 `git -C "<工作副本>" …`。** `cd <路径> && git …` 这种复合形式会被权限层拒绝
   （推送时连拒两次，换成 `git -C` 一次就过）。
3. ~~**五个数据目录是 junction**（`results/ experiments/ checkpoints/ wake_data/ offline_data/`
   实体在 D 盘）。~~ **2026-08-18 起不再成立**：五个目录已随迁移移入工作副本内部，是**真目录**。
   撤掉 junction 的理由不是整洁——Git Bash 的 `find` 不跟随 junction、`pathlib.rglob` 拒绝下降进
   symlink 目录，而本仓的完整性审计全靠目录枚举，多一层挂载就多一类「文件在、枚举没返回它」。
   撤后 `find` 与 `rglob` 在五个目录上逐一同数（60 / 4508 / 1060 / 1932 / 24）。
   它们仍**不随 `git clone` 到位**（全部 gitignore），这条没变；`benchmarks/` 仍是被 git 追踪的真目录。
   ⚠ `D:\rl_v2_data\` 未清空：`_archive` `_backup` 按裁决留在仓库外；同目录下另有一份
   `benchmarks` **旧副本**（与仓库内那份同名同项、各自独立、会漂移），读的时候别拿错。

## 1. 这条线分三层

| 层 | 内容 | 状态 |
|---|---|---|
| ② | 两份 line summary 的**真值核对**——逐条对照源文档、`results/`、CSV、`metadata.json` | ✅ 已完成 |
| ③ | 文档**归位与角色标注** | ✅ 已完成 |
| ① | 25 篇**论文承重文档**深核 | 🔧 **进行中** —— 第 1 批已落（12/25 已深核，见 §4.1） |
| 附 | 67 处**真失效指针** | ✅ **已闭环，但闭环方式须照 §3 引用**——不是「把 67 个文件补出来」 |

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

全部已推送。**主干 2026-08-20 起为 `main`**（`codex-arrival-v2-prototype` 已快进合入并退役，上面这批提交均在 `main` 上）。

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

### 4.1 进度（第 1 批，2026-08-20）

**已深核 12 篇**（沿上述优先级规则，先查引用三个高危源文档的）：广验 v2 report / plan / seed43 supplement plan、`rebrac_line_overview`、`rebrac_mainline_review`、`offline_rl_line_summary`、`data_integrity_open_items`、`td3bc_phase0c_experiment_report`、`rebrac_statistical_test_followup`、`td3bc_worldcomp_teacher_gap_experiment_report`、`rebrac_method_section_draft`、`rebrac_experiment_report`（仅 §10A 指针）。

**三个高危源文档中的两个已清**：广验 v2 supplement（2026-07-12）与 asym-critic addendum（2026-05-27）的传播逐处核过——作废的 `0.850 ± 0.024 / −5.2pp / 同向退化` 每一份副本都带撤销标记，论文侧 `boundary.tex` 已全面改三种子并自披露「配对诊断基于首轮两种子」。**第三个（data_integrity ①③⑤）则漏了一大片**，见下。

**本批 findings（均已落）**：

| # | 文档 | 问题 | 改法 |
|---|---|---|---|
| F1 | `offline_rl_line_summary.md` §4.5 | **四处过期**：标题与正文均写「处置未决/待裁决」（实已 2026-08-16 走 ①-c 落地）；「δ 仍未测」（实已测，源文档 `4295f68` 已订、本副本漏跟）；污染面「至今未封闭」（实为章内已封、全仓未封两层） | 订正 + 标明原文 |
| F2 | `rebrac_statistical_test_followup.md` | rev.1、**文内无任何日期**、paper 侧 10 处引用。它是 §5.7.2 「差距闭合过半」写作建议的源头，而第 5 章已改「约半」（留出 60 条上 48.9%）——文内零披露 | 加追注，**点估计不改** |
| F3 | `td3bc_worldcomp_teacher_gap_experiment_report.md`:431 | 「特权 critic 关闭约 48.5%」无嵌套限定；留出 60 条上为 35.6%（$-12.9$ pp） | 同上 |
| F4 | 本文 §7 | ①③⑤ 处置的指针只写 `discussion.tex` §5.10.4（那里只有章级收束句） | 改指正文三处 |

**本批新增一条筛法（与 §3 优先级规则并用）：数同一条陈述有几份平行副本。** F1 四处全出于同一机制：源文档改了，副本没改。同一条「处置未决」当时共三份（`CLAUDE.md`、本文 §7、`offline_rl_line_summary`），前两份已订而第三份漂了四天；`paper/thesis_ch5/status.md` 那份早已划掉。**做法**：每碰到一段复述别处结论的文字，先 grep 它的特征数字或措辞找兄弟副本，有兄弟的八成有一份掉队。机械检查对这类全部失盲。

**剩下 13 篇未深核**：`arrival_v2_experiment_report`、`arrival_v2_sac_collector_design`、`environment_design`、FQL 系六篇（`p2_results` / `p2_main_spec` / `p2_xbench_spec` / `p2_mechanism_diagnostic` / `p2_collection_log` / `paper_writing_index`）、`online_rl_line_summary`、`online_sac_reward_redesign`、`rebrac_paper_writing_index`、`systematic_improved_sac_experiment_report`、`rebrac_experiment_report`（全文，1200+ 行）。

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

### 归因（2026-08-18 独立核查，不采信当初提交自述）

这一层已由 `ff29c43` 闭环，读数从 `67` 归 `0`。**闭环方式不是「把 67 个文件补出来」**，
引用这条结论时必须带上机制，否则会被误读成断链已修复。

做法：把当前树同时喂给 `ff29c43^` 与今天的 `check_doc_pointers.py`（同一文件系统、同一
$130$ 份语料，先把 `.claude/worktrees/` 那份陈旧 agent 副本移开——旧版工具会走进去、语料翻倍）。

| 口径 | 真失效 | 自述缺席 |
|---|---:|---:|
| 旧工具（`ff29c43^`）@ 当前树 | **84** | 桶不存在 |
| 今天的工具 @ 当前树 | **0** | 65 |

$84 \to 0$ 的分解：

- **$65$ 处（占 77%）转入 `自述缺席`**——引用方在正文里写明为何不存在，工具读那句话。
  这**不是悄悄吸收**：同一个提交里工具改了 97 行、17 份文档各加 2–4 行声明，四条缘由是闭集。
- **约 $19$ 处根本不是断链**，是旧工具的误报，新版停止把它们当指针解析（解析总数 $3810\to3790$）：
  `.jsonl` 被按 `.json` 读、脚注定义被当成链接引用定义、围栏代码块里的 markdown 链接字面量、
  带 `:N` 行号的引用。修的是脚本，不是文档。
- **$0$ 处靠「把缺的文件建出来」修好。**
- **$4$ 处是真订正**（文件确实搬过／改过名，写「从未产出」会是假话），典型如
  `01_static_distribution_audit.md` 把 `_wake_stats.py` 的引用从 `experiments/` 下的旧位置
  改指到 [`auvhamnode_spike/_wake_stats.py`](auvhamnode_spike/_wake_stats.py)（`a68b4c2` 之后）。

**「链接不删」经查属实**：`ff29c43` 文档侧删除行里的 $2$ 个 markdown 链接全部在新增行重现，
净增 $13$ 个，**零链接被删**——没有靠删引用来降数。

**遗留的持续义务**：`自述缺席` 是全仓唯一装**声明**而非事实的桶。一份文档若对着历史上确实
新增过、后来只是被搬走的文件写「从未产出」，就永久掐掉了自己的警报。这次逐条拿
`git log --all --diff-filter=A -- <path>` 对质过：$28$ 条声明（对应桶内 $65$ 处引用）
**$0$ 处 SUSPECT**、$27$ 处 consistent、$1$ 处 weak。
weak 那条出自 v1 广验 plan，指的是它那个一次性 notebook 生成器脚本：声明说建过之后按计划删掉，
而 git 里查不到该路径被新增过。它由该 plan **自己 2026-05-04 的原文**佐证（第 1890 行
「this file is throw-away — delete after the notebook is generated」、第 2221 行的 `rm` 步骤），
早于那条补注三个多月，故判为可信而非可疑。
（此处刻意不写该脚本的完整路径——写了会让**本文件**也成为一处「自述缺席」声明，
把描述别人的声明变成自己发出声明。）

该对质已固化为 `python -m scripts.check_doc_pointers --verify-declarations`（配 `--strict`
时 SUSPECT 会让它退 1）。**每次文档搬家后重跑**——搬家正是让声明变假的那个动作。
探测器已用故障注入实测会响（对着 `a68b4c2` 搬走过的文件插一条「从未产出」，即报 SUSPECT 并退 1）。

⚠ 写这一节时踩到了工具自己：初稿举例用了 markdown 链接字面量、又把那个一次性脚本的完整路径
和「用后即删」写在同一行，结果 `真失效 0 → 1`、`自述缺席 65 → 66`。**描述指针问题的文档本身
也是被扫描的语料**——举例时避开可解析的写法，改动后重跑确认读数回到 $0$ / $65$。

## 6. 收尾必跑的三件事

```
python -m scripts.check_doc_pointers      # 真失效基线 = 0（2026-08-17 起；非 0 即为新引入）
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

**迁移收尾**（迁移本身见 `32ea60a`；原引的 memory `repo-migration-to-local` 已不存在）：

- 删 OneDrive 侧剩余四个数据目录（`checkpoints/ wake_data/ offline_data/ results/`，约 5.8 GB）——
  观察一两天后再动，**需用户再次确认**
- 删整个旧副本——最后一步
- **Mac 侧读数恢复**：OneDrive 双向同步，`experiments/` 的删除已传到 Mac，
  那边重建第 5 章插图会断；数据未丢（D 盘完整副本 + Google Drive 的 Colab 副本 + 回收站 30 天），
  补回只需**读数层约 125 MB**

**研究侧**：数据完整性 ①③⑤ 的**处置已于 2026-08-16 落地**——路线 ①-c（散文披露 + 登记敏感性读数），
入正文三处：`setup.tex` §5.3.5/§5.3.6（口径披露）、`rebrac.tex` §5.7.1/§5.7.2/§5.7.m（敏感性读数）、`discussion.tex` §5.10.4（章级收束）。**仍未闭合的是另一件事**：污染**枚举**只对第 5 章自有数据集闭合，
全仓未闭合——Drive 侧枚举本身被证残缺。入口 [`data_integrity_open_items.md`](data_integrity_open_items.md)
+ [`../paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../paper/thesis_ch5/data_integrity_impact_assessment_review.md)。
