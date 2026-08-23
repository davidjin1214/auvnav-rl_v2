# 交接：诚信审计固化成 pytest（2026-08-23，第一批）

承接 [`2026-08-23-tooling-followup.md`](2026-08-23-tooling-followup.md) 的**方向 2**。该文判
方向 1（skill 效力测试台）推迟、方向 3（环境加固）只剩零碎，本轮只做方向 2。

用户已定的批次顺序：**第一批 = 守卫类 + 引用校验**（本轮），第二批 = 刊值复算，第三批 =
`paper/thesis_ch5/tools/` 全覆盖。

---

## 一、审计考古结果——四类 12 项

上一轮交接没交代的难点：**审计清单本身不在任何一份文档里**。
[`../data_integrity_open_items.md`](../data_integrity_open_items.md) 记的是待办与已闭合项，
不是「历史上做过哪些审计」。清单靠三处凑出来：`git log --grep`（343 提交里 121 条候选，
剔掉「探针=传感探针」「复审=论文评审」等假阳性后集中在 2026-08）、10 份 `.tex` 的 67 行
rev 块核查动作、以及跨仓会话记录。

### A 类 · 有脚本 + 有测试（回归已受保护）

| # | 审计 | 覆盖 |
|---|---|---|
| 1 | [`../../scripts/audit_seed_overlap.py`](../../scripts/audit_seed_overlap.py) 种子重叠／污染面枚举／账本对帐 | 13 项测试，**含故意喂假枚举的负控**，退出码 2 契约在测 |
| 2 | [`../../scripts/audit_multimodality.py`](../../scripts/audit_multimodality.py) FQL 多模态审计契约 | 有测试 |

### B 类 · 有脚本、零测试

| # | 审计 | 缺口 | 本轮 |
|---|---|---|---|
| 3 | [`../../scripts/check_doc_pointers.py`](../../scripts/check_doc_pointers.py)（含 `--verify-declarations`） | 已由 `8222c70` 挂上 hook 自动跑，却零回归保护 | ✅ 已补 |
| 4 | [`../../scripts/build_doc_index.py`](../../scripts/build_doc_index.py) `--check` | `2d4ba97` 的 worktree 语料翻倍 bug 即出自它 | ✅ 已补 |
| 5 | [`../../paper/thesis_ch5/tools/`](../../paper/thesis_ch5/tools/README.md) 12 个 `.py` | `grep -rl thesis_ch5 tests/` 零命中 | ⬜ 第三批 |

第 5 项的前置：门槛类 6 个依赖 `latexmk` 产物 `main.aux`（本机有 MiKTeX，但仓库按约定编译后
即清中间文件）；证据类 5 个依赖 `results/`（本机 4508 文件／259 MB 在位，clone 为空）。

### C 类 · 只做过一次、无脚本（最容易漏的一批）

| # | 审计 | 出处 | 本轮 |
|---|---|---|---|
| 6 | 报告刊值 ⇄ `results/` 下逐 seed JSON 逐位复算，四条链 | `48b8d06` ReBRAC／`06ec295` td3bc phase0c／`6380082` arrival_v2／`2a8c311` FQL P2 | ⬜ 第二批 |
| 7 | 全章 ± 离散度口径 90 处归位 | `af6ee64`。`.tex` 侧有 `ch5_dispersion_audit.py`，`docs/*.md` 报告侧无工具 | ⬜ 第二批 |
| 8 | 文档→源码**行号**引用（全仓 42 处） | `2a8c311` 明写「这类引用 check_doc_pointers 验不了，只能实读」 | ✅ 已固化 |
| 9 | 文档→**函数名**引用 | `a49fb1e`：CLAUDE.md 里的 get_probe_positions 全仓只出现在 CLAUDE.md 自己 | ⚠ 判据待改，见 §四 |
| 10 | 文档自述横幅 ⇄ 引用它的表格状态栏 | `cce2b2d`：13 行核出 3 行对不上 | ⬜ 未做 |
| 11 | 刊出读数 ⇄ 评估 manifest 指纹归属 | `ch5_manifest_attribution.py`，3715 份读数零落空 | ⬜ 第二批 |
| 12 | `benchmarks/` 落库时逐条核（条数／种子连续／区间） | `0dc35a2`，纯手工 | ⬜ 第三批 |

### 跨仓那条线：已查，无遗漏，别再扫一遍

`data_integrity_open_items.md` 的诞生现场确实在 `D:\Codes\claude_windows_folder`
（2026-08-01 一次模拟审稿演练，对照臂 agent 翻了本仓 `paper/` 与 `offline_data/`）。逐条核过
该会话：它在代码层验的是「1000 回合数据集种子不相交」与 Table 1 transitions，**两条都落进了
文档**。其余跨仓命中（g3_5_5、Overleaf 等 15 个会话）全部是记忆文件里的引文，不是真跑过审计。
**结论：审计历史全在本仓的 git log ＋ rev 块 ＋ docs 里，跨仓无需再查。**

---

## 二、本轮已交付

| 文件 | 内容 |
|---|---|
| `scripts/check_doc_code_refs.py` | 新校验器。补两类引用：源码行锚与文档声称的函数名——`check_doc_pointers` 在自己的路径解析函数注释里承认这两类它验不了。全仓 153 处引用，1.8 s |
| `tests/test_check_doc_code_refs.py` | 23 项 |
| `tests/test_check_doc_pointers.py` | 29 项。含把 `a507eb3` 那次**手工**故障注入固化下来的 SUSPECT 用例 |
| `tests/test_build_doc_index.py` | 19 项 |

合跑 71 项全绿。运行方式（本机沙箱挡系统 temp，必须给 `--basetemp`）：

```bash
python -m pytest tests/test_check_doc_code_refs.py tests/test_check_doc_pointers.py \
    tests/test_build_doc_index.py -q --tb=short --basetemp=<scratchpad>/pytest_tmp
```

### 负控是实测的：24 次故障注入，24 次变红

上一轮交接要求「每条检查都要配负控」。做法是对被测工具逐条注入故障、确认对应用例变红。
下表是判据本身，shell 脚本是一次性的（含硬编码本机路径，未入库）。

| 被测工具 | 注入的故障 | 应变红的用例 |
|---|---|---|
| check_doc_pointers | worktree 不再豁免 | `test_agent_worktrees_are_not_part_of_the_corpus` |
| check_doc_pointers | `.claude` 被整体豁免 | `test_dot_claude_itself_is_still_walked` |
| check_doc_pointers | 去掉扩展名尾部守卫 | `test_an_extension_may_not_match_a_prefix_of_a_longer_one` |
| check_doc_pointers | 脚注当成引用定义 | `test_a_footnote_definition_is_not_a_link_reference` |
| check_doc_pointers | 围栏内链接也算指针 | `test_links_inside_a_fenced_block_are_not_pointers` |
| check_doc_pointers | 声明不要求同一行 | `test_a_declaration_must_sit_on_the_same_line_as_the_path` |
| check_doc_pointers | 声明作用域跨文件 | `test_a_declaration_is_scoped_to_the_file_that_makes_it` |
| check_doc_pointers | SUSPECT 判定失效 | `test_a_false_never_built_claim_is_flagged_and_fails_strict` |
| check_doc_pointers | `--strict` 不再退 1 | `test_an_undeclared_miss_is_not_excused` |
| build_doc_index | worktree 进索引 | `test_agent_worktrees_are_not_indexed` |
| build_doc_index | 数据目录进索引 | `test_gitignored_data_directories_are_skipped_at_the_top_level` |
| build_doc_index | 去掉横幅强调要求 | `test_a_plain_mention_inside_the_banner_without_emphasis_does_not_count` |
| build_doc_index | 路径里的关键词算横幅 | `test_a_keyword_inside_a_path_is_not_a_banner` |
| build_doc_index | `--check` 永远通过 | `test_check_fails_once_a_doc_is_added` |
| build_doc_index | 标题不截断 | `test_a_long_title_is_truncated` |
| check_doc_code_refs | 取消 路径 span 排除 | `test_a_backticked_path_is_not_read_as_the_fragment` |
| check_doc_code_refs | 取消 多引用不归属 | `test_several_citations_on_one_line_disable_fragment_attribution` |
| check_doc_code_refs | 取消 span 邻近约束 | `test_a_fragment_far_along_the_line_is_not_attributed` |
| check_doc_code_refs | 点号符号退回叶名 | `test_a_dotted_symbol_is_judged_by_its_class_not_its_method` |
| check_doc_code_refs | 取消 同名消歧 | `test_a_basename_collision_prefers_the_citing_files_own_subtree` |
| check_doc_code_refs | 偏移一律判缺陷 | `test_a_fragment_one_line_off_is_an_offset_not_rot` |
| check_doc_code_refs | 空行检查失效 | `test_a_citation_landing_on_a_blank_line` |
| check_doc_code_refs | 越界检查失效 | `test_line_number_past_the_end_of_the_file` |
| check_doc_code_refs | `--strict` 不返回 1 | `test_cli_exits_nonzero_only_under_strict` |

**两处不可注入、已就地标注**：`.tex`／`.md` 行锚的 CRLF 往返由两个机制共同保证（`main` 里的
显式归一化 ＋ Python `read_text` 的通用换行转换），拆掉任一个契约仍成立，故该用例钉的是可观测
行为而非某一实现；同理，`jsonl` 不被截成 `json` 由 alternation 顺序保证，与尾部守卫无关——
守卫真正拦的是 `.mdx`／`.python` 被读成 `.md`／`.py`，已拆成两条用例分别钉。

---

## 三、校验器已抓到、**尚未处置**的 8 处（逐条核过，非工具输出转述）

`python -m scripts.check_doc_code_refs --strict` 当前退出码 **1**。

### 5 处代码行锚落在空行——直接改号即可，被引符号都还在

| 引用方 | 现值 | 应为 | 依据 |
|---|---|---|---|
| `docs/archive/fql_succession/fql_succession_bug2_fix_decision.md` 第 17 行（显示文本与链接目标两处） | `train_utils.py` 185，区间 185-213 | **186**，区间 **186-216** | `_resolved_eval_episodes` 定义在 186；下一个顶层定义在 217 |
| 同上，第 128 行 | `train_utils.py` 185 | **186** | 同上 |
| `docs/arrival_v2_experiment_report.md` 第 287 行 | `train_sac.py` 164 | **165** | `saved_ckpt_dir = trainer_state.get("checkpoint_dir")` 在 165 |
| `docs/online_rl_thesis_plan.md` 第 49 行 | `env.py` 895 | **947** | `"privileged_obs": equivalent_body[:2]...` 在 947 |
| 同上，第 308 行 | `env.py` 895 | **947** | 同上 |

### 3 处符号引用——**先改判据再动文档**，见下节

- `docs/superpowers/plans/2026-04-04-improved-sac.md` 第 621 行：计划文档让实现者去找
  `auv_nav/env.py` 里的一个方法，而那个名字 **`git log -S` 确认从未在代码里存在过**，落地时
  定名为 `_build_info`（在 896 行）。与 `a49fb1e` 那条 CLAUDE.md 编函数名同类，是真发现。
- `docs/archive/fql_succession/fql_succession_p0p1_spec.md` 第 479 行：**假阳性**。原句是
  「……返回的 metrics dict 必须包含以下 key（与 `train_offline.py` 现有 logger 兼容）」——
  同现不是归属声明。
- `docs/auvhamnode_spike/04_swap_vehicle_decision_memo.md` 第 55 行：类名存在于
  `auv_nav/env.py`，但那个方法名全仓不存在。**归属判错、缺陷判对。**

### 4 处「行号偏移」——已判为提示、不判缺陷

含诚信账本 `docs/data_integrity_open_items.md` 第 202 行引 `collect_offline_data.py` 315 行
（`5f8228c` 已把它顶到 316，同一引用另在 `paper/thesis_ch5/notes/data_integrity_batch_review_findings.md`
第 18 行）。这类差 1–2 行的偏移**不进 `--strict`**：文档常引「效果落地的那一行」而在句中引上一行
的条件，两种写法都诚实。要收紧用 `--near-window 0`。

---

## 四、下一步（按优先级）

1. **改 `check_doc_code_refs` 的符号判据。** 现在的 Form-B 是「符号与某个 `.py` 路径同现在一行
   ⇒ 声称定义在该文件」。实测 `symbol-moved` 桶 2 条命中**全是假阳性**——同现推不出归属，散文里
   也没有可靠的归属标记。按证据砍掉文件归属，改判**「符号在全仓是否存在」**：裸符号判自身，
   点号符号 `A.b()` 在 `A` 存在时判叶名 `b`。这样三个真类全部命中（两个编造的函数名 ＋ 一个编造的
   方法名），`FQLAgent.update()` 正确放行。把「查不出『函数搬家了』」写进 docstring 的已知局限。
   改完**重跑注入表**里 check_doc_code_refs 的 9 条。
2. **处置 §三 的 8 处。** 5 处按表改号；3 处符号等第 1 步做完再看还剩几条，剩下的用工具里已就位
   的 `HISTORICAL` 同句自述（闭列表：当时的名字／当时叫／当时的行号／原计划名／落地时定名／
   后改名为，必须同时写出真名）——该分支**尚未实测会 fire**，动它之前先加用例。
3. **接线与落库决策。** `.claude/hooks/doc_pointers.py` 现在只跑 `check_doc_pointers --strict`；
   是否加挂新校验器待定——`--strict` 现仍退 1，接上会即刻变红，须先做完第 2 步。

---

## 五、方法论：本轮抓到的三个「测试跑绿但不承重」

写下来是因为第二、三批还会遇上。

1. **两个机制互相遮蔽。** 一条用例同时覆盖两个防御时，拆掉任一个另一个都会兜住，用例照绿。
   本轮撞上两次（路径 span 排除 ↔ 多引用不归属；横幅的引用块作用域 ↔ 强调要求）。**判据：
   一条用例只钉一个机制，且必须实测「拆掉它就红」。**
2. **注入脚本自己会撒谎。** 测试 ID 写错时 `pytest` 也退非零，被读成「突变被抓住」。已给注入
   脚本加 `--collect-only` 前置闸。这与上一轮交接第四节点名的是同一类错——**把工具的输出读成
   了自己以为的东西**。
3. **源注释里的例子可能归错因。** `check_doc_pointers` 的注释把 `jsonl` 不被截断归给尾部守卫，
   实测归因错误（是 alternation 顺序）。**写回归测试前先把注释里的因果实测一遍**，否则测出来的
   是注释而不是代码。
