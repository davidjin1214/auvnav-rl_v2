# 交接：诚信审计固化成 pytest（2026-08-23，第一批 ＋ 第二批进行中）

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
| 6 | 报告刊值 ⇄ `results/` 下逐 seed JSON 逐位复算，四条链 | `48b8d06` ReBRAC／`06ec295` td3bc phase0c／`6380082` arrival_v2／`2a8c311` FQL P2 | ◐ 3/4 已落表，见 §四 |
| 7 | 全章 ± 离散度口径 90 处归位 | `af6ee64`。`.tex` 侧有 `ch5_dispersion_audit.py`，`docs/*.md` 报告侧无工具 | ✅ `--ddof` 已补上报告侧，见 §四 |
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
| `scripts/check_doc_code_refs.py` | 新校验器。补两类引用：源码行锚与文档声称的函数名——`check_doc_pointers` 在自己的路径解析函数注释里承认这两类它验不了。全仓 156 处引用，1.8 s |
| `tests/test_check_doc_code_refs.py` | 29 项 |
| `tests/test_check_doc_pointers.py` | 32 项。含把 `a507eb3` 那次**手工**故障注入固化下来的 SUSPECT 用例，以及钩子的接线用例 |
| `tests/test_build_doc_index.py` | 19 项 |
| `.claude/hooks/doc_pointers.py` | 改成串跑两个 sweep。两者互补是构造出来的：前者验路径在不在，后者验行与名还在不在 |

合跑 82 项全绿；全仓 `pytest tests/` 245 passed / 4 skipped。运行方式（本机沙箱挡系统
temp，必须给 `--basetemp`）：

```bash
python -m pytest tests/test_check_doc_code_refs.py tests/test_check_doc_pointers.py \
    tests/test_build_doc_index.py -q --tb=short --basetemp=<scratchpad>/pytest_tmp
```

### 负控是实测的：第一批 40 条 ＋ 第二批 31 条，全部变红

上一轮交接要求「每条检查都要配负控」。做法是对被测工具逐条注入故障、确认对应用例变红。
下表是第一批的判据本身，shell 脚本是一次性的（含硬编码本机路径，未入库）；第二批
`audit_published_numbers` 那 31 条同法做过，判据见该脚本与
[`../tracebacks/README.md`](../tracebacks/README.md)。另有 1 次**刻意保持绿**的控制组注入，
用来隔离钩子的 markdown 门（见表下说明）。

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
| check_doc_code_refs | 取消 同名消歧 | `test_a_basename_collision_prefers_the_citing_files_own_subtree` |
| check_doc_code_refs | 偏移一律判缺陷 | `test_a_fragment_one_line_off_is_an_offset_not_rot` |
| check_doc_code_refs | 空行检查失效 | `test_a_citation_landing_on_a_blank_line` |
| check_doc_code_refs | 越界检查失效 | `test_line_number_past_the_end_of_the_file` |
| check_doc_code_refs | `--strict` 不返回 1 | `test_cli_exits_nonzero_only_under_strict` |

**符号判据改写后补跑的 11 条**（旧的「点号符号退回叶名」一条随文件归属一起作废，见 §三）：

| 被测工具 | 注入的故障 | 应变红的用例 |
|---|---|---|
| check_doc_code_refs | `_nearest` 退回「任一 token 最近」 | `test_one_shared_token_next_door_does_not_downgrade_a_real_drift` |
| check_doc_code_refs | 文件归属复活（判定改回「定义须在同现文件里」） | `test_a_function_that_lives_in_a_different_file_is_a_known_blind_spot` |
| check_doc_code_refs | 同上 | `test_a_symbol_named_beside_a_file_is_not_a_claim_about_that_file` |
| check_doc_code_refs | 存在性检查恒真 | `test_a_function_name_that_exists_nowhere` |
| check_doc_code_refs | 点号符号改判头名 | `test_a_dotted_symbol_is_judged_by_its_leaf_once_the_head_is_ours` |
| check_doc_code_refs | 未知头也判叶名 | `test_a_dotted_symbol_whose_head_is_unknown_is_skipped` |
| check_doc_code_refs | 模块名不算已知头 | `test_a_module_name_counts_as_a_known_head` |
| check_doc_code_refs | 多层点号也判 | `test_a_deeper_dotted_chain_is_beyond_what_can_be_attributed` |
| check_doc_code_refs | 取消「同句须提到仓内 .py」这道门 | `test_a_symbol_with_no_repo_file_on_the_line_is_not_judged` |
| check_doc_code_refs | 取消第三方前缀表 | `test_a_third_party_alias_beats_a_repo_class_of_the_same_name` |
| check_doc_code_refs | `HISTORICAL` 永不命中 | `test_a_renamed_symbol_may_be_declared_in_the_same_sentence` |
| 钩子 | 第二个 sweep 被摘掉 | `test_the_hook_runs_every_sweep_not_only_the_path_one` |
| 钩子 | 第三个 sweep 被摘掉（第二批加挂后补） | 同上 |
| 钩子 | 缺陷段过滤失效 | `test_the_defect_filter_keeps_only_the_failing_sections` |
| 钩子 | 忽略 sweep 的退出码 | `test_the_hook_is_quiet_on_this_repo_as_it_stands` |
| 钩子 | 坏 JSON 不再兜住 | `test_the_hook_survives_junk_on_stdin` |
| 钩子 | markdown 门被拆（**须先制造失败条件**，见下） | `test_the_hook_only_fires_on_markdown` |

**第三方前缀表差点被判成死码。** 新的「未知头则跳过」规则把 `np.zeros()` 一类全接住了，实测
`FOREIGN_PREFIX` 列的 16 个头在本仓既无定义也无同名模块——照原用例注入，拆掉它仍然全绿。它真正
承重的场景是**撞名**：仓里若有个类叫 `F`，`F.relu()` 的头就"是我们的"了。用例已改成那个场景，
注入才变红。删一个「测不出来」的机制之前，先看看是不是用例指错了地方。

**markdown 门也是同一类问题，且它自己测不出来。** 仓库干净时，拆掉门只是让钩子对 `.py` 也跑一遍
sweep，两个 sweep 都过 → 退 0 → 用例照绿。注入表里这一条因此是**两段式**的：先强制 sweep 报失败
造出可观测条件，再拆门。控制组（只强制失败、不拆门）实测**保持绿**，拆门后才红——绿的那次证明红
的那次是门被拆红的，不是强制失败红的。

**两处不可注入、已就地标注**：`.tex`／`.md` 行锚的 CRLF 往返由两个机制共同保证（`main` 里的
显式归一化 ＋ Python `read_text` 的通用换行转换），拆掉任一个契约仍成立，故该用例钉的是可观测
行为而非某一实现；同理，`jsonl` 不被截成 `json` 由 alternation 顺序保证，与尾部守卫无关——
守卫真正拦的是 `.mdx`／`.python` 被读成 `.md`／`.py`，已拆成两条用例分别钉。

---

## 三、校验器抓到的 9 处——**已全部处置**（逐条核过，非工具输出转述）

`python -m scripts.check_doc_code_refs --strict` 现在退出码 **0**，缺陷 0；
`check_doc_pointers --strict --verify-declarations` 同样退 0，真失效 0、SUSPECT 0。

### 5 处代码行锚落在空行——已按下表改号，被引符号都还在

| 引用方 | 现值 | 应为 | 依据 |
|---|---|---|---|
| `docs/archive/fql_succession/fql_succession_bug2_fix_decision.md` 第 17 行（显示文本与链接目标两处） | `train_utils.py` 185，区间 185-213 | **186**，区间 **186-216** | `_resolved_eval_episodes` 定义在 186；下一个顶层定义在 217 |
| 同上，第 128 行 | `train_utils.py` 185 | **186** | 同上 |
| `docs/arrival_v2_experiment_report.md` 第 287 行 | `train_sac.py` 164 | **165** | `saved_ckpt_dir = trainer_state.get("checkpoint_dir")` 在 165 |
| `docs/online_rl_thesis_plan.md` 第 49 行 | `env.py` 895 | **947** | `"privileged_obs": equivalent_body[:2]...` 在 947 |
| 同上，第 308 行 | `env.py` 895 | **947** | 同上 |

### 3 处符号引用——判据改写后，2 真 1 假，均已处置

判据改写见 §四之前的记录：砍掉「符号与 `.py` 路径同现 ⇒ 声称定义在该文件」，改判**符号在全仓
是否存在**（裸名判自身；点号 `A.b()` 在 `A` 是仓内类或模块时判叶名 `b`，头名不认识就跳过）。
代价写进了 docstring：**查不出「函数搬家了」**——名字只要在仓里有定义就放行，哪怕文档把读者指错
了文件。这个局限本身有用例钉着（`test_a_function_that_lives_in_a_different_file_is_a_known_blind_spot`），
免得日后无声地被改回去。

- `docs/superpowers/plans/2026-04-04-improved-sac.md` 第 621 行：**真**。计划文档让实现者去找
  `auv_nav/env.py` 里的一个方法，那名字 **`git log -S` 确认从未在代码里存在过**，落地时定名为
  `_build_info`（在 896 行）。与 `a49fb1e` 那条 CLAUDE.md 编函数名同类。**处置**：计划文档不改写
  它当初的计划，改用工具里的 `HISTORICAL` 同句自述（补 "shipped as `_build_info()`, now at L896"），
  落进「自述历史引用」桶。该分支上一轮尚未实测会 fire，本轮已补用例并注入验红。
- `docs/archive/fql_succession/fql_succession_p0p1_spec.md` 第 479 行：**假阳性，已随判据消失**。
  原句是「……返回的 metrics dict 必须包含以下 key（与 `train_offline.py` 现有 logger 兼容）」——
  同现不是归属声明。这一条现在是回归用例
  `test_a_symbol_named_beside_a_file_is_not_a_claim_about_that_file`。
- `docs/auvhamnode_spike/04_swap_vehicle_decision_memo.md` 第 55 行：**真**。`PlanarRemusEnv`
  在 `auv_nav/env.py` 里，但 `compute_flow_at_position` 全仓无定义（grep 实测 0 命中）。真正
  的采样口是 `FlowSampler.sample_probes_body()`（`auv_nav/flow.py`，`env.py:844` 调用）。
  **处置**：这句是对代码的事实描述、不是历史计划，直接改成真名，论断一字未动。

### 4 处「行号偏移」——手工复核后，1 处是被误降级的真漂移

偏移桶本身的设计站得住：文档常引「效果落地的那一行」而在句中引上一行的条件，两种写法都诚实，
所以差 1–2 行**不进 `--strict`**（要收紧用 `--near-window 0`）。唯一经手工复核确认诚实的是
`docs/environment_design.md:324 → auv_nav/env.py:377`——377 是赋值行，句中引的条件在 376。

但**逐条核这个提示桶，核出了工具自己的一个缺陷**：`_nearest` 原本取「离引用最近的、命中任一
token 的行」。`docs/arrival_v2_experiment_report.md:285` 引 `train_sac.py:467`，句中引了
`env_step`、`num_envs`、`range`、`start_step` 四个标识符，四个全在 480 行那条循环上；而最常见
的 `num_envs` 恰好也出现在 466 行。「任一 token」于是答 466，一处 **13 行的真漂移被判成 1 行
偏移**——降出 `--strict`，从此不再有人看见。改成**先比命中 token 数、再比距离**后，它正确落进
缺陷桶并指向 480。已改文档为 480，另两处 `collect_offline_data.py` 315 → **316** 也一并改直
（`5f8228c` 顶下来的，引的语句本身没变，只是移了位）。

教训写在 §五：**提示桶要手工逐条核**。它不进 `--strict`，所以工具在这里判错不会有任何东西报警。

---

## 四、下一步

上一轮列的三件（改符号判据 / 处置 §三 / 钩子接线）**本轮已全部做完**。钩子现在串跑两个 sweep；
`.claude/settings.json` 里的那条 PostToolUse 命令指的还是同一个脚本路径，**无需改配置**（该文件
gitignored，另一台机器要手工加，办法写在钩子自己的 docstring 里）。

**第二批 = 刊值复算**（用户已定顺序）。范围来自 §一 C 类，**已开工**：

| C 类 | 内容 | 状态 |
|---|---|---|
| 6 | 四条数字溯源链做成可重跑命令 | ◐ 工具 ＋ **3/4** 条链落表（FQL P2 35 ＋ td3bc phase0c 57 ＋ ReBRAC 85 = **177 处刊值全部吻合**）。arrival_v2 未落 |
| 7 | `docs/*.md` 报告侧的 ddof 口径工具化 | ✅ `--ddof` 三份报告合计 **57 处**：55 处 `ddof=0`、2 处 `ddof=1`，且那 2 处是同一个读数（`0.9340 ± 0.0261`，印在两张表里）。**机械复现了 `48b8d06` 手工查出的唯一一处 ddof=1** |
| 11 | 刊出读数 ⇄ manifest 指纹归属 | ⬜ 未动。工具 `paper/thesis_ch5/tools/ch5_manifest_attribution.py` 已存在，缺的是测试——与第三批第 5 项同源，可能合批 |

已交付：[`../../scripts/audit_published_numbers.py`](../../scripts/audit_published_numbers.py)
＋ [`../tracebacks/`](../tracebacks/README.md) 下的三份溯源表 ＋ 31 项测试（31 条注入判据全红，
其中一条注入的是**溯源表数据本身**而非代码）。已挂进 markdown 编辑钩子，串为第三个 sweep。

ReBRAC 那份还把该报告 §1 的**口径补注表本身**纳入核对——那张表印了逐种子值与两套口径，于是
「这份报告只有一处 ddof=1」这句话不再是被信任的，而是被重算出来的。

**arrival_v2 链要先扩工具**：它的源是 `eval_log.csv` 与 `final_eval.json`，现在的
`read_metric()` 只读 JSON。`6380082` 那轮的主发现是「论文侧已改、报告侧未回填」，属表述层，
不是本工具能验的；能验的是它逐格核过的 39 次周期评估与 11 个 run 的均值/peak/首达步。

`offline_data/` 与 `results/` 都是 gitignored 的，脚本一律支持把根指到 Drive 挂载，
**「本地扫不到」不能写成失败**（`no-data` 桶，不进 `--strict`）。

**第三批 = `paper/thesis_ch5/tools/` 全覆盖**（12 个脚本，其中 6 个需 `main.aux`、5 个需
`results/`）＋ C 类第 12 项。

---

## 五、方法论：本轮抓到的六个「测试跑绿但不承重」

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
4. **仓库干净时，「守门」类机制自己测不出来。** 拆掉门，被放进来的东西照样合格，用例照绿。本轮
   两次：钩子的 markdown 门、第三方前缀表。两种解法都用上了——**两段式注入**（先造出可观测的
   失败条件，再拆门；并跑一次只造条件、不拆门的控制组，确认红的是门），和**把用例改到该机制真正
   唯一承重的场景上**（前缀表→撞名）。删一个「测不出来」的机制之前，先确认不是用例指错了地方。
5. **不进 `--strict` 的提示桶要手工逐条核。** 工具在那里判错不会有任何东西报警。本轮正是逐条核
   偏移桶时，发现 `_nearest` 把一处 13 行的真漂移降级成了 1 行偏移（§三）。**分级本身要被审，
   不只是被信任。**
6. **一条什么都没改的注入，读起来和「测试没抓住」一模一样。** `sed` 匹配不上时静默留下原文，
   用例照过，表上是 GREEN。而**改动被测源码正是让旧注入失配的原因**——第二批扩了
   `audit_published_numbers` 的定位逻辑后，一条旧 sed 就此变成空操作。注入脚本已加校验：
   施加后比对文件校验和，没变就报 `NOOP` 而不是 GREEN。与第 2 条（测试 ID 写错也退非零）同源：
   **注入脚本的每一步都要有独立证据，不能只读最后那个退出码。**（一次遗留疑点：加校验前那轮
   还有一条 `capture failure ignored` 报 GREEN，加校验后重跑为 RED 且确认突变已施加；两次结果
   不一致，未能复现出成因，此处如实记下。）
