# 交接：诚信审计固化成 pytest（2026-08-23 起，三批已全部完成）

承接 [`2026-08-23-tooling-followup.md`](2026-08-23-tooling-followup.md) 的**方向 2**。该文判
方向 1（skill 效力测试台）推迟、方向 3（环境加固）只剩零碎，本轮只做方向 2。

用户已定的批次顺序：**第一批 = 守卫类 + 引用校验**、**第二批 = 刊值复算**、**第三批 =
`paper/thesis_ch5/tools/` 全覆盖**。三批均已完成（第三批于 2026-08-24 收尾，交付物见 §五）。

> **本文件里所有计数都是写下它那一轮的快照，不要当现值引用。** 用例数、注入条数、链数、
> 刊值数每一轮都在动，逐处回填是 2026-08-24 实测过会漏的（见 §一表下那条注）。现值一律
> 现算：`python -m scripts.audit_published_numbers` 头一行给链数与刊值数，`--ddof` 给口径
> 分布，`pytest tests/ -q` 给用例总数，`git log --oneline` 给批次边界。

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
| 5 | [`../../paper/thesis_ch5/tools/`](../../paper/thesis_ch5/tools/README.md) 12 个 `.py` | `grep -rl thesis_ch5 tests/` 零命中 | ✅ **12/12**（131 项用例，见 §五）|

第 5 项的前置：门槛类 6 个依赖 `latexmk` 产物 `main.aux`（本机有 MiKTeX，但仓库按约定编译后
即清中间文件）；证据类 5 个依赖 `results/`（本机 4508 文件／259 MB 在位，clone 为空）。

### C 类 · 只做过一次、无脚本（最容易漏的一批）

| # | 审计 | 出处 | 本轮 |
|---|---|---|---|
| 6 | 报告刊值 ⇄ `results/` 下逐 seed JSON 逐位复算，四条链 | `48b8d06` ReBRAC／`06ec295` td3bc phase0c／`6380082` arrival_v2／`2a8c311` FQL P2 | ✅ 4/4 已落表（第二批当时 320 处刊值），见 §四；2026-08-24 又加了在线线与论文 `.tex` 侧五条 |
| 7 | 全章 ± 离散度口径 90 处归位 | `af6ee64`。`.tex` 侧有 `ch5_dispersion_audit.py`，`docs/*.md` 报告侧无工具 | ✅ `--ddof` 已补上报告侧，见 §四 |
| 8 | 文档→源码**行号**引用（全仓 42 处） | `2a8c311` 明写「这类引用 check_doc_pointers 验不了，只能实读」 | ✅ 已固化 |
| 9 | 文档→**函数名**引用 | `a49fb1e`：CLAUDE.md 里的 get_probe_positions 全仓只出现在 CLAUDE.md 自己 | ✅ 判据已改写、3 处已处置，见 §三 |
| 10 | 文档自述横幅 ⇄ 引用它的表格状态栏 | `cce2b2d`：13 行核出 3 行对不上 | ⬜ 未做 |
| 11 | 刊出读数 ⇄ 评估 manifest 指纹归属 | `ch5_manifest_attribution.py`，3715 份读数零落空 | ✅ 已并入第三批（`db0c235`，19 项用例）|
| 12 | `benchmarks/` 落库时逐条核（条数／种子连续／区间） | `0dc35a2`，纯手工 | ⬜ 未做，**未分配批次**（第三批 `925d9c4` 收尾时未含它）|

**这张表自己漂过两处状态栏，两处都正是第 10 项要查的那个毛病**（2026-08-24 核出并已改）：

- **第 9 行**曾停在「⚠ 判据待改」，而判据在同一轮就改完了、3 处也已逐条处置——§三 有完整
  记录，`tests/test_check_doc_code_refs.py` 里两条盲点用例实测在位，**只有这张表没跟着动**。
  成因是处置写在 §三、状态栏在 §一，改一处不带动另一处。
- **第 12 行**曾写「⬜ 第三批」。那是**预告不是状态**：第三批的范围后来定成
  `paper/thesis_ch5/tools/` 12 个脚本，`925d9c4` 收尾时不含它，没人回来改这一格。

给第 10 项的两条现成判据：**（a）** 状态标记与它所依据的正文分处两节时必漂，核的时候要按
「它指的那一节现在说什么」判，不能按状态栏自述判；**（b）** 状态栏里写批次名／计划名而非
既成事实的，一律当预告处理——计划改了它不会自己变。

### 跨仓那条线：已查，无遗漏，别再扫一遍

`data_integrity_open_items.md` 的诞生现场确实在 `D:\Codes\claude_windows_folder`
（2026-08-01 一次模拟审稿演练，对照臂 agent 翻了本仓 `paper/` 与 `offline_data/`）。逐条核过
该会话：它在代码层验的是「1000 回合数据集种子不相交」与 Table 1 transitions，**两条都落进了
文档**。其余跨仓命中（g3_5_5、Overleaf 等 15 个会话）全部是记忆文件里的引文，不是真跑过审计。
**结论：审计历史全在本仓的 git log ＋ rev 块 ＋ docs 里，跨仓无需再查。**

---

## 二、第一批已交付（第二批的交付物见 §四）

| 文件 | 内容 |
|---|---|
| `scripts/check_doc_code_refs.py` | 新校验器。补两类引用：源码行锚与文档声称的函数名——`check_doc_pointers` 在自己的路径解析函数注释里承认这两类它验不了。全仓 156 处引用，1.8 s |
| `tests/test_check_doc_code_refs.py` | 29 项 |
| `tests/test_check_doc_pointers.py` | 33 项。含把 `a507eb3` 那次**手工**故障注入固化下来的 SUSPECT 用例，以及钩子的接线用例 |
| `tests/test_build_doc_index.py` | 20 项 |
| `.claude/hooks/doc_pointers.py` | 改成串跑 sweep（第二批加挂刊值复算后为三个）。互补是构造出来的：第一个验路径在不在，第二个验行与名还在不在，第三个验数还能不能由源算出来 |

表中三份测试合跑 82 项全绿（29 ＋ 33 ＋ 20；钩子的接线用例在第二份里）。两批做完时全仓 `pytest tests/` 为 300 passed ／ 4 skipped。运行方式（本机沙箱挡系统
temp，必须给 `--basetemp`）：

```bash
python -m pytest tests/test_check_doc_code_refs.py tests/test_check_doc_pointers.py \
    tests/test_build_doc_index.py -q --tb=short --basetemp=<scratchpad>/pytest_tmp
```

### 负控是实测的：第一批 40 条 ＋ 第二批 49 条，全部变红

上一轮交接要求「每条检查都要配负控」。做法是对被测工具逐条注入故障、确认对应用例变红。
下表是第一批的判据本身，shell 脚本是一次性的（含硬编码本机路径，未入库）；第二批
`audit_published_numbers` 那 49 条同法做过（扩了 CSV / scale / 勘误三组机制后从 31 条补到 49，旧的 31 条一并重跑），判据见该脚本与
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
| 6 | 四条数字溯源链做成可重跑命令 | ✅（第二批当时）**4/4** 条链落表（FQL P2 35 ＋ td3bc phase0c 57 ＋ ReBRAC 85 ＋ arrival_v2 143 = **320 处刊值**，订正 1 格后全部吻合） |
| 7 | `docs/*.md` 报告侧的 ddof 口径工具化 | ✅（第二批当时）`--ddof` 四份报告合计 **80 处**：63 `ddof=0`、14 `ddof=1`、2 处两套都对得上、1 处 `NEITHER`（是报告自己声明的重号格）。ReBRAC 那 1 处 `ddof=1` 读数机械复现了 `48b8d06` 的头号结论 |
| 11 | 刊出读数 ⇄ manifest 指纹归属 | ✅ 已随第三批交付（`db0c235`）。钉住的是「报警是 `match == "none"`、不是等价类的大小」与「prefix 是通过」|

已交付：[`../../scripts/audit_published_numbers.py`](../../scripts/audit_published_numbers.py)
＋ [`../tracebacks/`](../tracebacks/README.md) 下的四份溯源表 ＋ 50 项测试（49 条注入判据全红，
均为第二批当时的读数；
其中两条注入的是**溯源表数据本身**而非代码）。已挂进 markdown 编辑钩子，串为第三个 sweep。

ReBRAC 那份还把该报告 §1 的**口径补注表本身**纳入核对——那张表印了逐种子值与两套口径，于是
「这份报告只有一处 ddof=1」这句话不再是被信任的，而是被重算出来的。

**arrival_v2 链落表时给工具加了三样**（都各自配了负控）：`.csv` 源与列聚合
（`mean(...)` / `max(...)` / `argmax(y, x)` / `count_gt(y, thr)` / `nrows(...)`，因为
mean39 / peak / 首达步 / n_succ 都是对 `eval_log.csv` 一列做的归约）；`scale`，只用于单位换算
（`peak @ 475k` 对的是 475002 步）；以及 `expect: "mismatch"`——见下。

**`expect: "mismatch"`：给「报告明知有错、又要留着」的格子用。** §7.9.4 保留了 2026-05-19 的
中间态列，下面用 ⚠ 注写明其中两格重号。对这种格写普通 claim 会**永远红**，而一条本该红的检查
很快就没人看。翻过来钉那条**声明**：它必须继续复现不出来，复现出来了才是缺陷（`erratum-stale`
桶，进 `--strict`）。更有用的是配对：同一格再写一条普通 claim 指向那条注**声称**它其实是什么。
`0.064` 那格就是——一条证明它不是 σ_final（两 seed 的 final 同为 0.900，σ 精确为 0，判 `NEITHER`），
一条证明它正是同两 seed 的 mean39 σ（`ddof=0`）。**那条勘误注的诊断本身现在是可重跑的。**

**这条链查出 1 格**：§7.5 `eval_path_length_m` 的 tandem 格原印 `66.99`，原始值 `66.99537`，
两位小数的正确舍入是 `67.00`——是截断不是舍入。已订正（§7.3 正文同格一并；顺带订正该处
「四组中最短」，§7.1 的 63.58 更短）。**这不推翻 `6380082`**：那轮逐格核的是 §7.6–§7.9 的
11 个 run，§7.5 的指标网格不在它范围内。

**两件本工具验不了、已写进 spec 的 `note`**：`6380082` 的头号发现（论文侧已改、报告侧未回填）
属表述层；OOB 列要 `eval_termination_counts` 的分项除以 `num_eval_episodes`，本工具只读平铺键与
CSV 列。同类还有 §7.9.4 勘误 ② ③（三 seed 的值排在标着「2 seeds」的列里）——工具能证「这个数
确实要三个 seed 才算得出来」，样本量与列标题的矛盾仍是人的活。

`offline_data/` 与 `results/`（以及 online 线的 `experiments/`）都是 gitignored 的，脚本一律
支持把根指到 Drive 挂载，**「本地扫不到」不能写成失败**（`no-data` 桶，不进 `--strict`）。

**第三批已完成，交付物见 §五。** 依赖面的实测分布与上一轮估的不同：**4 个需 `main.aux`**
（`ch5_floats` / `ch5_order` / `ch5_refs` / `ch5_check_all`）、**4 个需 `results/`**
（`ch5_holdout_split_audit` / `ch5_clean_probe_readout` / `ch5_manifest_attribution` /
`ch5_sac_ladder_dispersion_check`）、其余 4 个只读 `sections/*.tex`。测试一律用夹具合成这两类
输入，所以**不需要 latexmk、也不需要 `results/` 同步**。

### `.tex` 侧 ± 口径的剩余量——已量过，比表面数字小一半（2026-08-24）

`ch5_dispersion_audit.py` 报「90 处 ± 中 23 处能自解、67 处需要 ground-truth 报告」。那 67 处
不是 67 份工作量，实测拆开是：

| 分类 | 处数 | 说明 |
|---|---|---|
| 已被 `ch5_sac_ladder_dispersion_check.py` 覆盖 | **12** | §5.9 那批，本来就是回 `results/` 复算的 |
| 同一对 (mean, sd) 在章内别处能自解，可继承判定 | **24** | 24 处**无一**映射到两种口径，均唯一 |
| 真正需要新建溯源 | **31** | `boundary` 3 ／ `online` 10 ／ `rebrac` 11 ／ `td3bc` 7 |

那 24 处只手工核过一处，**其余 23 处是同形状的候选、未逐条核**。核过的是
`algo_compare.tex:214`：散文写「该配置格的均值降为 $0.742 \pm 0.198$」并显式
`\ref{tab:ch5_rebrac_screen}`，而 `rebrac.tex:516`（$\beta_1=1.0,\beta_2=2.0$ 行）印着同一对数
与逐种子值 `0.950 | 0.800 | 0.475`——连散文里「难例种子跌至 $0.475$」都与该行下划线值对上。
该格自解为 `ddof=0`，所以散文那处随之定了。

顺带一个跨侧一致性证据：`rebrac.tex:375 / 392` 的 `0.934 ± 0.026` 判为 `ddof=1`，正是
`48b8d06` 手工查出的「该报告唯一一处 `ddof=1`」，`.tex` 侧与 `docs/*.md` 侧对上了。

### 现在还剩多少，以及给 `.tex` 侧建链要知道的（2026-08-24 收盘）

上一节那句「31 处真需新建溯源」按工作量算是错的。31 处只对应 **20 个不同的 (mean, sd)**，
A（`online_a0` 链）与 C（ReBRAC §7.4 第三格）做完后的分层：

| 层 | 对数 | 读数 | 是什么 |
|---|---|---|---|
| 报告侧已有 `--ddof` 判定 | 15 | **21** | 直接可比，不需新建 |
| 值在有链的报告里但那行没 `sd` claim | 1 | 3 | `0.858 ± 0.080`（rebrac.tex:374/384、td3bc.tex:240）|
| 值只在无链的文档里 | 2 | 4 | `0.862 ± 0.019` 与 `0.870 ± 0.024`（均在 `docs/data_integrity_open_items.md`）|
| `docs/` 里没有任何文档印这一对 | 2 | 3 | `0.76 ± 0.22`、`0.88 ± 0.04`——即上一节那张 k 阶梯表，源在 `experiments/` 不在任何报告里 |

> **这张表已被下一节的 B 交付取代**（31 处现已全部落链）。留着它，是因为它记下了
> 一个会重复发生的坑：**本文档自己也是被扫的语料**。上面刚把 `0.76 ± 0.22` 与 `0.88 ± 0.04`
> 写进表里，分层脚本下一次跑就把它们从 `NONE` 挪进了 `DOC`——「有文档印这一对」的判据
> 被这份交接文档本身满足了。凡是按「值出现在哪份文档里」分层的脚本都有这个毛病，
> 判据要排除 `docs/handoff/`，或者只认有溯源链的文档。

`0.858 ± 0.080` 那一格值得单独说：它是 **TD3BC 的数**，被引进 ReBRAC 报告作对照，娘家是
`docs/td3bc_worldcomp_teacher_gap_experiment_report.md:243`（那里 mean 与 sd 分列两栏）。
那份报告**没有链**，所以它不是「补一条 claim」，是「要不要建第六条链」。

**给 `.tex` 侧建链的可行性（已核，非推测）**：`scripts/audit_published_numbers.py` 对 `doc`
字段**没有 `.md` 硬要求**——它只按行读文件。不能用的只有 `section` 作用域，那个绑死在
`HEADING = re.compile(r"^(#{1,6}) ")` 上。`after` / `before` **可以脱离 `section` 单独使用**：
`after` 设下界、`before` 设上界，两者都在当前区间内解析且必须唯一命中。`.tex` 的表格行本身
够独特，`anchor` 通常单独就能唯一。`online_a0.json` 就是这么写的（两张同构表靠上方加粗标记
夹住），可作范本。

顺带记一处死代码：`_region()` 末尾的 `if lo >= hi: raise SpecError("empty scope")` **不可达**
——`after`/`before` 的命中都被限制在 `lo < i < hi` 内，只能收窄且始终保持 `lo < hi`。无害，
但别为它写测试（写了也红不了）。

### 两侧对照查出来的（2026-08-24）

按 anchor+capture 把 `.tex` 侧未决读数指回同一批源之后，31 处的分布是 **20 个不同的
(mean, sd)**，其中 14 处（8 对）报告侧本就有判定。对照两侧查出两件事。

**一 · `docs/data_integrity_open_items.md` 的干净集 `cross-2000` 一格印错了一位（已订正）。**
原印 $0.870\pm0.025$；逐种子 $[0.89, 0.87, 0.86, 0.83, 0.90]$ 的总体标准差是 $0.024495$，
三位舍入为 $0.024$，与真值差 $0.000505$，**超出**「半个末位」的可接受舍入。论文侧
`rebrac.tex` 一直印 $0.024$，故漂的是报告侧。成因：`ch5_clean_probe_readout.py` 的冻结期望
**只钉均值**，而均值本来就是对的。已订正该格、把四格的离散度一并纳入冻结期望，并补两条用例
（离散度漂移、口径被换成 ddof=1）；负控 3 条全红。

**二 · `docs/arrival_v2_experiment_report.md` 同一段里的两个 `σ_final` 是两套口径（未动）。**
§5.5 k 阶梯表的两行在 `docs/` 里搜不到，回 `experiments/arrival_v2_prototype/*_summary/
combined_gate_summary.json` 复算后对上了：

| 格 | 逐种子（42/0/7） | 均值 | ddof=0 | ddof=1 | 章节印 | 报告印 |
|---|---|---|---|---|---|---|
| k=8 | 0.900 / 0.500 / 0.867 | 0.756 → `0.76` | 0.181 | 0.222 | `± 0.22` | `σ_final = 0.181` |
| k=12 | 0.900 / 0.900 / 0.833 | 0.878 → `0.88` | 0.031 | 0.038 | `± 0.04` | `σ_final = 0.038` |

同段的 `mean39` 两处 σ（`0.192` / `0.113`）复算亦均为 ddof=1。**所以那段里唯一的异类是
k=8 的 `σ_final = 0.181`**，章节的 `± 0.22` 反而与报告自身的主导口径一致。两个数都没错，
错的是同一个符号在一段里指两套口径。**没有动它**——改报告正文属表述层，且要连带核 §7.9 其余
gate 读数用的是哪套，宜单独一轮。

> **订正（同日稍晚，做 B 时按文件逐格对照后）**：上面这句「未披露的口径混用」判错了。
> 该报告 **紧接在 843 行下面的 846–847 两行就自己披露了这件事**，并写出两套自洽配对：
> 「一致口径下应为 `0.222 → 0.038`（ddof=1）或 `0.181 → 0.031`（ddof=0）」，还指明
> §7.9.4 第三列用的是前者。论文侧 k 阶梯整条是 ddof=1，与 §7.10 表头自述口径、与该报告
> 推荐的自洽配对都一致。**因此「报告侧 σ_final 口径统一」不是待办，报告侧没有欠账**；
> 此前把它列为待办，是只读了 843 行、没往下读两行。

**未决**：要不要让 `ch5_dispersion_audit.py` 直接继承判定。**没有动手，这是用户的决定**——
该工具的设计原则就是「猜比不答更坏」（候选种子必须能复现刊出均值才被采纳），而 (mean, sd) 碰巧
撞车就会产出错判。做成提示桶也有代价：不进 `--strict` 的那一档，工具判错时没有任何东西报警
（这一条本仓吃过亏）。倾向是**先把那 31 处按 anchor+capture 建表**——那是无歧义的部分——继承
那 24 处留作单独议题。

重算这三个桶的一次性脚本在临时目录（`scope_tex_side.py` / `scope_crossref.py`），
不入库；它们只调 `ch5_dispersion_audit.collect()` 与 `ch5_sac_ladder_dispersion_check` 的两个
常量，重写一遍比找回来便宜。

**仍未做**：C 类第 12 项（`benchmarks/` 落库逐条核，出处 `0dc35a2`，纯手工）与 C 类第 10 项
（文档自述横幅 ⇄ 引用它的表格状态栏，出处 `cce2b2d`，13 行核出 3 行对不上）。两项都未分配批次。

---

### B 已交付：论文 `.tex` 侧四条链（2026-08-24）

31 处全部落链，`docs/tracebacks/` 由五条增至九条、刊值由 346 处增至 **408 处**，
`--strict` 吻合 408、缺陷 0、缺数据 0。`--ddof` 由 93 处增至 **124 处**
（99 `ddof=0` ／ 20 `ddof=1` ／ 4 `either` ／ 1 `NEITHER`）。

| 表 | `doc` | `root` | claim | 覆盖 |
|---|---|---|---|---|
| `ch5_online.json` | `sections/online.tex` | `experiments` | 20 | §5.5 传感表六格 ＋ 瓶颈 k 阶梯四行 |
| `ch5_boundary.json` | `sections/boundary.tex` | `experiments/arrival_v2_prototype` | 6 | §5.8 引的 k 阶梯两格（表 ＋ 散文） |
| `ch5_rebrac.json` | `sections/rebrac.tex` | `results/offline` | 22 | §5.6 TD3+BC 对照格、干净集两格、caption 那格 |
| `ch5_td3bc.json` | `sections/td3bc.tex` | `results/offline/td3bc/phase0c` | 14 | §5.4 数据规模六格 ＋ teacher-gap 可部署格 |

**四条而不是一条，是被工具形状逼出来的**：一份 spec 只有一个 `doc`、一个 `root`。31 处分布在
四个 section 文件里，四是下限。每条的 `root` 取该文件全部源的最近公共祖先——`online.tex` 同时
引 A0 与 k 阶梯两棵树，`rebrac.tex` 同时引 ReBRAC 与 TD3+BC 两棵树，所以这两条的根比对应报告链
高一层。glob 字符串因而差一段前缀，落到的文件相同。

**没有继承报告侧的判定。** 31 处里 21 处的同一格在报告侧已有 `--ddof` 结论，仍各写各的 claim。
理由是 `0993832`：两侧真的会漂，照抄会把漂掉的一侧固化成「一致」。

**跨侧对照结果：17 格两侧都刊，17 格口径一致，0 处相左。** 判「同一格」按**解析后的文件集合**，
不按刊值文本——文本相同的巧合真实存在（分层脚本就把 `0.967 ± 0.027` 同时匹到了 A0 的两个臂）。
另有 3 格只有论文侧有 claim：干净集两格与 TD3+BC 的 teacher-gap 格，娘家文档均无链。

**三处 provenance 规则是复算出来的**：k 阶梯 k=8／k=12（任何报告都没印过）、`0.858 ± 0.080`、
干净集两格。k 阶梯四行规则一致——逐种子 `final_eval.json`、mean ± 标准差；同批 run 的
`eval_log.csv` 上 `mean`／`max`／`last` 无一对得上（k=12 的 `max` 给 0.90，刊值 0.88）。
`0.858` 那格两个候选目录（`test_selected/alpha_0p0` 与 `test_bc_selected/alpha_0p0`）实测同值，
印证了 worldcomp 报告第 267 行的自述；同层 `privileged_final` 的 `0.922 ± 0.086` 复现，
作为「目录结构没读错」的独立旁证。

**取数按完整数值格走，不按 `\pm` 计数。** §5.6 有一句一行印四格。不能改成数 `\pm` 次数：
`tab:ch5_rebrac_perseed` 的 caption 里有一个「均值 $\pm$ 跨随机种子标准差」的**裸 `\pm`**。

**四条新测试，负控 15 条全红**（每条只对**一条指名的测试**判红，不拿别的测试的红顶数）：

| 测试 | 钉住什么 | 负控 |
|---|---|---|
| `..._capture_one_whole_published_cell` | 取数落在某一个完整格内，且 mean 与 sd 落同一格、同行两条不重格 | 3 |
| `..._captures_depend_on_the_cell_they_count_to` | 格序号承重：去掉它必须取到不同的数 | 2 |
| `..._read_the_files_the_reports_read` | 两侧解析到同一批文件；无对应的必须在 `UNSHARED_WITH_THE_REPORTS` 里逐条声明（声明过期也报） | 4 |
| `..._ladder_is_pinned_to_the_terminal_evaluation` | 阶梯的源形状是 `final_eval.json`，不是训练曲线 | 2 |
| 既有 `..._shipped_specs_still_anchor_to_their_reports` | 四条新链一并进它的扫描 | 1 |
| （数据侧 `--strict`） | 刊值被改、格序号成对滑动 | 3 |

**一处覆盖边界，实测出来的**：格序号若**成对**滑到同一行上**没有被任何 claim 钉住**的那一格，
data-free 的四条测试全部保持绿，只有数据侧复算报 `value-mismatch`。§5.6 那句四格里只钉两格，
正是这种行。所以两层不是冗余——clone 上只有前一层。

## 五、第三批已交付（2026-08-24）

`paper/thesis_ch5/tools/` 的 12 个脚本全部落测：**131 项用例、125 条注入判据全红**，
`pytest tests/` → 431 passed, 4 skipped——**均为第三批收尾当时的读数**。其后 `0993832` 又给
`ch5_clean_probe_readout.py` 加了离散度守卫（用例 +2、注入 +3），这七个文件现为 133 项。
七次提交：

| 提交 | 覆盖 | 用例 | 注入 |
|---|---|---|---|
| `cc94ba0` | `_ch5_corpus.py` 九条口径 ＋ `ch5_metrics.summarise` | 19 | 20 |
| `b87910b` | `ch5_floats` / `ch5_order` / `ch5_refs` ＋ `ch5_check_all` 编译门槛 | 26 | 21 |
| `5344e76` | `ch5_dispersion_audit.py` | 17 | 18 |
| `8586abf` | `ch5_sac_ladder_dispersion_check.py` | 15 | 15 |
| `db0c235` | `ch5_manifest_attribution.py`（C 类第 11 项）| 19 | 17 |
| `65f68ac` | `ch5_holdout_split_audit.py` ＋ `ch5_clean_probe_readout.py` | 18 | 20 |
| `925d9c4` | `ch5_lexcheck.py` | 17 | 14 |

**测试一律打在夹具上，不打在真章节上。** 唯一碰 `sections/` 的那条只断言结构不变量，
**不断言任何计数**——版面与计量数字随任意一次文本改动失效（2026-07-08 那句「全部浮动距首引
0–2 页」被沿用三轮、到 07-28 实测四处越线），spec §0.5.8 本就禁止预设量化目标。写一条
「198 段」的断言就是把同一个缺陷搬进测试里。`main.aux`、`main.log`、pdftotext 的分页、
`results/` 逐 seed 读数，全部由夹具合成。

**查出并修掉一个缺陷**（`b87910b`）：`ch5_floats.py` 判首引页用子串包含，而本章 21 张表、
15 张图，「表 5.1」是「表 5.10…5.19」的前缀。首引取 `min()`，只会把页码往前拽——距离被
撑大成假越线；标题页排在真首引之前时反而会盖掉真越线。已改为带数字边界的匹配。

**对当前章节零影响，这一条是实测的**：本地 latexmk 构建一次，用 HEAD 版与修后版跑同一套
产物，36 个浮动逐行 diff 为空。原因有结构性——编号序跟着首引序走时，编号更长的浮动必然
引用在后，`min()` 取不到它。**是 `ch5_order` 的零逆序判据一直在替 `ch5_floats` 掩着这个
bug**，也就是说它恰好会在 `ch5_order` 失败时浮出来。

顺带记下本次构建的实际读数（README 里那组停在 2026-07-28 的数字已被超越，**不要引用历史
值**）：65 页、浮动 36 个、Overfull 2 处（3.13pt／4.31pt，均在 10pt 判据内）、交叉引用
49 个、禁用词条 88，`ch5_check_all.py` 总判定 PASS。查完即 `latexmk -c`，`main.pdf` 还原
为已提交版本（重建只差压缩流里的时间戳）。

---

## 六、方法论：三批累计抓到的十个「测试跑绿但不承重」

第 1–7 条来自第一、二批，列在本节；第 8–10 条来自第三批，列在 §七。

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
7. **断言只盯一个桶，出错的东西落进别的桶时照绿。** 第二批新写的 CSV 用例只断言
   `buckets["value-mismatch"] == []`——可是一条**在取数阶段就死掉**的 claim 落进 `spec-error`，
   那个桶同样是空的。注入「聚合分派表里把 `max` 改名」实测 GREEN，抓出来的不是机制没用，是断言
   指错了地方（与第 4 条同源）。改法是写一个 `assert_reproduced()`：断言 `ok` 恰好一条，**且其余
   每个桶都为空**。判据：**「没报错」不等于「算对了」——正面断言那条 claim 真的走完了全程。**


---

## 七、第三批新增的三条方法论

**8 · 工作副本是 CRLF 时，`sed -i` 让「突变是否落盘」的校验和自检结构性失效。**
第二批给注入脚本加的第二道自检是「比对文件校验和，没变就报 NOOP」。但 `paper/thesis_ch5/tools/`
下的源码在工作副本里是 CRLF，而 Git Bash 的 `sed -i` 一律写回 LF——于是**模式匹配不上时校验和
照样变**。叠加上模式本身写错（反斜杠经「单引号 → `bash -c` → sed BRE」三层，写成了 16 个、
源码里只有 2 个），一条什么都没改的注入被判成 GREEN，读起来像「机制不承重」。
改法：指纹先 `tr -d '\r'` 再算。改完当轮就见效——mutate10 的三条引号写错的注入被正确报成
NOOP 而不是 GREEN。

推论比这条本身更重要：**自检也要有判别输入**。一道「文件变了没」的自检，在一个总会改写整份
文件的工具面前是恒真的。

**9 · 夹具照着被测常量去建输入时，改那个常量的注入本就不可观测。**
`ch5_sac_ladder_dispersion_check.py` 用 `XSOURCE` 绑定「正文那句话背后是哪个 run」。夹具原本
写的是 `seeds(h2h, ladder.XSOURCE, R1)`——注入改 `XSOURCE`，夹具跟着改，照绿。这不是机制不
承重，是夹具在跟着被测对象走。判据：**绑定是契约，不是夹具参数**（绑错等于把一个数算到别的
实验头上），所以夹具里把该路径写死。写死一个常量通常是坏味道，这里是例外，理由要写在用例里。

**10 · 对称输入会让不对称的公式看不出来。**
`welch()` 用 Satterthwaite 自由度。夹具原本给两组同样的离散度、同样的样本量——而**在等方差、
等 n 时 Satterthwaite 恰好等于 n1+n2−2**，于是「换成合并自由度」这条注入完全不可观测。改成
两组离散度不同后变红。这是第 4 条（守门机制在干净输入下测不出来）的数值版：**干净不只是
「没有缺陷」，也包括「太对称」**。
