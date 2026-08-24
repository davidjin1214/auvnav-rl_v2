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
| 10 | 文档自述横幅 ⇄ 引用它的表格状态栏 | `cce2b2d`：13 行核出 3 行对不上 | ✅ 已固化为 `check_status_claims.py`，见 §八 |
| 11 | 刊出读数 ⇄ 评估 manifest 指纹归属 | `ch5_manifest_attribution.py`，3715 份读数零落空 | ✅ 已并入第三批（`db0c235`，19 项用例）|
| 12 | `benchmarks/` 落库时逐条核（条数／种子连续／区间） | `0dc35a2`，纯手工 | ✅ 已固化为 `check_benchmark_manifests.py`，并与 §2／§3／§5 并作一批，见 §九 |

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

~~**仍未做**：C 类第 12 项（`benchmarks/` 落库逐条核，出处 `0dc35a2`，纯手工），未分配批次。~~
⚠ **2026-08-24 已交付，见 §九**；删划线保留只为留痕。
（同段原先并列的第 10 项亦已于同日交付，见 §八。）

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

### 这条线现在还欠什么（2026-08-24 收盘）

⚠ **本小节写于第 12 项交付之前，其「只剩第 12 项」现已过期——12 项已全部交付，见 §九。**
以下保留原文。审计考古 12 项里**只剩第 12 项**。第七条链、第 6/7 条链与生成器入库的注入判据、三处过期计数，
都在 2026-08-24 当轮收掉了；下面记的是剩下那一项，与收的过程中新查出来的两件事。

| 欠项 | 出处／规模 | 说明 |
|---|---|---|
| C 类第 **12** 项 | `0dc35a2` | `benchmarks/` 落库逐条核（条数／种子连续／区间）。纯手工，**未分配批次**——12 项里唯一剩下的 |

**第七条链已交付。** [`docs/tracebacks/data_integrity_open_items.json`](../tracebacks/data_integrity_open_items.json)，
**32 处**刊值，由 `_gen/gen_data_integrity_spec.py` 生成。覆盖 ①-c 干净集读数整段：四格
均值与离散度、两格位移、翻转两格、采集器基线三格加回合数，以及 2026-08-24 那条订正块。
provenance 规则**是复算确立的，不是从 `ch5_clean_probe_readout.py` 读来的**——那支工具正是
`0993832` 里因为只钉均值而漏掉离散度漂移的那一支，照抄它的路径等于继承它的盲点。
已刊列出自 `rebrac/formal/<dataset>/actorb_4p0__criticb_2p0/test/seed_*.json`，干净集列出自
`rebrac/clean_probe/cross-*/seed_*.json`，各五种子、`ddof=0`（cross-1000 那行两套口径可分：
`ddof=1` 会给 0.024，刊的是 0.021）。

**其中两条是勘误 claim**，钉的是订正块本身：`$\pm0.025$` 必须继续复算不出来。它的余量是
**0.000505 对半个末位 0.0005**——全仓最薄的一处，而这正是要的：逐种子读数哪天重生成、哪怕只
挪一点点，这条勘误就会翻，订正注也就该重读。工具明确留在链外的：全部 t／SE／95% CI（不是本
工具算得出的统计量）、差中差 `+0.80` pp（要四个源，`delta` 只收两个）、`0.000505` 本身
（它是「到一个错值的距离」，不是任何东西的复算），以及 §2／§3／§5——那几节的源在
`offline_data/` 与 `benchmarks/`，是另一族 provenance，正对着还没做的第 12 项。

**注入判据补齐：17 条，全部对着各自那一条指名检查变红**（脚本在会话 scratchpad，见 §四末）。
分两层判，因为这两条链本来就有两层守：

| 检查 | 钉住什么 | 注入 |
|---|---|---|
| `test_every_committed_spec_can_be_regenerated` | 手改 JSON、生成器不认 `out_dir` | 2 |
| `test_every_spec_declares_whether_it_has_a_generator` | 漏登记、登记指向不存在的生成器 | 2 |
| `test_a_spec_declared_ungenerated_is_not_emitted_by_any_generator`（新） | 假的「无生成器」声明 | 1 |
| `test_no_two_report_table_claims_read_the_same_cell_of_a_row`（新） | 同一行两条 claim 读同一格 | 2 |
| `test_the_report_table_chains_index_a_cell_that_carries_weight`（新） | 相邻格印同一个数（格序号从此不承重）、受控名单被摘空 | 2 |
| 既有 `..._shipped_specs_still_anchor_to_their_reports` | 锚点改到文档里找不到 | 1 |
| 既有 `..._read_the_files_the_reports_read` | 源改指另一格、复活一条已失效的 UNSHARED 声明 | 2 |
| （数据侧 `--strict`） | 刊值被改、勘误开始能复现、delta 源对调、去掉 pp 换算、第 6 条链格序号滑一格 | 5 |

**两条负控头一轮没变红，两次都是「被第二个机制兜住」——第 12 条方法论又中一次。**
一条是假声明那条：新测试原本只跑 `SPEC_GENERATORS` 里登记的生成器，而假声明恰恰是把生成器
从那张表里摘掉，于是它根本不会被跑到。改成跑 `_gen/` 下**全部** `gen_*.py`。另一条是
`_anchored` 这个测试辅助函数：它按全文匹配锚点，而工具是在 `section` 作用域内匹配；worldcomp
报告两张筛选表的 `| 0.1 |` 行一模一样，新测试一挂上就报「锚定到两行」。改成复用工具自己的
`apn._region`。

**顺带查出两条早就够不着的声明。** `UNSHARED_WITH_THE_REPORTS` 里 k 阶梯的
`s0_k8`／`s0_k12` 两条，理由写的是「任何报告都没印过这两行」——那句是真的，但那张表的判据是
**文件**有没有被报告链读到，而 `arrival_v2` 链为了 §7.9 的 σ_final 一直在读同样两个 glob，
`or` 在第一个分支就短路，两条声明从来没有被求值过。已撤下，那张表现在是空的；反向核对
（被声明为「没有报告读」而其实有报告读的要报错）加进了 `..._read_the_files_the_reports_read`。
**一条永远够不着的声明和一条错的声明一样危险**：它读起来比它能承担的更强。

**过期计数已改。** [`../tracebacks/README.md`](../tracebacks/README.md) 与
[`../../CLAUDE.md`](../../CLAUDE.md) 现写**十一条链、476 处**；README 新增「表是生成出来的」
一节写明 `_gen/` 的契约（`python docs/tracebacks/_gen/gen_*.py [out_dir]`，改生成器别改 JSON），
`metric` 的点号路径（`best.eval_success_rate`）写进了脚本 docstring——字段语义的唯一权威处
在那里，README 自己那条规矩就是「别在这里复制一份口径」。`--ddof` 合计由 124 变 130。

**单侧钉住的格：清零。** 两份娘家文档都有链之后，论文侧不再有任何源是报告侧读不到的。

**注入脚本在哪。** ✅ **2026-08-24 已入库，下面这条 scratchpad 路径只作出处、不再是唯一副本**
——引擎并入 [`../../scripts/mutation_probe.py`](../../scripts/mutation_probe.py)，行表并入
`scripts/mutations/`（本轮这批即 `tracebacks_chains.py` 17 行），见 [§十](#十注入器已入库2026-08-24这条线欠自己的那笔)。原路径：

```
C:\Users\jinxiang\AppData\Local\Temp\claude\D--Codes-rl-v2\1caf0b2d-afc5-4232-b554-9bd191702036\scratchpad\
    mutate_chains.py     本轮 17 条（溯源链 + 生成器入库）；两种判据层：pytest 用例 id，与
                         audit_published_numbers 的缺陷桶（配 bucket_probe.py 逐桶计数）
    mutate_status.py     上一轮 21 条（check_status_claims / build_doc_index）
    bucket_probe.py      把全部 spec 跑一遍、按桶计数输出 JSON，供 sweep 层判红
```

`mutate_chains.py` 比之前几支多两样，下一批照它改：edit 支持 `("claim", 表, 标签, 键, 值)`
直接改 spec 的某条 claim（比字面替换稳，JSON 重排后格式与生成器一致），以及一行可以带多处
edit（假声明那条要同时动两张表才构造得出来）。

**这件事本身值得注意**：审计固化做了四批，注入脚本重写了四遍，每一支都随会话消失——而这条线的
立论恰恰是「每次都没留下能再跑一遍的东西」。要不要把它收进 `scripts/` 或 `tests/` 变成一支
带参数的通用注入器，是一个还没拍板的决定，规模上已经够了。
→ ✅ **2026-08-24 已拍板并落地，见 [§十](#十注入器已入库2026-08-24这条线欠自己的那笔)。**

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

---

## 八、C 类第 10 项已交付（2026-08-24）

**结论和预期相反：出错的不是引用方，是横幅提取器。**

`cce2b2d` 当年手工核 `online_rl_line_summary.md` §3.2 的 13 行，判「表格状态栏 ⇄ 被引文档自述
横幅」，3 行对不上。本轮把它做成命令，先照原样复核了一遍——结果查出来的五处「相左」，
**五处都是表格写对了、`build_doc_index.py` 的 `status_of()` 读错了**：

| 文档 | 索引里曾写 | 它自己其实写的 | 那个关键词的真正主语 |
|---|---|---|---|
| [`../rebrac_experiment_plan.md`](../rebrac_experiment_plan.md) | DEPRECATED 2026-05-08 | 无自我标注（rev.8） | 它开篇让人先读的 `offline_rl_implementation_plan.md` |
| [`../rlpd_design.md`](../rlpd_design.md) | DEPRECATED 2026-05-08 | 无自我标注 | 同上 |
| [`../rebrac_mainline_review.md`](../rebrac_mainline_review.md) | SUPERSEDED | 无自我标注（rev.4） | 「v1 广验全套」那批实验 |
| [`../rebrac_broad_validation_v2_plan.md`](../rebrac_broad_validation_v2_plan.md) | SUPERSEDED 2026-05-04 | ✅ **PASS** | 「取代的 v1 文档（已加 SUPERSEDED banner）」 |
| [`../fql_succession_p2_main_spec.md`](../fql_succession_p2_main_spec.md) | SUPERSEDED 2026-05-20 | **CLOSED (2026-05-23)** | 它自己那些「已逐节标注 SUPERSEDED」的章节 |

**35 份自我标注的文档里错 5 份（14%），而 [`../DOC_INDEX.md`](../DOC_INDEX.md) 是已提交的生成物、
CLAUDE.md 让读者用它定位文档**——每一处都把一份活文档标成了死的。`status_of()` 原有的两道
护栏（只读 H1 后的引用块、关键词须带强调或 ⚠）都拦不住它：关键词**确实在横幅里**，只是主语
是别人。新增 `_self_declared()`，三条信号各管一种形状（行内更早出现链接／`.md` 文件名；关键词
落在未闭合的括号里；关键词前有超过 20 个字符的正文）。改动只减不增：30 份判对的一处未动，
第 6 处 `online_rl_line_summary.md` 由 CANCELLED 2026-07-28 改为「未自我标注」——那句「已撤销」
说的是 standalone paper，文档本身是在线线的活跃入口。

### 交付物

| 文件 | 内容 |
|---|---|
| `scripts/check_status_claims.py` | 新校验器，两条规则，0.8 s 扫全仓 |
| `tests/test_check_status_claims.py` | 22 项用例 |
| `tests/test_build_doc_index.py` | +10 项（`_self_declared` 三条信号各一条隔离用例 ＋ 四种标签写法的正控 ＋ 换行归一化） |
| `.claude/hooks/doc_pointers.py` | 挂成第 4 个 sweep——**编辑文档横幅正是引用它的表格变陈旧的那一刻** |
| [`../rebrac_broad_validation_v2_report.md`](../rebrac_broad_validation_v2_report.md) §7.1 | 五格 `TBD` 按实况改写（下详） |
| [`../DOC_INDEX.md`](../DOC_INDEX.md) | 重建，6 行状态变化 |

负控 **21 条，全部对着各自那一条指名用例变红**（脚本在会话 scratchpad，见 §四末）。

### 两条规则，以及为什么只有两条

**R1 交叉声称**：表头命中闭合词表（`状态`／`Status`／`state`）的列里，主语格就是一份仓内
markdown 的那些行，其状态格不得与该文档自述的横幅相左。全仓 20 行可比，现 16 吻合、4 未提。

**R2 预告占位**：进度列里整格只写一个预告词（`TBD`、`第三批`、`待定`…）的，是预告不是状态。
预告永远不会自己变成假的，所以没人回来看它。

**第三类没做，是量过之后决定不做的。** 用户给的判据 (a)「状态标记跨节引用正文的，按被引那节
现在说什么核」在这里**不可机械化**：全仓 61 处状态格带 `§` 指针，61 处全部解析得通——因为其中
绝大多数指的是**论文章节**（`.tex` 里的 `§5.5`），根本不是本文档的小节。按「§X 必须是本文标题」
建规则会把这 61 处全判错。该判据留在 §一 表下作人工判据，不进工具。

**「扫所有 ✅/⬜/⚠」也量过：全仓 117 张表、701 个状态格。** 绝大多数是**诚实的历史快照**
（「两批做完时全仓 300 passed」这种），报出来只会把真漂移淹掉。收窄是判据的一部分，不是省事。

三种假阳性形状，都是实测出来的，各有一条用例钉住：

- **状态栏评的是动作不是文档**：`✅ 已完成` 说的是「给它加横幅」这件事。判别式=主语格必须
  以链接开头、且去掉链接后残余 ≤ 24 字符（实测 20 个真候选残余最大 19，是 `(rev.5, 2026-04-22)`）。
- **列名带 ✅/❌ 但不是状态**：`持数字 ground truth？` 那一列的 ❌ 意思是「本文不持数字」。所以
  状态列**按表头识别，不按单元格多数决**。
- **裸 `TBD` 不在进度列**：表头 `Expected σ_final ↓` 下的 `TBD` 是「文献上没这个数」，表头
  `Notebook` 下的是「文件还没起名」。两处都在仓里。

R2 的列域比 R1 宽（多收 `本轮`／`处置`／`进度`），因为历史上那一格 `⬜ 第三批` 正是长在 `本轮` 底下。

### 自述冻结：声明不是事实

[`../rebrac_broad_validation_v2_plan.md`](../rebrac_broad_validation_v2_plan.md) §11.1 的 11 处 `TBD`
**不是缺陷**——该文 2026-07-28 在表格正上方写了「本注只记录落地实况，**不改本表任何一行**」，
并逐条说明了哪些文件未创建、哪些是 M1 未触发所以本就不该存在。工具照 `check_doc_pointers`
的「自述缺席」桶同样处理：声明须落在该表与上一个小节标题之间，且**原文照打进报告**——
一条读不到的声明是没法复核的。和那个桶一样的风险也一样在：**假声明会永久掐掉自己的告警**，
搬动或更新那张表时要一并复核。

### 真正修掉的五格

[`../rebrac_broad_validation_v2_report.md`](../rebrac_broad_validation_v2_report.md) §7.1「本 report
触发的下游修改」五行全写 `TBD`，而五项**早就落地了**，逐项复核：plan 顶部横幅已写 PASS 且 §12
清单逐项 `[x]`；line summary §5.2 该行已写「active（3-seed 收口 2026-07-12）」；mainline_review
§3.5 标题已带「v2 PASS 2026-05-19」。另两行落地了但**形式与原计划不同**，一并写进状态格：
paper_writing_index 那行——独立投稿 2026-06-02 撤销后已无 §experiments／§discussion 之分；
`auv_paper/` 那行——那是起草当时的**原计划名，git 里从未存在过**，LaTeX 实际长在 `paper/`，
`e6b729d`（2026-08-16）归档进 `paper/archive/rebrac_standalone/`，素材现落在论文 §5.8。

### 本轮新添的三条方法论（接 §六、§七）

**11 · 比对两个读数之前，先确认两边的读法都对。** 本项差点把五处「表格错了」当成结论报出去。
先手工核了被引文档的横幅原文，才发现错的是提取器。**判据：一个比对工具的两侧，谁都可能是
坏的那侧；先各自验，再比对。**

**12 · 被第二个机制兜住的机制，负控判不出来。** 首轮 20 条注入有 2 条「仍然绿」，不是机制不
承重，是那条用例同时被两个机制保护着（拆掉链接判定，括号判定接住了）。补一条**只由待验机制
把守**的隔离用例才判得动。同一轮还查出两处**永远执行不到的死分支**（`BANNER_LEAD`、R2 里的
`idx == subj`），都是这么暴露的——**注入判不动的分支，先怀疑它根本够不着**，删掉比留着好。
`FORECAST_ONLY` 的首尾锚定则是第三种：`^` 与 `.match` 重复，`$` 与 `.match` 各管一端，拆任何
单独一个都拆不动，于是删掉冗余的 `^`，两端各配一条用例（一格以批次名结尾、一格以批次名开头）。

**13 · 走管道才现形的崩溃，直接跑永远绿。** 新 sweep 挂进 hook 后立刻把全仓 markdown 编辑
堵死了：hook 用管道抓 stdout，Windows 上管道默认 cp936，而 `⚠`（U+26A0）在 cp936 里编不出来，
脚本报告打到一半崩掉、hook 把崩溃读成了发现。三个既有 sweep 都在 `main()` 头上
`sys.stdout.reconfigure(encoding="utf-8")`，新的漏了。**判据：凡是会被 hook 或 CI 用管道抓的
脚本，都要有一条真的走子进程管道的用例**，本仓这条是 `test_the_report_survives_a_non_utf8_stdout_pipe`。


---

## 九、C 类第 12 项已交付（2026-08-24）——12 项至此全部收口

范围按用户指定：第 12 项（`benchmarks/` 落库逐条核）＋
[`../data_integrity_open_items.md`](../data_integrity_open_items.md) 明确留在第七条链之外的
§2／§3／§5，合为一批，理由是同一族 provenance——源在 `benchmarks/` 与 `offline_data/`。

**做完之后这三族分了家，分法是被工具形状逼出来的、不是设计出来的**：

| 族 | 谁核 | 为什么不能合 |
|---|---|---|
| `results/` 逐 seed 读数 | 第七条链 `data_integrity_open_items.json` | — |
| `offline_data/` 采集侧 metadata | **新的第十二条链** `benchmarks_and_datasets.json` | 一份 spec 只有一个 `root` |
| `benchmarks/` 清单的条数与种子区间 | **新工具** `check_benchmark_manifests.py` | 那不是任何读数里的一个 metric，`audit_published_numbers` 没有源可取 |

### 交付物

| 文件 | 内容 |
|---|---|
| `scripts/check_benchmark_manifests.py` | 新校验器，四条规则，0.8 s 扫 24 份清单 ＋ 全仓 markdown |
| `tests/test_check_benchmark_manifests.py` | 28 项用例 |
| `docs/tracebacks/benchmarks_and_datasets.json` ＋ `_gen/gen_benchmarks_datasets_spec.py` | 第十二条链，16 处刊值 |
| `tests/test_audit_published_numbers.py` | ＋1 项（千分位逗号）、新链登记进 `SPEC_GENERATORS` |
| `scripts/audit_published_numbers.py` | `_norm` 现在也去掉千分位逗号——转移条数是这批文档里唯一的六位数，印作 `152,683` |
| `tests/test_check_doc_pointers.py` | 钩子接线用例扩到五个 sweep |
| `.claude/hooks/doc_pointers.py` | 挂成第 5 个 sweep |
| `benchmarks/README.md` | 新增「Checking this directory」一节 ＋ 截断修复记录 |
| [`../data_integrity_open_items.md`](../data_integrity_open_items.md) §2 | 按实况闭合（下详） |
| `paper/archive/rebrac_standalone/outline.md` | 一处计划值加日期声明 |

负控 **34 条，全部对着各自那一条指名检查变红**（脚本在会话 scratchpad，见本节末）。
分两层判：`pytest` 用例 id，与两支 sweep 的缺陷桶（`bucket_probe2.py` 逐桶计数）。

### 四条规则，以及为什么是四条

`0dc35a2` 的提交信息把验收写成了散文：「落库前逐个核过——条数与目录名相符、种子连续、其中
**8 个**与审计打印的区间逐条对上、val_40 确为 test_100 的前缀」。那次是手工、只核了 13 份里的
13 份，另外 11 份没人看过，且没留下能再跑的东西。四条规则就是把那句话变成命令，外加一条它
没想到、但正是本轮抓到缺陷的那条。

- **R1 结构自洽**：种子连续；`episode_id` 序号与位置相符（这一对正是
  `ch5_manifest_attribution.py` 拿 3715 份读数做归属的指纹）；路径声称的条数（`{val,test}_{N}/`
  目录或 `_ep{N}` 文件名）必须成立；首个种子等于生成器会用的那个——`BENCHMARK_SPECS[key]
  .manifest_seed`，或者文件名带 `_s{N}` 重播种标记时等于 `N`。**标记不是豁免，是把期望挪个位置**：
  `clean_probe/..._s3000.json` 照样被核，只是核的是 3000。
- **R2 族内嵌套**：同一任务配置（流场／几何／目标速度，就是 `audit_seed_overlap` 判「两者可比」
  用的那个三元组）＋同一起始种子的清单必须嵌套，短的是最长那份的开头一段。这是「每个 val_40 都是
  同级 test_100 的前缀」的机器形式，也顺带覆盖了 epoch_probe 那对「test 与 val 是同一个集合」。
- **R3 文档所印**：markdown 里印了某份清单的条数或种子区间的行，必须与文件相符。`{a,b}` 会展开，
  末尾 `dir/...` 解析到该目录下唯一那份——**这两样不是锦上添花**：`benchmarks/README.md` 的种子区间表
  与 `data_integrity_open_items.md` 里引的那段 OVERLAP 打印，恰恰全是这么写的，不支持就等于静默漏掉
  第 12 项要核的「区间」那一半。现判 29 行文档。
- **R4 协议同规模**：`benchmarks/<key>.json` 那 8 份 catalog 默认清单是同一个因子设计的各格，
  `BENCHMARK_GROUPS` 横着叉它们、`run_suite` 解析的就是这几个路径，所以条数必须一致。

### 查出四件事

**一 · `benchmarks/single_u10_upstream_tgt15.json` 被截断了四个月（R4，已修）。**
`bd00950`（2026-04-23，提交信息原文「代码整理，不知道是啥」）把 `9b96a7d` 冻的 30 回合清单换成了
前一天生成的 2 回合文件。八份默认清单里七份 30、它 2。**运行期抓不到**：给了 manifest 之后
`train_utils._resolved_eval_episodes` 返回清单里的 episode、**完全无视 `--eval-episodes`**，
于是声明 `eval_episodes: 30` 的 `flow_factor_v1` 与 `study_core_v1` 会在那一格上评 2 个回合、
照常记账。已从 `9b96a7d` 恢复，判据逐条核过：现存那 2 条与 30 条的前 2 条**逐字节相同**、
只有 `created_at` 不同、`results/`／`experiments/` 无一份读数指名它（两份 `suite_manifest.json`
用的都是 `single_u15_upstream_tgt15.json`）、恢复后污染面枚举**仍是 22 处**，与 `0dc35a2` 记的一致。

**二 · 第 ② 条早就该销号了，欠的是账本这一侧（已闭合）。**
§2 一直写着「两种可能，未确认是哪一种」。实况是：论文侧 `setup.tex` **rev.3（2026-06-18）**
就判原值「系无源误推」并改成了实测的 1.5e5／3.0e5／1.0e5，现行 §5.3.m 印的正是这三个数。
本轮另外两件补上：`crosscomp-2000` 那格的「无数据集」已过期（2026-08-09 就取回了，实测 304,967，
比值 1.57× 与 1000 回合那格**同值**，坐实「整列同一个每回合步数线性外推」）；假设 (b) 可以排除了
——加噪变体现已在位，156,882／313,719，离论文那一列比 deterministic 版更远。
成因与 2026-08-17 那条订正同类：**处置写在论文侧的 rev 块里，账本这一侧没人回来改状态。**

**三 · `paper/archive/rebrac_standalone/outline.md` 把 30 回合的清单写成了 100 回合（已声明）。**
那是起草当时的计划值，100 回合的 `_ep100.json` 要到 `ffa20cc`（2026-05-20）才生成，落差正是
`fql_succession_bug2_fix_decision.md` 记的 Bug 2。归档件不改原文，按本仓惯例在同一行加日期声明。

**四 · 三对清单之间存在浮点末位差（不判缺陷，已记录）。** 最大 `2.7e-15` rad，全在
`initial_heading` 上。其中一对正是 README 那句「`_repro_check_s1250.json` 逐条复现
`single_u10_cross_tgt15_ep100.json`」——**在容差内为真，按精确相等为假**。R2 因此分三档判：
逐条相同 / 末位差（提示）/ 真差异（缺陷），两侧各配一条负控。

### 本轮新添的三条方法论（接 §六、§七、§八）

**14 · 夹具从被测常量算出来，和夹具照着被测常量建输入是同一个坑。**
§七第 9 条记的是「夹具照着 `XSOURCE` 去建输入」。本轮撞到它的算术版：
`test_a_figure_far_along_the_line_is_not_attributed` 的填充写成 `'x' * (cbm.FIGURE_WINDOW + 5)`，
把窗口从 40 改到 500 时填充跟着变成 505，注入完全不可观测、报 GREEN。改成写死 60，
并补一条 `assert cbm.FIGURE_WINDOW < 60`——**写死一个数就要有东西在它失效时喊一声**。

**15 · 两道门守着同一个用例时，负控打在哪一道要先算清楚。**
「取数必须在路径之后」这条门的负控原本用了一行长句，图与路径相距 43 字符——而窗口是 40。
把方向判据拆掉之后窗口仍然拦着，用例照绿。这不是机制不承重，是**注入打在了被另一道门遮住的位置**。
改法是把用例的输入缩短到窗口之内，让那一格只剩方向判据把守。与 §六第 1 条同源，
但那条说的是「拆哪个都被另一个兜住」，这条是「同一道拆对了，输入却落在另一道的射程里」。

**16 · 一条只数缺陷个数的断言，分不出「少查了一半」。**
花括号展开的负控（只展开第一项）头一轮报 GREEN：用例断言「1 处不符」，而截断展开之后
**仍然是 1 处不符**——错的那半边还在，对的那半边没被查而已。改成同时断言对的那半边落进 `ok`。
与 §六第 7 条同源：**「报了该报的」不等于「查了该查的」，正面断言覆盖面。**

### 注入脚本在哪

✅ **2026-08-24 已入库**：本批 34 行现为 `scripts/mutations/benchmarks_manifests.py`，引擎为
[`../../scripts/mutation_probe.py`](../../scripts/mutation_probe.py)，见
[§十](#十注入器已入库2026-08-24这条线欠自己的那笔)。下面这条 scratchpad 路径只作出处：

```
C:\Users\jinxiang\AppData\Local\Temp\claude\D--Codes-rl-v2\889c680d-4a61-4824-a8fe-00a1ab958694\scratchpad\
    mutate_benchmarks.py   本轮 34 条；两种判据层：pytest 用例 id，与两支 sweep 的缺陷桶
    bucket_probe2.py       按模块名跑一支 sweep、按桶计数输出 JSON，供 sweep 层判红
```

比 §四末那支 `mutate_chains.py` 多一样：`buckets()` 一次跑**两支**工具、桶名加 `apn:`／`cbm:`
前缀，因为这一批的 sweep 层横跨两个工具。sweep 层的注入一律改**真文档的一行**，
~~**不动 `benchmarks/` 下任何文件**~~——那些是证据本身。
⚠ **2026-08-24 订正：这句自述与本批第 33／34 行不符**，那两行改的正是 `benchmarks/README.md`。
实际遵守的规矩是「不动**证据本身**（manifest 的 `.json`）」，而 README 是印着证据的文档、正是
`cbm:doc-mismatch` 要考的东西。入库时按后者写进护栏，详见 §十。

~~「要不要把注入器收进 `scripts/`」这个决定仍未拍板；本轮是第五次重写它。~~
→ ✅ **2026-08-24 已拍板并落地（第六次，也是最后一次重写），见 §十。**

---

## 十、注入器已入库（2026-08-24）——这条线欠自己的那笔

§四末与 §九末两次把「要不要把注入器收进 `scripts/`」记成未拍板，第二次还记了「本轮是第五次
重写它」。本节是那个决定的落地：**引擎入库，行表按批入库，并新写一批负控给引擎自己**。

理由不新，是这条线自己的立论：审计固化做了四批，每批的判据都靠一支随会话消失的脚本，而这条
线要证明的恰恰是「每次都没留下能再跑一遍的东西」。工具在，重跑就是一条命令；工具不在，下一
个人只能第六次重写它。

### 交付物

| 文件 | 内容 |
|---|---|
| [`../../scripts/mutation_probe.py`](../../scripts/mutation_probe.py) | 引擎。两种判据层（pytest 用例 id ／ `<工具>:<桶>`）、两种 edit（`text` 字面替换／`claim` 按标签改 spec 的一个键）、三条自检、`--list` ／ `--only` ／ `--out` ／ `--log-dir`。sweep 探针以 `--sweep-probe` 子模式内联，一批不再需要第二支脚本 |
| `../../scripts/mutations/status_claims.py` | 第 10 项那批，**21 行**（原 `mutate_status.py`，2026-08-23） |
| `../../scripts/mutations/tracebacks_chains.py` | 第六/七条链与生成器入库，**17 行**（原 `mutate_chains.py`） |
| `../../scripts/mutations/benchmarks_manifests.py` | 第 12 项那批，**34 行**（原 `mutate_benchmarks.py`） |
| `../../scripts/mutations/mutation_probe_self.py` | **新写的 12 行**——给引擎自己的负控 |
| [`../../tests/test_mutation_probe.py`](../../tests/test_mutation_probe.py) | **39 项**。一半测引擎（`apply_edit` 的各条子句、CRLF 往返、digest 归一），一半**测行表本身**：每行格式合法、每个注入目标文件还在、每个 sweep 行的工具前缀已注册、行号不重复、没有一行指向证据文件 |

**四批 84 行全部实跑通过，零自检失败**（21 ＋ 17 ＋ 34 ＋ 12）。顺带对上一个独立读数：
`tracebacks_chains` 的基线桶打出 `apn:ok = 492`，与 §九记的「十二条链 492 处」一致。

行表是**逐字搬运**的，不是重写：`status_claims` 那批原为 `(label, file, old, new, selector)`
五元组，用 `ast` 按源码 span 机械改写成统一的 `(label, kind, check, edits)`，多行字符串的
手写排布原样保留。搬运不动判据，才能说「跑的还是当初那批」。

### 新加的三条护栏（五支旧脚本都没有）

1. **脏工作树拒跑**（`--allow-dirty` 可绕）。恢复靠 `finally` 里的内存备份，进程被硬杀就丢，
   而旧脚本对此毫无防备——它们只在自己那一轮跑过，风险从没兑现，不等于不存在。
2. **收尾复核 `git status` 与开跑前一致**，不一致则退出码 3。
3. **证据文件拒绝注入**：`benchmarks/` ／ `offline_data/` ／ `results/` ／ `wake_data/` 下的
   非 markdown 文件一律拒绝。

### 查出两件事

**① 自检 A 的另一端一直空着——注入把模块改崩，红得与机制无关。** `mutation_probe_self` 首
轮 12 行「全绿」，其中第 8 行是假的：那条注入写成 `return False and (`，语法就断了，pytest
收集失败、退出码非零，**读起来与「机制被抓到」一模一样**。这正是自检 A 的病在注入后一侧的
复发——A 只管注入前。已补**自检 C**（注入后重跑一次 `--collect-only`，收不到即判无效），并把
那一行改成语法有效的 `return False and rel.startswith(PROTECTED)`，让它去考那条子句而不是
考解析器。

C 自己也做了对照：把第 8 行改回崩版本重跑，同一条注入由「✓ 变红」变成「✗ 自检 C 失败」。

**② 第 12 项那批的自述与它自己的行表不符。** §九写「sweep 层的注入一律改真文档的一行，
**不动 `benchmarks/` 下任何文件**」——而该批第 33／34 行改的正是 `benchmarks/README.md`。
照字面把护栏写进引擎，这两行当场被拒。它实际遵守的规矩是另一条：**不动证据本身**（manifest
的 `.json`），而 README 是**印着证据的文档**，正是 `cbm:doc-mismatch` 那类行要考的东西。
护栏已按后者写，两侧都有用例钉住（拒 `.json`、放行 `README.md`）。
**这又是第 10 项要查的那个毛病，这次长在做第 10 项的工具自己身上。**

### 本轮新添的两条方法论（接 §六、§七、§八、§九）

**17 · 「变红」要先问是哪一种红。** 用例失败是红，模块 import 不进去、选择器收不到、依赖缺
文件也都是红，退出码上分不出来。负控的判据必须钉在「那条用例失败了」，不能钉在「pytest 非零
退出」。同一个病 §六第 1 条在注入前一侧记过（自检 A），这次是注入后一侧。

**18 · 工具自述的护栏，要拿它自己的输入去核。** 「不动 `benchmarks/` 下任何文件」是一句读起
来更严的规矩，而它自己的两行注入违反它；把这句话直接实现成代码，反而把两条有效的负控挡在门
外。与 §八「自述冻结：声明不是事实」同源——区别是那条说的是文档自述，这条说的是**工具对自己
行为的自述**。

### 下一批怎么加

写一张新表放进 `scripts/mutations/`，`ROWS` 一个常量，docstring 首行会出现在 `--list` 里。
不必碰引擎。行表引用的是**具体某一行源码**，源码一改就报「注入点出现 0 次」——那是提醒该行
负控要跟着搬，不是工具坏了；`tests/test_mutation_probe.py` 的表检查会在文件被改名／前缀写错
／行号撞车时立刻喊，不必等到跑整批。
