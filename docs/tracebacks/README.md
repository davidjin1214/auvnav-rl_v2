# `docs/tracebacks/` — 刊值溯源表

> 收入时间：2026-08-23。每份 `.json` 把一份报告的「这个数从哪来」写成可执行的形式，由
> [`../../scripts/audit_published_numbers.py`](../../scripts/audit_published_numbers.py) 重跑。

## 为什么需要它

2026-05 到 08 之间做过四次**逐格数字回溯**——把报告里的刊值拿回 `results/` 的逐 seed JSON
重算一遍。每次花一个会话，每次都查出东西，**每次都没留下能再跑一遍的东西**：

| 提交 | 链 | 当时的结论 |
|---|---|---|
| [`2a8c311`](../fql_succession_p2_results.md) | FQL P2 | 全部读数与统计量精确复现；两处 scope 表述订正 |
| [`06ec295`](../td3bc_phase0c_experiment_report.md) | td3bc phase0c | 三个均值逐位吻合；`±` 是 `ddof=0`，报告原先没写明 |
| [`48b8d06`](../rebrac_experiment_report.md) | ReBRAC | 论文引用批全部逐位吻合；查出唯一一处 `ddof=1` 混入 |
| [`6380082`](../arrival_v2_experiment_report.md) | arrival_v2 | 读数全对；四处表述/口径已被论文侧取代却未回填 |

下一个怀疑某个刊出数字的人，得从头考古一遍。本目录把这件事变成一条命令。

顺带解决了那四次都撞上、都没能收口的问题：**某个 `±` 到底是哪套口径**。
[`../../paper/thesis_ch5/tools/ch5_dispersion_audit.py`](../../paper/thesis_ch5/tools/ch5_dispersion_audit.py)
能判 `.tex` 里那些「行内印了逐种子值」的行，并明说其余的「需要 ground-truth 报告」——
这里就是那个 ground truth：种子值从 `results/` 取，两套口径都算得出来，然后告诉刊值它是哪一套。

```bash
python -m scripts.audit_published_numbers            # 全部溯源表
python -m scripts.audit_published_numbers --ddof     # 只看 ± 口径判定
python -m scripts.audit_published_numbers --strict   # 有缺陷即退 1（markdown 编辑钩子跑的就是这个）
python -m scripts.audit_published_numbers --root /path/to/drive/results/fql_succession
```

## 一条表长什么样

字段与判据写在脚本的 docstring 里（那是唯一权威处，别在这里复制一份口径）。要点只有两条：

**一 · 刊值不抄进表里。** 表给的是「怎么在文档里定位到它」——一条只能命中一行的 `anchor`，
加一条从该行取数的 `capture`。抄进来的刊值在文档改了之后仍然自洽，**表会继续绿**，而这正是
本工具唯一要防的事。文档改写导致锚定不上，报的是缺陷，因为那恰恰是该重新核这个数的时刻。

**二 · 锚要按结构写，别按值写。** `td3bc_phase0c` 那份的教训：报告里 Stage B 与 Stage C 两张表
前两列一模一样（`| 500 | 0.5 |`），按前缀锚会同时命中两张表——工具拒绝猜、报了「锚定到多行」，
问题才暴露。改成数单元格个数（Stage C 行后面还有 6 格，Stage B 只有 4 格）后唯一。
同理，别拿刊值本身当锚：那样值一改就变成「锚定不到」，而不是它本该是的「刊值不符」。

## 报告明知有错、又要留着的那格

`arrival_v2_experiment_report.md` §7.9.4 保留了一列 2026-05-19 的中间态主张，下面用 ⚠ 注写明
其中两格重号（`0.064` 不是 σ_final——那两个 seed 的 final 同为 0.900，σ 精确为 0）。对这种格子
写一条普通 claim，它会**永远红**；而一条本来就该红的检查，很快就没人看了。

`"expect": "mismatch"` 把断言翻过来：**钉的是那条声明，不是那个数**——它必须继续复现不出来，
哪天复现出来了，说明表被改过、上面那条注该重读了（进 `erratum-stale` 缺陷桶，`--strict` 会红）。
写这种 claim 必须带 `note` 说清钉的是哪条声明。

更有用的是配对写法：同一格再写一条普通 claim，指向那条注**声称**它其实是什么。
§7.9.4 那格的两条就是——一条证明它不是 σ_final，一条证明它正是同两 seed 的 mean39 σ（ddof=0）。
于是那条勘误注的诊断本身变成可重跑的，而不是被信任的。

## 缺数据不算失败

`results/` 是 gitignored 的，clone 拿不到。缺文件报 `no-data`、**不进 `--strict`**；
要在应该有数据的机器上把缺失也判成失败，加 `--require-data`。

锚定与取数在查数据**之前**做，所以 clone 里跑仍然有意义：它能告诉你某份报告是不是已经被改得
和它的溯源表对不上了。回归测试 `tests/test_audit_published_numbers.py::test_the_shipped_specs_still_anchor_to_their_reports`
钉的就是这一条。

## 覆盖范围

| 表 | 报告 | 刊值数 | 状态 |
|---|---|---|---|
| `fql_succession_p2.json` | [`../fql_succession_p2_results.md`](../fql_succession_p2_results.md) | 35 | 全部吻合 |
| `td3bc_phase0c.json` | [`../td3bc_phase0c_experiment_report.md`](../td3bc_phase0c_experiment_report.md) | 57 | 全部吻合；23 处 `±` 判定为 `ddof=0` |
| `rebrac.json` | [`../rebrac_experiment_report.md`](../rebrac_experiment_report.md) | 87 | 全部吻合；覆盖面按「第 5 章实际引用的那批」，非全部 47 处。2026-08-24 补上 §7.4 的第三格 $(\beta_1,\beta_2)=(4.0,\,1.0)$——`48b8d06` 只核了那张表加粗的两行，而章节在 `tab:ch5_rebrac_perseed` 的 caption 里引了第三行 |
| `arrival_v2.json` | [`../arrival_v2_experiment_report.md`](../arrival_v2_experiment_report.md) | 143 | 全部吻合（订正 1 格，见下）；§7.5 四向对照表 + §7.9 全部 gate 读数 + §7.10 传感九宫格 |
| `online_a0.json` | [`../online_rl_line_summary.md`](../online_rl_line_summary.md) | 24 | 全部吻合；12 处 `±` 判定为 `ddof=0` 11 处、`either` 1 处（该格逐种子同值）|

五条链共 **346 处**刊值。

**`online_a0` 不是那四次手工回溯之一。** 前四条链的授权范围是「把 2026-05 那四次逐格回溯变成
命令」，在线线的 A0 传感筛选从来不在其中——它不是被声明豁免的，是原本就不在范围内。加它的
理由是：`ch5_dispersion_audit` 在 `.tex` 侧的未决读数里有 6 处只能追到
[`../online_rl_line_summary.md`](../online_rl_line_summary.md)，而那份文档此前没有任何机械溯源，
在线线因此是唯一一条完全靠人记住口径的线。

它的 provenance 规则是**复算出来的，不是从正文读来的**：逐种子值取 `final_eval.json`
（60 万步终评、30 回合，与 `eval_log.csv` 末行同值），聚合是 mean ± 总体标准差，种子 46/47/50。
同一批 run 的 `eval_log.csv` 上另外三种归约对全部 24 处刊值**无一对得上**——包括 `max`，
而该文档正文那句「best per-cell success ≥ 70%」恰好会把人引向 `max`（0.989 对 0.967）。
两张结果表的表头与行标签完全相同，所以每条 claim 都靠上方那行加粗标记 `after`／`before` 夹住；
`tests/test_audit_published_numbers.py::test_the_online_a0_chain_is_ambiguous_without_its_table_scope`
把这层作用域本身当负控钉住：拆掉它，24 条 claim 必须全部退化成「锚定到多行」。

**全仓 `±` 口径现状**（`--ddof`，合计 **93 处**）：**75 处 `ddof=0`、14 处 `ddof=1`、
3 处两套口径都对得上（`either`，因为该格离散度本身接近 0）、1 处 `NEITHER` 且那 1 处正是
上面说的、被自己报告声明为重号的那格**。

其中 `ddof=1` 的 14 处集中在 arrival_v2 §7.9.4 第三列一系（3-seed、报告自己标了 ddof=1）
与 ReBRAC 的 `0.9340 ± 0.0261`。ReBRAC 那处是**同一个读数**（Stage D Phase 2 privileged）
印在两张表里，机械复现了 `48b8d06` 手工查出的「本报告只有一处 ddof=1」。

`rebrac.json` 把该报告 §1 的**口径补注表本身**也纳入核对：那张表印了逐种子值与两套口径，
于是那句结论不再是被信任的，而是被重算出来的。

## 落表时查出来的

**`arrival_v2` §7.5 `eval_path_length_m` 的 tandem 一格**：原印 `66.99`，原始值 `66.99537`，
两位小数的正确舍入是 `67.00`——差 0.0054 m，是截断不是舍入。已订正（§7.3 正文同格一并）。
2026-05 那次手工回溯（`6380082`）逐格核的是 §7.6–§7.9 的 11 个 run，§7.5 的指标网格不在其范围内，
所以这不是推翻它、是它没覆盖到的地方。顺带：§7.3 原写该格「四组中最短」，但 §7.1 single+cross
的 63.58 更短；已改为「三个 upstream 组中最短」并点名 §7.1。

**这条链有两件本工具验不了的**，写在 spec 的 `note` 里：`6380082` 的头号发现（论文侧已改、
报告侧未回填）属表述层；OOB 列要 `eval_termination_counts` 的分项除以 `num_eval_episodes`，
本工具只读平铺键与 CSV 列。
