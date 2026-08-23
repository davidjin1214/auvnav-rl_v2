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
| `rebrac.json` | [`../rebrac_experiment_report.md`](../rebrac_experiment_report.md) | 85 | 全部吻合；覆盖面按 `48b8d06` 的原范围（论文实际引用的那批，非全部 47 处） |

**全仓 `±` 口径现状**（`--ddof`，三份报告合计 **57 处**）：**55 处 `ddof=0`，2 处 `ddof=1`**，
而那 2 处是**同一个读数**——`0.9340 ± 0.0261`（Stage D Phase 2 privileged），分别印在
ReBRAC 报告的口径补注表与四向对比表里。这机械复现了 `48b8d06` 手工查出的那条结论，
并且现在每次跑都会再验一遍。

`rebrac.json` 把该报告 §1 的**口径补注表本身**也纳入核对：那张表印了逐种子值与两套口径，
于是「这份报告只有一处 ddof=1」这句话不再是被信任的，而是被重算出来的。

arrival_v2 那条链尚未落表：它的源是 `eval_log.csv` 与 `final_eval.json`，读 CSV 的能力
本脚本还没有——落它之前要先加。
