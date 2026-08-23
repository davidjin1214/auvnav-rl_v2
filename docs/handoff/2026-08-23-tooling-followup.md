# 交接：/insights 报告的后续处置（2026-08-23）

面向接手会话。本轮把 `/insights` 报告的建议逐条核过并落地了一部分，剩下的是报告
「On the Horizon」三个方向。**核心结论：那三个方向不该各开一个独立会话**，理由见下。

## 一、本轮已落地

| 内容 | 提交 |
|---|---|
| 全局 `CLAUDE.md` 加 3 条（`git -C`、heredoc/`jq` 缺失、验证纪律） | 未版本化时落地，现随 `claude-config` 仓受控 |
| 项目 `CLAUDE.md`：pytest 禁接 `head`/`tail` | `d381a55` |
| 编辑 markdown 后自动跑 `scripts/check_doc_pointers.py --strict` 的 hook | `8222c70` |
| `.gitattributes` 把 `scripts/*.sh` 工作副本钉成 LF | `151cfc7` |
| `probe-the-rule` skill（成对臂协议成文） | claude-skills 仓 `a63f33b` |
| `~/.claude` 纳入 git（private 仓 `claude-config`）＋ `push_guard` hook | claude-config 仓 `d90dd74`→`3cfcd44` |

另外清理了项目 `.claude/settings.local.json` 的 allow 规则（102→53），其中删掉了
`Bash(rm -f main.*)`——它在 `paper/thesis_ch5/` 下会命中 `main.tex`。

**报告里被证伪、未采纳的**：`encoding=` AST 扫描器（此仓不存在，`import ast` 零命中）；
「预授权 git push」（那是 `soft_deny` 的刻意闸门，不是缺失的授权）；`handoff` skill
（`strategic-compact` 已覆盖）；`ship` skill（4/5 步是默认行为，改做成了 hook）。

## 二、第 6 块：不是三个会话

### 方向 1 — skill 效力测试台：**推迟，先别做**

它本质是「把 `probe-the-rule` 自动化」。而那个 skill 今天刚写出来，`status` 字段明写
`UNPROBED`，并预判了自己哪两节最可能是 no-op。**先手工用它探几次真规则**，攒够题面设计
经验（尤其是那几类会让结果读不出来的题面缺陷），再谈自动化。现在做等于把一套没验证过的
协议固化进 harness。

### 方向 2 — 诚信审计固化成 pytest：**值得独立一个会话，工作量最大**

> **已执行完毕（2026-08-24）。** 下面这一节是当时的立项说明，保留原样；实际做成什么样、
> 还剩哪两项，一律看 [`2026-08-23-integrity-audit-pytest.md`](2026-08-23-integrity-audit-pytest.md)。
> 那份是该方向的账本，本节不是。

纯本仓工作，自成体系。已有素材：

- [`docs/data_integrity_open_items.md`](../data_integrity_open_items.md) — 待办项与已闭合项
- [`paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../../paper/thesis_ch5/data_integrity_impact_assessment_review.md) — 分级发现与处置选项
- `paper/thesis_ch5/tools/ch5_holdout_split_audit.py` — 已有的切分审计工具
- `scripts/audit_seed_overlap.py`、`scripts/audit_multimodality.py` — 已有的两个审计脚本

要点：**每条检查都要配负控**（往临时副本注入违规、断言检查确实报警）。静默通过的检查
等于没有检查——本轮的两个 hook 都是这么验的，可照搬。

一个已知约束：污染面枚举 2026-08-21 在 Drive 侧闭合（35 datasets × 24 manifests），而
`offline_data/` 是 gitignored 的，clone 只能扫到更少的 dataset。审计脚本要支持
`--data-dir` / `--benchmarks-dir` 指向 Drive 挂载，别把「本地扫不到」写成失败。

### 方向 3 — 环境加固：**本轮已吃掉大半，剩余零碎**

已落地：`jq` 缺失与 heredoc 陷阱、`git -C`、验证纪律（全局 `CLAUDE.md`）；两个 hook；
CRLF 修复；权限规则清理。

剩余且未实测的：OneDrive 长路径与非 ASCII 路径处理、pytest 临时目录权限、PowerShell
递归删除是否真被分类器拦（本轮未验证，所以**没有**写进 `CLAUDE.md`）。这些零碎到不值得
独立会话，附在别的工作里顺手做即可。

## 三、未了事项

1. `claude-config` 远端残留 `master` 分支需在 GitHub 网页删（`git push --delete` 属于
   必须拒绝的一类，agent 不做）。
2. `claude-config` 的 Mac 侧未接入；那边 `~/.claude/` 已有同名文件，须 `git fetch` 后
   逐个 `git diff` 比对，不可 checkout 覆盖。
3. 全局 `CLAUDE.md` 两机不能共用同一版本（首段是「本机环境（Windows 11 副机）」），
   需拆分或分支。
4. 项目 `.claude/settings.json` 仍 gitignored，所以 [`.claude/hooks/doc_pointers.py`](../../.claude/hooks/doc_pointers.py)
   在 clone 或另一台机器上**不会自动生效**，挂载方法写在该脚本的 docstring 里。

## 四、一个方法论提醒

本轮有两次测量错误，都是同一类：**把工具的输出读成了自己以为的东西**。一次是
`find | xargs grep` 在 find 无输出时 grep 根本没跑，空输出被当成「没有匹配」；一次是
`grep -c $'\r'` 在某个上下文里 `$'\r'` 没被解释，返回的是总行数，被读成了 CR 行数——
据此得出「仓库里 16 个 `.sh` 全是 CRLF」的错误结论，实际 blob 一直是 LF，坏的是
checkout 往返。两次都是先给结论、后被自己推翻。接手时对任何计数类结论，先确认命令本身
跑通了、且输出的确是你以为的那个量。
