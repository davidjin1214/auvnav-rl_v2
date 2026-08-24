# 数据完整性待核项（2026-08-02 记录）

> ## 核实结论（2026-08-09 起，含 2026-08-16 增补）
>
> **第 1 条与第 3 条均已核实成立**，第 3 条比原怀疑更强。第 2、4 条状态不变。逐条追注见各节末。
>
> **2026-08-16 增补（本块标题原只写 08-09，与下表内容已不符，2026-08-17 订正）**：第 ⑤ 条经 Drive 侧探针钉死成立（与 ① 同因同病），第 ③ 条另得逐回合 seed 直证。**成立项因此是 ①③⑤ 三条，不是两条**；且污染面**尚未封闭**——08-09 那次枚举只覆盖本机 `offline_data/`，⑤ 正是漏在枚举之外的那个（**2026-08-21 已销号**，读数见下方 ✅ 块末尾的「第二轮」小节）。波及面的独立复核见 [`../paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../paper/thesis_ch5/data_integrity_impact_assessment_review.md)。
>
> | 条目 | 结论 | 证据 |
> |---|---|---|
> | ① `crosscomp-2000` 种子重叠 | ⚠ **成立**。评估 manifest 的 **100/100** 条 episode 是训练 episode，任务实例逐条相同 | 数据集 `metadata.json` + reset RNG 重放（[`scripts/audit_seed_overlap.py`](../scripts/audit_seed_overlap.py)） |
> | ② Table 1 transitions 数字 | 仅影响已撤销的 standalone paper 旧稿；论文第 5 章 `setup.tex` rev.3 已用实测值纠正 | 第 5 章 rev 头注四源互证 |
> | ③ 验证集 ⊄ 测试集 | ⚠ **成立且更强**：val **不是重叠而是前缀子集**，40/40 配置下两份 manifest 完全相同 | manifest 生成器无 seed 偏移 + 三个 launcher + notebook 实际启动命令；2026-08-16 用 `validation/seed_*/*.json` 与 `screening/**/test/*.json` 的逐回合 seed 直证，并**订正**一处对 §5.3.6 措辞的误称（见该节末） |
> | ④ 行为策略 vs ReBRAC 成功率 | 未动，仍是建议项 | — |
> | ⑤ 含噪 2000 数据集种子 | ⚠ **成立**（2026-08-16 Drive 侧探针钉死）。`seed=0`、2000 回合，与评估 manifest **100/100 任务实例逐条相同** —— 与 ① 同因同病 | 见第 5 节 |
>
> **污染面（本机范围内）已封闭**：**本机** `offline_data/` 下 10 个数据集，只有 `crosscomp-2000`
> 一个受影响。其余 9 个均为 `seed=0 / 1000 回合`（种子 0..999），而全部 8 个 benchmark key 的
> `manifest_seed` 最小为 **1100**，故不可能相交。复核命令：
>
> ```bash
> python -m scripts.audit_seed_overlap
> ```
>
> ⚠ **该枚举的范围是本机目录，不是全仓**（2026-08-16 独立复核订正）。`audit_seed_overlap` 只扫
> 本机 `offline_data/` 与本机 `benchmarks/`，而第 5 章至少引用了三个不在其中的数据集：500 回合集
> （0..499，安全）、含噪 1000 集（0..999，安全）、**含噪 2000 集（见下方第 5 条）**。
>
> ### ✅ 污染面枚举**已封闭**（2026-08-21 销号）——原为 ⚠⚠「Drive 侧枚举本身不可信」（2026-08-16 探针发现）
>
> **销号读数在本块末尾的「第二轮」小节**；病因是 `rglob` 不进符号链接目录，已由 `06e6d1f` 修掉。
> 以下保留 2026-08-16 发现当时的原文，以便回溯判据是怎么定下来的。
>
> Drive 侧跑同一命令，range pass 报 `datasets: 24  manifests: 22`，只有 `crosscomp-2000` 一个
> `[OVERLAP]`。**但这份清单是残缺的**：同一次会话里 `--verify` 直接按路径读到了
> `crosscomp_..._noise0p05clip0p15_ep2000/metadata.json`（`seed=0`、2000 回合、100/100 逐条相同），
> 而该数据集**从未出现在那 24 行里**。
>
> 证据层面这是硬矛盾，不是解释问题：[`_load_datasets()`](../scripts/audit_seed_overlap.py) 只做
> `offline_data/*/metadata.json` 的 glob，对每个命中都无条件打一行（`[clean]` 或 `[OVERLAP]`，
> 无任何过滤）；而 `run_identity_pass()` 按 `OFFLINE_DATA_DIR / name / "metadata.json"` 直读、
> **无回退分支**，读不到就抛异常。它没抛。→ **文件存在，glob 没返回它。**
>
> 成因未证实，最可能是 Google Drive FUSE 的目录列举分页/缓存不完整（直接 `stat` 已知路径正常）。
> **后果是方法论的**：Drive 侧「只有一个数据集受影响」这个结论**不成立**，因为枚举漏掉了至少一个
> ——而漏掉的那个恰好也是污染的。含噪 1000 集等其余 Drive-only 数据集同样未被真正扫到。
>
> ~~复核前须先修枚举：用 `os.listdir("offline_data")` 逐名 `stat`，与 glob 结果对拍，数量不等即以
> listdir 为准。~~ ⚠ **这个药方无效，见下方 2026-08-17 订正。**
>
> #### ⚠ 2026-08-17 订正与补充（两轮探针原始输出的逐行比对）
>
> 证据是两份 completed notebook 里 §2 的原始输出——首轮 `ch5_data_integrity_probe_completed.ipynb`、
> 次轮 `ch5_data_integrity_probe1_completed.ipynb`，逐行比对而非转述。
>
> **(1) 漏掉的是两个，不是「至少一个」。** 首轮 range pass 的 24 行里，缺的是**含噪 1000 与含噪
> 2000 两个顶层目录**（`..._noise0p05clip0p15_ep1000` 与 `..._ep2000`）；次轮同一份代码报
> `datasets: 26`，两个都在。上面「含噪 1000 集……同样未被真正扫到」当时是推测，现由次轮读数证实。
> 两者在排序里**相邻**，与「目录列举丢了连续一段」的形态相符。
>
> **(2) 证据比原记的更硬：不是单个进程的问题。** 首轮 cell 8（内核内 `glob`，`exec_count=5`）与
> cell 9（子进程跑 `audit_seed_overlap`，`exec_count=7`）**两个不同进程**都只看到 24 个、都缺同样
> 两个；cell 10（第三个进程，`exec_count=9`）按路径直读含噪 2000 却成功。→ 不是某个进程的缓存，
> 是挂载层的目录项。（另：上一段说 `_load_datasets()` 做的是单层 glob，那是 `0aa42ac` 之前的状态，
> 现已改 `rglob`；但**首轮漏的是顶层，单层 glob 本就该扫到**，故与该修复无关。）
>
> **(3) 那次漏扫是瞬时的——于是「重跑对上了」这条路子会误判。** 次轮就是那次重跑，它「对上了」，
> 而问题只是当时没复现。**任何以「再跑一次数目对得上」为由的销号都不成立**，包括次轮自己。
>
> **(4) 原来开的药方（`os.listdir` ↔ `glob` 对拍）结构上抓不到这个病。** 两者是同一次 `readdir`
> 的两个消费者：目录列举短了就一起短，对拍照样干净收场——等于给盲区发合格证。而且那一格
> **从未被执行过**（次轮 notebook 里 `execution_count=None`、outputs 为空），所以这个缺陷此前
> 没被发现。要区分「文件没了」与「列举撒谎」，参照物必须**不经列举**产生。
>
> **(5) 现在的做法：账本对帐（2026-08-17 落地）。**
> [`../scripts/offline_dataset_ledger.txt`](../scripts/offline_dataset_ledger.txt) 记录**跨机并集**的
> 数据集名（本机 29 ＋ Drive 首轮 24 ＋ Drive 次轮 26 ＝ 39 个，含首轮漏掉的那两个）；
> `audit_seed_overlap` 每次 range pass 前先拿账本里每个名字做**直接 `stat`**（lookup，不是列举）：
>
> | 判定 | 含义 |
> |---|---|
> | `[ENUM MISS]` | `metadata.json` stat 得到、枚举却没返回它 —— **就是 Drive 那次的形态**，退出码 `2` |
> | `[SHADOWED ]` | `os.scandir` 看得到而递归 glob 丢了 —— 另一类：`rglob` 不进入符号链接目录（本仓数据经链接挂载，已由测试实证） |
> | `[absent]` | 该名字这台机器上没有 —— 正常，账本是跨机并集 |
> | `[new]` | 枚举到但账本里没有 —— 跑 `--record` 收进去（并集，只增不删，短列举缩不了账本） |
>
> 检测本身经**故意喂假枚举**实测会 fire，不是只在健康文件系统上跑通就算数：
> [`../tests/test_audit_seed_overlap.py`](../tests/test_audit_seed_overlap.py) 共 13 项，含顶层漏项、
> 嵌套漏项、符号链接（数据集与整个集合两种形态，以及链接成环必须收敛），以及退出码 `2`（notebook 里唯一的硬信号；该契约在进程内钉，另有一项子进程测试只钉「`python -m` 起得来」——它跑在夹具树上，不扫宿主的 `offline_data/`，否则一次真发现会表现成测试失败）。
>
> **销号条件**：对 Drive 那棵树跑一次 `python -m scripts.audit_seed_overlap`，对帐块判 clean（零
> `[ENUM MISS]`、零 `[SHADOWED]`、退出码 `0`），且账本已含 Drive 全部数据集名（先跑一次
> `--record` 并回传账本）。可执行版见
> [`../notebooks/ch5_data_integrity_enum_closure.ipynb`](../notebooks/ch5_data_integrity_enum_closure.ipynb)
> ——**别再用** [`../notebooks/ch5_data_integrity_probe.ipynb`](../notebooks/ch5_data_integrity_probe.ipynb)
> §2 末两格：那几格在 Drive 树里就地 `!python -m scripts.audit_seed_overlap`，跑的是 Drive 自带的旧脚本（下一节）。
> **在那之前本块不销号。**
>
> #### 第一次实跑的读数（2026-08-21）：**未销号**
>
> `datasets: 32   manifests: 24`，两遍退出码都是 `2`：**3 条 `[ENUM MISS]` ＋ 同名 3 条
> `[SHADOWED ]`，两遍逐字相同**——所以不是 2026-08-16 那种瞬时短列举，是结构性的。
>
> - 三个名字：`crosscomp_..._noise0p05clip0p15_ep1000`、`..._noise0p05clip0p15_ep2000`、
>   `worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone`。
> - 它们在本机全部枚举正常，其中 `..._noise0p05clip0p15_ep2000` 本机就报 `[OVERLAP]`
>   （seeds 0..1999 × manifest 1250..1349）——**而在 Drive 那一侧它从未进过扫描**。这正是账本
>   要拦的失效形态，账本拦住了；但拦到的盲区仍旧是盲区，仓级封闭谈不上。
> - 病因：`rglob` 不进符号链接目录。`06e6d1f` 已换成显式下行——碰到条目就试着往下走，由文件
>   系统拒绝为准，因而 FUSE 上 `is_dir()` 说谎的情形一并覆盖。
> - 账本 39 → 41（收到 `fql_succession/xbench_u15cross/` 下两个），已回传提交。
>
> **销号仍需在 `06e6d1f` 之后对 Drive 侧重跑一次**——这一轮不能追认。执行记录：
> [`../notebooks/ch5_data_integrity_enum_closure_completed.ipynb`](../notebooks/ch5_data_integrity_enum_closure_completed.ipynb)。
>
> 同一轮还带回两条读数，都不影响上面的判定，但都得记下来：
>
> 1. Drive 的 `benchmarks/` 有 **24** 个 manifest，git 里只有 11 个，且 git 那 11 个 Drive 全有——
>    即 **13 个实际用过的评估 manifest 从未进过 git**（`offline_rebrac_broad/`、
>    `offline_rebrac_screen/`、`offline_rebrac_worldcomp_final/`、`offline_rebrac_worldcomp_epoch_probe/`、
>    `c1_reward_ablation/` 各自的 `val_40/` 与 `test_100/` 切分，加一个 `single_u15_cross_tgt15_ep100.json`）。
>    `benchmarks/` 是被跟踪目录，本意就是让评估集可复现；这 13 个在本机不可见，**任何只在本机跑的
>    审计都数不到它们**。~~是否补进 git 待定。~~ ✅ **同日已全部补进 `benchmarks/`**——读数与核验
>    见本块「第二轮」小节末尾；上面这句「从未进过 git」是当日的发现，保留原文。
> 2. 同一个 `crosscomp_..._ep2000`，在本机 11 个 manifest 里撞 3 个，在 Drive 24 个里撞 **11** 个；
>    多出的 8 处全落在那 13 个未提交 manifest 的 `val_40/` 与 `test_100/` 上。其中 `val_40`
>    （seeds 1250..1289）整段落在训练区间 0..1999 内——这是第 ③ 条「选点用的 40 条本身就是训练
>    episode」的直接物证，此前只有推断。
>
> #### 第二轮（同日，`06e6d1f` 之后）：**三个零成立，本块销号**
>
> clone `859a39f`，`datasets: 35   manifests: 24`，两遍均 **0 条 `[ENUM MISS]` ／ 0 条
> `[SHADOWED ]` ／ `exit=0`**。账本 41 个名字全部落定：**枚举到 35 ＋ `[absent]` 6 ＝ 41**，
> 无一悬空；`--record` 收到 0 个新名字，故本轮不必回传账本。
>
> 病因由自检第 (4) 步**直接坐实**，不再是推断：那三个名字全报
> `is_symlink=True is_dir=True metadata=True` —— 确是符号链接，不是 FUSE 上 `is_dir()` 说谎。
>
> 补回三个集后 `OVERLAP` 由 11 处增至 **22 处 ＝ 2 个数据集 × 11 个 manifest**：
> `crosscomp_..._ep2000` 与 `crosscomp_..._noise0p05clip0p15_ep2000`，二者均 `seeds 0..1999`，
> 撞的是同一批 manifest。含噪 2000 集此前只在本机核过，Drive 侧这是**第一次**进枚举，结论与
> 本机一致（第 ⑤ 条）。执行记录：
> [`../notebooks/ch5_data_integrity_enum_closure1_completed.ipynb`](../notebooks/ch5_data_integrity_enum_closure1_completed.ipynb)。
>
> ⚠ **销号的前提曾写在这里**：本轮覆盖面含 Drive 上那 13 个未进 git 的评估 manifest，它们不进
> git 则任何只从 clone 出发的审计都覆盖不到，这次封闭就不可复现。**已办（同日）**：13 个全部
> 补进 `benchmarks/`，本机重跑 `audit_seed_overlap` 由 `manifests: 11` 变为 **`24`**，OVERLAP 明细
> 与 Drive 那轮**逐行相同**（22 处 ＝ 2 × 11）——封闭现在从 clone 就能复现。不走重生成：
> `auv_nav/env.py` 在协议冻结（`9b96a7d`）之后被 `813096e` 动过，重跑生成器不保证逐条复现，而这是
> 评估记录。落库时逐个核过：条数与目录名相符、种子连续、8 个与审计打印的区间逐条对上。清单与
> 两条新读数见 [`../benchmarks/README.md`](../benchmarks/README.md)。
>
> #### 怎么在 Colab 上跑（Drive 侧不是 git 仓库）
>
> Drive 上那份 `rl_v2_5/` 是同步副本，不是 clone —— 所以**别跑它自带的
> `scripts/audit_seed_overlap.py`**：那份的版本停在最后一次同步，很可能早于 `0aa42ac`（单层
> glob 改 rglob）、`8d03fcc`（账本对帐 ＋ 退出码 `2`）与 `95a8302`（`--data-dir`），跑了会静默
> 重现旧行为，且没有对帐块可判。**代码从 git 取、数据从 Drive 读**，两边不必是同一棵树：
>
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> !git clone --depth 1 https://github.com/davidjin1214/auvnav-rl_v2.git /content/rl_v2
> %cd /content/rl_v2
> DRIVE = '/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5'
> # 1) 先收名字进账本（并集，只增不删）
> !python -m scripts.audit_seed_overlap --data-dir "$DRIVE/offline_data" \
>     --benchmarks-dir "$DRIVE/benchmarks" --record ; echo "exit=$?"
> # 2) 再判对帐；exit=0 才算 clean
> !python -m scripts.audit_seed_overlap --data-dir "$DRIVE/offline_data" \
>     --benchmarks-dir "$DRIVE/benchmarks" ; echo "exit=$?"
> ```
>
> 上面四行是手跑的最小配方；照着跑一遍的 notebook 是
> [`../notebooks/ch5_data_integrity_enum_closure.ipynb`](../notebooks/ch5_data_integrity_enum_closure.ipynb)，
> 它另外做两件手跑做不到的事：把 clone 与 Drive 两边的 manifest 取**并集**再审（覆盖面缺一边都不算
> 仓级封闭），以及把对帐块解析成三个零的判定。**判据与理由留在本文，可执行步骤留在 notebook**，
> 两边不复述彼此。
>
> 三个必看的点：
>
> 1. **`!` 行的非零退出不会让 cell 失败**，而退出码是这里唯一的硬信号 —— 所以每行都跟一句
>    `echo "exit=$?"`，并**以打印出来的数字为准**，不要只看输出好不好看。
> 2. **回传账本**：clone 里的 `scripts/offline_dataset_ledger.txt` 是被 `--record` 改的那份，
>    要取回本机提交（`files.download` 或写回 Drive 再同步），否则这次收到的名字下轮就没了。
> 3. **`--record` 只能记下枚举**看得见**的名字** —— 枚举本身正是嫌疑对象，所以短列举会
>    少记。这不构成销号，只是让账本长期变厚；判据仍是上面那两个零加退出码 `0`，且照
>    (3) 条，「重跑一次数目对上了」不算数。

三条在别的工作里顺带撞见、**已在本机核实过证据但尚未处理**的记账问题。都不影响方法本身，
但都会在投稿/返修阶段被审稿人问到，且第 1 条一旦成立会直接推翻一格主结果。

发现场景：跑一组与本仓库无关的对照测试时，两个独立 agent 各自翻了 `paper/` 与
`offline_data/`，报出前两条；第 3 条是本次核实时顺带发现的。**下面所有事实均已亲自核过，
路径与数字可直接复用；未核实的部分逐条标注。**

---

## 1. `crosscomp-2000` 的采集种子可能与评估 manifest 重叠 ⚠️ 最高优先

### 已核实的机制

- 采集：`scripts/collect_offline_data.py:316` → `episode_seed = base_seed + ep`，
  `base_seed` 来自 `--seed`（:619）。
- 评估 manifest 生成：`scripts/generate_standard_benchmarks.py:30` → `seed = manifest_seed + idx`。
- `benchmarks/single_u10_cross_tgt15_ep100.json`：100 条，种子 **1250..1349**。
- **`offline_data/` 下现存 9 个数据集，`metadata.json` 里 `seed` 全部 = 0、
  `num_episodes` 全部 = 1000** → 训练种子区间 0..999。

### 因此

对所有 **1000-episode 数据集**：训练 0..999 vs 评估 1250..1349，**不相交，安全**。

对 **`crosscomp-2000`**：该数据集**本机不存在**（`offline_data/` 下只有 `_ep1000` 目录），
无法核。但现存 9 个数据集的 `base_seed` **无一例外都是 0**，若 2000-episode 那次采集沿用
同样的调用，种子区间就是 **0..1999，完整包含评估的 1250..1349** —— 即 100 条评估 episode
逐条出现在训练集里。采集与 manifest 生成调的是同一个 `env.reset(seed=s, ...)`，同种子
⇒ 同 `flow_time`/`start_xy`/`goal_xy`/`initial_heading`，是完全相同的任务实例。

**这正好命中 `paper` Table 3 里 `cross-2000` 那一行，也就是全文最大的那个数字（+32.2pp）。**

### 怎么查（几分钟）

1. `crosscomp-2000` 数据集的 `metadata.json` 里的 `seed` 与 `num_episodes`（数据集若在 Mac 侧）；
2. 或 `docs/td3bc_phase0c_experiment_report.md` 里那次采集的命令行。

若 `base_seed` 是 1000 / 2000 → 不相交，只需在正文写明种子机制即可。
若是 0 → 需换 base_seed 重采并重跑该单元。

### ✅ 核实结论（2026-08-09）：**`base_seed` = 0，重叠成立**

数据集已从 Drive 取回（`offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/`）：

```
"seed": 0, "num_episodes": 2000, "num_transitions": 304967, "success_rate": 0.8855
```

→ 训练 episode 种子 **0..1999**，完整包含评估的 1250..1349。种子层面 **100/100 相交**；
reset RNG 重放（不跑仿真）确认任务实例**逐条相同** —— `flow_time` / `start_xy` / `goal_xy` /
`initial_heading` 四项在 `1e-6` 容差下 **100/100 一致**（30 回合的
`single_u10_cross_tgt15.json` 同样 30/30）。工具与复现命令：

```bash
python -m scripts.audit_seed_overlap --verify crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000 single_u10_cross_tgt15_ep100
```

**成因**：`collect_offline_data.py` 的 `--seed` 默认值就是 `0`，采集命令从未传过它——这也是现存
10 个数据集 `seed` 全部为 0 的原因。**不是某次操作失误，是默认值 + 数据规模跨过 1250 的必然结果。**

**影响面（限本条）**：只波及 2000 回合一格。第 5 章 §5.7 结果表该行（ReBRAC-Q $0.918\pm0.030$ /
TD3+BC $0.596\pm0.036$）与 §5.7.1「数据规模退化的翻转」（$0.918$ 对 1000 回合的 $0.902$）都建立在
这一格上，**最脆弱的是"翻转"这条论述**——它本身就只有 $+1.6$pp。

**但 §5.6 的「$2000<1000$」结论方向相反、不受威胁**：污染只会**抬高** 2000 格的表现，而 TD3+BC
与纯 BC 在该格上均**更差**；污染制造不出这个方向的结果，至多说明真实差距比报告的更大。（这是定性
推断，非实测。）

**处置选项**：(a) 换 `base_seed`（如 `--seed 5000`）重采 2000 回合数据集并重跑该单元；
(b) 保留数字，在正文与表注显式披露该格的训练/评估任务实例重合，并把 §5.7.1 的翻转论述降级为
不可用；**(c)** 在**现有检查点**上补一次干净 manifest 终检，把 δ 测出来再决定（独立复核补的第三条
路径，纯评估开销、不重采不重训）。**未决，需用户裁决。**

> ✅ **2026-08-24 改状态：上面这句「未决」已过期。** 本文末「处置与复核状态」记着**2026-08-16 用户已裁决走 ①-c**（正文披露 ＋ 敏感性读数），①③⑤ 合成一批整改并已落地（第 5 章 4 个 `.tex`、9 处）。选项列表保留原文，它是当时考虑过什么的记录；**状态以本注为准**。成因同第 ② 条：裁决与落地写在别处，本节没人回来改这一格。

### ✅ Drive 侧探针 A（2026-08-16）：**选中检查点全部在位 → (c) 可做**

`notebooks/ch5_data_integrity_probe_completed.ipynb` §1：cross-1000 与 cross-2000 两格 ×
`actorb_4p0__criticb_2p0` × 种子 42–46 共 **10 个单元，每个 run 目录 10 个 `.pt`，
`selected_checkpoint.json` 指名的那一个 100% 在位**（cross-1000 四个 `agent_step_00033432.pt` +
一个 `agent_final.pt` + 一个 `agent_step_00028656.pt`；cross-2000 四个 `agent_step_00066752.pt` +
一个 `agent_step_00057216.pt`）。本机 `results/offline/**` 下 `.pt` 计数为 0，故此前无法判断。

### ✅ 污染幅度 $\delta$ 已实测（2026-08-16，①-c 路径走完）

在干净 manifest `benchmarks/clean_probe/single_u10_cross_tgt15_ep100_s3000.json`（100 条，种子
3000..3099，与训练区间 0..1999 和现测试集 1250..1349 **均不相交**）上，用**原封不动的那 10 个
选中检查点**重跑终检。纯评估，无训练、无重采。复算：`python paper/thesis_ch5/tools/ch5_clean_probe_readout.py`

| 量 | 已刊（1250..1349） | 干净集（3000..3099） | 位移 |
|---|---|---|---|
| ReBRAC-Q cross-1000 | $0.902\pm0.021$ | $0.862\pm0.019$ | $-4.0$ pp（t=4.78，5/5 种子同向） |
| ReBRAC-Q cross-2000 | $0.918\pm0.030$ | $0.870\pm0.024$ | $-4.8$ pp（t=4.71，5/5 种子同向） |
| **翻转（2000 − 1000）** | **$+1.6$ pp**（t=1.06，4/5 不为负） | **$+0.8$ pp**（t=0.43，3/5 不为负） | — |
| 采集器基线（解析式，无训练无选点） | $0.950$ | $0.930$ | $-2.0$ pp（n=100，SE≈3.4 pp） |

> **2026-08-24 订正**：干净集 `cross-2000` 一格原印 $\pm0.025$。逐种子值
> $[0.89,\ 0.87,\ 0.86,\ 0.83,\ 0.90]$ 的总体标准差为 $0.024495$，三位舍入为 $0.024$；
> $0.025$ 与真值相差 $0.000505$，**超出**「半个末位」的可接受舍入范围。论文侧
> `paper/thesis_ch5/sections/rebrac.tex` 一直印的就是 $0.024$，故本表是漂的那一侧。
> 均值、位移、t 值、差中差均未变。成因是这一格的离散度此前不在
> `ch5_clean_probe_readout.py` 的冻结期望里——该期望只钉均值，而均值本来就是对的；
> 现已把四格的离散度一并纳入。

**① 可归因的抬高量 = 差中差 $+0.80$ pp**（t=0.512，95% CI $[-3.5, +5.1]$，逐种子
$[0.03, 0.03, 0.00, 0.03, -0.05]$）。逐格位移里的另外两项——实例抽样难度与选点泛化——**对两格
共模**，故在两格之差上抵消；这也是唯一该拿来做裁决的量，单格 $\delta$ 的绝对值不是。

对照复核文档 §4 ★3 预设的门槛：$0.80$ pp **低于** $2$ pp（「点估计反而略高」半句失效线），
**远低于** $5$–$7$ pp（翻转论述被推翻线）。→ **章内的真实断言「未再检出回落」在干净集上依然成立**，
点估计仍为正。

**但同次读数带出三条不利事实，处置时必须一并计入**：

1. **两格的绝对水平都掉了约 $4$–$5$ pp，且高度显著**（两格皆 5/5 种子同向、CI 不含 0）。这个量
   **比 ① 干的事大得多**。
2. **其中只有约 $2$ pp 是抽样难度**——采集器策略是解析式的，从未训练、从未选点，其位移即纯难度
   偏移，实测 $-2.0$ pp（且在 $n=100$ 下与 0 不可区分）。**剩下的约 $2$–$3$ pp 更像选点泛化**：
   检查点是在种子 1250 族的 `val_40` 上选的，换一族要付代价。**这是 ③ 家族的效应，而
   `hold60` 结构上看不见它**——留出 60 条仍在 1250 族内。
3. **本次只补评了 ReBRAC-Q 两格**。TD3+BC 与纯 BC 未重评，故 §5.7.1 的基线差距（$23.0$ /
   $32.2$ pp）**不能**在干净集上重述——那些差距的两端会一起移动，方向未知。

### ⚠ 同次探针的附带发现：① 与 ③ 在 2000 格上**叠加**

Drive 侧 range pass 显示 `crosscomp-2000` 的训练区间 0..1999 不只吞掉终检的 100 条，还吞掉了
**全部选点用的 val manifest**：

```
OVERLAP 100/100  offline_rebrac_broad/test_100/single_u10_cross_tgt15.json   (1250..1349)
OVERLAP  40/40   offline_rebrac_broad/val_40/single_u10_cross_tgt15.json     (1250..1289)
OVERLAP 100/100  offline_rebrac_screen/test_100/...                          (1250..1349)
OVERLAP  40/40   offline_rebrac_screen/val_40/...                            (1250..1289)
OVERLAP  40/40   offline_rebrac_worldcomp_epoch_probe/{test_40,val_40}/...   (1250..1289)
OVERLAP 100/100  offline_rebrac_worldcomp_final/test_100/...                 (1250..1349)
OVERLAP  40/40   offline_rebrac_worldcomp_final/val_40/...                   (1250..1289)
OVERLAP  30/30   single_u10_cross_tgt15.json                                 (1250..1279)
OVERLAP 100/100  single_u10_cross_tgt15_ep100.json                           (1250..1349)
```

即：在 2000 格上，**选检查点用的那 40 条本身就是训练 episode**。③ 单看是"在测试集的前缀上选点"，
叠上 ① 之后在这一格变成"**在训练 episode 上选点、再在训练 episode 上报告**"。这一点此前的波及面
评估与独立复核都未单列，处置时须计入：它意味着 (c) 路径测出的 δ 是 ①③ 合并效应，而这恰好是
正确的口径——两条本就要合成一批整改。

### 顺带建议

不管结果如何，都值得跑一次 overlap audit：只调 `env.reset` 不跑仿真，把训练种子区间的
`(flow_time, start_xy, goal_xy, initial_heading)` dump 出来与 manifest 的 100 条做集合比对。
注意 `transitions.npz` 里只有 `obs/actions/.../privileged_obs`，**不存** `flow_time`/`start_xy`，
所以不能直接从数据集比对，必须重放 reset RNG。

---

## 2. `paper` Table 1 的 transitions 数字与本机 metadata 对不上

`paper/archive/rebrac_standalone/sections/setup.tex:89-91`（`tab:dataset_matrix`）写的是：

| Alias | 论文 Transitions | 本机实测 | 比值 |
|---|---|---|---|
| `crosscomp-1000` | $\sim$2.4$\times 10^5$ | **152,683**（= `mean_episode_length` 152.683 × 1000） | 1.57× |
| `worldcomp-1000` | $\sim$2.0$\times 10^5$ | **104,360**（= 104.36 × 1000） | 1.92× |
| `crosscomp-2000` | $\sim$4.8$\times 10^5$ | **304,967**（= 152.4835 × 2000） | 1.57× |

`crosscomp-2000` 的 4.8e5 恰好是 2.4e5 的两倍，说明这一列很可能是按 episode 数线性外推的
估算值，而非从 metadata 读出的实测值。

两种可能：(a) Table 1 的数字过时；(b) 论文主线用的是**加噪采集变体**
（`docs/td3bc_phase0c_experiment_report.md` 里区分过 deterministic / noisy 单元），而本机只
留了 deterministic 版。任何一种都要改表或在表注里写清口径。

### ✅ 核实结论（2026-08-24）：**是 (a)，且论文侧早已改完——欠的是这份账本**

三件事，逐条：

1. **(b) 已被排除。** 加噪采集变体本机现已在位，两档的 `num_transitions` 离论文那一列比
   deterministic 版**更远**——**没有任何本机采集变体能给出 2.4e5 / 4.8e5**：

   | 加噪变体（`offline_data/` 下） | 本机实测 | 论文同规模那格 |
   |---|---|---|
   | `crosscomp_..._noise0p05clip0p15_ep1000` | **156,882** | $\sim$2.4$\times 10^5$ |
   | `crosscomp_..._noise0p05clip0p15_ep2000` | **313,719** | $\sim$4.8$\times 10^5$ |
2. **`crosscomp-2000` 不再「无数据集」。** 上表该格原写「无数据集」，是 2026-08-02 的实况；
   数据集已于 2026-08-09 从 Drive 取回（见第 1 条的核实块），实测 **304,967** 条，比值 1.57×
   与 `crosscomp-1000` 那格**同值**——这坐实了「整列按同一个每回合步数线性外推」的判断，
   而不是逐行各错各的。
3. **论文侧在 2026-06-18 就改完了，这份账本没跟上。** `paper/thesis_ch5/sections/setup.tex`
   rev.3 判原值「系无源误推（疑把 reward spec 中 240 步 fixture 示例误当平均回合长度反推）」，
   并改以二位有效数字写：$1.5\times10^5$ / $3.0\times10^5$（crosscomp 1000 / 2000）、
   $1.0\times10^5$（worldcomp 1000）。现行 `setup.tex` §5.3.m 印的正是这三个数，与本表
   「本机实测」列在二位有效数字下逐格吻合。**独立投稿撤销后，`paper/archive/rebrac_standalone/`
   下那张 Table 1 是归档件，不再是活稿**；本条到此闭合，不需要再改表。

> 本条闭合迟了两个月，成因与 2026-08-17 那条订正同类：**处置写在论文侧的 rev 块里，账本这一侧
> 没人回来改状态**。本条与第 1／4／5 条的 `offline_data/` 读数现已由
> [`tracebacks/benchmarks_and_datasets.json`](tracebacks/benchmarks_and_datasets.json)
> 机械复算，改动 metadata 或改动这里的刊值都会当场报出来。

---

## 3. 两个同前缀评估 manifest 并存，种子区间是包含关系

- `benchmarks/single_u10_cross_tgt15.json` —— **30 条**，种子 1250..1279
- `benchmarks/single_u10_cross_tgt15_ep100.json` —— **100 条**，种子 1250..1349

前者的种子区间是后者的**子集**。`docs/archive/fql_succession/fql_succession_bug2_fix_decision.md:29` 记着
`single_u10_cross_tgt15.json` 曾经是 30 ep、导致"所有 in-training eval + final test 都是 30 ep"。

`paper/archive/rebrac_standalone/sections/experiments.tex` §5.1 写的是 val = 40 episodes 选 checkpoint、test = 100
episodes 一次性报告。**40 这个数字对不上上面任何一个文件，未确认它来自哪里。**

需要确认的是：**选 checkpoint 用的那批 episode 与最终报告用的那批是否互斥。**
若 val 用的是 30ep 文件、test 用的是 100ep 文件，则 val ⊂ test，等于在测试集上选模型。
查法：ReBRAC 主线 run 的 `trainer_state.json` 里的 `eval_manifest` 字段。

### ✅ 核实结论（2026-08-09）：**不互斥，且不是"重叠"而是"前缀子集"**

不必等 Drive 侧的 `trainer_state.json` —— 从生成器代码与实际启动命令两头就能对上：

1. [`scripts/generate_standard_benchmarks.py`](../scripts/generate_standard_benchmarks.py) **没有任何
   seed 偏移参数**（只有 `--benchmarks / --episodes / --output-dir / --output-name`），种子恒为
   `manifest_seed + idx`。**同一个 benchmark key 生成的任何 manifest，都从同一个种子起算。**
2. 三个 launcher 的 val 与 test manifest 都用**同一个** `BENCHMARK_KEY` 调它，只是落到不同目录
   （`run_offline_rebrac_screen.sh` 的 `ensure_manifest`，`run_offline_rebrac_broad.sh`、
   `run_offline_td3bc_phase0c.sh` 同构）。
3. notebook 里的实际启动命令：**39 处全部**是 `BENCHMARK_KEY=single_u10_cross_tgt15`
   （`manifest_seed = 1250`），配 `VAL_MANIFEST_EPISODES=40` 与 `TEST_MANIFEST_EPISODES=100`（或 `40`）。

于是：

| 配置 | 出处 | val 种子 | test 种子 | 关系 |
|---|---|---|---|---|
| 40 + 100 | broad / phase0c Stage C / worldcomp final | 1250..1289 | 1250..1349 | val = test 的**前 40 条** |
| 40 + 40 | screen / phase0c Stage B / epoch probe | 1250..1289 | 1250..1289 | **两份 manifest 逐条相同** |

即：**报告的 100 回合测试集里，有 40 条正是用来选 checkpoint 的那批**；在 40+40 的单元里两者完全重合。

> **⚠ 2026-08-16 订正（独立复核发现的事实错误）**：本条初稿写「第 5 章 §5.3.6（rev.6）写的
> 『在验证集上按字典序规则选点、在**独立**测试集上评估』，"独立"不成立」——**正文里没有"独立"
> 这个词**。`sections/*.tex` 的 body 检索"独立测试"零命中；`setup.tex:215` 的实际措辞是
> 「在评估集上对该次训练选定的检查点重新评估……或为训练期保存的检查点中按验证集指标的字典序
> 选出者」。"独立测试集"只出现在 `td3bc.tex` rev.6 **头注**里对 GT 报告统一选择规则的转述。
> → **③ 的正文敞口比原描述小**：不是撤回一个错误断言，而是**补一个未写明的限定**。这一订正
> 直接影响处置取舍（披露路线更可行），且该误称已随文档扩散过一轮，务必不要回退。

「40 回合」的出处也随之明确：不是某个 `benchmarks/` 下的文件，而是 launcher 的
`VAL_MANIFEST_EPISODES` 默认值现场生成的 `benchmarks/<root>/val_40/single_u10_cross_tgt15.json`
（**2026-08-21 起本机已有**：`0dc35a2` 把这批只在 Drive 的评估 manifest 补进了 `benchmarks/`；
括注原作「本机无，只在 Drive」，是当时的实况，现已不成立）。

> **✅ 2026-08-21 补证：上表不再只是从生成器代码与启动命令推出来的，已由读数逐条实测。**
> `paper/thesis_ch5/tools/ch5_manifest_attribution.py` 拿每份读数的 `eval_episode_results`
> 逐条取 `(episode_id, seed)` 当指纹，去比 `benchmarks/` 下的 24 份 manifest：3715 份读数
> **无一落空**（3280 份逐条相同，435 份是某份更长 manifest 的开头一段），刊出的每个单元都归到
> 上表三批实例之一 —— cross 100 条 1250..1349、cross 40 条 1250..1289、upstream 100 条
> 1400..1499。于是 2026-08-21 那次只扫 24 份 manifest 的污染面枚举，**覆盖了全部刊出评估实例**，
> 这一步此前是缺的。
>
> 同一份工具从 launcher 侧反查物理文件名（路径是 `${MANIFEST_ROOT}/{val,test}_${N}/${BENCHMARK_KEY}.json`
> 推出来的，不是写死的）：ReBRAC 主线 `results/offline/rebrac/formal` 用的是
> `benchmarks/offline_rebrac_screen/test_100/single_u10_cross_tgt15.json` —— **screen 那个根，
> 不是目录名会让人猜的 `worldcomp_final` 或顶层文件**。另有两处 launcher 至今仍在指名、而文件
> 连 Drive 上都没有：`offline_rebrac_screen/test_40/`，以及 `scripts/run_offline_td3bc_phase0*.sh`
> 名下整个 `benchmarks/offline_phase0*/` 家族（7 个根）。**文件没了，它们装的实例集没丢** —— 用过
> 它们的读数指纹都落在上面两批 cross 实例里。判据与等价类明细见 `benchmarks/README.md`。

**处置选项**：(a) 给 test manifest 换一段种子（生成器加 `--seed-offset`）并重跑终检评估；
(b) 保留数字，在 §5.3.6 把"独立"改为如实描述选点集与测试集的包含关系。**未决，需用户裁决。**

> ✅ **2026-08-24 改状态：上面这句「未决」已过期。** 本文末「处置与复核状态」记着**2026-08-16 用户已裁决走 ①-c**（正文披露 ＋ 敏感性读数），①③⑤ 合成一批整改并已落地（第 5 章 4 个 `.tex`、9 处）。选项列表保留原文，它是当时考虑过什么的记录；**状态以本注为准**。成因同第 ② 条：裁决与落地写在别处，本节没人回来改这一格。落地形态是 (b)：`setup` §5.3.6 已披露选点集嵌套并定义「留出回合」。

---

## 4. 一条不算缺陷但会被问到的

`offline_data/worldcomp_..._ep1000/metadata.json` 的采集策略 `success_rate` = **0.958**，
而论文报告 ReBRAC-Q 在 `worldcomp-1000` 上是 **0.928**。两者分布不同（采集在 seeds 0..999
上、评估在冻结的 100-episode manifest 上），**不能直接比**，但数字太接近，容易被问出
"离线策略打得过产生数据的行为策略吗"。

建议：把两个 behaviour policy 在**评估 manifest 上**的成功率跑出来，作为一行加进 Table 3。
纯 eval 开销，且顺带回应"最强的 baseline 其实就是数据采集策略本身"这个必然会有的质疑。

---

## 5. 含噪 2000 回合数据集的种子未核 ⚠️ 新增（2026-08-16，独立复核发现）

第 1 条的污染面枚举只覆盖本机 `offline_data/`，**漏掉了一个规模跨过 1250 的数据集**：

- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_noise0p05clip0p15_ep2000`
  （结果目录 `results/offline/td3bc/phase0c/noisy_support_screen/noisy_std0p05_clip0p15/` 可证其存在；
  数据集本身只在 Drive——**2026-08-24 注：括注是当时的实况，该集本机现已在位**，故下方核实块里的
  `seed` 与回合数已由 [`tracebacks/benchmarks_and_datasets.json`](tracebacks/benchmarks_and_datasets.json)
  机械复算，不再只能靠 Drive 侧探针）

`collect_offline_data.py` 的 `--seed` 默认为 `0`，本机现存 10 个数据集**无一例外都是 0**。若含噪 2000
沿用同一调用，其训练种子区间同为 **0..1999**，完整包含评估 manifest 的 1250..1349 —— 与
`crosscomp-2000` 同因同病。

**波及**：第 5 章 §5.6.2 的 noisy-support 诊断句「两千回合数据上的纯行为克隆成功率上升（由约
$0.60$ 升至约 $0.74$）」——

- `0.60` 出自**确定性 2000**（即已确认受第 1 条污染的那个数据集），必然要重出或披露；
- `0.74` 出自含噪 2000，**本条待核**。

减轻情节：`td3bc.tex` rev.5 已把该段的机制归属从两千那条腿挪到**干净的一千回合反证**上
（0.68/0.76 → 0.51），受影响的只是被降级为"上升"的那半句。另：该筛查为 40+40 单元，val 与 test
逐条相同（第 3 条），故它同时是第 3 条最严重的一类实例。

**怎么查（分钟级，需 Drive）**：`notebooks/ch5_data_integrity_probe.ipynb` §2，或直接

```bash
python -m scripts.audit_seed_overlap
python -m scripts.audit_seed_overlap --verify crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_noise0p05clip0p15_ep2000 single_u10_cross_tgt15_ep100
```

### ✅ 核实结论（2026-08-16 Drive 侧探针）：**成立，与 ① 同因同病**

```
dataset : crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_noise0p05clip0p15_ep2000  seeds 0..1999
manifest: single_u10_cross_tgt15_ep100.json  100 episodes
seeds in both ranges     : 100/100
identical task instances : 100/100
```

`seed=0`、2000 回合 → 训练区间 0..1999，完整包含 1250..1349，reset RNG 重放确认逐条相同。
于是 §5.6.2 那句「两千回合数据上的纯行为克隆成功率上升（由约 $0.60$ 升至约 $0.74$）」**两个端点
都坐在污染数据上**：`0.60` 出自确定性 2000（第 ① 条），`0.74` 出自本条。且该筛查是 40+40 单元，
val 与 test 逐条相同（第 ③ 条）——**三条问题在这一句上同时命中**。

减轻情节不变：`td3bc.tex` rev.5 已把机制归属挪到干净的一千回合反证（0.68/0.76 → 0.51），受影响的
只是被降级为"上升"的那半句。处置随 ① 一并裁决。

---

## 处置与复核状态（2026-08-16）

- 波及面评估：[`../paper/thesis_ch5/data_integrity_impact_assessment.md`](../paper/thesis_ch5/data_integrity_impact_assessment.md)（初稿）
- **独立复核**：[`../paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../paper/thesis_ch5/data_integrity_impact_assessment_review.md)
  —— 判定「可作裁决依据但须打补丁」；6 处漏项、2 处"守得住"要打折、3 条推断的可靠度分级、
  以及一条被漏掉的处置路径（①-c：在现有检查点上补一次干净 manifest 终检）。
- **第 3 条已零成本量化**：[`../paper/thesis_ch5/tools/ch5_holdout_split_audit.py`](../paper/thesis_ch5/tools/ch5_holdout_split_audit.py)
  把已刊 100 回合劈成「参与过选点的 40 条」与「选点规则未见的 60 条」。留出 60 条上组间差**全部
  保号且变大**（翻转 +1.6→+3.0 pp、基线差距 23.0→28.0 / 32.2→40.0 pp），**唯一例外**是 §5.7.2 的
  「差距闭合过半」53.0%→**48.9%**。
- **Drive 侧探针已回（2026-08-16，`notebooks/ch5_data_integrity_probe_completed.ipynb`）**：
  - 探针 A **通过** —— 10 个选中检查点全部在位，路径 ①-c（干净 manifest 补评）可做；
  - 探针 B **钉死第 ⑤ 条**（含噪 2000 同样 100/100 污染），同时**暴露枚举残缺**，污染面仍未封闭
    （见文首 ⚠⚠ 块）；
  - 附带发现 **① 与 ③ 在 2000 格上叠加**：该格选点用的 40 条 val 本身就是训练 episode。
- ~~**第 1 条的污染幅度 δ 仍未测**，但已具备条件：`ch5_data_integrity_probe.ipynb` §3 放开
  `RUN_CLEAN_PROBE` 即在干净 manifest（`--seed 3000`）上补评两格 × 5 种子（纯评估开销）。~~
  ⚠ **2026-08-17 订正：本行早已过期，删划线保留只为留痕。** δ 于 2026-08-16 随下一条的 ①-c
  处置一并测出并进正文（`6618161`）：干净 manifest（`--seed 3000`）补评已落地，差中差
  **$+0.80$ 个百分点**（种子级 95% CI $[-3.5,+5.1]$ 跨零），见 `rebrac.tex` §5.7.1 与 §5.7.m。
  复算工具 [`../paper/thesis_ch5/tools/ch5_clean_probe_readout.py`](../paper/thesis_ch5/tools/ch5_clean_probe_readout.py)。
  **本行滞留一天即误导了一次任务派单**（2026-08-17 的轮次简报据此写「δ 从未测过」）——
  本文件是事实账本，条目办完即须就地改状态，追加新块不算办完。
- ✅ **处置已裁决并落地（2026-08-16）**：用户裁决走 **①-c**（正文披露 + 敏感性读数），①③⑤ 合成**一批**整改，一次落地，触及第 5 章 4 个 `.tex` 文件、9 处。
  - `setup` §5.3.5 披露种子区间关系与两千回合的实例重合、§5.3.6 披露选点集嵌套并定义「留出回合」；
  - `rebrac` §5.7.1 增补充终检段、§5.7.2 补留出读数并把「差距闭合过半」改为「约半」、§5.7.m 增两段集中登记全部读数与其边界、筛查表 caption 补 val≡test；
  - `td3bc` §5.6.2 就地限定含噪段两个端点的口径、§5.6.m 补种子区间与复核读数；
  - `discussion` §5.10.4 章级限制补一句（绝对水平系于这一组固定评估实例）。
  - 验收：编译 65 页、0 undefined、2 处 Overfull（与整改前逐条相同）；既有数字零漂移；逐处理由见各 `.tex` 头注 rev 块。章状态见 [`../paper/thesis_ch5/status.md`](../paper/thesis_ch5/status.md)。
- **污染面枚举（2026-08-16 更新，分两层）**：
  - ✅ **第 5 章范围已封闭**。本机补齐全部章内依赖数据集后，`audit_seed_overlap` 对 $29$ 个集跑通，**OVERLAP 仍只有两格**（`crosscomp-2000` 确定性集与其含噪变体，各 $100/100$），其余全 clean；以 `results/offline/**` 目录名反查，章内每个训练单元所依的数据集无一缺失。含噪 $2000$ 集本机独立复现 $100/100$，同族 $1000$ 集 $0/100$。
  - ✅ **仓库全域已封闭（2026-08-21 销号）**。Drive 侧实跑：`datasets: 35 manifests: 24`，两遍均零 `[ENUM MISS]`／零 `[SHADOWED ]`／`exit=0`，账本 41 个名字全部落定（35 枚举到 ＋ 6 `[absent]`）。**病因不是 FUSE**：那三个漏项经自检实测全为 `is_symlink=True`，`rglob` 拒绝进入符号链接目录，已由 `06e6d1f` 换成显式下行修掉。此前「漏扫是瞬时的」这一判断只对 2026-08-16 那次成立，08-21 首轮两遍逐字相同、属结构性，二者是不同的失效。⚠ 封闭依赖 Drive 上 13 个**未进 git** 的评估 manifest，它们不进 git 则本次封闭不可复现——该项另开，见文首 ✅ 块末尾。
  - ⚠ 另查出一个**独立**缺陷并已修（`0aa42ac`）：`_load_datasets()` 用单层 `glob` 而 `_load_manifests()` 用 `rglob`，两侧不对称，`offline_data/fql_succession/` 与 `audit_dryrun_2026-05-19/` 下的 $12$ 个嵌套集从未进过枚举（全部 clean）。**它不解释 Drive 那次**——两者是不同的缺陷，勿相互冒充。
  - 此项不影响已落地的整改——正文披露覆盖的是已实核的三条。
