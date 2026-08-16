# 数据完整性待核项（2026-08-02 记录）

> ## 核实结论（2026-08-09）
>
> **第 1 条与第 3 条均已核实成立**，第 3 条比原怀疑更强。第 2、4 条状态不变。逐条追注见各节末。
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
> ### ⚠⚠ 污染面**仍未封闭**——Drive 侧枚举本身不可信（2026-08-16 探针发现）
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
> 复核前须先修枚举：用 `os.listdir("offline_data")` 逐名 `stat`，与 glob 结果对拍，数量不等即以
> listdir 为准。修复格已加进 [`../notebooks/ch5_data_integrity_probe.ipynb`](../notebooks/ch5_data_integrity_probe.ipynb) §2。

三条在别的工作里顺带撞见、**已在本机核实过证据但尚未处理**的记账问题。都不影响方法本身，
但都会在投稿/返修阶段被审稿人问到，且第 1 条一旦成立会直接推翻一格主结果。

发现场景：跑一组与本仓库无关的对照测试时，两个独立 agent 各自翻了 `paper/` 与
`offline_data/`，报出前两条；第 3 条是本次核实时顺带发现的。**下面所有事实均已亲自核过，
路径与数字可直接复用；未核实的部分逐条标注。**

---

## 1. `crosscomp-2000` 的采集种子可能与评估 manifest 重叠 ⚠️ 最高优先

### 已核实的机制

- 采集：`scripts/collect_offline_data.py:315` → `episode_seed = base_seed + ep`，
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
| ReBRAC-Q cross-2000 | $0.918\pm0.030$ | $0.870\pm0.025$ | $-4.8$ pp（t=4.71，5/5 种子同向） |
| **翻转（2000 − 1000）** | **$+1.6$ pp**（t=1.06，4/5 不为负） | **$+0.8$ pp**（t=0.43，3/5 不为负） | — |
| 采集器基线（解析式，无训练无选点） | $0.950$ | $0.930$ | $-2.0$ pp（n=100，SE≈3.4 pp） |

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

`paper/sections/setup.tex:89-91`（`tab:dataset_matrix`）写的是：

| Alias | 论文 Transitions | 本机实测 | 比值 |
|---|---|---|---|
| `crosscomp-1000` | $\sim$2.4$\times 10^5$ | **152,683**（= `mean_episode_length` 152.683 × 1000） | 1.57× |
| `worldcomp-1000` | $\sim$2.0$\times 10^5$ | **104,360**（= 104.36 × 1000） | 1.92× |
| `crosscomp-2000` | $\sim$4.8$\times 10^5$ | 无数据集 | — |

`crosscomp-2000` 的 4.8e5 恰好是 2.4e5 的两倍，说明这一列很可能是按 episode 数线性外推的
估算值，而非从 metadata 读出的实测值。

两种可能，**未确认是哪一种**：(a) Table 1 的数字过时；(b) 论文主线用的是**加噪采集变体**
（`docs/td3bc_phase0c_experiment_report.md` 里区分过 deterministic / noisy 单元），而本机只
留了 deterministic 版。任何一种都要改表或在表注里写清口径。

---

## 3. 两个同前缀评估 manifest 并存，种子区间是包含关系

- `benchmarks/single_u10_cross_tgt15.json` —— **30 条**，种子 1250..1279
- `benchmarks/single_u10_cross_tgt15_ep100.json` —— **100 条**，种子 1250..1349

前者的种子区间是后者的**子集**。`docs/fql_succession_bug2_fix_decision.md:27` 记着
`single_u10_cross_tgt15.json` 曾经是 30 ep、导致"所有 in-training eval + final test 都是 30 ep"。

`paper/sections/experiments.tex` §5.1 写的是 val = 40 episodes 选 checkpoint、test = 100
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
（本机无，只在 Drive）。

**处置选项**：(a) 给 test manifest 换一段种子（生成器加 `--seed-offset`）并重跑终检评估；
(b) 保留数字，在 §5.3.6 把"独立"改为如实描述选点集与测试集的包含关系。**未决，需用户裁决。**

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
  数据集本身只在 Drive）

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
- **第 1 条的污染幅度 δ 仍未测**，但已具备条件：`ch5_data_integrity_probe.ipynb` §3 放开
  `RUN_CLEAN_PROBE` 即在干净 manifest（`--seed 3000`）上补评两格 × 5 种子（纯评估开销）。
- ✅ **处置已裁决并落地（2026-08-16）**：用户裁决走 **①-c**（正文披露 + 敏感性读数），①③⑤ 合成**一批**整改，一次落地，触及第 5 章 4 个 `.tex` 文件、9 处。
  - `setup` §5.3.5 披露种子区间关系与两千回合的实例重合、§5.3.6 披露选点集嵌套并定义「留出回合」；
  - `rebrac` §5.7.1 增补充终检段、§5.7.2 补留出读数并把「差距闭合过半」改为「约半」、§5.7.m 增两段集中登记全部读数与其边界、筛查表 caption 补 val≡test；
  - `td3bc` §5.6.2 就地限定含噪段两个端点的口径、§5.6.m 补种子区间与复核读数；
  - `discussion` §5.10.4 章级限制补一句（绝对水平系于这一组固定评估实例）。
  - 验收：编译 65 页、0 undefined、2 处 Overfull（与整改前逐条相同）；既有数字零漂移；逐处理由见各 `.tex` 头注 rev 块。章状态见 [`../paper/thesis_ch5/status.md`](../paper/thesis_ch5/status.md)。
- **仍未闭合的一项**：污染面枚举因 Drive FUSE 目录列举残缺而不可信（见文首 ⚠⚠ 块），须以 `os.listdir` 逐名 stat 重建全集后重判。已在 `notebooks/ch5_data_integrity_probe.ipynb` §2 备好修复格，未跑。此项不影响已落地的整改——正文披露覆盖的是已实核的三条。
