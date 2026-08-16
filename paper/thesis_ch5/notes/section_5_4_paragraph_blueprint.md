# §5.4“强化学习方法与算法框架”段落级写作蓝图

> 本文件是 §5.4 正文起草前的段落级蓝图。它规定每段的功能、公式边界、可写 claim 与禁止 claim。正式写入 LaTeX 时应转化为连续学术正文，不应保留“本段目的”“后文将”等写作管理语言。

## 0. 方法节功能

§5.4 的功能是把第 5 章后续结果节共用的强化学习方法语言一次性定义清楚，使 §5.6--§5.9 可以集中讨论证据与机制，而不必重复解释算法基础。

本节的最低验收标准：

- 读者能够区分在线 SAC、TD3+BC、ReBRAC-Q 和 FQL 的训练范式与损失结构。
- 读者能够理解 deployable 协议与 privileged-critic 协议在训练和部署阶段的信息边界。
- 读者能够看出 TD3+BC、ReBRAC-Q 和 FQL 的比较核心不是“算法名排序”，而是行为约束、参考动作质量、critic 稳定性和数据条件之间的关系。
- 本节不泄露 §5.6--§5.9 的主要实验结论。

## 1. 建议小节结构

```tex
\section{强化学习方法与算法框架}\label{sec:ch5_methodology}
\subsection{从部分可观测任务到异策略学习问题}
\subsection{在线异策略 Actor-Critic 与 SAC}
\subsection{行为约束离线 Actor-Critic：TD3+BC}
\subsection{ReBRAC-Q：Q 归一化双正则 TD3+BC 变体}
\subsection{FQL：flow-matching 行为建模与蒸馏策略}
\subsection{训练期特权信息与部署期策略接口}
\subsection{实施细节与可复现性}
```

若篇幅需要压缩，可将 SAC 与 TD3+BC 合并为“异策略 Actor-Critic 基础”，将特权信息协议并入实施细节。但不建议删除 ReBRAC-Q 与 FQL 的独立小节，因为 §5.7 和 §5.9 依赖这两个方法定义。

## 2. 段落级蓝图

### P1. 方法节开场：为何需要统一方法框架

功能：从 §5.3 的任务、观测、奖励和数据集协议过渡到算法层。

应写内容：

- 本章后续比较的对象并非互不相关的算法，而是同一部署约束下不同信息利用方式。
- 这些方法共享 actor/critic、离线数据或交互数据、Q 估计和行为约束等基本构件。
- §5.4 先统一符号和方法边界。

禁止内容：

- 不写“本节不宣称新算法”这种防御性表述。
- 不写后续各节目录清单。
- 不写任何实验数字。

### P2. POMDP 表述与部署观测

功能：把 §5.3 的 AUV 任务抽象为部分可观测控制问题。

应写内容：

- 真实流场与 AUV 动力学状态可记为 \(s_t\)，策略只能读取可部署观测 \(o_t\)。
- 策略网络(actor)输出二维连续动作 \(a_t\)。
- 回报由 §5.3 的奖励设定给出，目标是最大化折扣回报或固定评估协议下的任务成功率。
- 局部单点观测与 hull-integral 特权观测之间的差异，是后续方法比较的核心信息边界。

可写公式：

```tex
o_t = \Omega(s_t), \qquad a_t \sim \pi_\theta(\cdot|o_t), \qquad
J(\pi)=\mathbb{E}_{\pi}\sum_{t=0}^{T}\gamma^t r_t .
```

### P3. 在线与离线数据来源

功能：区分在线交互和固定数据集学习。

应写内容：

- 在线 RL 在训练时通过环境交互生成 replay buffer。
- 离线 RL 只能使用固定数据集 \(\mathcal D=\{(o_t,a_t,r_t,o_{t+1},d_t)\}\)。
- 本章离线部分的关键困难是策略改进不能离开数据支持太远，否则价值估计会外推。

禁止内容：

- 不把离线数据采集器称为 teacher。
- 不提前讨论哪类数据源更好。

### P4. 异策略 Actor-Critic 共同框架

功能：定义 actor、critic、target network 和 bootstrap。

应写内容：

- 策略网络(actor)负责产生动作。
- 价值网络(critic)估计 \(Q(o,a)\)。
- 异策略训练允许用行为数据更新当前策略。
- 双 critic 和 target smoothing 是 TD3 系方法用于减轻过估计和训练震荡的基础设施。

可写公式：

```tex
y_t = r_t + \gamma(1-d_t)\min_{j=1,2}Q_{\bar\phi_j}(o_{t+1},a_{t+1}).
```

### P5. SAC 方法定位

功能：说明 SAC 在本章中的作用。

应写内容：

- SAC 是最大熵异策略 Actor-Critic 方法，优化回报与策略熵。
- 它在本章中承担两个角色：在线可学性基准和 SAC collector 数据源。
- SAC 的核心 actor loss 为熵正则化策略改进。

可写公式：

```tex
\mathcal L_{\pi}^{\mathrm{SAC}}
= \mathbb E_{o\sim\mathcal B,\,a\sim\pi_\theta}
\left[\alpha_{\mathrm{ent}}\log\pi_\theta(a|o)-Q_\phi(o,a)\right].
```

禁止内容：

- 不写 A0 sensor screen 的成功率。
- 不写 history length 如何闭合 s0--s1 gap。
- 不把 SAC 写成第 5 章主线算法。

### P6. TD3+BC 方法定位

功能：定义离线基线，说明行为约束的必要性。

应写内容：

- TD3+BC 在 TD3 确定性策略梯度上加入行为克隆约束。
- 其作用是在离线数据支持范围内进行策略改进。
- TD3+BC 是 §5.6 的离线基线，也是理解 ReBRAC-Q 的参照。

可写公式：

```tex
\mathcal L_{\pi}^{\mathrm{TD3+BC}}
= -\lambda_{\mathrm{TD3+BC}}
\mathbb E_{o\sim\mathcal D}Q_\phi(o,\pi_\theta(o))
+ \mathbb E_{(o,a)\sim\mathcal D}
\|\pi_\theta(o)-a\|^2 ,
\qquad
\lambda_{\mathrm{TD3+BC}}
=\frac{\alpha}{\mathbb E|Q_\phi(o,\pi_\theta(o))|}.
```

禁止内容：

- 不写 TD3+BC 1000 episodes 优于 2000 episodes 的结果。
- 不写 worldcomp privileged gap。

### P7. TD3+BC 的 claim 边界

功能：给 §5.6 留出叙事空间。

应写内容：

- TD3+BC 提供最小行为约束基线，但其单一 BC anchor 无法区分数据质量、动作噪声和价值外推的不同来源。
- 这为后续引入双侧约束和更强参考动作建模提供方法动机。

禁止内容：

- 不提前说 TD3+BC “失败”。
- 不把 baseline 写成低质量对照。

### P8. ReBRAC-Q 命名与方法关系

功能：定义本文方法名，避免与原始 ReBRAC 混淆。

应写内容：

- ReBRAC-Q 是 Q 归一化双正则 TD3+BC 变体。
- 它继承 TD3+BC 的 Q-normalized actor improvement，并引入 actor-side 与 critic-side 两类行为约束。
- 与原始 ReBRAC 的主要差异在于 actor 端 Q 项是否采用 TD3+BC 风格的 batch mean \(|Q|\) 归一化。

禁止内容：

- 不写“关键诚信声明”“不是复刻”等审稿防御式话语。
- 不把 ReBRAC-Q 简写成裸“ReBRAC”。

### P9. ReBRAC-Q actor loss

功能：给出 actor-side BC anchor 和 Q-normalization 的准确公式。

可写公式：

```tex
\bar Q =
\mathbb E_{o\sim\mathcal D}
\left|Q_\phi(o,\pi_\theta(o))\right|_{\mathrm{detach}},
\qquad
\lambda_Q=\frac{1}{\max(\bar Q,\varepsilon)},
```

```tex
\mathcal L_{\pi}^{\mathrm{ReBRAC\text{-}Q}}
= -\lambda_Q
\mathbb E_{o\sim\mathcal D}Q_\phi(o,\pi_\theta(o))
+ \beta_1
\mathbb E_{(o,a)\sim\mathcal D}
\|\pi_\theta(o)-a\|^2 .
```

应写内容：

- \(\beta_1\) 控制 actor 输出贴近数据动作的强度。
- \(\lambda_Q\) 只做尺度归一化，不改变 reward 或任务定义。
- \(\beta_1\) 与 TD3+BC 的 \(\alpha\) 只在本文 Q-normalized 写法下有近似换算关系。

禁止内容：

- 不把 \(\beta_1\) 写成唯一解释变量。
- 不写 \(\beta_1=4.0\) 或 \(\beta_1=1.0\) 哪个最终更好，除非只作为“超参数含义”不带结果判断。

### P10. ReBRAC-Q critic-side BC penalty

功能：准确写出当前实现口径。

可写公式：

```tex
a' = \mathrm{clip}\{\pi_{\bar\theta}(o')+\epsilon\},
\qquad
\epsilon\sim\mathrm{clip}(\mathcal N(0,\sigma^2I),-c,c),
```

```tex
y =
r+\gamma(1-d)
\left[
\min_j Q_{\bar\phi_j}(o',a')
-\beta_2\|a'-a'_{\mathcal D}\|^2
\right].
```

应写内容：

- \(a'_{\mathcal D}\) 是离线轨迹中的下一步数据动作。
- \(\beta_2\) 惩罚 bootstrap target action 偏离数据支持的程度。
- 该项作用于 target-Q 链路，区别于 actor-side BC anchor。

禁止内容：

- 不把该项写成 \(\|a'-\pi_{\bar\theta}(o')\|^2\) 并称其为 critic-side BC penalty。
- 不提前讲 \(\beta_2=0\) 导致 Q drift 的实验结果。

### P11. ReBRAC-Q 中 LayerNorm 的方法角色

功能：把 LayerNorm 放在正确层级。

应写内容：

- LayerNorm 是 critic 表征稳定性的基础设施，与行为约束项共同服务于离线价值估计稳定。
- 它不是 deployment information，也不是额外传感能力。
- 它与特权信息是否必要这个问题正交。

禁止内容：

- 不写“可选增强能力”。
- 不 claim LayerNorm 比 dual penalty 更重要。
- 不写 LN-off 的具体退化数字。

### P12. FQL 方法定位

功能：定义 FQL 与 TD3+BC/ReBRAC-Q 的关系。

应写内容：

- FQL 将行为数据中的动作分布建模为从高斯噪声到数据动作的连续流。
- flow-matching 模型学习条件速度场 \(v_\eta(x_t,t|o)\)。
- 蒸馏策略使用该模型积分得到的参考动作作为去噪 anchor，同时仍通过 critic 做策略改进。

可写公式：

```tex
x_t=(1-t)x_0+t a,\qquad v^\star=a-x_0,\qquad
\mathcal L_{\mathrm{flow}}
=\mathbb E\|v_\eta(x_t,t|o)-v^\star\|^2 .
```

### P13. FQL student / deployed actor loss

功能：说明 FQL 的部署策略来自蒸馏，不是直接部署 flow 模型。

可写公式：

```tex
a_{\mathrm{FM}}(o)=\mathrm{Integrate}(v_\eta(\cdot,\cdot|o)),
```

```tex
\mathcal L_{\pi}^{\mathrm{FQL}}
=-\lambda_Q\mathbb E Q_\phi(o,\mu_\theta(o))
+\alpha_{\mathrm{FQL}}
\mathbb E\|\mu_\theta(o)-a_{\mathrm{FM}}(o)\|^2 .
```

应写内容：

- FQL 的 anchor 指向 flow-matching 生成的参考动作，而不是原始数据动作。
- 这一差异是 §5.9 分析算法与数据条件交互的基础。

禁止内容：

- 不写 FQL “更强”或“失败”。
- 不把 teacher 作为中文概念主语。
- 不提前写 FQL P2 的负结果。

### P14. deployable 协议与 privileged-critic 协议

功能：把训练期和部署期信息边界写清楚。

应写内容：

- deployable 协议下，策略网络(actor)和价值网络(critic)都只读取可部署观测 \(o_t\)。
- privileged-critic 协议下，价值网络(critic)在训练中额外读取 \(o_t^{\mathrm{priv}}\)，策略网络(actor)仍只读取 \(o_t\)。
- 评估与部署阶段只执行策略网络(actor)，critic 被弃用，因此两种协议训练所得策略的部署传感接口一致。

禁止内容：

- 不把 privileged critic 写成部署能力。
- 不把特权信息与全场 teacher 混用。

### P15. 方法框架总表

功能：用一张表固定算法关系。

建议列：

| 算法 | 训练范式 | actor 输出 | 行为约束/先验 | actor 观测 | critic 观测 | 本章角色 |
|---|---|---|---|---|---|---|
| SAC | 在线异策略 | 随机策略 | 熵正则 | deployable | deployable 或特权 ablation | 在线可学性与 collector |
| TD3+BC | 离线异策略 | 确定性策略 | raw-action BC | deployable | deployable/privileged 协议 | 离线基线 |
| ReBRAC-Q | 离线异策略 | 确定性策略 | actor/critic 双侧 BC + Q normalization + LayerNorm | deployable | deployable/privileged 协议 | 离线主线 |
| FQL | 离线异策略 | 蒸馏确定性策略 | flow-matching 参考动作 | deployable | deployable | 算法-数据交互 |

注意：

- “本章角色”列只能是方法定位，不写结果结论。
- LayerNorm 可列在 ReBRAC-Q 的方法构件中，但不能放进“额外能力”或“部署能力”列。

### P16. 损失函数与信息协议汇总表

功能：让后续 §5.6--§5.9 可引用统一符号。

建议列：

| 方法 | actor 目标 | critic 目标 | anchor 对象 | 特权信息可能出现的位置 |
|---|---|---|---|---|
| SAC | 熵正则策略改进 | soft Bellman target | 无显式 BC anchor | 可用于 critic ablation |
| TD3+BC | Q-normalized actor improvement + raw-action BC | TD3 Bellman target | 数据动作 \(a\) | critic 可选 |
| ReBRAC-Q | Q-normalized actor improvement + \(\beta_1\) raw-action BC | 带 \(\beta_2\) next-action support penalty 的 TD target | 数据动作 \(a,a'_{\mathcal D}\) | critic 可选 |
| FQL | Q-normalized actor improvement + distilled reference BC | TD-style critic | flow-matching 参考动作 \(a_{\mathrm{FM}}\) | 本章 FQL 线不依赖 |

### P17. 实施细节与可复现性

功能：放 §5.4 中必须统一的实现口径。

应写内容：

- 统一符号：\(o_t,a_t,r_t,d_t,o_{t+1}\)、\(\mathcal D\)、\(\pi_\theta\)、\(Q_\phi\)、\(\gamma\)、\(\tau\)、\(\beta_1\)、\(\beta_2\)。
- Q-normalization 的 \(\varepsilon\) 下界只为数值稳定。
- TD3-style policy noise、target smoothing、policy update frequency 作为共同实现口径列出。
- privileged observation 的维度和含义回指 §5.3，不在本节重复推导物理公式。

禁止内容：

- 不写 per-seed 统计。
- 不写具体实验 manifest。
- 不写数据集成功率。

### P18. 收束段：把方法边界转为后文可检验问题

功能：从方法框架自然过渡到结果节，但不写目录清单。

应写内容：

- 在相同部署接口下，后文证据将检验三类方法差异：是否需要更多训练期信息、行为约束如何与数据质量匹配、以及单点观测在不同流动强度下的边界。
- 句子应以问题逻辑收束，而不是“下一节将……”式安排。

禁止内容：

- 不把 §5.4 末段写成章节目录。
- 不提前宣布结论。

## 3. 允许 claim 与禁止 claim

### 3.1 允许 claim

- 本章所有学习方法都在同一 AUV 部署接口下比较。
- SAC、TD3+BC、ReBRAC-Q 和 FQL 都可放入异策略 Actor-Critic 或其离线扩展框架中理解。
- TD3+BC 与 ReBRAC-Q 的差异在于行为约束的位置、Q normalization 口径和 critic 稳定性构件。
- FQL 与 ReBRAC-Q 的关键差异在于 actor anchor 的目标来源：原始数据动作 vs flow-matching 参考动作。
- privileged critic 只改变训练期价值估计信息，不改变部署期 actor 传感接口。

### 3.2 禁止 claim

- 禁止说 ReBRAC-Q 的提升由单一 actor anchor 解释。
- 禁止说 LayerNorm 是可选附加能力。
- 禁止说 FQL 全局优于或劣于 ReBRAC-Q。
- 禁止说 privileged critic 全局无用；只能在具体结果节中按证据范围讨论。
- 禁止说 N2' 证明 s0 actor 信息论不可能。
- 禁止把 cross-source magnitude 写成严格可比。

## 4. 正文起草前检查表

- [ ] §5.1 roadmap 已同步到十节结构。
- [ ] `main.tex` 中 §5.4 占位与实际文件名一致。
- [ ] `sections/methodology.tex` 创建前确认标签命名：`sec:ch5_methodology`。
- [ ] ReBRAC-Q critic-side penalty 采用 \(a'_{\mathcal D}\) 口径。
- [ ] ReBRAC-Q 首次出现时完整定义，不裸写 ReBRAC。
- [ ] off-policy 统一写“异策略”。
- [ ] actor/critic 首次出现写作“策略网络(actor)”和“价值网络(critic)”。
- [ ] FQL 不把 teacher 作为概念主语。
- [ ] LayerNorm 写成表征稳定性基础设施。
- [ ] §5.4 表格不含成功率、提升百分点或最终排名。
- [ ] §5.4 末段不写成章节目录。

## 5. 与后续章节的交接边界

| 后续节 | §5.4 交付什么 | 后续节保留什么 |
|---|---|---|
| §5.5 Online RL | SAC 框架、在线异策略定义 | A0 sensor screen、history/sensor 机制、online floor |
| §5.6 TD3+BC | TD3+BC loss 和行为约束动机 | TD3+BC baseline 结果、数据规模反常、BC 对照 |
| §5.7 ReBRAC-Q | ReBRAC-Q loss、β1/β2、LayerNorm 方法角色 | 四个 finding、+23--32pp、deployable vs privileged、Q drift、LN ablation |
| §5.8 泛化边界 | privileged-critic 协议定义 | N0/N2'、asym-critic ablation、critical regime 边界 |
| §5.9 算法比较 | FQL 方法定义、anchor 目标差异 | FQL P2 负结果、SAC collector interaction、direction robust caveat |
| §5.10 讨论 | 统一符号与方法关系 | β1 reconciliation、中心命题回收、limitations |

