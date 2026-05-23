# FQL Succession P2 — Cross-Benchmark Confirmation Spec (`single_u15_cross`)

> **状态**:DRAFT(2026-05-23,待 collection + Colab 执行)
> **作者**:Claude Code session
> **依据**:[`fql_succession_p2_results.md`](fql_succession_p2_results.md)(P2 主报告)+ [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9(权威 lab 记录)
> **目的**:在更难的 benchmark 上**精简复核** P2 的两条 load-bearing 结论,硬化机制 claim 的泛化性。**不是**重跑 P2 的全展开。

---

## 0. 为什么做这个(scope 一句话)

P2 的负面 + 机制结论目前只在 **`single_u10_cross`(U=1.0/Re150)** 一个 benchmark 上成立。投稿时最大的审稿风险是「**单 benchmark → 能泛化吗?**」。本 spec 用**最小代价**在 online 线已在用的更难 benchmark **`single_u15_cross`(U=1.5/Re250)** 上复核,**只测两条 load-bearing claim**:

- **C1(noise 是 discriminator)**:clean vs noisy 是真正区分两算法的轴。
- **C2(β1-tuning artifact)**:固定 ReBRAC β1=1.0 在 worst-case-over-noise 上 **≥ FQL**(即"FQL 赢"是 ReBRAC β1 mis-tuning 的产物),且 β1 4→1 的噪声恢复复现。

modality 轴(E-multi)在 P2 已判 NULL → modality 已被排除,**本复核不再测**。

---

## 1. 从 P2 继承(不重复定义)

| 维度 | 沿用 P2 设定 |
|---|---|
| Algos | FQL(flow-matching teacher + 1-step distill,**冻结** `--flow-steps 10 --distill-alpha-bc 1.0 --teacher-lr 3e-4 --flow-time-embed-dim 32`)vs ReBRAC(`train_offline --algo rebrac`,`--critic-penalty-coef 2.0` 冻结) |
| Primary metric | test JSON 顶层 `eval_success_rate`,在**固定 ep100 manifest** 上 → episode_ids/env-seed 一致 → 对比**配对**,seed-spread = 纯训练方差(σ_train ≈ 3.8pp) |
| Train protocol | s0 / h4 / target 1.5 / cross_stream / arrival_v2 / 200k steps / batch 256 / eval-every 10k / eval-episodes 100 / test 100;`train_offline` → `evaluate_offline`;skip-resume on `agent_final.pt` |
| Framing | **B+A:机制发现 + 诚实负面**(NOT "FQL wins")。本复核是 generalization 证据,不改 verdict 性质 |

唯一变化:**flow + benchmark + 数据集**从 `re150_u10cross` 换到 `re250_u15cross`。

---

## 2. 最小矩阵(noise 轴,12 runs)

| cell | noise | configs | seeds | runs |
|---|---|---|---|---|
| `e_uni_clean` | privileged σ=0 | FQL / ReBRAC β1=4.0 / ReBRAC β1=1.0 | 42, 0 | 6 |
| `m_uni_noise` | privileged σ=0.5 | FQL / ReBRAC β1=4.0 / ReBRAC β1=1.0 | 42, 0 | 6 |

- **3 configs 缺一不可**:FQL(对手);ReBRAC β1=4.0(冻结默认,噪声下应塌);ReBRAC β1=1.0(canonical,应救回)。β1=4.0 是复现 headline 机制所必需。
- **2 seeds [42, 0]**:与 P2 一致,**不补 seed**(更多 seed 只收紧 NULL,翻不了 dominance + 机制)。
- ReBRAC flags:β1=4.0 → `--actor-penalty-coef 4.0 --critic-penalty-coef 2.0`;β1=1.0 → `--actor-penalty-coef 1.0 --critic-penalty-coef 2.0`。

输出树:`checkpoints/fql_succession/p2_xbench/<cell>/<name>`、`results/fql_succession/p2_xbench/<cell>/test/<name>.json`,`<name>` ∈ {`fql_seed{S}`, `rebrac_b1p4_seed{S}`, `rebrac_b1p1_seed{S}`}。

---

## 3. 前置(manifest + 2 datasets)

**3.1 ep100 manifest**(P2 配对 rigor 需要 100-scenario;现仅有 30-ep `single_u15_cross_tgt15.json`)。用现有 key `single_u15_cross_tgt15` + `--output-name` 生成 100-ep 变体(与 u10 ep100 同模式):
```bash
python -m scripts.generate_standard_benchmarks \
  --benchmarks single_u15_cross_tgt15 \
  --episodes 100 \
  --output-name single_u15_cross_tgt15_ep100
# → benchmarks/single_u15_cross_tgt15_ep100.json（flow = wake_v8_U1p50_Re250_* 由 benchmark spec 自带）
```

**3.2 两个 dataset**(本机 6-worker CPU,各 ~3-4 min;复用 P2 配方,仅换 flow):
```bash
# clean-uni（对应 P2 E-uni；σ=0）
python -m scripts.collect_offline_data --policy privileged \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 1000 --seed 0 --num-workers 6 --action-noise-std 0.0 \
  --output-dir offline_data/fql_succession/xbench_u15cross/e_uni_clean_1000

# noisy-uni（对应 P2 M-uni-noise；σ=0.5，physically single privileged + Gaussian widening）
python -m scripts.collect_offline_data --policy privileged \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 1000 --seed 1 --num-workers 6 --action-noise-std 0.5 \
  --output-dir offline_data/fql_succession/xbench_u15cross/m_uni_noise_eps0p5_1000
```

> audit 不重跑(D22:advisory)。noisy cell 物理上是 single privileged + Gaussian widening,继承 P2 §2.2 的 GMM false-positive 透明披露。

---

## 4. 地板效应门(必跑,挡在 12-run 矩阵之前)

u15_cross 更难(U=1.5,正是 AUVHamNODE 审计里 wake current 2-4× OOD 的速度;online 线记过 §7.9.7 universal-floor finding)。**若三个 config 全塌到地板或全顶天花板,就没有区分信号**,矩阵白跑。所以先挡一道:

1. **看 collection 自报 success_rate**:clean privileged 应远高于 0.5(P2 u10 是 0.985);noisy privileged P2 u10 是 0.632,u15 预计更低但应 > ~0.4。
2. **先只跑 1 个 run**:`rebrac_b1p1` / `e_uni_clean` / seed 42,确认 test `eval_success_rate`:
   - **PASS**:∈ (0.50, 0.97) → 既可学又有 headroom → 解锁剩余 11 runs。
   - **FLOOR**(≤ 0.50)→ benchmark 对 offline-from-privileged 太难,**STOP**,回报后再决定(换稍易配置 / 接受地板本身是 finding)。
   - **CEILING**(≥ 0.97)→ 无区分 headroom(u15 下不太可能),回报后再议。

---

## 5. 预注册判据(verdict)

对每个 config 取 **worst-case-over-noise = min(clean SR, noisy SR)**。

| 结果 | 判定 |
|---|---|
| ReBRAC β1=1.0 worst-case **≥** FQL worst-case(FQL 不赢)**AND** β1=4.0 noisy < β1=1.0 noisy(4→1 噪声恢复复现) | ✅ **CONFIRM** —— P2 的 C1+C2 在更难 benchmark 上泛化,机制 claim 硬化 |
| FQL worst-case > ReBRAC β1=1.0 worst-case **> 5pp** | ⚠ **NEW POSITIVE** —— FQL 在更难 benchmark 意外赢 worst-case,**重开**(不是协议失败),细看再定 framing |
| 差距 < 5pp 同向/方向不一 | **NULL/TIE** —— 与 P2 相容(FQL 无 edge);记入报告,verdict 性质不变 |

统计提醒:n=2,固定 manifest ⇒ 配对,σ_train ≈ 3.8pp;小差距(<5pp)属 NULL 分辨率。

---

## 6. 算力预算

- Collection:2 dataset × ~3-4 min(本机 CPU)≈ <10 min。
- Manifest gen:秒级。
- Training:floor gate 1 run + 矩阵 11 runs,各 ~30-45 min/run(L4)≈ **6-9h** → 1-2 个 Colab session(skip-resume 跨 session 接续)。
- Eval:12 × 数 min。

---

## 7. 交付物

- 1 个 run-notebook(builder `scripts/_build_fql_succession_p2_xbench_notebook.py`,镜像 P2 builder 模式)→ 用户上 Colab 跑。
- 跑完后:把 xbench 结果**追加**进 [`fql_succession_p2_results.md`](fql_succession_p2_results.md)(generalization 小节)+ verdict notebook 增一个 xbench 行;**不新开**报告文档。

---

## 8. 明确不做(carveouts)

- ❌ modality 轴(E-multi / M-multi-mix)—— P2 已 NULL,modality 已排除。
- ❌ C-1 FQL `distill_alpha_bc` 复赛 —— 除非 §5 命中 NEW POSITIVE。
- ❌ 补 seed(> 2)。
- ❌ 重 tune 算法:FQL flags + ReBRAC β2 全冻结;只 β1 取 {4.0, 1.0} 两点。
- ❌ 改 Gate-B-frozen `auv_nav/{fql,rebrac}.py`(只动 CLI 值)。
- ❌ 重跑 GMM audit(advisory per D22)。
