# FQL Succession P2 — test-eval primary metric

n=2 seeds, 100 ep / seed, primary = `eval_success_rate`.

| cell | rebrac s42 | rebrac s0 | rebrac μ | fql s42 | fql s0 | fql μ | δ=F−R | spec verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e_uni | 0.890 | 0.880 | **0.885** | 0.800 | 0.910 | **0.855** | **-0.030** | GRAY |
| m_uni_noise | 0.680 | 0.730 | **0.705** | 0.910 | 0.910 | **0.910** | **+0.205** | POSITIVE |
| m_multi_mix | 1.000 | 0.980 | **0.990** | 0.960 | 0.950 | **0.955** | **-0.035** | GRAY |
