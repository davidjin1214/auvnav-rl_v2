"""Generate docs/tracebacks/arrival_v2.json.

Hand-editing 100-odd anchor regexes is how a column gets mis-pointed silently, so the
table columns are enumerated here and the locators derived from position.

Usage:
    python docs/tracebacks/_gen/gen_arrival_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Optional output directory, so the regression test in tests/test_audit_published_numbers.py
# can regenerate into a scratch tree instead of overwriting the committed specs.
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"
OUT = OUT_DIR / "arrival_v2.json"
VAN = "single_u15_cross_tgt15/arrival_v2/sac_vanilla"


def cell(c):
    """Regex prefix that steps over the row label plus `c-1` data cells."""
    return r"^\|[^|]*\|" + r"[^|]*\|" * (c - 1)


def num(c):        return cell(c) + r" \*{0,2}([0-9.]+)"
def mean_of(c):    return cell(c) + r" \*{0,2}([0-9.]+) ±"
def std_of(c):     return cell(c) + r" \*{0,2}[0-9.]+ ± ([0-9.]+)"
def peak_of(c):    return cell(c) + r" \*{0,2}([0-9.]+) @"
def step_of(c):    return cell(c) + r" \*{0,2}[0-9.]+ @ \*{0,2}([0-9]+)k"


claims = []


def add(label, anchor, capture, stat, sources, **kw):
    claim = {"label": label, "anchor": anchor, "capture": capture, "stat": stat,
             "sources": sources}
    claim.update(kw)
    claims.append(claim)


# ---------------------------------------------------------------- 7.5, the 4-way grid
SEC75 = r"^### 7\.5 4-way"
RUNS75 = [
    ("§7.1 single+cross",    "single_u15_cross_tgt15/arrival_v2/sac_vanilla/s1_k4/seed_42"),
    ("§7.2 single+upstream", "single_u15_upstream_tgt15/arrival_v2/sac_vanilla/s1_k4/seed_42"),
    ("§7.3 tandem",          "tandem_u15_upstream_tgt15/arrival_v2/sac_vanilla/s1_k4/seed_42"),
    ("§7.4 sbs",             "sbs_u15_upstream_tgt15/arrival_v2/sac_vanilla/s1_k4/seed_42"),
]
PAIRED = ["eval_return", "eval_safety_cost", "eval_time_s", "eval_path_length_m",
          "eval_progress_ratio", "eval_path_efficiency"]

for c, (name, run) in enumerate(RUNS75, start=1):
    fe = [f"{run}/results/final_eval.json"]
    csv = [f"{run}/results/eval_log.csv"]
    add(f"7.5 {name} final", r"^\| final_success_rate \|", num(c), "mean", fe,
        section=SEC75)
    add(f"7.5 {name} peak", r"^\| peak \(first-hit step\) \|", peak_of(c), "mean", csv,
        metric="max(eval_success_rate)", section=SEC75)
    add(f"7.5 {name} peak step (k steps)", r"^\| peak \(first-hit step\) \|", step_of(c),
        "mean", csv, metric="argmax(eval_success_rate, env_step)", scale=1000,
        section=SEC75)
    for key in PAIRED:
        add(f"7.5 {name} {key}", rf"^\| {key} \|", mean_of(c), "mean", fe,
            metric=key, section=SEC75)
        add(f"7.5 {name} {key} ±", rf"^\| {key} \|", std_of(c), "mean", fe,
            metric=key + "_std", section=SEC75,
            note="终检自带的 std，不是跨 seed 离散度——单 run 30 episodes 的样本标准差")
    add(f"7.5 {name} eval_energy", r"^\| eval_energy \|",
        cell(c) + r" ([0-9][0-9 ]*[0-9])", "mean", fe, metric="eval_energy",
        section=SEC75)

# ------------------------------------------------- 7.9.1 / 7.9.2 per-run gate readings
GATE = [
    # (label, row anchor, k, seed)
    ("7.9.1 k=8 seed=42 (§7.8 anchor)", r"^\| §7\.8 anchor \(重列\) \|", 8, 42),
    ("7.9.1' k=8 seed=0",               r"^\| §7\.9\.1' sister", 8, 0),
    ("7.9.1'' k=8 seed=7",              r"^\| §7\.9\.1'' sister", 8, 7),
    ("7.9.2 k=12 seed=42",              r"^\| §7\.9\.2 anchor", 12, 42),
    ("7.9.2' k=12 seed=0",              r"^\| §7\.9\.2' sister", 12, 0),
]
for label, anchor, k, seed in GATE:
    run = f"{VAN}/s0_k{k}/seed_{seed}"
    fe = [f"{run}/results/final_eval.json"]
    csv = [f"{run}/results/eval_log.csv"]
    add(f"{label} final", anchor, num(2), "mean", fe)
    add(f"{label} peak", anchor, peak_of(3), "mean", csv,
        metric="max(eval_success_rate)")
    add(f"{label} peak step (k steps)", anchor, step_of(3), "mean", csv,
        metric="argmax(eval_success_rate, env_step)", scale=1000)
    add(f"{label} mean39", anchor, num(4), "mean", csv,
        metric="mean(eval_success_rate)")
    add(f"{label} n_succ", anchor, cell(6) + r" ([0-9]+)/", "mean", csv,
        metric="count_gt(eval_success_rate, 0)")
    add(f"{label} n_evals", anchor, cell(6) + r" [0-9]+/([0-9]+)", "mean", csv,
        metric="nrows(eval_success_rate)")

# ------------------------------------------------------- 7.9.3 seed=0 across history k
HISTORY = [
    ("7.9.3 seed=0 k=4",  r"^\| §7\.7\.1 sister \|", 4),
    ("7.9.3 seed=0 k=8",  r"^\| §7\.9\.1' \|", 8),
    ("7.9.3 seed=0 k=12", r"^\| §7\.9\.2' \|", 12),
]
for label, anchor, k in HISTORY:
    run = f"{VAN}/s0_k{k}/seed_0"
    fe = [f"{run}/results/final_eval.json"]
    csv = [f"{run}/results/eval_log.csv"]
    add(f"{label} final", anchor, num(3), "mean", fe)
    add(f"{label} peak", anchor, peak_of(4), "mean", csv,
        metric="max(eval_success_rate)")
    add(f"{label} peak step (k steps)", anchor, step_of(4), "mean", csv,
        metric="argmax(eval_success_rate, env_step)", scale=1000)
    add(f"{label} mean39", anchor, num(6), "mean", csv,
        metric="mean(eval_success_rate)")

# ------------------------------------------------------- 7.9.4 the two dispersion rows
FINAL8 = [f"{VAN}/s0_k8/seed_*/results/final_eval.json"]
FINAL12 = [f"{VAN}/s0_k12/seed_*/results/final_eval.json"]
MEAN8 = [f"{VAN}/s0_k8/seed_*/results/eval_log.csv"]
MEAN12 = [f"{VAN}/s0_k12/seed_*/results/eval_log.csv"]
# The 2026-05-19 column had only seeds {42, 0}; `seed_[04]*` is exactly those two, and
# the `n` claim below holds that glob to the "2 seeds" the column itself prints.
PAIR_FINAL = [f"{VAN}/s0_k12/seed_[04]*/results/final_eval.json"]
PAIR_MEAN = [f"{VAN}/s0_k12/seed_[04]*/results/eval_log.csv"]
MEAN_METRIC = "mean(eval_success_rate)"

ROW_F = r"^\| σ_final \|"
ROW_M = r"^\| σ_mean39 \|"
COL2_L = cell(2) + r" k=8: ([0-9.]+) →"
COL2_R = cell(2) + r" k=8: [0-9.]+ → k=12: ([0-9.]+)"
COL3_L = cell(3) + r" k=8: ([0-9.]+) →"
COL3_R = cell(3) + r" k=8: [0-9.]+ → k=12: \*\*([0-9.]+)"

ERR1 = ("勘误 ①（§7.9.4 表下 ⚠ 注）：k=12 当时只有 seed {42, 0}，两者 final 同为 0.900，"
        "σ_final 精确为 0。此格本就不该由 final 复现出来——它复现了才说明表被改过。")
add("7.9.4 col2 σ_final 0.064 — 不是 σ_final（勘误 ①上半）", ROW_F, COL2_R, "sd",
    PAIR_FINAL, expect="mismatch", note=ERR1)
add("7.9.4 col2 σ_final 0.064 — 实为同两 seed 的 mean39 σ（勘误 ①下半）", ROW_F, COL2_R,
    "sd", PAIR_MEAN, metric=MEAN_METRIC,
    note="勘误 ① 的诊断本身：同一格的 0.064 由这两个 seed 的 mean39 按 ddof=0 复现，"
         "与下一行 σ_mean39 第二列同值——重号的痕迹。")
add("7.9.4 col2 「2 seeds」的样本量", ROW_F, r"\(([0-9]+) seeds\) \| k=8:", "n",
    PAIR_FINAL,
    note="把 seed_[04]* 这个 glob 钉在该列自己印的样本量上：解析出 3 个文件就报错。")
add("7.9.4 col2 σ_final 0.181（k=8 三 seed，ddof=0）", ROW_F, COL2_L, "sd", FINAL8,
    note="勘误 ②：0.181 复现得出，但要三个 seed——它排在标着「2 seeds」的列里。"
         "样本量与列标题的矛盾属表述层，本工具只能证前半句。")
add("7.9.4 col3 σ_final k=8（ddof=1）", ROW_F, COL3_L, "sd", FINAL8)
add("7.9.4 col3 σ_final k=12（ddof=1）", ROW_F, COL3_R, "sd", FINAL12)

add("7.9.4 col2 σ_mean39 k=8 0.157（三 seed，ddof=0）", ROW_M, COL2_L, "sd", MEAN8,
    metric=MEAN_METRIC,
    note="勘误 ③：同样是三 seed 的值排在 2-seed 列里。")
add("7.9.4 col2 σ_mean39 k=12 0.064（两 seed，ddof=0）", ROW_M, COL2_R, "sd", PAIR_MEAN,
    metric=MEAN_METRIC)
add("7.9.4 col3 σ_mean39 k=8（ddof=1）", ROW_M, COL3_L, "sd", MEAN8, metric=MEAN_METRIC)
add("7.9.4 col3 σ_mean39 k=12（ddof=1）", ROW_M, COL3_R, "sd", MEAN12,
    metric=MEAN_METRIC)

# ------------------------------------------------------------- the prose that quotes σ
add("7.9.1 F1.1 σ_final = 0.181", r"^\*\*Finding F1\.1", r"σ_final = ([0-9.]+)", "sd",
    FINAL8)
add("7.9.1 口径注 ddof=1 为 0.222", r"^> final 值按 \*\*ddof=1\*\*",
    r"是 \*\*`([0-9.]+)`\*\*", "sd", FINAL8)
add("7.9.1 口径注 一致口径 0.222 → 0.038（ddof=1，左）", r"^> 故「0\.181 → 0\.038」",
    r"应为 `([0-9.]+) → [0-9.]+`（ddof=1）", "sd", FINAL8)
add("7.9.1 口径注 一致口径 0.222 → 0.038（ddof=1，右）", r"^> 故「0\.181 → 0\.038」",
    r"应为 `[0-9.]+ → ([0-9.]+)`（ddof=1）", "sd", FINAL12)
add("7.9.1 口径注 一致口径 0.181 → 0.031（ddof=0，右）", r"^> 故「0\.181 → 0\.038」",
    r"或 `[0-9.]+ → ([0-9.]+)`（ddof=0）", "sd", FINAL12)
add("7.9.2 F2.1 σ_mean39 k=8 三 seed 0.157", r"^\*\*Finding F2\.1",
    r"k=8 三 seed σ=([0-9.]+)", "sd", MEAN8, metric=MEAN_METRIC)
add("7.9.2 F2.1 σ_mean39 k=12 两 seed 0.064", r"^\*\*Finding F2\.1",
    r"k=12 两 seed σ=([0-9.]+)", "sd", PAIR_MEAN, metric=MEAN_METRIC)
add("7.9.4 ⚠ 注 第三列 σ_final 0.222", r"^> \*\*第三列（3-seed、ddof=1）经重算逐位无误\*\*",
    r"σ_final `([0-9.]+) →", "sd", FINAL8)
add("7.9.4 ⚠ 注 第三列 σ_final 0.038", r"^> \*\*第三列（3-seed、ddof=1）经重算逐位无误\*\*",
    r"σ_final `[0-9.]+ → ([0-9.]+)`", "sd", FINAL12)
add("7.9.4 ⚠ 注 第三列 σ_mean39 0.192", r"^> \*\*第三列（3-seed、ddof=1）经重算逐位无误\*\*",
    r"σ_mean39 `([0-9.]+) →", "sd", MEAN8, metric=MEAN_METRIC)
add("7.9.4 ⚠ 注 第三列 σ_mean39 0.113", r"^> \*\*第三列（3-seed、ddof=1）经重算逐位无误\*\*",
    r"σ_mean39 `[0-9.]+ → ([0-9.]+)`", "sd", MEAN12, metric=MEAN_METRIC)

# -------------------------------------------------------------- 7.10 the sensing grid
SEC710 = r"^### 7\.10 "
PROBES = [("s0", r"^\| s0（10-D"), ("s1", r"^\| s1（12-D"), ("s2", r"^\| s2（16-D")]
for probe, anchor in PROBES:
    for c, seed in enumerate((0, 7, 42), start=1):
        add(f"7.10 {probe} seed={seed}", anchor, num(c), "mean",
            [f"{VAN}/{probe}_k4/seed_{seed}/results/final_eval.json"], section=SEC710)
    glob3 = [f"{VAN}/{probe}_k4/seed_*/results/final_eval.json"]
    add(f"7.10 {probe} mean", anchor, mean_of(4), "mean", glob3, section=SEC710)
    add(f"7.10 {probe} std", anchor, std_of(4), "sd", glob3, section=SEC710)
add("7.10 gap(s1 − s0)", r"^gap\(s1 − s0\) = ", r"= ([0-9.]+)（约", "delta",
    [f"{VAN}/s0_k4/seed_*/results/final_eval.json",
     f"{VAN}/s1_k4/seed_*/results/final_eval.json"], section=SEC710)

spec = {
    "chain": "arrival_v2",
    "doc": "docs/arrival_v2_experiment_report.md",
    "note": (
        "provenance：本报告 §7.10 自述的引用链「论文 → 本报告 → final_eval.json」，"
        "以及 §7.9 各表自述的 gate JSON 源。逐 run 的 mean39 / peak / 首达步 / n_succ "
        "取自该 run 的 results/eval_log.csv（39 次周期评估），final 取自 results/final_eval.json。"
        "OOB 列未落表：它要 eval_termination_counts 的分项除以 num_eval_episodes，"
        "本工具只读平铺键。6380082 的主发现（论文侧已改、报告侧未回填）属表述层，同样不在覆盖内。"),
    "root": "experiments/arrival_v2_prototype",
    "metric": "eval_success_rate",
    "claims": claims,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"{len(claims)} claims -> {OUT}")
