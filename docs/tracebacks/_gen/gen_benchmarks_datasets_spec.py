"""Generate docs/tracebacks/benchmarks_and_datasets.json -- the `offline_data/` provenance family.

Why a twelfth chain rather than more claims on the seventh. `data_integrity_open_items.json`
is rooted at `results/offline` and says so in its own `note`: items 2/3/5 of that document
"whose sources are offline_data/ and benchmarks/" were left out because a spec carries one
`root`. This is that root. The two families really are different -- one recomputes a model's
published success rate from per-seed evaluation readouts, the other reads a collection run's
own metadata -- and keeping them apart is what lets `--root` point either one at a Drive
mount without dragging the other along.

What it covers, and where each provenance rule comes from:

  §1 核实块   The four values quoted verbatim from the 2000-episode dataset's `metadata.json`
              (`seed` / `num_episodes` / `num_transitions` / `success_rate`). The document
              prints them as a code block claiming to be that file; this checks that it is.
  §2 表       The 本机实测 column: `num_transitions` for the three deterministic datasets,
              and the `mean_episode_length` each cell shows its arithmetic from. Also the
              2026-08-24 verification block's two noisy-variant counts, which are what rules
              hypothesis (b) out.
  §4         The collector's own `success_rate` 0.958, the figure that whole item is about.
  §5 核实块   `seed` and `num_episodes` for the noisy 2000-episode dataset -- the two numbers
              the contamination finding rests on.

Deliberately not covered, so the count is not read as completeness:

  * the upper end of every `seeds 0..1999`. It is `seed + num_episodes - 1`, an expression
    over two metrics, and this tool reads one metric per source. Both operands are pinned,
    so a drift cannot hide -- it would surface on the lower bound or the count.
  * `100/100`, `40/40` and the other intersection counts. Those are set intersections
    between a dataset range and a manifest, which is `scripts/audit_seed_overlap.py`'s range
    pass, tested separately; the identity half needs the flow field and an env replay.
  * every episode count and seed range of a `benchmarks/` manifest, including §3 entirely.
    Those are `scripts/check_benchmark_manifests.py`'s R3, which reads the manifest rather
    than a metric and can therefore also check the counts this tool has no source for.
  * the paper-side 2.4e5 column, typeset there in maths. It is the figure the item says is
    wrong; it has no source to recompute from, which is the finding.
  * `0.928`, the ReBRAC-Q worldcomp-1000 success rate §4 compares against. That is a
    `results/` readout and belongs to the ReBRAC chain's root, not this one.
  * the dataset *counts* the document quotes in prose (「现存 9 个数据集」, 「现存 10 个」).
    They are dated snapshots of a gitignored directory that has since grown; pinning them
    would fail on every machine for a reason that is not a defect.

Usage:
    python docs/tracebacks/_gen/gen_benchmarks_datasets_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"
DOC = "docs/data_integrity_open_items.md"

CROSS = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep%s/metadata.json"
NOISY = ("crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_"
         "noise0p05clip0p15_ep%s/metadata.json")
WORLD = "worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/metadata.json"

# The four scopes. Each must match exactly one line: this document carries two
# `### ✅ 核实结论（...）` headings whose prefixes differ only in the date.
SEC_SEED = r"^### ✅ 核实结论（2026-08-09）：\*\*`base_seed`"
SEC_TABLE = r"^## 2\. `paper` Table 1 的 transitions"
SEC_BEHAVIOUR = r"^## 4\. 一条不算缺陷但会被问到的"
SEC_NOISY = r"^### ✅ 核实结论（2026-08-16 Drive 侧探针）"

claims: list[dict] = []


def claim(label: str, section: str, anchor: str, capture: str, source: str, metric: str,
          expected: str) -> None:
    entry = {
        "label": label,
        "section": section,
        "anchor": anchor,
        "capture": capture,
        "stat": "mean",
        "sources": [source],
        "metric": metric,
    }
    claims.append(entry)
    _verify(section, anchor, capture, expected, label)


def _verify(section: str, anchor: str, capture: str, expected: str, label: str) -> None:
    """Resolve against the live document now, so a mis-pointed capture fails here.

    The expectation lives in this generator and never reaches the spec -- a spec that
    carried the figure would keep passing after the document changed, which is the one
    thing these tables exist to notice.

    `capture` is applied to the whole line, alone, because that is what the tool does.
    Verifying `anchor + capture` concatenated instead is how 28 claims in the worldcomp
    chain passed generation with their column two cells off.
    """
    lines = (REPO / DOC).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    heads = [i for i, ln in enumerate(lines) if re.search(section, ln)]
    assert len(heads) == 1, f"{label}: section matched {len(heads)} lines"
    lo = heads[0]
    level = len(re.match(r"^(#{1,6}) ", lines[lo]).group(1))
    hi = next((i for i in range(lo + 1, len(lines))
               if re.match(r"^(#{1,6}) ", lines[i])
               and len(re.match(r"^(#{1,6}) ", lines[i]).group(1)) <= level), len(lines))
    hits = [i for i in range(lo, hi) if re.search(anchor, lines[i])]
    assert len(hits) == 1, f"{label}: anchor matched {len(hits)} lines in section"
    found = re.search(capture, lines[hits[0]])
    assert found, f"{label}: capture did not fire on {lines[hits[0]]!r}"
    assert found.group(1) == expected, \
        f"{label}: captured {found.group(1)!r}, expected {expected!r}"


# --- §1: the metadata block the document quotes as the dataset's own ------------------------
# Anchored on the shape of the quoted line, not on any value in it: a value anchor turns a
# changed figure into "anchor missing" instead of the "value mismatch" it is.
BLOCK = r'^"seed": .*"num_transitions"'
for field, capture, expected in (
        ("seed", r'"seed": ([0-9]+)', "0"),
        ("num_episodes", r'"num_episodes": ([0-9]+)', "2000"),
        ("num_transitions", r'"num_transitions": ([0-9]+)', "304967"),
        ("success_rate", r'"success_rate": ([0-9.]+)', "0.8855")):
    claim(f"§1 核实块 | {field}", SEC_SEED, BLOCK, capture, CROSS % "2000", field, expected)

# --- §2: the 本机实测 column, and the arithmetic each cell shows ---------------------------
for alias, source, transitions, mel in (
        ("crosscomp-1000", CROSS % "1000", "152,683", "152.683"),
        ("crosscomp-2000", CROSS % "2000", "304,967", "152.4835"),
        ("worldcomp-1000", WORLD, "104,360", "104.36")):
    row = r"^\| `" + alias + r"` \|"
    claim(f"§2 表 | {alias} num_transitions", SEC_TABLE, row,
          r"\*\*([0-9,]+)\*\*", source, "num_transitions", transitions)
    claim(f"§2 表 | {alias} mean_episode_length", SEC_TABLE, row,
          r"（= (?:`mean_episode_length` )?([0-9.]+) ×", source,
          "mean_episode_length", mel)

# The 2026-08-24 block's noisy-variant counts: the measurement that rules out hypothesis (b).
for episodes, expected in (("1000", "156,882"), ("2000", "313,719")):
    claim(f"§2 排除 (b) | 加噪 {episodes} num_transitions", SEC_TABLE,
          r"^   \| `crosscomp_\.\.\._noise0p05clip0p15_ep" + episodes + r"` \|",
          r"\*\*([0-9,]+)\*\*", NOISY % episodes, "num_transitions", expected)

# --- §4: the collection policy's own success rate ------------------------------------------
claim("§4 | 采集策略 success_rate", SEC_BEHAVIOUR,
      r"^`offline_data/worldcomp_\.\.\._ep1000/metadata\.json`",
      r"\*\*([0-9.]+)\*\*", WORLD, "success_rate", "0.958")

# --- §5: the two numbers the noisy-2000 contamination finding rests on ---------------------
claim("§5 核实块 | 含噪 2000 起始种子", SEC_NOISY,
      r"^dataset : crosscomp_.*seeds [0-9]+\.\.[0-9]+$",
      r"seeds ([0-9]+)\.\.", NOISY % "2000", "seed", "0")
claim("§5 正文 | 含噪 2000 起始种子（复述）", SEC_NOISY,
      r"^`seed=[0-9]+`、[0-9]+ 回合 → 训练区间",
      r"`seed=([0-9]+)`", NOISY % "2000", "seed", "0")
claim("§5 正文 | 含噪 2000 回合数", SEC_NOISY,
      r"^`seed=[0-9]+`、[0-9]+ 回合 → 训练区间",
      r"、([0-9]+) 回合", NOISY % "2000", "num_episodes", "2000")

spec = {
    "chain": "benchmarks_and_datasets",
    "doc": DOC,
    "root": "offline_data",
    "metric": "num_transitions",
    "note": (
        "The offline_data/ half of docs/data_integrity_open_items.md, split off from "
        "data_integrity_open_items.json because a spec carries one root and that one is "
        "rooted at results/offline. Every claim reads a collection run's own "
        "metadata.json: §1's quoted four-field block, §2's 本机实测 column plus the "
        "mean_episode_length each cell shows its arithmetic from, the two noisy-variant "
        "counts that rule out hypothesis (b), §4's collector success_rate, and §5's seed "
        "and num_episodes for the noisy 2000-episode set. Deliberately not covered: the "
        "upper end of every `seeds 0..1999` (seed + num_episodes - 1 is an expression over "
        "two metrics, and both operands are pinned separately); the 100/100 intersection "
        "counts (audit_seed_overlap's range and identity passes); every benchmarks/ "
        "manifest fact including §3 entirely (check_benchmark_manifests R3, which reads "
        "the manifest and can also check counts this tool has no source for); the paper-"
        "side ~2.4e5 column, which is the figure with no source and therefore the finding; "
        "0.928, a results/ readout under another root; and the prose dataset counts, which "
        "are dated snapshots of a gitignored directory that has since grown."),
    "claims": claims,
}

path = OUT_DIR / "benchmarks_and_datasets.json"
path.parent.mkdir(parents=True, exist_ok=True)
io.open(path, "w", encoding="utf-8", newline="\n").write(
    json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {path}: {len(claims)} claims")
