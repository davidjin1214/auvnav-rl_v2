"""Check the frozen evaluation manifests under `benchmarks/` against themselves and the docs.

Why this exists: `0dc35a2` (2026-08-21) brought 13 evaluation manifests off Drive into the
tracked `benchmarks/` tree, and its commit message records the acceptance pass as prose --
"落库前逐个核过——条数与目录名相符、种子连续、其中 8 个与审计打印的区间逐条对上、val_40
确为 test_100 的前缀". That pass was done by hand, on 13 of the 24 files, once. Nothing
re-runs it, and nothing looked at the other 11. This makes it one command over all of them.

These files are *records*, not derived artefacts: `auv_nav/env.py` changed after the
benchmark protocol was frozen in `9b96a7d`, so regenerating a manifest is not guaranteed to
reproduce it and a diff against the generator is not available as a check. What is available
is that the manifests constrain each other, and that several documents print their episode
counts and seed ranges as fact.

Four rules.

  R1 结构自洽    Per manifest, with no document involved:
                 * seeds ascend by exactly one, with no gap and no repeat;
                 * `episode_id` carries index 0..n-1 in file order -- this pairing is the
                   fingerprint `paper/thesis_ch5/tools/ch5_manifest_attribution.py` matches
                   3715 published readouts on, so a manifest that loses it silently
                   un-attributes every readout that used it;
                 * a path that declares a count holds that many (`{val,test}_{N}/` directory,
                   or an `_ep{N}` filename);
                 * the first seed is the one the generator would have used --
                   `BENCHMARK_SPECS[key].manifest_seed` for the key the filename names, or
                   `N` when the filename carries an explicit `_s{N}` reseed marker. The
                   marker is not an exemption: it moves the expected value, so
                   `clean_probe/..._s3000.json` is checked against 3000 rather than excused.

  R2 族内嵌套    Manifests sharing a task configuration (flow field, geometry, target speed --
                 the same triple `audit_seed_overlap` uses to decide two things are even
                 comparable) and a starting seed must nest: every one of them is the opening
                 segment of the longest. This is the machine form of "every val_40 is a
                 prefix of its sibling test_100", and it subsumes the epoch_probe pair being
                 one set rather than two.

  R3 文档所印    A markdown line naming a manifest and printing its episode count or its seed
                 range must agree with the file. `{a,b}` expands and a trailing `dir/...`
                 resolves to the single JSON under that directory, because the tables this
                 rule exists for -- `benchmarks/README.md`'s seed-range table and the audit
                 printout quoted in `docs/data_integrity_open_items.md` -- are written that
                 way and would otherwise be skipped in silence.

  R4 协议同规模  The eight default manifests `benchmarks/<key>.json`, one per catalog key, are
                 the frozen cells of one factorial protocol: `BENCHMARK_GROUPS` crosses them
                 and `run_suite` resolves exactly these paths. A cell evaluated on a different
                 number of episodes is not comparable with its siblings, the same reason
                 `num_envs` is held constant across a study. They must therefore all hold the
                 same count. Note that supplying a manifest makes `--eval-episodes` dead:
                 `train_utils._resolved_eval_episodes` returns the manifest's episodes and
                 ignores the requested number, so a short manifest is silent, not an error.

What these rules deliberately do NOT do. Two floats that went through JSON months apart can
differ in the last bit -- measured, `initial_heading` differs by up to 2.7e-15 rad between
manifests that are otherwise the same instance. R2 grades that as `float-noise` and prints
it rather than failing: at nine orders of magnitude below any real difference in an instance
it is a property of the formatter, not of the task. And nothing here replays the reset RNG;
whether a frozen instance is the one the seed actually produces is `audit_seed_overlap
--verify`, which needs the flow field and an env.

Buckets and grading follow `check_doc_pointers`: only the defect buckets fail --strict.

Usage:
    python -m scripts.check_benchmark_manifests            # report, always exit 0
    python -m scripts.check_benchmark_manifests --strict   # exit 1 if any defect bucket fills
    python -m scripts.check_benchmark_manifests --benchmarks-dir /mnt/drive/benchmarks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.benchmark_catalog import BENCHMARK_SPECS

ROOT = Path(__file__).resolve().parents[1]

# Directories whose markdown R3 does not read. Mirrors `build_doc_index`, kept local so a
# checker over `benchmarks/` does not inherit that module's index-specific exemptions --
# `benchmarks/README.md` is the single most important document this rule grades, and it
# lives inside a directory the doc index skips.
R3_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
                ".mypy_cache", ".ipynb_checkpoints", ".venv", "venv", ".tmp", "tmp",
                "worktrees", "results", "experiments", "checkpoints", "wake_data",
                "offline_data"}

# `val_40/`, `test_100/`, `test_40/` -- the directory naming the launchers generate into.
DIR_COUNT = re.compile(r"^(?:val|test)_(\d+)$")
# `..._ep100.json` -- the same declaration made in a filename instead of a directory.
NAME_COUNT = re.compile(r"_ep(\d+)$")
# `..._s3000.json` -- an explicit reseed, which moves the expected first seed rather than
# excusing the manifest from having one.
NAME_SEED = re.compile(r"_s(\d+)$")

# Two floats that made a round trip through JSON in different months may differ in the last
# bit. Measured worst case across this tree: 2.7e-15 rad on `initial_heading`. Nine orders
# below that is still nine orders above nothing and nine orders below a real difference.
FLOAT_NOISE = 1e-9

# --- R3 line parsing ---------------------------------------------------------------------
# A manifest path, with brace groups left intact for `_expand` to deal with.
DOC_PATH = re.compile(r"((?:benchmarks/)?[A-Za-z0-9_{}][A-Za-z0-9_{},./-]*"
                      r"(?:\.json|/\.\.\.))")
DOC_RANGE = re.compile(r"(\d{3,5})\s*\.\.\s*(\d{3,5})")
# `100 条` / `40 episodes`, and the `100 / 40 ep` form the tables use for a brace pair.
DOC_COUNT = re.compile(r"(?<![\d./])((?:\d{1,4}\s*/\s*)*\d{1,4})\s*(?:条|回合|episodes|ep\b)")
# `前 30 条` / `首 40 回合` says *which* episodes, not how many the file holds. Filtered
# after matching rather than by a lookbehind: the quantifier is variable-width because the
# space is optional, and `re` will not take a variable-width lookbehind.
COUNT_IS_A_SLICE = re.compile(r"[前首]\s*$")
# A table cell holding nothing but numbers: how `benchmarks/README.md`'s seed-range table
# writes the count, under an `Episodes` header this rule does not read.
CELL_COUNT = re.compile(r"(?:^|\|)\s*(\d{1,4}(?:\s*/\s*\d{1,4})*)\s*(?=\||$)")
BRACE = re.compile(r"\{([^{}]*)\}")

# How far *after* the path a figure may sit and still be read as a statement about it.
# Only after: `- 100 episodes，前 30 与 \`single_u10_cross_tgt15.json\` byte-identical` puts a
# count 14 characters ahead of the path whose subject is the bullet above, and reading
# backwards calls that correct sentence wrong. Measured against the lines this rule exists
# for, the widest genuine gap is 12 characters (`：100 条，种子 **1250..1349**`).
FIGURE_WINDOW = 40

# A figure a document keeps on purpose because it records what was planned or believed at
# the time. Same shape as `check_doc_pointers`'s 自述缺席 bucket: the declaration must sit on
# the same line as the figure it excuses, and must be dated, so a reader can check it. And
# the same risk applies -- a false declaration silences its own alarm permanently.
DECLARED = re.compile(r"20\d{2}-\d{2}-\d{2}[^。\n]{0,40}?(?:注|订正|当时|计划值|历史)")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def manifests(benchmarks_dir: Path) -> dict[str, dict]:
    """Every JSON under the tree that carries an `episodes` list, keyed by relative path."""
    out = {}
    for p in sorted(benchmarks_dir.rglob("*.json")):
        try:
            payload = _read(p)
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            out[p.relative_to(benchmarks_dir).as_posix()] = payload
    return out


def declared_count(rel: str) -> tuple[int, str] | None:
    """What the path itself says the episode count is, and which part of it said so."""
    parts = rel.split("/")
    if len(parts) > 1:
        m = DIR_COUNT.match(parts[-2])
        if m:
            return int(m.group(1)), f"目录 {parts[-2]}/"
    m = NAME_COUNT.search(Path(rel).stem)
    if m:
        return int(m.group(1)), f"文件名 {Path(rel).name}"
    return None


def expected_first_seed(rel: str) -> tuple[int, str] | None:
    """The seed the generator would have started from, and where that expectation comes from.

    An `_s{N}` marker wins over the catalog: `clean_probe/single_u10_cross_tgt15_ep100_s3000`
    is a deliberate reseed of a catalog key, and reading the key alone would call it wrong.
    """
    stem = Path(rel).stem
    m = NAME_SEED.search(stem)
    if m:
        return int(m.group(1)), f"文件名的 _s{m.group(1)} 重播种标记"
    spec = BENCHMARK_SPECS.get(stem)
    if spec is not None:
        return int(spec.manifest_seed), f"catalog 里 {stem} 的 manifest_seed"
    return None


def task_key(payload: dict) -> tuple[str, str, float]:
    """Flow field, geometry, target speed -- `audit_seed_overlap`'s test for comparability."""
    options = payload.get("base_reset_options") or {}
    return (
        str(payload.get("flow_path")),
        str(options.get("task_geometry")),
        float(options.get("target_auv_max_speed_mps") or 0.0),
    )


def _instance(episode: dict) -> tuple:
    options = episode["reset_options"]
    return (
        float(options["flow_time"]),
        tuple(float(v) for v in options["start_xy"]),
        tuple(float(v) for v in options["goal_xy"]),
        float(options["initial_heading"]),
    )


def _flat(value) -> list[float]:
    out: list[float] = []
    for item in value:
        if isinstance(item, tuple):
            out.extend(item)
        else:
            out.append(item)
    return out


def nesting(short: list[dict], long: list[dict]) -> tuple[str, str]:
    """Is `short` the opening segment of `long`? Returns a verdict and what separates them."""
    if len(short) > len(long):
        return "not-nested", f"更长（{len(short)} > {len(long)}）"
    worst, worst_at = 0.0, 0
    for idx, (a, b) in enumerate(zip(short, long)):
        if int(a["seed"]) != int(b["seed"]):
            return "not-nested", f"第 {idx} 条种子 {a['seed']} ≠ {b['seed']}"
        gaps = [abs(x - y) for x, y in zip(_flat(_instance(a)), _flat(_instance(b)))]
        if max(gaps) > worst:
            worst = max(gaps)
            worst_at = idx
    if worst == 0.0:
        return "nested", f"{len(short)}/{len(long)} 逐条相同"
    if worst <= FLOAT_NOISE:
        return "float-noise", (f"{len(short)}/{len(long)} 相同，最大偏差 {worst:.3g}"
                               f"（第 {worst_at} 条）")
    return "not-nested", f"第 {worst_at} 条实例相差 {worst:.3g}"


def _expand(pattern: str) -> list[str]:
    """`a/{b,c}/d` -> `a/b/d`, `a/c/d`. The seed-range tables are written this way."""
    m = BRACE.search(pattern)
    if not m:
        return [pattern]
    out = []
    for option in m.group(1).split(","):
        out.extend(_expand(pattern[:m.start()] + option.strip() + pattern[m.end():]))
    return out


def resolve(raw: str, known: dict[str, dict]) -> list[str] | None:
    """A path as a document writes it -> the manifests it names, or None if it names none."""
    hits = []
    for candidate in _expand(raw):
        rel = candidate[len("benchmarks/"):] if candidate.startswith("benchmarks/") else candidate
        if rel.endswith("/..."):
            here = [k for k in known if k.startswith(rel[:-3]) and "/" not in k[len(rel) - 3:]]
            if len(here) != 1:
                return None  # ambiguous shorthand: refuse rather than pick
            hits.append(here[0])
        elif rel in known:
            hits.append(rel)
        else:
            # Prose names a file without its directory (`single_u10_cross_tgt15_ep100_s3000
            # .json`). Accept only when the basename is unique in the tree -- the bare
            # `single_u10_cross_tgt15.json` is not, and guessing which of four is meant is
            # how a checker starts inventing findings.
            same = [k for k in known if k.rsplit("/", 1)[-1] == rel]
            if len(same) != 1:
                return None
            hits.append(same[0])
    return hits or None


def _near(matches, span: tuple[int, int]) -> list:
    """Matches that follow the cited path within the window, in document order."""
    out = []
    for m in matches:
        gap = m.start() - span[1]
        if 0 <= gap <= FIGURE_WINDOW:
            out.append((m, gap))
    return out


def _split(text: str) -> list[int]:
    return [int(part) for part in text.split("/")]


def figures(line: str, span: tuple[int, int]):
    """The counts and the seed ranges this line states about the path at `span`."""
    counts = _near((m for m in DOC_COUNT.finditer(line)
                    if not COUNT_IS_A_SLICE.search(line[:m.start()])), span)
    if not counts:
        # `benchmarks/README.md`'s seed-range table writes the count as a bare table cell,
        # under a header this rule does not read.
        counts = _near(CELL_COUNT.finditer(line), span)
    ranges = _near(DOC_RANGE.finditer(line), span)
    return ([(value, gap) for m, gap in counts for value in _split(m.group(1))],
            [((int(m.group(1)), int(m.group(2))), gap) for m, gap in ranges])


def pair(figs: list, targets: list[str]):
    """Which figure describes which manifest, or None when the line does not say.

    One figure among several manifests is a claim about all of them, and the seed-range
    table writes it that way (`offline_rebrac_{broad,screen,worldcomp_final}/test_100/...`
    | 100 | 1250..1349). k figures among k manifests pair in document order, which is the
    other way that table writes it (`{test_100,val_40}` | 100 / 40 | 1400..1499 / 1400..1439).
    Several figures about one manifest resolve to the nearest, which is what makes
    `- 100 episodes，前 30 与 X byte-identical` read as the sentence it is. Anything else --
    four manifests against two figures, in the equivalence-class table -- is prose about a
    group, and picking a pairing there would be invention.
    """
    if not figs:
        return None
    if len(figs) == len(targets):
        return [(value, t) for (value, _), t in zip(figs, targets)]
    if len(figs) == 1:
        return [(figs[0][0], t) for t in targets]
    if len(targets) == 1:
        return [(min(figs, key=lambda f: f[1])[0], targets[0])]
    return None


def md_files() -> list[Path]:
    return [p for p in sorted(ROOT.rglob("*.md")) if not (R3_SKIP_DIRS & set(p.parts))]


def scan(benchmarks_dir: Path) -> dict[str, list[str]]:
    known = manifests(benchmarks_dir)
    buckets: dict[str, list[str]] = {k: [] for k in (
        "seed-gap", "id-drift", "count-mismatch", "seed-origin", "not-nested",
        "doc-mismatch", "cohort", "float-noise", "doc-unresolved",
        "declared", "ok")}

    # ---- R1 ------------------------------------------------------------------------------
    for rel, payload in known.items():
        eps = payload["episodes"]
        seeds = [int(e["seed"]) for e in eps]
        checks = 0
        if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            gaps = [i for i in range(1, len(seeds)) if seeds[i] != seeds[i - 1] + 1]
            buckets["seed-gap"].append(
                f"{rel}  种子非连续，首处断点在第 {gaps[0]} 条（{seeds[gaps[0] - 1]} → "
                f"{seeds[gaps[0]]}）" if gaps else f"{rel}  种子非升序连续")
        else:
            checks += 1
        bad = [i for i, e in enumerate(eps)
               if not str(e.get("episode_id", "")).endswith(f"_{i:04d}")]
        if bad:
            buckets["id-drift"].append(
                f"{rel}  episode_id 的序号与位置不符，首处在第 {bad[0]} 条"
                f"（{eps[bad[0]].get('episode_id')!r}）")
        else:
            checks += 1
        want = declared_count(rel)
        if want is not None:
            if want[0] != len(eps):
                buckets["count-mismatch"].append(
                    f"{rel}  {want[1]} 声称 {want[0]} 条，实际 {len(eps)} 条")
            else:
                checks += 1
        origin = expected_first_seed(rel)
        if origin is not None:
            if origin[0] != seeds[0]:
                buckets["seed-origin"].append(
                    f"{rel}  首个种子 {seeds[0]}，而 {origin[1]} 给的是 {origin[0]}")
            else:
                checks += 1
        buckets["ok"].extend([rel] * checks)

    # ---- R2 ------------------------------------------------------------------------------
    families: dict[tuple, list[str]] = {}
    for rel, payload in known.items():
        seeds = [int(e["seed"]) for e in payload["episodes"]]
        families.setdefault(task_key(payload) + (seeds[0],), []).append(rel)
    for family in families.values():
        if len(family) < 2:
            continue
        longest = max(family, key=lambda r: len(known[r]["episodes"]))
        for rel in sorted(family):
            if rel == longest:
                continue
            verdict, why = nesting(known[rel]["episodes"], known[longest]["episodes"])
            where = f"{rel}  ⊂ {longest}：{why}"
            buckets["ok" if verdict == "nested" else verdict].append(where)

    # ---- R3 ------------------------------------------------------------------------------
    for p in md_files():
        rel_doc = p.relative_to(ROOT).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        for lineno, line in enumerate(text.split("\n"), 1):
            if not (DOC_RANGE.search(line) or DOC_COUNT.search(line)
                    or CELL_COUNT.search(line)):
                continue
            for m in DOC_PATH.finditer(line):
                targets = resolve(m.group(1), known)
                if not targets:
                    continue
                counts, ranges = figures(line, m.span())
                for kind, unit, figs in (("条数", " 条", counts), ("种子区间", "", ranges)):
                    paired = pair(figs, targets)
                    if paired is None:
                        if figs:
                            buckets["doc-unresolved"].append(
                                f"{rel_doc}:{lineno}  一行里 {len(figs)} 个{kind}对 "
                                f"{len(targets)} 份 manifest，配不出唯一对应，未判")
                        continue
                    for value, target in paired:
                        seeds = [int(e["seed"]) for e in known[target]["episodes"]]
                        actual = len(seeds) if kind == "条数" else (seeds[0], seeds[-1])
                        head = f"{rel_doc}:{lineno}  ← {target}"
                        if value == actual:
                            buckets["ok"].append(head)
                        elif DECLARED.search(line):
                            buckets["declared"].append(
                                f"{head}\n      文档印 {value}{unit}，实际 {actual}"
                                f"\n      本行声明：{DECLARED.search(line).group(0)}")
                        else:
                            buckets["doc-mismatch"].append(
                                f"{head}\n      文档印 {value}{unit}，实际 {actual}")

    # ---- R4 ------------------------------------------------------------------------------
    cohort = {key: known[f"{key}.json"] for key in BENCHMARK_SPECS
              if f"{key}.json" in known}
    sizes = {key: len(payload["episodes"]) for key, payload in cohort.items()}
    if len(set(sizes.values())) > 1:
        protocol = max(set(sizes.values()), key=list(sizes.values()).count)
        for key, size in sorted(sizes.items()):
            if size == protocol:
                buckets["ok"].append(f"{key}.json")
            else:
                buckets["cohort"].append(
                    f"{key}.json  {size} 条，而同为协议默认清单的其余 "
                    f"{len(sizes) - 1} 份里多数是 {protocol} 条")
    else:
        buckets["ok"].extend(f"{key}.json" for key in sorted(sizes))
    return buckets


# `★` marks a defect section, matching `check_status_claims`: the PostToolUse hook greps for
# it to decide which part of the report to hand back.
DEFECT = ("seed-gap", "id-drift", "count-mismatch", "seed-origin", "not-nested",
          "doc-mismatch", "cohort")
TITLES = {
    "seed-gap": "★ 种子不是连续升序",
    "id-drift": "★ episode_id 的序号与位置不符（读数归属的指纹就此失效）",
    "count-mismatch": "★ 条数与路径自己声称的不符",
    "seed-origin": "★ 首个种子不是生成器会用的那个",
    "not-nested": "★ 同一任务配置同一起始种子，短的不是长的开头一段",
    "doc-mismatch": "★ 文档印的条数／种子区间与文件不符",
    "cohort": "★ 协议默认清单之间条数不齐（同一因子设计的各格不可比）",
    "float-noise": "嵌套成立，但浮点末位有差（提示，不判失败）",
    "doc-unresolved": "印了条数／区间、但路径落不到本树上的行（提示，不判失败）",
    "declared": "文档自述该数是当时的计划值／历史值（声明不是事实，见下）",
}


def main(argv: list[str] | None = None) -> int:
    # The PostToolUse hook captures stdout through a pipe, which on Windows defaults to
    # cp936, where `★` and `⊂` are not encodable. Without this the report dies mid-print and
    # the hook reads the crash as a finding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any defect bucket is non-empty")
    ap.add_argument("--benchmarks-dir", default=str(ROOT / "benchmarks"),
                    help="the manifest tree to check (a Drive mount, for instance)")
    args = ap.parse_args(argv)

    buckets = scan(Path(args.benchmarks_dir))
    defects = sum(len(buckets[b]) for b in DEFECT)
    graded = sum(len(v) for k, v in buckets.items() if k != "ok")
    # Coverage, printed rather than left implicit: a rule that quietly stops matching
    # anything reads exactly like a rule that finds nothing wrong.
    docs = len({row.split("\n")[0].split("  ←")[0]
                for name in ("ok", "doc-mismatch", "declared", "doc-unresolved")
                for row in buckets[name] if ".md:" in row})

    print(f"评估清单核查：{len(manifests(Path(args.benchmarks_dir)))} 份清单，"
          f"吻合 {len(buckets['ok'])} 项 / 提出 {graded} 项；R3 判了 {docs} 行文档")
    for name in DEFECT + ("float-noise", "doc-unresolved", "declared"):
        rows = buckets[name]
        if not rows:
            continue
        print(f"\n--- {TITLES[name]}：{len(rows)} ---")
        for row in rows:
            print(f"  {row}")
    if buckets["declared"]:
        print("\n  ⚠ 自述桶装的是声明，不是事实。改动那一行时要一并复核这条声明，"
              "否则它会永久静音自己的告警。")
    print()
    if defects and args.strict:
        print(f"FAIL: {defects} 处缺陷（--strict）")
        return 1
    print("OK: 无缺陷" if not defects else f"{defects} 处缺陷（未加 --strict，不判失败）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
