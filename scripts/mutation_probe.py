"""Run a batch of negative controls: disable one mechanism, assert one named check turns red.

Why this exists: the integrity-audit line was hardened in five batches, and each batch was
graded by an injection script written from scratch in that session's scratchpad -- five
rewrites of the same engine, none of which survived the session. That is precisely the defect
the line is about. `docs/handoff/2026-08-23-integrity-audit-pytest.md` records the decision as
open twice ("要不要把注入器收进 `scripts/`", §4 and §9). This is that decision taken: the
engine lives here, and each batch keeps only its own table of rows under `scripts/mutations/`.

What a row is
-------------
`(label, kind, named-check, [edits])`. Each row disables exactly one mechanism and names the
ONE check that exists to catch it -- never a suite selector, because a union lets an unrelated
failure stand in for the one being graded.

Two kinds of named check, because the guards are not all pytest:

  pytest  a single test id (`tests/test_x.py::test_y`). This is the layer that still exists
          in a clone with no `results/` and no `offline_data/`.
  sweep   a repo-wide run of a graded tool, named `<tool>:<bucket>` and passed only if THAT
          bucket gains a row. Registered tools are in `SWEEPS`.

An edit is one of:

  ("text",  relpath, old, new)                 literal replace; must occur exactly once
  ("claim", relpath, claim-label, key, value)  set a spec claim's key; None deletes it

The `claim` form exists because a spec is JSON: a literal replace over re-serialised JSON is
brittle, while addressing the claim by label survives a reordering.

Three self-checks, all learned the hard way
-------------------------------------------
  A. the named check must actually pass on the unmutated tree first. A typo'd test id
     collects nothing and pytest exits non-zero, which reads exactly like a caught mutation;
     a bucket name that no longer exists never gains a row and reads exactly like a green.
  B. the mutation must land on disk, compared with line endings normalised. The working copy
     is CRLF and a raw checksum otherwise compares equal by accident.
  C. the named case must still be collectable *after* the injection. A row that leaves the
     module unimportable also exits non-zero -- same symptom as A, other end of the run.
     Found on the first run of `mutation_probe_self`, where row 8 read green on a syntax
     error rather than on the clause it names.

When a row stops applying
-------------------------
`("text", ...)` edits quote the source they disable, so editing that source makes the row fail
with "注入点出现 0 次". **That is the intended signal, not a broken tool**: the mechanism moved
and its negative control has to move with it. Same for a renamed test id -- self-check A fires.
Rows are evidence about a specific line of code; they are not expected to be refactor-proof.

Guard rails
-----------
Rows edit the real tree and restore from an in-memory backup in a `finally`. A hard kill
between the two loses the working copy, so this refuses to start on a dirty tree unless
`--allow-dirty` is given, and re-checks `git status` at the end. A sweep-layer row edits the
*document* that prints a figure, never the file the figure came from: manifests, datasets and
readouts are the evidence, and `is_evidence()` refuses them (markdown under those trees is a
document about the evidence, and is allowed).

Usage
-----
    python -m scripts.mutation_probe --list
    python -m scripts.mutation_probe benchmarks_manifests
    python -m scripts.mutation_probe tracebacks_chains --only 13,15,16 --out report.txt
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_PACKAGE = "scripts.mutations"

# Directories holding evidence: a manifest, a dataset, a readout. Never an injection target.
# Markdown under them is exempt -- `benchmarks/README.md` prints the counts and seed ranges the
# `cbm:doc-mismatch` rows exist to test, so it is a document that quotes the evidence, not the
# evidence. (`mutate_benchmarks.py` stated the rule as "不动 `benchmarks/` 下任何文件" while two
# of its own rows edited that README; the rule it was actually following is this one.)
PROTECTED = ("benchmarks/", "offline_data/", "results/", "wake_data/")
PROTECTED_EXEMPT_SUFFIX = ".md"


def is_evidence(rel: str) -> bool:
    rel = str(rel).replace("\\", "/")
    return rel.startswith(PROTECTED) and not rel.endswith(PROTECTED_EXEMPT_SUFFIX)


# --------------------------------------------------------------------------- sweep probes
# Each returns {bucket: count} for one graded tool. They run in a *subprocess* (see
# `sweep_buckets`): a row that edits a tool's source is invisible to a module this process
# already imported, so grading in-process would report the pre-mutation tree as green.

def _probe_apn(root: Path) -> dict[str, int]:
    from scripts import audit_published_numbers as apn
    specs = apn.load_specs(apn.SPEC_DIR)
    merged = apn.merge([apn.run_spec(s, None, False) for s in specs])
    return {k: len(v) for k, v in merged.items()}


def _probe_cbm(root: Path) -> dict[str, int]:
    from scripts import check_benchmark_manifests as cbm
    return {k: len(v) for k, v in cbm.scan(root / "benchmarks").items()}


SWEEPS = {"apn": _probe_apn, "cbm": _probe_cbm}


def sweep_buckets(tools: list[str], log: Path | None, root: Path, python: str) -> dict:
    """Run each named tool in its own interpreter; return {"<tool>:<bucket>": count}."""
    out: dict[str, int] = {}
    text = ""
    for tool in tools:
        r = subprocess.run([python, "-X", "utf8", "-m", "scripts.mutation_probe",
                            "--sweep-probe", tool],
                           cwd=root, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        text += f"--- {tool}\n{r.stdout or ''}{r.stderr or ''}"
        if r.returncode != 0:
            raise RuntimeError(f"sweep probe {tool} failed:\n{r.stderr}")
        for name, count in json.loads(r.stdout).items():
            out[f"{tool}:{name}"] = count
    if log is not None:
        log.write_text(text, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- edit engine

def digest(path: Path) -> str:
    """Content hash with line endings normalised -- a CRLF working copy defeats a raw one."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def apply_edit(edit, root: Path = ROOT) -> None:
    kind, rel = edit[0], edit[1]
    if is_evidence(rel):
        raise AssertionError(f"{rel} 是证据文件，不得注入")
    path = root / rel
    raw = path.read_bytes()

    if kind == "text":
        _, _, old, new = edit
        crlf = b"\r\n" in raw
        norm = raw.decode("utf-8").replace("\r\n", "\n")
        needle = old.replace("\r\n", "\n")
        n = norm.count(needle)
        if n != 1:
            raise AssertionError(f"注入点在 {rel} 中出现 {n} 次（应为 1）")
        out = norm.replace(needle, new.replace("\r\n", "\n"), 1)
        path.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
        return

    if kind == "claim":
        _, _, label, key, value = edit
        spec = json.loads(raw.decode("utf-8"))
        hits = [c for c in spec["claims"] if c["label"] == label]
        if len(hits) != 1:
            raise AssertionError(f"{rel} 里标签 {label!r} 命中 {len(hits)} 条（应为 1）")
        if value is None:
            del hits[0][key]
        else:
            hits[0][key] = value
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
        return

    raise AssertionError(f"未知的 edit 类型 {kind!r}")


def _pytest(args: list[str], root: Path, python: str, basetemp: Path):
    return subprocess.run([python, "-X", "utf8", "-m", "pytest", *args,
                           "--basetemp", str(basetemp)],
                          cwd=root, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def pytest_red(selector: str, log: Path | None, root: Path, python: str,
               basetemp: Path) -> bool:
    r = _pytest(["-q", "--tb=line", selector], root, python, basetemp)
    if log is not None:
        log.write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    return r.returncode != 0


def pytest_collects(selector: str, root: Path, python: str, basetemp: Path) -> bool:
    """Does the named case still exist and import cleanly?

    Self-check C, learned 2026-08-24 while grading this engine with itself: a row that makes
    the module unimportable turns pytest red for a reason that has nothing to do with the
    mechanism -- collection fails, the exit code is non-zero, and it reads exactly like a
    caught mutation. Run *after* the injection; self-check A already covers the before side.
    """
    return _pytest(["--collect-only", "-q", selector], root, python, basetemp).returncode == 0


# --------------------------------------------------------------------------- batch runner

def run_batch(rows, root: Path, python: str, out_dir: Path | None) -> tuple[list[str], int]:
    """Run every row; return (report lines, count of rows that did not turn red)."""
    lines: list[str] = []
    bad = 0

    def log(name: str) -> Path | None:
        return None if out_dir is None else out_dir / name

    tools = sorted({c.split(":")[0] for _, k, c, _ in rows if k == "sweep"})
    base = sweep_buckets(tools, log("00_baseline.txt"), root, python) if tools else {}
    if tools:
        lines.append("基线桶：" + json.dumps({k: v for k, v in base.items() if v},
                                             ensure_ascii=False))

    basetemp = Path(tempfile.mkdtemp(prefix="mutprobe_"))

    for i, (label, kind, check, edits) in enumerate(rows, 1):
        # --- self-check A: the named check must be green / empty before we touch anything
        if kind == "pytest":
            if pytest_red(check, log(f"{i:02d}_pre.txt"), root, python, basetemp):
                lines.append(f"{label}\n    ✗ 自检 A 失败：注入前该用例就不是绿的")
                bad += 1
                continue
        elif base.get(check, 0) != 0:
            where = "不为 0" if check in base else "不存在（桶名已改？）"
            lines.append(f"{label}\n    ✗ 自检 A 失败：桶 {check} 基线{where}")
            bad += 1
            continue

        touched = sorted({e[1] for e in edits})
        backup = {rel: (root / rel).read_bytes() for rel in touched}
        before = {rel: digest(root / rel) for rel in backup}
        try:
            try:
                for edit in edits:
                    apply_edit(edit, root)
            except (AssertionError, KeyError) as exc:
                lines.append(f"{label}\n    ✗ {exc}")
                bad += 1
                continue

            # --- self-check B: the mutation actually reached the disk
            if all(digest(root / rel) == before[rel] for rel in touched):
                lines.append(f"{label}\n    ✗ 自检 B 失败：改动没落盘")
                bad += 1
                continue

            if kind == "pytest":
                # --- self-check C: the injection must not have broken collection
                if not pytest_collects(check, root, python, basetemp):
                    lines.append(f"{label}\n    ✗ 自检 C 失败：注入后收集不到该用例"
                                 f"（多半是注入把模块改崩了，红得与机制无关）")
                    bad += 1
                    continue
                red = pytest_red(check, log(f"{i:02d}_run.txt"), root, python, basetemp)
                detail = ""
                if out_dir is not None:
                    detail = (out_dir / f"{i:02d}_run.txt").read_text(
                        encoding="utf-8").strip().split("\n")[-1][:110]
                name = check.split("::")[-1]
            else:
                after = sweep_buckets(tools, log(f"{i:02d}_run.txt"), root, python)
                red = after.get(check, 0) > base.get(check, 0)
                detail = json.dumps({k: v for k, v in after.items()
                                     if v and not k.endswith(("ok", "no-data"))},
                                    ensure_ascii=False)
                name = f"sweep 桶 {check}"
            lines.append(f"{label}\n    {'✓ 变红' if red else '✗ 仍然绿'}  {name}"
                         + (f"\n      {detail}" if detail else ""))
            bad += 0 if red else 1
        finally:
            for rel, raw in backup.items():
                (root / rel).write_bytes(raw)
                assert digest(root / rel) == before[rel], f"restore failed for {rel}"

    lines.append("")
    lines.append(f"共 {len(rows)} 行注入，未按预期变红 {bad} 行")
    return lines, bad


# --------------------------------------------------------------------------- batches & CLI

def available_batches() -> list[str]:
    d = ROOT / "scripts" / "mutations"
    return sorted(p.stem for p in d.glob("*.py") if not p.stem.startswith("_"))


def load_batch(name: str):
    mod = importlib.import_module(f"{BATCH_PACKAGE}.{name}")
    rows = getattr(mod, "ROWS")
    return rows, (mod.__doc__ or "").strip().split("\n")[0]


def git_dirty(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return "\n".join(ln for ln in (r.stdout or "").splitlines()
                     if not ln.startswith("?? "))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("batch", nargs="?", help="which table under scripts/mutations/ to run")
    ap.add_argument("--list", action="store_true", help="list the batches and exit")
    ap.add_argument("--only", default="",
                    help="comma-separated row prefixes, matched against the label's first "
                         "token (e.g. --only 13,15,16)")
    ap.add_argument("--out", default="", help="write the report here (default: stdout only)")
    ap.add_argument("--log-dir", default="",
                    help="keep the per-row pytest/sweep output in this directory")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the graded subprocesses")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even though tracked files are modified (a kill mid-row then "
                         "loses them)")
    args = ap.parse_args(argv)

    if args.list or not args.batch:
        for name in available_batches():
            rows, headline = load_batch(name)
            print(f"{name:24s} {len(rows):3d} 行  {headline}")
        return 0

    dirty = git_dirty(ROOT)
    if dirty and not args.allow_dirty:
        print("工作树不干净，拒绝运行——注入中途被杀会丢掉这些改动：\n" + dirty)
        print("确认要跑：--allow-dirty")
        return 2

    rows, _ = load_batch(args.batch)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r[0].split()[0] in want]
        if not rows:
            print(f"--only {args.only} 没有匹配到任何行")
            return 2

    out_dir = Path(args.log_dir) if args.log_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    lines, bad = run_batch(rows, ROOT, args.python, out_dir)
    report = "\n".join(lines)
    print(report)
    if args.out:
        io.open(args.out, "w", encoding="utf-8", newline="\n").write(report + "\n")

    after = git_dirty(ROOT)
    if after != dirty:
        print("\n⚠ 收尾时工作树与开跑前不一致，恢复可能没走完：\n" + after)
        return 3
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--sweep-probe":
        # Subprocess mode: one tool, {bucket: count} as JSON on stdout. Kept inside this file
        # so a batch has no second script to carry.
        sys.path.insert(0, str(ROOT))
        name = sys.argv[2]
        if name not in SWEEPS:
            raise SystemExit(f"unknown sweep {name!r}; known: {', '.join(SWEEPS)}")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.write(json.dumps(SWEEPS[name](ROOT), ensure_ascii=False))
        raise SystemExit(0)
    raise SystemExit(main())
