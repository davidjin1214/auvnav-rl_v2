"""One-off: emit docs/tracebacks/rebrac.json.

Scope follows 48b8d06, which deliberately covered only the batch chapter 5 actually
cites -- not all 47 published readings. The spec covers the same batch and says so.

Usage:
    python docs/tracebacks/_gen/gen_rebrac_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Optional output directory, so the regression test in tests/test_audit_published_numbers.py
# can regenerate into a scratch tree instead of overwriting the committed specs.
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"

FORMAL = "formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{n}/actorb_4p0__criticb_2p0/test/seed_*.json"
SCREEN = "screening/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{n}/actorb_{a}__criticb_{c}/test/seed_*.json"
STAGE_E = "stage_e_critic_penalty_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/test/seed_*.json"
WORLD = "worldcomp_teacher_gap/{track}/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test/seed_*.json"
LN_OFF = "critic_ln_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test/seed_*.json"

claims = []


def add(label, anchor, capture, stat, sources, metric=None, **scope):
    re.compile(capture)
    re.compile(anchor)
    claim = {"label": label, "anchor": anchor, "capture": capture,
             "stat": stat, "sources": sources}
    claim.update(scope)
    if metric:
        claim["metric"] = metric
    claims.append(claim)


def skip(n: int) -> str:
    return r"[^|]+\|" * n


# --- 1. the dispersion note in section 1, which is itself a published claim ----------
# Each row prints the reading, its per-seed values, and both conventions. Checking it
# against `results/` is what turns 48b8d06's hand pass into something re-runnable --
# including its headline finding, the single ddof=1 reading.
NOTE_SECTION = r"^## 1\. 报告概述"
NOTE = [
    ("Stage C crosscomp-1000", r"（Stage C crosscomp-1000）", FORMAL.format(n=1000)),
    ("Stage C crosscomp-2000", r"（Stage C crosscomp-2000）", FORMAL.format(n=2000)),
    ("Stage E beta2=0", r"（Stage E β2=0）", STAGE_E),
    ("Stage D Phase 1 deployable", r"（Stage D Phase 1 deployable）", WORLD.format(track="deployable")),
    ("Stage D Phase 2 privileged", r"（Stage D Phase 2 privileged，rev\.8）",
     WORLD.format(track="privileged_critic")),
]
for label, tag, src in NOTE:
    add(f"dispersion note | {label} | published mean", tag,
        r"`\*{0,2}`?([0-9.]+) ±", "mean", [src], section=NOTE_SECTION)
    add(f"dispersion note | {label} | published ±", tag,
        r"± ([0-9.]+)`", "sd", [src], section=NOTE_SECTION)
    add(f"dispersion note | {label} | per-seed column", tag,
        tag + r"\*{0,2} \| ([0-9./]+) \|", "seeds", [src], section=NOTE_SECTION)
    add(f"dispersion note | {label} | ddof=0 column", tag,
        tag + r"\*{0,2} \| [0-9./]+ \| \*{0,2}([0-9.]+)", "sd0", [src], section=NOTE_SECTION)
    add(f"dispersion note | {label} | ddof=1 column", tag,
        tag + r"\*{0,2} \| [0-9./]+ \| \*{0,2}[0-9.]+\*{0,2} \| \*{0,2}([0-9.]+)",
        "sd1", [src], section=NOTE_SECTION)

# --- 2. Stage B screening grids (section 6.3) ---------------------------------------
# Both grids repeat the same six beta pairs, and the rows are identical line by line;
# only the bold dataset marker above them differs, and that marker repeats again under
# 6.4. Hence section + after/before rather than a cleverer row regex.
SCREEN_SECTION = r"^### 6\.3 主结果"
MARK_1000 = r"^\*\*`crosscomp-1000`\*\*"
MARK_2000 = r"^\*\*`crosscomp-2000`\*\*"
BETAS = [("4.0", "2.0", "4p0", "2p0"), ("2.0", "2.0", "2p0", "2p0"),
         ("4.0", "1.0", "4p0", "1p0"), ("2.0", "1.0", "2p0", "1p0"),
         ("1.0", "2.0", "1p0", "2p0"), ("1.0", "1.0", "1p0", "1p0")]
for size, scope in ((1000, dict(after=MARK_1000, before=MARK_2000)),
                    (2000, dict(after=MARK_2000))):
    for b1, b2, a, c in BETAS:
        src = [SCREEN.format(n=size, a=a, c=c)]
        row = rf"^\| \*{{0,2}}{re.escape(b1)}\*{{0,2}} \| \*{{0,2}}{re.escape(b2)}\*{{0,2}} \|"
        tag = f"stage B screen | crosscomp-{size} | b1={b1} b2={b2}"
        add(f"{tag} | success mean", row, row + r" \*{0,2}([0-9.]+) ±", "mean", src,
            "eval_success_rate", section=SCREEN_SECTION, **scope)
        add(f"{tag} | success ±", row, row + r" \*{0,2}[0-9.]+ ± ([0-9.]+)", "sd", src,
            "eval_success_rate", section=SCREEN_SECTION, **scope)
        add(f"{tag} | return mean", row, row + skip(1) + r" ([−-][0-9.]+) ±", "mean", src,
            "eval_return", section=SCREEN_SECTION, **scope)
        add(f"{tag} | return ±", row, row + skip(1) + r" [−-][0-9.]+ ± ([0-9.]+)", "sd", src,
            "eval_return", section=SCREEN_SECTION, **scope)

# --- 3. Stage C formal 5-seed overview (section 7.4) --------------------------------
for size in (1000, 2000):
    row = rf"^\| `crosscomp-{size}` \| \*\*4\.0\*\* \|"
    src = [FORMAL.format(n=size)]
    add(f"stage C formal | crosscomp-{size} | success mean", row,
        row + skip(1) + r" \*\*([0-9.]+) ±", "mean", src,
        section=r"^### 7\.4 主结果")
    add(f"stage C formal | crosscomp-{size} | success ±", row,
        row + skip(1) + r" \*\*[0-9.]+ ± ([0-9.]+)", "sd", src,
        section=r"^### 7\.4 主结果")

# --- 3b. the third cell of the same table, the (4.0, 1.0) alternative ---------------
# 48b8d06 covered the two bold rows only. The chapter cites this one as well, in the
# caption of tab:ch5_rebrac_perseed -- it says the alternative was trained on the same
# 2000-episode data under the same formal protocol and that its result entered no
# argument, which is exactly the kind of aside that goes unchecked. The row is the only
# unbolded one of the three, so the bold in the neighbours' anchors is what keeps all
# three apart; anchoring this one on the plain `| 4.0 | 1.0 |` cells does the same.
B2_1P0 = ("formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/"
          "actorb_4p0__criticb_1p0/test/seed_*.json")
ALT_ROW = r"^\| `crosscomp-2000` \| 4\.0 \| 1\.0 \|"
add("stage C formal | crosscomp-2000 (4.0, 1.0) | success mean", ALT_ROW,
    ALT_ROW + r" ([0-9.]+) ±", "mean", [B2_1P0], section=r"^### 7\.4 主结果")
add("stage C formal | crosscomp-2000 (4.0, 1.0) | success ±", ALT_ROW,
    ALT_ROW + r" [0-9.]+ ± ([0-9.]+)", "sd", [B2_1P0], section=r"^### 7\.4 主结果")


# --- 4. Stage D Phase 1, in its own 5-seed overview (section 7.10.3) ----------------
P1_ROW = r"^\| `worldcomp-1000` \| 4\.0 \| 2\.0 \|"
add("stage D Phase 1 | success mean", P1_ROW, P1_ROW + r" \*\*([0-9.]+) ±", "mean",
    [WORLD.format(track="deployable")], section=r"^#### 7\.10\.3 主结果")
add("stage D Phase 1 | success ±", P1_ROW, P1_ROW + r" \*\*[0-9.]+ ± ([0-9.]+)", "sd",
    [WORLD.format(track="deployable")], section=r"^#### 7\.10\.3 主结果")

# --- 5. the four-way Stage D comparison table (section 7.12.5) ----------------------
FOUR = r"^#### 7\.12\.5 与 Phase 1"
for label, row, track in (
        ("Phase 1 deployable", r"^\| ReBRAC worldcomp deployable Phase 1", "deployable"),
        ("Phase 2 privileged", r"^\| ReBRAC worldcomp privileged-critic Phase 2", "privileged_critic")):
    src = [WORLD.format(track=track)]
    add(f"stage D four-way | {label} | mean", row, row + r"[^|]*\| \*{0,2}([0-9.]+)",
        "mean", src, section=FOUR)
    add(f"stage D four-way | {label} | std", row,
        row + r"[^|]*\| \*{0,2}[0-9.]+\*{0,2} \| \*{0,2}([0-9.]+)", "sd", src, section=FOUR)

# --- 6. Stage F critic-LayerNorm-off (section 7.15.3) -------------------------------
LN_ROW = r"^\| mean_test_success_rate \|"
add("stage F | LN-off mean", LN_ROW, LN_ROW + r" \*\*([0-9.]+)\*\*", "mean",
    [LN_OFF], section=r"^#### 7\.15\.3 主结果")
add("stage F | the 5-seed baseline it is compared with", LN_ROW,
    LN_ROW + r" \*\*[0-9.]+\*\* \| ([0-9.]+)", "mean", [FORMAL.format(n=1000)],
    section=r"^#### 7\.15\.3 主结果")

spec = {
    "chain": "rebrac",
    "doc": "docs/rebrac_experiment_report.md",
    "note": ("Scope follows 48b8d06 (2026-05): the batch chapter 5 actually cites, not "
             "all 47 published readings -- Stage B screening grids, all three Stage C "
             "formal cells -- the two finalists plus the (4.0, 1.0) alternative that only "
             "the perseed table's caption cites, both Stage D worldcomp tracks, Stage E critic-penalty-off and "
             "Stage F LayerNorm-off. Sources are results/offline/rebrac/**/test/seed_*.json. "
             "The section-1 dispersion note is itself checked, so its headline finding "
             "-- one reading in the whole report written at ddof=1 -- is re-derived "
             "rather than trusted."),
    "root": "results/offline/rebrac",
    "metric": "eval_success_rate",
    "claims": claims,
}

out = OUT_DIR / "rebrac.json"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"wrote {out}: {len(claims)} claims")
