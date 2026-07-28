"""
paper/thesis_ch5/tools/_ch5_corpus.py

Shared corpus loader for the Chapter 5 register/layout checks.

This module is the executable form of spec §0.5.9 (d) 语体计量诊断读数 —— the nine
caliber rules written there are implemented here one-for-one, so the prose spec and
the tooling cannot drift apart. If you change a rule, change it in both places.

Caliber (spec §0.5.9 (d), 2026-07-28):
  1. Corpus  = sections/*.tex in main.tex \\input order (SECTION_ORDER below).
  2. Comments: content from an unescaped '%' to EOL is dropped; a line that was a
     full-line comment is removed entirely -- in LaTeX a full-line comment does NOT
     break a paragraph, so keeping a blank line here would fabricate paragraphs.
     '%' counts as a comment start only when preceded by an even number of
     backslashes (the chapter contains 24 literal `\\%`, all inside math).
  3. Floats: \\begin{table|figure}[*] ... \\end{...} removed from the body stream;
     their \\caption{...} argument is brace-matched out into a separate caption
     stream (27 of 36 captions contain nested braces -- regex alone gets this wrong).
  4. Dropped lines: \\section \\subsection \\subsubsection \\label \\input \\clearpage
     \\newpage \\centering \\includegraphics when they own the line.
  5. Body paragraph = maximal run of non-blank lines, >= MIN_NET_CHARS net chars.
     Display equations do NOT break paragraphs -- every equation in this chapter sits
     mid-sentence, and treating them as breaks inflates the paragraph count by 15.
  6. Macro strip before counting: equation/cases/align bodies, $...$, \\S\\ref{}
     \\ref{} \\eqref{} \\cite*{} \\label{}, then remaining \\macroname and braces.
  7. Net chars = CJK unified ideographs U+4E00-U+9FFF.
  8. Sentence terminators: 。！？ only (；，、 do not terminate). A trailing fragment
     with >= 1 net char counts as a sentence.
  9. Dash unit: one 破折号 = one occurrence of "——" (U+2014 twice). The chapter also
     contains 2 isolated single U+2014 used as 连接号 (e.g. 训练—部署信息边界);
     counting characters instead of pairs would wrongly report 130 instead of 128.

Two knobs above are load-bearing and two are not (measured 2026-07-28):
  - LOAD-BEARING: dropping structure lines (198 -> 252 paragraphs if kept) and
    treating equations as paragraph breaks (198 -> 213).
  - NOT load-bearing on this corpus: MIN_NET_CHARS (filters 0 blocks at any value
    1..20) and the trailing-fragment rule (all 198 paragraphs end in 。！？).

Usage:
    from _ch5_corpus import load, SECTION_ORDER, SECTION_NO, net_chars, sentences
    doc = load()                     # working tree
    doc = load(base=some_other_dir)  # an exported git baseline
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(CHAPTER_DIR, "..", ".."))

# main.tex \input order
SECTION_ORDER = [
    "intro", "related_work", "setup", "methodology", "online",
    "td3bc", "rebrac", "boundary", "algo_compare", "discussion",
]
SECTION_NO = dict(zip(SECTION_ORDER, [
    "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "5.9", "5.10",
]))

MIN_NET_CHARS = 5
SENTENCE_END = "。！？"
DASH = "\u2014\u2014"

CJK = re.compile(r"[\u4e00-\u9fff]")
FLOAT_BEGIN = re.compile(r"\\begin\{(table\*?|figure\*?)\}")
SUBSECTION = re.compile(r"\\subsection\{(.*?)\}")
DROP_LINE = re.compile(
    r"^\s*\\(section|subsection|subsubsection|label|input"
    r"|clearpage|newpage|centering|includegraphics)\b"
)


# --------------------------------------------------------------------------- #
# caliber rule 2 -- comment stripping
# --------------------------------------------------------------------------- #
def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        kept, i = [], 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                kept.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            kept.append(line[i])
            i += 1
        stripped = "".join(kept)
        if not stripped.strip() and line.strip().startswith("%"):
            continue  # full-line comment: drop the line, do not leave a blank
        out.append(stripped)
    return "\n".join(out)


def brace_match(text: str, start: int) -> tuple[str, int]:
    """`start` is the index of '{'. Returns (inner_content, index_after_close)."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)


# --------------------------------------------------------------------------- #
# caliber rule 3 -- float / caption separation
# --------------------------------------------------------------------------- #
def split_floats(text: str) -> tuple[str, list[str]]:
    captions: list[str] = []
    body: list[str] = []
    pos = 0
    while True:
        m = FLOAT_BEGIN.search(text, pos)
        if not m:
            body.append(text[pos:])
            break
        body.append(text[pos:m.start()])
        env = m.group(1)
        end = re.search(r"\\end\{" + re.escape(env) + r"\}", text[m.end():])
        stop = len(text) if not end else m.end() + end.end()
        chunk = text[m.start():stop]
        cpos = 0
        while True:
            cm = re.search(r"\\caption(\[[^\]]*\])?\{", chunk[cpos:])
            if not cm:
                break
            open_brace = cpos + cm.end() - 1
            content, after = brace_match(chunk, open_brace)
            captions.append(content)
            cpos = after
        pos = stop
    return "".join(body), captions


# --------------------------------------------------------------------------- #
# caliber rules 6-8 -- macro strip, net chars, sentence split
# --------------------------------------------------------------------------- #
def strip_macros(s: str) -> str:
    s = re.sub(r"\\begin\{(equation|cases|align)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\S?\s*\\ref\{[^}]*\}", " ", s)
    s = re.sub(r"\\(ref|eqref|label)\{[^}]*\}", " ", s)
    s = re.sub(r"\\cite[a-z]*\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return s.replace("{", " ").replace("}", " ")


def net_chars(s: str) -> int:
    return len(CJK.findall(strip_macros(s)))


def sentences(paragraph: str) -> list[str]:
    text = strip_macros(paragraph)
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in SENTENCE_END:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return [s for s in out if CJK.search(s)]


# --------------------------------------------------------------------------- #
# caliber rules 4-5 -- paragraph segmentation
# --------------------------------------------------------------------------- #
def load(base: str | None = None) -> dict:
    """Return {section_stem: {"paras": [{"sub", "text"}], "caps": [str]}}."""
    root = base or CHAPTER_DIR
    doc: dict = {}
    for stem in SECTION_ORDER:
        path = os.path.join(root, "sections", stem + ".tex")
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        body, caps = split_floats(strip_comments(raw))

        paras: list[dict] = []
        buf: list[str] = []
        current_sub = "(节导语)"

        def flush() -> None:
            joined = " ".join(x.strip() for x in buf).strip()
            if joined and net_chars(joined) >= MIN_NET_CHARS:
                paras.append({"sub": current_sub, "text": joined})
            buf.clear()

        for line in body.split("\n"):
            m = SUBSECTION.search(line)
            if m:
                flush()
                current_sub = m.group(1)
                continue
            if DROP_LINE.match(line):
                continue
            if not line.strip():
                flush()
                continue
            buf.append(line)
        flush()

        doc[stem] = {
            "paras": paras,
            "caps": [c for c in caps if net_chars(c) >= MIN_NET_CHARS],
        }
    return doc


def load_baseline(commit: str) -> tuple[dict, str]:
    """Export sections/*.tex at `commit` into a temp dir and load it.

    Returns (doc, tmpdir). The caller owns tmpdir; it is left on disk so the
    export can be inspected. Used for the cross-baseline trend table -- historical
    metric values from earlier review rounds must NOT be compared against current
    ones directly (spec §0.5.9 (d): they cannot even be assumed to share an
    implementation), so re-measuring each baseline with this code is the only
    defensible way to state a trend.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"ch5_{commit[:8]}_")
    archive = subprocess.run(
        ["git", "archive", commit, "paper/thesis_ch5/sections"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    tar = subprocess.Popen(["tar", "-x", "-C", tmpdir], stdin=subprocess.PIPE)
    tar.communicate(archive)
    if tar.returncode != 0:
        raise RuntimeError(f"tar extraction failed for {commit}")
    return load(os.path.join(tmpdir, "paper", "thesis_ch5")), tmpdir


def use_utf8_stdout() -> None:
    """Windows consoles default to a legacy codepage; CJK output needs this."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # pragma: no cover
        pass
