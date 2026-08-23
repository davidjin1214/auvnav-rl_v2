"""Tests for the Chapter 5 corpus loader -- the executable form of spec §0.5.9 (d).

`_ch5_corpus.py` is not a helper: its nine caliber rules ARE the measurement convention.
Three review rounds each rolled their own dash density and reported 0.635 / 0.618 / 0.593
with identical numerators and mutually incompatible denominators, which made the readings
unsubtractable after the fact (findings §12.1). Consolidating into one module fixed that
only for as long as the module keeps meaning what the spec says -- so each rule gets a
case here, and each case is built to change its answer if that rule is removed.

Two of the rules are documented as load-bearing on the real chapter (dropping structure
lines: 198 -> 252 paragraphs if kept; equations not breaking paragraphs: 198 -> 213). The
fixture reproduces both discriminations in miniature.

The corpus under test is a fixture, never the real chapter. The one test that does touch
`sections/` asserts structural invariants only, never a count: quantitative claims about
this chapter expire on any text edit (the 2026-07-08 layout claim survived three rounds
and was wrong in four places by 2026-07-28), and spec §0.5.8 forbids preset numeric
targets outright. A test asserting "198 paragraphs" would be the same defect in a new
place.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

# `paper/thesis_ch5/tools/` is not a package -- the scripts `from _ch5_corpus import`
# each other, which resolves only when their own directory is on sys.path. That is how
# they are actually run (`python tools/ch5_metrics.py`), so reproduce it rather than
# repackaging the chapter around the tests.
if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

corpus = importlib.import_module("_ch5_corpus")
metrics = importlib.import_module("ch5_metrics")

DASH_CHAR = "—"

# One section carrying the discriminating input for every caliber rule. Every construct
# below is here because removing the rule that handles it changes an assertion; nothing
# is decoration.
ALPHA = r"""\section{甲节}\label{sec:alpha}
第一段第一句。第二句；分号不断句，仍算一句。

第二段起头。
% 段中整行注释：若留成空行，这一段会被劈成两段
第二段续行，与上一行同段。

\begin{table}[t]
\centering
\caption{表题含嵌套花括号 \textbf{加粗} 与 $\mathrm{Re}=150$，共两层。}
\label{tab:alpha}
表体文字不进正文流，这一段不应出现在 paras 里。
\end{table}

第三段有一个破折号——只该算一处；另有单个连接号—不该算。
\begin{equation}
x = 1
\end{equation}
公式不断段，这一行仍属第三段。
\centering
\includegraphics{nowhere.pdf}
结构行被丢掉，这一行仍属第三段。
\subsection{甲小节}\label{subsec:alpha}
小节下第一段，sub 字段应变。

行内百分号 \% 后面的这些字必须保留。  % 但这个尾注释要去掉

\ref{tab:alpha}

末段没有句末标点，仍算一句
"""

BETA = r"""\section{乙节}\label{sec:beta}
乙节唯一一段，用来确认跨文件汇总。
"""

UNLISTED = "这个文件不在 SECTION_ORDER 里，不该被读进语料。\n"


@pytest.fixture
def chapter(tmp_path, monkeypatch):
    """A two-section chapter plus one .tex that the input order does not name."""
    root = tmp_path / "chapter"
    (root / "sections").mkdir(parents=True)
    (root / "sections" / "alpha.tex").write_text(ALPHA, encoding="utf-8")
    (root / "sections" / "beta.tex").write_text(BETA, encoding="utf-8")
    (root / "sections" / "unlisted.tex").write_text(UNLISTED, encoding="utf-8")
    monkeypatch.setattr(corpus, "SECTION_ORDER", ["alpha", "beta"])
    monkeypatch.setattr(metrics, "SECTION_ORDER", ["alpha", "beta"])
    return root


@pytest.fixture
def doc(chapter):
    return corpus.load(base=str(chapter))


def texts(doc: dict, stem: str = "alpha") -> list[str]:
    return [p["text"] for p in doc[stem]["paras"]]


# --------------------------------------------------------------------------- #
# rule 1 -- corpus membership
# --------------------------------------------------------------------------- #
def test_only_the_sections_named_by_the_input_order_are_loaded(doc):
    assert sorted(doc) == ["alpha", "beta"]
    assert not any("不该被读进语料" in t for t in texts(doc, "beta"))


def test_a_section_the_order_names_but_the_tree_lacks_is_an_error(chapter, monkeypatch):
    monkeypatch.setattr(corpus, "SECTION_ORDER", ["alpha", "missing"])
    # Silently skipping would shrink the corpus without saying so, and every density in
    # the chapter is a ratio over exactly this denominator.
    with pytest.raises(FileNotFoundError):
        corpus.load(base=str(chapter))


# --------------------------------------------------------------------------- #
# rule 2 -- comment stripping
# --------------------------------------------------------------------------- #
def test_a_trailing_comment_is_dropped(doc):
    (para,) = [t for t in texts(doc) if "行内百分号" in t]
    assert "但这个尾注释要去掉" not in para


def test_an_escaped_percent_does_not_open_a_comment(doc):
    (para,) = [t for t in texts(doc) if "行内百分号" in t]
    # `%` opens a comment only after an even number of backslashes. Lose that and the
    # line truncates at the backslash, taking 10 CJK characters of published text with it.
    assert "后面的这些字必须保留" in para
    assert corpus.net_chars(para) == 15


def test_a_full_line_comment_inside_a_paragraph_does_not_split_it(doc):
    joined = [t for t in texts(doc) if "第二段起头" in t]
    # In LaTeX a full-line comment does not break a paragraph. Leaving a blank line in
    # its place would fabricate one -- here, two paragraphs of 8 CJK chars each.
    assert len(joined) == 1
    assert "第二段续行" in joined[0]


# --------------------------------------------------------------------------- #
# rule 3 -- float / caption separation
# --------------------------------------------------------------------------- #
def test_a_float_body_leaves_the_text_stream(doc):
    assert not any("表体文字不进正文流" in t for t in texts(doc))


def test_a_caption_is_brace_matched_not_regex_matched(doc):
    (cap,) = doc["alpha"]["caps"]
    # 27 of the chapter's 36 captions nest braces. A non-greedy `\caption\{(.*?)\}`
    # stops at the first `}` and silently truncates -- here, at `\textbf{加粗}`.
    assert cap.endswith("共两层。")
    assert r"\textbf{加粗}" in cap


# --------------------------------------------------------------------------- #
# rules 4-5 -- paragraph segmentation
# --------------------------------------------------------------------------- #
def test_structure_lines_are_dropped_without_breaking_the_paragraph(doc):
    (para,) = [t for t in texts(doc) if "第三段有一个破折号" in t]
    # LOAD-BEARING on the real chapter: keeping these lines takes 198 paragraphs to 252.
    assert r"\centering" not in para
    assert r"\includegraphics" not in para
    assert "结构行被丢掉，这一行仍属第三段。" in para


def test_display_equations_do_not_break_a_paragraph(doc):
    (para,) = [t for t in texts(doc) if "第三段有一个破折号" in t]
    # LOAD-BEARING: every equation in this chapter sits mid-sentence, and treating them
    # as breaks inflates the paragraph count by 15.
    assert "公式不断段，这一行仍属第三段。" in para


def test_a_block_below_the_net_char_floor_is_not_a_paragraph(doc):
    assert not any(t.strip().startswith(r"\ref{") for t in texts(doc))
    assert all(corpus.net_chars(t) >= corpus.MIN_NET_CHARS for t in texts(doc))


def test_a_subsection_flushes_the_paragraph_and_relabels_what_follows(doc):
    subs = [p["sub"] for p in doc["alpha"]["paras"]]
    assert subs == ["(节导语)"] * 3 + ["甲小节"] * 3
    assert not any("甲小节" in t for t in texts(doc))


# --------------------------------------------------------------------------- #
# rules 6-7 -- macro strip and net chars
# --------------------------------------------------------------------------- #
def test_net_chars_counts_cjk_only_and_strips_macros_first():
    assert corpus.net_chars(r"\textbf{四个汉字} abc 123 $x=1$") == 4
    assert corpus.net_chars(r"\ref{tab:x}\cite{a}\label{b}") == 0
    # Stripping must happen BEFORE counting -- this chapter carries CJK inside math.
    assert corpus.net_chars(r"正文 $公式里不算$") == 2


# --------------------------------------------------------------------------- #
# rule 8 -- sentence splitting
# --------------------------------------------------------------------------- #
def test_only_the_three_terminators_end_a_sentence():
    # ；，、 are not terminators. Counting them would nearly double every reading.
    assert len(corpus.sentences("一句；分号，逗号、顿号都不断。")) == 1
    assert len(corpus.sentences("句号。感叹！问号？")) == 3


def test_a_trailing_fragment_with_net_chars_counts_as_a_sentence(doc):
    (para,) = [t for t in texts(doc) if "末段没有句末标点" in t]
    assert len(corpus.sentences(para)) == 1


def test_a_fragment_with_no_net_chars_does_not_count():
    assert corpus.sentences("一句话。 $x=1$") == ["一句话。"]


# --------------------------------------------------------------------------- #
# rule 9 -- dash unit
# --------------------------------------------------------------------------- #
def test_a_dash_is_a_pair_of_em_dashes_not_a_character(doc):
    (para,) = [t for t in texts(doc) if "第三段有一个破折号" in t]
    # The chapter also carries isolated single U+2014 as 连接号 (训练—部署信息边界);
    # counting characters reports 130 where the caliber says 128.
    assert para.count(corpus.DASH) == 1
    assert para.count(DASH_CHAR) == 3


# --------------------------------------------------------------------------- #
# ch5_metrics.summarise -- the aggregation the trend table is built from
# --------------------------------------------------------------------------- #
def test_summarise_aggregates_the_loaded_corpus(doc):
    s = metrics.summarise(doc)
    assert s == {
        "paras": 7,          # 6 in alpha + 1 in beta
        "dash": 1,
        "cap_units": 8,      # paragraphs + captions
        "cap_dash": 1,       # body dashes + caption dashes
        "over": {},          # nothing over the 4-sentence trigger
        "sentences": 11,
        "longest": 24,
    }


def test_summarise_counts_a_paragraph_over_the_sentence_trigger(chapter, monkeypatch):
    long_para = "".join(f"第{n}句话在这里。" for n in "一二三四五")
    (chapter / "sections" / "beta.tex").write_text(BETA + "\n" + long_para + "\n",
                                                   encoding="utf-8")
    s = metrics.summarise(corpus.load(base=str(chapter)))
    # The trigger is a diagnostic, not a cap (spec §0.5.7) -- but it has to fire.
    assert dict(s["over"]) == {5: 1}


# --------------------------------------------------------------------------- #
# the real chapter -- structural invariants only, never a count
# --------------------------------------------------------------------------- #
def test_the_real_chapter_still_loads_under_this_caliber():
    doc = corpus.load()
    assert sorted(doc) == sorted(corpus.SECTION_ORDER)
    for stem in corpus.SECTION_ORDER:
        assert doc[stem]["paras"], f"{stem} loaded with no body paragraphs"
        for para in doc[stem]["paras"]:
            assert corpus.net_chars(para["text"]) >= corpus.MIN_NET_CHARS
            assert corpus.sentences(para["text"])
