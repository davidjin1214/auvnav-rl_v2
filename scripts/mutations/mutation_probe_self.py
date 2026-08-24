"""注入器自身的负控（2026-08-24，12 行）。

The engine grades other people's tests; this is the batch that grades the engine. Written
the day the engine moved into `scripts/` for the sixth and last time, because a negative
control that cannot itself be shown to fail is a green light with nothing behind it.

Two halves, matching `tests/test_mutation_probe.py`:

  1..10  the engine. Each disables one clause of `apply_edit` / `digest` and names the case
         written for that clause. Rows 1 and 2 split `if n != 1` into its two sides -- `n < 1`
         still refuses an absent needle, `n > 1` still refuses an ambiguous one -- because a
         single `if False:` would collapse both and prove only that one of them fires.
         Row 9 is a *positive* control: dropping the markdown exemption makes
         `benchmarks/README.md` unreachable, and the two `cbm:doc-mismatch` rows in
         `benchmarks_manifests` become unrunnable while the suite stays green.

  11..12 the row-table checks. Their injections edit a *row table*, not code: that is the
         layer that notices a batch has rotted, and the only way to show it works is to rot
         one on purpose.

Note the self-reference is safe: the engine is already in memory when a row rewrites its own
source, so the restore in `finally` runs the original code, and only the pytest subprocess
sees the mutation.
"""

MP = "scripts/mutation_probe.py"
TM = "tests/test_mutation_probe.py"
CHAINS = "scripts/mutations/tracebacks_chains.py"

ROWS = [
    # ---- apply_edit: the match-count clause, one side at a time ----------------------
    ("1  命中多处时不再拒绝（取第一处）", "pytest",
     f"{TM}::test_an_injection_point_that_occurs_twice_is_refused",
     [("text", MP, "        if n != 1:", "        if n < 1:")]),
    ("2  命中 0 处时不再拒绝", "pytest",
     f"{TM}::test_an_injection_point_that_is_gone_is_refused",
     [("text", MP, "        if n != 1:", "        if n > 1:")]),
    # ---- apply_edit: line endings ---------------------------------------------------
    ("3  CRLF 文件被写回成 LF", "pytest", f"{TM}::test_a_crlf_file_stays_crlf",
     [("text", MP, "(out.replace(\"\\n\", \"\\r\\n\") if crlf else out)", "out")]),
    ("4  读入时不再归一行尾（多行 needle 匹配不到）", "pytest",
     f"{TM}::test_a_multiline_needle_matches_across_line_endings",
     [("text", MP, 'norm = raw.decode("utf-8").replace("\\r\\n", "\\n")',
       'norm = raw.decode("utf-8")')]),
    ("5  digest 不再归一行尾（还原自检失效）", "pytest",
     f"{TM}::test_the_digest_ignores_line_endings",
     [("text", MP, 'replace(b"\\r\\n", b"\\n")', "replace(b\"\", b\"\")")]),
    # ---- apply_edit: the claim form -------------------------------------------------
    ("6  claim 标签唯一性不再检查", "pytest",
     f"{TM}::test_a_claim_label_that_is_not_unique_is_refused",
     [("text", MP, "        if len(hits) != 1:", "        if False:")]),
    ("7  claim 的 None 改成写空值而不是删键", "pytest",
     f"{TM}::test_a_claim_edit_with_none_deletes_the_key",
     [("text", MP, "            del hits[0][key]", "            hits[0][key] = None")]),
    # ---- the evidence rule, both sides ----------------------------------------------
    # `return False and (` would also turn this red -- by making the module unimportable, so
    # pytest collects nothing and exits non-zero. Self-check C now catches that; the row is
    # written to stay syntactically valid so it grades the clause and not the parser.
    ("8  证据判据整个失效", "pytest",
     f"{TM}::test_the_evidence_files_are_never_an_injection_target",
     [("text", MP, "    return rel.startswith(PROTECTED)",
       "    return False and rel.startswith(PROTECTED)")]),
    ("9  markdown 豁免被摘掉（正控：文档不得被当成证据）", "pytest",
     f"{TM}::test_markdown_under_an_evidence_tree_is_still_a_document",
     [("text", MP, " and not rel.endswith(PROTECTED_EXEMPT_SUFFIX)", "")]),
    ("10 未知的 edit 类型被静默接受", "pytest",
     f"{TM}::test_an_unknown_edit_kind_is_refused",
     [("text", MP, '    raise AssertionError(f"未知的 edit 类型 {kind!r}")', "    return")]),
    # ---- the row-table checks: rot one table on purpose ------------------------------
    ("11 一条 sweep 行的工具前缀查无此人", "pytest",
     f"{TM}::test_every_sweep_row_names_a_registered_tool",
     [("text", CHAINS, '"sweep", "apn:erratum-stale"', '"sweep", "xyz:erratum-stale"')]),
    ("12 一条注入行指向已不存在的文件", "pytest",
     f"{TM}::test_every_file_a_row_names_still_exists",
     [("text", CHAINS, 'MD = "docs/data_integrity_open_items.md"',
       'MD = "docs/data_integrity_open_items_RENAMED.md"')]),
]
