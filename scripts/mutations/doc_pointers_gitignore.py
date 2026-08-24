"""`artifact` 桶改由 .gitignore 判定（2026-08-24，5 行）。

Negative controls for the change that made `check_doc_pointers`'s `artifact` bucket ask git
instead of matching a hand-written list of five directory names.

Every row here corresponds to something that actually went wrong while making that change,
in the order it was found: the list was too short to begin with, then the batch query came
back quoted (Windows text mode), then one out-of-repo path voided the whole batch, then a
directory rule missed a directory that does not exist yet. Three of the four were fail-open
-- the tool answered "nothing is ignored" and every artefact reference became a live defect.

The fifth row guards the other direction: with git unavailable the fallback list must still
apply, or a tree that is not a checkout reports every artefact path as a failure.

See `scripts/mutation_probe.py` for the row format and the three self-checks.
"""

CDP = "scripts/check_doc_pointers.py"
T = "tests/test_check_doc_pointers.py"

ROWS = [
    ("1  git 判定被摘掉，只剩写死的五个目录名", "pytest",
     f"{T}::test_a_gitignored_directory_outside_the_fallback_list_is_an_artifact",
     [("text", CDP, 'if ignored or ARTIFACT_DIR.search("/" + raw):',
       'if ARTIFACT_DIR.search("/" + raw):')]),
    ("2  目录规则的尾斜杠探针被摘掉", "pytest",
     f"{T}::test_a_missing_directory_is_matched_by_a_directory_rule",
     [("text", CDP, 'for suffix in ("", "/")', 'for suffix in ("",)')]),
    ("3  越界路径不再剔除（一条坏输入吞掉整批答案）", "pytest",
     f"{T}::test_a_path_outside_the_repo_does_not_sink_the_whole_batch",
     [("text", CDP,
       '    rels = [p for p in rels if not p.startswith("../") '
       'and not re.match(r"^[A-Za-z]:", p)]',
       "    rels = list(rels)")]),
    # The original defect here was quoting, not truncation: text mode put `\r` on every path
    # but the last, so git answered about one of them. Injecting the truncation reproduces
    # the property under test -- every path in the batch must come back answered -- without
    # depending on the host's newline translation to express it.
    ("4  整批只问第一条", "pytest",
     f"{T}::test_every_path_in_a_batch_is_answered_not_only_the_last",
     [("text", CDP, "input=\"\\0\".join(probes).encode(\"utf-8\")",
       "input=\"\\0\".join(probes[:1]).encode(\"utf-8\")")]),
    ("5  无 git 时的兜底名单被拆掉（正控）", "pytest",
     f"{T}::test_a_tree_without_git_falls_back_to_the_hardcoded_list",
     [("text", CDP, 'if ignored or ARTIFACT_DIR.search("/" + raw):', "if ignored:")]),
]
