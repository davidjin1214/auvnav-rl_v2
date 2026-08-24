"""C 类第 10 项：横幅提取与状态栏对帐（2026-08-23，21 行）。

Negative controls for `scripts/build_doc_index.py` (whose banner belongs to the document
itself, not one it cites) and `scripts/check_status_claims.py`. All pytest-layer; one row
is a *positive* control -- tightening the budget to 0 must not kill an ordinary banner.

Recovered from `mutate_status.py` (session scratchpad, 2026-08-23) when the engine moved
into `scripts/mutation_probe.py`. Its rows were the older 5-tuple shape and were rewritten
into `(label, kind, check, edits)` mechanically, source spans preserved. See
`scripts/mutation_probe.py` for the row format and the two self-checks.
"""

BDI = "scripts/build_doc_index.py"
CSC = "scripts/check_status_claims.py"
TB = "tests/test_build_doc_index.py"
TC = "tests/test_check_status_claims.py"

ROWS = [
    # ---- build_doc_index: the banner belongs to this doc, not one it cites ----------
    ("1  横幅归属判定整个拆掉", "pytest", f"{TB}::test_a_banner_quoting_another_documents_state_is_not_this_documents_status",
     [("text", BDI,
       "            if not _self_declared(head, kw_start):\n                continue\n",
       "")]),
    ("2  只拆「行内更早有链接」这一条", "pytest", f"{TB}::test_a_keyword_after_a_link_on_the_same_line_is_not_a_banner",
     [("text", BDI,
       '    if "](" in prefix or re.search(r"[A-Za-z0-9_\\-]\\.md\\b", prefix):',
       '    if False:')]),
    ("3  只拆「括号内旁白」这一条", "pytest", f"{TB}::test_a_keyword_inside_a_parenthetical_aside_is_not_a_banner",
     [("text", BDI,
       '    if prefix.count("（") > prefix.count("）") or prefix.count("(") > prefix.count(")"):',
       '    if False:')]),
    ("4  只把前缀预算放到无穷", "pytest", f"{TB}::test_a_keyword_buried_mid_sentence_is_not_a_banner",
     [("text", BDI,
       "SELF_DECL_BUDGET = 20",
       "SELF_DECL_BUDGET = 9999")]),
    ("5  预算收到 0（正控：正常横幅不得被误杀）", "pytest", f"{TB}::test_a_status_label_still_reaches_its_own_keyword",
     [("text", BDI,
       "SELF_DECL_BUDGET = 20",
       "SELF_DECL_BUDGET = 0")]),
    ("6  拆掉换行归一化", "pytest", f"{TB}::test_status_of_is_unchanged_by_whether_lines_carry_newlines",
     [("text", BDI,
       '    head = "".join(ln if ln.endswith("\\n") else ln + "\\n" for ln in quote)',
       '    head = "".join(quote)')]),
    # ---- check_status_claims: R1 -----------------------------------------------------
    ("7  状态栏改成任何表头都算", "pytest", f"{TC}::test_a_column_that_is_not_a_status_column_is_not_graded",
     [("text", CSC,
       'STATUS_HEADER = re.compile(r"状态|\\bstatus\\b|\\bstate\\b", re.I)',
       'STATUS_HEADER = re.compile(r"", re.I)')]),
    ("8  主语格不再要求以链接开头", "pytest", f"{TC}::test_a_row_whose_subject_merely_mentions_a_doc_is_not_graded",
     [("text", CSC,
       'SUBJECT_OPENER = re.compile(r"^[*`~\\s]*\\[")',
       'SUBJECT_OPENER = re.compile(r"^")')]),
    ("9  「自称仍有效」不再算相左", "pytest", f"{TC}::test_a_status_cell_contradicting_the_cited_banner_is_a_defect",
     [("text", CSC,
       'ALIVE = re.compile(r"\\bactive\\b|\\bcurrent\\b|\\blive\\b|仍有效|仍然有效|现行|在用|活跃", re.I)',
       'ALIVE = re.compile(r"(?!x)x")')]),
    ("10 日期比对拆掉", "pytest", f"{TC}::test_a_matching_keyword_with_a_different_date_is_a_defect",
     [("text", CSC,
       '                    bucket = "date" if (bdate and dates and bdate not in dates) else "agree"',
       '                    bucket = "agree"')]),
    ("11 单行引两份文档也照判", "pytest", f"{TC}::test_a_row_citing_two_docs_is_not_graded",
     [("text", CSC,
       "    if len(hrefs) != 1:",
       "    if len(hrefs) < 1:")]),
    # ---- check_status_claims: R2 -----------------------------------------------------
    ("12 预告规则不再限定进度列", "pytest", f"{TC}::test_a_forecast_outside_a_progress_column_is_not_graded",
     [("text", CSC,
       "                    if idx not in graded or not FORECAST_ONLY",
       "                    if not FORECAST_ONLY")]),
    ("13 只拆尾锚（去掉 $）", "pytest", f"{TC}::test_a_cell_beginning_with_a_batch_name_is_not_a_defect",
     [("text", CSC,
       '|第[一二三四五六七八九十\\d]+批|待第[一二三四五六七八九十\\d]+批|下一批|后续批次)$",',
       '|第[一二三四五六七八九十\\d]+批|待第[一二三四五六七八九十\\d]+批|下一批|后续批次)",')]),
    ("14 只拆首锚（match 改 search）", "pytest", f"{TC}::test_a_cell_ending_in_a_batch_name_is_not_a_defect",
     [("text", CSC,
       "or not FORECAST_ONLY.match(CELL_NOISE.sub(\"\", cell))",
       "or not FORECAST_ONLY.search(CELL_NOISE.sub(\"\", cell))")]),
    # ---- the freeze declaration ------------------------------------------------------
    ("15 自述冻结不再豁免", "pytest", f"{TC}::test_a_table_declared_frozen_is_excused_but_still_printed",
     [("text", CSC,
       '                    buckets["declared" if frozen else "forecast"].append(',
       '                    buckets["forecast"].append(')]),
    ("16 声明不再受小节边界约束", "pytest", f"{TC}::test_the_freeze_declaration_must_sit_under_the_same_heading",
     [("text", CSC,
       "        if HEADING.match(lines[i]):\n            return None\n",
       "")]),
    # ---- grading and corpus ----------------------------------------------------------
    ("17 把提示桶也算成缺陷", "pytest", f"{TC}::test_a_cell_that_omits_the_banner_keyword_is_advisory_not_a_defect",
     [("text", CSC,
       'DEFECT = ("contradict", "date", "forecast")',
       'DEFECT = ("contradict", "date", "forecast", "silent")')]),
    ("18 生成物不再豁免", "pytest", f"{TC}::test_the_generated_index_is_not_graded",
     [("text", CSC,
       'GENERATED = {"docs/DOC_INDEX.md"}',
       "GENERATED = set()")]),
    ("19 表格切分改成朴素 split", "pytest", f"{TC}::test_pipes_inside_inline_code_do_not_split_cells",
     [("text", CSC,
       "        if c == \"`\":\n            in_tick = not in_tick\n",
       "")]),
    ("21 拆掉 stdout 的 utf-8 重配（只有走管道才现形）", "pytest", f"{TC}::test_the_report_survives_a_non_utf8_stdout_pipe",
     [("text", CSC,
       '    if hasattr(sys.stdout, "reconfigure"):\n'
     '        sys.stdout.reconfigure(encoding="utf-8")\n',
       "")]),
    ("20 数据目录不再跳过", "pytest", f"{TC}::test_gitignored_data_trees_are_not_walked",
     [("text", CSC,
       "        if rel.split(\"/\", 1)[0] in DATA_DIRS or rel in GENERATED:",
       "        if rel in GENERATED:")]),
]
