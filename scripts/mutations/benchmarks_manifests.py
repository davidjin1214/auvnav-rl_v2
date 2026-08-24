"""C 类第 12 项：manifest 校验器与第十二条链（2026-08-24，34 行）。

Negative controls for `scripts/check_benchmark_manifests.py` and the
`benchmarks_and_datasets` chain. Graded across two tools, so the sweep rows carry both
prefixes: `cbm:` for the manifest checker, `apn:` for the published-number audit.

The sweep-layer rows edit a real markdown line and never a file under `benchmarks/` --
those manifests are the evidence itself, and the engine refuses to inject into them.

Recovered verbatim from `mutate_benchmarks.py` / `mutate_benchmarks_retry.py` (session
scratchpad, 2026-08-24) when the engine moved into `scripts/mutation_probe.py`. See that
module's docstring for the row format and the two self-checks.
"""

CBM = "scripts/check_benchmark_manifests.py"
APN = "scripts/audit_published_numbers.py"
T = "tests/test_check_benchmark_manifests.py"
TA = "tests/test_audit_published_numbers.py"
TP = "tests/test_check_doc_pointers.py"
HOOK = ".claude/hooks/doc_pointers.py"
SPEC = "docs/tracebacks/benchmarks_and_datasets.json"
GEN = "docs/tracebacks/_gen/gen_benchmarks_datasets_spec.py"
MD = "docs/data_integrity_open_items.md"
README = "benchmarks/README.md"

# (label, kind, named-check, [edits]) where an edit is
#   ("text",  relpath, old, new)                 literal replace, must occur exactly once
#   ("claim", relpath, claim-label, key, value)  set a spec claim's key; None deletes it
ROWS = [
    # ---- R1 结构自洽 -----------------------------------------------------------------
    ("01 种子连续性检查失效", "pytest", f"{T}::test_a_seed_that_skips_one_is_a_gap",
     [("text", CBM, "if seeds != list(range(seeds[0], seeds[0] + len(seeds))):",
       "if False:")]),
    ("02 episode_id 序号检查失效", "pytest",
     f"{T}::test_an_episode_id_whose_index_left_its_position",
     [("text", CBM, 'if not str(e.get("episode_id", "")).endswith(f"_{i:04d}")',
       'if not str(e.get("episode_id", "")).endswith("")')]),
    ("03 目录声明的条数不再核", "pytest",
     f"{T}::test_a_directory_that_declares_a_count_the_file_does_not_hold",
     [("text", CBM, 'DIR_COUNT = re.compile(r"^(?:val|test)_(\\d+)$")',
       'DIR_COUNT = re.compile(r"^(?!)(\\d+)$")')]),
    ("04 文件名声明的条数不再核", "pytest",
     f"{T}::test_a_filename_can_declare_the_count_too",
     [("text", CBM, 'NAME_COUNT = re.compile(r"_ep(\\d+)$")',
       'NAME_COUNT = re.compile(r"(?!)_ep(\\d+)$")')]),
    ("05 catalog 那一支起始种子期望被摘掉", "pytest",
     f"{T}::test_a_default_manifest_that_left_its_catalog_seed",
     [("text", CBM, "    spec = BENCHMARK_SPECS.get(stem)\n    if spec is not None:",
       "    spec = BENCHMARK_SPECS.get(stem)\n    if False:")]),
    ("06 _s{N} 标记退化成豁免（不再是期望）", "pytest",
     f"{T}::test_a_reseed_marker_moves_the_expectation_rather_than_lifting_it",
     [("text", CBM,
       '    m = NAME_SEED.search(stem)\n    if m:\n'
       '        return int(m.group(1)), f"文件名的 _s{m.group(1)} 重播种标记"',
       '    m = NAME_SEED.search(stem)\n    if m:\n        return None')]),
    # ---- R2 族内嵌套 -----------------------------------------------------------------
    ("07 嵌套不再比种子", "pytest",
     f"{T}::test_a_shorter_manifest_whose_seeds_diverge_partway",
     [("text", CBM, '        if int(a["seed"]) != int(b["seed"]):',
       "        if False:")]),
    ("08 浮点容差归零（末位差被判成缺陷）", "pytest",
     f"{T}::test_a_last_bit_difference_is_not_a_defect",
     [("text", CBM, "FLOAT_NOISE = 1e-9", "FLOAT_NOISE = 0.0")]),
    ("09 浮点容差放宽到 1e-3（真差异被吞掉）", "pytest",
     f"{T}::test_the_tolerance_does_not_swallow_a_real_difference",
     [("text", CBM, "FLOAT_NOISE = 1e-9", "FLOAT_NOISE = 1e-3")]),
    ("10 分组键丢掉流场（不同任务配置也拿来比）", "pytest",
     f"{T}::test_manifests_of_different_task_configurations_are_never_compared",
     [("text", CBM, '        str(payload.get("flow_path")),\n', '        "",\n')]),
    # ---- R3 文档所印 -----------------------------------------------------------------
    ("11 条数不再与文档比", "pytest", f"{T}::test_a_document_printing_the_wrong_count",
     [("text", CBM, '("条数", " 条", counts), ("种子区间", "", ranges)',
       '("种子区间", "", ranges),')]),
    ("12 种子区间不再与文档比", "pytest",
     f"{T}::test_a_document_printing_the_wrong_seed_range",
     [("text", CBM, '("条数", " 条", counts), ("种子区间", "", ranges)',
       '("条数", " 条", counts),')]),
    ("13 花括号只展开第一项", "pytest",
     f"{T}::test_a_brace_group_is_checked_against_every_file_it_expands_to",
     [("text", CBM, "    for option in m.group(1).split(\",\"):",
       "    for option in m.group(1).split(\",\")[:1]:")]),
    ("14 k 图对 k 份的按序配对被摘掉", "pytest",
     f"{T}::test_a_row_pairs_its_figures_with_its_expansions_in_order",
     [("text", CBM,
       "    if len(figs) == len(targets):\n"
       "        return [(value, t) for (value, _), t in zip(figs, targets)]\n", "")]),
    ("15 取数窗口不再要求在路径之后", "pytest",
     f"{T}::test_a_figure_ahead_of_the_path_is_not_a_claim_about_it",
     [("text", CBM, "        gap = m.start() - span[1]\n        if 0 <= gap",
       "        gap = abs(m.start() - span[1])\n        if 0 <= gap")]),
    ("16 取数窗口放到 500 字符", "pytest",
     f"{T}::test_a_figure_far_along_the_line_is_not_attributed",
     [("text", CBM, "FIGURE_WINDOW = 40", "FIGURE_WINDOW = 500")]),
    ("17 「前 N 条」不再排除", "pytest",
     f"{T}::test_a_prefix_count_says_which_episodes_not_how_many",
     [("text", CBM, 'COUNT_IS_A_SLICE = re.compile(r"[前首]\\s*$")',
       'COUNT_IS_A_SLICE = re.compile(r"(?!)")')]),
    ("18 dir/... 简写在多义时挑第一个而不是拒绝", "pytest",
     f"{T}::test_a_directory_shorthand_over_several_manifests_is_refused",
     [("text", CBM,
       '            if len(here) != 1:\n                return None  # ambiguous shorthand:'
       ' refuse rather than pick\n            hits.append(here[0])',
       '            if not here:\n                return None\n'
       '            hits.append(sorted(here)[0])')]),
    ("19 裸文件名在多义时挑第一个", "pytest",
     f"{T}::test_a_bare_basename_is_accepted_only_where_it_is_unique",
     [("text", CBM, "            if len(same) != 1:\n                return None\n"
                    "            hits.append(same[0])",
       "            if not same:\n                return None\n"
       "            hits.append(sorted(same)[0])")]),
    ("20 自述声明这一支被摘掉", "pytest",
     f"{T}::test_a_dated_declaration_on_the_line_excuses_the_figure",
     [("text", CBM, "                        elif DECLARED.search(line):",
       "                        elif False:")]),
    ("21 声明改成全文搜索（不再要求同一行）", "pytest",
     f"{T}::test_a_declaration_must_sit_on_the_line_it_excuses",
     [("text", CBM, "                        elif DECLARED.search(line):",
       "                        elif DECLARED.search(text):")]),
    # ---- R4 协议同规模 ---------------------------------------------------------------
    ("22 R4 整条摘掉", "pytest",
     f"{T}::test_a_default_manifest_shorter_than_its_protocol_siblings",
     [("text", CBM, "    if len(set(sizes.values())) > 1:", "    if False:")]),
    ("23 R4 改按顶层列举而非 catalog 键", "pytest",
     f"{T}::test_only_the_catalog_defaults_are_in_the_cohort",
     [("text", CBM,
       '    cohort = {key: known[f"{key}.json"] for key in BENCHMARK_SPECS\n'
       '              if f"{key}.json" in known}',
       '    cohort = {k[:-5]: v for k, v in known.items() if "/" not in k}')]),
    # ---- CLI 与管道 ------------------------------------------------------------------
    ("24 --strict 不再退 1", "pytest", f"{T}::test_strict_exits_nonzero_only_on_a_defect",
     [("text", CBM, "    if defects and args.strict:", "    if False:")]),
    ("25 少了 stdout 的 utf-8 重设（走管道即崩）", "pytest",
     f"{T}::test_the_report_survives_a_non_utf8_stdout_pipe",
     [("text", CBM, '    if hasattr(sys.stdout, "reconfigure"):\n'
                    '        sys.stdout.reconfigure(encoding="utf-8")\n', "")]),
    # ---- 钩子接线 --------------------------------------------------------------------
    ("26 新 sweep 没挂进钩子", "pytest",
     f"{TP}::test_the_hook_runs_every_sweep_not_only_the_path_one",
     [("text", HOOK, '    ("scripts.check_benchmark_manifests", "★",', '    ("x.y", "★",')]),
    # ---- 第 12 条链：工具、生成器、登记 -----------------------------------------------
    ("27 千分位逗号不再规范化", "pytest",
     f"{TA}::test_a_thousands_separator_is_still_a_number",
     [("text", APN, '.replace(" ", "").replace(",", "")', '.replace(" ", "")')]),
    ("28 手改一处 capture（生成器还写着原样）", "pytest",
     f"{TA}::test_every_committed_spec_can_be_regenerated",
     [("claim", SPEC, "§4 | 采集策略 success_rate", "capture", r"\*\*([0-9.]*)\*\*")]),
    ("29 SPEC_GENERATORS 里漏登记第 12 条链", "pytest",
     f"{TA}::test_every_spec_declares_whether_it_has_a_generator",
     [("text", TA, '    "benchmarks_and_datasets.json": "gen_benchmarks_datasets_spec.py",\n',
       "")]),
    ("30 假的「无生成器」声明（声明会消掉自己的警报）", "pytest",
     f"{TA}::test_a_spec_declared_ungenerated_is_not_emitted_by_any_generator",
     [("text", TA, '    "benchmarks_and_datasets.json": "gen_benchmarks_datasets_spec.py",\n',
       ""),
      ("text", TA, "UNGENERATED_SPECS = {\n",
       'UNGENERATED_SPECS = {\n    "benchmarks_and_datasets.json": "误登记为手工维护",\n')]),
    # ---- 数据侧：真文档改一处，看桶 ---------------------------------------------------
    ("31 改掉一个 offline_data 刊值", "sweep", "apn:value-mismatch",
     [("text", MD, "| **152,683**（= `mean_episode_length` 152.683 × 1000） |",
       "| **152,783**（= `mean_episode_length` 152.683 × 1000） |")]),
    ("32 改掉排除 (b) 的加噪读数", "sweep", "apn:value-mismatch",
     [("text", MD, "| `crosscomp_..._noise0p05clip0p15_ep2000` | **313,719** |",
       "| `crosscomp_..._noise0p05clip0p15_ep2000` | **313,619** |")]),
    ("33 README 的种子区间改错一位", "sweep", "cbm:doc-mismatch",
     [("text", README,
       "| `offline_rebrac_worldcomp_epoch_probe/{test_40,val_40}/single_u10_cross_tgt15.json`"
       " | 40 | 1250..1289 |",
       "| `offline_rebrac_worldcomp_epoch_probe/{test_40,val_40}/single_u10_cross_tgt15.json`"
       " | 40 | 1250..1290 |")]),
    ("34 README 的条数改错一格", "sweep", "cbm:doc-mismatch",
     [("text", README,
       "| `offline_rebrac_broad/{test_100,val_40}/single_u10_upstream_tgt15.json`"
       " | 100 / 40 | 1400..1499 / 1400..1439 |",
       "| `offline_rebrac_broad/{test_100,val_40}/single_u10_upstream_tgt15.json`"
       " | 100 / 50 | 1400..1499 / 1400..1439 |")]),
]
