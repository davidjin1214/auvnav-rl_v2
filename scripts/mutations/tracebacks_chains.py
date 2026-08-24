"""第六/七条溯源链与生成器入库（2026-08-24，17 行）。

Negative controls for the 6th/7th traceback chains and the generator-regression tests.
Two layers guard these chains and only one of them is pytest: the `pytest` rows are what
still exists in a clone with no `results/`, the `apn:` sweep rows need the data and are what
the markdown hook runs here.

Recovered verbatim from `mutate_chains.py` (session scratchpad, 2026-08-24) when the engine
moved into `scripts/mutation_probe.py`; the sweep checks gained the `apn:` tool prefix the
engine now requires. See that module's docstring for the row format and the two self-checks.
"""
T = "tests/test_audit_published_numbers.py"
DI = "docs/tracebacks/data_integrity_open_items.json"
WC = "docs/tracebacks/td3bc_worldcomp_teacher_gap.json"
GEN = "docs/tracebacks/_gen/gen_data_integrity_spec.py"
MD = "docs/data_integrity_open_items.md"

REGEN = f"{T}::test_every_committed_spec_can_be_regenerated"
DECLARE = f"{T}::test_every_spec_declares_whether_it_has_a_generator"
FALSE_DECL = f"{T}::test_a_spec_declared_ungenerated_is_not_emitted_by_any_generator"
WEIGHT = f"{T}::test_the_report_table_chains_index_a_cell_that_carries_weight"
ONECELL = f"{T}::test_no_two_report_table_claims_read_the_same_cell_of_a_row"
ANCHORS = f"{T}::test_the_shipped_specs_still_anchor_to_their_reports"
SHARED = f"{T}::test_the_chapter_chains_read_the_files_the_reports_read"

FORMAL_1000 = ("rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"
               "/actorb_4p0__criticb_2p0/test/seed_*.json")
CLEAN_1000 = "rebrac/clean_probe/cross-1000/seed_*.json"
CLEAN_2000 = "rebrac/clean_probe/cross-2000/seed_*.json"

# (label, kind, named-check, [edits]) where an edit is
#   ("text",  relpath, old, new)                 literal replace, must occur exactly once
#   ("claim", relpath, claim-label, key, value)  set a spec claim's key; None deletes it
ROWS = [
    # ---- 生成器入库：手改 spec / 生成器不认 out_dir --------------------------------
    ("1  手改一处 capture（生成器还写着原样）", "pytest", REGEN,
     [("claim", DI, "δ表 | 采集器基线 回合数", "capture", "n=([0-9]*)")]),
    ("2  生成器忽略 out_dir，永远写进仓里", "pytest", REGEN,
     [("text", GEN,
       'OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"',
       'OUT_DIR = REPO / "docs" / "tracebacks"')]),
    # ---- 生成器入库：声明表 ----------------------------------------------------------
    ("3  SPEC_GENERATORS 里漏登记第七条链", "pytest", DECLARE,
     [("text", T, '    "data_integrity_open_items.json": "gen_data_integrity_spec.py",\n', "")]),
    ("4  SPEC_GENERATORS 指向不存在的生成器", "pytest", DECLARE,
     [("text", T, '"data_integrity_open_items.json": "gen_data_integrity_spec.py",',
       '"data_integrity_open_items.json": "gen_nope.py",')]),
    ("5  假的「无生成器」声明（声明会消掉自己的警报）", "pytest", FALSE_DECL,
     [("text", T, '    "data_integrity_open_items.json": "gen_data_integrity_spec.py",\n', ""),
      ("text", T, 'UNGENERATED_SPECS = {\n',
       'UNGENERATED_SPECS = {\n    "data_integrity_open_items.json": "误登记为手工维护",\n')]),
    # ---- 第 6/7 条链：格序号 ---------------------------------------------------------
    ("6  第 6 条链两条 claim 读同一格", "pytest", ONECELL,
     [("claim", WC, "progress ratio privileged-critic", "capture",
       "(?:[^|]*\\|){2}\\s*(-?[0-9.]+)\\s*[ |]")]),
    ("7  第 7 条链两条 claim 读同一格", "pytest", ONECELL,
     [("claim", DI, "δ表 | cross-1000 干净集 mean", "capture",
       "(?:[^|]*\\|){2}\\s*\\$([-+]?[0-9.]+)\\\\pm")]),
    ("8  文档把相邻两格印成同一个数（格序号从此不承重）", "pytest", WEIGHT,
     [("text", MD, "| ReBRAC-Q cross-2000 | $0.918\\pm0.030$ | $0.870\\pm0.024$ |",
       "| ReBRAC-Q cross-2000 | $0.918\\pm0.030$ | $0.918\\pm0.024$ |")]),
    ("9  把第七条链从受控名单里摘掉（掏空负控本身）", "pytest", WEIGHT,
     [("text", T, 'REPORT_TABLE_CHAINS = ("td3bc_worldcomp_teacher_gap", "data_integrity_open_items")',
       'REPORT_TABLE_CHAINS = ("td3bc_worldcomp_teacher_gap",)')]),
    ("10 锚点改到文档里找不到", "pytest", ANCHORS,
     [("claim", DI, "δ表 | 采集器基线 已刊", "anchor", "^\\| 采集器基座（解析式")]),
    # ---- 第 7 条链撑起的跨侧比对 -----------------------------------------------------
    ("11 第 7 条链的源改指另一格（章节侧 cross-1000 就此无人复核）", "pytest", SHARED,
     [("claim", DI, "δ表 | cross-1000 干净集 mean", "sources", [CLEAN_2000]),
      ("claim", DI, "δ表 | cross-1000 干净集 sd", "sources", [CLEAN_2000]),
      ("claim", DI, "δ表 | cross-1000 位移 pp", "sources", [FORMAL_1000, CLEAN_2000]),
      ("claim", DI, "δ表 | 翻转 干净集 pp", "sources", [CLEAN_2000, CLEAN_2000])]),
    ("12 复活一条已经失效的 UNSHARED 声明", "pytest", SHARED,
     [("text", T, "UNSHARED_WITH_THE_REPORTS: dict[str, str] = {}",
       'UNSHARED_WITH_THE_REPORTS: dict[str, str] = {\n    "results/offline/'
       'rebrac/clean_probe/cross-2000/seed_*.json": "已过期的声明",\n}')]),
    # ---- 数据侧：--strict 的各个缺陷桶 ----------------------------------------------
    ("13 改掉一个刊值", "sweep", "apn:value-mismatch",
     [("text", MD, "| ReBRAC-Q cross-1000 | $0.902\\pm0.021$",
       "| ReBRAC-Q cross-1000 | $0.912\\pm0.021$")]),
    ("14 勘误值改成能复算出来的（订正注就此失效）", "sweep", "apn:erratum-stale",
     [("text", MD, "一格原印 $\\pm0.025$", "一格原印 $\\pm0.024$")]),
    ("15 delta 的两个源对调", "sweep", "apn:value-mismatch",
     [("claim", DI, "δ表 | cross-1000 位移 pp", "sources", [CLEAN_1000, FORMAL_1000])]),
    ("16 去掉 pp 的单位换算", "sweep", "apn:value-mismatch",
     [("claim", DI, "δ表 | cross-2000 位移 pp", "scale", None)]),
    ("17 第 6 条链的格序号滑一格", "sweep", "apn:value-mismatch",
     [("claim", WC, "deployable_screen alpha=0.0 val success", "capture",
       "(?:[^|]*\\|){3}\\s*(-?[0-9.]+)\\s*[ |]")]),
]
