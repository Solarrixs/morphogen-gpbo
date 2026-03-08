# BUGFIX_SWARM_PROMPT.md

> **Usage:** Feed this entire file to Claude Code as a prompt. It will orchestrate an agentic swarm to implement all 38 bug fixes across 8 spec files, using git worktrees for isolation and respecting the dependency graph.

---

## Role

You are a **master orchestrator**. Your job is to read the bug fix specs, create a task plan, then spawn specialized agents in git worktrees to implement all fixes. You merge results sequentially in dependency order. You do NOT write code yourself — you delegate to agents and verify their work.

## Step 0: Bootstrap

1. Read `specs/SPEC-MASTER.md` and all 8 spec files:
   - `specs/spec-config.md`
   - `specs/spec-morphogen-parser.md`
   - `specs/spec-03-fidelity-scoring.md`
   - `specs/spec-02-map-to-hnoca.md`
   - `specs/spec-00-downloads.md`
   - `specs/spec-01-load-and-convert.md`
   - `specs/spec-04-gpbo-loop.md`
   - `specs/spec-tests-coverage.md`

2. Use `/planning-with-files` to initialize `task_plan.md`, `progress.md`, and `findings.md`.

3. Create a Task per spec with dependency relationships matching the graph in SPEC-MASTER.md:
   ```
   spec-config (FIRST)
     ├── spec-morphogen-parser → spec-04-gpbo-loop
     ├── spec-03-fidelity-scoring
     ├── spec-02-map-to-hnoca
     ├── spec-00-downloads
     └── spec-01-load-and-convert
                                    ↓
                          spec-tests-coverage (LAST)
   ```

4. Ensure you are on the `main` branch (or create a `bugfix/swarm` branch from `main`).

---

## Complexity: complex

---

## Phase 1: Foundation (1 agent, blocking)

**Spec:** `spec-config.md`

Spawn **1 agent** in `isolation: "worktree"` with this prompt:

> You are implementing bug fixes for `gopro/config.py` and all pipeline files. Read the spec below carefully.
>
> **Your task:**
> 1. Create `gopro/config.py` with:
>    - `PROJECT_DIR` resolved from `GPBO_PROJECT_DIR` env var, fallback to `Path(__file__).resolve().parent.parent`
>    - `DATA_DIR` resolved from `GPBO_DATA_DIR` env var, fallback to `PROJECT_DIR / "data"`
>    - `MODEL_DIR` resolved from `GPBO_MODEL_DIR` env var
>    - `MORPHOGEN_COLUMNS` — canonical 20-column ordering (indices 16-18: Dorsomorphin, purmorphamine, cyclopamine)
>    - `ANNOT_LEVEL_1`, `ANNOT_LEVEL_2`, `ANNOT_REGION`, `ANNOT_LEVEL_3` constants
>    - `get_logger(name)` factory with `GPBO_LOG_LEVEL` env var support
>    - `LOG_FORMAT`, `LOG_DATE_FORMAT` constants
>
> 2. Create `.env` at project root with `GPBO_PROJECT_DIR` (optional, for documentation)
>
> 3. Update ALL 6 pipeline files to import from config:
>    - `00_zenodo_download.py`: Remove `PROJECT_DIR = Path(...)`, import `PROJECT_DIR, DATA_DIR, get_logger`
>    - `01_load_and_convert_data.py`: Remove `PROJECT_DIR`, `DATA_DIR`, `DATA_DIR.mkdir()`, import `DATA_DIR, get_logger`
>    - `02_map_to_hnoca.py`: Remove `PROJECT_DIR`, `DATA_DIR`, `MODEL_DIR`, `ANNOT_*` constants, import all from config
>    - `03_fidelity_scoring.py`: Remove `PROJECT_DIR`, `DATA_DIR`, `ANNOT_*` constants, import all from config
>    - `04_gpbo_loop.py`: Remove `PROJECT_DIR`, `DATA_DIR`, `MORPHOGEN_COLUMNS`, import from config
>    - `morphogen_parser.py`: Remove local `MORPHOGEN_COLUMNS`, import from config
>
> 4. Replace ALL `print()` calls with `logger.info/warning/error()` across all 6 pipeline files (131 total). Conversion rules:
>    - `print(f"  Loading ...")` → `logger.info("Loading ...")`
>    - `print(f"ERROR: ...")` → `logger.error("...")`
>    - `print(f"WARNING: ...")` → `logger.warning("...")`
>    - f-string interpolation → lazy `%s` formatting: `logger.info("Loaded %d cells", n_cells)`
>    - Separator lines (`print("=" * 60)`) → remove or replace with `logger.info("--- Section Name ---")`
>    - Exception: Keep `print()` for progress bars using `\r`/`end=""` (in `00_zenodo_download.py`)
>
> 5. Replace 5 `exit(1)` calls with `raise SystemExit(1)`:
>    - `02_map_to_hnoca.py` lines 349, 353
>    - `03_fidelity_scoring.py` lines 652, 694, 706
>
> 6. Add 9 new tests to `gopro/tests/test_unit.py`:
>    - `test_config_project_dir_fallback`
>    - `test_config_project_dir_env_override` (monkeypatch)
>    - `test_config_data_dir_env_override` (monkeypatch)
>    - `test_morphogen_columns_length` (== 20)
>    - `test_morphogen_columns_canonical_order` (indices 16-18)
>    - `test_morphogen_columns_unique`
>    - `test_annot_level_values`
>    - `test_get_logger_returns_logger`
>    - `test_get_logger_respects_env_level` (monkeypatch)
>
> 7. Update any test imports that reference `MORPHOGEN_COLUMNS` from old locations.
>
> 8. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Constraints:**
> - `config.py` must NOT import from any `gopro/` module (no circular imports)
> - Zero hardcoded `/Users/maxxyung/Projects/morphogen-gpbo` strings remaining in any `.py` file under `gopro/`
> - All existing tests must pass
>
> **Full spec:** [Read `specs/spec-config.md`]

**After agent completes:** Merge the worktree branch into your working branch. Update `progress.md`.

**Commit message:** `fix: create centralized config.py with paths, constants, logging`

---

## Phase 2: Core Fixes (2 batches)

### Batch A — 3 parallel agents in worktrees

**Agent 1: Morphogen Parser** (`spec-morphogen-parser.md`)

> You are implementing bug fixes for `gopro/morphogen_parser.py`. The `config.py` module has already been merged — import `MORPHOGEN_COLUMNS` from `gopro.config`.
>
> **Your task:**
> 1. Remove the local `MORPHOGEN_COLUMNS` definition, import from `gopro.config`
> 2. Remove `_time_fraction()` entirely
> 3. Add `_set_morphogen(v, col, concentration, start_day, end_day)` helper
> 4. Add `_STANDARD_WINDOW = (6, 21)` constant
> 5. Update ALL 14 affected condition handlers to use `_set_morphogen` with true concentrations
> 6. Update `_zeros()` to accept `include_temporal` parameter
> 7. Add `TEMPORAL_MORPHOGEN_COLUMNS` to `gopro/config.py` (19 morphogens * 3 + 1 = 58 columns)
> 8. Update `parse_condition_name()` and `build_morphogen_matrix()` to accept `include_temporal` flag
> 9. Add tests:
>    - `test_temporal_conditions_are_distinct` (CHIR-d6-11 vs d11-16 vs d16-21)
>    - `test_backward_compatible_column_count` (20 columns)
>    - `test_temporal_column_count` (58 columns)
>    - `test_concentration_is_true_value` (1.5, not 0.5)
>    - `test_switch_condition_temporal`
>    - `test_full_window_defaults`
>    - `test_absent_morphogen_zero_temporal`
>    - `test_no_duplicate_vectors_after_fix`
> 10. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Constraints:** Don't break existing tests. `include_temporal=False` must be backward compatible with 20-column output.
>
> **Full spec:** [Read `specs/spec-morphogen-parser.md`]

**Agent 2: Fidelity Scoring** (`spec-03-fidelity-scoring.md`)

> You are implementing bug fixes for `gopro/03_fidelity_scoring.py`. The `config.py` module has already been merged — import paths, constants, and `get_logger` from `gopro.config`.
>
> **Your task:**
> 1. **P0 BUG FIX:** Wire `align_composition_to_braun()` into `score_all_conditions()`:
>    - Add `label_map` parameter to `score_all_conditions()`
>    - Call `align_composition_to_braun(level1_fracs, label_map)` before `compute_rss()`
> 2. Remove unused `from scipy.spatial.distance import cosine` import
> 3. Remove dead code: `compute_condition_composition()` (verify it's uncalled first via grep)
> 4. Scope warnings: Replace `warnings.filterwarnings("ignore")` with targeted filters for FutureWarning/DeprecationWarning from anndata/scanpy
> 5. Replace 3 `exit(1)` with `raise SystemExit(1)` (lines 652, 694, 706)
> 6. Replace ALL `print()` calls (~45) with `logger` calls
> 7. Decompose `main()` into: `load_references()`, `load_query_data()`, `score_and_annotate()`, `save_results()`, `print_summary()`
> 8. Add tests:
>    - `test_rss_with_aligned_labels`
>    - `test_rss_without_alignment_fails` (regression test proving the bug)
>    - `test_score_all_conditions_uses_alignment`
>    - `test_extract_braun_region_profiles_from_cache`
>    - `test_extract_braun_region_profiles_computes`
>    - `test_extract_braun_celltype_profiles`
> 9. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Full spec:** [Read `specs/spec-03-fidelity-scoring.md`]

**Agent 3: Atlas Mapping** (`spec-02-map-to-hnoca.md`)

> You are implementing bug fixes for `gopro/02_map_to_hnoca.py`. The `config.py` module has already been merged — import paths, constants, and `get_logger` from `gopro.config`.
>
> **Your task:**
> 1. Refactor `prepare_query_for_scpoli()` into 3 sub-functions:
>    - `load_and_filter_query(query, ref)` — gene symbol mapping
>    - `align_genes_to_model(query, ref)` — gene intersection, zero-fill missing, log overlap warning if <90%
>    - `setup_scpoli_query(query, batch_column)` — set batch, add snapseed labels, clear obsm/varm
> 2. Create `gopro/utils.py` with shared `compute_cell_type_fractions(obs, condition_key, label_key)`
> 3. Update both `02_map_to_hnoca.py` and `03_fidelity_scoring.py` to import from `gopro/utils.py`
> 4. Scope warnings: Replace `warnings.filterwarnings("ignore")` with targeted filters
> 5. Replace 2 `exit(1)` with `raise SystemExit(1)` (lines 349, 353)
> 6. Replace `print()` calls (~30) with `logger` calls
> 7. Add unit tests:
>    - `TestPrepareQueryForScpoli` (9 tests: gene symbol mapping, alignment, batch column, scpoli labels, obsm cleared, output shape, sparse input)
>    - `TestAlignGenesToModel` (4 tests: zero-fill columns, shared values preserved, overlap warning, output sparse)
>    - `TestTransferLabelsKnn` (6 tests: perfect separation, confidence range, multiple labels, missing column, output columns, k parameter)
> 8. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Full spec:** [Read `specs/spec-02-map-to-hnoca.md`]

**After all 3 agents complete:** Merge branches sequentially:
1. Morphogen parser first
2. Then fidelity scoring (03)
3. Then atlas mapping (02)

Run `python -m pytest gopro/tests/ -v` after each merge to catch conflicts early.

### Batch B — 2 parallel agents in worktrees

**Agent 4: Downloads** (`spec-00-downloads.md`)

> You are implementing bug fixes for `gopro/00_zenodo_download.py`. The `config.py` module has already been merged — import `PROJECT_DIR, DATA_DIR, get_logger` from `gopro.config`.
>
> **Your task:**
> 1. Remove unused `from urllib.parse import urljoin`
> 2. Remove hardcoded `PROJECT_DIR`, import from config
> 3. Replace `PROJECT_DIR / "data"` references with `DATA_DIR`
> 4. Fix download resume logic:
>    - Separate handling for HTTP 200 (write `"wb"`) vs 206 (append `"ab"`)
>    - Add `sha256_file()` helper alongside existing `md5_file()`
>    - Add `expected_sha256` parameter to `download_file()`
>    - Extract progress bar to `_print_progress()` helper
> 5. Replace 13 `print()` calls with `logger` calls (keep `print()` for progress bar `\r`/`end=""`)
> 6. Add 10 tests with mocked HTTP:
>    - `test_md5_file`, `test_sha256_file`
>    - `test_download_file_skips_when_checksum_matches`
>    - `test_download_file_redownloads_on_checksum_mismatch`
>    - `test_download_file_resume_http_206`
>    - `test_download_file_resume_fallback_on_http_200`
>    - `test_download_file_http_error`
>    - `test_get_record_url_construction`
>    - `test_known_records_structure`
>    - `test_search_zenodo_params`
> 7. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Full spec:** [Read `specs/spec-00-downloads.md`]

**Agent 5: Load & Convert** (`spec-01-load-and-convert.md`)

> You are implementing bug fixes for `gopro/01_load_and_convert_data.py`. The `config.py` module has already been merged — import `DATA_DIR, get_logger` from `gopro.config`.
>
> **Your task:**
> 1. Remove unused `import numpy as np`
> 2. Remove hardcoded `PROJECT_DIR`, `DATA_DIR`, `DATA_DIR.mkdir()`, import from config
> 3. Add type annotations to `convert_geo_to_anndata()` and `verify_references()`
> 4. Replace 20 `print()` calls with `logger` calls
> 5. Add tests:
>    - `test_convert_geo_dimension_mismatch_cells`
>    - `test_convert_geo_dimension_mismatch_genes`
>    - `test_convert_geo_output_compressed`
>    - `test_verify_references_all_missing` (monkeypatch DATA_DIR)
>    - `test_verify_references_all_present` (monkeypatch DATA_DIR)
> 6. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Full spec:** [Read `specs/spec-01-load-and-convert.md`]

**After both agents complete:** Merge both branches. Run tests.

**Commit message:** `fix: temporal encoding, fidelity alignment, atlas mapping refactor, download/load cleanup`

---

## Phase 3: GP-BO (1 agent, blocking)

**Spec:** `spec-04-gpbo-loop.md`

Spawn **1 agent** in `isolation: "worktree"`:

> You are implementing bug fixes for `gopro/04_gpbo_loop.py` — the core Bayesian optimization engine. The `config.py` and morphogen parser fixes have already been merged. Import `PROJECT_DIR, DATA_DIR, MORPHOGEN_COLUMNS` from `gopro.config`.
>
> **Your task (ordered by priority):**
>
> 1. **P0: Zero-variance filtering** (A-C-001, A-C-002, OPT-CRIT-01)
>    - Add `filter_zero_variance_columns(X, bounds)` → returns `(X_filtered, dropped_cols, dropped_values)`
>    - Call it in `fit_gp_botorch()` after constructing train_X, before fitting model
>    - In recommendations, restore dropped columns with their constant values
>    - Make `build_training_set()` only add fidelity column when `include_fidelity=True`
>
> 2. **Adaptive ILR pseudocount** (A-W-003)
>    - Change `ilr_transform()` to accept `cell_counts: Optional[np.ndarray]`
>    - Pseudocount = `1/(2*n_cells)` when counts available, else `1e-3` fallback
>    - Load cell counts from `fidelity_report.csv` if available
>
> 3. **Multi-objective ref_point** (A-W-004)
>    - Replace `ref_point = [0.0] * D` with data-derived: `y_min - 0.1 * y_range`
>    - Extract to `_compute_ref_point(train_Y)` helper
>
> 4. **Multi-fidelity GP wiring** (OPT-CRIT-02)
>    - Add `low_fidelity_data` parameter to `run_gpbo_loop()`
>    - Support concatenating training sets at different fidelity levels
>    - Fix fidelity dimension to 1.0 during acquisition optimization
>
> 5. **Scalarized weight warning** (A-W-006)
>    - Log warning about ILR-space weights not corresponding to equal cell-type importance
>
> 6. **Remove unused import** (OPT-MAJ-04)
>    - Delete `from botorch.utils.transforms import normalize, standardize`
>
> 7. **Refactor `recommend_next_experiments()`** (RF-HIGH-4)
>    - Break 117-line function into 4 helpers:
>      - `build_bounds_tensor(columns, bounds)`
>      - `build_acquisition_function(model, train_X, train_Y, use_multi_objective, ref_point)`
>      - `optimize_acquisition_function(acqf, bounds_tensor, n_recommendations)`
>      - `format_recommendations(candidates, acq_values, model, columns, dropped_cols, dropped_values, n_recommendations)`
>    - Top-level function becomes ~20-line orchestrator
>
> 8. **Add tests:**
>    - `TestFilterZeroVarianceColumns` (4 tests: drops constants, keeps all, drops fidelity, realistic morphogens)
>    - `TestAdaptiveILRPseudocount` (3 tests: with counts, without counts, scaling)
>    - `TestMultiObjectiveRefPoint` (1 test: auto ref below worst)
>    - `TestBuildBoundsTensor` (2 tests: basic, fidelity fixed)
>    - `TestMultiFidelityGP` (2 integration tests: activates, finite predictions)
>    - `TestMultiObjectiveAcquisition` (1 integration test: qNEHVI runs)
>    - `TestZeroVarianceEndToEnd` (1 integration test: GP fits with zero-var columns)
>    - ILR property tests: `test_pseudocount_always_positive`, `test_ilr_finite_with_zeros`
>    - Update existing `test_ilr_inverse_roundtrip` tolerance to `atol=5e-3`
>
> 9. Run `python -m pytest gopro/tests/ -v` and report results.
>
> **Constraints:** All GP operations on CPU (MPS doesn't support float64). Don't break existing 65 tests.
>
> **Full spec:** [Read `specs/spec-04-gpbo-loop.md`]

**After agent completes:** Merge branch. Run tests.

**Commit message:** `fix: GP zero-variance filtering, ILR pseudocount, multi-fidelity/objective`

---

## Phase 4: Test Coverage (1 agent, blocking)

**Spec:** `spec-tests-coverage.md`

Spawn **1 agent** in `isolation: "worktree"`:

> You are adding ~28 new tests to improve coverage from 64% to 85%+. All previous bug fixes have been merged.
>
> **Your task:**
>
> 1. **Fixture cleanup in `conftest.py`:**
>    - Remove unused `step01`, `step02`, `step04` fixtures (never used — tests do their own imports)
>    - Keep `_import_pipeline_module()` helper
>    - Add shared `data_dir` fixture pointing to project `data/` directory
>
> 2. **Integration tests** (`test_integration.py`) — all gated with `@pytest.mark.skipif(not Path(...).exists())`:
>    - `TestBraunReference::test_extract_braun_region_profiles`
>    - `TestBraunReference::test_extract_braun_celltype_profiles`
>    - `TestFidelityScoringIntegration::test_score_real_mapped_data`
>    - `TestGPBOIntegration::test_real_data_gpbo_loop`
>
> 3. **Unit tests** (`test_unit.py`):
>    - `TestMultiFidelityGP::test_fit_multi_fidelity_gp` (UT-01)
>    - `TestMultiFidelityGP::test_single_fidelity_uses_single_task_gp` (UT-02)
>    - `TestMultiObjectiveAcquisition::test_multi_objective_recommendations` (UT-03)
>    - `TestMultiObjectiveAcquisition::test_custom_ref_point` (UT-04)
>    - `TestConditionComposition::test_compute_condition_composition` (UT-05)
>    - `TestConditionComposition::test_compute_condition_region_fractions` (UT-06)
>    - `TestScPoliPreparation::test_prepare_query_basic` (UT-07)
>    - `TestScPoliPreparation::test_prepare_query_no_gene_name_column` (UT-08)
>    - `TestLabelTransfer::test_transfer_labels_knn` (UT-09)
>    - `TestLabelTransfer::test_transfer_labels_knn_missing_column` (UT-10)
>    - `TestZenodoDownload::test_get_record_mock` (UT-11)
>    - `TestZenodoDownload::test_md5_file` (UT-12)
>    - `TestZenodoDownload::test_download_file_skips_existing` (UT-13)
>    - `TestHelmertBasis::test_helmert_orthogonality` (UT-14)
>    - `TestBuildTrainingSet::test_empty_intersection` (UT-15)
>
> 4. **Error path tests** (`test_unit.py`):
>    - `test_compute_fractions_empty_dataframe` (EP-01)
>    - `test_unknown_condition_name` (EP-02)
>    - `test_ilr_all_zeros_row` (EP-03)
>    - `test_on_target_fraction_empty_condition` (EP-04)
>    - `test_cosine_similarity_nan_input` (EP-05)
>    - `test_ilr_inverse_wrong_d` (EP-06)
>    - `test_score_all_conditions_missing_columns` (EP-07)
>
> 5. **Property-based tests** (`test_properties.py`):
>    - `test_ilr_preserves_dominance` (PBT-01)
>    - `test_condition_composition_valid_distribution` (PBT-02)
>
> 6. Run `python -m pytest gopro/tests/ --cov=gopro --cov-report=term-missing -v` and report:
>    - Total test count (target: >= 90)
>    - Per-file coverage percentages
>    - Any test failures
>
> **Constraints:**
> - Add tests to EXISTING files only (no new test files)
> - Tests that need real data files must use `@pytest.mark.skipif`
> - Error path tests should document actual behavior in comments
> - Don't duplicate tests already added by previous agents
>
> **Full spec:** [Read `specs/spec-tests-coverage.md`]

**After agent completes:** Merge branch. Run full test suite.

**Commit message:** `test: add ~28 new tests, improve coverage 64% → 85%`

---

## Phase 5: QA (1 agent, final pass)

Spawn **1 agent** (no worktree needed — read-only verification):

> You are the QA agent. Run all verification checks on the merged codebase. Do NOT modify any files.
>
> **Run these commands and report results:**
>
> ```bash
> # 1. All tests pass
> python -m pytest gopro/tests/ -v
>
> # 2. Coverage target met
> python -m pytest gopro/tests/ --cov=gopro --cov-report=term-missing
>
> # 3. Zero print() calls in pipeline files (excluding tests and progress bars)
> grep -rn "print(" gopro/*.py | grep -v test | grep -v "_print_progress"
>
> # 4. Zero exit(1) calls
> grep -rn "exit(1)" gopro/*.py
>
> # 5. Single PROJECT_DIR definition (in config.py only)
> grep -rn "PROJECT_DIR = Path" gopro/
>
> # 6. Single MORPHOGEN_COLUMNS definition (in config.py only)
> grep -rn "MORPHOGEN_COLUMNS" gopro/*.py | grep -v "import" | grep -v "from" | grep -v "#" | grep -v "test"
>
> # 7. No hardcoded absolute paths
> grep -rn "/Users/maxxyung" gopro/*.py
>
> # 8. Recent git log
> git log --oneline -10
> ```
>
> **Report format:**
> - For each check: PASS or FAIL with details
> - List any remaining issues that need manual attention
> - If all checks pass, output: `QA COMPLETE — ALL CHECKS PASSED`

---

## Step 6: Completion

After the QA agent reports:

1. Update `task_plan.md` — mark all tasks complete
2. Update `progress.md` with final status and coverage numbers
3. Output a summary:

```
## BUGFIX COMPLETE

### Files Created
- gopro/config.py — centralized paths, constants, logging
- gopro/utils.py — shared compute_cell_type_fractions()
- .env — optional env var documentation

### Files Modified
- gopro/00_zenodo_download.py — config imports, resume fix, logging
- gopro/01_load_and_convert_data.py — config imports, type annotations, logging
- gopro/02_map_to_hnoca.py — refactored prepare_query, utils extraction, logging
- gopro/03_fidelity_scoring.py — P0 alignment fix, main() decomposition, logging
- gopro/04_gpbo_loop.py — zero-variance filter, ILR pseudocount, multi-fidelity/objective, refactor
- gopro/morphogen_parser.py — temporal encoding fix, config import
- gopro/tests/conftest.py — fixture cleanup
- gopro/tests/test_unit.py — ~40 new tests
- gopro/tests/test_integration.py — ~8 new tests
- gopro/tests/test_properties.py — ~4 new property tests

### P0 Critical Fixes Resolved
1. Zero-variance morphogen columns → filtered before GP fitting
2. align_composition_to_braun() → wired into scoring pipeline
3. Lossy temporal encoding → true concentrations + optional temporal columns
4. MORPHOGEN_COLUMNS ordering → single source of truth in config.py

### Commits
- fix: create centralized config.py with paths, constants, logging
- fix: temporal encoding, fidelity alignment, atlas mapping refactor
- fix: GP zero-variance filtering, ILR pseudocount, multi-fidelity/objective
- test: add ~28 new tests, coverage 64% → 85%
```

---

## Subtasks
- [ ] Phase 1: Foundation — create gopro/config.py, migrate all pipeline files to use centralized config, replace print() with logging, replace exit(1) with SystemExit | Acceptance: spec-config.md fully implemented, all existing tests pass, zero hardcoded paths
- [ ] Phase 2A-1: Morphogen Parser — temporal encoding fix, true concentrations, _set_morphogen helper | Acceptance: spec-morphogen-parser.md fully implemented, 8 new tests pass, backward compatible 20-column output
- [ ] Phase 2A-2: Fidelity Scoring — wire align_composition_to_braun into scoring, decompose main(), logging | Acceptance: spec-03-fidelity-scoring.md fully implemented, 6 new tests pass
- [ ] Phase 2A-3: Atlas Mapping — refactor prepare_query_for_scpoli, create utils.py, logging | Acceptance: spec-02-map-to-hnoca.md fully implemented, 19 new tests pass
- [ ] Phase 2B-1: Downloads — fix resume logic, sha256, logging | Acceptance: spec-00-downloads.md fully implemented, 10 new tests pass
- [ ] Phase 2B-2: Load & Convert — type annotations, logging, config imports | Acceptance: spec-01-load-and-convert.md fully implemented, 5 new tests pass
- [ ] Phase 3: GP-BO Loop — zero-variance filtering, adaptive ILR pseudocount, multi-objective ref_point, refactor recommend_next_experiments | Acceptance: spec-04-gpbo-loop.md fully implemented, 14 new tests pass
- [ ] Phase 4: Test Coverage — add ~28 new tests across unit/integration/property files | Acceptance: spec-tests-coverage.md fully implemented, coverage >= 85%
- [ ] Phase 5: QA Verification — run all checks, confirm zero print/exit/hardcoded paths | Acceptance: all 8 QA checks pass

---

## Verification (run manually after swarm completes)

```bash
python -m pytest gopro/tests/ -v                    # all pass
python -m pytest gopro/tests/ --cov=gopro            # >80%
grep -r "PROJECT_DIR = Path" gopro/                  # only config.py
grep -r "exit(1)" gopro/                             # zero results
grep -rn "print(" gopro/*.py | grep -v test          # zero results (except progress bar)
git log --oneline -10                                # structured commits
```
