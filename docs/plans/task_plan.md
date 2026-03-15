# Implementation Plan -- GP-BO Pipeline Production Readiness
> Created: 2026-03-15
> Goal: Make the pipeline production-ready with trusted recommendations, full virtual data stack, and any-region targeting
> Hardware: macOS 48GB RAM, no GPU (GPU tasks deferred to TODO.md)
> Approach: Quality over speed, phased rollout

## Phase Overview

```
Phase 1: Foundations (validation, refactoring, API surface)     ~3-4 days
Phase 2: Region Targeting System                                ~2-3 days
Phase 3: Virtual Data Stack (CellRank2 + CellFlow)             ~3-4 days
Phase 4: Pipeline Integration + Config-Driven Datasets          ~2-3 days
Phase 5: Test Coverage + Polish                                 ~2-3 days
                                                          Total: ~12-17 days
```

## Dependency Graph

```
Phase 1A (validation) ─────────┐
Phase 1B (step 05 decompose) ──┤
Phase 1C (importable API) ──────┼──> Phase 2 (region targeting) ──> Phase 4 (integration)
                                │                                        │
                                └──> Phase 3A (temporal atlas) ──────────┤
                                     Phase 3B (CellRank2) ──────────────┤
                                     Phase 3C (CellFlow heuristic) ─────┘
                                                                        │
                                                                   Phase 5 (tests + polish)
```

Phases 1A/1B/1C can run in parallel.
Phases 3A/3B/3C are sequential (each depends on the previous).
Phase 2 can run in parallel with Phase 3.

---

## Phase 1: Foundations

### 1A. Inter-Step Data Validation (~1 day)

**Goal**: Add pre-flight validation so steps fail fast with clear messages instead of crashing mid-computation.

**Files to create/modify**:
- NEW: `gopro/validation.py` -- schema definitions and validators

**Tasks**:

1. **Define output schemas for each step** as lightweight dataclasses or typed dicts:
   - Step 02 output: `*_mapped.h5ad` must have `predicted_annot_level_1`, `predicted_annot_level_2`, `predicted_annot_region_rev2`, `predicted_annot_level_3_rev2`, plus `condition` column in obs
   - Step 02 CSV output: `gp_training_labels_*.csv` must have condition index, all values in [0,1], rows sum to ~1.0
   - Step 03 output: `fidelity_report.csv` must have `composite_fidelity`, `rss_score`, `dominant_region` columns
   - Step 04 output: `gp_recommendations_round*.csv` must have morphogen columns + prediction columns
   - Morphogen matrix: columns must be subset of `MORPHOGEN_COLUMNS`, index must match fractions CSV

2. **Add `validate_mapped_adata(path)` function**: checks obs columns, returns clear error listing what's missing

3. **Add `validate_training_csvs(fractions_path, morphogen_path)` function**: checks alignment, NaN, row sums, column names

4. **Wire validators into step entry points**:
   - Step 03 `main()`: validate mapped h5ad before loading Braun (saves 2+ min on failure)
   - Step 04 `run_gpbo_loop()`: validate CSVs before GP fitting
   - Step 05 `main()`: validate temporal atlas has `day` column and expected structure

**Tests**: 8-10 new tests in `test_unit.py` for validation functions (valid input, missing columns, NaN, wrong types)

**Acceptance**: Pipeline steps fail within 5 seconds when given bad input, with a message naming the missing/invalid field.

---

### 1B. Decompose `project_query_forward` (~1 day)

**Goal**: Break the 327-line monolith into testable, single-responsibility functions.

**File**: `gopro/05_cellrank2_virtual.py`

**Decomposition plan**:

```python
# Extract these functions from project_query_forward:

def embed_query_in_atlas_pca(
    query_adata, atlas_adata, source_mask
) -> tuple[np.ndarray, np.ndarray]:
    """Project query cells into atlas PCA space.
    Returns (query_embedding, source_pca) with matched dimensions.
    Handles 3 cases: existing X_pca, PCA loadings projection, joint PCA fallback.
    """

def find_nearest_atlas_neighbors(
    query_embedding, source_pca, k=10
) -> tuple[np.ndarray, np.ndarray]:
    """KNN in PCA space. Returns (distances, neighbor_indices)."""

def compute_knn_weights(
    distances: np.ndarray
) -> np.ndarray:
    """Convert distances to inverse-distance weights, row-normalized."""

def project_via_push_api(
    problem, source_dist, source_tp, target_tp,
    target_labels, target_ct_fracs, n_target_atlas
) -> pd.Series:
    """Forward-project using moscot .push() API.
    Returns cell type fractions or raises ProjectionError.
    """

def project_via_transport_matrix(
    transport, neighbor_idx, weights, cond_indices,
    local_idx_map, target_labels_arr, target_ct_fracs
) -> pd.Series:
    """Forward-project using manual sparse transport composition.
    Returns cell type fractions or raises ProjectionError.
    """

def harmonize_atlas_labels(
    target_labels: pd.Series, source_col: str, target_col: str
) -> pd.Series:
    """Map atlas cell type labels to query vocabulary.
    LABEL_HARMONIZATION dict moved to module-level constant.
    """
```

Move `LABEL_HARMONIZATION` dict to module level (currently buried at line 385).

The refactored `project_query_forward` becomes a ~60 line orchestrator that calls these functions in sequence with try/except fallback logic.

**Tests**: 6-8 new tests in `test_phase4_5.py`:
- `test_embed_query_pca_matching_dims`
- `test_embed_query_pca_joint_fallback`
- `test_knn_weights_sum_to_one`
- `test_harmonize_labels_known_mapping`
- `test_harmonize_labels_unmapped_passthrough`
- `test_project_via_transport_matrix_sparse`

**Acceptance**: `project_query_forward` is under 80 lines. Each extracted function has at least one unit test. All 194+ tests still pass.

---

### 1C. Importable API Surface (~1 day)

**Goal**: Make core pipeline functions importable from notebooks/scripts without running CLI.

**Files to modify**:
- `gopro/__init__.py` -- add public API exports
- Steps 02, 03, 04, 05, 06 -- ensure `main()` functions are decomposed into reusable pieces

**Tasks**:

1. **Update `gopro/__init__.py`** with curated exports:
   ```python
   # Core pipeline functions
   from gopro.config import MORPHOGEN_COLUMNS, DATA_DIR, get_logger
   from gopro.morphogen_parser import build_morphogen_matrix, parse_condition_name
   from gopro.validation import validate_training_csvs, validate_mapped_adata

   # GP-BO
   from gopro._04_gpbo_loop import (
       build_training_set, fit_gp_botorch, recommend_next_experiments,
       ilr_transform, ilr_inverse, run_gpbo_loop,
   )

   # Scoring
   from gopro._03_fidelity_scoring import (
       score_all_conditions, compute_composite_fidelity,
       extract_braun_region_profiles,
   )

   # Mapping
   from gopro._02_map_to_hnoca import (
       filter_quality_cells, compute_cell_type_fractions,
       compute_soft_cell_type_fractions,
   )
   ```

2. **Decompose `main()` in step 03** into `run_fidelity_scoring(mapped_path, braun_path, condition_key) -> (report_df, annotated_adata)` that returns data instead of only writing files.

3. **Decompose `main()` in step 02** into `run_mapping(query_path, ref_path, model_dir, ...) -> (mapped_adata, fractions_df, soft_probs)`.

4. **Add module aliases** for numeric-prefixed files: since `from gopro.04_gpbo_loop import ...` is invalid Python, the `__init__.py` uses the `_import_pipeline_module` pattern from conftest.py or adds symlinks like `gopro/gpbo.py -> 04_gpbo_loop.py`.

**Tests**: Verify imports work: `from gopro import build_training_set, score_all_conditions`

**Acceptance**: A Jupyter notebook cell can `from gopro import run_gpbo_loop, score_all_conditions` without errors.

---

## Phase 2: Region Targeting System

### 2A. Region Discovery + Named Profiles (~1.5 days)

**Goal**: Let users target ANY brain region by showing what's available and providing named presets.

**Files to create/modify**:
- NEW: `gopro/region_targets.py` -- region profile management
- MODIFY: `gopro/03_fidelity_scoring.py` -- parameterize region dicts
- MODIFY: `gopro/04_gpbo_loop.py` -- accept target region as input

**Tasks**:

1. **Create `gopro/region_targets.py`** with:
   ```python
   def discover_available_regions(ref_path: Path) -> dict[str, dict]:
       """Scan a reference atlas and return available regions with metadata.
       Returns: {region_name: {n_cells, top_cell_types, annotation_level}}
       """

   def load_region_profile(region_name: str, ref_path: Path) -> pd.Series:
       """Load the cell type composition profile for a named region.
       Returns: Series of cell type fractions (sums to 1).
       """

   # Named presets (built-in profiles for common targets)
   NAMED_REGION_PROFILES: dict[str, dict] = {
       "dorsal_telencephalon": {
           "description": "Cortical excitatory neurons, radial glia",
           "source": "Braun fetal brain",
           "annotation_level": "annot_region_rev2",
       },
       "ventral_midbrain": {
           "description": "Dopaminergic neurons (Parkinson's disease relevant)",
           "source": "Braun fetal brain",
           "annotation_level": "annot_region_rev2",
       },
       # ... all 9 HNOCA regions plus additional from Braun
   }

   def list_named_profiles() -> pd.DataFrame:
       """List all built-in region profiles with descriptions."""

   def build_custom_target(cell_type_fractions: dict[str, float]) -> pd.Series:
       """Create a custom target from user-specified cell type fractions."""
   ```

2. **Refactor `03_fidelity_scoring.py`**:
   - Move `HNOCA_TO_BRAUN_REGION`, `OFF_TARGET_LEVEL1`, `build_hnoca_to_braun_label_map()` into `region_targets.py` as default configs
   - `score_all_conditions()` accepts an optional `target_profile: pd.Series` parameter
   - When `target_profile` is provided, RSS scoring compares against that single profile instead of finding the best-matching region
   - The "composite fidelity" score becomes "how close is this condition to the target"

3. **Add `--target-region` CLI arg to step 04**:
   ```
   python 04_gpbo_loop.py --target-region ventral_midbrain
   python 04_gpbo_loop.py --target-region custom --target-profile path/to/profile.csv
   python 04_gpbo_loop.py --list-regions  # print available regions and exit
   ```

4. **Wire target region into GP objective**:
   - When a target region is specified, the GP objective becomes: maximize cosine similarity between predicted composition and target profile
   - This replaces the current "maximize all cell type fractions equally" default

**Tests**: 10-12 new tests:
- `test_discover_regions_from_hnoca_ref`
- `test_load_region_profile_known_region`
- `test_load_region_profile_unknown_region_raises`
- `test_named_profiles_all_valid`
- `test_custom_target_sums_to_one`
- `test_score_with_target_profile`
- `test_gp_objective_with_target_region`

**Acceptance**: `python 04_gpbo_loop.py --list-regions` prints available regions. `--target-region ventral_midbrain` produces recommendations optimized for that region.

---

### 2B. Dynamic Label Map Construction (~1 day)

**Goal**: Auto-build label mappings between any two atlases instead of hardcoding them.

**File**: `gopro/region_targets.py`

**Tasks**:

1. **Add `build_label_map(source_labels, target_labels, method="fuzzy")` function**:
   - Exact string matching first
   - Fuzzy matching (Levenshtein distance) for close matches
   - Synonym table for known aliases (e.g., "NPC" = "Radial glia" = "Neural progenitor")
   - Returns mapping dict + confidence scores + list of unmapped labels

2. **Add a curated synonym table** (`CELL_TYPE_SYNONYMS`) covering HNOCA, Braun, and common scRNA-seq vocabularies

3. **Wire into `score_all_conditions()`**: if no explicit label_map is provided, auto-build one with a warning for low-confidence mappings

**Tests**: 5-6 tests for fuzzy matching, synonym resolution, unmapped handling

**Acceptance**: Adding a new reference atlas requires zero code changes to label mapping (auto-discovery with manual override option).

---

## Phase 3: Virtual Data Stack

### 3A. Build Temporal Atlas (~0.5 day, run on Mac)

**Goal**: Actually run step 00c to produce `azbukina_temporal_atlas.h5ad`.

**Prerequisite**: Patterning screen data already downloaded (22GB on disk).

**Tasks**:

1. **Inspect the source h5ad** to understand its structure:
   ```bash
   python 00c_build_temporal_atlas.py --inspect-only
   ```
   Document what columns exist, what timepoints are present, how many cells.

2. **Run the build**:
   ```bash
   python 00c_build_temporal_atlas.py
   ```
   This decompresses the .h5ad.gz and reformats into CellRank2-compatible format.

3. **Validate output**: check `day` column has expected timepoints, cell counts per timepoint, whether `X_pca` exists or needs computing.

4. **Document findings** in `data/README.md` or `data/temporal_atlas_notes.md`.

**Memory estimate**: The source file is ~2-3GB uncompressed. Should fit in 48GB RAM.

**Acceptance**: `data/azbukina_temporal_atlas.h5ad` exists with correct `day` column and at least 4 timepoints.

---

### 3B. CellRank2 Virtual Data Generation (~2 days)

**Goal**: Run step 05 end-to-end to produce virtual training data.

**Prerequisites**: Phase 1B (decomposed step 05), Phase 3A (temporal atlas).

**Tasks**:

1. **Dry-run the decomposed step 05** with the temporal atlas:
   - Verify moscot transport map computation works on Mac (memory check)
   - Check if atlas timepoints match `ATLAS_TIMEPOINTS` constant
   - Monitor RAM usage during transport map computation

2. **Run full CellRank2 pipeline**:
   ```bash
   python 05_cellrank2_virtual.py
   ```
   This produces `cellrank2_virtual_fractions.csv` and `cellrank2_virtual_morphogens.csv`.

3. **Validate virtual data quality**:
   - Check that virtual fractions are not all atlas-average (which would mean fallback was used for everything)
   - Compare virtual vs real fractions for overlapping conditions
   - Check transport quality report for high-cost or non-converged transitions

4. **Wire transport quality into filtering** (addresses TODO item):
   - In `merge_multi_fidelity_data()`, optionally load `cellrank2_transport_quality.csv`
   - Down-weight or exclude virtual points from transitions flagged `HIGH_COST` or `NOT_CONVERGED`

**Tests**: 4-5 new tests for transport quality filtering logic

**Risk**: moscot OT may be memory-intensive. If temporal atlas is >500K cells, may need to subsample. Fallback: subsample to 100K cells per timepoint.

**Acceptance**: Virtual fractions CSV exists with at least 50 virtual data points. At least 30% of virtual points use actual transport (not atlas average fallback).

---

### 3C. CellFlow Heuristic Improvement (~1 day)

**Goal**: Improve the baseline heuristic predictor. Defer actual CellFlow training to GPU TODO.

**File**: `gopro/06_cellflow_virtual.py`

**Tasks**:

1. **Check for pre-trained CellFlow model** on disk (address user requirement):
   ```python
   # At startup, check these paths:
   DATA_DIR / "cellflow_model"
   DATA_DIR / "cellflow_model.pt"
   DATA_DIR / "patterning_screen" / "cellflow_model"
   ```
   If found, use it. If not, log a clear message and fall back to heuristic.

2. **Improve `_predict_baseline()` heuristic**:
   - Replace crude fixed increments with a **morphogen-pathway lookup table** that maps signaling pathway activation levels to expected cell type shifts
   - Use the real training data distribution as the Dirichlet prior (instead of uniform 0.05)
   - Add dose-response curves: effect should saturate at high concentrations (sigmoid, not linear)
   - Add pathway antagonism: e.g., WNT agonist + WNT inhibitor should partially cancel

3. **Wire confidence scores into `merge_multi_fidelity_data()`** (addresses TODO item):
   - Load `cellflow_screening_report.csv` confidence column
   - Filter virtual points below confidence threshold (default 0.3)
   - Log how many points were filtered

4. **Run CellFlow virtual screen**:
   ```bash
   python 06_cellflow_virtual.py
   ```

**Tests**: 6-8 tests for improved heuristic (dose-response saturation, pathway cancellation, confidence filtering)

**Acceptance**: Heuristic predictions show region-specific composition patterns (not near-uniform). Confidence filtering removes >20% of extrapolated points.

---

## Phase 4: Pipeline Integration + Config-Driven Datasets

### 4A. Dataset Configuration System (~1.5 days)

**Goal**: Adding a new dataset should require only a YAML config, not code changes.

**Files to create/modify**:
- NEW: `gopro/dataset_config.py` -- dataset config loader
- NEW: `datasets/amin_kelley.yaml` -- example config
- NEW: `datasets/sag_screen.yaml` -- example config
- MODIFY: Steps 01, 02, 04 to read from config

**Tasks**:

1. **Define dataset config schema**:
   ```yaml
   # datasets/amin_kelley.yaml
   name: amin_kelley_2024
   source:
     type: geo
     accession: GSE233574
     format: mtx

   preprocessing:
     quality_column: quality
     quality_keep_value: keep
     batch_column: sample
     condition_column: condition

   morphogens:
     parser: AminKelleyParser  # class name in morphogen_parser.py
     harvest_day: 72

   mapping:
     reference: hnoca
     model: scpoli
     n_epochs: 500

   outputs:
     prefix: amin_kelley
     fractions_csv: gp_training_labels_amin_kelley.csv
     morphogen_csv: morphogen_matrix_amin_kelley.csv
     mapped_h5ad: amin_kelley_mapped.h5ad
   ```

2. **Create `gopro/dataset_config.py`**:
   - `load_dataset_config(path) -> DatasetConfig` (validated dataclass)
   - `discover_datasets(config_dir) -> list[DatasetConfig]` (find all .yaml files)
   - `resolve_paths(config) -> DatasetConfig` (resolve relative paths to absolute)

3. **Add `--config` CLI arg to step 02**:
   ```bash
   python 02_map_to_hnoca.py --config datasets/sag_screen.yaml
   ```
   This reads all parameters from the config instead of CLI args.

4. **Step 04 auto-discovers datasets**: scan `datasets/` directory, load all configs, merge training data from all discovered datasets.

**Tests**: 5-6 tests for config loading, validation, path resolution

**Acceptance**: Adding the SAG screen requires creating one YAML file. `python 02_map_to_hnoca.py --config datasets/sag_screen.yaml` runs correctly.

---

### 4B. Pipeline Orchestrator (Lightweight) (~1 day)

**Goal**: Single entry point that runs the pipeline with dependency tracking.

**File**: NEW `gopro/run_pipeline.py`

**Tasks**:

1. **Create a simple DAG-based runner**:
   ```python
   def run_pipeline(
       datasets: list[str] = None,     # dataset config names
       start_step: int = 0,            # resume from step N
       end_step: int = 6,              # stop after step N
       dry_run: bool = False,          # print what would run
       target_region: str = None,      # region targeting
   ) -> dict:
       """Run pipeline steps in order, checking prerequisites."""
   ```

2. **Step dependency checking**: before running step N, verify step N-1 outputs exist (using validators from Phase 1A)

3. **Progress reporting**: log which step is running, elapsed time, and output paths

4. **Skip logic**: if outputs already exist and are newer than inputs, skip the step (make-style)

**No tests needed**: This is a thin orchestration layer over tested functions.

**Acceptance**: `python -m gopro.run_pipeline --datasets amin_kelley --start-step 3 --end-step 4` runs steps 3 and 4 in sequence.

---

## Phase 5: Test Coverage + Polish

### 5A. Test Coverage Push (~1.5 days)

**Goal**: Get `02_map_to_hnoca.py` and `05_cellrank2_virtual.py` above 60% coverage.

**Strategy**: Use mock-heavy tests (these modules depend on scArches, moscot, CellRank which are hard to install in CI).

**Tasks**:

1. **Step 02 tests** (target: 60%+):
   - `test_filter_quality_cells_primary_screen`
   - `test_filter_quality_cells_sag_screen`
   - `test_prepare_query_gene_mapping` (mock ref with known genes)
   - `test_prepare_query_missing_batch_column`
   - `test_map_to_hnoca_scpoli_smoke` (mock scPoli model)
   - `test_transfer_labels_knn_class_balanced`
   - `test_transfer_labels_knn_unbalanced`
   - `test_soft_fractions_match_hard_fractions_uniform` (when all cells have same type, soft == hard)

2. **Step 05 tests** (target: 60%+, enabled by Phase 1B decomposition):
   - Test each extracted function independently
   - Mock moscot problem object with known transport matrices
   - Test fallback chain: push API fails -> transport matrix -> atlas average

3. **Step 03 tests** (target: 80%+):
   - `test_score_all_conditions_with_target_profile` (from Phase 2)
   - `test_compute_rss_perfect_match`
   - `test_compute_rss_no_overlap`
   - `test_normalized_entropy_edge_cases`

**Acceptance**: `python -m pytest gopro/tests/ -v` shows 230+ tests passing. Coverage report shows no module below 50%.

---

### 5B. Code Quality Polish (~0.5 day)

**Tasks**:
1. Remove suppressed warnings (`warnings.filterwarnings("ignore")`) -- replace with targeted suppression
2. Deduplicate `md5_file` across download scripts into a shared utility
3. Fix remaining type annotation gaps in download scripts
4. Clean up dead code (unused fixtures in conftest.py, unused imports)
5. Update `CLAUDE.md` with new architecture (region_targets.py, validation.py, dataset_config.py)

---

## Deferred Items (add to TODO.md, do NOT plan implementation)

These are explicitly deferred per user requirements:

| Item | Reason | Prerequisite |
|------|--------|-------------|
| Train CellFlow model | Requires GPU | GPU access |
| GPU acceleration for scPoli | Requires CUDA/MPS testing | GPU access |
| Literature scraping tool | User deferred | None |
| Cross-species reference integration | Low priority | Region targeting system |
| Heavy RDS conversion (22GB) | Run on server | Server access |
| disease_atlas.h5ad integration | Pending data exploration results | Parallel agent findings |
| Run step 02 for SAG screen | Needs scPoli (30-60 min CPU) | Verified data on disk |

---

## Execution Strategy

### For a Single Developer

**Week 1**: Phases 1A + 1B + 1C in parallel (3 different files, no conflicts)
**Week 2**: Phase 2A + 3A (region targeting + build temporal atlas)
**Week 3**: Phase 2B + 3B + 3C (label maps + CellRank2 + CellFlow heuristic)
**Week 4**: Phase 4A + 4B + 5A + 5B (integration + tests + polish)

### For Parallel Agents (recommended)

**Batch 1** (no dependencies):
- Agent A: Phase 1A (validation.py)
- Agent B: Phase 1B (decompose step 05)
- Agent C: Phase 1C (importable API)

**Batch 2** (depends on Batch 1):
- Agent A: Phase 2A (region targets)
- Agent B: Phase 3A (build temporal atlas) then Phase 3B (CellRank2)
- Agent C: Phase 3C (CellFlow heuristic)

**Batch 3** (depends on Batch 2):
- Agent A: Phase 4A + 4B (config + orchestrator)
- Agent B: Phase 5A + 5B (tests + polish)
- Agent C: Phase 2B (dynamic label maps)

---

## Success Criteria

The pipeline is "production-ready" when:

1. `python -m gopro.run_pipeline --datasets amin_kelley --target-region ventral_midbrain` runs end-to-end
2. `python 04_gpbo_loop.py --list-regions` shows all available brain regions
3. Virtual data (CellRank2 + CellFlow) is integrated into GP fitting
4. 230+ tests passing, no module below 50% coverage
5. Adding a new dataset requires only a YAML config file
6. Core functions are importable: `from gopro import run_gpbo_loop, score_all_conditions`
7. All inter-step transitions have validation with clear error messages
