# GP-BO Pipeline Comprehensive Audit Report

**Date:** 2026-03-14
**Scope:** Full codebase review — data, math, architecture, code quality, data standardization
**Method:** 6 parallel audit agents covering orthogonal aspects

---

## Executive Summary

The pipeline has **5 critical bugs** that likely explain why GP recommendations don't make biological sense, **10 major issues** that degrade quality or block extensibility, and **15+ minor issues**. The most impactful finding: **the GP is optimizing over harvest days with zero training data** (Day 2–45 when all data is Day 72), wasting exploration budget on a meaningless dimension.

---

## Critical Issues (5)

### C1. `log_harvest_day` bounds cause nonsensical recommendations
**File:** `04_gpbo_loop.py:113-116`
**Impact:** GP recommendations include harvest days from Day 2 to Day 45 when ALL training data is at Day 72.

When `log_harvest_day` has zero variance (single timepoint), the code sets bounds to `(log(7), log(120))` — the full literature range. The GP has zero information at any other harvest day, so the acquisition function explores this dimension blindly. This wastes the GP's exploration budget and produces biologically meaningless recommendations.

**Fix:** When constant, fix bounds to `(constant_value, constant_value)` or add only ±10% padding. Only expand to literature range when multiple timepoints exist in training.

### C2. CellFlow virtual morphogens missing 4 base media columns
**File:** `06_cellflow_virtual.py` → `cellflow_virtual_morphogens_200.csv`
**Impact:** GP learns that CellFlow experiments have no base media (BDNF=0, NT3=0, cAMP=0, AscorbicAcid=0) when real experiments have cAMP=50µM and AscorbicAcid=200µM.

`merge_multi_fidelity_data()` zero-fills missing columns. The massive feature discrepancy (200µM vs 0µM for AscorbicAcid) corrupts the GP's learned morphogen-outcome relationships.

**Fix:** In `generate_virtual_screen_grid()`, include base media columns from `_BASE_MEDIA` dict with their constant values, not 0.0.

### C3. CellRank2 data chain completely broken
**File:** `azbukina_temporal_atlas.h5ad` does not exist on disk
**Impact:** The multi-fidelity GP has ZERO medium-fidelity (0.5) data points. Step 00c was never run.

22 GB of patterning screen data sits unused. The `cellrank2_virtual_fractions.csv` and `cellrank2_virtual_morphogens.csv` were never generated.

**Fix:** Run `python 00c_build_temporal_atlas.py`, then `python 05_cellrank2_virtual.py`.

### C4. Predictions output in ILR space without inverse transform
**File:** `04_gpbo_loop.py:524`
**Impact:** `gp_recommendations_round1.csv` shows `predicted_y0_mean`, `predicted_y1_mean` — these are ILR coordinates (log-ratios), not cell type fractions. Users cannot interpret the GP's predictions.

**Fix:** Apply `ilr_inverse()` to the predicted means before saving. Map columns back to cell type names.

### C5. Multi-objective `ref_point=[0]*D` invalid for ILR-transformed data
**File:** `04_gpbo_loop.py:468-469`
**Impact:** ILR coordinates range over all of R (including negative). A zero reference point may not be dominated by any training point, making the hypervolume computation degenerate.

**Fix:** Set `ref_point = train_Y.min(dim=0).values - 0.1 * (train_Y.max(dim=0).values - train_Y.min(dim=0).values)`.

---

## Major Issues (10)

### M1. Equal-weight scalarization in ILR space has no compositional interpretation
**File:** `04_gpbo_loop.py:484-494`
Equal weighting of ILR coordinates does not correspond to equal weighting of cell type fractions. The GP optimizes an arbitrary objective in log-ratio space.

**Fix:** Define weights in composition space and transform to ILR, or use multi-objective path.

### M2. GP lower bounds prevent exploring zero concentration
**File:** `04_gpbo_loop.py:125`
`lower = max(0.0, col_min)` means if CHIR was always ≥1.5µM in training, the GP cannot recommend trying 0µM CHIR.

**Fix:** Use `lower = 0.0` for all morphogen dimensions.

### M3. Normalized entropy uses wrong denominator
**File:** `03_fidelity_scoring.py:283`
Divides by `log2(n_nonzero_types)` instead of `log2(total_types)`. A condition with 3 equal types scores 1.0, same as one with 15 equal types.

**Fix:** Use `h_max = np.log2(total_cell_types)` where total is the number of columns in the fractions matrix.

### M4. `build_training_set()` mutates caller's DataFrame
**File:** `04_gpbo_loop.py:235`
`X["fidelity"] = fidelity` modifies the CSV-loaded DataFrame in-place.

**Fix:** Add `X = X.copy()` before adding fidelity column.

### M5. Time-fraction encoding conflates duration with concentration
**File:** `morphogen_parser.py:82-92`
"CHIR applied days 11-16" → `1.5 * 5/15 = 0.5 µM`. The GP interprets this as "half the concentration applied continuously." This is a design limitation that prevents the GP from learning time-dependent effects.

**Fix (long-term):** Encode temporal windows as separate dimensions (start_day, end_day) rather than scaling concentration.

### M6. `warnings.filterwarnings("ignore")` in 4 files
**Files:** `02_map_to_hnoca.py:34`, `03_fidelity_scoring.py:33`, `05_cellrank2_virtual.py:41`, `06_cellflow_virtual.py:38`
GP convergence failures, scipy warnings, and pandas deprecations are silently swallowed.

**Fix:** Replace with targeted suppression for known harmless warnings only.

### M7. `preprocess_for_moscot` mutates input AnnData in-place
**File:** `05_cellrank2_virtual.py:97-125`
`sc.pp.normalize_total` and `sc.pp.log1p` modify `adata.X` in-place, corrupting raw counts.

**Fix:** Add `adata = adata.copy()` at start of function.

### M8. O(n*k) Python double-loop in transport push API
**File:** `05_cellrank2_virtual.py:469-473`
Nested Python loops over cells × neighbors. Should be vectorized.

**Fix:** Use `np.add.at(source_dist, cond_nn[valid_mask], cond_w[valid_mask])`.

### M9. Pipeline locked to HNOCA/Braun regions
**File:** `03_fidelity_scoring.py` — `OFF_TARGET_LEVEL1`, `BRAUN_NEURAL_CLASSES`, `HNOCA_TO_BRAUN_REGION`
Three hardcoded dicts prevent targeting arbitrary brain regions.

**Fix:** Extract to configurable `ReferenceProfile` class loadable from YAML.

### M10. Bounds padding uses multiplicative scaling
**File:** `04_gpbo_loop.py:122`
`col_max * (1.0 + padding)` fails for near-zero values (almost no exploration room).

**Fix:** Use `col_max + padding * (col_max - col_min)` or additive minimum margin.

---

## Minor Issues (15+)

| # | File | Issue |
|---|------|-------|
| 1 | `04_gpbo_loop.py:531` | Well labels assume 24-well plate, no validation for other sizes |
| 2 | `04_gpbo_loop.py:277` | `_extract_lengthscales` silently swallows errors |
| 3 | `02_map_to_hnoca.py:549` | Soft fractions uses hardcoded `condition_key="condition"` ignoring CLI arg |
| 4 | `02_map_to_hnoca.py:551` | Soft fractions output ignores `--output-prefix` |
| 5 | `02_map_to_hnoca.py:96-100` | `prepare_query_for_scpoli` mutates `query.var_names` in-place |
| 6 | `05_cellrank2_virtual.py:798` | Docstring says Day 21, code uses Day 72; `PROJECTION_TARGETS` constant misleading |
| 7 | `03_fidelity_scoring.py:363` | `compute_on_target_fraction` crashes on empty subset (IndexError) |
| 8 | `03_fidelity_scoring.py:112` | `IsNeural == True` may fail if stored as string |
| 9 | `03_fidelity_scoring.py:410` | Entropy penalty parameters (0.55, 0.2) are uncalibrated heuristics |
| 10 | `06_cellflow_virtual.py:407-466` | Baseline predictor positional fallbacks produce meaningless predictions for level-2 labels |
| 11 | `config.py` + `gruffi_qc.py` | Duplicate Gruffi constants not synchronized |
| 12 | `convert_rds_to_h5ad.py:27` | Hardcodes macOS R framework path |
| 13 | `gruffi_qc.py:236` | `identify_stressed_clusters` copies entire AnnData (doubles memory) |
| 14 | `morphogen_parser.py:407` | `assert len == 48` causes import failure when adding conditions |
| 15 | `visualize_report.py:86` | `load_cell_type_fractions` assumes index named "condition" |

---

## Data File Inventory

### Actively Used (keep)
| File | Size | Used By |
|------|------|---------|
| `hnoca_minimal_for_mapping.h5ad` | 2.9 GB | Steps 02, 03 |
| `braun-et-al_minimal_for_mapping.h5ad` | 11 GB | Step 03 |
| `amin_kelley_2024.h5ad` | 694 MB | Step 02 |
| `amin_kelley_sag_screen.h5ad` | 132 MB | Step 02 (needs to be run) |
| `amin_kelley_mapped.h5ad` | 24 MB | Steps 03, 05, visualization |
| `scpoli_model_params/` | ~few MB | Step 02 |
| All pipeline CSVs | small | Various |

### Unused / Redundant (47 GB savings possible)
| File | Size | Status |
|------|------|--------|
| `OSMGT_processed_files.tar.gz` | 23 GB | Redundant (already extracted) |
| `OSMGT.rds.gz` | 12 GB | Unused (h5ad available) |
| `4_M_vs_sM_21d_clean.rds.gz` | 3.1 GB | Unused |
| `d9_mistr_cleaned.rds.gz` | 2.5 GB | Unused |
| `disease_atlas.h5ad` | 2.2 GB | Never referenced in pipeline |
| GEO `.rds` files (2 Seurat objects) | 2 GB | Redundant (MTX+CSV used) |
| `amin_kelley_fidelity.h5ad` | 24 MB | Orphaned output (no downstream consumer) |
| `gp_training_labels_demo.csv` | 8 KB | Stale demo data |
| `HumanFetalBrainPool_cluster_expr_highvar.tsv` | 39 MB | Unused |

### Broken Chain
| File | Expected From | Needed By | Status |
|------|--------------|-----------|--------|
| `azbukina_temporal_atlas.h5ad` | Step 00c | Step 05 | **MISSING** |
| `cellrank2_virtual_fractions.csv` | Step 05 | Step 04 | **NOT GENERATED** |
| `cellrank2_virtual_morphogens.csv` | Step 05 | Step 04 | **NOT GENERATED** |
| `gp_training_labels_sag_screen.csv` | Step 02 (SAG) | Step 04 | **MISSING** (step 02 never run for SAG) |

---

## Mathematical Verification Summary

### Confirmed Correct
- Helmert basis ILR construction (matches Egozcue et al., 2003)
- ILR forward/inverse transforms (epsilon=1e-10 appropriate)
- `SingleTaskMultiFidelityGP` API usage and fidelity column handling
- Fidelity fixed at 1.0 during acquisition (correct practice)
- `best_f` computation for scalarized qLogEI
- KNN class-balanced `1/sqrt(freq)` weighting (standard technique)
- Soft probability distributions from weighted votes
- Transport map composition via matrix multiplication
- ng_mL_to_uM and nM_to_uM conversion formulas
- All 48 morphogen handler functions produce correct µM values
- Cosine similarity zero-vector handling
- CellFlow confidence decay function

### Issues Found
See Critical C4, C5 and Major M1, M2, M3, M5, M10 above.

---

## Architecture & Modularity Summary

### Extensibility Assessment
| Requirement | Status | Effort |
|---|---|---|
| Target any brain region | Blocked (3 hardcoded dicts) | Medium |
| Add new datasets | Partially supported (CLI in step 02) | Medium |
| Configurable reference profiles | Not possible | Medium |
| Mouse atlas support | Blocked (human-only gene sets, HNOCA columns) | High |
| Checkpoint/resume | Not implemented | Medium |
| Data validation between steps | Not implemented | Low |

### CellFlow Assessment
- Real code: `encode_protocol_cellflow()`, `generate_virtual_screen_grid()`, `compute_prediction_confidence()`, SMILES/identity mappings
- Placeholder: `_predict_with_cellflow()` (untested API), `_predict_baseline()` (heuristic)
- **To make functional:** Install CellFlow, train model on patterning screen data (requires GPU)

---

## Prioritized Fix Plan

### Phase 1: Fix Critical Bugs (must do before next GP run)
1. Fix `log_harvest_day` bounds to constant when single timepoint
2. Add base media columns to CellFlow virtual morphogen grid
3. Apply `ilr_inverse()` to GP predictions before saving
4. Fix multi-objective reference point for ILR data
5. Add `X = X.copy()` in `build_training_set()`

### Phase 2: Run Missing Pipeline Steps
6. Run step 02 for SAG screen: `python 02_map_to_hnoca.py --input data/amin_kelley_sag_screen.h5ad --output-prefix sag_screen`
7. Run step 00c to build temporal atlas
8. Run step 05 for CellRank2 virtual data

### Phase 3: Fix Major Issues
9. Fix GP lower bounds (always allow 0.0 for morphogens)
10. Fix normalized entropy denominator
11. Replace global warning suppression with targeted filters
12. Fix bounds padding to use additive/range-relative scaling
13. Fix label harmonization vocabulary in step 05

### Phase 4: Architecture Improvements
14. Extract hardcoded region/cell type dicts to configurable profiles
15. Decompose `project_query_forward()` into smaller functions
16. Add data validation between pipeline steps
17. Make morphogen parser extensible via config files
18. Add checkpoint/resume mechanism

### Phase 5: Feature Development
19. Train CellFlow model from patterning screen data
20. Build literature scraping tool
21. Add GPU acceleration for scPoli
22. Add cross-species reference support

---

## Appendix: Missing Brain Regions Analysis

### Current Coverage

HNOCA has 10 regions at `annot_region_rev2` level. Critically, **HNOCA already has hippocampal cell types at level 3** (`hippocampal_npc`, `hippocampal_excitatory_neuron`) grouped under "Dorsal telencephalon" — but the pipeline never uses level-3 for GP-BO targeting.

| Requested Region | HNOCA Status | Braun Status | Gap |
|---|---|---|---|
| Entorhinal cortex | Not present at any level | Not a separate region | Need external reference (Siletti 2023 or Grubman 2019) |
| Hippocampus | Level-3 cell types exist! | Grouped under Dorsal telencephalon | Enable level-3 targeting — immediate win |
| Substantia nigra | Grouped under Ventral midbrain | No DaN-specific labels | Need BrainSTEM (Toh 2025) or Agarwal 2020 |
| Basal ganglia | Not present | Not present | Need Siletti 2023 |

### Recommended Reference Datasets

| Dataset | DOI | Size | Regions Covered | Priority |
|---|---|---|---|---|
| **Siletti et al. 2023** (Human Brain Cell Atlas) | 10.1126/science.add7046 | 3M+ nuclei, ~100 dissections | ALL — hippocampus, EC, SN, basal ganglia, thalamic nuclei, cortical areas | **Highest** |
| **BrainSTEM (Toh 2025)** | 10.1126/sciadv.adu7944 | 680K cells | Midbrain subatlas with DaN subtypes | High (for PD) |
| **Agarwal 2020** | 10.1038/s41467-020-17876-0 | 17K nuclei | Substantia nigra specifically | Medium |
| **Su 2022** | 10.1016/j.stem.2022.09.010 | Hippocampal glia | Hippocampus glial subtypes | Medium |
| **Grubman 2019** | 10.1038/s41593-019-0539-4 | 13K nuclei | Entorhinal cortex (AD + control) | Medium |

### Immediate Wins (No New Data Needed)

1. **Enable level-3 targeting** — HNOCA already distinguishes hippocampal cell types. Add `--target-level level3` to step 04.
2. **Integrate patterning screen as GP training data** — Sanchis-Calleja has 176 conditions with diverse regionalization (cerebellum, pons, medulla subtypes). Map through scPoli (step 02 already supports `--input`).
3. **Use disease atlas controls** — Additional training points from different protocols.

### Architecture for Multi-Reference Support

Refactor `03_fidelity_scoring.py` to support pluggable reference profiles:
```python
class FidelityReference:
    def extract_profiles(self) -> pd.DataFrame: ...
    def get_region_mapping(self) -> dict: ...

class BraunReference(FidelityReference): ...
class SilettiReference(FidelityReference): ...
class BrainSTEMReference(FidelityReference): ...
```

Add marker-gene-based scoring for regions not covered by any atlas (RELN for entorhinal stellate cells, TH/NURR1/PITX3 for dopaminergic neurons).
