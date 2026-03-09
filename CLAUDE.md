# CLAUDE.md

## Project Overview

GP-BO (Gaussian Process Bayesian Optimization) pipeline for brain organoid morphogen protocol optimization. Uses active learning to find optimal morphogen combinations that drive brain organoids toward cell type compositions matching the human fetal brain.

## Repository Structure

```
morphogen-gpbo/
├── gopro/                    # Pipeline code (Python)
│   ├── __init__.py
│   ├── config.py                       # Centralized paths, constants, logging (import from here)
│   ├── 00_zenodo_download.py           # Download HNOCA + Braun references from Zenodo
│   ├── 00a_download_geo.py             # Download GEO GSE233574 raw data (primary + SAG screens)
│   ├── 00b_download_patterning_screen.py # Download Sanchis-Calleja/Azbukina from Zenodo
│   ├── 00c_build_temporal_atlas.py     # Build temporal atlas from patterning screen
│   ├── 01_load_and_convert_data.py     # Convert GEO MTX → AnnData h5ad
│   ├── 02_map_to_hnoca.py             # scArches/scPoli mapping + KNN label transfer (multi-dataset via CLI)
│   ├── 03_fidelity_scoring.py          # Two-tier fidelity scoring vs Braun fetal brain
│   ├── 04_gpbo_loop.py                 # BoTorch GP-BO with ILR transform, auto-discovers SAG data
│   ├── 05_cellrank2_virtual.py         # CellRank 2 temporal projection via moscot
│   ├── 05_visualize.py                 # CLI wrapper → generates interactive HTML report
│   ├── 06_cellflow_virtual.py          # CellFlow virtual protocol screening
│   ├── gruffi_qc.py                    # Gruffi cell stress filtering (GO-term pathway scoring)
│   ├── morphogen_parser.py             # Parse condition names → 24D concentration vectors (48 total)
│   ├── qc_cross_screen.py             # Cross-screen QC validation (cosine similarity)
│   ├── convert_rds_to_h5ad.py         # RDS→h5ad conversion via R subprocess (requires R 4.2+)
│   ├── visualize_report.py             # Plotly report generation (called by 05_visualize.py)
│   ├── requirements.txt
│   ├── README.md
│   └── tests/                          # pytest test suite (165 tests)
│       ├── conftest.py                 # Shared fixtures + dynamic module loader
│       ├── test_unit.py                # Unit tests (ILR, morphogen parsing, QC, GP functions)
│       ├── test_integration.py         # Integration tests (all 46 primary conditions)
│       ├── test_properties.py          # Property-based tests (Hypothesis)
│       └── test_phase4_5.py            # Phase 4-5 tests (CellRank2 + CellFlow)
├── data/                     # All large data files (gitignored, see data/README.md)
│   ├── *.h5ad                          # Reference atlases + pipeline outputs
│   ├── GSE233574_*                     # GEO morphogen screen raw data
│   └── patterning_screen/             # Sanchis-Calleja/Azbukina dataset
├── data/neural_organoid_atlas/  # Cloned theislab HNOCA repo (gitignored, inside data/)
│   └── supplemental_files/scpoli_model_params/  # Pre-trained scPoli model
└── docs/                     # Research notes and architecture docs
    └── plans/                          # Design docs and implementation plans
```

## Config Module (`gopro/config.py`)

Centralized configuration — import paths and constants from here instead of defining locally.

- `PROJECT_DIR`, `DATA_DIR`, `MODEL_DIR` — derived from env vars (`GPBO_PROJECT_DIR`, `GPBO_DATA_DIR`, `GPBO_MODEL_DIR`) or auto-detected from `__file__`
- `MORPHOGEN_COLUMNS` — canonical list of 24 morphogen dimension names (all concentrations in µM)
- `PROTEIN_MW_KDA` — molecular weights (kDa) for recombinant protein morphogens
- `ng_mL_to_uM(ng_per_mL, mw_kda)`, `nM_to_uM(nM)` — unit conversion functions
- `ANNOT_LEVEL_1`, `ANNOT_LEVEL_2`, `ANNOT_REGION`, `ANNOT_LEVEL_3` — HNOCA annotation column names
- `get_logger(name)` — configured logger factory (level via `GPBO_LOG_LEVEL` env var, default INFO)

## Pipeline Steps

Run sequentially from `gopro/`:

```bash
# Download data
python 00_zenodo_download.py            # Download HNOCA + Braun references to data/
python 00a_download_geo.py              # Download GEO GSE233574 raw data to data/ (primary + SAG screens)
python 00b_download_patterning_screen.py # Download patterning screen to data/
python 00c_build_temporal_atlas.py       # Build temporal atlas from patterning screen

# Convert and parse
python 01_load_and_convert_data.py       # Convert GEO MTX → AnnData h5ad
python morphogen_parser.py              # Parse condition names → morphogen_matrix_amin_kelley.csv + morphogen_matrix_sag_screen.csv

# Map to reference atlas (run once per dataset)
python 02_map_to_hnoca.py               # Primary screen (default: amin_kelley_2024.h5ad)
python 02_map_to_hnoca.py --input data/amin_kelley_sag_screen.h5ad --output-prefix sag_screen  # SAG screen

# Score and optimize
python 03_fidelity_scoring.py            # Two-tier fidelity scoring vs Braun fetal brain
python 04_gpbo_loop.py                   # Fit GP, acquisition function, output plate map (auto-discovers SAG data)

# Virtual data augmentation
python 05_cellrank2_virtual.py           # CellRank 2 temporal projection → virtual data (fidelity=0.5)
python 06_cellflow_virtual.py            # CellFlow virtual protocol screening (fidelity=0.0)

# Visualization
python 05_visualize.py                   # Generate interactive HTML report (data/report_round{N}.html)

# Utilities (run as needed)
python convert_rds_to_h5ad.py INPUT.rds.gz              # Convert Seurat RDS → h5ad
python convert_rds_to_h5ad.py INPUT.rds.gz --check-only  # Inspect RDS metadata only
```

### Step 02 CLI Options

```
--input PATH           Input h5ad file (default: data/amin_kelley_2024.h5ad)
--output-prefix NAME   Output file prefix (default: amin_kelley)
--condition-key COL    obs column for conditions (default: condition)
--batch-key COL        obs column for batch/sample (default: sample)
```

Produces: `data/gp_training_labels_{prefix}.csv`, `data/gp_training_regions_{prefix}.csv`, `data/{prefix}_mapped.h5ad`

### Step 04 CLI Options

```
--fractions PATH       Cell type fractions CSV (default: data/gp_training_labels_amin_kelley.csv)
--morphogens PATH      Morphogen matrix CSV (default: data/morphogen_matrix_amin_kelley.csv)
--sag-fractions PATH   SAG screen fractions CSV (default: auto-discovers data/gp_training_labels_sag_screen.csv)
--sag-morphogens PATH  SAG screen morphogens CSV (default: auto-discovers data/morphogen_matrix_sag_screen.csv)
--cellrank2-fractions/--cellrank2-morphogens  CellRank2 virtual data (fidelity=0.5)
--cellflow-fractions/--cellflow-morphogens    CellFlow virtual data (fidelity=0.0)
--target-cell-types    Cell types to optimize for (default: all)
--n-recommendations N  Number of experiments to recommend (default: 24)
--round N              Optimization round number (default: 1)
--no-ilr               Disable ILR transform
--multi-objective      Use multi-objective acquisition (qLogNEHVI)
```

## Data

All data lives in `data/` (gitignored). See `data/README.md` for download instructions. Three sources:

- **Zenodo 15004817**: HNOCA reference (2.9 GB) + Braun fetal brain (11.2 GB)
- **GEO GSE233574** (Amin/Kelley): Primary morphogen screen (46 conditions) + SAG secondary screen (4 conditions, 2 unique new: SAG_50nM, SAG_2uM), scRNA-seq
- **Zenodo 17225179** (Sanchis-Calleja/Azbukina): Patterning screen (22.8 GB) — RDS files, convertible via `convert_rds_to_h5ad.py`

The `data/neural_organoid_atlas/` repo (cloned separately into `data/`) provides the pre-trained scPoli model for step 02.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r gopro/requirements.txt
```

Key dependencies: scanpy, anndata, scvi-tools, scarches, hnoca, scikit-learn, scipy, botorch, gpytorch, torch, plotly, pytest, hypothesis.

## Architecture

**Data flow:** Raw scRNA-seq (GEO) → AnnData → scArches atlas mapping → cell type fractions per condition → fidelity scoring → GP-BO recommendation of next 24 conditions → interactive visualization report. Multiple datasets (primary screen + SAG secondary screen) are mapped independently, producing separate CSVs that are merged at GP fitting time.

**Key domain concepts:**
- **24 morphogen dimensions** (`MORPHOGEN_COLUMNS` in `config.py`): WNT, BMP, SHH, RA, FGF, Notch, EGF signaling + harvest time + 4 base media columns (BDNF_uM, NT3_uM, cAMP_uM, AscorbicAcid_uM). All concentrations in µM.
- **48 parsed conditions**: 46 primary screen (Amin/Kelley, Day 72) + 2 SAG secondary screen (SAG_50nM, SAG_2uM, Day 70). SAG_250nM and SAG_1uM are near-duplicates of primary screen and are excluded.
- **Morphogen parser class hierarchy**: `MorphogenParser` base class → `AminKelleyParser` (46 conditions, Day 72), `SAGSecondaryParser` (2 conditions, Day 70), `CombinedParser` (merges parsers). `_BASE_MEDIA` dict provides default base media concentrations (BDNF, NT3, cAMP, Ascorbic Acid) appended to all 24-column vectors. Legacy functions (`parse_condition_name`, `build_morphogen_matrix`, `ALL_CONDITIONS`) remain for backward compatibility.
- **scArches/scPoli architecture surgery**: Transfer learning to map query organoid cells onto HNOCA reference atlas. Step 02 supports multiple datasets via `--input`/`--output-prefix` CLI args.
- **Cell filtering**: Primary screen uses `quality == 'keep'`; SAG screen uses `ClusterLabel != 'filtered'` (handled automatically by `filter_quality_cells()`).
- **Two-tier fidelity scoring** (step 03): Tier 1 = brain region assignment, Tier 2 = subtype fidelity via cosine similarity to Braun fetal brain
- **GP fitting**: BoTorch `SingleTaskGP` / `SingleTaskMultiFidelityGP` with Matérn 5/2 + ARD kernel; `--saasbo` flag enables `SaasFullyBayesianSingleTaskGP` with half-Cauchy sparsity prior
- **SAASBO**: Fully Bayesian GP via NUTS; wraps per-output models in `ModelListGP`; automatic variable selection in high-D morphogen space; `_extract_lengthscales` helper for diagnostics
- **Soft fractions**: `compute_soft_cell_type_fractions` averages per-cell KNN probabilities instead of hard argmax; saved as `gp_training_labels_soft_*.csv`
- **ILR transform**: Isometric log-ratio via Helmert basis for compositional Y data
- **Multi-objective acquisition**: `qLogNoisyExpectedHypervolumeImprovement` or scalarized `qLogExpectedImprovement`
- **Multi-fidelity GP integration**: `merge_multi_fidelity_data()` in step 04 combines real data (fidelity=1.0, from both primary and SAG screens) + CellRank2 virtual (0.5) + CellFlow virtual (0.0). SAG screen CSVs are auto-discovered if present.
- **Cross-screen QC** (`qc_cross_screen.py`): Validates overlapping conditions between screens via cosine similarity on cell type fraction vectors. Flags conditions below threshold (default 0.8).
- **CellRank 2 virtual data** (step 05): moscot OT maps on Azbukina temporal atlas → forward-project query cells via `.push()` API → medium-fidelity (0.5) training points; falls back to manual transport composition or atlas average
- **CellFlow virtual screening** (step 06): Protocol encoding (RDKit + ESM2) → generative model → low-fidelity (0.0) training points
- **RDS→h5ad converter** (`convert_rds_to_h5ad.py`): Converts Seurat RDS files via R subprocess (Seurat/SeuratDisk). Auto-discovers R at `/Library/Frameworks/R.framework/`, auto-installs missing R packages. Supports `--check-only` inspection mode and `.rds.gz` decompression.
- **Visualization report**: Self-contained Plotly HTML report showing optimization state per round

## Critical Files Reference

- `gopro/config.py` — `MORPHOGEN_COLUMNS`, `PROTEIN_MW_KDA`, `nM_to_uM`, `ng_mL_to_uM`, `get_logger`
- `gopro/gruffi_qc.py` — `score_gruffi()`, Gruffi cell stress filtering via GO-term pathway scoring
- `gopro/morphogen_parser.py` — `parse_condition_name()`, `build_morphogen_matrix()`, `ALL_CONDITIONS`, `SAG_SECONDARY_CONDITIONS`, `_BASE_MEDIA`, `AminKelleyParser`, `SAGSecondaryParser`, `CombinedParser`
- `gopro/02_map_to_hnoca.py` — `filter_quality_cells()`, `prepare_query_for_scpoli()`, `map_to_hnoca_scpoli()`, `transfer_labels_knn()`, `compute_cell_type_fractions()`
- `gopro/03_fidelity_scoring.py` — `score_all_conditions()`, `compute_composite_fidelity()`, `compute_rss()`, `build_hnoca_to_braun_label_map()`, `align_composition_to_braun()`
- `gopro/04_gpbo_loop.py` — `build_training_set()`, `merge_multi_fidelity_data()`, `run_gpbo_loop()`, `fit_gp_botorch()`, `recommend_next_experiments()`, `ilr_transform()`, `ilr_inverse()`, `_compute_active_bounds()`
- `gopro/05_cellrank2_virtual.py` — `load_temporal_atlas()`, `compute_transport_maps()`, `project_query_forward()`, `generate_virtual_training_data()`, `build_virtual_morphogen_matrix()`
- `gopro/06_cellflow_virtual.py` — `encode_protocol_cellflow()`, `generate_virtual_screen_grid()`, `predict_cellflow()`, `run_virtual_screen()`
- `gopro/qc_cross_screen.py` — `compute_cross_screen_similarity()`, `validate_cross_screen()`
- `gopro/convert_rds_to_h5ad.py` — `convert_rds_to_h5ad()`, `inspect_rds()`, `decompress_rds()`, `ensure_r_packages()`
- `gopro/visualize_report.py` — `generate_report()`, `assemble_html_report()`, figure builders (`build_morphogen_pca_figure`, `build_plate_map_figure`, `build_importance_figure`, `build_leaderboard_figure`, `build_composition_figure`, `build_convergence_figure`, `build_cell_umap_figure`, `extract_cell_umap_from_h5ad`)
- `gopro/tests/conftest.py` — `_import_pipeline_module()` dynamic module loader for numeric-prefixed filenames

## Conventions

- Import paths and constants from `gopro.config` (not hardcoded per script); env vars override defaults
- Import `MORPHOGEN_COLUMNS` and all shared constants from `gopro.config` — never use `importlib` to load other pipeline steps for constants
- Use `get_logger(__name__)` from `gopro.config` for logging
- Use `.copy()` before mutating DataFrames passed as function arguments (avoid in-place modification of caller's data)
- Avoid deprecated pandas APIs: use `isinstance(series.dtype, pd.CategoricalDtype)` instead of `pd.api.types.is_categorical_dtype()`
- Tests in `gopro/tests/`, run with `python -m pytest gopro/tests/ -v`
- Tests use dynamic module loading (via `conftest.py`) to handle numeric-prefixed filenames
- Test coverage target: prioritize `02_map_to_hnoca.py` and `05_cellrank2_virtual.py` (both below 20%); 165 tests total
- BoTorch uses CPU (MPS doesn't support float64 required by GP fitting)
- scPoli model expects `snapseed_pca_rss_level_*` column names (mapped from `annot_level_*`)

## Known Issues

- **Low test coverage**: `02_map_to_hnoca.py` (16%) and `05_cellrank2_virtual.py` (18%) have critical untested code paths.
