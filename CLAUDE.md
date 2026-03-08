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
│   ├── 00a_download_geo.py             # Download GEO GSE233574 raw data
│   ├── 00b_download_patterning_screen.py # Download Sanchis-Calleja/Azbukina from Zenodo
│   ├── 00c_build_temporal_atlas.py     # Build temporal atlas from patterning screen
│   ├── 01_load_and_convert_data.py     # Convert GEO MTX → AnnData h5ad
│   ├── 02_map_to_hnoca.py             # scArches/scPoli mapping + KNN label transfer
│   ├── 03_fidelity_scoring.py          # Two-tier fidelity scoring vs Braun fetal brain
│   ├── 04_gpbo_loop.py                 # BoTorch GP-BO with ILR transform, plate map output
│   ├── 05_cellrank2_virtual.py         # CellRank 2 temporal projection via moscot
│   ├── 05_visualize.py                 # CLI wrapper → generates interactive HTML report
│   ├── 06_cellflow_virtual.py          # CellFlow virtual protocol screening
│   ├── morphogen_parser.py             # Parse condition names → 20D concentration vectors
│   ├── visualize_report.py             # Plotly report generation (called by 05_visualize.py)
│   ├── requirements.txt
│   ├── README.md
│   └── tests/                          # pytest test suite
│       ├── conftest.py                 # Shared fixtures + dynamic module loader
│       ├── test_unit.py                # Unit tests (ILR, morphogen parsing, GP functions)
│       ├── test_integration.py         # Integration tests (all 46 real conditions)
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
- `MORPHOGEN_COLUMNS` — canonical list of 20 morphogen dimension names
- `ANNOT_LEVEL_1`, `ANNOT_LEVEL_2`, `ANNOT_REGION`, `ANNOT_LEVEL_3` — HNOCA annotation column names
- `get_logger(name)` — configured logger factory (level via `GPBO_LOG_LEVEL` env var, default INFO)

## Pipeline Steps

Run sequentially from `gopro/`:

```bash
python 00_zenodo_download.py            # Download HNOCA + Braun references to data/
python 00a_download_geo.py              # Download GEO GSE233574 raw data to data/
python 00b_download_patterning_screen.py # Download patterning screen to data/
python 00c_build_temporal_atlas.py       # Build temporal atlas from patterning screen
python 01_load_and_convert_data.py       # Convert GEO MTX → AnnData h5ad
python morphogen_parser.py              # Parse condition names → morphogen_matrix_amin_kelley.csv
python 02_map_to_hnoca.py               # scArches/scPoli mapping + KNN label transfer
python 03_fidelity_scoring.py            # Two-tier fidelity scoring vs Braun fetal brain
python 04_gpbo_loop.py                   # Fit GP, acquisition function, output plate map
python 05_cellrank2_virtual.py           # CellRank 2 temporal projection → virtual data (fidelity=0.5)
python 05_visualize.py                   # Generate interactive HTML report (data/report_round{N}.html)
python 06_cellflow_virtual.py            # CellFlow virtual protocol screening (fidelity=0.0)
```

## Data

All data lives in `data/` (gitignored). See `data/README.md` for download instructions. Three sources:

- **Zenodo 15004817**: HNOCA reference (2.9 GB) + Braun fetal brain (11.2 GB)
- **GEO GSE233574** (Amin/Kelley): Primary morphogen screen — 46 conditions, scRNA-seq
- **Zenodo 17225179** (Sanchis-Calleja/Azbukina): Patterning screen (22.8 GB)

The `data/neural_organoid_atlas/` repo (cloned separately into `data/`) provides the pre-trained scPoli model for step 02.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r gopro/requirements.txt
```

Key dependencies: scanpy, anndata, scvi-tools, scarches, hnoca, scikit-learn, scipy, botorch, gpytorch, torch, plotly, pytest, hypothesis.

## Architecture

**Data flow:** Raw scRNA-seq (GEO) → AnnData → scArches atlas mapping → cell type fractions per condition → fidelity scoring → GP-BO recommendation of next 24 conditions → interactive visualization report.

**Key domain concepts:**
- **20 morphogen dimensions** (`MORPHOGEN_COLUMNS` in `config.py`): WNT, BMP, SHH, RA, FGF, Notch, EGF signaling + harvest time
- **scArches/scPoli architecture surgery**: Transfer learning to map query organoid cells onto HNOCA reference atlas
- **Two-tier fidelity scoring** (step 03): Tier 1 = brain region assignment, Tier 2 = subtype fidelity via cosine similarity to Braun fetal brain
- **GP fitting**: BoTorch `SingleTaskGP` / `SingleTaskMultiFidelityGP` with Matérn 5/2 + ARD kernel
- **ILR transform**: Isometric log-ratio via Helmert basis for compositional Y data
- **Multi-objective acquisition**: `qLogNoisyExpectedHypervolumeImprovement` or scalarized `qLogExpectedImprovement`
- **CellRank 2 virtual data** (step 05): moscot OT maps on Azbukina temporal atlas → forward-project query cells → medium-fidelity (0.5) training points
- **CellFlow virtual screening** (step 06): Protocol encoding (RDKit + ESM2) → generative model → low-fidelity (0.0) training points
- **Multi-fidelity GP integration**: `merge_multi_fidelity_data()` in step 04 combines real (1.0) + CellRank2 (0.5) + CellFlow (0.0) data
- **Visualization report**: Self-contained Plotly HTML report showing optimization state per round

## Conventions

- Import paths and constants from `gopro.config` (not hardcoded per script); env vars override defaults
- Use `get_logger(__name__)` from `gopro.config` for logging
- Tests in `gopro/tests/`, run with `python -m pytest gopro/tests/ -v`
- Tests use dynamic module loading (via `conftest.py`) to handle numeric-prefixed filenames
- BoTorch uses CPU (MPS doesn't support float64 required by GP fitting)
- scPoli model expects `snapseed_pca_rss_level_*` column names (mapped from `annot_level_*`)
