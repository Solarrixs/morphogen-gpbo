# CLAUDE.md

## Project Overview

GP-BO (Gaussian Process Bayesian Optimization) pipeline for brain organoid morphogen protocol optimization. Uses active learning to find optimal morphogen combinations that drive brain organoids toward cell type compositions matching the human fetal brain.

## Repository Structure

```
morphogen-gpbo/
├── gopro/                    # Pipeline code (Python)
│   ├── 00_zenodo_download.py           # Download HNOCA + Braun references from Zenodo
│   ├── 00b_download_patterning_screen.py # Download patterning screen from Zenodo
│   ├── 01_load_and_convert_data.py     # Convert GEO MTX → AnnData h5ad
│   ├── 02_map_to_hnoca.py             # scArches/scPoli mapping + KNN label transfer
│   ├── 03_fidelity_scoring.py          # Two-tier fidelity scoring vs Braun fetal brain
│   ├── 04_gpbo_loop.py                 # BoTorch GP-BO with ILR transform, plate map output
│   ├── morphogen_parser.py             # Parse condition names → 20D concentration vectors
│   ├── requirements.txt
│   └── tests/                          # pytest test suite (65 tests)
│       ├── test_unit.py                # 35 unit tests
│       ├── test_integration.py         # 18 integration tests
│       └── test_properties.py          # 12 property-based tests (Hypothesis)
├── data/                     # All large data files (gitignored, see data/README.md)
│   ├── *.h5ad                          # Reference atlases + pipeline outputs
│   ├── GSE233574_*                     # GEO morphogen screen raw data
│   └── patterning_screen/             # Sanchis-Calleja/Azbukina dataset
├── neural_organoid_atlas/    # Cloned theislab HNOCA repo (gitignored)
│   └── supplemental_files/scpoli_model_params/  # Pre-trained scPoli model
└── docs/                     # Research notes and architecture docs
```

## Pipeline Steps

Run sequentially from `gopro/`:

```bash
python 00_zenodo_download.py            # Download reference atlases to data/
python 00b_download_patterning_screen.py # Download patterning screen to data/
python 01_load_and_convert_data.py       # Convert GEO MTX → AnnData h5ad
python 02_map_to_hnoca.py               # scArches/scPoli mapping + KNN label transfer
python 03_fidelity_scoring.py            # Two-tier fidelity scoring vs Braun fetal brain
python 04_gpbo_loop.py                   # Fit GP, acquisition function, output plate map
```

## Data

All data lives in `data/` (gitignored). See `data/README.md` for download instructions. Three sources:

- **Zenodo 15004817**: HNOCA reference (2.9 GB) + Braun fetal brain (11.2 GB)
- **GEO GSE233574** (Amin/Kelley): Primary morphogen screen — 46 conditions, scRNA-seq
- **Zenodo 17225179** (Sanchis-Calleja/Azbukina): Patterning screen (22.8 GB)

The `neural_organoid_atlas/` repo (cloned separately) provides the pre-trained scPoli model for step 02.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r gopro/requirements.txt
```

Key dependencies: scanpy, anndata, scvi-tools, scarches, hnoca, scikit-learn, scipy, botorch, gpytorch, torch, pytest, hypothesis.

## Architecture

**Data flow:** Raw scRNA-seq (GEO) → AnnData → scArches atlas mapping → cell type fractions per condition → fidelity scoring → GP-BO recommendation of next 24 conditions.

**Key domain concepts:**
- **20 morphogen dimensions** (`MORPHOGEN_COLUMNS` in `04_gpbo_loop.py`): WNT, BMP, SHH, RA, FGF, Notch, EGF signaling + harvest time
- **scArches/scPoli architecture surgery**: Transfer learning to map query organoid cells onto HNOCA reference atlas
- **Two-tier fidelity scoring** (step 03): Tier 1 = brain region assignment, Tier 2 = subtype fidelity via cosine similarity to Braun fetal brain
- **GP fitting**: BoTorch `SingleTaskGP` / `SingleTaskMultiFidelityGP` with Matérn 5/2 + ARD kernel
- **ILR transform**: Isometric log-ratio via Helmert basis for compositional Y data
- **Multi-objective acquisition**: `qLogNoisyExpectedHypervolumeImprovement` or scalarized `qLogExpectedImprovement`

## Conventions

- All pipeline scripts use `PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")` and `DATA_DIR = PROJECT_DIR / "data"`
- Tests in `gopro/tests/`, run with `python -m pytest gopro/tests/ -v`
- BoTorch uses CPU (MPS doesn't support float64 required by GP fitting)
- scPoli model expects `snapseed_pca_rss_level_*` column names (mapped from `annot_level_*`)
