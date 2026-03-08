# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GP-BO (Gaussian Process Bayesian Optimization) pipeline for brain organoid morphogen protocol optimization. Uses active learning to find optimal morphogen combinations that drive brain organoids toward cell type compositions matching the human fetal brain.

## Repository Structure

- **`gopro/`** — Main pipeline code (Python). This is where active development happens.
- **`neural_organoid_atlas/`** — Cloned HNOCA reproducibility repo (theislab). Contains the trained scPoli model params at `supplemental_files/scpoli_model_params/`.
- **`patterning_screen/`** — Downloaded patterning screen data.
- Root-level `.h5ad` files and `GSE233574_*` files are large data assets.

## Pipeline Steps (run sequentially from `gopro/`)

```bash
python 01_load_and_convert_data.py   # Convert GEO MTX → AnnData h5ad
python 02_map_to_hnoca.py            # scArches/scPoli mapping + KNN label transfer
python 03_fidelity_scoring.py        # Two-tier fidelity scoring vs Braun fetal brain
python 04_gpbo_loop.py               # Fit GP, UCB/EI acquisition, output plate map CSV
```

Data download scripts: `00_zenodo_download.py`, `00b_download_patterning_screen.py`.

## Environment

```bash
pip install -r gopro/requirements.txt
```

Python venv at `.venv/`. Key dependencies: scanpy, anndata, scvi-tools, scarches, hnoca, scikit-learn, scipy. BoTorch/GPyTorch planned but not yet installed.

## Architecture

**Data flow:** Raw scRNA-seq (GEO) → AnnData → scArches atlas mapping → cell type fractions per condition → fidelity scoring → GP-BO recommendation of next 24 conditions.

**Key domain concepts:**
- **17 morphogen dimensions** (`MORPHOGEN_COLUMNS` in `04_gpbo_loop.py`): WNT, BMP, SHH, RA, FGF, Notch, EGF signaling + harvest time
- **scArches/scPoli architecture surgery**: Transfer learning to map query organoid cells onto HNOCA reference atlas
- **Two-tier fidelity scoring** (step 03): Tier 1 = brain region assignment, Tier 2 = subtype fidelity
- **GP fitting**: Currently scikit-learn `GaussianProcessRegressor` with Matérn 5/2 + ARD kernel; production upgrade to BoTorch planned

**Data sources:**
- GSE233574 (Amin/Kelley): Primary morphogen screen (46 conditions)
- HNOCA (Zenodo 15004817): 1.77M cell reference atlas
- Braun et al.: Fetal brain reference for fidelity scoring

## No Tests / CI / Linting

This is a research pipeline — no test suite, CI/CD, or linting configuration exists.
