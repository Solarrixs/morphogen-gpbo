# Morphogen GP-BO Pipeline

Pipeline for Gaussian Process Bayesian Optimization of brain organoid morphogen protocols.

## Setup

```bash
pip install -r requirements.txt
```

## Data Downloads

### Already downloaded:
- GSE233574 (Amin/Kelley morphogen screen) - in project root
- scPoli trained model - in `neural_organoid_atlas/supplemental_files/scpoli_model_params/`
- Disease atlas - `disease_atlas.h5ad`

### Still needed:
```bash
bash ../download_zenodo.sh
```
This downloads from Zenodo 15004817:
- `hnoca_minimal_for_mapping.h5ad` (2.9 GB) - HNOCA reference for scArches projection
- `braun-et-al_minimal_for_mapping.h5ad` (11.2 GB) - Fetal brain reference for fidelity scoring

## Pipeline Steps

```
01_load_and_convert_data.py   → Convert Seurat RDS to AnnData h5ad
02_map_to_hnoca.py            → Project onto HNOCA, get cell type labels
03_fidelity_scoring.py        → Score against fetal brain atlas
04_gpbo_loop.py               → Fit GP, recommend next experiments
```

Run in order:
```bash
python 01_load_and_convert_data.py
python 02_map_to_hnoca.py
python 03_fidelity_scoring.py
python 04_gpbo_loop.py
```
