# Data Files

All large data files live in this directory and are **not tracked by git**. Follow the instructions below to download them.

## 1. HNOCA + Braun Fetal Brain References (Zenodo)

**Source:** Zenodo record [15004817](https://zenodo.org/records/15004817)

| File | Size | Description |
|------|------|-------------|
| `hnoca_minimal_for_mapping.h5ad` | 2.9 GB | HNOCA reference atlas for scArches/scPoli projection |
| `braun-et-al_minimal_for_mapping.h5ad` | 11.2 GB | Braun/Linnarsson fetal brain reference for fidelity scoring |

**Download:**
```bash
cd gopro
python 00_zenodo_download.py
```

Optional (not required for main pipeline):

| File | Size | Source |
|------|------|--------|
| `disease_atlas.h5ad` | 2.2 GB | Zenodo record [14161275](https://zenodo.org/records/14161275) — uncomment in `00_zenodo_download.py` |

## 2. GSE233574 — Amin/Kelley Morphogen Screen (GEO)

**Source:** [GSE233574](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233574)

Download these files from GEO and place them in this `data/` directory:

| File | Size | Description |
|------|------|-------------|
| `GSE233574_OrganoidScreen_counts.mtx.gz` | 445 MB | Primary screen count matrix (46 conditions) |
| `GSE233574_OrganoidScreen_cellMetaData.csv.gz` | 1.1 MB | Primary screen cell metadata |
| `GSE233574_OrganoidScreen_geneInfo.csv.gz` | 574 KB | Gene info |
| `GSE233574_Organoid.SAG.secondaryScreen_counts.mtx.gz` | 82 MB | SAG secondary screen counts |
| `GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv.gz` | 161 KB | SAG screen cell metadata |
| `GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv.gz` | 574 KB | SAG screen gene info |
| `GSE233574_OrganoidScreen_processed_SeuratObject.rds.gz` | 1.7 GB | Processed Seurat object (optional, for R users) |
| `GSE233574_Organoid.SAG.secondaryScreen_processed_SeuratObject.rds.gz` | 302 MB | SAG Seurat object (optional) |

## 3. Patterning Screen — Sanchis-Calleja/Azbukina et al.

**Source:** Zenodo record [17225179](https://zenodo.org/records/17225179)
**Paper:** "Systematic scRNAseq screens profile neural organoid response to morphogens" (2025, Nature Methods)

| File | Size | Description |
|------|------|-------------|
| `patterning_screen/OSMGT_processed_files.tar.gz` | 22.8 GB | Processed scRNA-seq data |

**Download:**
```bash
cd gopro
python 00b_download_patterning_screen.py
```

## 4. Neural Organoid Atlas (HNOCA)

The `neural_organoid_atlas/` directory (in project root, not in `data/`) is cloned from the [theislab HNOCA reproducibility repo](https://github.com/theislab/neural_organoid_atlas). It contains the trained scPoli model parameters needed for step 02:

```
neural_organoid_atlas/supplemental_files/scpoli_model_params/
├── model_params.pt
├── attr.pkl
└── var_names.csv
```

Clone it with:
```bash
git clone https://github.com/theislab/neural_organoid_atlas.git
```

## Pipeline-Generated Files

These are created by the pipeline steps and don't need to be downloaded:

| File | Created by | Description |
|------|-----------|-------------|
| `amin_kelley_2024.h5ad` | Step 01 | Converted primary screen AnnData |
| `amin_kelley_sag_screen.h5ad` | Step 01 | Converted SAG screen AnnData |
| `amin_kelley_mapped.h5ad` | Step 02 | HNOCA-mapped with cell type labels |
| `gp_training_labels_amin_kelley.csv` | Step 02 | Cell type fractions per condition |
| `amin_kelley_fidelity.h5ad` | Step 03 | With fidelity scores |
| `fidelity_report.csv` | Step 03 | Per-condition fidelity summary |
