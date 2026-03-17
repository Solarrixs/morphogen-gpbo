# Data Directory Inventory Report

Generated: 2026-03-15
Total data directory size: **63 GB**

## Summary

| Category | Size | Files | Status |
|----------|------|-------|--------|
| Actively used by pipeline | ~15 GB | 6 h5ad + CSVs + scPoli model | OK |
| Raw GEO source (consumed by Step 01) | ~2.3 GB | 6 files (MTX + CSV) | Keep (input) |
| Redundant GEO RDS (Seurat objects) | ~2.0 GB | 2 files | Delete candidate |
| Patterning screen (tar.gz + extracted) | ~42 GB | 1 tar.gz + 4 extracted files | Partially used |
| Disease atlas | ~2.2 GB | 1 h5ad | Not consumed |
| HNOCA repo (notebooks/scripts) | ~562 MB | Entire cloned repo | Only scPoli model used |
| Pipeline-generated CSVs | ~2 MB | ~15 files | Keep (outputs) |

---

## File-by-File Inventory

### h5ad Files

| File | Size | Shape | Used By | Recommendation |
|------|------|-------|---------|----------------|
| `hnoca_minimal_for_mapping.h5ad` | 2.9 GB | - | Step 02, 03 | **Keep** (actively used) |
| `braun-et-al_minimal_for_mapping.h5ad` | 11 GB | - | Step 03 | **Keep** (actively used) |
| `amin_kelley_2024.h5ad` | 694 MB | - | Step 01/02 | **Keep** (actively used) |
| `amin_kelley_sag_screen.h5ad` | 132 MB | - | Step 02 | **Keep** (actively used) |
| `amin_kelley_mapped.h5ad` | 24 MB | 36265x3000 | Step 03, 05 | **Keep** (actively used) |
| `amin_kelley_fidelity.h5ad` | 24 MB | 36265x3000 | Step 03 output | **Keep** (pipeline output; same as mapped + 4 fidelity columns: `fidelity_score`, `rss_score`, `on_target_fraction`, `is_off_target`) |
| `disease_atlas.h5ad` | **2.2 GB** | 409277x14323 | **Only checked for existence** in `01_load_and_convert_data.py:verify_references()`. Never actually loaded or processed by any pipeline step. Contains disease organoid data (ALS/FTD, glioblastoma, ASD, etc.) with 15 obs columns including `annot_level_*_plus` annotations. | **Defer/Delete** — could be valuable for disease-specific optimization but currently unused. Remove from `verify_references()` if deleting. |

### GEO Raw Data (GSE233574)

| File | Size | Used By | Recommendation |
|------|------|---------|----------------|
| `GSE233574_OrganoidScreen_counts.mtx` | 1.9 GB | Step 01 (`load_and_convert_data.py`) | **Keep** (raw input, consumed to produce `amin_kelley_2024.h5ad`) |
| `GSE233574_OrganoidScreen_cellMetaData.csv` | 4.3 MB | Step 01 | **Keep** (53k cells, raw metadata) |
| `GSE233574_OrganoidScreen_geneInfo.csv` | 2.7 MB | Step 01 | **Keep** (gene annotations) |
| `GSE233574_Organoid.SAG.secondaryScreen_counts.mtx` | 323 MB | Step 01 | **Keep** (raw input for SAG screen) |
| `GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv` | 641 KB | Step 01 | **Keep** (6k cells, raw metadata) |
| `GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv` | 2.7 MB | Step 01 | **Keep** (gene annotations) |
| `GSE233574_OrganoidScreen_processed_SeuratObject.rds` | **1.7 GB** | **Not used** — pipeline uses MTX+CSV, not RDS | **Delete** — redundant with counts.mtx + cellMetaData.csv above |
| `GSE233574_Organoid.SAG.secondaryScreen_processed_SeuratObject.rds` | **304 MB** | **Not used** — pipeline uses MTX+CSV, not RDS | **Delete** — redundant with SAG MTX+CSV above |

Note: `01_load_and_convert_data.py` also references two optional RDS files (`GSE233574_hCbO_processed_SeuratObject.rds`, `GSE233574_hMPO_processed_SeuratObject.rds`) for cerebellar and medial pallium organoid validation data. These are **not present** in the data directory and are listed as optional future downloads.

### Patterning Screen (42 GB total)

| File | Size | Used By | Recommendation |
|------|------|---------|----------------|
| `patterning_screen/OSMGT_processed_files.tar.gz` | **23 GB** | `00b_download_patterning_screen.py` downloads it; extracted to `OSMGT_processed_files/` | **Delete** — already extracted; can re-download from Zenodo 17225179 |
| `patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz` | **4.7 GB** | `00c_build_temporal_atlas.py` — input for building temporal atlas for CellRank 2 | **Keep** (only patterning screen file actively referenced by pipeline) |
| `patterning_screen/OSMGT_processed_files/OSMGT.rds.gz` | **12 GB** | **Not used** — the full merged Seurat object. Referenced only in `convert_rds_to_h5ad.py` example usage. | **Defer** — could provide additional temporal data if converted; very large |
| `patterning_screen/OSMGT_processed_files/4_M_vs_sM_21d_clean.rds.gz` | **3.1 GB** | **Not used** — day 21 mistr vs sM comparison. Referenced in `convert_rds_to_h5ad.py` example. | **Defer** — day 21 data could enrich temporal coverage |
| `patterning_screen/OSMGT_processed_files/d9_mistr_cleaned.rds.gz` | **2.5 GB** | **Not used** — day 9 mistr dataset. | **Defer** — early timepoint data; needs RDS conversion |

**Patterning screen status**: Only `exp1_processed_8.h5ad.gz` is consumed by the pipeline (Step 00c). The temporal atlas output (`azbukina_temporal_atlas.h5ad`) has not yet been generated (Step 00c has not been run). The 3 RDS files would need R-based conversion before integration.

### Pipeline-Generated CSVs

| File | Size | Rows | Used By | Recommendation |
|------|------|------|---------|----------------|
| `gp_training_labels_amin_kelley.csv` | 8.9 KB | ~46 | Step 04 (GP training) | **Keep** |
| `gp_training_regions_amin_kelley.csv` | 9.2 KB | ~46 | Step 04 | **Keep** |
| `morphogen_matrix_amin_kelley.csv` | 8.1 KB | ~46 | Step 04 | **Keep** |
| `morphogen_matrix_sag_screen.csv` | 563 B | 2 | Step 04 | **Keep** |
| `fidelity_report.csv` | 4.2 KB | 46 | Step 03 output (condition-level fidelity scores) | **Keep** |
| `braun_reference_profiles.csv` | 2.6 KB | 16 regions | Step 03 cache (Braun fetal brain region-level composition, 12 cell types) | **Keep** |
| `braun_reference_celltype_profiles.csv` | 7.2 KB | 16 regions | Step 03 cache (Braun region x 68 fine cell types) | **Keep** |
| `hnoca_region_profiles_level3.csv` | 1.9 KB | 9 regions | Step 03 cache (HNOCA level-3 annotation profiles, 29 cell types) | **Keep** |
| `gp_training_labels_demo.csv` | 7.9 KB | 46 | Step 04 fallback demo data (used when real training labels missing; has generic `celltype_0..7` columns) | **Keep** (fallback) |
| `gp_diagnostics_round1.csv` | 305 B | 1 | Step 04 output | **Keep** |
| `gp_recommendations_round1.csv` | 1.4 KB | ~24 | Step 04 output (next experiments) | **Keep** |
| `cellflow_virtual_fractions.csv` | 1.5 MB | 5000 | Step 06 output (full virtual screen) | **Keep** |
| `cellflow_virtual_morphogens.csv` | 737 KB | 5000 | Step 06 output | **Keep** |
| `cellflow_virtual_fractions_200.csv` | 56 KB | 200 | Step 04 input (top-200 subset fed to GP) | **Keep** |
| `cellflow_virtual_morphogens_200.csv` | 28 KB | 200 | Step 04 input | **Keep** |
| `cellflow_screening_report.csv` | 168 KB | 5000 | Step 06 output (quality metrics per virtual prediction) | **Keep** |
| `report_round1.html` | 1.1 MB | - | Step 05 visualization output | **Keep** |

### Neural Organoid Atlas Repo (562 MB)

Cloned from theislab. Only the scPoli model is used.

| Component | Size | Used By | Recommendation |
|-----------|------|---------|----------------|
| `supplemental_files/scpoli_model_params/` (3 files) | ~81 MB | Step 02 (scPoli architecture surgery) | **Keep** (essential) |
| `supplemental_files/HumanFetalBrainPool_cluster_expr_highvar.tsv` | 39 MB | **Not used** — 3927 genes x N clusters expression matrix from fetal brain. | **Integrate?** — could serve as alternative reference for fidelity scoring or marker gene analysis |
| `supplemental_files/organoids_scpoli_herarchical123_sample_embedding.h5ad` | 163 KB | **Not used** — 396 sample-level embeddings (5D). No obs metadata. | **Defer** — sample-level comparison tool |
| `supplemental_files/Data_S1_snapseed_markers.yaml` | 4.8 KB | **Not used** | **Defer** — annotation markers, could inform QC |
| `supplemental_files/abstract.jpg` | 139 KB | Not used | **Delete** |
| `Fig1_HNOCA_establishment/` | ~57 MB | **Not used** — HNOCA construction notebooks (scPoli, pseudotime, integration) | **Delete** (reference only; no pipeline dependency) |
| `Fig2_map_to_primary/` | ~14 MB | **Not used** — Braun mapping notebooks | **Delete** |
| `Fig3_DE_to_primary/` | ~43 MB | **Not used** — DE analysis notebooks | **Delete** |
| `Fig4_Amin_mapping/` | ~65 KB | **Not used** — Amin/Kelley mapping scripts (NOMS). Contains `wknn.py` with weighted KNN implementation. | **Defer** — `wknn.py` could inform KNN label transfer improvements |
| `Fig5_disease_atlas/` | ~22 MB | **Not used** — disease atlas construction notebook | **Delete** (unless disease_atlas.h5ad is integrated) |
| `Fig6_HNOCA-extended/` | ~37 MB | **Not used** — HNOCA-extended model and dataloading notebooks. Contains `hnoca_extended_scpoli_model.tar.gz` (29 MB). | **Defer** — extended scPoli model could enable mapping of additional protocol datasets |
| `supplemental_files/08_export_h5ads.ipynb` | 57 KB | Not used | **Delete** |

---

## Deletion Candidates (Reclaim ~27 GB)

| File | Size | Reason |
|------|------|--------|
| `patterning_screen/OSMGT_processed_files.tar.gz` | 23 GB | Already extracted; re-downloadable from Zenodo |
| `GSE233574_OrganoidScreen_processed_SeuratObject.rds` | 1.7 GB | Redundant with MTX+CSV raw files |
| `GSE233574_Organoid.SAG.secondaryScreen_processed_SeuratObject.rds` | 304 MB | Redundant with MTX+CSV raw files |
| `disease_atlas.h5ad` | 2.2 GB | Not consumed by any pipeline step |
| `neural_organoid_atlas/Fig{1,2,3,5}/` + misc notebooks | ~136 MB | Reference notebooks, not needed by pipeline |

## Integration Candidates

| File | Size | Potential Use |
|------|------|---------------|
| `patterning_screen/OSMGT.rds.gz` | 12 GB | Full patterning screen Seurat object — richer temporal coverage for CellRank 2 if converted |
| `patterning_screen/4_M_vs_sM_21d_clean.rds.gz` | 3.1 GB | Day 21 comparison data — fills early-timepoint gap |
| `patterning_screen/d9_mistr_cleaned.rds.gz` | 2.5 GB | Day 9 data — earliest available timepoint |
| `supplemental_files/HumanFetalBrainPool_cluster_expr_highvar.tsv` | 39 MB | Cluster-level expression profiles for alternative fidelity metric |
| `Fig6 hnoca_extended_scpoli_model.tar.gz` | 29 MB | Extended scPoli model for mapping non-Amin datasets |
| `Fig4 wknn.py` | 4 KB | Weighted KNN implementation from original HNOCA authors |

## Missing Files (Expected but Not Present)

| File | Expected From | Status |
|------|--------------|--------|
| `azbukina_temporal_atlas.h5ad` | Step 00c output | Step 00c has not been run yet |
| `gp_training_labels_sag_screen.csv` | Step 02 (SAG screen mapping) | SAG screen not yet mapped |
| `sag_screen_mapped.h5ad` | Step 02 (SAG screen mapping) | SAG screen not yet mapped |
| `GSE233574_hCbO_processed_SeuratObject.rds` | GEO (optional) | Not downloaded — cerebellar organoid validation |
| `GSE233574_hMPO_processed_SeuratObject.rds` | GEO (optional) | Not downloaded — medial pallium organoid validation |
