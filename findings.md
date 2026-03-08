# Findings

## Data Inventory (Initial)
- All GEO files present (NO .gz extensions — bug in code)
- scPoli model at `data/neural_organoid_atlas/supplemental_files/scpoli_model_params/`
- Patterning screen tarball at `data/patterning_screen/OSMGT_processed_files.tar.gz` (not yet extracted)
- HNOCA reference: 2.9 GB, Braun fetal brain: 11 GB, Disease atlas: 2.2 GB

## Amin/Kelley Metadata (Primary Screen)
- Columns: sample, bc_index, species, tscp_count, tscp_count_50dup, gene_count, quality, condition
- 46 unique conditions (see below)
- Quality column has "keep" and "low-quality" values

## Condition Names (encode morphogens)
BMP4 CHIR, BMP4 CHIR d11-16, BMP4 SAG, BMP7, BMP7 CHIR, BMP7 SAG,
C/L/S/FGF8, C/S/BMP7/D, C/S/D/FGF4, C/S/R/E/FGF2/D,
CHIR SAG FGF4, CHIR SAG FGF8, CHIR switch IWP2, CHIR-d11-16,
CHIR-d16-21, CHIR-d6-11, CHIR-SAG-d16-21, CHIR-SAG-LDN,
CHIR-SAGd10-21, CHIR1.5-SAG1000, CHIR1.5-SAG250, CHIR1.5,
CHIR3-SAG1000, CHIR3-SAG250, CHIR3, DAPT, FGF-20/EGF, FGF2-20,
FGF2-50, FGF4, FGF8, I/Activin/DAPT/SR11, IWP2, IWP2 switch CHIR,
IWP2-SAG, LDN, RA10, RA100, S/I/E/FGF2, SAG-CHIR-d16-21,
SAG-CHIRd10-21, SAG-d11-16, SAG-d16-21, SAG-d6-11, SAG1000, SAG250

## SAG Secondary Screen Metadata
- Has extra columns: ClusterLabel, seurat_clusters, Order
- Conditions include SAG concentration variants

## Installed Packages Status
- scanpy 1.12, anndata 0.12.10, torch 2.10.0, botorch 0.17.2 ✓
- scvi-tools 1.4.2 ✓, gpytorch 1.15.2 ✓, hnoca ✓
- pytest ✓, hypothesis ✓
- scarches 0.6.1 BROKEN (anndata.read removed in 0.12.10)

## Morphogen Concentrations (from docs/MORPHOGEN_SCREEN_DATASETS_ML_READINESS.md)
Key concentrations for Amin/Kelley:
- SAG: 50, 250, 1000, 2000 nM
- CHIR99021: 1.5, 3 uM
- IWP2: standard
- RA: 10, 100 nM
- FGF2: 20, 50 ng/mL
- FGF8: 100 ng/mL
- BMP4: standard
- BMP7: standard
- LDN-193189: standard
- DAPT: 2.5 uM
- Dorsomorphin: 2.5 uM
- Activin A: 50 ng/mL
- EGF: standard
