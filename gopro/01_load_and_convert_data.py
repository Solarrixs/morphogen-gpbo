"""
Step 1: Load all datasets and convert to unified AnnData format.

Inputs:
  - GSE233574 Amin/Kelley data (Seurat RDS → AnnData via MTX + CSV)
  - HNOCA minimal reference (already h5ad)
  - Braun fetal brain reference (already h5ad)
  - scPoli trained model (already .pt)

Output:
  - data/amin_kelley_2024.h5ad (converted from Seurat)
  - data/amin_kelley_sag_screen.h5ad (SAG secondary screen)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.io import mmread
from pathlib import Path

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

DATA_DIR.mkdir(exist_ok=True)


def convert_geo_to_anndata(counts_path, metadata_path, genes_path, output_path, name):
    """Convert GEO MTX + CSV files to AnnData h5ad."""
    logger.info("--- Converting %s ---", name)

    # Load count matrix (genes x cells in MTX format, need to transpose)
    logger.info("Loading counts from %s...", counts_path.name)
    counts = mmread(str(counts_path)).T.tocsr()
    logger.info("Matrix shape: %s (cells x genes)", counts.shape)

    # Load cell metadata
    logger.info("Loading metadata from %s...", metadata_path.name)
    metadata = pd.read_csv(str(metadata_path), index_col=0)
    logger.info("Metadata columns: %s", list(metadata.columns))
    logger.info("Cells in metadata: %d", len(metadata))

    # Load gene info
    logger.info("Loading gene info from %s...", genes_path.name)
    genes = pd.read_csv(str(genes_path), index_col=0)
    logger.info("Genes: %d", len(genes))

    # Verify dimensions match
    assert counts.shape[0] == len(metadata), \
        f"Cell count mismatch: matrix has {counts.shape[0]}, metadata has {len(metadata)}"
    assert counts.shape[1] == len(genes), \
        f"Gene count mismatch: matrix has {counts.shape[1]}, gene info has {len(genes)}"

    # Create AnnData
    adata = sc.AnnData(
        X=counts,
        obs=metadata,
        var=genes,
    )

    # Store raw counts in layers
    adata.layers["counts"] = adata.X.copy()

    logger.info("AnnData created: %s", adata.shape)
    logger.info("Saving to %s...", output_path)
    adata.write(str(output_path), compression="gzip")
    logger.info("Done. File size: %.2f GB", output_path.stat().st_size / 1e9)

    return adata


def verify_references():
    """Check that reference files exist."""
    logger.info("--- Verifying reference files ---")

    files = {
        "HNOCA minimal": DATA_DIR / "hnoca_minimal_for_mapping.h5ad",
        "Braun fetal brain": DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad",
        "scPoli model_params.pt": DATA_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params/model_params.pt",
        "scPoli attr.pkl": DATA_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params/attr.pkl",
        "scPoli var_names.csv": DATA_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params/var_names.csv",
        "Disease atlas": DATA_DIR / "disease_atlas.h5ad",
    }

    all_ok = True
    for name, path in files.items():
        exists = path.exists()
        size = f"({path.stat().st_size / 1e9:.2f} GB)" if exists else ""
        status = "OK" if exists else "MISSING"
        logger.info("  [%s] %s: %s %s", status, name, path.name, size)
        if not exists:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    # 1. Verify references exist
    refs_ok = verify_references()
    if not refs_ok:
        logger.warning("Some reference files are missing!")
        logger.warning("Run: bash download_zenodo.sh")
        logger.warning("to download the HNOCA and Braun fetal brain references.")

    # 2. Convert Amin/Kelley primary screen (46 conditions, Day 72-74)
    convert_geo_to_anndata(
        counts_path=DATA_DIR / "GSE233574_OrganoidScreen_counts.mtx",
        metadata_path=DATA_DIR / "GSE233574_OrganoidScreen_cellMetaData.csv",
        genes_path=DATA_DIR / "GSE233574_OrganoidScreen_geneInfo.csv",
        output_path=DATA_DIR / "amin_kelley_2024.h5ad",
        name="Amin/Kelley 2024 Primary Morphogen Screen (GSE233574)",
    )

    # 3. Convert Amin/Kelley SAG secondary screen
    convert_geo_to_anndata(
        counts_path=DATA_DIR / "GSE233574_Organoid.SAG.secondaryScreen_counts.mtx",
        metadata_path=DATA_DIR / "GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv",
        genes_path=DATA_DIR / "GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv",
        output_path=DATA_DIR / "amin_kelley_sag_screen.h5ad",
        name="Amin/Kelley 2024 SAG Secondary Screen",
    )

    # 4. Convert hCbO cerebellar organoid validation (1 protocol, high-fidelity anchor)
    hcbo_rds = DATA_DIR / "GSE233574_hCbO_processed_SeuratObject.rds"
    if hcbo_rds.exists():
        logger.info("hCbO Seurat object found (%.2f GB)", hcbo_rds.stat().st_size / 1e9)
        logger.info("This is an RDS file — needs R/sceasy conversion.")
        logger.info('  sceasy::convertFormat(readRDS("GSE233574_hCbO_processed_SeuratObject.rds.gz"),')
        logger.info('    from="seurat", to="anndata", outFile="data/amin_kelley_hcbo.h5ad")')
    else:
        logger.info("hCbO Seurat object not found (optional)")

    # 5. Convert hMPO medial pallium organoid validation (1 protocol, high-fidelity anchor)
    hmpo_rds = DATA_DIR / "GSE233574_hMPO_processed_SeuratObject.rds"
    if hmpo_rds.exists():
        logger.info("hMPO Seurat object found (%.2f GB)", hmpo_rds.stat().st_size / 1e9)
        logger.info("This is an RDS file — needs R/sceasy conversion.")
        logger.info('  sceasy::convertFormat(readRDS("GSE233574_hMPO_processed_SeuratObject.rds.gz"),')
        logger.info('    from="seurat", to="anndata", outFile="data/amin_kelley_hmpo.h5ad")')
    else:
        logger.info("hMPO Seurat object not found (optional)")

    logger.info("--- SUMMARY ---")
    logger.info("Converted datasets saved to: data/")
    for f in sorted(DATA_DIR.glob("*.h5ad")):
        logger.info("  %s: %.2f GB", f.name, f.stat().st_size / 1e9)
