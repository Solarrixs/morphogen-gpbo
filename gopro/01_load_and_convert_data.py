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

# Paths
PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def convert_geo_to_anndata(counts_path, metadata_path, genes_path, output_path, name):
    """Convert GEO MTX + CSV files to AnnData h5ad."""
    print(f"\n{'='*60}")
    print(f"Converting {name}")
    print(f"{'='*60}")

    # Load count matrix (genes x cells in MTX format, need to transpose)
    print(f"  Loading counts from {counts_path.name}...")
    counts = mmread(str(counts_path)).T.tocsr()
    print(f"  Matrix shape: {counts.shape} (cells x genes)")

    # Load cell metadata
    print(f"  Loading metadata from {metadata_path.name}...")
    metadata = pd.read_csv(str(metadata_path), index_col=0)
    print(f"  Metadata columns: {list(metadata.columns)}")
    print(f"  Cells in metadata: {len(metadata)}")

    # Load gene info
    print(f"  Loading gene info from {genes_path.name}...")
    genes = pd.read_csv(str(genes_path), index_col=0)
    print(f"  Genes: {len(genes)}")

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

    print(f"  AnnData created: {adata.shape}")
    print(f"  Saving to {output_path}...")
    adata.write(str(output_path), compression="gzip")
    print(f"  Done. File size: {output_path.stat().st_size / 1e9:.2f} GB")

    return adata


def verify_references():
    """Check that reference files exist."""
    print("\n" + "="*60)
    print("Verifying reference files")
    print("="*60)

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
        print(f"  [{status}] {name}: {path.name} {size}")
        if not exists:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    # 1. Verify references exist
    refs_ok = verify_references()
    if not refs_ok:
        print("\n  WARNING: Some reference files are missing!")
        print("  Run: bash download_zenodo.sh")
        print("  to download the HNOCA and Braun fetal brain references.\n")

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
        print(f"\n  NOTE: hCbO Seurat object found ({hcbo_rds.stat().st_size / 1e9:.2f} GB)")
        print("  This is an RDS file — needs R/sceasy conversion.")
        print("  To convert in R:")
        print('    sceasy::convertFormat(readRDS("GSE233574_hCbO_processed_SeuratObject.rds.gz"),')
        print('      from="seurat", to="anndata", outFile="data/amin_kelley_hcbo.h5ad")')
    else:
        print("  hCbO Seurat object not found (optional)")

    # 5. Convert hMPO medial pallium organoid validation (1 protocol, high-fidelity anchor)
    hmpo_rds = DATA_DIR / "GSE233574_hMPO_processed_SeuratObject.rds"
    if hmpo_rds.exists():
        print(f"\n  NOTE: hMPO Seurat object found ({hmpo_rds.stat().st_size / 1e9:.2f} GB)")
        print("  This is an RDS file — needs R/sceasy conversion.")
        print("  To convert in R:")
        print('    sceasy::convertFormat(readRDS("GSE233574_hMPO_processed_SeuratObject.rds.gz"),')
        print('      from="seurat", to="anndata", outFile="data/amin_kelley_hmpo.h5ad")')
    else:
        print("  hMPO Seurat object not found (optional)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Converted datasets saved to: data/")
    for f in sorted(DATA_DIR.glob("*.h5ad")):
        print(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB")
