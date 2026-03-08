"""
Step 2: Map morphogen screen data onto HNOCA reference via scArches.

This script:
  1. Loads the pre-trained scPoli model + HNOCA minimal reference
  2. Preprocesses query data (QC, normalization)
  3. Projects query cells onto HNOCA latent space (architecture surgery)
  4. Transfers cell type labels from HNOCA → query
  5. Computes cell type fractions per morphogen condition (= GP training labels)

Inputs:
  - data/amin_kelley_2024.h5ad (from step 01)
  - hnoca_minimal_for_mapping.h5ad (from Zenodo 15004817)
  - scpoli_model_params/ (from GitHub)

Outputs:
  - data/amin_kelley_mapped.h5ad (with cell type annotations)
  - data/gp_training_data.csv (X=morphogens, Y=cell type fractions)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params"


def preprocess_query(adata):
    """Standard scRNA-seq preprocessing for query data."""
    print("  Preprocessing query data...")

    # QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Filter cells
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=8000)
    adata = adata[adata.obs.pct_counts_mt < 20].copy()
    print(f"  Filtered: {n_before} → {adata.n_obs} cells ({n_before - adata.n_obs} removed)")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=None)

    # PCA + neighbors
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=30)

    return adata


def map_to_hnoca(query_adata, ref_adata_path, model_dir):
    """
    Map query data onto HNOCA using scArches/scPoli.

    This performs "architecture surgery":
    - Loads the pre-trained scPoli model (trained on 1.77M HNOCA cells)
    - Freezes most weights (preserves cell biology knowledge)
    - Adds a small new layer for the query batch
    - Trains only that layer (~100 epochs, minutes)
    - Result: query cells in the same latent space as HNOCA
    """
    try:
        import scvi
        from hnoca.map import AtlasMapper
    except ImportError:
        print("\n  ERROR: Required packages not installed. Run:")
        print("    pip install hnoca scvi-tools scarches")
        return None, None

    # Load reference
    print("  Loading HNOCA minimal reference...")
    ref_adata = sc.read_h5ad(str(ref_adata_path))
    print(f"  Reference: {ref_adata.shape}")

    # Load pre-trained model
    print(f"  Loading pre-trained scPoli model from {model_dir}...")
    ref_model = scvi.model.SCANVI.load(str(model_dir), adata=ref_adata)

    # Create mapper
    mapper = AtlasMapper(ref_model)

    # Architecture surgery: map query onto reference
    print("  Running architecture surgery (partial retrain)...")
    mapper.map_query(
        query_adata,
        retrain="partial",
        max_epochs=100,
        batch_size=1024,
    )

    # Transfer labels
    print("  Computing weighted KNN...")
    mapper.compute_wknn(k=100)

    print("  Transferring cell type labels...")
    celltype_labels = mapper.transfer_labels(label_key="cell_type")

    # Compute presence scores (how well represented each cell type is)
    presence_scores = mapper.get_presence_scores(split_by="batch")

    return celltype_labels, presence_scores


def compute_gp_training_data(adata, condition_key="condition"):
    """
    Compute cell type fractions per condition = GP training labels (Y).

    Returns DataFrame: rows=conditions, columns=cell types, values=fractions
    """
    print("  Computing cell type fractions per condition...")
    fractions = (
        adata.obs
        .groupby(condition_key)["predicted_cell_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print(f"  Result: {fractions.shape[0]} conditions × {fractions.shape[1]} cell types")
    return fractions


if __name__ == "__main__":
    # Check prerequisites
    ref_path = PROJECT_DIR / "hnoca_minimal_for_mapping.h5ad"
    if not ref_path.exists():
        print("ERROR: hnoca_minimal_for_mapping.h5ad not found!")
        print("Run: bash download_zenodo.sh")
        exit(1)

    # Load converted Amin/Kelley data
    print("Loading Amin/Kelley data...")
    query = sc.read_h5ad(str(DATA_DIR / "amin_kelley_2024.h5ad"))
    print(f"  Loaded: {query.shape}")

    # Print condition info
    if "condition" in query.obs.columns:
        cond_key = "condition"
    else:
        # Try to find the condition column
        print(f"  Available metadata columns: {list(query.obs.columns)}")
        print("  You may need to set the condition_key manually.")
        cond_key = None

    # Preprocess
    query = preprocess_query(query)

    # Map to HNOCA
    labels, scores = map_to_hnoca(query, ref_path, MODEL_DIR)

    if labels is not None:
        query.obs["predicted_cell_type"] = labels

        # Compute GP training labels
        if cond_key:
            fractions = compute_gp_training_data(query, condition_key=cond_key)
            fractions.to_csv(str(DATA_DIR / "gp_training_labels_amin_kelley.csv"))
            print(f"\n  Saved GP training labels to data/gp_training_labels_amin_kelley.csv")

        # Save annotated data
        output_path = DATA_DIR / "amin_kelley_mapped.h5ad"
        query.write(str(output_path), compression="gzip")
        print(f"  Saved mapped data to {output_path}")
    else:
        print("\n  Mapping skipped — install hnoca + scvi-tools first.")
        print("  The data conversion (step 01) still works without these.")
