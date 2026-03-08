"""
Step 2: Map morphogen screen data onto HNOCA reference via scArches/scPoli.

Based on the reference implementation in:
  data/neural_organoid_atlas/Fig4_Amin_mapping/03_NOMS_to_HNOCA_mapping.ipynb

This script:
  1. Loads the pre-trained scPoli model + HNOCA minimal reference
  2. Prepares query data (subset to ref HVGs, set X to counts, add batch column)
  3. Projects query cells onto HNOCA latent space (architecture surgery)
  4. Transfers cell type labels via weighted KNN in latent space
  5. Computes cell type fractions per morphogen condition (= GP training labels)

Inputs:
  - data/amin_kelley_2024.h5ad (from step 01)
  - data/hnoca_minimal_for_mapping.h5ad (HNOCA reference with scPoli latent)
  - data/neural_organoid_atlas/supplemental_files/scpoli_model_params/

Outputs:
  - data/amin_kelley_mapped.h5ad (with cell type annotations in obs)
  - data/gp_training_labels_amin_kelley.csv (cell type fractions per condition)
"""

from __future__ import annotations

import warnings
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = DATA_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params"

# Annotation columns in the HNOCA reference
ANNOT_LEVEL_1 = "annot_level_1"       # 13 broad types: Neuron, NPC, etc.
ANNOT_LEVEL_2 = "annot_level_2"       # 17 types: Dorsal Telencephalic Neuron, etc.
ANNOT_REGION = "annot_region_rev2"    # 10 brain regions
ANNOT_LEVEL_3 = "annot_level_3_rev2"  # 29 detailed types


def filter_quality_cells(adata: sc.AnnData) -> sc.AnnData:
    """Filter to 'keep' quality cells based on Amin/Kelley QC annotations.

    Args:
        adata: AnnData with 'quality' column in obs.

    Returns:
        Filtered AnnData with only 'keep' quality cells.
    """
    if "quality" in adata.obs.columns:
        n_before = adata.n_obs
        adata = adata[adata.obs["quality"] == "keep"].copy()
        print(f"  Quality filter: {n_before} → {adata.n_obs} cells "
              f"({n_before - adata.n_obs} removed)")
    return adata


def prepare_query_for_scpoli(
    query: sc.AnnData,
    ref: sc.AnnData,
    batch_column: str = "sample",
) -> sc.AnnData:
    """Prepare query AnnData for scPoli mapping.

    Following the reference implementation:
    1. Subset to shared genes (intersection with reference HVGs)
    2. Set X to raw counts
    3. Add batch column
    4. Add placeholder annotation labels

    Args:
        query: Query AnnData (from step 01).
        ref: Reference AnnData (HNOCA minimal).
        batch_column: Column in query.obs to use as batch identifier.

    Returns:
        Prepared query AnnData ready for scPoli.
    """
    print("  Preparing query for scPoli mapping...")

    # Map query var_names to gene symbols (ref uses gene symbols, query uses Ensembl IDs)
    if "gene_name_unique" in query.var.columns:
        print("  Query uses Ensembl IDs — mapping to gene symbols...")
        query.var_names = query.var["gene_name_unique"].values
        query.var_names_make_unique()
    elif "gene_symbol" in query.var.columns:
        query.var_names = query.var["gene_symbol"].values
        query.var_names_make_unique()

    # Align query to reference var_names (scPoli expects exact same genes)
    shared_genes = query.var_names.intersection(ref.var_names)
    print(f"  Shared genes with reference: {len(shared_genes)} / {ref.n_vars}")

    # Efficient reindexing: build a permutation matrix to map query genes → ref genes
    # Start with query counts
    if "counts" in query.layers:
        if sparse.issparse(query.layers["counts"]):
            X_counts = query.layers["counts"].copy()
        else:
            X_counts = sparse.csr_matrix(query.layers["counts"])
    else:
        X_counts = query.X.copy() if sparse.issparse(query.X) else sparse.csr_matrix(query.X)

    # Build mapping: for each ref gene, find its column index in query (or -1)
    query_gene_to_idx = {g: i for i, g in enumerate(query.var_names)}
    ref_gene_indices = []  # indices into query columns
    ref_gene_mask = []     # which ref genes are found in query
    for g in ref.var_names:
        if g in query_gene_to_idx:
            ref_gene_indices.append(query_gene_to_idx[g])
            ref_gene_mask.append(True)
        else:
            ref_gene_indices.append(0)  # placeholder
            ref_gene_mask.append(False)

    # Build reindexed matrix efficiently
    X_reindexed = X_counts[:, ref_gene_indices].copy()
    # Zero out columns for genes not found in query
    not_found = np.where(~np.array(ref_gene_mask))[0]
    if len(not_found) > 0:
        X_reindexed = X_reindexed.tolil()
        X_reindexed[:, not_found] = 0
        X_reindexed = X_reindexed.tocsr()

    # Create new AnnData with ref var_names
    import anndata
    query_aligned = anndata.AnnData(
        X=X_reindexed,
        obs=query.obs.copy(),
    )
    query_aligned.var_names = ref.var_names.copy()
    # Copy ref var metadata
    for col in ref.var.columns:
        query_aligned.var[col] = ref.var[col].values

    print(f"  Reindexed query: {query_aligned.shape} ({len(shared_genes)} genes filled, "
          f"{len(not_found)} zero-filled)")

    query = query_aligned

    # Set batch column
    if batch_column in query.obs.columns:
        query.obs["batch"] = query.obs[batch_column].astype(str)
    else:
        query.obs["batch"] = "query"
    print(f"  Batch column: {query.obs['batch'].nunique()} unique batches")

    # scPoli model expects these specific column names (from original HNOCA)
    # Map: annot_level_1 → snapseed_pca_rss_level_1, etc.
    SCPOLI_LABEL_COLS = [
        "snapseed_pca_rss_level_1",
        "snapseed_pca_rss_level_12",
        "snapseed_pca_rss_level_123",
    ]
    for col in SCPOLI_LABEL_COLS:
        query.obs[col] = "unknown"

    # Clear obsm/varm that might cause issues
    query.obsm = {}
    query.varm = {}

    return query


def map_to_hnoca_scpoli(
    query: sc.AnnData,
    ref: sc.AnnData,
    model_dir: Path,
    n_epochs: int = 500,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Map query cells onto HNOCA latent space using scPoli architecture surgery.

    Following the reference implementation in:
      Fig4_Amin_mapping/03_NOMS_to_HNOCA_mapping.ipynb

    Args:
        query: Prepared query AnnData.
        ref: Reference AnnData (HNOCA minimal).
        model_dir: Path to scPoli model parameters.
        n_epochs: Number of training epochs for query mapping.
        batch_size: Training batch size.

    Returns:
        Tuple of (query_latent, ref_latent) numpy arrays.
    """
    from scarches.models.scpoli import scPoli

    # Prepare reference: set X to counts
    print("  Setting reference X to counts layer...")
    if sparse.issparse(ref.layers["counts"]):
        ref.X = ref.layers["counts"].toarray().copy()
    else:
        ref.X = ref.layers["counts"].copy()

    # scPoli model expects old HNOCA column names
    # Map: annot_level_1 → snapseed_pca_rss_level_1, etc.
    col_map = {
        "annot_level_1": "snapseed_pca_rss_level_1",
        "annot_level_2": "snapseed_pca_rss_level_12",
        "annot_level_3_rev2": "snapseed_pca_rss_level_123",
    }
    for old_col, new_col in col_map.items():
        if old_col in ref.obs.columns and new_col not in ref.obs.columns:
            ref.obs[new_col] = ref.obs[old_col].values

    # Load pre-trained scPoli model
    print(f"  Loading pre-trained scPoli model from {model_dir}...")
    scpoli_model = scPoli.load(str(model_dir), ref, map_location="cpu")
    print("  Model loaded successfully.")

    # Load query data into the model (architecture surgery)
    print("  Preparing query data for scPoli...")
    # Ensure query X is dense
    if sparse.issparse(query.X):
        query.X = query.X.toarray().copy()

    scpoli_query = scPoli.load_query_data(
        adata=query,
        reference_model=scpoli_model,
        labeled_indices=[],
    )

    # Train on query data (partial retrain — only new batch embedding)
    print(f"  Training scPoli on query data ({n_epochs} epochs)...")
    scpoli_query.train(
        n_epochs=n_epochs,
        pretraining_epochs=n_epochs,
        eta=10,
        unlabeled_prototype_training=False,
    )
    print("  Training complete.")

    # Get latent representations
    print("  Computing latent representations...")
    query_latent = scpoli_query.get_latent(query, mean=True)
    ref_latent = scpoli_model.get_latent(ref, mean=True)

    print(f"  Query latent: {query_latent.shape}")
    print(f"  Reference latent: {ref_latent.shape}")

    return query_latent, ref_latent


def transfer_labels_knn(
    ref_latent: np.ndarray,
    query_latent: np.ndarray,
    ref_obs: pd.DataFrame,
    query_obs: pd.DataFrame,
    label_columns: list[str],
    k: int = 50,
) -> pd.DataFrame:
    """Transfer cell type labels from reference to query via KNN in latent space.

    Uses sklearn KNeighborsClassifier (CPU) instead of cuml (GPU).

    Args:
        ref_latent: Reference latent space embeddings.
        query_latent: Query latent space embeddings.
        ref_obs: Reference cell metadata.
        query_obs: Query cell metadata.
        label_columns: List of label columns to transfer.
        k: Number of neighbors.

    Returns:
        DataFrame with transferred labels and confidence scores.
    """
    from sklearn.neighbors import KNeighborsClassifier

    results = pd.DataFrame(index=query_obs.index)

    for label_col in label_columns:
        if label_col not in ref_obs.columns:
            print(f"  WARNING: {label_col} not in reference, skipping")
            continue

        labels = ref_obs[label_col].values
        print(f"  Transferring {label_col} ({len(np.unique(labels))} classes)...")

        knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(ref_latent, labels)

        predicted = knn.predict(query_latent)
        probas = knn.predict_proba(query_latent)
        confidence = probas.max(axis=1)

        results[f"predicted_{label_col}"] = predicted
        results[f"{label_col}_confidence"] = confidence

    return results


def compute_cell_type_fractions(
    obs: pd.DataFrame,
    condition_key: str = "condition",
    label_key: str = "predicted_annot_level_2",
    quality_filter: bool = True,
) -> pd.DataFrame:
    """Compute cell type fractions per condition (= GP training labels Y).

    Args:
        obs: Cell metadata with condition and predicted cell type columns.
        condition_key: Column identifying experimental conditions.
        label_key: Column with predicted cell type labels.
        quality_filter: Whether to filter to 'keep' quality cells.

    Returns:
        DataFrame: rows=conditions, columns=cell types, values=fractions (sum to 1).
    """
    print(f"  Computing cell type fractions per {condition_key}...")

    df = obs.copy()
    if quality_filter and "quality" in df.columns:
        df = df[df["quality"] == "keep"]

    fractions = (
        df.groupby(condition_key)[label_key]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print(f"  Result: {fractions.shape[0]} conditions × {fractions.shape[1]} cell types")

    # Verify fractions sum to 1
    row_sums = fractions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        f"Fractions don't sum to 1: {row_sums[~np.isclose(row_sums, 1.0)]}"

    return fractions


if __name__ == "__main__":
    import time
    start = time.time()

    # Check prerequisites
    ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"
    query_path = DATA_DIR / "amin_kelley_2024.h5ad"

    for path, name in [(ref_path, "HNOCA reference"), (query_path, "Amin/Kelley data")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            exit(1)

    if not MODEL_DIR.exists():
        print(f"ERROR: scPoli model not found at {MODEL_DIR}")
        exit(1)

    # Load data
    print("Loading HNOCA minimal reference...")
    ref = sc.read_h5ad(str(ref_path))
    print(f"  Reference: {ref.shape}")

    print("Loading Amin/Kelley data...")
    query = sc.read_h5ad(str(query_path))
    print(f"  Query: {query.shape}")

    # Filter to quality cells
    query = filter_quality_cells(query)

    # Prepare query
    query = prepare_query_for_scpoli(query, ref, batch_column="sample")

    # Map to HNOCA via scPoli
    query_latent, ref_latent = map_to_hnoca_scpoli(
        query, ref, MODEL_DIR,
        n_epochs=500,
        batch_size=1024,
    )

    # Store latent in query
    query.obsm["X_scpoli"] = query_latent

    # Transfer labels
    print("\nTransferring cell type labels...")
    label_cols = [ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3]
    transferred = transfer_labels_knn(
        ref_latent, query_latent,
        ref.obs, query.obs,
        label_columns=label_cols,
        k=50,
    )

    # Add transferred labels to query
    for col in transferred.columns:
        query.obs[col] = transferred[col].values

    # Compute cell type fractions for GP training
    fractions = compute_cell_type_fractions(
        query.obs,
        condition_key="condition",
        label_key=f"predicted_{ANNOT_LEVEL_2}",
    )

    # Also compute region fractions
    region_fractions = compute_cell_type_fractions(
        query.obs,
        condition_key="condition",
        label_key=f"predicted_{ANNOT_REGION}",
    )

    # Save outputs
    print("\nSaving outputs...")

    fractions.to_csv(str(DATA_DIR / "gp_training_labels_amin_kelley.csv"))
    print(f"  Cell type fractions → data/gp_training_labels_amin_kelley.csv")

    region_fractions.to_csv(str(DATA_DIR / "gp_training_regions_amin_kelley.csv"))
    print(f"  Region fractions → data/gp_training_regions_amin_kelley.csv")

    output_path = DATA_DIR / "amin_kelley_mapped.h5ad"
    query.write(str(output_path), compression="gzip")
    print(f"  Mapped data → {output_path}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")

    # Summary
    print("\n" + "=" * 60)
    print("MAPPING SUMMARY")
    print("=" * 60)
    print(f"  Cells mapped: {query.n_obs}")
    print(f"  Conditions: {query.obs['condition'].nunique()}")
    for label_col in label_cols:
        pred_col = f"predicted_{label_col}"
        if pred_col in query.obs.columns:
            print(f"  {label_col}: {query.obs[pred_col].nunique()} types")
            top3 = query.obs[pred_col].value_counts().head(3)
            for t, c in top3.items():
                print(f"    {t}: {c} cells ({c/query.n_obs*100:.1f}%)")
