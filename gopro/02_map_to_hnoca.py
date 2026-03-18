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

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Optional

from gopro.config import (
    DATA_DIR, MODEL_DIR,
    ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3,
    get_logger,
)

logger = get_logger(__name__)


def filter_quality_cells(adata: sc.AnnData) -> sc.AnnData:
    """Filter to quality cells based on dataset-specific QC annotations.

    Handles two formats:
    - Primary screen: 'quality' column, keep rows where quality == 'keep'
    - SAG screen: 'ClusterLabel' column, drop rows where ClusterLabel == 'filtered'

    Args:
        adata: AnnData with QC annotation columns in obs.

    Returns:
        Filtered AnnData.
    """
    n_before = adata.n_obs
    if "quality" in adata.obs.columns:
        adata = adata[adata.obs["quality"] == "keep"].copy()
        logger.info("Quality filter: %d -> %d cells (%d removed)",
                    n_before, adata.n_obs, n_before - adata.n_obs)
    elif "ClusterLabel" in adata.obs.columns:
        adata = adata[adata.obs["ClusterLabel"] != "filtered"].copy()
        logger.info("ClusterLabel filter: %d -> %d cells (%d 'filtered' removed)",
                    n_before, adata.n_obs, n_before - adata.n_obs)
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
    logger.info("Preparing query for scPoli mapping...")

    # Map query var_names to gene symbols (ref uses gene symbols, query uses Ensembl IDs)
    if "gene_name_unique" in query.var.columns:
        logger.info("Query uses Ensembl IDs — mapping to gene symbols...")
        query.var_names = query.var["gene_name_unique"].values
        query.var_names_make_unique()
    elif "gene_symbol" in query.var.columns:
        query.var_names = query.var["gene_symbol"].values
        query.var_names_make_unique()

    # Align query to reference var_names (scPoli expects exact same genes)
    shared_genes = query.var_names.intersection(ref.var_names)
    logger.info("Shared genes with reference: %d / %d", len(shared_genes), ref.n_vars)

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

    logger.info("Reindexed query: %s (%d genes filled, %d zero-filled)",
                query_aligned.shape, len(shared_genes), len(not_found))

    query = query_aligned

    # Set batch column
    if batch_column in query.obs.columns:
        query.obs["batch"] = query.obs[batch_column].astype(str)
    else:
        query.obs["batch"] = "query"
    logger.info("Batch column: %d unique batches", query.obs['batch'].nunique())

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
    logger.info("Setting reference X to counts layer...")
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
    logger.info("Loading pre-trained scPoli model from %s...", model_dir)
    scpoli_model = scPoli.load(str(model_dir), ref, map_location="cpu")
    logger.info("Model loaded successfully.")

    # Load query data into the model (architecture surgery)
    logger.info("Preparing query data for scPoli...")
    # Ensure query X is dense
    if sparse.issparse(query.X):
        query.X = query.X.toarray().copy()

    scpoli_query = scPoli.load_query_data(
        adata=query,
        reference_model=scpoli_model,
        labeled_indices=[],
    )

    # Train on query data (partial retrain — only new batch embedding)
    logger.info("Training scPoli on query data (%d epochs)...", n_epochs)
    scpoli_query.train(
        n_epochs=n_epochs,
        pretraining_epochs=n_epochs,
        eta=10,
        unlabeled_prototype_training=False,
    )
    logger.info("Training complete.")

    # Get latent representations
    logger.info("Computing latent representations...")
    query_latent = scpoli_query.get_latent(query, mean=True)
    ref_latent = scpoli_model.get_latent(ref, mean=True)

    logger.info("Query latent: %s", query_latent.shape)
    logger.info("Reference latent: %s", ref_latent.shape)

    return query_latent, ref_latent


def transfer_labels_knn(
    ref_latent: np.ndarray,
    query_latent: np.ndarray,
    ref_obs: pd.DataFrame,
    query_obs: pd.DataFrame,
    label_columns: list[str],
    k: int = 50,
    class_balanced: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Transfer cell type labels from reference to query via KNN in latent space.

    Uses class-balanced KNN by default: each neighbor's vote is weighted by
    ``(1/distance) * 1/sqrt(class_frequency)`` to correct for class imbalance
    in the reference atlas (e.g. HNOCA is ~43% dorsal telencephalon).

    Args:
        ref_latent: Reference latent space embeddings.
        query_latent: Query latent space embeddings.
        ref_obs: Reference cell metadata.
        query_obs: Query cell metadata.
        label_columns: List of label columns to transfer.
        k: Number of neighbors.
        class_balanced: If True, apply inverse-sqrt class frequency weighting
            to correct for reference class imbalance.

    Returns:
        Tuple of:
        - DataFrame with transferred labels and confidence scores.
        - Dict mapping label_col -> DataFrame of soft probabilities (cells x types).
    """
    from sklearn.neighbors import NearestNeighbors

    results = pd.DataFrame(index=query_obs.index)
    soft_probs = {}

    # Fit a single NearestNeighbors model (shared across label columns)
    logger.info("Fitting NearestNeighbors (k=%d) on %d reference cells...", k, len(ref_latent))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(ref_latent)

    # Get distances and indices for all query cells
    logger.info("Querying %d cells...", len(query_latent))
    distances, indices = nn.kneighbors(query_latent)

    # Distance-based weights (inverse distance, avoid division by zero)
    dist_weights = 1.0 / (distances + 1e-10)  # shape: (n_query, k)

    for label_col in label_columns:
        if label_col not in ref_obs.columns:
            logger.warning("%s not in reference, skipping", label_col)
            continue

        labels = ref_obs[label_col].values
        classes = np.unique(labels)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        logger.info("Transferring %s (%d classes, class_balanced=%s)...",
                     label_col, n_classes, class_balanced)

        # Map all reference labels to integer indices
        label_indices = np.empty(len(labels), dtype=np.int32)
        for c, idx in class_to_idx.items():
            label_indices[labels == c] = idx

        # Get neighbor labels as integer indices: shape (n_query, k)
        neighbor_label_idx = label_indices[indices]

        # Compute combined weights
        if class_balanced:
            # Inverse-sqrt class frequency weighting per label column.
            # Pipeline-specific heuristic for HNOCA class imbalance (~43%
            # dorsal telencephalon). sqrt-frequency is a compromise between
            # no correction (majority bias) and full 1/freq (rare class
            # overweighting). Not from published scRNA-seq methodology.
            class_counts = np.bincount(label_indices, minlength=n_classes).astype(float)
            class_freq = class_counts / class_counts.sum()
            class_weight = 1.0 / np.sqrt(class_freq + 1e-10)  # sqrt correction
            class_weight /= class_weight.sum()  # normalize so weights sum to 1

            # Map class weights to each neighbor
            neighbor_class_wt = class_weight[neighbor_label_idx]  # (n_query, k)
            weights = dist_weights * neighbor_class_wt

            # Log the effective boost
            max_wt = class_weight.max()
            min_wt = class_weight.min()
            logger.info("  Class weight range: %.3f - %.3f (max/min ratio: %.1fx)",
                         min_wt, max_wt, max_wt / min_wt if min_wt > 0 else float('inf'))
        else:
            weights = dist_weights

        # Accumulate weighted votes per class for each query cell
        vote_matrix = np.zeros((len(query_latent), n_classes))
        for j in range(k):
            np.add.at(vote_matrix,
                       (np.arange(len(query_latent)), neighbor_label_idx[:, j]),
                       weights[:, j])

        # Soft probabilities: normalize vote matrix to per-cell probability distribution
        row_totals = vote_matrix.sum(axis=1, keepdims=True)
        prob_matrix = vote_matrix / (row_totals + 1e-10)
        soft_probs[label_col] = pd.DataFrame(
            prob_matrix,
            index=query_obs.index,
            columns=[idx_to_class[i] for i in range(n_classes)],
        )

        # Predict: class with highest vote
        predicted_idx = vote_matrix.argmax(axis=1)
        predicted = np.array([idx_to_class[i] for i in predicted_idx])

        # Confidence: fraction of total weight going to the winning class
        total_weights = vote_matrix.sum(axis=1)
        confidence = vote_matrix.max(axis=1) / (total_weights + 1e-10)

        results[f"predicted_{label_col}"] = predicted
        results[f"{label_col}_confidence"] = confidence

    return results, soft_probs


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
    logger.info("Computing cell type fractions per %s...", condition_key)

    df = obs.copy()
    if quality_filter and "quality" in df.columns:
        df = df[df["quality"] == "keep"]

    fractions = (
        df.groupby(condition_key)[label_key]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    logger.info("Result: %d conditions x %d cell types", fractions.shape[0], fractions.shape[1])

    # Verify fractions sum to 1
    row_sums = fractions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        f"Fractions don't sum to 1: {row_sums[~np.isclose(row_sums, 1.0)]}"

    return fractions


def compute_soft_cell_type_fractions(
    obs: pd.DataFrame,
    soft_probs: pd.DataFrame,
    condition_key: str = "condition",
) -> pd.DataFrame:
    """Compute cell type fractions by averaging soft probabilities per condition.

    Instead of argmax followed by value_counts (hard assignment), this averages the
    per-cell probability vectors within each condition. This preserves
    annotation uncertainty and reduces noise in GP training labels.

    Args:
        obs: Cell metadata with condition column.
        soft_probs: DataFrame (cells x cell types) of soft probabilities.
        condition_key: Column identifying experimental conditions.

    Returns:
        DataFrame: rows=conditions, columns=cell types, values=fractions (sum to 1).
    """
    logger.info("Computing soft cell type fractions per %s...", condition_key)

    conditions = obs[condition_key]
    fractions = soft_probs.groupby(conditions).mean()

    # Re-normalize rows to sum to 1 (should be close already)
    row_sums = fractions.sum(axis=1)
    zero_mask = row_sums == 0
    if zero_mask.any():
        logger.warning(
            "Zero-sum conditions detected (%d): %s. Setting to uniform.",
            zero_mask.sum(), list(fractions.index[zero_mask]),
        )
    fractions = fractions.div(row_sums.replace(0, 1), axis=0)

    logger.info("Result: %d conditions x %d cell types", fractions.shape[0], fractions.shape[1])
    return fractions


def compute_bootstrap_uncertainty(
    obs: pd.DataFrame,
    soft_probs: pd.DataFrame,
    condition_key: str = "condition",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap variance on cell type fractions per condition.

    Resamples cells (with replacement) within each condition ``n_bootstrap``
    times and computes the fraction vector for each resample.  Returns the
    per-condition, per-cell-type variance across bootstrap replicates.

    These variances can be passed as heteroscedastic observation noise
    (``train_Yvar``) to ``SingleTaskGP`` in the GP-BO loop.

    Args:
        obs: Cell metadata with *condition_key* column.  Index must align
            with *soft_probs*.
        soft_probs: DataFrame (cells x cell types) of soft KNN probabilities.
        condition_key: Column identifying experimental conditions.
        n_bootstrap: Number of bootstrap resamples per condition.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame (conditions x cell types) of bootstrap variance estimates.
    """
    rng = np.random.default_rng(seed)
    conditions = obs[condition_key]
    cell_types = soft_probs.columns

    grouped = soft_probs.groupby(conditions)
    result_rows: dict[str, np.ndarray] = {}

    for cond, group_df in grouped:
        n_cells = len(group_df)
        values = group_df.values  # (n_cells, n_types)

        # Vectorized bootstrap: draw all indices at once
        idx = rng.integers(0, n_cells, size=(n_bootstrap, n_cells))
        boot_means = values[idx].mean(axis=1)  # (n_bootstrap, n_types)
        # Renormalize each resample to the simplex
        totals = boot_means.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        boot_means /= totals

        result_rows[cond] = np.var(boot_means, axis=0, ddof=0)

    variance_df = pd.DataFrame.from_dict(
        result_rows, orient="index", columns=cell_types,
    )
    variance_df.index.name = condition_key

    logger.info(
        "Bootstrap uncertainty (%d resamples): %d conditions, "
        "mean var=%.2e, max var=%.2e",
        n_bootstrap,
        len(variance_df),
        variance_df.values.mean(),
        variance_df.values.max(),
    )
    return variance_df


def run_mapping_pipeline(
    query_path: Path,
    ref_path: Path,
    model_dir: Path,
    output_prefix: str = "amin_kelley",
    condition_key: str = "condition",
    batch_key: str = "sample",
    n_epochs: int = 500,
    run_gruffi: bool = True,
    gruffi_threshold: float = 0.15,
) -> tuple[sc.AnnData, pd.DataFrame, pd.DataFrame]:
    """Run the full mapping pipeline: load, filter, map, transfer labels, compute fractions.

    Args:
        query_path: Path to input h5ad file.
        ref_path: Path to HNOCA reference h5ad.
        model_dir: Path to scPoli model parameters directory.
        output_prefix: Prefix for output files.
        condition_key: obs column identifying experimental conditions.
        batch_key: obs column identifying batch/sample.
        n_epochs: Number of scPoli training epochs.
        run_gruffi: Whether to run Gruffi stress filtering.
        gruffi_threshold: Gruffi stress score threshold.

    Returns:
        Tuple of (mapped_adata, fractions_df, region_fractions_df).
    """
    # Load data
    logger.info("Loading HNOCA minimal reference...")
    ref = sc.read_h5ad(str(ref_path))
    logger.info("Reference: %s", ref.shape)

    logger.info("Loading query data from %s...", query_path.name)
    query = sc.read_h5ad(str(query_path))
    logger.info("Query: %s", query.shape)

    # Filter to quality cells
    query = filter_quality_cells(query)

    # Gruffi stress filtering (optional)
    if run_gruffi:
        from gopro.gruffi_qc import filter_stressed_cells
        query = filter_stressed_cells(
            query,
            threshold=gruffi_threshold,
            condition_key=condition_key,
        )

    # Prepare query
    query = prepare_query_for_scpoli(query, ref, batch_column=batch_key)

    # Map to HNOCA via scPoli
    query_latent, ref_latent = map_to_hnoca_scpoli(
        query, ref, model_dir,
        n_epochs=n_epochs,
        batch_size=1024,
    )

    # Store latent in query
    query.obsm["X_scpoli"] = query_latent

    # Transfer labels
    logger.info("Transferring cell type labels...")
    label_cols = [ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3]
    transferred, soft_probs = transfer_labels_knn(
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
        condition_key=condition_key,
        label_key=f"predicted_{ANNOT_LEVEL_2}",
    )

    # Also compute region fractions
    region_fractions = compute_cell_type_fractions(
        query.obs,
        condition_key=condition_key,
        label_key=f"predicted_{ANNOT_REGION}",
    )

    # Compute soft cell type fractions (probability-averaged) + bootstrap uncertainty
    if ANNOT_LEVEL_2 in soft_probs:
        soft_fractions = compute_soft_cell_type_fractions(
            query.obs,
            soft_probs[ANNOT_LEVEL_2],
            condition_key=condition_key,
        )
        query.uns["soft_fractions"] = soft_fractions

        # Compare hard vs soft
        diff = (fractions - soft_fractions.reindex_like(fractions).fillna(0)).abs()
        logger.info("Hard vs soft fraction max diff: %.4f, mean diff: %.4f",
                    diff.values.max(), diff.values.mean())

        # Bootstrap uncertainty on soft fractions
        bootstrap_var = compute_bootstrap_uncertainty(
            query.obs,
            soft_probs[ANNOT_LEVEL_2],
            condition_key=condition_key,
        )
        query.uns["bootstrap_variance"] = bootstrap_var

    return query, fractions, region_fractions


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Map query data to HNOCA via scPoli")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input h5ad (default: amin_kelley_2024.h5ad)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Output file prefix (default: amin_kelley)")
    parser.add_argument("--condition-key", type=str, default="condition",
                        help="obs column identifying experimental conditions")
    parser.add_argument("--batch-key", type=str, default="sample",
                        help="obs column identifying batch/sample")
    parser.add_argument("--no-gruffi", action="store_true",
                        help="Skip Gruffi cell stress filtering")
    parser.add_argument("--gruffi-threshold", type=float, default=0.15,
                        help="Gruffi stress score threshold for cluster filtering")
    args = parser.parse_args()

    start = time.time()

    # Resolve paths
    query_path = Path(args.input) if args.input else DATA_DIR / "amin_kelley_2024.h5ad"
    output_prefix = args.output_prefix or "amin_kelley"
    ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"

    for path, name in [(ref_path, "HNOCA reference"), (query_path, "Query data")]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            raise SystemExit(1)
    if not MODEL_DIR.exists():
        logger.error("scPoli model not found at %s", MODEL_DIR)
        raise SystemExit(1)

    query, fractions, region_fractions = run_mapping_pipeline(
        query_path=query_path,
        ref_path=ref_path,
        model_dir=MODEL_DIR,
        output_prefix=output_prefix,
        condition_key=args.condition_key,
        batch_key=args.batch_key,
        run_gruffi=not args.no_gruffi,
        gruffi_threshold=args.gruffi_threshold,
    )

    # Save outputs
    logger.info("Saving outputs...")

    fractions.to_csv(str(DATA_DIR / f"gp_training_labels_{output_prefix}.csv"))
    logger.info("Cell type fractions -> data/gp_training_labels_%s.csv", output_prefix)

    region_fractions.to_csv(str(DATA_DIR / f"gp_training_regions_{output_prefix}.csv"))
    logger.info("Region fractions -> data/gp_training_regions_%s.csv", output_prefix)

    # Save soft fractions and bootstrap variance if computed
    if "soft_fractions" in query.uns:
        query.uns["soft_fractions"].to_csv(
            str(DATA_DIR / f"gp_training_labels_soft_{output_prefix}.csv")
        )
        logger.info("Soft cell type fractions -> data/gp_training_labels_soft_%s.csv", output_prefix)

    if "bootstrap_variance" in query.uns:
        query.uns["bootstrap_variance"].to_csv(
            str(DATA_DIR / f"gp_noise_variance_{output_prefix}.csv")
        )
        logger.info("Bootstrap noise variance -> data/gp_noise_variance_%s.csv", output_prefix)

    output_path = DATA_DIR / f"{output_prefix}_mapped.h5ad"
    query.write(str(output_path), compression="gzip")
    logger.info("Mapped data -> %s", output_path)

    elapsed = time.time() - start
    logger.info("Done in %.1f minutes.", elapsed / 60)

    # Summary
    logger.info("--- MAPPING SUMMARY ---")
    logger.info("Cells mapped: %d", query.n_obs)
    logger.info("Conditions: %d", query.obs[args.condition_key].nunique())
    label_cols = [ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3]
    for label_col in label_cols:
        pred_col = f"predicted_{label_col}"
        if pred_col in query.obs.columns:
            logger.info("%s: %d types", label_col, query.obs[pred_col].nunique())
            top3 = query.obs[pred_col].value_counts().head(3)
            for t, c in top3.items():
                logger.info("  %s: %d cells (%.1f%%)", t, c, c/query.n_obs*100)
