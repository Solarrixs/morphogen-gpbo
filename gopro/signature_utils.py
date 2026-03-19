"""Transcriptomic fidelity scoring beyond cell type proportions.

Implements NEST-Score (Naas et al. 2025, Cell Reports) and gene signature
scoring for measuring organoid transcriptomic maturity against fetal
brain references.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from gopro.config import get_logger

logger = get_logger(__name__)


def compute_nest_score(
    query_obs: pd.DataFrame,
    condition_key: str = "condition",
    knn_dist_col: str = "mean_knn_dist_to_ref",
) -> pd.Series:
    """Compute per-condition NEST-inspired transcriptomic fidelity score.

    Uses mean KNN distance to reference (from step 02 transfer_labels_knn)
    as a proxy for transcriptomic neighborhood coverage. Lower distance
    means the organoid cell is transcriptomically closer to its reference
    counterpart.

    Score is computed as: exp(-mean_dist / median_dist_global)
    where median_dist_global normalizes across the entire dataset.

    Args:
        query_obs: Cell-level obs DataFrame with condition and KNN distance columns.
        condition_key: Column identifying experimental conditions.
        knn_dist_col: Column with per-cell mean KNN distance to reference.

    Returns:
        Per-condition NEST score in (0, 1]. Higher = more transcriptomically faithful.

    Raises:
        ValueError: If knn_dist_col is not in query_obs.
    """
    if knn_dist_col not in query_obs.columns:
        raise ValueError(
            f"Column '{knn_dist_col}' not found in query_obs. "
            f"Available columns: {list(query_obs.columns)}"
        )

    # Drop NaN distances before computing global median
    valid_dists = query_obs[knn_dist_col].dropna()
    if len(valid_dists) == 0:
        logger.warning("All KNN distances are NaN; returning empty NEST scores")
        return pd.Series(dtype=float)

    median_global = float(valid_dists.median())
    if median_global <= 0:
        median_global = 1.0  # guard against degenerate case
        logger.warning("Global median KNN distance is <= 0; using 1.0 as fallback")

    # Group by condition, compute mean distance per condition
    mean_dist_per_cond = (
        query_obs
        .groupby(condition_key)[knn_dist_col]
        .mean()
    )

    # Convert to score: exp(-mean_dist / median_global)
    nest_scores = np.exp(-mean_dist_per_cond / median_global)

    logger.info(
        "NEST scores computed for %d conditions: mean=%.3f, min=%.3f, max=%.3f",
        len(nest_scores), nest_scores.mean(), nest_scores.min(), nest_scores.max(),
    )

    return nest_scores


def score_gene_signatures(
    adata,  # sc.AnnData
    signatures: dict[str, list[str]],
    condition_key: str = "condition",
    n_permutations: int = 0,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Score conditions on gene signature enrichment using scanpy.tl.score_genes.

    Args:
        adata: AnnData with gene expression in X.
        signatures: Dict mapping signature_name -> list of gene names.
        condition_key: Column identifying conditions.
        n_permutations: If > 0, compute permutation p-values by scoring
            random gene sets of the same size. 0 = no permutation testing.

    Returns:
        Tuple of (scores_df, pvalues_df). scores_df has conditions as rows,
        signatures as columns. pvalues_df is None if n_permutations == 0.
    """
    import scanpy as sc

    adata = adata.copy()  # avoid mutating caller's data

    all_genes = list(adata.var_names)
    rng = np.random.default_rng(42)

    # Score each signature
    for sig_name, gene_list in signatures.items():
        # Filter to genes present in the data
        valid_genes = [g for g in gene_list if g in adata.var_names]
        if len(valid_genes) == 0:
            logger.warning("Signature '%s': no genes found in adata.var_names", sig_name)
            adata.obs[sig_name] = 0.0
            continue
        n_ctrl = min(len(valid_genes), max(1, len(all_genes) // 10))
        n_bins = min(25, max(1, (len(all_genes) - len(valid_genes)) // n_ctrl))
        sc.tl.score_genes(
            adata, gene_list=valid_genes, score_name=sig_name,
            ctrl_size=n_ctrl, n_bins=n_bins,
        )

    # Aggregate per condition (mean score)
    sig_names = list(signatures.keys())
    scores_df = (
        adata.obs
        .groupby(condition_key)[sig_names]
        .mean()
    )

    # Permutation testing
    pvalues_df = None
    if n_permutations > 0:
        pvalues = {}
        for sig_name, gene_list in signatures.items():
            valid_genes = [g for g in gene_list if g in adata.var_names]
            n_genes = max(len(valid_genes), 1)
            observed = scores_df[sig_name]

            # Build null distribution
            null_scores = np.zeros((n_permutations, len(observed)))
            for i in range(n_permutations):
                perm_genes = list(rng.choice(all_genes, size=n_genes, replace=False))
                perm_col = f"_perm_{sig_name}_{i}"
                perm_n_ctrl = min(len(perm_genes), max(1, len(all_genes) // 10))
                perm_n_bins = min(25, max(1, (len(all_genes) - len(perm_genes)) // perm_n_ctrl))
                sc.tl.score_genes(
                    adata, gene_list=perm_genes, score_name=perm_col,
                    ctrl_size=perm_n_ctrl, n_bins=perm_n_bins,
                )
                null_per_cond = (
                    adata.obs
                    .groupby(condition_key)[perm_col]
                    .mean()
                )
                null_scores[i] = null_per_cond.reindex(observed.index).values

            # p-value = fraction of null scores >= observed
            observed_arr = observed.values[np.newaxis, :]  # (1, n_conditions)
            p_vals = (null_scores >= observed_arr).mean(axis=0)
            pvalues[sig_name] = pd.Series(p_vals, index=observed.index)

        pvalues_df = pd.DataFrame(pvalues)

    return scores_df, pvalues_df
