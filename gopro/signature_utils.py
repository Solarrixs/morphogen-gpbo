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

    # Track columns added to adata.obs so we can clean up without copying
    # the full AnnData (which duplicates the potentially large X matrix).
    added_cols: list[str] = []

    all_genes_arr = np.array(adata.var_names)  # numpy array for efficient rng.choice
    rng = np.random.default_rng(42)

    # Precompute condition group indices for fast per-permutation aggregation
    _cond_labels = adata.obs[condition_key].values
    _unique_conds = np.unique(_cond_labels)
    _cond_masks = {c: (_cond_labels == c) for c in _unique_conds}

    # Cache filtered gene lists to avoid recomputing in the permutation loop
    filtered_genes: dict[str, list[str]] = {}

    try:
        # Score each signature
        for sig_name, gene_list in signatures.items():
            # Filter to genes present in the data
            valid_genes = [g for g in gene_list if g in adata.var_names]
            filtered_genes[sig_name] = valid_genes
            if len(valid_genes) == 0:
                logger.warning("Signature '%s': no genes found in adata.var_names", sig_name)
                adata.obs[sig_name] = 0.0
                added_cols.append(sig_name)
                continue
            n_ctrl = min(len(valid_genes), max(1, len(all_genes_arr) // 10))
            n_bins = min(25, max(1, (len(all_genes_arr) - len(valid_genes)) // n_ctrl))
            sc.tl.score_genes(
                adata, gene_list=valid_genes, score_name=sig_name,
                ctrl_size=n_ctrl, n_bins=n_bins,
            )
            added_cols.append(sig_name)

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
            for sig_name in signatures:
                valid_genes = filtered_genes[sig_name]
                n_genes = max(len(valid_genes), 1)
                observed = scores_df[sig_name]

                # Build null distribution
                null_scores = np.zeros((n_permutations, len(observed)))
                # Precompute ctrl/bins outside the loop (same for all perms of same size)
                perm_n_ctrl = min(n_genes, max(1, len(all_genes_arr) // 10))
                perm_n_bins = min(25, max(1, (len(all_genes_arr) - n_genes) // perm_n_ctrl))
                for i in range(n_permutations):
                    perm_genes = list(rng.choice(all_genes_arr, size=n_genes, replace=False))
                    perm_col = f"_perm_{sig_name}_{i}"
                    added_cols.append(perm_col)
                    sc.tl.score_genes(
                        adata, gene_list=perm_genes, score_name=perm_col,
                        ctrl_size=perm_n_ctrl, n_bins=perm_n_bins,
                    )
                    # Fast per-condition mean using precomputed masks
                    perm_vals = adata.obs[perm_col].values
                    null_per_cond = pd.Series(
                        {c: float(perm_vals[mask].mean()) for c, mask in _cond_masks.items()}
                    )
                    null_scores[i] = null_per_cond.reindex(observed.index).values
                    del adata.obs[perm_col]
                    added_cols.pop()

                # p-value = fraction of null scores >= observed
                observed_arr = observed.values[np.newaxis, :]  # (1, n_conditions)
                p_vals = (null_scores >= observed_arr).mean(axis=0)
                pvalues[sig_name] = pd.Series(p_vals, index=observed.index)

            pvalues_df = pd.DataFrame(pvalues)

        return scores_df, pvalues_df
    finally:
        # Clean up columns added to caller's adata.obs
        for col in added_cols:
            if col in adata.obs.columns:
                del adata.obs[col]


def refine_signatures(
    prior_signatures: dict[str, list[str]],
    adata,  # sc.AnnData
    fidelity_report: pd.DataFrame,
    condition_key: str = "condition",
    alpha: float = 0.7,
    top_k: int = 50,
    score_col: str = "composite_fidelity",
) -> dict[str, list[str]]:
    """Refine gene signatures using observed fidelity data.

    For each signature, identifies genes that are differentially expressed
    between high-fidelity and low-fidelity conditions, then blends the
    data-derived gene set with the prior signature using EMA interpolation.

    new_sig = alpha * data_derived_genes + (1-alpha) * prior_genes

    Implemented as set interpolation: keep genes that appear in either
    the top-k data-derived genes OR the prior, weighted by alpha
    (data-derived genes need rank < top_k * alpha to be included;
    prior genes are kept if they rank < top_k * (1-alpha) in the prior).

    Args:
        prior_signatures: Dict of signature_name -> gene list.
        adata: AnnData with gene expression and condition labels.
        fidelity_report: Per-condition fidelity scores.
        condition_key: Column identifying conditions.
        alpha: Blending weight for data-derived genes (0=all prior, 1=all data).
        top_k: Number of genes per refined signature.
        score_col: Column in fidelity_report to rank conditions by.

    Returns:
        Dict of signature_name -> refined gene list.
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Validate inputs
    if condition_key not in adata.obs.columns:
        logger.warning(
            "condition_key '%s' not in adata.obs; returning prior signatures unchanged",
            condition_key,
        )
        return {k: list(v) for k, v in prior_signatures.items()}

    if score_col not in fidelity_report.columns:
        logger.warning(
            "score_col '%s' not in fidelity_report; returning prior signatures unchanged",
            score_col,
        )
        return {k: list(v) for k, v in prior_signatures.items()}

    # Get conditions present in both adata and fidelity_report
    adata_conditions = set(adata.obs[condition_key].unique())
    report_conditions = set(fidelity_report.index) if fidelity_report.index.name == condition_key else set(
        fidelity_report[condition_key] if condition_key in fidelity_report.columns else fidelity_report.index
    )
    shared_conditions = adata_conditions & report_conditions

    if len(shared_conditions) < 2:
        logger.warning(
            "Need at least 2 shared conditions between adata and fidelity_report "
            "(found %d); returning prior signatures unchanged",
            len(shared_conditions),
        )
        return {k: list(v) for k, v in prior_signatures.items()}

    # Subset adata to shared conditions
    adata_sub = adata[adata.obs[condition_key].isin(shared_conditions)].copy()

    if adata_sub.n_obs < 10:
        logger.warning(
            "Too few cells (%d) after filtering to shared conditions; "
            "returning prior signatures unchanged",
            adata_sub.n_obs,
        )
        return {k: list(v) for k, v in prior_signatures.items()}

    # Build fidelity ranking and split into high/low
    if fidelity_report.index.name == condition_key or condition_key not in fidelity_report.columns:
        scores_series = fidelity_report[score_col]
    else:
        scores_series = fidelity_report.set_index(condition_key)[score_col]

    scores_shared = scores_series.loc[scores_series.index.isin(shared_conditions)].sort_values(ascending=False)
    midpoint = len(scores_shared) // 2
    high_conditions = set(scores_shared.iloc[:midpoint].index)

    # Assign fidelity_bin label
    adata_sub.obs["fidelity_bin"] = adata_sub.obs[condition_key].map(
        lambda c: "high" if c in high_conditions else "low"
    )

    # alpha=0 means return prior unchanged (truncated to top_k)
    if alpha == 0.0:
        return {k: list(v[:top_k]) for k, v in prior_signatures.items()}

    # Run differential expression: high vs low fidelity
    # Use scipy t-test directly to avoid scanpy sparse-utility compatibility issues
    try:
        from scipy.stats import ttest_ind

        # Get expression matrices for high and low groups
        X = adata_sub.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        high_mask = (adata_sub.obs["fidelity_bin"] == "high").values
        low_mask = ~high_mask

        if high_mask.sum() < 2 or low_mask.sum() < 2:
            logger.warning(
                "Too few cells in high (%d) or low (%d) group; "
                "returning prior signatures unchanged",
                high_mask.sum(), low_mask.sum(),
            )
            return {k: list(v) for k, v in prior_signatures.items()}

        X_high = X[high_mask]
        X_low = X[low_mask]

        t_stats, p_values = ttest_ind(X_high, X_low, axis=0, equal_var=False)
        # Replace NaN t-stats with 0 (e.g. zero-variance genes)
        t_stats = np.nan_to_num(t_stats, nan=0.0)

        # Rank genes by t-statistic (descending = upregulated in high-fidelity)
        gene_names = list(adata_sub.var_names)
        gene_order = np.argsort(-t_stats)
        de_genes = [gene_names[i] for i in gene_order]
    except Exception as e:
        logger.warning(
            "Differential expression failed (%s); returning prior signatures unchanged", e
        )
        return {k: list(v) for k, v in prior_signatures.items()}

    # Build refined signatures
    n_data = int(top_k * alpha)
    n_prior = top_k - n_data

    refined = {}
    for sig_name, prior_genes in prior_signatures.items():
        # Data-derived: top n_data DE genes
        data_genes = de_genes[:n_data]

        # Prior: keep up to n_prior from prior (excluding those already in data set)
        data_set = set(data_genes)
        prior_remaining = [g for g in prior_genes if g not in data_set]
        prior_keep = prior_remaining[:n_prior]

        # Combine: data-derived first, then prior fill
        combined = list(data_genes) + list(prior_keep)

        # If we still have fewer than top_k (due to overlap removal), backfill from DE
        if len(combined) < top_k:
            combined_set = set(combined)
            for g in de_genes:
                if g not in combined_set:
                    combined.append(g)
                    combined_set.add(g)
                    if len(combined) >= top_k:
                        break

        refined[sig_name] = combined[:top_k]

    return refined
