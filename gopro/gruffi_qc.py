"""Gruffi-inspired stress cell filtering for brain organoid scRNA-seq data.

Implements granular filtering of unhealthy cells based on stress pathway
enrichment (glycolysis, ER stress, UPR, apoptosis). Cells are scored per
pathway, clustered at high resolution, and entire stressed clusters are
removed -- mirroring the Gruffi approach (Varga et al., 2022).

Usage::

    from gopro.gruffi_qc import filter_stressed_cells
    adata_clean = filter_stressed_cells(adata, threshold=0.15)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gopro.config import get_logger, DATA_DIR

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRESS_GO_TERMS: dict[str, str] = {
    "glycolysis": "GO:0006096",
    "er_stress": "GO:0006986",
    "upr": "GO:0030968",
    "apoptosis": "GO:0006915",
}

DEFAULT_STRESS_THRESHOLD = 0.15
DEFAULT_RESOLUTION = 2.0
MIN_CELLS_PER_CONDITION = 50

# Path to bundled gene-set JSON
_GENE_SETS_JSON = Path(__file__).resolve().parent / "data" / "gruffi_gene_sets.json"

# ---------------------------------------------------------------------------
# Gene-set loading
# ---------------------------------------------------------------------------

def fetch_go_gene_sets(organism: str = "human") -> dict[str, list[str]]:
    """Load stress-pathway gene sets from the bundled JSON file.

    Parameters
    ----------
    organism
        Currently only ``"human"`` is supported.

    Returns
    -------
    dict
        Mapping from pathway name (e.g. ``"glycolysis"``) to a list of
        HGNC gene symbols.

    Raises
    ------
    FileNotFoundError
        If the bundled JSON cannot be located.
    ValueError
        If *organism* is not ``"human"``.
    """
    if organism != "human":
        raise ValueError(f"Only 'human' organism is supported, got '{organism}'")

    if _GENE_SETS_JSON.exists():
        with open(_GENE_SETS_JSON) as fh:
            gene_sets: dict[str, list[str]] = json.load(fh)
        logger.info(
            "Loaded %d gene sets from %s", len(gene_sets), _GENE_SETS_JSON
        )
        return gene_sets

    raise FileNotFoundError(
        f"Bundled gene-set file not found at {_GENE_SETS_JSON}. "
        "Please ensure gopro/data/gruffi_gene_sets.json is present."
    )


# ---------------------------------------------------------------------------
# Pathway scoring
# ---------------------------------------------------------------------------

def _score_with_decoupler(
    adata,
    gene_sets: dict[str, list[str]],
) -> None:
    """Score pathways using decoupler's AUCell implementation."""
    import decoupler as dc  # type: ignore[import-untyped]

    # Build a net DataFrame in decoupler format
    rows = []
    for pathway, genes in gene_sets.items():
        for gene in genes:
            rows.append({"source": pathway, "target": gene, "weight": 1.0})
    net = pd.DataFrame(rows)

    dc.run_aucell(adata, net=net, source="source", target="target", use_raw=False)
    # decoupler stores results in adata.obsm["aucell_estimate"]
    estimates = adata.obsm["aucell_estimate"]
    for pathway in gene_sets:
        col = f"gruffi_{pathway}"
        if pathway in estimates.columns:
            adata.obs[col] = estimates[pathway].values
        else:
            logger.warning("Pathway '%s' missing from decoupler output; setting to 0", pathway)
            adata.obs[col] = 0.0


def _score_with_scanpy(
    adata,
    gene_sets: dict[str, list[str]],
) -> None:
    """Score pathways using scanpy.tl.score_genes (fallback)."""
    import scanpy as sc  # type: ignore[import-untyped]

    var_names = set(adata.var_names)
    for pathway, genes in gene_sets.items():
        col = f"gruffi_{pathway}"
        overlap = [g for g in genes if g in var_names]
        if len(overlap) < 3:
            logger.warning(
                "Pathway '%s': only %d/%d genes found in adata; skipping",
                pathway, len(overlap), len(genes),
            )
            adata.obs[col] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=overlap, score_name=col)
        logger.info(
            "Scored pathway '%s' with %d/%d genes (scanpy)",
            pathway, len(overlap), len(genes),
        )


def score_stress_pathways(
    adata,
    gene_sets: Optional[dict[str, list[str]]] = None,
    method: str = "auto",
):
    """Score stress pathways and store results in ``adata.obs``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalised expected).
    gene_sets
        Mapping of pathway name to gene list. If ``None``, loads from
        the bundled JSON via :func:`fetch_go_gene_sets`.
    method
        One of ``"auto"``, ``"decoupler"``, or ``"scanpy"``.
        ``"auto"`` tries decoupler first, falling back to scanpy.

    Returns
    -------
    AnnData
        The same *adata* object (modified in-place) with new obs columns:
        ``gruffi_glycolysis``, ``gruffi_er_stress``, ``gruffi_upr``,
        ``gruffi_apoptosis``, and ``gruffi_stress_score``.
    """
    if gene_sets is None:
        gene_sets = fetch_go_gene_sets()

    if method == "auto":
        try:
            _score_with_decoupler(adata, gene_sets)
            logger.info("Stress scoring completed with decoupler (AUCell)")
        except Exception as exc:
            logger.info("decoupler unavailable (%s); falling back to scanpy", exc)
            _score_with_scanpy(adata, gene_sets)
    elif method == "decoupler":
        _score_with_decoupler(adata, gene_sets)
    elif method == "scanpy":
        _score_with_scanpy(adata, gene_sets)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'auto', 'decoupler', or 'scanpy'")

    # Combined stress score = max across pathways
    pathway_cols = [f"gruffi_{p}" for p in gene_sets]
    existing = [c for c in pathway_cols if c in adata.obs.columns]
    if existing:
        adata.obs["gruffi_stress_score"] = adata.obs[existing].max(axis=1)
    else:
        adata.obs["gruffi_stress_score"] = 0.0

    return adata


# ---------------------------------------------------------------------------
# Stressed-cluster identification
# ---------------------------------------------------------------------------

def identify_stressed_clusters(
    adata,
    score_key: str = "gruffi_stress_score",
    resolution: float = DEFAULT_RESOLUTION,
    threshold: float = DEFAULT_STRESS_THRESHOLD,
) -> np.ndarray:
    """Identify stressed cells via high-resolution Leiden clustering.

    Clusters whose *median* stress score exceeds *threshold* are flagged.

    Parameters
    ----------
    adata : AnnData
        Must already contain ``adata.obs[score_key]`` (run
        :func:`score_stress_pathways` first).
    score_key
        Column in ``adata.obs`` holding per-cell stress scores.
    resolution
        Leiden resolution (higher = more granular clusters).
    threshold
        Median score above which a cluster is deemed stressed.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_obs,)``; ``True`` for stressed cells.
    """
    import scanpy as sc  # type: ignore[import-untyped]

    if score_key not in adata.obs.columns:
        raise KeyError(
            f"'{score_key}' not found in adata.obs. "
            "Run score_stress_pathways() first."
        )

    # Work on a copy so we don't pollute the caller's object with
    # temporary PCA / neighbor graphs
    tmp = adata.copy()

    # PCA + neighbors + Leiden
    if "X_pca" not in tmp.obsm:
        sc.pp.pca(tmp, n_comps=min(50, tmp.n_vars - 1))
    if "neighbors" not in tmp.uns:
        sc.pp.neighbors(tmp, n_pcs=min(30, tmp.obsm["X_pca"].shape[1]))

    leiden_key = "_gruffi_leiden"
    sc.tl.leiden(tmp, resolution=resolution, key_added=leiden_key)

    # Median stress per cluster
    cluster_scores = (
        tmp.obs.groupby(leiden_key, observed=True)[score_key].median()
    )
    stressed_clusters = set(cluster_scores[cluster_scores > threshold].index)

    logger.info(
        "Leiden found %d clusters (resolution=%.1f); %d flagged as stressed (median > %.2f)",
        cluster_scores.shape[0], resolution, len(stressed_clusters), threshold,
    )

    mask = tmp.obs[leiden_key].isin(stressed_clusters).values
    return mask


# ---------------------------------------------------------------------------
# Main filtering entry point
# ---------------------------------------------------------------------------

def filter_stressed_cells(
    adata,
    threshold: float = DEFAULT_STRESS_THRESHOLD,
    resolution: float = DEFAULT_RESOLUTION,
    min_cells_per_condition: int = MIN_CELLS_PER_CONDITION,
    method: str = "auto",
    condition_key: str = "condition",
):
    """Filter stressed cells from *adata* using Gruffi-style QC.

    This is the main entry point. It:

    1. Scores stress pathways (glycolysis, ER stress, UPR, apoptosis).
    2. Clusters at high Leiden resolution and flags stressed clusters.
    3. Removes stressed cells, with a safety check that no condition
       drops below *min_cells_per_condition* cells.

    Parameters
    ----------
    adata : AnnData
        Log-normalised AnnData.
    threshold
        Median stress score above which a cluster is marked stressed.
    resolution
        Leiden resolution for granular clustering.
    min_cells_per_condition
        Safety floor: if removing stressed cells would leave fewer than
        this many cells for a condition, stressed cells in that condition
        are retained.
    method
        Scoring method (``"auto"``, ``"decoupler"``, ``"scanpy"``).
    condition_key
        Column in ``adata.obs`` identifying experimental conditions.

    Returns
    -------
    AnnData
        Filtered copy of *adata* with ``adata.obs["gruffi_is_stressed"]``
        added (``True`` for cells that were identified as stressed).
    """
    adata = adata.copy()

    # Step 1: score
    score_stress_pathways(adata, method=method)

    # Step 2: identify stressed clusters
    stressed_mask = identify_stressed_clusters(
        adata,
        score_key="gruffi_stress_score",
        resolution=resolution,
        threshold=threshold,
    )

    adata.obs["gruffi_is_stressed"] = stressed_mask
    n_stressed = int(stressed_mask.sum())
    pct = 100.0 * n_stressed / len(adata) if len(adata) > 0 else 0.0
    logger.info(
        "Stressed cells: %d / %d (%.1f%%)", n_stressed, len(adata), pct,
    )

    # Step 3: safety check per condition
    if condition_key in adata.obs.columns:
        keep_mask = ~stressed_mask  # start with removing all stressed
        conditions = adata.obs[condition_key]
        for cond in conditions.unique():
            cond_idx = conditions == cond
            cond_clean = int((cond_idx & keep_mask).sum())
            if cond_clean < min_cells_per_condition:
                # Restore stressed cells for this condition
                rescued = int((cond_idx & stressed_mask).sum())
                keep_mask = keep_mask | cond_idx
                logger.warning(
                    "Condition '%s': would have only %d cells after filtering "
                    "(min=%d); rescued %d stressed cells",
                    cond, cond_clean, min_cells_per_condition, rescued,
                )
    else:
        keep_mask = ~stressed_mask
        logger.info(
            "Condition key '%s' not found in adata.obs; "
            "skipping per-condition safety check",
            condition_key,
        )

    # Per-condition breakdown logging
    if condition_key in adata.obs.columns:
        breakdown = compute_stress_fraction_per_condition(adata, condition_key)
        for _, row in breakdown.iterrows():
            logger.info(
                "  %s: %d/%d stressed (%.1f%%)",
                row["condition"],
                int(row["n_stressed"]),
                int(row["n_total"]),
                row["fraction_stressed"] * 100,
            )

    n_removed = int((~keep_mask).sum())
    logger.info("Removing %d cells (%.1f%% of total)", n_removed, 100.0 * n_removed / len(adata) if len(adata) else 0)

    return adata[keep_mask].copy()


# ---------------------------------------------------------------------------
# Reporting utility
# ---------------------------------------------------------------------------

def compute_stress_fraction_per_condition(
    adata,
    condition_key: str = "condition",
) -> pd.DataFrame:
    """Compute fraction of stressed cells per experimental condition.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs["gruffi_is_stressed"]``.
    condition_key
        Column in ``adata.obs`` with condition labels.

    Returns
    -------
    pd.DataFrame
        Columns: ``condition``, ``n_total``, ``n_stressed``,
        ``fraction_stressed``.
    """
    if "gruffi_is_stressed" not in adata.obs.columns:
        raise KeyError(
            "'gruffi_is_stressed' not in adata.obs. "
            "Run filter_stressed_cells() or identify_stressed_clusters() first."
        )

    df = adata.obs.groupby(condition_key, observed=True).agg(
        n_total=("gruffi_is_stressed", "size"),
        n_stressed=("gruffi_is_stressed", "sum"),
    ).reset_index()
    df = df.rename(columns={condition_key: "condition"})
    df["fraction_stressed"] = df["n_stressed"] / df["n_total"]
    return df.sort_values("fraction_stressed", ascending=False).reset_index(drop=True)
