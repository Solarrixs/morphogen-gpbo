"""Step 3: Score organoid cell fidelity against Braun fetal brain reference.

KNOWN LIMITATION: This module optimizes cell type proportions (compositional
vectors) rather than per-cell-type transcriptomic fidelity. He et al. 2024
(Nature, DOI:10.1038/s41586-024-08172-8) demonstrated that organoid cell types
can match primary references by proportion while remaining transcriptomically
immature. The Tier 2 RSS scoring partially addresses this by measuring subtype
similarity, but does not capture gene-level maturation signatures.

Future enhancement: incorporate per-cell-type transcriptomic similarity scores
from the HNOCA mapping (snapseed_pca_rss_level_* annotations in step 02) as
additional GP objectives. See also: NEST-Score (Naas et al. 2025,
DOI:10.1016/j.celrep.2025.116168) for a more comprehensive organoid evaluation
metric.

BrainSTEM-inspired two-tier fidelity scoring:
  Tier 1: Use HNOCA-transferred region labels to identify brain region identity
  Tier 2: Compare cell type composition to Braun fetal brain reference by region

The Braun fetal brain reference (~1.65M cells, 11 GB) is loaded in backed mode
to extract region-level cell type composition profiles. These serve as the
ground truth "ideal" composition vectors. Each organoid condition is scored by
Aitchison similarity of its cell type composition to the fetal reference.

This module produces a **composite fidelity score** as a convenience summary,
but the sub-scores (RSS, on-target, off-target, entropy) are also available
individually in the fidelity report.  For rigorous optimization, the
recommended approach is multi-objective BO (``--multi-objective`` in step 04),
which treats each sub-score as a separate objective and returns Pareto-optimal
conditions without requiring weight selection.  See
:data:`DEFAULT_COMPOSITE_WEIGHTS` and :func:`sensitivity_analysis_weights` for
details on the heuristic weighting and how to verify ranking robustness.

Inputs:
  - data/amin_kelley_mapped.h5ad     (from step 02, with HNOCA cell type labels)
  - data/braun-et-al_minimal_for_mapping.h5ad (Braun fetal brain, from Zenodo)

Outputs:
  - data/amin_kelley_fidelity.h5ad   (mapped data with fidelity scores in obs)
  - data/fidelity_report.csv         (per-condition fidelity summary)
  - data/braun_reference_profiles.csv (cached fetal brain composition profiles)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from gopro.config import (
    DATA_DIR,
    ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3,
    get_logger,
)
from gopro.region_targets import (
    OFF_TARGET_LEVEL1,
    HNOCA_TO_BRAUN_REGION,
    build_hnoca_to_braun_label_map,
)
logger = get_logger(__name__)

# Braun fetal brain CellClass values considered neural
BRAUN_NEURAL_CLASSES: set[str] = {
    "Neuron",
    "Neuroblast",
    "Neuronal IPC",
    "Radial glia",
    "Glioblast",
    "Oligo",
}


# ---------------------------------------------------------------------------
# Braun reference profile extraction
# ---------------------------------------------------------------------------

def extract_braun_region_profiles(
    braun_path: Path,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract cell type composition profiles from the Braun fetal brain atlas.

    Loads the Braun reference in backed mode to avoid loading the full 11 GB
    into memory. Computes cell class fractions per SummarizedRegion.

    Args:
        braun_path: Path to braun-et-al_minimal_for_mapping.h5ad.
        cache_path: If provided and exists, load cached profiles instead.

    Returns:
        DataFrame with rows=SummarizedRegion, columns=CellClass, values=fractions.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached Braun profiles from %s", cache_path.name)
        return pd.read_csv(str(cache_path), index_col=0)

    logger.info("Loading Braun fetal brain reference (backed mode)...")
    braun = ad.read_h5ad(str(braun_path), backed="r")
    logger.info("Braun reference: %s cells x %s genes", f"{braun.shape[0]:,}", f"{braun.shape[1]:,}")

    # Extract only the metadata we need (avoids loading expression data)
    obs_df = braun.obs[["SummarizedRegion", "CellClass", "IsNeural"]].copy()
    braun.file.close()

    # Filter to non-doublet neural cells for cleaner profiles
    obs_df = obs_df[obs_df["IsNeural"] == True]  # noqa: E712
    logger.info("Neural cells: %s", f"{len(obs_df):,}")

    # Compute cell class fractions per region
    profiles = (
        obs_df
        .groupby("SummarizedRegion")["CellClass"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    logger.info("Extracted profiles: %d regions x %d cell classes", profiles.shape[0], profiles.shape[1])
    for region in profiles.index:
        n_cells = (obs_df["SummarizedRegion"] == region).sum()
        top_class = profiles.loc[region].idxmax()
        top_frac = profiles.loc[region].max()
        logger.info("  %s: %s cells, top=%s (%.1f%%)", region, f"{n_cells:,}", top_class, top_frac * 100)

    # Cache for future runs
    if cache_path is not None:
        profiles.to_csv(str(cache_path))
        logger.info("Cached profiles to %s", cache_path.name)

    return profiles


def extract_braun_celltype_profiles(
    braun_path: Path,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract detailed CellType composition profiles per SummarizedRegion.

    Uses the finer-grained Braun CellType annotation (69 types) rather than
    CellClass (12 types) for more detailed fidelity comparison.

    Args:
        braun_path: Path to braun-et-al_minimal_for_mapping.h5ad.
        cache_path: If provided and exists, load cached profiles instead.

    Returns:
        DataFrame with rows=SummarizedRegion, columns=CellType, values=fractions.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached Braun CellType profiles from %s", cache_path.name)
        return pd.read_csv(str(cache_path), index_col=0)

    logger.info("Loading Braun fetal brain reference for CellType profiles (backed mode)...")
    braun = ad.read_h5ad(str(braun_path), backed="r")

    obs_df = braun.obs[["SummarizedRegion", "CellType", "IsNeural"]].copy()
    braun.file.close()

    # Keep neural cells with valid CellType annotation
    obs_df = obs_df[obs_df["IsNeural"] == True]  # noqa: E712
    obs_df = obs_df.dropna(subset=["CellType"])

    profiles = (
        obs_df
        .groupby("SummarizedRegion")["CellType"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    if cache_path is not None:
        profiles.to_csv(str(cache_path))
        logger.info("Cached CellType profiles to %s", cache_path.name)

    return profiles


# ---------------------------------------------------------------------------
# Composition vector construction
# ---------------------------------------------------------------------------

def compute_condition_composition(
    obs: pd.DataFrame,
    condition_key: str = "condition",
    label_key: str = "predicted_annot_level_2",
) -> pd.DataFrame:
    """Compute cell type fraction vectors per experimental condition.

    Args:
        obs: Cell metadata with condition and predicted cell type columns.
        condition_key: Column identifying experimental conditions.
        label_key: Column with predicted cell type labels.

    Returns:
        DataFrame: rows=conditions, columns=cell types, values=fractions (sum to 1).
    """
    fractions = (
        obs
        .groupby(condition_key)[label_key]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    return fractions


# ---------------------------------------------------------------------------
# Fidelity scoring functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, handling zero vectors.

    .. deprecated::
        Use :func:`aitchison_similarity` for comparing compositional data
        (cell type fractions).  Cosine similarity does not respect the
        geometry of the simplex.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [0, 1]. Returns 0 if either vector is all zeros.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def aitchison_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Aitchison distance between two compositional vectors.

    The Aitchison distance is the Euclidean distance in CLR (centered
    log-ratio) space.  Unlike cosine similarity it respects the simplex
    geometry of compositional data — it is scale-invariant, permutation-
    invariant, and satisfies sub-compositional dominance (Quinn et al.
    2018, *Bioinformatics* 34(16):2870-2878, DOI:10.1093/bioinformatics/bty175).

    Args:
        a: First composition vector (non-negative, sums to ~1).
        b: Second composition vector (non-negative, sums to ~1).

    Returns:
        Aitchison distance (>= 0).  Returns 0 for identical compositions.
    """
    # Additive pseudo-count for CLR computation. Adequate for distance
    # calculation where zeros are rare. The ILR path in 04_gpbo_loop.py
    # uses multiplicative replacement (Martin-Fernandez 2003) which better
    # preserves simplex geometry.
    eps = 1e-6
    a_safe = np.maximum(a, eps)
    b_safe = np.maximum(b, eps)
    # Closure: ensure each sums to 1
    a_safe = a_safe / a_safe.sum()
    b_safe = b_safe / b_safe.sum()
    # CLR transform
    clr_a = np.log(a_safe) - np.mean(np.log(a_safe))
    clr_b = np.log(b_safe) - np.mean(np.log(b_safe))
    return float(np.linalg.norm(clr_a - clr_b))


def aitchison_similarity(
    a: np.ndarray,
    b: np.ndarray,
    scale: float | None = None,
) -> float:
    """Convert Aitchison distance to a similarity score in [0, 1].

    Uses an exponential decay: ``sim = exp(-dist / scale)``.

    When *scale* is provided, it is used directly.  When *scale* is
    ``None``, falls back to ``_AITCHISON_FALLBACK_SCALE`` (2.0).
    Callers that have access to a corpus of pairwise distances should
    pre-compute the median and pass it as *scale* (the **median
    heuristic**; Garreau, Jitkrittum & Kanagawa, 2017).

    Args:
        a: First composition vector.
        b: Second composition vector.
        scale: Kernel bandwidth.  ``None`` → fallback constant.

    Returns:
        Similarity in [0, 1].  1 = identical, 0 = maximally different.
    """
    dist = aitchison_distance(a, b)
    if scale is None:
        scale = _AITCHISON_FALLBACK_SCALE
    return float(np.exp(-dist / scale))


# Fallback when no pairwise corpus is available (e.g. single comparison).
_AITCHISON_FALLBACK_SCALE = 2.0


def shannon_entropy(fractions: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        fractions: Array of non-negative values summing to ~1.

    Returns:
        Shannon entropy in bits. Higher = more diverse.
    """
    p = fractions[fractions > 0]
    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(fractions: np.ndarray, total_types: int | None = None) -> float:
    """Compute Shannon entropy normalized to [0, 1].

    Normalized by log2(total_types) when provided, otherwise log2(n_nonzero).
    Using total_types ensures consistent normalization across conditions with
    different numbers of detected cell types.

    Args:
        fractions: Array of non-negative values summing to ~1.
        total_types: Total number of possible cell types for normalization.
            If None, uses the number of nonzero elements in fractions.

    Returns:
        Normalized entropy in [0, 1].
    """
    p = fractions[fractions > 0]
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log2(p))
    n = total_types if total_types is not None else len(p)
    if n <= 1:
        return 0.0
    h_max = np.log2(n)
    return float(h / h_max)


def compute_rss(
    condition_vec: pd.Series,
    reference_profiles: pd.DataFrame,
) -> tuple[str, float]:
    """Compute Reference Similarity Spectrum: find best-matching fetal region.

    Uses Aitchison similarity (Euclidean distance in CLR space) to compare
    the condition's cell type composition against each fetal brain region
    profile.  Aitchison distance is the correct metric for compositional
    data (Quinn et al., *Bioinformatics* 2018).

    Args:
        condition_vec: Cell type fractions for one condition (Series, index=cell types).
        reference_profiles: Fetal brain composition profiles
            (rows=regions, columns=cell classes).

    Returns:
        Tuple of (best_matching_region, similarity_score).
    """
    # Align cell types: use union of both label sets, fill missing with 0
    all_labels = sorted(set(condition_vec.index) | set(reference_profiles.columns))
    cond_aligned = np.array([condition_vec.get(l, 0.0) for l in all_labels])

    # Compute all Aitchison distances first, then derive the scale via an
    # adapted median heuristic (Garreau, Jitkrittum & Kanagawa, 2017).
    # NOTE: This uses per-query distances (condition → all references), not
    # all-pairwise distances. Per-query bandwidth makes absolute similarity
    # values non-comparable across conditions, but relative rankings within
    # a single scoring call are valid.
    ref_aligned_vecs = {}
    distances = {}
    for region in reference_profiles.index:
        ref_vec = np.array([reference_profiles.loc[region].get(l, 0.0) for l in all_labels])
        ref_aligned_vecs[region] = ref_vec
        distances[region] = aitchison_distance(cond_aligned, ref_vec)

    if len(distances) >= 2:
        scale = float(np.median(list(distances.values())))
        # Guard against degenerate case where median is ~0
        if scale < 1e-8:
            scale = _AITCHISON_FALLBACK_SCALE
    else:
        scale = _AITCHISON_FALLBACK_SCALE

    best_region = "none"
    best_sim = 0.0
    for region, dist in distances.items():
        sim = float(np.exp(-dist / scale))
        if sim > best_sim:
            best_sim = sim
            best_region = region

    return best_region, best_sim


def compute_off_target_fraction(
    obs_subset: pd.DataFrame,
    level1_key: str = "predicted_annot_level_1",
) -> float:
    """Compute fraction of cells with off-target (non-neural) identity.

    Off-target cell types include: PSC (pluripotent), MC (mesenchymal),
    EC (endothelial), Microglia, NC Derivatives.

    Args:
        obs_subset: Cell metadata for one condition.
        level1_key: Column with HNOCA level-1 cell type labels.

    Returns:
        Fraction of off-target cells in [0, 1].
    """
    if level1_key not in obs_subset.columns:
        return np.nan
    return float(obs_subset[level1_key].isin(OFF_TARGET_LEVEL1).mean())


def compute_on_target_fraction(
    obs_subset: pd.DataFrame,
    region_key: str = "predicted_annot_region_rev2",
) -> tuple[str, float]:
    """Compute dominant brain region and its fraction for a condition.

    The "on-target" fraction is the proportion of cells assigned to the
    single most common brain region. Higher values indicate a more focused
    regionalization.

    Args:
        obs_subset: Cell metadata for one condition.
        region_key: Column with predicted brain region labels.

    Returns:
        Tuple of (dominant_region, on_target_fraction).
    """
    if region_key not in obs_subset.columns:
        return "unknown", np.nan
    region_counts = obs_subset[region_key].value_counts(normalize=True)
    dominant = region_counts.index[0]
    return str(dominant), float(region_counts.iloc[0])


def compute_braun_entropy_center(braun_profiles: pd.DataFrame) -> float:
    """Compute mean normalized entropy across Braun fetal brain region profiles.

    Each row of braun_profiles is a region's cell type composition (fractions).
    Returns the mean normalized entropy, which serves as the data-driven center
    for the Gaussian entropy penalty in composite fidelity scoring.

    Args:
        braun_profiles: DataFrame with rows=regions, columns=cell types,
            values=fractions summing to ~1 per row.

    Returns:
        Mean normalized entropy across all regions, in [0, 1].
    """
    n_types = braun_profiles.shape[1]
    entropies = braun_profiles.apply(
        lambda row: normalized_entropy(row.values.astype(float), total_types=n_types),
        axis=1,
    )
    return float(entropies.mean())


# Fallback when no Braun reference is available
_DEFAULT_ENTROPY_CENTER = 0.55
# Heuristic: no literature basis for sigma=0.2. Consider deriving from
# variance of Braun region entropies for a data-driven alternative.
_ENTROPY_SIGMA = 0.2

# ---------------------------------------------------------------------------
# Default composite fidelity weights — HEURISTIC, NOT LITERATURE-BACKED.
#
# These weights are engineering choices that produce reasonable condition
# rankings for the Amin & Kelley 2024 morphogen screen.  No published
# organoid evaluation framework uses fixed-weight composite scores.
#
# Literature alternatives:
#   - Per-cell-type transcriptomic similarity
#     (He, Treutlein et al. 2024, DOI:10.1038/s41586-024-08172-8)
#   - NEST-Score for organoid benchmarking
#     (Naas, Knoblich et al. 2025, DOI:10.1016/j.celrep.2025.116168)
#   - Hierarchical staged QC with pass/fail gating
#     (Castiglione et al. 2025, DOI:10.1038/s41598-025-14425-x)
#
# Principled approach: use multi-objective BO (--multi-objective in step 04)
# which treats each sub-score as a separate objective and returns Pareto-
# optimal conditions without weight selection (Daulton et al. 2020, qEHVI).
#
# Use sensitivity_analysis_weights() to verify ranking robustness before
# relying on this composite score for decision-making.
# ---------------------------------------------------------------------------
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "rss": 0.35,
    "on_target": 0.25,
    "off_target": 0.25,
    "entropy": 0.15,
}

# Weights when maturity proxy is available (still heuristic, not literature-backed).
# Maturity gets 0.15 weight, taken proportionally from other sub-scores.
DEFAULT_COMPOSITE_WEIGHTS_WITH_MATURITY: dict[str, float] = {
    "rss": 0.30,
    "on_target": 0.20,
    "off_target": 0.20,
    "entropy": 0.15,
    "maturity": 0.15,
}


def compute_composite_fidelity(
    rss_score: float,
    on_target_frac: float,
    off_target_frac: float,
    norm_entropy: float,
    weights: Optional[dict[str, float]] = None,
    entropy_center: Optional[float] = None,
    maturity_score: Optional[float] = None,
) -> float:
    """Compute a single composite fidelity score in [0, 1].

    Combines four sub-scores with configurable weights:
    - RSS (Aitchison similarity to fetal brain): higher is better
    - On-target fraction: higher is better
    - Off-target fraction: lower is better (inverted)
    - Normalized entropy: moderate is best (penalize extremes)

    .. note:: **Heuristic weighting — not literature-backed.**

       The default weights (RSS=0.35, on-target=0.25, off-target=0.25,
       entropy=0.15) are engineering choices tuned for reasonable ranking
       behaviour.  No published organoid evaluation framework uses fixed-
       weight composite scores.  See ``DEFAULT_COMPOSITE_WEIGHTS`` docstring
       for literature alternatives.

       **Recommended principled approach**: use multi-objective Bayesian
       optimization (``--multi-objective`` flag in step 04), which treats
       RSS, on-target, off-target, and entropy as separate objectives and
       returns Pareto-optimal conditions without requiring weight selection
       (Daulton et al. 2020, qEHVI; DOI:10.48550/arXiv.2006.05078).

       Use :func:`sensitivity_analysis_weights` to verify that condition
       rankings are robust to weight perturbation.

    Args:
        rss_score: Aitchison similarity to best-matching fetal region [0, 1].
        on_target_frac: Fraction of cells in dominant region [0, 1].
        off_target_frac: Fraction of non-neural cells [0, 1].
        norm_entropy: Normalized Shannon entropy [0, 1].
        weights: Dict with keys 'rss', 'on_target', 'off_target', 'entropy'.
            Must sum to ~1.0. Defaults to ``DEFAULT_COMPOSITE_WEIGHTS``.
        entropy_center: Center of Gaussian entropy penalty [0, 1].
            Derived from Braun fetal brain reference via
            ``compute_braun_entropy_center()``. Falls back to 0.55 if None.

    Returns:
        Composite fidelity score in [0, 1].
    """
    if weights is None:
        if maturity_score is not None and not np.isnan(maturity_score):
            weights = dict(DEFAULT_COMPOSITE_WEIGHTS_WITH_MATURITY)
        else:
            weights = dict(DEFAULT_COMPOSITE_WEIGHTS)

    if entropy_center is None:
        entropy_center = _DEFAULT_ENTROPY_CENTER

    # Handle NaN inputs gracefully
    rss_score = rss_score if not np.isnan(rss_score) else 0.0
    on_target_frac = on_target_frac if not np.isnan(on_target_frac) else 0.0
    off_target_frac = off_target_frac if not np.isnan(off_target_frac) else 1.0
    norm_entropy = norm_entropy if not np.isnan(norm_entropy) else 0.0

    # Entropy contribution: penalize both too low (monoculture) and too high
    # (disorganized). Center is data-driven from Braun fetal brain reference.
    entropy_score = np.exp(-((norm_entropy - entropy_center) ** 2) / (2 * _ENTROPY_SIGMA ** 2))

    score = (
        weights["rss"] * rss_score
        + weights["on_target"] * on_target_frac
        + weights["off_target"] * (1.0 - off_target_frac)
        + weights["entropy"] * entropy_score
    )

    # Add maturity contribution if available and weighted
    if maturity_score is not None and not np.isnan(maturity_score) and "maturity" in weights:
        score += weights["maturity"] * maturity_score

    return float(np.clip(score, 0.0, 1.0))


def sensitivity_analysis_weights(
    report: pd.DataFrame,
    n_samples: int = 200,
    seed: int = 42,
    entropy_center: Optional[float] = None,
) -> pd.DataFrame:
    """Sweep random weight combinations and report ranking stability.

    Generates ``n_samples`` random weight vectors drawn from a symmetric
    Dirichlet distribution, re-scores all conditions under each weight
    vector, and computes the Spearman rank correlation between the
    resulting ranking and the default-weight ranking.

    Inspired by the sensitivity-to-assumptions principle in Castiglione
    et al. 2025 (DOI:10.1038/s41598-025-14425-x): before trusting a
    composite score, verify that the ranking is not an artefact of the
    particular weight choice.

    If the median Spearman rho is high (>0.85), the ranking is robust
    and the composite score is a reliable summary.  If it is low, prefer
    multi-objective optimization (``--multi-objective`` in step 04) which
    avoids weight selection entirely.

    Args:
        report: Fidelity report DataFrame from :func:`score_all_conditions`,
            must contain columns ``rss_score``, ``on_target_fraction``,
            ``off_target_fraction``, ``normalized_entropy``.
        n_samples: Number of random weight vectors to draw.
        seed: Random seed for reproducibility.
        entropy_center: Entropy center for scoring.  If None, uses the
            module default (``_DEFAULT_ENTROPY_CENTER``).

    Returns:
        DataFrame with one row per weight sample, columns:
        ``w_rss``, ``w_on_target``, ``w_off_target``, ``w_entropy``,
        ``spearman_rho``, ``spearman_pvalue``.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)

    required = {"rss_score", "on_target_fraction", "off_target_fraction", "normalized_entropy"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"Report is missing columns needed for sensitivity analysis: {missing}")

    # Baseline ranking under default weights
    baseline_scores = np.array([
        compute_composite_fidelity(
            row["rss_score"], row["on_target_fraction"],
            row["off_target_fraction"], row["normalized_entropy"],
            entropy_center=entropy_center,
        )
        for _, row in report.iterrows()
    ])
    baseline_rank = np.argsort(-baseline_scores)  # descending

    results: list[dict] = []
    # Draw weight vectors from Dirichlet(1,1,1,1) — uniform on the simplex
    weight_samples = rng.dirichlet(np.ones(4), size=n_samples)

    for w in weight_samples:
        wdict = {"rss": w[0], "on_target": w[1], "off_target": w[2], "entropy": w[3]}
        scores = np.array([
            compute_composite_fidelity(
                row["rss_score"], row["on_target_fraction"],
                row["off_target_fraction"], row["normalized_entropy"],
                weights=wdict,
                entropy_center=entropy_center,
            )
            for _, row in report.iterrows()
        ])
        rho, pval = spearmanr(baseline_rank, np.argsort(-scores))
        results.append({
            "w_rss": w[0],
            "w_on_target": w[1],
            "w_off_target": w[2],
            "w_entropy": w[3],
            "spearman_rho": rho,
            "spearman_pvalue": pval,
        })

    result_df = pd.DataFrame(results)
    median_rho = result_df["spearman_rho"].median()
    logger.info(
        "Weight sensitivity analysis (%d samples): median Spearman rho = %.3f "
        "(>0.85 suggests robust ranking)",
        n_samples, median_rho,
    )
    return result_df


# ---------------------------------------------------------------------------
# Per-condition fidelity report
# ---------------------------------------------------------------------------

def score_all_conditions(
    query_adata: sc.AnnData,
    braun_profiles: pd.DataFrame,
    condition_key: str = "condition",
    hnoca_level3_profiles: Optional[pd.DataFrame] = None,
    label_map: Optional[dict[str, str]] = None,
    target_profile: Optional[pd.Series] = None,
    entropy_center: Optional[float] = None,
    control_condition: Optional[str] = None,
) -> pd.DataFrame:
    """Compute full fidelity report for all conditions in the query data.

    For each condition, computes:
    - n_cells: number of cells
    - dominant_region: most common brain region (HNOCA label)
    - on_target_fraction: fraction of cells in dominant region
    - off_target_fraction: fraction of non-neural cells
    - shannon_entropy: cell type diversity (bits)
    - normalized_entropy: entropy normalized to [0, 1]
    - rss_best_region: best-matching brain region via level-3 composition
    - rss_score: cosine similarity to best region profile
    - composite_fidelity: weighted combination of all scores [0, 1]

    When ``target_profile`` is provided, the RSS score is computed as cosine
    similarity between each condition's composition and the target profile
    (instead of searching across all reference regions). The ``rss_best_region``
    column will report "custom_target" in this case.

    Args:
        query_adata: Mapped AnnData with HNOCA-transferred labels in obs.
        braun_profiles: Fetal brain composition profiles (fallback for RSS).
        condition_key: Column identifying experimental conditions.
        hnoca_level3_profiles: HNOCA region profiles at level-3 granularity.
            If provided, used for RSS instead of braun_profiles (much better
            region discrimination).
        label_map: Mapping from HNOCA labels to Braun CellClass.
        target_profile: Optional target cell type composition profile.
            When provided, RSS is computed against this single profile instead
            of searching all reference regions.
        control_condition: Optional name of an untreated/baseline condition.
            When provided, an ``is_hit`` boolean column is added to the report
            using :func:`compute_hit_threshold` (3-MAD above control).

    Returns:
        DataFrame with one row per condition, sorted by composite_fidelity
        descending.
    """
    obs = query_adata.obs
    pred_level1 = f"predicted_{ANNOT_LEVEL_1}"
    pred_level2 = f"predicted_{ANNOT_LEVEL_2}"
    pred_level3 = f"predicted_{ANNOT_LEVEL_3}"
    pred_region = f"predicted_{ANNOT_REGION}"

    # When a target profile is provided, score against it directly
    use_target_profile = target_profile is not None

    if use_target_profile:
        logger.info("Scoring against custom target profile (%d cell types)",
                     len(target_profile))
        # Wrap target profile as a single-row DataFrame for compute_rss
        target_as_df = pd.DataFrame([target_profile], index=["custom_target"])
        rss_profiles = target_as_df
        # Use the label level that best matches the target profile's index
        rss_label_key = pred_level2  # default to level-2
        # Check overlap with different label levels
        for candidate_key, candidate_col in [
            (pred_level3, ANNOT_LEVEL_3),
            (pred_level2, ANNOT_LEVEL_2),
            (pred_level1, ANNOT_LEVEL_1),
        ]:
            if candidate_key in obs.columns:
                overlap = set(target_profile.index) & set(obs[candidate_key].unique())
                if len(overlap) > 0:
                    rss_label_key = candidate_key
                    break
    elif hnoca_level3_profiles is not None and pred_level3 in obs.columns:
        rss_profiles = hnoca_level3_profiles
        rss_label_key = pred_level3
        logger.info("Using HNOCA level-3 profiles for RSS (region-specific cell types)")
    else:
        rss_profiles = braun_profiles
        rss_label_key = pred_level1
        logger.info("Using Braun CellClass profiles for RSS (level-1 fallback)")

    conditions = obs[condition_key].unique()
    logger.info("Scoring %d conditions...", len(conditions))

    # Check if KNN latent distance is available (transcriptomic maturity proxy)
    has_maturity = "mean_knn_dist_to_ref" in obs.columns
    if has_maturity:
        # Compute a global scale factor for converting distance → similarity.
        # Use median distance so the score is relative to the dataset.
        all_dists = obs["mean_knn_dist_to_ref"].dropna()
        dist_scale = float(all_dists.median()) if len(all_dists) > 0 else 1.0
        logger.info(
            "Maturity proxy available: median KNN dist=%.3f (scale factor)",
            dist_scale,
        )
    else:
        logger.info(
            "No mean_knn_dist_to_ref in obs — skipping maturity sub-score. "
            "Run step 02 with updated transfer_labels_knn to enable."
        )

    # Total unique cell types across all conditions for consistent entropy normalization
    total_cell_types = obs[pred_level2].nunique()

    results: list[dict] = []

    for cond in conditions:
        mask = obs[condition_key] == cond
        subset = obs[mask]
        n_cells = int(mask.sum())

        # Cell type composition (level 2 — 17 types)
        ct_fracs = subset[pred_level2].value_counts(normalize=True)

        # On-target: dominant region fraction
        dominant_region, on_target = compute_on_target_fraction(
            subset, region_key=pred_region,
        )

        # Off-target: non-neural fraction
        off_target = compute_off_target_fraction(
            subset, level1_key=pred_level1,
        )

        # Entropy
        frac_array = ct_fracs.values
        h = shannon_entropy(frac_array)
        h_norm = normalized_entropy(frac_array, total_types=total_cell_types)

        # RSS: cosine similarity to region profiles (or target profile)
        rss_fracs = subset[rss_label_key].value_counts(normalize=True)
        if label_map is not None and rss_label_key == pred_level1 and not use_target_profile:
            rss_fracs = align_composition_to_braun(rss_fracs, label_map)
        rss_region, rss_score = compute_rss(rss_fracs, rss_profiles)

        # Maturity proxy: mean KNN latent distance per condition → similarity score.
        # Lower distance = closer to reference in latent space = more mature.
        # Converted to [0, 1] via exp(-dist/scale) so 1.0 = transcriptomically close.
        if has_maturity:
            cond_dists = subset["mean_knn_dist_to_ref"].dropna()
            if len(cond_dists) > 0:
                mean_dist = float(cond_dists.mean())
                maturity_score = float(np.exp(-mean_dist / dist_scale))
            else:
                maturity_score = 0.0
        else:
            maturity_score = np.nan

        # Composite fidelity
        fidelity = compute_composite_fidelity(
            rss_score=rss_score,
            on_target_frac=on_target,
            off_target_frac=off_target,
            norm_entropy=h_norm,
            entropy_center=entropy_center,
            maturity_score=maturity_score if has_maturity else None,
        )

        results.append({
            "condition": cond,
            "n_cells": n_cells,
            "n_cell_types": len(ct_fracs),
            "dominant_region": dominant_region,
            "on_target_fraction": round(on_target, 4),
            "off_target_fraction": round(off_target, 4),
            "shannon_entropy": round(h, 4),
            "normalized_entropy": round(h_norm, 4),
            "rss_best_region": rss_region,
            "rss_score": round(rss_score, 4),
            "maturity_score": round(maturity_score, 4) if not np.isnan(maturity_score) else np.nan,
            "composite_fidelity": round(fidelity, 4),
        })

    report = (
        pd.DataFrame(results)
        .set_index("condition")
        .sort_values("composite_fidelity", ascending=False)
    )

    if control_condition is not None:
        hit_info = compute_hit_threshold(report, control_condition)
        report["is_hit"] = report.index.isin(hit_info["hit_conditions"])
        logger.info(
            "Hit calling: %d/%d conditions exceed threshold %.4f (control=%s)",
            hit_info["n_hits"], len(report), hit_info["threshold"], control_condition,
        )

    return report


def compute_hit_threshold(
    report: pd.DataFrame,
    control_condition: str,
    score_col: str = "composite_fidelity",
    n_mad: float = 3.0,
) -> dict:
    """Compute hit threshold from a control/untreated baseline condition.

    Uses MAD (median absolute deviation) for robustness to outliers.
    A condition is a "hit" if its score exceeds control_median + n_mad * MAD.

    Args:
        report: Per-condition fidelity report (from score_all_conditions).
        control_condition: Name of the untreated/baseline condition.
        score_col: Column to threshold on.
        n_mad: Number of MADs above control median for hit calling.

    Returns:
        Dict with keys: control_median, mad, threshold, n_hits, hit_conditions.

    Raises:
        KeyError: If control_condition is not found in the report index.
    """
    if control_condition not in report.index:
        raise KeyError(
            f"Control condition '{control_condition}' not found in report. "
            f"Available conditions: {list(report.index)}"
        )

    control_score = float(report.loc[control_condition, score_col])
    scores = report[score_col].dropna()

    # MAD with consistency constant (1.4826) for normality equivalence
    median_scores = float(scores.median())
    mad = float(np.median(np.abs(scores - median_scores))) * 1.4826

    threshold = control_score + n_mad * mad
    hit_mask = scores > threshold
    hit_conditions = list(scores[hit_mask].index)

    logger.info(
        "Hit threshold: control=%s (%.4f), MAD=%.4f, threshold=%.4f, hits=%d/%d",
        control_condition, control_score, mad, threshold, len(hit_conditions), len(scores),
    )

    return {
        "control_median": control_score,
        "mad": mad,
        "threshold": threshold,
        "n_hits": len(hit_conditions),
        "hit_conditions": hit_conditions,
    }


def assign_cell_level_fidelity(
    query_adata: sc.AnnData,
    report: pd.DataFrame,
    condition_key: str = "condition",
) -> sc.AnnData:
    """Propagate per-condition fidelity scores to individual cells in obs.

    Adds the following columns to query_adata.obs:
    - fidelity_score: composite fidelity of the cell's condition
    - rss_score: RSS score of the cell's condition
    - on_target_fraction: on-target fraction of the cell's condition
    - is_off_target: whether this cell's level-1 type is off-target

    Args:
        query_adata: AnnData with condition column.
        report: Per-condition fidelity report.
        condition_key: Column identifying conditions.

    Returns:
        AnnData with fidelity columns added to obs.
    """
    pred_level1 = f"predicted_{ANNOT_LEVEL_1}"

    # Map condition-level scores to cells
    cond_to_fidelity = report["composite_fidelity"].to_dict()
    cond_to_rss = report["rss_score"].to_dict()
    cond_to_on_target = report["on_target_fraction"].to_dict()

    query_adata.obs["fidelity_score"] = (
        query_adata.obs[condition_key].map(cond_to_fidelity).astype(float)
    )
    query_adata.obs["rss_score"] = (
        query_adata.obs[condition_key].map(cond_to_rss).astype(float)
    )
    query_adata.obs["on_target_fraction"] = (
        query_adata.obs[condition_key].map(cond_to_on_target).astype(float)
    )

    # Per-cell off-target flag
    if pred_level1 in query_adata.obs.columns:
        query_adata.obs["is_off_target"] = (
            query_adata.obs[pred_level1].isin(OFF_TARGET_LEVEL1)
        )

    return query_adata


# ---------------------------------------------------------------------------
# Label alignment utilities
# ---------------------------------------------------------------------------

def build_hnoca_region_profiles_level3(
    ref_path: Path,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build region-level cell type profiles from HNOCA reference using level-3 labels.

    Level-3 labels (e.g. "Cerebellar NPC", "Thalamic Neuron") are region-specific,
    giving perfect discrimination between brain regions (cosine similarity ≈ 0
    between all region pairs). This is much more informative than level-1 labels
    for RSS scoring.

    Args:
        ref_path: Path to hnoca_minimal_for_mapping.h5ad.
        cache_path: If provided and exists, load cached profiles instead.

    Returns:
        DataFrame with rows=annot_region_rev2, columns=annot_level_3_rev2,
        values=fractions.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached HNOCA level-3 region profiles from %s", cache_path.name)
        return pd.read_csv(str(cache_path), index_col=0)

    logger.info("Building HNOCA region profiles from level-3 labels...")
    ref = ad.read_h5ad(str(ref_path), backed="r")

    profiles = (
        ref.obs
        .groupby("annot_region_rev2")["annot_level_3_rev2"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    ref.file.close()

    # Drop "Unspecific" region — not a real brain region
    if "Unspecific" in profiles.index:
        profiles = profiles.drop("Unspecific")

    logger.info("Built profiles: %d regions x %d cell types", profiles.shape[0], profiles.shape[1])

    if cache_path is not None:
        profiles.to_csv(str(cache_path))
        logger.info("Cached to %s", cache_path.name)

    return profiles


def align_composition_to_braun(
    hnoca_fracs: pd.Series,
    label_map: dict[str, str],
) -> pd.Series:
    """Re-key HNOCA-labeled composition vector to Braun CellClass labels.

    When multiple HNOCA types map to the same Braun class, their fractions
    are summed.

    Args:
        hnoca_fracs: Series with HNOCA level-1 labels as index.
        label_map: Mapping from HNOCA labels to Braun CellClass.

    Returns:
        Series with Braun CellClass labels as index, fractions summed.
    """
    mapped = {}
    for hnoca_label, frac in hnoca_fracs.items():
        braun_label = label_map.get(hnoca_label, hnoca_label)
        mapped[braun_label] = mapped.get(braun_label, 0.0) + frac
    return pd.Series(mapped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fidelity_scoring(
    mapped_path: Path,
    braun_path: Path,
    ref_path: Optional[Path] = None,
    condition_key: str = "condition",
) -> tuple[pd.DataFrame, sc.AnnData]:
    """Run the full fidelity scoring pipeline without writing files.

    Args:
        mapped_path: Path to mapped query h5ad (from step 02).
        braun_path: Path to Braun fetal brain reference h5ad.
        ref_path: Path to HNOCA reference h5ad (for level-3 RSS). Optional.
        condition_key: Column identifying experimental conditions.

    Returns:
        Tuple of (fidelity_report_df, annotated_adata).
    """
    # Validate mapped h5ad BEFORE loading expensive Braun data
    from gopro.validation import validate_mapped_adata
    validate_mapped_adata(mapped_path, condition_key=condition_key)

    # Step 1: Extract Braun fetal brain reference profiles
    logger.info("--- STEP 1: Extract Braun fetal brain reference profiles ---")

    braun_profiles = extract_braun_region_profiles(
        braun_path,
        cache_path=braun_path.parent / "braun_reference_profiles.csv",
    )

    # Also extract detailed CellType profiles for reporting
    extract_braun_celltype_profiles(
        braun_path,
        cache_path=braun_path.parent / "braun_reference_celltype_profiles.csv",
    )

    # Step 2: Load mapped query data
    logger.info("--- STEP 2: Load mapped query data ---")

    logger.info("Loading mapped query data...")
    query = sc.read_h5ad(str(mapped_path))
    logger.info("Query: %s cells x %s genes", f"{query.n_obs:,}", f"{query.n_vars:,}")

    # Verify expected columns exist
    pred_level1 = f"predicted_{ANNOT_LEVEL_1}"
    pred_level2 = f"predicted_{ANNOT_LEVEL_2}"
    pred_region = f"predicted_{ANNOT_REGION}"

    required_cols = [pred_level1, pred_level2, pred_region]
    missing = [c for c in required_cols if c not in query.obs.columns]
    if missing:
        raise ValueError(
            f"Missing predicted label columns: {missing}. "
            f"These should have been added by step 02. "
            f"Available: {list(query.obs.columns)}"
        )

    # Detect condition column
    if condition_key not in query.obs.columns:
        for alt in ["sample", "protocol", "well", "batch"]:
            if alt in query.obs.columns:
                condition_key = alt
                break
        else:
            raise ValueError(
                f"No condition column found. Available: {list(query.obs.columns)}"
            )
    logger.info("Using condition column: '%s' (%d unique conditions)",
                condition_key, query.obs[condition_key].nunique())

    # Step 3: Build region profiles for RSS scoring
    logger.info("--- STEP 3: Build region profiles for RSS ---")

    hnoca_l3_profiles = None
    if ref_path is not None and ref_path.exists():
        hnoca_l3_profiles = build_hnoca_region_profiles_level3(
            ref_path,
            cache_path=ref_path.parent / "hnoca_region_profiles_level3.csv",
        )
    else:
        logger.warning("HNOCA reference not found — falling back to Braun CellClass RSS")

    label_map = build_hnoca_to_braun_label_map()

    # Compute data-driven entropy center from Braun reference
    entropy_center = compute_braun_entropy_center(braun_profiles)
    logger.info("Data-driven entropy center from Braun reference: %.4f", entropy_center)

    # Step 4: Score all conditions
    logger.info("--- STEP 4: Compute fidelity scores ---")

    report = score_all_conditions(
        query_adata=query,
        braun_profiles=braun_profiles,
        condition_key=condition_key,
        hnoca_level3_profiles=hnoca_l3_profiles,
        label_map=label_map,
        entropy_center=entropy_center,
    )

    # Step 5: Assign cell-level fidelity scores
    logger.info("--- STEP 5: Assign cell-level fidelity scores ---")

    query = assign_cell_level_fidelity(query, report, condition_key=condition_key)
    logger.info("Added fidelity columns to %s cells", f"{query.n_obs:,}")

    return report, query


def main() -> None:
    """Run the full fidelity scoring pipeline."""
    start = time.time()

    # Check prerequisites
    braun_path = DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
    mapped_path = DATA_DIR / "amin_kelley_mapped.h5ad"

    for path, name, hint in [
        (braun_path, "Braun fetal brain reference", "Run: python 00_zenodo_download.py"),
        (mapped_path, "Mapped query data", "Run step 02 first: python 02_map_to_hnoca.py"),
    ]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            logger.error("%s", hint)
            raise SystemExit(1)

    ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"

    report, query = run_fidelity_scoring(
        mapped_path=mapped_path,
        braun_path=braun_path,
        ref_path=ref_path if ref_path.exists() else None,
    )

    # Save outputs
    logger.info("--- Saving outputs ---")

    report_path = DATA_DIR / "fidelity_report.csv"
    report.to_csv(str(report_path))
    logger.info("Fidelity report -> %s", report_path)

    output_path = DATA_DIR / "amin_kelley_fidelity.h5ad"
    query.write(str(output_path), compression="gzip")
    logger.info("Annotated data -> %s", output_path)

    # Summary
    elapsed = time.time() - start
    logger.info("--- FIDELITY SCORING SUMMARY ---")
    logger.info("Cells scored: %s", f"{query.n_obs:,}")
    logger.info("Conditions: %d", len(report))
    logger.info("Time elapsed: %.1fs", elapsed)

    logger.info("Top 5 conditions by composite fidelity:")
    for cond, row in report.head(5).iterrows():
        logger.info("  %s  fidelity=%.3f  rss=%.3f  region=%s  off_target=%.1f%%",
                     cond, row['composite_fidelity'], row['rss_score'],
                     row['dominant_region'], row['off_target_fraction'] * 100)

    logger.info("Bottom 5 conditions by composite fidelity:")
    for cond, row in report.tail(5).iterrows():
        logger.info("  %s  fidelity=%.3f  rss=%.3f  region=%s  off_target=%.1f%%",
                     cond, row['composite_fidelity'], row['rss_score'],
                     row['dominant_region'], row['off_target_fraction'] * 100)

    logger.info("Score distributions:")
    for metric in ["composite_fidelity", "rss_score", "on_target_fraction", "off_target_fraction"]:
        vals = report[metric].dropna()
        logger.info("  %s  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                     metric, vals.mean(), vals.std(), vals.min(), vals.max())

    logger.info("Region distribution across conditions:")
    region_counts = report["dominant_region"].value_counts()
    for region, count in region_counts.items():
        logger.info("  %s  %d conditions (%.0f%%)", region, count, count/len(report)*100)


if __name__ == "__main__":
    main()
