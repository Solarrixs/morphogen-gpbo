"""Step 3: Score organoid cell fidelity against Braun fetal brain reference.

BrainSTEM-inspired two-tier fidelity scoring:
  Tier 1: Use HNOCA-transferred region labels to identify brain region identity
  Tier 2: Compare cell type composition to Braun fetal brain reference by region

The Braun fetal brain reference (~1.65M cells, 11 GB) is loaded in backed mode
to extract region-level cell type composition profiles. These serve as the
ground truth "ideal" composition vectors. Each organoid condition is scored by
cosine similarity of its cell type composition to the fetal reference.

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
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cosine as cosine_distance

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"

# HNOCA annotation column names (must match step 02 output)
ANNOT_LEVEL_1 = "annot_level_1"
ANNOT_LEVEL_2 = "annot_level_2"
ANNOT_REGION = "annot_region_rev2"
ANNOT_LEVEL_3 = "annot_level_3_rev2"

# Cell classes considered off-target for brain organoids.
# These are HNOCA level-1 labels that do not correspond to neural lineage.
OFF_TARGET_LEVEL1: set[str] = {
    "PSC",            # Pluripotent stem cells (undifferentiated)
    "MC",             # Mesenchymal cells
    "EC",             # Endothelial cells
    "Microglia",      # Immune — not generated in standard protocols
    "NC Derivatives", # Neural crest derivatives (PNS, not CNS)
}

# Braun fetal brain CellClass values considered neural
BRAUN_NEURAL_CLASSES: set[str] = {
    "Neuron",
    "Neuroblast",
    "Neuronal IPC",
    "Radial glia",
    "Glioblast",
    "Oligo",
}

# Mapping from HNOCA annot_region_rev2 to Braun SummarizedRegion.
# Used to align organoid region labels with fetal reference regions.
HNOCA_TO_BRAUN_REGION: dict[str, str] = {
    "Dorsal telencephalon": "Dorsal telencephalon",
    "Ventral telencephalon": "Ventral telencephalon",
    "Hypothalamus": "Hypothalamus",
    "Thalamus": "Thalamus",
    "Dorsal midbrain": "Dorsal midbrain",
    "Ventral midbrain": "Ventral midbrain",
    "Cerebellum": "Cerebellum",
    "Pons": "Pons",
    "Medulla": "Medulla",
    # "Unspecific" has no direct Braun counterpart — handled separately
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
        print(f"  Loading cached Braun profiles from {cache_path.name}")
        return pd.read_csv(str(cache_path), index_col=0)

    print("  Loading Braun fetal brain reference (backed mode)...")
    braun = ad.read_h5ad(str(braun_path), backed="r")
    print(f"  Braun reference: {braun.shape[0]:,} cells x {braun.shape[1]:,} genes")

    # Extract only the metadata we need (avoids loading expression data)
    obs_df = braun.obs[["SummarizedRegion", "CellClass", "IsNeural"]].copy()
    braun.file.close()

    # Filter to non-doublet neural cells for cleaner profiles
    obs_df = obs_df[obs_df["IsNeural"] == True]  # noqa: E712
    print(f"  Neural cells: {len(obs_df):,}")

    # Compute cell class fractions per region
    profiles = (
        obs_df
        .groupby("SummarizedRegion")["CellClass"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    print(f"  Extracted profiles: {profiles.shape[0]} regions x {profiles.shape[1]} cell classes")
    for region in profiles.index:
        n_cells = (obs_df["SummarizedRegion"] == region).sum()
        top_class = profiles.loc[region].idxmax()
        top_frac = profiles.loc[region].max()
        print(f"    {region}: {n_cells:,} cells, top={top_class} ({top_frac:.1%})")

    # Cache for future runs
    if cache_path is not None:
        profiles.to_csv(str(cache_path))
        print(f"  Cached profiles to {cache_path.name}")

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
        print(f"  Loading cached Braun CellType profiles from {cache_path.name}")
        return pd.read_csv(str(cache_path), index_col=0)

    print("  Loading Braun fetal brain reference for CellType profiles (backed mode)...")
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
        print(f"  Cached CellType profiles to {cache_path.name}")

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


def compute_condition_region_fractions(
    obs: pd.DataFrame,
    condition_key: str = "condition",
    region_key: str = "predicted_annot_region_rev2",
) -> pd.DataFrame:
    """Compute brain region fraction vectors per condition.

    Args:
        obs: Cell metadata with condition and predicted region columns.
        condition_key: Column identifying experimental conditions.
        region_key: Column with predicted brain region labels.

    Returns:
        DataFrame: rows=conditions, columns=regions, values=fractions.
    """
    fractions = (
        obs
        .groupby(condition_key)[region_key]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    return fractions


# ---------------------------------------------------------------------------
# Fidelity scoring functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, handling zero vectors.

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


def shannon_entropy(fractions: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        fractions: Array of non-negative values summing to ~1.

    Returns:
        Shannon entropy in bits. Higher = more diverse.
    """
    p = fractions[fractions > 0]
    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(fractions: np.ndarray) -> float:
    """Compute Shannon entropy normalized to [0, 1].

    Normalized by log2(n_nonzero) so that maximum diversity = 1.0.

    Args:
        fractions: Array of non-negative values summing to ~1.

    Returns:
        Normalized entropy in [0, 1].
    """
    p = fractions[fractions > 0]
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log2(p))
    h_max = np.log2(len(p))
    return float(h / h_max)


def compute_rss(
    condition_vec: pd.Series,
    reference_profiles: pd.DataFrame,
) -> tuple[str, float]:
    """Compute Reference Similarity Spectrum: find best-matching fetal region.

    Computes cosine similarity between the condition's cell type composition
    and each fetal brain region's composition. Returns the best-matching
    region and its similarity score.

    Args:
        condition_vec: Cell type fractions for one condition (Series, index=cell types).
        reference_profiles: Fetal brain composition profiles
            (rows=regions, columns=cell classes).

    Returns:
        Tuple of (best_matching_region, cosine_similarity_score).
    """
    # Align cell types: use union of both label sets, fill missing with 0
    all_labels = sorted(set(condition_vec.index) | set(reference_profiles.columns))
    cond_aligned = np.array([condition_vec.get(l, 0.0) for l in all_labels])

    best_region = "none"
    best_sim = 0.0

    for region in reference_profiles.index:
        ref_aligned = np.array([reference_profiles.loc[region].get(l, 0.0) for l in all_labels])
        sim = cosine_similarity(cond_aligned, ref_aligned)
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


def compute_composite_fidelity(
    rss_score: float,
    on_target_frac: float,
    off_target_frac: float,
    norm_entropy: float,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute a single composite fidelity score in [0, 1].

    Combines four sub-scores with configurable weights:
    - RSS (cosine similarity to fetal brain): higher is better
    - On-target fraction: higher is better
    - Off-target fraction: lower is better (inverted)
    - Normalized entropy: moderate is best (penalize extremes)

    Args:
        rss_score: Cosine similarity to best-matching fetal region [0, 1].
        on_target_frac: Fraction of cells in dominant region [0, 1].
        off_target_frac: Fraction of non-neural cells [0, 1].
        norm_entropy: Normalized Shannon entropy [0, 1].
        weights: Dict with keys 'rss', 'on_target', 'off_target', 'entropy'.
            Defaults to equal weighting.

    Returns:
        Composite fidelity score in [0, 1].
    """
    if weights is None:
        weights = {
            "rss": 0.35,
            "on_target": 0.25,
            "off_target": 0.25,
            "entropy": 0.15,
        }

    # Handle NaN inputs gracefully
    rss_score = rss_score if not np.isnan(rss_score) else 0.0
    on_target_frac = on_target_frac if not np.isnan(on_target_frac) else 0.0
    off_target_frac = off_target_frac if not np.isnan(off_target_frac) else 1.0
    norm_entropy = norm_entropy if not np.isnan(norm_entropy) else 0.0

    # Entropy contribution: penalize both too low (monoculture) and too high
    # (disorganized). Optimal is moderate diversity (~0.4-0.7).
    # Use a Gaussian-like penalty centered at 0.55.
    entropy_score = np.exp(-((norm_entropy - 0.55) ** 2) / (2 * 0.2 ** 2))

    score = (
        weights["rss"] * rss_score
        + weights["on_target"] * on_target_frac
        + weights["off_target"] * (1.0 - off_target_frac)
        + weights["entropy"] * entropy_score
    )

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-condition fidelity report
# ---------------------------------------------------------------------------

def score_all_conditions(
    query_adata: sc.AnnData,
    braun_profiles: pd.DataFrame,
    condition_key: str = "condition",
) -> pd.DataFrame:
    """Compute full fidelity report for all conditions in the query data.

    For each condition, computes:
    - n_cells: number of cells
    - dominant_region: most common brain region (HNOCA label)
    - on_target_fraction: fraction of cells in dominant region
    - off_target_fraction: fraction of non-neural cells
    - shannon_entropy: cell type diversity (bits)
    - normalized_entropy: entropy normalized to [0, 1]
    - rss_best_region: best-matching Braun fetal brain region
    - rss_score: cosine similarity to best fetal region
    - composite_fidelity: weighted combination of all scores [0, 1]

    Args:
        query_adata: Mapped AnnData with HNOCA-transferred labels in obs.
        braun_profiles: Fetal brain composition profiles (from
            extract_braun_region_profiles).
        condition_key: Column identifying experimental conditions.

    Returns:
        DataFrame with one row per condition, sorted by composite_fidelity
        descending.
    """
    obs = query_adata.obs
    pred_level1 = f"predicted_{ANNOT_LEVEL_1}"
    pred_level2 = f"predicted_{ANNOT_LEVEL_2}"
    pred_region = f"predicted_{ANNOT_REGION}"

    conditions = obs[condition_key].unique()
    print(f"  Scoring {len(conditions)} conditions...")

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
        h_norm = normalized_entropy(frac_array)

        # RSS: cosine similarity to fetal brain region profiles
        # Map the condition's level-1 composition to Braun CellClass space
        level1_fracs = subset[pred_level1].value_counts(normalize=True)
        rss_region, rss_score = compute_rss(level1_fracs, braun_profiles)

        # Composite fidelity
        fidelity = compute_composite_fidelity(
            rss_score=rss_score,
            on_target_frac=on_target,
            off_target_frac=off_target,
            norm_entropy=h_norm,
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
            "composite_fidelity": round(fidelity, 4),
        })

    report = (
        pd.DataFrame(results)
        .set_index("condition")
        .sort_values("composite_fidelity", ascending=False)
    )

    return report


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

def build_hnoca_to_braun_label_map() -> dict[str, str]:
    """Build a mapping from HNOCA level-1 labels to Braun CellClass labels.

    The two atlases use different cell type naming schemes. This mapping
    allows cosine similarity comparison between organoid (HNOCA-labeled)
    and fetal (Braun-labeled) composition vectors.

    Returns:
        Dict mapping HNOCA annot_level_1 values to Braun CellClass values.
    """
    return {
        # HNOCA level 1  →  Braun CellClass
        "Neuron": "Neuron",
        "NPC": "Radial glia",         # Neural progenitors ~ radial glia
        "IP": "Neuronal IPC",         # Intermediate progenitors
        "Neuroepithelium": "Radial glia",
        "Glioblast": "Glioblast",
        "Astrocyte": "Glioblast",     # Braun doesn't separate astrocytes
        "OPC": "Oligo",
        "CP": "Neuron",              # Choroid plexus — neural origin
        "NC Derivatives": "Neural crest",
        "MC": "Fibroblast",          # Mesenchymal ~ fibroblast in Braun
        "EC": "Vascular",
        "Microglia": "Immune",
        "PSC": "Radial glia",        # Map PSC to closest, but flagged off-target
    }


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

def main() -> None:
    """Run the full fidelity scoring pipeline."""
    start = time.time()

    # -----------------------------------------------------------------------
    # Check prerequisites
    # -----------------------------------------------------------------------
    braun_path = DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
    mapped_path = DATA_DIR / "amin_kelley_mapped.h5ad"

    for path, name, hint in [
        (braun_path, "Braun fetal brain reference", "Run: python 00_zenodo_download.py"),
        (mapped_path, "Mapped query data", "Run step 02 first: python 02_map_to_hnoca.py"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            print(hint)
            exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Extract Braun fetal brain reference profiles
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Extract Braun fetal brain reference profiles")
    print("=" * 60)

    braun_profiles = extract_braun_region_profiles(
        braun_path,
        cache_path=DATA_DIR / "braun_reference_profiles.csv",
    )

    # Also extract detailed CellType profiles for reporting
    braun_celltype_profiles = extract_braun_celltype_profiles(
        braun_path,
        cache_path=DATA_DIR / "braun_reference_celltype_profiles.csv",
    )

    # -----------------------------------------------------------------------
    # Step 2: Load mapped query data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Load mapped query data")
    print("=" * 60)

    print("  Loading mapped query data...")
    query = sc.read_h5ad(str(mapped_path))
    print(f"  Query: {query.n_obs:,} cells x {query.n_vars:,} genes")

    # Verify expected columns exist
    pred_level1 = f"predicted_{ANNOT_LEVEL_1}"
    pred_level2 = f"predicted_{ANNOT_LEVEL_2}"
    pred_region = f"predicted_{ANNOT_REGION}"

    required_cols = [pred_level1, pred_level2, pred_region]
    missing = [c for c in required_cols if c not in query.obs.columns]
    if missing:
        print(f"ERROR: Missing predicted label columns: {missing}")
        print("These should have been added by step 02 (map_to_hnoca.py).")
        print(f"Available obs columns: {list(query.obs.columns)}")
        exit(1)

    # Detect condition column
    condition_key = "condition"
    if condition_key not in query.obs.columns:
        # Try alternative names
        for alt in ["sample", "protocol", "well", "batch"]:
            if alt in query.obs.columns:
                condition_key = alt
                break
        else:
            print(f"ERROR: No condition column found. Available: {list(query.obs.columns)}")
            exit(1)
    print(f"  Using condition column: '{condition_key}' "
          f"({query.obs[condition_key].nunique()} unique conditions)")

    # -----------------------------------------------------------------------
    # Step 3: Build label alignment map
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Align HNOCA labels to Braun CellClass space")
    print("=" * 60)

    label_map = build_hnoca_to_braun_label_map()
    print("  HNOCA → Braun label mapping:")
    for hnoca, braun in sorted(label_map.items()):
        print(f"    {hnoca:20s} → {braun}")

    # Re-key per-condition level-1 compositions into Braun space for RSS
    # This is done inside score_all_conditions via compute_rss

    # -----------------------------------------------------------------------
    # Step 4: Score all conditions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Compute fidelity scores")
    print("=" * 60)

    report = score_all_conditions(
        query_adata=query,
        braun_profiles=braun_profiles,
        condition_key=condition_key,
    )

    # -----------------------------------------------------------------------
    # Step 5: Assign cell-level fidelity scores
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Assign cell-level fidelity scores")
    print("=" * 60)

    query = assign_cell_level_fidelity(query, report, condition_key=condition_key)
    print(f"  Added fidelity columns to {query.n_obs:,} cells")

    # -----------------------------------------------------------------------
    # Step 6: Save outputs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Save outputs")
    print("=" * 60)

    # Save fidelity report
    report_path = DATA_DIR / "fidelity_report.csv"
    report.to_csv(str(report_path))
    print(f"  Fidelity report → {report_path}")

    # Save annotated AnnData
    output_path = DATA_DIR / "amin_kelley_fidelity.h5ad"
    query.write(str(output_path), compression="gzip")
    print(f"  Annotated data  → {output_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("FIDELITY SCORING SUMMARY")
    print("=" * 60)
    print(f"  Cells scored:     {query.n_obs:,}")
    print(f"  Conditions:       {len(report)}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print()

    # Top and bottom conditions
    print("  Top 5 conditions by composite fidelity:")
    for cond, row in report.head(5).iterrows():
        print(f"    {cond:30s}  fidelity={row['composite_fidelity']:.3f}  "
              f"rss={row['rss_score']:.3f}  region={row['dominant_region']}  "
              f"off_target={row['off_target_fraction']:.1%}")

    print()
    print("  Bottom 5 conditions by composite fidelity:")
    for cond, row in report.tail(5).iterrows():
        print(f"    {cond:30s}  fidelity={row['composite_fidelity']:.3f}  "
              f"rss={row['rss_score']:.3f}  region={row['dominant_region']}  "
              f"off_target={row['off_target_fraction']:.1%}")

    print()
    print("  Score distributions:")
    for metric in ["composite_fidelity", "rss_score", "on_target_fraction", "off_target_fraction"]:
        vals = report[metric].dropna()
        print(f"    {metric:25s}  mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")

    print()
    print("  Region distribution across conditions:")
    region_counts = report["dominant_region"].value_counts()
    for region, count in region_counts.items():
        print(f"    {region:30s}  {count} conditions ({count/len(report)*100:.0f}%)")


if __name__ == "__main__":
    main()
