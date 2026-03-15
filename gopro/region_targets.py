"""Region targeting system for the GP-BO pipeline.

Provides named brain region profiles, custom target construction, and
region discovery from reference atlases. Used by step 03 (fidelity scoring)
and step 04 (GP-BO objective) to optimize for specific brain regions.

The 9 HNOCA brain regions and their mapping to Braun fetal brain regions
are defined here as the canonical source of truth. Step 03 imports them
rather than defining its own copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from gopro.config import (
    ANNOT_LEVEL_1,
    ANNOT_LEVEL_2,
    ANNOT_REGION,
    ANNOT_LEVEL_3,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical region/label mappings (moved from 03_fidelity_scoring.py)
# ---------------------------------------------------------------------------

# Cell classes considered off-target for brain organoids.
# These are HNOCA level-1 labels that do not correspond to neural lineage.
OFF_TARGET_LEVEL1: set[str] = {
    "PSC",            # Pluripotent stem cells (undifferentiated)
    "MC",             # Mesenchymal cells
    "EC",             # Endothelial cells
    "Microglia",      # Immune — not generated in standard protocols
    "NC Derivatives", # Neural crest derivatives (PNS, not CNS)
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


_STATIC_HNOCA_TO_BRAUN: dict[str, str] = {
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


def _build_dynamic_label_map(
    hnoca_obs: pd.DataFrame,
    braun_obs: pd.DataFrame,
    hnoca_label_col: str = "annot_level_1",
    braun_label_col: str = "CellClass",
    shared_gene_col: str | None = None,
) -> dict[str, str]:
    """Discover HNOCA→Braun label mapping from co-expressed gene profiles.

    For each HNOCA label, builds a region-distribution vector (fraction of
    cells per brain region) and matches it to the Braun label whose region
    distribution is most similar (cosine similarity). This works because
    cell types with the same biological identity tend to cluster in the
    same brain regions.

    Args:
        hnoca_obs: HNOCA reference obs DataFrame. Must contain
            ``hnoca_label_col`` and a region column (``annot_region_rev2``).
        braun_obs: Braun reference obs DataFrame. Must contain
            ``braun_label_col`` and ``SummarizedRegion``.
        hnoca_label_col: Column with HNOCA cell type labels.
        braun_label_col: Column with Braun cell class labels.
        shared_gene_col: Unused, reserved for future gene-expression matching.

    Returns:
        Dict mapping each HNOCA label to the best-matching Braun label.
    """
    hnoca_obs = hnoca_obs.copy()
    braun_obs = braun_obs.copy()

    # Determine region columns
    hnoca_region_col = None
    for col in ["annot_region_rev2", ANNOT_REGION]:
        if col in hnoca_obs.columns:
            hnoca_region_col = col
            break
    if hnoca_region_col is None:
        raise ValueError(
            f"No region column found in HNOCA obs. "
            f"Expected annot_region_rev2 or {ANNOT_REGION}. "
            f"Available: {list(hnoca_obs.columns)}"
        )

    braun_region_col = "SummarizedRegion"
    if braun_region_col not in braun_obs.columns:
        raise ValueError(
            f"Column '{braun_region_col}' not found in Braun obs. "
            f"Available: {list(braun_obs.columns)}"
        )

    # Build region-distribution profiles for each label in both atlases
    # HNOCA: fraction of cells per region for each level-1 label
    hnoca_profiles = (
        hnoca_obs
        .groupby(hnoca_label_col)[hnoca_region_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    # Braun: fraction of cells per region for each CellClass
    braun_profiles = (
        braun_obs
        .groupby(braun_label_col)[braun_region_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    # Align on shared regions (union, fill missing with 0)
    all_regions = sorted(set(hnoca_profiles.columns) | set(braun_profiles.columns))
    hnoca_aligned = hnoca_profiles.reindex(columns=all_regions, fill_value=0.0)
    braun_aligned = braun_profiles.reindex(columns=all_regions, fill_value=0.0)

    # For each HNOCA label, find the Braun label with highest cosine similarity
    label_map: dict[str, str] = {}
    for hnoca_label in hnoca_aligned.index:
        h_vec = hnoca_aligned.loc[hnoca_label].values
        h_norm = np.linalg.norm(h_vec)
        if h_norm == 0:
            continue

        best_braun = None
        best_sim = -1.0
        for braun_label in braun_aligned.index:
            b_vec = braun_aligned.loc[braun_label].values
            b_norm = np.linalg.norm(b_vec)
            if b_norm == 0:
                continue
            sim = float(np.dot(h_vec, b_vec) / (h_norm * b_norm))
            if sim > best_sim:
                best_sim = sim
                best_braun = braun_label

        if best_braun is not None:
            label_map[str(hnoca_label)] = str(best_braun)

    logger.info(
        "Dynamic label map: matched %d/%d HNOCA labels to Braun labels",
        len(label_map), len(hnoca_aligned),
    )
    return label_map


def build_hnoca_to_braun_label_map(
    hnoca_obs: pd.DataFrame | None = None,
    braun_obs: pd.DataFrame | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a mapping from HNOCA level-1 labels to Braun CellClass labels.

    When ``hnoca_obs`` and ``braun_obs`` are provided, mappings are discovered
    dynamically from region co-occurrence patterns. The static map is used as
    fallback for any HNOCA labels not matched dynamically. Optional
    ``overrides`` take highest precedence.

    When called with no arguments, returns the static curated map (backward
    compatible).

    Args:
        hnoca_obs: HNOCA reference obs DataFrame with region and label columns.
            If None, only the static map is used.
        braun_obs: Braun reference obs DataFrame with SummarizedRegion and
            CellClass columns. If None, only the static map is used.
        overrides: Manual label overrides applied on top of everything else.

    Returns:
        Dict mapping HNOCA annot_level_1 values to Braun CellClass values.
    """
    # Start with the static curated map as the base
    label_map = _STATIC_HNOCA_TO_BRAUN.copy()

    # Layer dynamic discovery on top if both reference obs are available
    if hnoca_obs is not None and braun_obs is not None:
        try:
            dynamic = _build_dynamic_label_map(hnoca_obs, braun_obs)
            # Dynamic mappings override static for labels present in both
            label_map.update(dynamic)
            logger.info("Applied %d dynamic label mappings", len(dynamic))
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Dynamic label map discovery failed, using static map: %s", exc,
            )

    # User overrides always win
    if overrides is not None:
        label_map.update(overrides)
        logger.info("Applied %d user label overrides", len(overrides))

    return label_map


# ---------------------------------------------------------------------------
# Named region profiles
# ---------------------------------------------------------------------------

# Named presets for all 9 HNOCA brain regions.
# These describe the regions available for targeting. The actual cell type
# composition profiles are loaded from the reference atlas at runtime.
NAMED_REGION_PROFILES: dict[str, dict] = {
    "dorsal_telencephalon": {
        "display_name": "Dorsal telencephalon",
        "description": "Cortical excitatory neurons, radial glia, intermediate progenitors",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Dorsal telencephalon",
    },
    "ventral_telencephalon": {
        "display_name": "Ventral telencephalon",
        "description": "GABAergic interneurons, medial/lateral ganglionic eminence",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Ventral telencephalon",
    },
    "hypothalamus": {
        "display_name": "Hypothalamus",
        "description": "Hypothalamic neurons, neuroendocrine cells",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Hypothalamus",
    },
    "thalamus": {
        "display_name": "Thalamus",
        "description": "Thalamic relay neurons, reticular nucleus",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Thalamus",
    },
    "dorsal_midbrain": {
        "display_name": "Dorsal midbrain",
        "description": "Superior/inferior colliculus neurons",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Dorsal midbrain",
    },
    "ventral_midbrain": {
        "display_name": "Ventral midbrain",
        "description": "Dopaminergic neurons (substantia nigra, VTA)",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Ventral midbrain",
    },
    "cerebellum": {
        "display_name": "Cerebellum",
        "description": "Purkinje cells, granule cells, cerebellar progenitors",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Cerebellum",
    },
    "pons": {
        "display_name": "Pons",
        "description": "Pontine neurons, hindbrain progenitors",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Pons",
    },
    "medulla": {
        "display_name": "Medulla",
        "description": "Medullary neurons, respiratory/cardiovascular centers",
        "source": "Braun fetal brain / HNOCA",
        "annotation_level": "annot_region_rev2",
        "hnoca_region": "Medulla",
    },
}

# Required keys in each named profile entry
_REQUIRED_PROFILE_KEYS = {"display_name", "description", "source", "annotation_level", "hnoca_region"}


# ---------------------------------------------------------------------------
# Region discovery from reference atlas
# ---------------------------------------------------------------------------

def discover_available_regions(ref_path: Path) -> dict[str, dict]:
    """Scan a reference atlas h5ad and return available regions with metadata.

    Uses anndata in backed mode to avoid loading the full expression matrix.

    Args:
        ref_path: Path to reference atlas h5ad (e.g., hnoca_minimal_for_mapping.h5ad
            or braun-et-al_minimal_for_mapping.h5ad).

    Returns:
        Dict mapping region names to metadata dicts containing:
        - n_cells: number of cells in the region
        - top_cell_types: list of (cell_type, fraction) tuples for top 5
        - annotation_level: which obs column was used for region labels

    Raises:
        FileNotFoundError: If ref_path does not exist.
        ValueError: If no suitable region column is found.
    """
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference atlas not found: {ref_path}")

    adata = ad.read_h5ad(str(ref_path), backed="r")
    obs = adata.obs

    # Detect region column
    region_col = None
    cell_type_col = None
    for candidate_region in [ANNOT_REGION, "annot_region_rev2", "SummarizedRegion"]:
        if candidate_region in obs.columns:
            region_col = candidate_region
            break

    if region_col is None:
        adata.file.close()
        raise ValueError(
            f"No region column found in {ref_path.name}. "
            f"Expected one of: {ANNOT_REGION}, annot_region_rev2, SummarizedRegion. "
            f"Available: {list(obs.columns)}"
        )

    # Detect cell type column for top_cell_types
    for candidate_ct in [ANNOT_LEVEL_3, "annot_level_3_rev2", ANNOT_LEVEL_2,
                         "annot_level_2", "CellClass", "CellType"]:
        if candidate_ct in obs.columns:
            cell_type_col = candidate_ct
            break

    # Extract region metadata
    # Read just the columns we need to avoid loading full obs for large atlases
    region_series = obs[region_col]
    ct_series = obs[cell_type_col] if cell_type_col else None

    regions: dict[str, dict] = {}
    for region_name in region_series.cat.categories if hasattr(region_series, "cat") else region_series.unique():
        if region_name == "Unspecific":
            continue

        mask = region_series == region_name
        n_cells = int(mask.sum())

        top_cell_types = []
        if ct_series is not None:
            ct_counts = ct_series[mask].value_counts(normalize=True)
            for ct_name, frac in ct_counts.head(5).items():
                top_cell_types.append((str(ct_name), round(float(frac), 4)))

        regions[str(region_name)] = {
            "n_cells": n_cells,
            "top_cell_types": top_cell_types,
            "annotation_level": region_col,
        }

    adata.file.close()
    logger.info("Discovered %d regions in %s", len(regions), ref_path.name)
    return regions


def load_region_profile(
    region_name: str,
    ref_path: Path,
    cell_type_level: str = "auto",
) -> pd.Series:
    """Load the cell type composition profile for a named region.

    Computes the cell type frequency distribution within the specified region
    of the reference atlas.

    Args:
        region_name: Name of the region (e.g., "Dorsal telencephalon").
            Can also be a snake_case key from NAMED_REGION_PROFILES.
        ref_path: Path to reference atlas h5ad.
        cell_type_level: Which cell type annotation to use. "auto" tries
            level-3, then level-2, then CellClass.

    Returns:
        Series of cell type fractions (sums to 1.0), indexed by cell type name.

    Raises:
        FileNotFoundError: If ref_path does not exist.
        ValueError: If region_name not found in the atlas.
    """
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference atlas not found: {ref_path}")

    # Resolve snake_case keys to display names
    resolved_name = region_name
    if region_name in NAMED_REGION_PROFILES:
        resolved_name = NAMED_REGION_PROFILES[region_name]["hnoca_region"]

    adata = ad.read_h5ad(str(ref_path), backed="r")
    obs = adata.obs

    # Find region column
    region_col = None
    for candidate in [ANNOT_REGION, "annot_region_rev2", "SummarizedRegion"]:
        if candidate in obs.columns:
            region_col = candidate
            break

    if region_col is None:
        adata.file.close()
        raise ValueError(f"No region column found in {ref_path.name}")

    # Check region exists
    available_regions = obs[region_col].unique().tolist()
    if resolved_name not in available_regions:
        adata.file.close()
        raise ValueError(
            f"Region '{resolved_name}' not found in {ref_path.name}. "
            f"Available: {sorted(available_regions)}"
        )

    # Find cell type column
    if cell_type_level == "auto":
        for candidate in [ANNOT_LEVEL_3, "annot_level_3_rev2", ANNOT_LEVEL_2,
                          "annot_level_2", "CellClass"]:
            if candidate in obs.columns:
                ct_col = candidate
                break
        else:
            adata.file.close()
            raise ValueError(f"No cell type column found in {ref_path.name}")
    else:
        if cell_type_level not in obs.columns:
            adata.file.close()
            raise ValueError(f"Column '{cell_type_level}' not in {ref_path.name}")
        ct_col = cell_type_level

    # Compute profile
    mask = obs[region_col] == resolved_name
    profile = obs.loc[mask, ct_col].value_counts(normalize=True)

    adata.file.close()

    logger.info("Loaded profile for '%s': %d cell types from %d cells",
                resolved_name, len(profile), int(mask.sum()))
    return profile


# ---------------------------------------------------------------------------
# Listing and custom targets
# ---------------------------------------------------------------------------

def list_named_profiles() -> pd.DataFrame:
    """List all built-in region profiles with descriptions.

    Returns:
        DataFrame with columns: name, display_name, description, source.
    """
    rows = []
    for name, info in NAMED_REGION_PROFILES.items():
        rows.append({
            "name": name,
            "display_name": info["display_name"],
            "description": info["description"],
            "source": info["source"],
        })
    return pd.DataFrame(rows)


def build_custom_target(cell_type_fractions: dict[str, float]) -> pd.Series:
    """Create a custom target from user-specified cell type fractions.

    Args:
        cell_type_fractions: Dict mapping cell type names to desired fractions.
            Values must be non-negative and sum to approximately 1.0.

    Returns:
        Series of normalized cell type fractions (sums to exactly 1.0).

    Raises:
        ValueError: If fractions are negative or sum is too far from 1.0.
    """
    if not cell_type_fractions:
        raise ValueError("cell_type_fractions must not be empty")

    series = pd.Series(cell_type_fractions, dtype=float)

    if (series < 0).any():
        neg = series[series < 0].to_dict()
        raise ValueError(f"Fractions must be non-negative, got: {neg}")

    total = series.sum()
    if total <= 0:
        raise ValueError(f"Fractions must sum to a positive number, got {total}")

    if abs(total - 1.0) > 0.05:
        raise ValueError(
            f"Fractions sum to {total:.4f}, expected ~1.0 (tolerance 0.05). "
            f"Normalize your fractions first or adjust values."
        )

    # Normalize to exactly 1.0
    return series / total


def load_target_profile_csv(path: Path) -> pd.Series:
    """Load a target profile from a CSV file.

    The CSV should have two columns: cell_type and fraction (or be a single-row
    DataFrame where columns are cell types and values are fractions).

    Args:
        path: Path to CSV file.

    Returns:
        Series of cell type fractions (sums to 1.0).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If CSV format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Target profile CSV not found: {path}")

    df = pd.read_csv(str(path))

    # Format 1: two columns (cell_type, fraction)
    if df.shape[1] == 2:
        df.columns = ["cell_type", "fraction"]
        series = pd.Series(df["fraction"].values, index=df["cell_type"].values)
    # Format 2: single row with cell types as columns
    elif df.shape[0] == 1:
        series = df.iloc[0]
        if df.columns[0] in ("Unnamed: 0", "index"):
            series = series.iloc[1:]  # drop index column
    # Format 3: index column + fraction columns (like braun_reference_profiles.csv)
    else:
        if df.columns[0] in ("Unnamed: 0", "index"):
            df = df.set_index(df.columns[0])
        # Take the first row as the profile
        series = df.iloc[0]

    series = series.astype(float)
    total = series.sum()
    if total <= 0:
        raise ValueError(f"Profile fractions sum to {total}, expected positive")

    return series / total
