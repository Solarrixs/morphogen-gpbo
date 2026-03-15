"""Inter-step data validation for the GP-BO pipeline.

Validates data schemas and constraints between pipeline steps to catch
errors early, before expensive computations run.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from gopro.config import (
    MORPHOGEN_COLUMNS,
    ANNOT_LEVEL_1,
    ANNOT_LEVEL_2,
    ANNOT_REGION,
    ANNOT_LEVEL_3,
    get_logger,
)

logger = get_logger(__name__)


class ValidationError(ValueError):
    """Raised when pipeline data fails schema validation."""


def validate_mapped_adata(
    path: str | Path,
    condition_key: str = "condition",
    require_counts_layer: bool = False,
) -> list[str]:
    """Check mapped h5ad has predicted_annot_level_* + condition columns.

    Uses backed='r' mode to avoid loading expression data into memory.

    Args:
        path: Path to mapped h5ad file.
        condition_key: Expected condition column name.
        require_counts_layer: If True, raise on missing counts layer.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        ValidationError: On missing required columns or file not found.
    """
    path = Path(path)
    warnings: list[str] = []

    if not path.exists():
        raise ValidationError(f"Mapped h5ad not found: {path}")

    adata = ad.read_h5ad(str(path), backed="r")
    obs_columns = list(adata.obs.columns)

    # Check predicted annotation columns
    required_predicted = [
        f"predicted_{ANNOT_LEVEL_1}",
        f"predicted_{ANNOT_LEVEL_2}",
        f"predicted_{ANNOT_REGION}",
    ]
    missing = [c for c in required_predicted if c not in obs_columns]
    if missing:
        adata.file.close()
        raise ValidationError(
            f"Missing predicted label columns: {missing}. "
            f"These should have been added by step 02 (map_to_hnoca.py). "
            f"Available: {obs_columns}"
        )

    # Check condition column
    if condition_key not in obs_columns:
        adata.file.close()
        raise ValidationError(
            f"Condition column '{condition_key}' not found. "
            f"Available: {obs_columns}"
        )

    # Check counts layer (warning only by default)
    if require_counts_layer and "counts" not in adata.layers:
        adata.file.close()
        raise ValidationError("Required 'counts' layer not found in mapped h5ad")
    elif "counts" not in adata.layers:
        warnings.append("No 'counts' layer found — some downstream steps may need it")

    adata.file.close()

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Mapped h5ad validation passed: %s", path.name)

    return warnings


def validate_training_csvs(
    fractions_path: str | Path,
    morphogen_path: str | Path,
    check_morphogen_columns: bool = True,
) -> list[str]:
    """Check CSV alignment: index overlap, no NaN, rows sum to ~1.0.

    Args:
        fractions_path: Path to cell type fractions CSV.
        morphogen_path: Path to morphogen concentration matrix CSV.
        check_morphogen_columns: If True, warn on unrecognized morphogen columns.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        ValidationError: On zero overlap, NaN values, or missing files.
    """
    fractions_path = Path(fractions_path)
    morphogen_path = Path(morphogen_path)
    warnings: list[str] = []

    if not fractions_path.exists():
        raise ValidationError(f"Fractions CSV not found: {fractions_path}")
    if not morphogen_path.exists():
        raise ValidationError(f"Morphogen CSV not found: {morphogen_path}")

    fractions = pd.read_csv(str(fractions_path), index_col=0)
    morphogens = pd.read_csv(str(morphogen_path), index_col=0)

    # Check index overlap
    overlap = fractions.index.intersection(morphogens.index)
    if len(overlap) == 0:
        raise ValidationError(
            f"Zero overlap between fractions ({list(fractions.index[:5])}) "
            f"and morphogens ({list(morphogens.index[:5])})"
        )
    if len(overlap) < len(fractions):
        warnings.append(
            f"{len(fractions) - len(overlap)} conditions in fractions "
            f"not found in morphogens"
        )

    # Check for NaN in fractions
    if fractions.isna().any().any():
        n_nan = int(fractions.isna().sum().sum())
        raise ValidationError(
            f"Fractions CSV contains {n_nan} NaN values"
        )

    # Check row sums
    row_sums = fractions.sum(axis=1)
    bad_rows = row_sums[~np.isclose(row_sums, 1.0, atol=0.05)]
    if len(bad_rows) > 0:
        raise ValidationError(
            f"{len(bad_rows)} rows don't sum to ~1.0. "
            f"Range: [{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )

    # Check morphogen columns
    if check_morphogen_columns:
        known = set(MORPHOGEN_COLUMNS)
        unknown = [c for c in morphogens.columns if c not in known and c != "fidelity"]
        if unknown:
            warnings.append(f"Unrecognized morphogen columns: {unknown}")

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Training CSV validation passed: %d conditions", len(overlap))

    return warnings


def validate_temporal_atlas(
    path: str | Path,
    time_key: str = "day",
    expected_timepoints: list[int] | None = None,
) -> list[str]:
    """Check time_key exists, is numeric, has >= 2 timepoints.

    Uses backed='r' mode to avoid loading expression data.

    Args:
        path: Path to temporal atlas h5ad file.
        time_key: Column in obs containing timepoint information.
        expected_timepoints: If provided, check these timepoints exist.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        ValidationError: On missing time_key or insufficient timepoints.
    """
    path = Path(path)
    warnings: list[str] = []

    if not path.exists():
        raise ValidationError(f"Temporal atlas not found: {path}")

    adata = ad.read_h5ad(str(path), backed="r")

    if time_key not in adata.obs.columns:
        adata.file.close()
        raise ValidationError(
            f"Time key '{time_key}' not found in atlas obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Check numeric
    time_vals = adata.obs[time_key]
    try:
        time_numeric = pd.to_numeric(time_vals)
    except (ValueError, TypeError):
        adata.file.close()
        raise ValidationError(
            f"Time key '{time_key}' is not numeric. "
            f"Sample values: {list(time_vals.head())}"
        )

    timepoints = sorted(time_numeric.unique())
    if len(timepoints) < 2:
        adata.file.close()
        raise ValidationError(
            f"Need >= 2 timepoints, found {len(timepoints)}: {timepoints}"
        )

    if expected_timepoints is not None:
        missing_tps = [t for t in expected_timepoints if t not in timepoints]
        if missing_tps:
            warnings.append(f"Expected timepoints not found: {missing_tps}")

    adata.file.close()

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Temporal atlas validation passed: %d timepoints", len(timepoints))

    return warnings


def validate_fidelity_report(
    path: str | Path,
) -> list[str]:
    """Check required columns and score ranges in fidelity report.

    Args:
        path: Path to fidelity report CSV.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        ValidationError: On missing required columns or invalid scores.
    """
    path = Path(path)
    warnings: list[str] = []

    if not path.exists():
        raise ValidationError(f"Fidelity report not found: {path}")

    report = pd.read_csv(str(path), index_col=0)

    required_cols = ["composite_fidelity", "rss_score", "dominant_region"]
    missing = [c for c in required_cols if c not in report.columns]
    if missing:
        raise ValidationError(
            f"Missing required columns in fidelity report: {missing}. "
            f"Available: {list(report.columns)}"
        )

    # Check score ranges
    for score_col in ["composite_fidelity", "rss_score"]:
        if score_col in report.columns:
            vals = report[score_col].dropna()
            if len(vals) > 0:
                if vals.min() < -0.01 or vals.max() > 1.01:
                    warnings.append(
                        f"{score_col} outside [0, 1]: "
                        f"[{vals.min():.4f}, {vals.max():.4f}]"
                    )

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Fidelity report validation passed: %d conditions", len(report))

    return warnings
