"""
Step 0c: Build temporal atlas from Azbukina patterning screen data.

Converts the patterning screen h5ad into the format expected by
05_cellrank2_virtual.py for CellRank 2 temporal projection.

Inputs:
  - data/patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz

Outputs:
  - data/azbukina_temporal_atlas.h5ad (with 'day' column in .obs)

Usage:
  python 00c_build_temporal_atlas.py --inspect-only   # inspect file metadata
  python 00c_build_temporal_atlas.py                  # build atlas
  python 00c_build_temporal_atlas.py --time-col stage --label-col cell_type
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import scanpy as sc

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

# Source file path
SOURCE_GZ = DATA_DIR / "patterning_screen" / "OSMGT_processed_files" / "exp1_processed_8.h5ad.gz"
SOURCE_H5AD = SOURCE_GZ.with_suffix("")  # strip .gz

# Output path
OUTPUT_PATH = DATA_DIR / "azbukina_temporal_atlas.h5ad"

# Expected timepoints for CellRank 2 temporal projection
EXPECTED_TIMEPOINTS = [7, 15, 30, 60, 90, 120]

# Candidate column names for timepoint metadata
TIME_CANDIDATES = [
    "day", "timepoint", "time", "age", "stage",
    "sample_day", "collection_day", "Day", "Timepoint",
    "Time", "Age", "Stage",
]

# Candidate column names for cell type labels (order matters — first match wins)
LABEL_CANDIDATES = [
    "predicted_annot_level_2",
    "annot_level_2",
    "cell_type",
    "CellType",
    "celltype",
]


def decompress_source(gz_path: Path, h5ad_path: Path) -> Path:
    """Decompress .h5ad.gz to .h5ad if the decompressed file doesn't exist.

    Args:
        gz_path: Path to the gzipped h5ad file.
        h5ad_path: Path where the decompressed file should be written.

    Returns:
        Path to the decompressed h5ad file.
    """
    if h5ad_path.exists():
        logger.info("Decompressed file already exists: %s", h5ad_path.name)
        return h5ad_path

    if not gz_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {gz_path}\n"
            f"Run 00b_download_patterning_screen.py first."
        )

    logger.info("Decompressing %s ...", gz_path.name)
    with gzip.open(str(gz_path), "rb") as f_in:
        with open(str(h5ad_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logger.info("Decompressed to %s (%.1f MB)", h5ad_path.name, h5ad_path.stat().st_size / 1e6)
    return h5ad_path


def load_source(h5ad_path: Path) -> sc.AnnData:
    """Load the source h5ad file.

    Args:
        h5ad_path: Path to the h5ad file.

    Returns:
        Loaded AnnData object.
    """
    logger.info("Loading %s ...", h5ad_path.name)
    adata = sc.read_h5ad(str(h5ad_path))
    logger.info("Loaded: %s cells x %s genes", f"{adata.n_obs:,}", f"{adata.n_vars:,}")
    return adata


def inspect_metadata(adata: sc.AnnData) -> None:
    """Log metadata to help identify the right columns.

    Args:
        adata: AnnData object to inspect.
    """
    logger.info("=== .obs columns (%d) ===", len(adata.obs.columns))
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype
        n_unique = adata.obs[col].nunique()
        logger.info("  %-30s  dtype=%-12s  nunique=%d", col, str(dtype), n_unique)

    logger.info("=== .obs dtypes ===")
    for dtype, cols in adata.obs.columns.to_series().groupby(adata.obs.dtypes):
        logger.info("  %s: %s", dtype, list(cols))

    # Show unique values for likely time-related columns
    logger.info("=== Likely time columns (unique values) ===")
    found_any = False
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["day", "time", "age", "stage", "point", "sample"]):
            unique_vals = sorted(adata.obs[col].unique())
            display_vals = unique_vals[:20]
            suffix = f" ... ({len(unique_vals)} total)" if len(unique_vals) > 20 else ""
            logger.info("  %s: %s%s", col, display_vals, suffix)
            found_any = True
    if not found_any:
        logger.info("  (no columns matched time-related keywords)")

    # Show unique values for likely label columns
    logger.info("=== Likely label columns (unique values) ===")
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["type", "label", "annot", "cell", "cluster"]):
            unique_vals = sorted(adata.obs[col].unique())
            display_vals = unique_vals[:30]
            suffix = f" ... ({len(unique_vals)} total)" if len(unique_vals) > 30 else ""
            logger.info("  %s: %s%s", col, display_vals, suffix)

    logger.info("=== .obsm keys ===")
    logger.info("  %s", list(adata.obsm.keys()))

    logger.info("=== .uns keys ===")
    logger.info("  %s", list(adata.uns.keys()) if adata.uns else "(empty)")


def find_time_column(adata: sc.AnnData, user_col: str | None = None) -> str:
    """Find the timepoint column in .obs.

    Args:
        adata: AnnData object.
        user_col: User-specified column name (takes priority).

    Returns:
        Name of the timepoint column.

    Raises:
        ValueError: If no timepoint column is found.
    """
    if user_col is not None:
        if user_col not in adata.obs.columns:
            raise ValueError(
                f"Specified time column '{user_col}' not in .obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        logger.info("Using user-specified time column: %s", user_col)
        return user_col

    for candidate in TIME_CANDIDATES:
        if candidate in adata.obs.columns:
            logger.info("Auto-detected time column: %s", candidate)
            return candidate

    # Fuzzy search: any column containing time-related keywords
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["day", "time", "age", "stage"]):
            logger.info("Auto-detected time column (fuzzy): %s", col)
            return col

    raise ValueError(
        "Could not auto-detect a timepoint column. "
        "Run with --inspect-only to see available columns, "
        "then use --time-col to specify the correct one.\n"
        f"Available columns: {list(adata.obs.columns)}"
    )


def find_label_column(adata: sc.AnnData, user_col: str | None = None) -> str:
    """Find the cell type label column in .obs.

    Args:
        adata: AnnData object.
        user_col: User-specified column name (takes priority).

    Returns:
        Name of the label column.

    Raises:
        ValueError: If no label column is found.
    """
    if user_col is not None:
        if user_col not in adata.obs.columns:
            raise ValueError(
                f"Specified label column '{user_col}' not in .obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        logger.info("Using user-specified label column: %s", user_col)
        return user_col

    for candidate in LABEL_CANDIDATES:
        if candidate in adata.obs.columns:
            logger.info("Auto-detected label column: %s", candidate)
            return candidate

    raise ValueError(
        "Could not auto-detect a cell type label column. "
        "Run with --inspect-only to see available columns, "
        "then use --label-col to specify the correct one.\n"
        f"Available columns: {list(adata.obs.columns)}"
    )


def map_timepoints(adata: sc.AnnData, time_col: str) -> sc.AnnData:
    """Map timepoint metadata to a numeric 'day' column.

    If the time column is already numeric, uses it directly.
    If categorical or string, attempts to extract numeric day values.

    Args:
        adata: AnnData object.
        time_col: Column name containing timepoint metadata.

    Returns:
        AnnData with numeric 'day' column added to .obs.
    """
    import re

    import numpy as np
    import pandas as pd

    raw_values = adata.obs[time_col]
    unique_raw = sorted(raw_values.unique())
    logger.info("Time column '%s' has %d unique values: %s", time_col, len(unique_raw), unique_raw)

    # Try direct numeric conversion
    try:
        numeric_days = pd.to_numeric(raw_values)
        adata.obs["day"] = numeric_days
        logger.info("Time column is numeric — used directly as 'day'.")
        return adata
    except (ValueError, TypeError):
        pass

    # Try extracting numbers from string values (e.g. "Day 7", "d30", "7d")
    day_map = {}
    for val in unique_raw:
        val_str = str(val)
        numbers = re.findall(r"\d+\.?\d*", val_str)
        if numbers:
            day_map[val] = float(numbers[0])
        else:
            logger.warning("Could not extract numeric day from '%s'", val)

    if not day_map:
        raise ValueError(
            f"Could not extract numeric day values from column '{time_col}'. "
            f"Unique values: {unique_raw}"
        )

    logger.info("Mapped timepoints: %s", day_map)
    adata.obs["day"] = raw_values.map(day_map).astype(np.float64)

    n_missing = adata.obs["day"].isna().sum()
    if n_missing > 0:
        logger.warning("%d cells have unmapped timepoints — dropping them.", n_missing)
        adata = adata[~adata.obs["day"].isna()].copy()

    return adata


def validate_atlas(adata: sc.AnnData, label_col: str) -> None:
    """Validate the atlas has the expected structure.

    Args:
        adata: AnnData object with 'day' column.
        label_col: Cell type label column name.
    """
    timepoints = sorted(adata.obs["day"].unique())
    logger.info("Final timepoints: %s", timepoints)

    for tp in timepoints:
        n = (adata.obs["day"] == tp).sum()
        logger.info("  Day %g: %s cells", tp, f"{n:,}")

    # Check overlap with expected timepoints
    expected_set = set(EXPECTED_TIMEPOINTS)
    actual_set = set(timepoints)
    overlap = expected_set & actual_set
    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if overlap:
        logger.info("Timepoints matching expected %s: %s", EXPECTED_TIMEPOINTS, sorted(overlap))
    if missing:
        logger.warning("Missing expected timepoints: %s", sorted(missing))
    if extra:
        logger.info("Additional timepoints (not in expected list): %s", sorted(extra))

    # Label column stats
    n_types = adata.obs[label_col].nunique()
    logger.info("Cell type labels ('%s'): %d unique types", label_col, n_types)

    # Check for PCA
    if "X_pca" in adata.obsm:
        logger.info("X_pca present: shape %s", adata.obsm["X_pca"].shape)
    else:
        logger.info("X_pca not present — will be computed by 05_cellrank2_virtual.py")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build temporal atlas from Azbukina patterning screen data.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect file metadata (columns, dtypes, unique values), do not build atlas.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Column name containing timepoint metadata. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Column name containing cell type labels. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: {OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """Build temporal atlas from patterning screen data."""
    args = parse_args()
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    # --- Decompress if needed ---
    h5ad_path = decompress_source(SOURCE_GZ, SOURCE_H5AD)

    # --- Load ---
    adata = load_source(h5ad_path)

    # --- Inspect-only mode ---
    if args.inspect_only:
        inspect_metadata(adata)
        return

    # --- Find time column and map to numeric 'day' ---
    time_col = find_time_column(adata, user_col=args.time_col)
    adata = map_timepoints(adata, time_col)

    # --- Find label column ---
    label_col = find_label_column(adata, user_col=args.label_col)

    # --- Validate ---
    validate_atlas(adata, label_col)

    # --- Save ---
    logger.info("Saving temporal atlas to %s ...", output_path)
    adata.write_h5ad(str(output_path))
    logger.info(
        "Done. Saved %s cells x %s genes to %s",
        f"{adata.n_obs:,}",
        f"{adata.n_vars:,}",
        output_path.name,
    )


if __name__ == "__main__":
    main()
