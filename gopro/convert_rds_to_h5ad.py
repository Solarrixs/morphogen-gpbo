"""Convert Seurat RDS files to AnnData h5ad format.

Uses R (via subprocess) with Seurat and SeuratDisk packages.
Auto-installs missing R packages on first run.

R location: /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/Rscript

Usage:
    python gopro/convert_rds_to_h5ad.py data/patterning_screen/OSMGT_processed_files/4_M_vs_sM_21d_clean.rds.gz
    python gopro/convert_rds_to_h5ad.py input.rds.gz --output output.h5ad
    python gopro/convert_rds_to_h5ad.py input.rds.gz --check-only  # inspect without converting
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

# R binary location (not in PATH on this system)
R_FRAMEWORK = Path("/Library/Frameworks/R.framework/Versions")
RSCRIPT = None
for _version_dir in sorted(R_FRAMEWORK.glob("*/Resources/bin/Rscript"), reverse=True):
    RSCRIPT = _version_dir
    break

if RSCRIPT is None:
    # Fallback: try PATH
    RSCRIPT = Path("Rscript")


def _run_r(script: str, timeout: int = 3600) -> str:
    """Run an R script via Rscript and return stdout.

    Args:
        script: R code to execute.
        timeout: Max runtime in seconds.

    Returns:
        stdout from the R process.

    Raises:
        RuntimeError: If R execution fails.
    """
    result = subprocess.run(
        [str(RSCRIPT), "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )
    return result.stdout


def ensure_r_packages() -> None:
    """Install required R packages if missing."""
    check_script = """
    packages <- c("Seurat", "SeuratObject", "SeuratDisk")
    missing <- packages[!sapply(packages, requireNamespace, quietly=TRUE)]
    if (length(missing) > 0) {
        cat("MISSING:", paste(missing, collapse=","), "\\n")
    } else {
        cat("ALL_INSTALLED\\n")
    }
    """
    output = _run_r(check_script)

    if "ALL_INSTALLED" in output:
        logger.info("All required R packages are installed.")
        return

    logger.info("Installing missing R packages (this may take several minutes)...")

    install_script = """
    if (!requireNamespace("remotes", quietly=TRUE)) {
        install.packages("remotes", repos="https://cloud.r-project.org")
    }
    if (!requireNamespace("Seurat", quietly=TRUE)) {
        install.packages("Seurat", repos="https://cloud.r-project.org")
    }
    if (!requireNamespace("SeuratDisk", quietly=TRUE)) {
        remotes::install_github("mojaveazure/seurat-disk", quiet=TRUE)
    }
    cat("INSTALL_COMPLETE\\n")
    """
    output = _run_r(install_script, timeout=1800)

    if "INSTALL_COMPLETE" not in output:
        raise RuntimeError(f"R package installation may have failed:\n{output}")

    logger.info("R packages installed successfully.")


def decompress_rds(rds_gz_path: Path) -> Path:
    """Decompress .rds.gz to .rds if needed.

    Args:
        rds_gz_path: Path to gzipped RDS file.

    Returns:
        Path to decompressed RDS file.
    """
    if rds_gz_path.suffix != ".gz":
        return rds_gz_path

    rds_path = rds_gz_path.with_suffix("")
    if rds_path.exists():
        logger.info("Decompressed file exists: %s", rds_path.name)
        return rds_path

    logger.info("Decompressing %s (%.1f GB)...", rds_gz_path.name,
                rds_gz_path.stat().st_size / 1e9)

    with gzip.open(str(rds_gz_path), "rb") as f_in:
        with open(str(rds_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Decompressed to %s (%.1f GB)", rds_path.name,
                rds_path.stat().st_size / 1e9)
    return rds_path


def inspect_rds(rds_path: Path) -> str:
    """Inspect an RDS file's structure without converting.

    Args:
        rds_path: Path to RDS file.

    Returns:
        String summary of the object's metadata.
    """
    script = f"""
    obj <- readRDS("{rds_path}")
    cat("Class:", class(obj), "\\n")
    if (inherits(obj, "Seurat")) {{
        cat("Cells:", ncol(obj), "\\n")
        cat("Genes:", nrow(obj), "\\n")
        cat("Assays:", paste(names(obj@assays), collapse=", "), "\\n")
        cat("Metadata columns:", paste(colnames(obj@meta.data), collapse=", "), "\\n")
        cat("\\nMetadata head:\\n")
        print(head(obj@meta.data, 3))
        cat("\\nUnique values for key columns:\\n")
        for (col in colnames(obj@meta.data)) {{
            n <- length(unique(obj@meta.data[[col]]))
            if (n < 100) {{
                cat(col, "(", n, "unique):", paste(head(unique(obj@meta.data[[col]]), 20), collapse=", "), "\\n")
            }} else {{
                cat(col, "(", n, "unique)\\n")
            }}
        }}
    }} else {{
        cat("Not a Seurat object. str():\\n")
        str(obj, max.level=2)
    }}
    """
    return _run_r(script, timeout=600)


def convert_rds_to_h5ad(rds_path: Path, h5ad_path: Path) -> Path:
    """Convert a Seurat RDS file to h5ad format.

    Uses SeuratDisk as intermediate (RDS -> h5seurat -> h5ad).

    Args:
        rds_path: Path to input RDS file.
        h5ad_path: Path for output h5ad file.

    Returns:
        Path to the created h5ad file.
    """
    h5seurat_path = h5ad_path.with_suffix(".h5seurat")

    script = f"""
    library(Seurat)
    library(SeuratDisk)

    cat("Loading RDS:", "{rds_path}", "\\n")
    obj <- readRDS("{rds_path}")
    cat("Loaded:", ncol(obj), "cells x", nrow(obj), "genes\\n")

    # Ensure default assay has counts
    if (!"counts" %in% names(obj@assays[[DefaultAssay(obj)]]@layers) &&
        length(obj@assays[[DefaultAssay(obj)]]@counts) > 0) {{
        cat("Counts matrix found in default assay\\n")
    }}

    # Save as h5seurat (intermediate format)
    cat("Saving h5Seurat to:", "{h5seurat_path}", "\\n")
    SaveH5Seurat(obj, filename = "{h5seurat_path}", overwrite = TRUE)

    # Convert h5seurat to h5ad
    cat("Converting to h5ad:", "{h5ad_path}", "\\n")
    Convert("{h5seurat_path}", dest = "h5ad", overwrite = TRUE)

    cat("CONVERSION_COMPLETE\\n")
    """

    logger.info("Converting %s -> %s ...", rds_path.name, h5ad_path.name)
    output = _run_r(script, timeout=3600)

    if "CONVERSION_COMPLETE" not in output:
        raise RuntimeError(f"Conversion may have failed:\n{output}")

    # Clean up intermediate h5seurat
    if h5seurat_path.exists():
        h5seurat_path.unlink()
        logger.info("Cleaned up intermediate %s", h5seurat_path.name)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Expected output not found: {h5ad_path}")

    logger.info("Conversion complete: %s (%.1f GB)",
                h5ad_path.name, h5ad_path.stat().st_size / 1e9)
    return h5ad_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Seurat RDS files to AnnData h5ad format.",
    )
    parser.add_argument("input", type=str,
                        help="Path to input RDS or RDS.gz file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output h5ad path (default: same name with .h5ad extension)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only inspect the RDS file, don't convert")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip R package installation check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    logger.info("R binary: %s", RSCRIPT)

    # Ensure R packages
    if not args.skip_install:
        ensure_r_packages()

    # Decompress if needed
    rds_path = decompress_rds(input_path)

    # Inspect-only mode
    if args.check_only:
        logger.info("=== Inspecting %s ===", rds_path.name)
        output = inspect_rds(rds_path)
        print(output)
        return

    # Determine output path
    if args.output:
        h5ad_path = Path(args.output)
    else:
        # Strip .rds.gz or .rds, add .h5ad
        stem = input_path.name
        for suffix in [".rds.gz", ".rds"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        h5ad_path = DATA_DIR / f"{stem}.h5ad"

    # Convert
    convert_rds_to_h5ad(rds_path, h5ad_path)


if __name__ == "__main__":
    main()
