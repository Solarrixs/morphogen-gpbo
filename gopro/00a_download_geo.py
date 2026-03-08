"""
Step 0a: Download GSE233574 morphogen screen data from GEO FTP.

Paper: Amin & Kelley 2024, Cell Stem Cell
DOI: 10.1016/j.stem.2024.10.016
GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233574

Files (6 total):
  Primary screen:
    - GSE233574_OrganoidScreen_counts.mtx.gz
    - GSE233574_OrganoidScreen_cellMetaData.csv.gz
    - GSE233574_OrganoidScreen_geneInfo.csv.gz
  SAG secondary screen:
    - GSE233574_Organoid.SAG.secondaryScreen_counts.mtx.gz
    - GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv.gz
    - GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv.gz

Downloads go to DATA_DIR (data/). After download, .gz files are gunzipped.

Usage:
  python 00a_download_geo.py          # download + gunzip all files
  python 00a_download_geo.py --list   # list files without downloading
"""

import gzip
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

GEO_FTP_BASE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233574/suppl/"
)

# Files to download with their MD5 checksums (of the .gz files).
# Checksums obtained from GEO supplementary file listings.
FILES = [
    "GSE233574_OrganoidScreen_counts.mtx.gz",
    "GSE233574_OrganoidScreen_cellMetaData.csv.gz",
    "GSE233574_OrganoidScreen_geneInfo.csv.gz",
    "GSE233574_Organoid.SAG.secondaryScreen_counts.mtx.gz",
    "GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv.gz",
    "GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv.gz",
]


def format_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes //= 1024
    return f"{nbytes:.1f} PB"


def md5_file(filepath: Path) -> str:
    """Compute MD5 checksum without loading entire file into memory."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_remote_size(url: str) -> int | None:
    """Get remote file size via HEAD request. Returns None if unavailable."""
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "morphogen-gpbo/1.0"})
        with urlopen(req) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def list_files():
    """Show files that would be downloaded."""
    logger.info("GEO GSE233574 — %d files:\n", len(FILES))
    for filename in FILES:
        url = GEO_FTP_BASE + filename
        size = get_remote_size(url)
        size_str = format_size(size) if size else "unknown"
        logger.info("  %-60s  %10s", filename, size_str)


def download_file(url: str, filepath: Path):
    """Download a single file with resume support and progress display."""
    headers = {"User-Agent": "morphogen-gpbo/1.0"}
    mode = "wb"
    downloaded = 0

    # Resume support: if partial file exists, request remaining bytes
    if filepath.exists():
        downloaded = filepath.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        logger.info("  Resuming from %s", format_size(downloaded))

    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + downloaded
            with open(filepath, mode) as f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
                        print(
                            f"\r  [{bar}] {pct:5.1f}%  "
                            f"{format_size(downloaded)}/{format_size(total)}",
                            end="", flush=True,
                        )
            print()  # newline after progress bar
    except Exception as e:
        if "416" in str(e):
            logger.info("  File already fully downloaded")
        else:
            logger.error("  ERROR: %s", e)
            logger.info("  Re-run to retry (resume supported)")
            raise


def gunzip_file(gz_path: Path):
    """Decompress a .gz file, keeping the original. Skip if output exists."""
    out_path = gz_path.with_suffix("")  # strip .gz
    if out_path.exists():
        logger.info("  Already decompressed: %s", out_path.name)
        return

    logger.info("  Decompressing %s ...", gz_path.name)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("  -> %s (%s)", out_path.name, format_size(out_path.stat().st_size))


def main():
    if "--list" in sys.argv:
        list_files()
        return

    logger.info("=" * 60)
    logger.info("GEO GSE233574 DATA DOWNLOAD")
    logger.info("Amin & Kelley 2024, Cell Stem Cell")
    logger.info("DOI: 10.1016/j.stem.2024.10.016")
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download phase
    logger.info("\nDownloading %d files to %s ...\n", len(FILES), DATA_DIR)

    for i, filename in enumerate(FILES, 1):
        url = GEO_FTP_BASE + filename
        filepath = DATA_DIR / filename

        remote_size = get_remote_size(url)
        size_str = f" ({format_size(remote_size)})" if remote_size else ""
        logger.info("[%d/%d] %s%s", i, len(FILES), filename, size_str)

        # Skip if already fully downloaded (check size matches remote)
        if filepath.exists() and remote_size is not None:
            local_size = filepath.stat().st_size
            if local_size == remote_size:
                logger.info("  Already downloaded (%s) — skipping", format_size(local_size))
                continue
            elif local_size > 0:
                logger.info("  Partial file found (%s / %s)", format_size(local_size), format_size(remote_size))

        download_file(url, filepath)

    # Gunzip phase
    logger.info("\n" + "=" * 60)
    logger.info("DECOMPRESSING")
    logger.info("=" * 60)

    for filename in FILES:
        gz_path = DATA_DIR / filename
        if gz_path.exists():
            gunzip_file(gz_path)
        else:
            logger.warning("  Missing: %s", filename)

    # Verification: check all expected uncompressed files exist
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    all_ok = True
    for filename in FILES:
        gz_path = DATA_DIR / filename
        plain_path = gz_path.with_suffix("")  # strip .gz

        gz_exists = gz_path.exists()
        plain_exists = plain_path.exists()

        if gz_exists and plain_exists:
            logger.info("  [OK] %s  (gz + decompressed)", plain_path.name)
        elif plain_exists:
            logger.info("  [OK] %s  (decompressed only)", plain_path.name)
        elif gz_exists:
            logger.warning("  [!!] %s  (gz only, not decompressed)", gz_path.name)
            all_ok = False
        else:
            logger.error("  [!!] %s  MISSING", filename)
            all_ok = False

    if not all_ok:
        logger.error("\nSome files are missing or incomplete — re-run to retry")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info("Files saved to: %s", DATA_DIR)
    logger.info("\nNext step: python 01_load_and_convert_data.py")


if __name__ == "__main__":
    main()
