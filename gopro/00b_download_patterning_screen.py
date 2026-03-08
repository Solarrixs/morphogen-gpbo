"""
Step 0b: Download Sanchis-Calleja/Azbukina organoid patterning screen data.

Zenodo record: 17225179
Paper: "Systematic scRNAseq screens profile neural organoid response to morphogens"
GitHub: https://github.com/quadbio/organoid_patterning_screen

Files:
  - OSMGT_processed_files.tar.gz (22.8 GB) — processed scRNA-seq data
  - H9_H1_hg38_diversed_parse_true.vcf.gz (3.0 MB) — genotyping VCF
  - WIBJ2_WTC_hg38_parse.vcf.gz (2.7 MB) — genotyping VCF

Strategy: zenodo_get generates URL list + MD5s, aria2c downloads in parallel.
Fallback: pure Python streaming download with resume support if aria2c unavailable.

Usage:
  python 00b_download_patterning_screen.py              # download all files
  python 00b_download_patterning_screen.py --list       # list files only
  python 00b_download_patterning_screen.py --no-extract # download without extracting
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

RECORD_ID = "17225179"
ZENODO_API = "https://zenodo.org/api"
PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
OUTPUT_DIR = PROJECT_DIR / "data" / "patterning_screen"


def get_record_files(record_id: str) -> list[dict]:
    """Fetch file metadata from Zenodo REST API."""
    url = f"{ZENODO_API}/records/{record_id}"
    req = Request(url, headers={"User-Agent": "morphogen-gpbo/1.0"})
    with urlopen(req) as resp:
        data = json.loads(resp.read())

    files = []
    for f in data.get("files", []):
        checksum = f["checksum"].replace("md5:", "")
        files.append({
            "filename": f["key"],
            "size": f["size"],
            "md5": checksum,
            "url": f["links"]["self"],
        })
    return files


def list_files():
    """Print file listing for the record."""
    files = get_record_files(RECORD_ID)
    print(f"Zenodo record {RECORD_ID} — {len(files)} files:\n")
    for f in files:
        size_str = format_size(f["size"])
        print(f"  {f['filename']:50s}  {size_str:>10s}  md5:{f['md5']}")
    total = sum(f["size"] for f in files)
    print(f"\n  {'Total':50s}  {format_size(total):>10s}")
    return files


def format_size(nbytes: int) -> str:
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


def file_already_valid(filepath: Path, expected_md5: str) -> bool:
    """Check if file exists and checksum matches."""
    if not filepath.exists():
        return False
    print(f"  File exists: {filepath.name}, verifying checksum...")
    actual = md5_file(filepath)
    if actual == expected_md5:
        print(f"  Checksum OK — skipping")
        return True
    print(f"  Checksum mismatch (expected {expected_md5}, got {actual}) — re-downloading")
    return False


def has_aria2c() -> bool:
    return shutil.which("aria2c") is not None


def has_zenodo_get() -> bool:
    return shutil.which("zenodo_get") is not None


def download_aria2c(files: list[dict], output_dir: Path):
    """Download using aria2c for parallel multi-connection transfers."""
    # Write URL file in aria2c input format
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        url_file = f.name
        for file_info in files:
            filepath = output_dir / file_info["filename"]
            if file_already_valid(filepath, file_info["md5"]):
                continue
            # aria2c input file format: URL\n  out=FILENAME\n  checksum=md5=HASH\n
            f.write(f"{file_info['url']}\n")
            f.write(f"  out={file_info['filename']}\n")
            f.write(f"  checksum=md5={file_info['md5']}\n")

    # Check if anything to download
    with open(url_file) as f:
        if not f.read().strip():
            print("\nAll files already downloaded and verified.")
            Path(url_file).unlink()
            return True

    cmd = [
        "aria2c",
        "-i", url_file,
        "-d", str(output_dir),
        "-x", "16",              # max connections per server
        "-s", "16",              # split file into 16 parts
        "-k", "1M",              # min split size
        "-j", "3",               # 3 concurrent downloads
        "--max-tries=10",
        "--retry-wait=5",
        "--continue=true",       # resume partial downloads
        "--auto-file-renaming=false",
        "--console-log-level=notice",
        "--summary-interval=10",
    ]

    print(f"\nStarting aria2c with 16 connections per file...\n")
    result = subprocess.run(cmd)
    Path(url_file).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\naria2c exited with code {result.returncode}")
        return False
    return True


def download_python(files: list[dict], output_dir: Path):
    """Fallback: pure Python streaming download with resume and progress."""
    for i, file_info in enumerate(files, 1):
        filepath = output_dir / file_info["filename"]
        print(f"\n[{i}/{len(files)}] {file_info['filename']} ({format_size(file_info['size'])})")

        if file_already_valid(filepath, file_info["md5"]):
            continue

        headers = {"User-Agent": "morphogen-gpbo/1.0"}
        mode = "wb"
        downloaded = 0

        # Resume support
        if filepath.exists():
            downloaded = filepath.stat().st_size
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
            print(f"  Resuming from {format_size(downloaded)}")

        req = Request(file_info["url"], headers=headers)
        try:
            with urlopen(req) as resp:
                total = int(resp.headers.get("content-length", 0)) + downloaded
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
                print()
        except Exception as e:
            if "416" in str(e):
                print(f"  File already fully downloaded")
            else:
                print(f"  ERROR: {e}")
                print(f"  Re-run to retry (resume supported)")
                continue

        # Verify
        print(f"  Verifying checksum...")
        actual = md5_file(filepath)
        if actual == file_info["md5"]:
            print(f"  Checksum OK")
        else:
            print(f"  CHECKSUM MISMATCH — delete and re-run")
            print(f"    Expected: {file_info['md5']}")
            print(f"    Got:      {actual}")


def extract_tar(output_dir: Path):
    """Extract the tar.gz archive if present."""
    tar_file = output_dir / "OSMGT_processed_files.tar.gz"
    if not tar_file.exists():
        print("tar.gz not found, skipping extraction")
        return

    extract_dir = output_dir / "OSMGT_processed_files"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted to {extract_dir}")
        return

    print(f"\nExtracting {tar_file.name} ({format_size(tar_file.stat().st_size)})...")
    print("  This may take a while for 22.8 GB...")
    subprocess.run(
        ["tar", "xzf", str(tar_file), "-C", str(output_dir)],
        check=True,
    )
    print("  Extraction complete")


def main():
    if "--list" in sys.argv:
        list_files()
        return

    do_extract = "--no-extract" not in sys.argv

    print("=" * 60)
    print("PATTERNING SCREEN DATA DOWNLOAD")
    print("Sanchis-Calleja/Azbukina et al. (2025, Nature Methods)")
    print(f"Zenodo record: {RECORD_ID}")
    print("=" * 60)

    # Fetch file metadata from API
    print("\nFetching record metadata...")
    files = get_record_files(RECORD_ID)
    total_size = sum(f["size"] for f in files)
    print(f"  {len(files)} files, {format_size(total_size)} total\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Choose download strategy
    if has_aria2c():
        print("Using aria2c (parallel multi-connection download)")
        success = download_aria2c(files, OUTPUT_DIR)
        if not success:
            print("\naria2c failed, falling back to Python downloader...")
            download_python(files, OUTPUT_DIR)
    else:
        if has_zenodo_get():
            print("aria2c not found. Install with: brew install aria2")
            print("Tip: zenodo_get + aria2c is 5-10x faster than single-threaded download")
        else:
            print("For fastest downloads, install: brew install aria2 && pip install zenodo-get")
        print("Using Python streaming download (single-threaded, resumable)\n")
        download_python(files, OUTPUT_DIR)

    # Post-download verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    all_ok = True
    for file_info in files:
        filepath = OUTPUT_DIR / file_info["filename"]
        if filepath.exists():
            actual = md5_file(filepath)
            status = "OK" if actual == file_info["md5"] else "MISMATCH"
            if status != "OK":
                all_ok = False
            print(f"  {'[OK]' if status == 'OK' else '[!!]'} {file_info['filename']:50s}  {format_size(filepath.stat().st_size)}")
        else:
            all_ok = False
            print(f"  [!!] {file_info['filename']:50s}  MISSING")

    if not all_ok:
        print("\nSome files failed — re-run to retry (downloads are resumable)")
        sys.exit(1)

    # Extract
    if do_extract:
        extract_tar(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Files saved to: {OUTPUT_DIR}")
    print(f"\nNext step: integrate with existing GSE233574 data in the GP-BO pipeline")


if __name__ == "__main__":
    main()
