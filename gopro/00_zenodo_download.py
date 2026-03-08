"""
Step 0: Download datasets from Zenodo using the REST API.

Two modes:
  1. Download known records (HNOCA, Braun fetal brain) by record ID
  2. Search for additional brain organoid datasets by keyword

Usage:
  python 00_zenodo_download.py                    # download known records
  python 00_zenodo_download.py --search           # search for more datasets
  python 00_zenodo_download.py --search "morphogen screen organoid"  # custom query
"""

import requests
import hashlib
import sys
from pathlib import Path
from urllib.parse import urljoin

from gopro.config import PROJECT_DIR, DATA_DIR, get_logger

logger = get_logger(__name__)

ZENODO_API = "https://zenodo.org/api"

# Known records we need
KNOWN_RECORDS = {
    "15004817": {
        "description": "HNOCA + Braun fetal brain minimal mapping files",
        "files": {
            "hnoca_minimal_for_mapping.h5ad": {
                "md5": "078675d6108e93cebc99676b6b0626aa",
                "description": "HNOCA reference for scArches projection (2.7 GB)",
            },
            "braun-et-al_minimal_for_mapping.h5ad": {
                "md5": "324e42c616f957d3be182afb857b1ade",
                "description": "Braun/Linnarsson fetal brain reference (10.4 GB)",
            },
        },
    },
    # Uncomment to also grab the full atlas + disease atlas:
    # "14161275": {
    #     "description": "Full HNOCA + extended + disease atlas",
    #     "files": {
    #         "disease_atlas.h5ad": {
    #             "md5": "8832126bb6ebc3a5a0b946f02391fddc",
    #             "description": "Disease atlas (2.2 GB)",
    #         },
    #     },
    # },
}


def get_record(record_id):
    """Fetch record metadata from Zenodo REST API."""
    url = f"{ZENODO_API}/records/{record_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def list_files(record_id):
    """List all files in a Zenodo record with sizes and checksums."""
    data = get_record(record_id)

    logger.info("Record %s: %s", record_id, data['metadata']['title'])
    logger.info("  DOI: %s", data['doi'])
    logger.info("  Published: %s", data['metadata']['publication_date'])
    logger.info("  License: %s", data['metadata'].get('license', {}).get('id', 'unknown'))
    logger.info("  Files:")

    files = []
    for f in data.get("files", []):
        size_gb = f["size"] / 1e9
        checksum = f["checksum"].replace("md5:", "")
        download_url = f["links"]["self"]
        logger.info("    %s  %6.2f GB  md5:%s", f['key'], size_gb, checksum)
        files.append({
            "filename": f["key"],
            "size": f["size"],
            "size_gb": size_gb,
            "md5": checksum,
            "url": download_url,
        })

    return files


def md5_file(filepath, chunk_size=8192 * 1024):
    """Compute MD5 of a file without loading it all into memory."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url, output_path, expected_md5=None):
    """Download a file with progress, resume support, and checksum verification."""
    output_path = Path(output_path)

    # Check if already downloaded
    if output_path.exists() and expected_md5:
        logger.info("  File exists: %s", output_path.name)
        actual_md5 = md5_file(output_path)
        if actual_md5 == expected_md5:
            logger.info("  Checksum OK — skipping")
            return True
        else:
            logger.warning("  Checksum mismatch — re-downloading")

    # Stream download with progress
    logger.info("  Downloading %s...", output_path.name)
    headers = {}
    mode = "wb"
    downloaded = 0

    # Resume support
    if output_path.exists():
        downloaded = output_path.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        logger.info("  Resuming from %.2f GB", downloaded / 1e9)

    resp = requests.get(url, headers=headers, stream=True)

    if resp.status_code == 416:
        # Range not satisfiable — file is complete
        logger.info("  File already fully downloaded")
    elif resp.status_code in (200, 206):
        total = int(resp.headers.get("content-length", 0)) + downloaded
        with open(output_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                    print(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB", end="", flush=True)
        print()  # newline after progress bar
    else:
        logger.error("HTTP %d", resp.status_code)
        return False

    # Verify checksum
    if expected_md5:
        logger.info("  Verifying checksum...")
        actual_md5 = md5_file(output_path)
        if actual_md5 == expected_md5:
            logger.info("  MD5 OK")
            return True
        else:
            logger.error("MD5 MISMATCH: expected %s, got %s", expected_md5, actual_md5)
            return False

    return True


def download_known_records():
    """Download all files from known Zenodo records."""
    logger.info("--- DOWNLOADING KNOWN RECORDS ---")

    for record_id, info in KNOWN_RECORDS.items():
        logger.info("--- %s ---", info['description'])

        # Get actual download URLs from API
        record_data = get_record(record_id)
        api_files = {f["key"]: f for f in record_data.get("files", [])}

        for filename, file_info in info["files"].items():
            logger.info("  %s", file_info['description'])
            output_path = DATA_DIR / filename

            if filename in api_files:
                url = api_files[filename]["links"]["self"]
            else:
                # Fallback: construct URL
                url = f"{ZENODO_API}/records/{record_id}/files/{filename}/content"

            download_file(url, output_path, expected_md5=file_info["md5"])


def search_zenodo(query, max_results=20):
    """
    Search Zenodo for datasets using Elasticsearch syntax.

    Examples:
      search_zenodo("brain organoid scRNA-seq morphogen")
      search_zenodo('metadata.title:"neural organoid" AND metadata.resource_type.type:"dataset"')
    """
    logger.info("--- SEARCHING ZENODO: %s ---", query)

    params = {
        "q": query,
        "type": "dataset",
        "size": max_results,
        "sort": "mostrecent",
    }

    resp = requests.get(f"{ZENODO_API}/records", params=params)
    resp.raise_for_status()
    data = resp.json()

    total = data["hits"]["total"]
    logger.info("Found %d results (showing top %d)", total, min(max_results, total))

    for i, hit in enumerate(data["hits"]["hits"], 1):
        meta = hit["metadata"]
        title = meta.get("title", "No title")
        doi = hit.get("doi", "no DOI")
        date = meta.get("publication_date", "unknown")
        description = meta.get("description", "")[:200].replace("\n", " ")

        # Count files and total size
        files = hit.get("files", [])
        n_files = len(files)
        total_size = sum(f.get("size", 0) for f in files) / 1e9

        # File types
        extensions = set(Path(f["key"]).suffix for f in files if "key" in f)

        logger.info("  [%d] %s", i, title)
        logger.info("      DOI: %s", doi)
        logger.info("      Date: %s", date)
        logger.info("      Files: %d (%.2f GB total)", n_files, total_size)
        logger.info("      Types: %s", ', '.join(extensions) if extensions else 'unknown')
        logger.info("      Record ID: %s", hit['id'])
        if description:
            logger.info("      Desc: %s...", description)

    return data["hits"]["hits"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        # Search mode
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "brain organoid morphogen scRNA-seq"
        results = search_zenodo(query)

        # Also try more specific queries
        additional_queries = [
            "neural organoid single-cell differentiation protocol",
            "morphogen screen organoid h5ad",
            "brain organoid atlas scRNA-seq 2024 2025",
        ]
        for q in additional_queries:
            logger.info("--- Additional search: %s ---", q)
            search_zenodo(q, max_results=5)

    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        # List files in a specific record
        record_id = sys.argv[2] if len(sys.argv) > 2 else "15004817"
        list_files(record_id)

    else:
        # Default: download known records
        download_known_records()

        logger.info("--- SUMMARY ---")
        for filename in ["hnoca_minimal_for_mapping.h5ad", "braun-et-al_minimal_for_mapping.h5ad"]:
            path = DATA_DIR / filename
            if path.exists():
                logger.info("  OK %s: %.2f GB", filename, path.stat().st_size / 1e9)
            else:
                logger.warning("  MISSING %s", filename)
