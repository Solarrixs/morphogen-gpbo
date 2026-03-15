"""CLI entry point for the literature scraping pipeline.

Subcommands
-----------
scrape   Run scrapers (all sources or a single named source).
review   Interactive review of pending papers.
status   Print counts of papers/datasets by status and source.
export   Export approved papers/datasets to CSV or JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Optional

from sqlalchemy.orm import Session

from literature.config import load_config
from literature.db import init_db
from literature.models import Dataset, Paper


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_scrape(args: argparse.Namespace) -> None:
    """Run enabled scrapers and persist results to the database."""
    from literature.scheduler import run_scrape

    config = load_config()

    if args.source:
        # Disable every source except the one requested.
        source = args.source
        sources_cfg = config.get("sources", {})

        # Paper sources
        for key in ("pubmed", "biorxiv"):
            if key in sources_cfg:
                sources_cfg[key]["enabled"] = key == source

        # Dataset sources
        datasets_cfg = sources_cfg.get("datasets", {})
        for key in ("geo", "zenodo", "cellxgene"):
            if key in datasets_cfg:
                datasets_cfg[key]["enabled"] = key == source

    if args.dry_run:
        sources_cfg = config.get("sources", {})
        print("=== Dry-run: configured queries ===")
        for src_name in ("pubmed", "biorxiv"):
            src = sources_cfg.get(src_name, {})
            if src.get("enabled"):
                print(f"\n[{src_name}]")
                for q in src.get("queries", []):
                    print(f"  - {q}")
        datasets_cfg = sources_cfg.get("datasets", {})
        for src_name in ("geo", "zenodo", "cellxgene"):
            src = datasets_cfg.get(src_name, {})
            if src.get("enabled"):
                print(f"\n[{src_name}]")
                for q in src.get("queries", []):
                    print(f"  - {q}")
        return

    summary = run_scrape(config)
    print("Scrape complete:")
    for source_name, counts in summary.items():
        print(f"  {source_name}: {counts['n_results']} results, {counts['n_new']} new")


def cmd_review(args: argparse.Namespace) -> None:
    """Interactive review of pending papers."""
    from literature.review import review_papers

    config = load_config()
    db_path = config.get("db_path")
    review_papers(db_path=db_path, limit=args.limit)


def cmd_status(args: argparse.Namespace) -> None:
    """Print a summary table of paper/dataset counts by status and source."""
    config = load_config()
    db_path = config.get("db_path")
    engine = init_db(db_path)

    with Session(engine) as session:
        papers = session.query(Paper).all()
        datasets = session.query(Dataset).all()

    # Papers by status
    paper_status: dict[str, int] = {}
    paper_source: dict[str, int] = {}
    for p in papers:
        paper_status[p.status] = paper_status.get(p.status, 0) + 1
        paper_source[p.source] = paper_source.get(p.source, 0) + 1

    # Datasets by status
    dataset_status: dict[str, int] = {}
    dataset_source: dict[str, int] = {}
    for d in datasets:
        dataset_status[d.status] = dataset_status.get(d.status, 0) + 1
        dataset_source[d.source] = dataset_source.get(d.source, 0) + 1

    print(f"\n=== Papers ({len(papers)} total) ===")
    print("  By status:")
    for status, count in sorted(paper_status.items()):
        print(f"    {status:12s} {count}")
    print("  By source:")
    for source, count in sorted(paper_source.items()):
        print(f"    {source:12s} {count}")

    print(f"\n=== Datasets ({len(datasets)} total) ===")
    print("  By status:")
    for status, count in sorted(dataset_status.items()):
        print(f"    {status:12s} {count}")
    print("  By source:")
    for source, count in sorted(dataset_source.items()):
        print(f"    {source:12s} {count}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export papers (and optionally datasets) to CSV or JSON."""
    config = load_config()
    db_path = config.get("db_path")
    engine = init_db(db_path)

    with Session(engine) as session:
        query = session.query(Paper)
        if args.status:
            query = query.filter(Paper.status == args.status)
        papers = query.order_by(Paper.discovered_at.desc()).all()

        rows = []
        for p in papers:
            row: dict = {
                "id": p.id,
                "doi": p.doi,
                "pmid": p.pmid,
                "title": p.title,
                "authors": p.authors,
                "journal": p.journal,
                "year": p.year,
                "abstract": p.abstract,
                "source": p.source,
                "url": p.url,
                "has_scrna_seq": p.has_scrna_seq,
                "has_spatial": p.has_spatial,
                "species": p.species,
                "brain_regions": p.brain_regions,
                "cell_count": p.cell_count,
                "accession_numbers": p.accession_numbers,
                "status": p.status,
                "discovered_at": p.discovered_at.isoformat() if p.discovered_at else None,
                "reviewed_at": p.reviewed_at.isoformat() if p.reviewed_at else None,
            }
            if args.with_datasets:
                row["datasets"] = [
                    {
                        "accession": d.accession,
                        "source": d.source,
                        "title": d.title,
                        "cell_count": d.cell_count,
                        "format": d.format,
                        "download_url": d.download_url,
                    }
                    for d in p.datasets
                ]
            rows.append(row)

    if args.format == "json":
        print(json.dumps(rows, indent=2, default=str))
    else:
        # CSV — flatten datasets list to a semicolon-separated accessions field
        if not rows:
            print("No papers found.")
            return
        fieldnames = [k for k in rows[0].keys() if k != "datasets"]
        if args.with_datasets:
            fieldnames.append("dataset_accessions")

        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            datasets_val = row.pop("datasets", [])
            if args.with_datasets:
                row["dataset_accessions"] = ";".join(
                    d["accession"] for d in datasets_val
                )
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m literature",
        description="Literature scraping pipeline for brain sc/spatial papers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- scrape ---
    scrape_p = subparsers.add_parser("scrape", help="Run scrapers and store results.")
    scrape_p.add_argument(
        "--source",
        metavar="SOURCE",
        default=None,
        help="Limit to a single source (pubmed, biorxiv, geo, zenodo, cellxgene).",
    )
    scrape_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configured queries without executing any scrape.",
    )

    # --- review ---
    review_p = subparsers.add_parser("review", help="Interactive paper review.")
    review_p.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of pending papers to review (default: 50).",
    )

    # --- status ---
    subparsers.add_parser("status", help="Show counts by status and source.")

    # --- export ---
    export_p = subparsers.add_parser("export", help="Export papers to CSV or JSON.")
    export_p.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv).",
    )
    export_p.add_argument(
        "--status",
        metavar="STATUS",
        default=None,
        help="Filter by status (pending, approved, rejected).",
    )
    export_p.add_argument(
        "--with-datasets",
        action="store_true",
        help="Include linked datasets in the export.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "scrape": cmd_scrape,
        "review": cmd_review,
        "status": cmd_status,
        "export": cmd_export,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
