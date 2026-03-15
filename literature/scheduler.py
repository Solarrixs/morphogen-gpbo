"""Scheduler/orchestrator for the literature scraping pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from Levenshtein import ratio as levenshtein_ratio
from sqlalchemy.orm import Session

from literature.db import init_db
from literature.models import Dataset, Paper, SearchRun
from literature.scrapers.base import DatasetResult, PaperResult, detect_scrna_seq, detect_spatial
from literature.scrapers.pubmed import PubMedScraper
from literature.scrapers.biorxiv import BioRxivScraper
from literature.scrapers.dataset_sources import GEOScraper, ZenodoScraper, CellxGeneScraper

logger = logging.getLogger(__name__)

_TITLE_SIMILARITY_THRESHOLD = 0.9


def deduplicate_papers(papers: list[PaperResult]) -> list[PaperResult]:
    """Remove duplicate papers by DOI, then by fuzzy title match (ratio > 0.9).

    DOI-based deduplication takes priority: if two PaperResult objects share the
    same non-None DOI, only the first is kept.  After DOI dedup, remaining papers
    are compared pairwise by Levenshtein title similarity; if the ratio exceeds
    ``_TITLE_SIMILARITY_THRESHOLD`` they are considered duplicates and the later
    one is dropped.

    Args:
        papers: List of PaperResult objects, possibly containing duplicates.

    Returns:
        Deduplicated list preserving original ordering of first occurrences.
    """
    if not papers:
        return []

    seen_dois: set[str] = set()
    doi_deduped: list[PaperResult] = []
    for paper in papers:
        if paper.doi:
            doi_norm = paper.doi.strip().lower()
            if doi_norm in seen_dois:
                continue
            seen_dois.add(doi_norm)
        doi_deduped.append(paper)

    # Title-based fuzzy deduplication on remainder
    result: list[PaperResult] = []
    for candidate in doi_deduped:
        title_lower = candidate.title.strip().lower()
        is_dup = False
        for kept in result:
            if levenshtein_ratio(title_lower, kept.title.strip().lower()) > _TITLE_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            result.append(candidate)

    return result


def run_scrape(config: dict, db_path=None) -> dict:
    """Run all enabled scrapers, deduplicate, store new papers/datasets.

    Args:
        config: Configuration dictionary (same schema as config.yaml).
        db_path: Override path to the SQLite database.  Falls back to
            ``config["db_path"]`` and then the default ``literature/papers.db``.

    Returns:
        Dict mapping ``source`` -> ``{"n_results": int, "n_new": int}``.
    """
    effective_db_path = str(db_path) if db_path is not None else config.get("db_path")
    engine = init_db(effective_db_path)

    sources_cfg = config.get("sources", {})
    ncbi_api_key: str = config.get("ncbi_api_key", "") or ""
    summary: dict[str, dict] = {}

    with Session(engine) as session:
        # --- Paper sources ---
        summary.update(_scrape_pubmed(session, sources_cfg, ncbi_api_key))
        summary.update(_scrape_biorxiv(session, sources_cfg))

        # --- Dataset sources ---
        datasets_cfg = sources_cfg.get("datasets", {})
        summary.update(_scrape_geo(session, datasets_cfg, ncbi_api_key))
        summary.update(_scrape_zenodo(session, datasets_cfg))
        summary.update(_scrape_cellxgene(session, datasets_cfg))

        session.commit()

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_from(date_range_days: int) -> str:
    """Return an ISO date string N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=date_range_days)
    return dt.strftime("%Y/%m/%d")


def _scrape_pubmed(
    session: Session,
    sources_cfg: dict,
    ncbi_api_key: str,
) -> dict:
    pubmed_cfg = sources_cfg.get("pubmed", {})
    if not pubmed_cfg.get("enabled", False):
        return {}

    scraper = PubMedScraper(api_key=ncbi_api_key or None)
    queries: list[str] = pubmed_cfg.get("queries", [])
    max_results: int = pubmed_cfg.get("max_results_per_query", 100)
    date_range_days: int = pubmed_cfg.get("date_range_days", 90)
    date_from = _date_from(date_range_days)

    all_results: list[PaperResult] = []
    for query in queries:
        results = scraper.search(query, max_results=max_results, date_from=date_from)
        n_results = len(results)
        all_results.extend(results)

        run = SearchRun(source="pubmed", query=query, n_results=n_results, n_new=0)
        session.add(run)
        session.flush()
        logger.debug("PubMed query %r → %d results", query, n_results)

    deduped = deduplicate_papers(all_results)
    n_new = _store_papers(session, deduped, source="pubmed")

    # Back-fill n_new on the last SearchRun for this source (best-effort)
    _update_last_run_n_new(session, "pubmed", n_new)

    return {"pubmed": {"n_results": len(all_results), "n_new": n_new}}


def _scrape_biorxiv(session: Session, sources_cfg: dict) -> dict:
    biorxiv_cfg = sources_cfg.get("biorxiv", {})
    if not biorxiv_cfg.get("enabled", False):
        return {}

    scraper = BioRxivScraper()
    queries: list[str] = biorxiv_cfg.get("queries", [])
    max_results: int = biorxiv_cfg.get("max_results_per_query", 100)
    date_range_days: int = biorxiv_cfg.get("date_range_days", 90)
    date_from = _date_from(date_range_days)

    all_results: list[PaperResult] = []
    for query in queries:
        results = scraper.search(query, max_results=max_results, date_from=date_from)
        n_results = len(results)
        all_results.extend(results)

        run = SearchRun(source="biorxiv", query=query, n_results=n_results, n_new=0)
        session.add(run)
        session.flush()
        logger.debug("bioRxiv query %r → %d results", query, n_results)

    deduped = deduplicate_papers(all_results)
    n_new = _store_papers(session, deduped, source="biorxiv")
    _update_last_run_n_new(session, "biorxiv", n_new)

    return {"biorxiv": {"n_results": len(all_results), "n_new": n_new}}


def _scrape_geo(session: Session, datasets_cfg: dict, ncbi_api_key: str) -> dict:
    geo_cfg = datasets_cfg.get("geo", {})
    if not geo_cfg.get("enabled", False):
        return {}

    scraper = GEOScraper(api_key=ncbi_api_key or None)
    queries: list[str] = geo_cfg.get("queries", [])
    max_results: int = geo_cfg.get("max_results_per_query", 50)

    all_results: list[DatasetResult] = []
    for query in queries:
        results = scraper.search(query, max_results=max_results)
        all_results.extend(results)

        run = SearchRun(source="geo", query=query, n_results=len(results), n_new=0)
        session.add(run)
        session.flush()

    n_new = _store_datasets(session, all_results)
    _update_last_run_n_new(session, "geo", n_new)
    return {"geo": {"n_results": len(all_results), "n_new": n_new}}


def _scrape_zenodo(session: Session, datasets_cfg: dict) -> dict:
    zenodo_cfg = datasets_cfg.get("zenodo", {})
    if not zenodo_cfg.get("enabled", False):
        return {}

    scraper = ZenodoScraper()
    queries: list[str] = zenodo_cfg.get("queries", [])
    max_results: int = zenodo_cfg.get("max_results_per_query", 50)

    all_results: list[DatasetResult] = []
    for query in queries:
        results = scraper.search(query, max_results=max_results)
        all_results.extend(results)

        run = SearchRun(source="zenodo", query=query, n_results=len(results), n_new=0)
        session.add(run)
        session.flush()

    n_new = _store_datasets(session, all_results)
    _update_last_run_n_new(session, "zenodo", n_new)
    return {"zenodo": {"n_results": len(all_results), "n_new": n_new}}


def _scrape_cellxgene(session: Session, datasets_cfg: dict) -> dict:
    cxg_cfg = datasets_cfg.get("cellxgene", {})
    if not cxg_cfg.get("enabled", False):
        return {}

    scraper = CellxGeneScraper()
    queries: list[str] = cxg_cfg.get("queries", [])
    max_results: int = cxg_cfg.get("max_results_per_query", 50)

    all_results: list[DatasetResult] = []
    for query in queries:
        results = scraper.search(query, max_results=max_results)
        all_results.extend(results)

        run = SearchRun(source="cellxgene", query=query, n_results=len(results), n_new=0)
        session.add(run)
        session.flush()

    n_new = _store_datasets(session, all_results)
    _update_last_run_n_new(session, "cellxgene", n_new)
    return {"cellxgene": {"n_results": len(all_results), "n_new": n_new}}


def _store_papers(session: Session, papers: list[PaperResult], source: str) -> int:
    """Insert new papers, skipping those whose DOI already exists.

    Returns number of newly inserted papers.
    """
    n_new = 0
    for result in papers:
        if result.doi:
            existing = session.query(Paper).filter_by(doi=result.doi).first()
            if existing:
                logger.debug("Skipping duplicate DOI %s", result.doi)
                continue

        text_for_detection = " ".join(
            filter(None, [result.title, result.abstract])
        )
        paper = Paper(
            doi=result.doi,
            pmid=result.pmid,
            title=result.title,
            authors=result.authors,
            journal=result.journal,
            year=result.year,
            abstract=result.abstract,
            source=result.source or source,
            url=result.url,
            has_scrna_seq=detect_scrna_seq(text_for_detection),
            has_spatial=detect_spatial(text_for_detection),
            status="pending",
        )
        session.add(paper)
        n_new += 1

    return n_new


def _store_datasets(session: Session, datasets: list[DatasetResult]) -> int:
    """Insert new datasets, skipping those whose accession already exists.

    Returns number of newly inserted datasets.
    """
    n_new = 0
    for result in datasets:
        existing = session.query(Dataset).filter_by(accession=result.accession).first()
        if existing:
            continue

        dataset = Dataset(
            accession=result.accession,
            source=result.source,
            title=result.title,
            species=result.species,
            cell_count=result.cell_count,
            format=result.format,
            download_url=result.download_url,
            size_bytes=result.size_bytes,
            status="pending",
        )
        session.add(dataset)
        n_new += 1

    return n_new


def _update_last_run_n_new(session: Session, source: str, n_new: int) -> None:
    """Update the most recently flushed SearchRun for this source with n_new."""
    run = (
        session.query(SearchRun)
        .filter_by(source=source)
        .order_by(SearchRun.id.desc())
        .first()
    )
    if run is not None:
        run.n_new = n_new
