"""Literature scraper classes."""
from literature.scrapers.base import BaseScraper, PaperResult, DatasetResult
from literature.scrapers.pubmed import PubMedScraper
from literature.scrapers.biorxiv import BioRxivScraper
from literature.scrapers.dataset_sources import GEOScraper, ZenodoScraper, CellxGeneScraper

__all__ = [
    "BaseScraper",
    "PaperResult",
    "DatasetResult",
    "PubMedScraper",
    "BioRxivScraper",
    "GEOScraper",
    "ZenodoScraper",
    "CellxGeneScraper",
]
