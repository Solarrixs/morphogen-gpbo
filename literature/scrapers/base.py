"""Base scraper ABC and shared utilities."""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

SCRNA_KEYWORDS = [
    "single-cell",
    "single cell",
    "scrna-seq",
    "snrna-seq",
    "10x genomics",
    "drop-seq",
    "smart-seq",
    "cel-seq",
    "single-nucleus",
    "single nucleus",
]

SPATIAL_KEYWORDS = [
    "spatial transcriptom",
    "merfish",
    "visium",
    "slide-seq",
    "seqfish",
    "starmap",
    "osmfish",
    "10x xenium",
]


def detect_scrna_seq(text: str) -> bool:
    """Check if text contains any scRNA-seq keyword (case-insensitive)."""
    lower = text.lower()
    return any(kw in lower for kw in SCRNA_KEYWORDS)


def detect_spatial(text: str) -> bool:
    """Check if text contains any spatial transcriptomics keyword (case-insensitive)."""
    lower = text.lower()
    return any(kw in lower for kw in SPATIAL_KEYWORDS)


@dataclass
class PaperResult:
    title: str
    source: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    accession_numbers: Optional[dict] = field(default=None)


@dataclass
class DatasetResult:
    accession: str
    source: str
    title: Optional[str] = None
    species: Optional[str] = None
    cell_count: Optional[int] = None
    format: Optional[str] = None
    download_url: Optional[str] = None
    size_bytes: Optional[int] = None


class BaseScraper(ABC):
    """Abstract base class for literature and dataset scrapers."""

    name: str = "base"
    rate_limit_delay: float = 1.0

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list:
        """Search for papers or datasets matching the query.

        Returns a list of PaperResult or DatasetResult objects.
        """

    def _rate_limit(self) -> None:
        """Sleep to respect rate limits."""
        time.sleep(self.rate_limit_delay)
