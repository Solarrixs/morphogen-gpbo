"""Base scraper ABC and result dataclasses."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PaperResult:
    """Structured result from a paper search."""
    title: str
    authors: list[str]
    abstract: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    biorxiv_id: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    source: str = "unknown"
    search_query: Optional[str] = None


@dataclass
class DatasetResult:
    """Structured result from a dataset search."""
    name: str
    accession: Optional[str] = None
    repository: str = "unknown"
    url: Optional[str] = None
    species: Optional[str] = None
    tissue: Optional[str] = None
    n_cells: Optional[int] = None
    description: Optional[str] = None


class BaseScraper(ABC):
    """Abstract base class for literature scrapers."""

    @abstractmethod
    def search_papers(self, query: str, max_results: int = 100) -> list[PaperResult]:
        """Search for papers matching the query."""
        ...

    def search_datasets(self, query: str, max_results: int = 100) -> list[DatasetResult]:
        """Search for datasets. Override in subclasses that support this."""
        return []

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of this data source (e.g., 'pubmed', 'biorxiv')."""
        ...
