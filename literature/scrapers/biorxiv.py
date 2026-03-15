"""bioRxiv/medRxiv scraper using the bioRxiv REST API."""
from datetime import datetime, timedelta
from typing import Optional

import requests

from literature.scrapers.base import BaseScraper, PaperResult


class BioRxivScraper(BaseScraper):
    """Scraper for bioRxiv/medRxiv preprints via the bioRxiv REST API."""

    name = "biorxiv"
    rate_limit_delay = 1.0

    BASE_URL = "https://api.biorxiv.org/details"

    def __init__(self, server: str = "biorxiv") -> None:
        """Initialize with the target server.

        Args:
            server: "biorxiv" or "medrxiv"
        """
        self.server = server
        self.name = server  # use actual server name as source label

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list[PaperResult]:
        """Search bioRxiv/medRxiv for papers matching the query.

        Args:
            query: Search terms to filter results by (matched against title + abstract).
            max_results: Maximum number of results to return.
            date_from: Start date in YYYY-MM-DD format. Defaults to 7 days ago.

        Returns:
            List of PaperResult objects matching the query.
        """
        if date_from is None:
            date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        date_to = datetime.now().strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{self.server}/{date_from}/{date_to}/0/{max_results}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        collection = data.get("collection", [])

        results: list[PaperResult] = []
        query_lower = query.lower()

        for item in collection:
            title = item.get("title", "") or ""
            abstract = item.get("abstract", "") or ""

            # Filter: only keep items where query appears in title or abstract
            if query_lower not in title.lower() and query_lower not in abstract.lower():
                continue

            doi = item.get("doi")
            authors_raw = item.get("authors", "") or ""
            # Replace ";" separator with "," for consistency
            authors = authors_raw.replace(";", ",")

            # Extract year from date string (YYYY-MM-DD)
            date_str = item.get("date", "") or ""
            year: Optional[int] = None
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                except ValueError:
                    pass

            url_str = f"https://doi.org/{doi}" if doi else None

            results.append(
                PaperResult(
                    title=title,
                    source=self.server,
                    doi=doi,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    url=url_str,
                )
            )

        return results
