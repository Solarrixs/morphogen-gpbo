"""Dataset scrapers for GEO, Zenodo, and CellxGene."""
import logging
import os
from typing import Optional

import requests
from Bio import Entrez

from literature.scrapers.base import BaseScraper, DatasetResult

logger = logging.getLogger(__name__)

# File extensions that indicate scRNA-seq dataset files
DATASET_EXTENSIONS = {".h5ad", ".rds", ".mtx.gz", ".loom"}


class GEOScraper(BaseScraper):
    """Scraper for NCBI Gene Expression Omnibus (GEO) datasets."""

    name = "geo"
    rate_limit_delay = 0.5

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "literature-scraper@local",
    ) -> None:
        Entrez.email = email
        # Use provided key or fall back to environment variable
        key = api_key or os.environ.get("NCBI_API_KEY")
        if key:
            Entrez.api_key = key

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list[DatasetResult]:
        """Search GEO for datasets matching the query.

        Args:
            query: Search terms.
            max_results: Maximum number of results to return.
            date_from: Not used for GEO (GEO API does not support date filtering easily).

        Returns:
            List of DatasetResult objects.
        """
        handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
        search_results = Entrez.read(handle)
        handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return []

        self._rate_limit()

        summary_handle = Entrez.esummary(db="gds", id=",".join(id_list))
        summaries = Entrez.read(summary_handle)
        summary_handle.close()

        results: list[DatasetResult] = []
        for summary in summaries:
            try:
                accession = summary.get("Accession", "")
                title = summary.get("title") or summary.get("Title")
                description = summary.get("summary") or summary.get("Summary")
                species = summary.get("taxon") or summary.get("Taxon")
                n_samples_raw = summary.get("n_samples") or summary.get("Samples")
                cell_count: Optional[int] = None
                if n_samples_raw is not None:
                    try:
                        cell_count = int(n_samples_raw)
                    except (ValueError, TypeError):
                        pass

                results.append(
                    DatasetResult(
                        accession=accession,
                        source=self.name,
                        title=title,
                        species=species,
                        cell_count=cell_count,
                    )
                )
            except Exception:
                continue

        return results


class ZenodoScraper(BaseScraper):
    """Scraper for Zenodo dataset records."""

    name = "zenodo"
    rate_limit_delay = 0.5

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list[DatasetResult]:
        """Search Zenodo for dataset records matching the query.

        Only returns DatasetResult entries for files with recognised scRNA-seq
        extensions (.h5ad, .rds, .mtx.gz, .loom).

        Args:
            query: Search terms.
            max_results: Maximum number of results to return.
            date_from: Not used (Zenodo API does not require it here).

        Returns:
            List of DatasetResult objects, one per matching file.
        """
        url = "https://zenodo.org/api/records"
        params = {
            "q": query,
            "type": "dataset",
            "size": max_results,
            "sort": "mostrecent",
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])

        results: list[DatasetResult] = []
        for hit in hits:
            doi = hit.get("doi")
            metadata = hit.get("metadata", {})
            title = metadata.get("title")
            files = hit.get("files", [])

            for file_info in files:
                key: str = file_info.get("key", "")
                # Check whether the filename ends with a recognised extension
                matched_ext = None
                for ext in DATASET_EXTENSIONS:
                    if key.endswith(ext):
                        matched_ext = ext.lstrip(".")
                        break
                if matched_ext is None:
                    continue

                size_bytes = file_info.get("size")
                download_url = file_info.get("links", {}).get("self")

                # Use doi as accession if available, else fall back to key
                accession = doi or key

                results.append(
                    DatasetResult(
                        accession=accession,
                        source=self.name,
                        title=title,
                        format=matched_ext,
                        download_url=download_url,
                        size_bytes=size_bytes,
                    )
                )

        return results


class CellxGeneScraper(BaseScraper):
    """Scraper for CellxGene Census datasets."""

    name = "cellxgene"
    rate_limit_delay = 1.0

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list[DatasetResult]:
        """Search CellxGene Census for datasets matching the query tissue.

        Requires the ``cellxgene_census`` package. If it is not installed this
        method logs a warning and returns an empty list.

        Args:
            query: Tissue or cell type term to filter datasets by.
            max_results: Maximum number of results to return.
            date_from: Not used.

        Returns:
            List of DatasetResult objects, or [] if package is unavailable.
        """
        try:
            import cellxgene_census  # noqa: F401
        except ImportError:
            logger.warning(
                "cellxgene_census is not installed; skipping CellxGene search. "
                "Install with: pip install cellxgene-census"
            )
            return []

        try:
            with cellxgene_census.open_soma() as census:
                datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()

            query_lower = query.lower()
            # Filter rows where tissue_general contains the query term
            if "tissue_general" in datasets_df.columns:
                mask = datasets_df["tissue_general"].str.lower().str.contains(
                    query_lower, na=False
                )
                filtered = datasets_df[mask].head(max_results)
            else:
                filtered = datasets_df.head(max_results)

            results: list[DatasetResult] = []
            for _, row in filtered.iterrows():
                dataset_id = row.get("dataset_id", "") or ""
                title = row.get("title") or row.get("dataset_title")
                results.append(
                    DatasetResult(
                        accession=dataset_id,
                        source=self.name,
                        title=title,
                        format="h5ad",
                    )
                )
            return results

        except Exception as exc:
            logger.warning("CellxGene Census query failed: %s", exc)
            return []
