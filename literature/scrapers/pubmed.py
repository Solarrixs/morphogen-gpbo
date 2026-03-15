"""PubMed scraper using NCBI Entrez API."""
import xml.etree.ElementTree as ET
from typing import Optional

from Bio import Entrez

from literature.scrapers.base import BaseScraper, PaperResult


class PubMedScraper(BaseScraper):
    """Scraper for PubMed literature via NCBI Entrez."""

    name = "pubmed"
    rate_limit_delay = 0.1

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "literature-scraper@local",
    ) -> None:
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
    ) -> list[PaperResult]:
        """Search PubMed and return a list of PaperResult objects."""
        search_kwargs: dict = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "date",
        }
        if date_from:
            search_kwargs["mindate"] = date_from
            search_kwargs["datetype"] = "pdat"

        handle = Entrez.esearch(**search_kwargs)
        search_results = Entrez.read(handle)
        handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return []

        self._rate_limit()

        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="xml",
        )
        xml_data = fetch_handle.read()
        fetch_handle.close()

        # Handle both bytes and str
        if isinstance(xml_data, bytes):
            xml_data = xml_data.decode("utf-8")

        return self._parse_xml(xml_data)

    def _parse_xml(self, xml_data: str) -> list[PaperResult]:
        """Parse PubMed XML response into PaperResult objects."""
        results: list[PaperResult] = []

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return results

        for article_el in root.findall(".//PubmedArticle"):
            try:
                citation = article_el.find("MedlineCitation")
                if citation is None:
                    continue

                # PMID
                pmid_el = citation.find("PMID")
                pmid = pmid_el.text if pmid_el is not None else None

                article = citation.find("Article")
                if article is None:
                    continue

                # Title
                title_el = article.find("ArticleTitle")
                title = title_el.text if title_el is not None else ""

                # Abstract
                abstract_el = article.find("Abstract/AbstractText")
                abstract = abstract_el.text if abstract_el is not None else None

                # Authors
                author_els = article.findall("AuthorList/Author")
                author_parts = []
                for auth in author_els:
                    last = auth.find("LastName")
                    initials = auth.find("Initials")
                    if last is not None:
                        name = last.text or ""
                        if initials is not None and initials.text:
                            name = f"{name} {initials.text}"
                        author_parts.append(name)
                authors = ", ".join(author_parts) if author_parts else None

                # Journal
                journal_el = article.find("Journal/Title")
                journal = journal_el.text if journal_el is not None else None

                # Year — prefer ArticleDate, fall back to PubDate
                year: Optional[int] = None
                year_el = article.find("ArticleDate/Year")
                if year_el is None:
                    year_el = article.find("Journal/JournalIssue/PubDate/Year")
                if year_el is not None and year_el.text:
                    try:
                        year = int(year_el.text)
                    except ValueError:
                        pass

                # DOI from ELocationID
                doi: Optional[str] = None
                for loc in article.findall("ELocationID"):
                    if loc.get("EIdType") == "doi":
                        doi = loc.text
                        break

                results.append(
                    PaperResult(
                        title=title or "",
                        source=self.name,
                        doi=doi,
                        pmid=pmid,
                        authors=authors,
                        journal=journal,
                        year=year,
                        abstract=abstract,
                    )
                )
            except Exception:
                # Skip malformed entries
                continue

        return results
