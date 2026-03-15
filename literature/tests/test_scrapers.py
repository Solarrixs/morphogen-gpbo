"""Tests for literature scrapers."""
import pytest
from unittest.mock import patch, MagicMock
from literature.scrapers.pubmed import PubMedScraper
from literature.scrapers.biorxiv import BioRxivScraper
from literature.scrapers.dataset_sources import GEOScraper, ZenodoScraper
from literature.scrapers.base import PaperResult, detect_scrna_seq, detect_spatial

MOCK_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Brain organoid scRNA-seq atlas</ArticleTitle>
        <Abstract>
          <AbstractText>We performed single-cell RNA-seq on brain organoids.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
          <Author><LastName>Chen</LastName><Initials>L</Initials></Author>
        </AuthorList>
        <Journal><Title>Nature</Title></Journal>
        <ArticleDate><Year>2026</Year></ArticleDate>
        <ELocationID EIdType="doi">10.1038/test-doi</ELocationID>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestKeywordDetection:
    def test_scrna_detection_positive(self):
        assert detect_scrna_seq("We performed single-cell RNA-seq")
        assert detect_scrna_seq("scRNA-seq analysis revealed")
        assert detect_scrna_seq("using 10x Genomics platform")

    def test_scrna_detection_negative(self):
        assert not detect_scrna_seq("We studied protein expression")
        assert not detect_scrna_seq("bulk RNA-seq analysis")

    def test_spatial_detection_positive(self):
        assert detect_spatial("MERFISH spatial transcriptomics")
        assert detect_spatial("Visium spatial gene expression")

    def test_spatial_detection_negative(self):
        assert not detect_spatial("bulk RNA-seq analysis")


class TestPubMedScraper:
    def test_parse_xml(self):
        scraper = PubMedScraper(api_key="test")
        results = scraper._parse_xml(MOCK_EFETCH_XML)
        assert len(results) == 1
        assert isinstance(results[0], PaperResult)
        assert results[0].source == "pubmed"
        assert results[0].title == "Brain organoid scRNA-seq atlas"
        assert results[0].doi == "10.1038/test-doi"
        assert results[0].pmid == "12345678"
        assert "Smith" in results[0].authors
        assert results[0].year == 2026

    def test_search_with_mocked_entrez(self):
        scraper = PubMedScraper(api_key="test")
        with patch("literature.scrapers.pubmed.Entrez") as mock_entrez:
            mock_entrez.esearch.return_value = MagicMock()
            mock_entrez.read.return_value = {"IdList": ["12345678"]}
            mock_entrez.efetch.return_value = MagicMock(read=lambda: MOCK_EFETCH_XML)
            results = scraper.search("brain organoid", max_results=10)
        assert len(results) == 1
        assert results[0].source == "pubmed"


# ---------------------------------------------------------------------------
# bioRxiv scraper
# ---------------------------------------------------------------------------

MOCK_BIORXIV_RESPONSE = {
    "messages": [{"status": "ok", "count": 2, "total": 2}],
    "collection": [
        {
            "doi": "10.1101/2026.03.15.123456",
            "title": "Neural organoid patterning screen",
            "authors": "Smith, J.; Chen, L.",
            "abstract": "We performed single-cell RNA-seq on neural organoids.",
            "date": "2026-03-15",
            "server": "biorxiv",
            "category": "neuroscience",
        },
        {
            "doi": "10.1101/2026.03.14.999999",
            "title": "Unrelated protein folding paper",
            "authors": "Jones, A.",
            "abstract": "Protein structure prediction using deep learning.",
            "date": "2026-03-14",
            "server": "biorxiv",
            "category": "bioinformatics",
        },
    ],
}


class TestBioRxivScraper:
    def test_parse_and_filter(self):
        scraper = BioRxivScraper()
        with patch("literature.scrapers.biorxiv.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: MOCK_BIORXIV_RESPONSE)
            mock_get.return_value.raise_for_status = lambda: None
            results = scraper.search("neural organoid", max_results=10)
        assert len(results) == 1  # Only the neural organoid paper matches
        assert results[0].source == "biorxiv"
        assert "10.1101" in results[0].doi


# ---------------------------------------------------------------------------
# Zenodo scraper
# ---------------------------------------------------------------------------


class TestZenodoScraper:
    def test_parse_results(self):
        scraper = ZenodoScraper()
        mock_response = {
            "hits": {"hits": [{
                "doi": "10.5281/zenodo.12345",
                "metadata": {"title": "Brain atlas h5ad", "description": "scRNA-seq atlas"},
                "files": [{"key": "atlas.h5ad", "size": 1000000,
                           "links": {"self": "https://zenodo.org/files/atlas.h5ad"}}],
            }]}
        }
        with patch("literature.scrapers.dataset_sources.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response)
            mock_get.return_value.raise_for_status = lambda: None
            results = scraper.search("brain atlas h5ad")
        assert len(results) >= 1
        assert results[0].source == "zenodo"
        assert results[0].format == "h5ad"


# ---------------------------------------------------------------------------
# GEO scraper
# ---------------------------------------------------------------------------


class TestGEOScraper:
    def test_parse_results(self):
        scraper = GEOScraper()
        with patch("literature.scrapers.dataset_sources.Entrez") as mock_entrez:
            mock_entrez.esearch.return_value = MagicMock()
            mock_entrez.read.return_value = {"IdList": ["200233574"]}
            mock_entrez.esummary.return_value = MagicMock()
            # esummary returns a list of dicts
            mock_summary = [{"Accession": "GSE233574", "title": "Brain organoid data",
                            "summary": "scRNA-seq of brain organoids",
                            "taxon": "Homo sapiens", "n_samples": "46"}]
            mock_entrez.read.side_effect = [
                {"IdList": ["200233574"]},
                mock_summary,
            ]
            results = scraper.search("brain organoid")
        assert len(results) >= 1
        assert results[0].accession == "GSE233574"
