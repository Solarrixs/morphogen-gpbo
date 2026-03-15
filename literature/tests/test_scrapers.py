"""Tests for literature scrapers."""
import pytest
from unittest.mock import patch, MagicMock
from literature.scrapers.pubmed import PubMedScraper
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
