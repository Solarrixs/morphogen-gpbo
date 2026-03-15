"""Tests for the scrape scheduler/orchestrator."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from literature.models import Base, Paper, SearchRun
from literature.scheduler import run_scrape, deduplicate_papers
from literature.scrapers.base import PaperResult


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session, db_path


class TestDeduplication:
    def test_dedup_by_doi(self):
        papers = [
            PaperResult(title="Paper A", source="pubmed", doi="10.1038/test"),
            PaperResult(title="Paper A copy", source="biorxiv", doi="10.1038/test"),
        ]
        deduped = deduplicate_papers(papers)
        assert len(deduped) == 1

    def test_dedup_by_title_similarity(self):
        papers = [
            PaperResult(title="Brain organoid single-cell atlas", source="pubmed"),
            PaperResult(title="Brain organoid single-cell atlas", source="biorxiv"),
        ]
        deduped = deduplicate_papers(papers)
        assert len(deduped) == 1

    def test_different_papers_kept(self):
        papers = [
            PaperResult(title="Paper A", source="pubmed", doi="10.1038/a"),
            PaperResult(title="Paper B", source="pubmed", doi="10.1038/b"),
        ]
        deduped = deduplicate_papers(papers)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert deduplicate_papers([]) == []


class TestRunScrape:
    def test_stores_papers_to_db(self, db_session):
        session, db_path = db_session
        config = {
            "db_path": str(db_path),
            "ncbi_api_key": "",
            "sources": {
                "pubmed": {
                    "enabled": True,
                    "queries": ["brain organoid"],
                    "max_results_per_query": 5,
                    "date_range_days": 7,
                },
                "biorxiv": {"enabled": False},
                "datasets": {
                    "geo": {"enabled": False},
                    "zenodo": {"enabled": False},
                    "cellxgene": {"enabled": False},
                },
            },
        }
        mock_results = [
            PaperResult(title="Test Paper", source="pubmed", doi="10.1038/test123",
                       abstract="single-cell RNA-seq of brain organoids"),
        ]
        with patch("literature.scheduler.PubMedScraper") as MockPubMed:
            instance = MockPubMed.return_value
            instance.search.return_value = mock_results
            run_scrape(config, db_path=db_path)

        # Verify paper was stored
        papers = session.query(Paper).all()
        assert len(papers) == 1
        assert papers[0].doi == "10.1038/test123"
        assert papers[0].status == "pending"
        assert papers[0].has_scrna_seq is True

    def test_skips_existing_doi(self, db_session):
        session, db_path = db_session
        # Pre-insert a paper
        existing = Paper(doi="10.1038/existing", title="Existing", source="pubmed", status="approved")
        session.add(existing)
        session.commit()

        config = {
            "db_path": str(db_path),
            "ncbi_api_key": "",
            "sources": {
                "pubmed": {
                    "enabled": True,
                    "queries": ["test"],
                    "max_results_per_query": 5,
                    "date_range_days": 7,
                },
                "biorxiv": {"enabled": False},
                "datasets": {
                    "geo": {"enabled": False},
                    "zenodo": {"enabled": False},
                    "cellxgene": {"enabled": False},
                },
            },
        }
        mock_results = [
            PaperResult(title="Existing paper", source="pubmed", doi="10.1038/existing"),
        ]
        with patch("literature.scheduler.PubMedScraper") as MockPubMed:
            instance = MockPubMed.return_value
            instance.search.return_value = mock_results
            run_scrape(config, db_path=db_path)

        # Should still be just 1 paper (no duplicate)
        papers = session.query(Paper).all()
        assert len(papers) == 1

    def test_logs_search_run(self, db_session):
        session, db_path = db_session
        config = {
            "db_path": str(db_path),
            "ncbi_api_key": "",
            "sources": {
                "pubmed": {
                    "enabled": True,
                    "queries": ["test query"],
                    "max_results_per_query": 5,
                    "date_range_days": 7,
                },
                "biorxiv": {"enabled": False},
                "datasets": {
                    "geo": {"enabled": False},
                    "zenodo": {"enabled": False},
                    "cellxgene": {"enabled": False},
                },
            },
        }
        with patch("literature.scheduler.PubMedScraper") as MockPubMed:
            instance = MockPubMed.return_value
            instance.search.return_value = []
            run_scrape(config, db_path=db_path)

        runs = session.query(SearchRun).all()
        assert len(runs) >= 1
        assert runs[0].source == "pubmed"
