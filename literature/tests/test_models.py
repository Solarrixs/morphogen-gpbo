"""Tests for SQLAlchemy ORM models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from literature.models import Base, Dataset, Paper, SearchRun


@pytest.fixture
def db_session():
    """In-memory SQLite engine with all tables created; yields a Session."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


class TestPaper:
    def test_create_paper(self, db_session):
        paper = Paper(
            doi="10.1234/test.001",
            title="A test paper about brain organoids",
            source="pubmed",
        )
        db_session.add(paper)
        db_session.commit()

        assert paper.id is not None
        assert paper.discovered_at is not None
        assert paper.doi == "10.1234/test.001"
        assert paper.title == "A test paper about brain organoids"
        assert paper.source == "pubmed"

    def test_doi_unique_constraint(self, db_session):
        paper1 = Paper(doi="10.1234/dup.001", title="First paper", source="pubmed")
        paper2 = Paper(doi="10.1234/dup.001", title="Second paper", source="biorxiv")
        db_session.add(paper1)
        db_session.commit()

        db_session.add(paper2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_paper_without_doi_allowed(self, db_session):
        paper = Paper(title="No DOI paper", source="biorxiv")
        db_session.add(paper)
        db_session.commit()

        assert paper.id is not None
        assert paper.doi is None

    def test_status_default_pending(self, db_session):
        paper = Paper(title="Status check paper", source="pubmed")
        db_session.add(paper)
        db_session.commit()

        assert paper.status == "pending"


class TestDataset:
    def test_create_dataset(self, db_session):
        dataset = Dataset(accession="GSE123456", source="geo")
        db_session.add(dataset)
        db_session.commit()

        assert dataset.id is not None
        assert dataset.accession == "GSE123456"
        assert dataset.source == "geo"
        assert dataset.discovered_at is not None

    def test_dataset_linked_to_paper(self, db_session):
        paper = Paper(title="Paper with dataset", source="pubmed")
        db_session.add(paper)
        db_session.commit()

        dataset = Dataset(
            accession="GSE999999",
            source="geo",
            paper_id=paper.id,
        )
        db_session.add(dataset)
        db_session.commit()

        assert dataset.paper_id == paper.id
        assert dataset.paper is paper


class TestSearchRun:
    def test_create_search_run(self, db_session):
        run = SearchRun(
            source="pubmed",
            query="brain organoid scRNA-seq",
            n_results=42,
            n_new=5,
        )
        db_session.add(run)
        db_session.commit()

        assert run.id is not None
        assert run.timestamp is not None
        assert run.source == "pubmed"
        assert run.query == "brain organoid scRNA-seq"
        assert run.n_results == 42
        assert run.n_new == 5
