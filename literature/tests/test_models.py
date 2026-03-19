"""Tests for ORM models and database setup."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from literature.models import Base, Paper, Dataset, SearchRun, paper_dataset
from literature.db import load_config, get_engine, init_db, get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    """Yield a Session bound to the in-memory database."""
    with Session(engine) as sess:
        yield sess


# ---------------------------------------------------------------------------
# Paper tests
# ---------------------------------------------------------------------------

def test_paper_creation(session):
    paper = Paper(
        title="Brain organoid morphogen screen",
        authors="Smith, J; Doe, A",
        source="pubmed",
        doi="10.1234/test.001",
    )
    session.add(paper)
    session.commit()

    assert paper.id is not None
    assert paper.title == "Brain organoid morphogen screen"
    assert paper.authors == "Smith, J; Doe, A"
    assert paper.source == "pubmed"
    assert paper.doi == "10.1234/test.001"
    assert paper.status == "pending"
    assert paper.discovered_at is not None


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_dataset_creation(session):
    ds = Dataset(
        name="Amin Kelley 2024 primary screen",
        accession="GSE233574",
        repository="geo",
        species="Homo sapiens",
        tissue="brain organoid",
        n_cells=50000,
    )
    session.add(ds)
    session.commit()

    assert ds.id is not None
    assert ds.name == "Amin Kelley 2024 primary screen"
    assert ds.accession == "GSE233574"
    assert ds.repository == "geo"
    assert ds.species == "Homo sapiens"
    assert ds.n_cells == 50000
    assert ds.status == "pending"
    assert ds.discovered_at is not None


# ---------------------------------------------------------------------------
# Paper <-> Dataset relationship
# ---------------------------------------------------------------------------

def test_paper_dataset_relationship(session):
    paper = Paper(
        title="Paper with datasets",
        authors="Author A",
        source="biorxiv",
    )
    ds = Dataset(
        name="Associated dataset",
        accession="GSE999999",
        repository="geo",
    )
    paper.datasets.append(ds)
    session.add(paper)
    session.commit()

    # Verify bidirectional relationship
    assert len(paper.datasets) == 1
    assert paper.datasets[0].accession == "GSE999999"
    assert len(ds.papers) == 1
    assert ds.papers[0].title == "Paper with datasets"


# ---------------------------------------------------------------------------
# SearchRun tests
# ---------------------------------------------------------------------------

def test_search_run_creation(session):
    run = SearchRun(
        source="pubmed",
        query="brain organoid scRNA-seq",
        n_results=42,
        n_new=5,
        n_duplicates=37,
    )
    session.add(run)
    session.commit()

    assert run.id is not None
    assert run.source == "pubmed"
    assert run.query == "brain organoid scRNA-seq"
    assert run.n_results == 42
    assert run.n_new == 5
    assert run.n_duplicates == 37
    assert run.started_at is not None
    assert run.completed_at is None
    assert run.error is None


# ---------------------------------------------------------------------------
# Database init tests
# ---------------------------------------------------------------------------

def test_init_db():
    eng = create_engine("sqlite:///:memory:", echo=False)
    returned_engine = init_db(engine=eng)
    assert returned_engine is eng

    inspector = inspect(eng)
    table_names = inspector.get_table_names()
    assert "papers" in table_names
    assert "datasets" in table_names
    assert "search_runs" in table_names
    assert "paper_dataset" in table_names
    eng.dispose()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_load_config():
    config = load_config()
    assert "database" in config
    assert "url" in config["database"]
    assert "pubmed" in config
    assert "biorxiv" in config
    assert "search_queries" in config
    assert "deduplication" in config


def test_env_var_override():
    env_overrides = {
        "LITERATURE_DB_URL": "sqlite:///test_override.db",
        "NCBI_API_KEY": "test_key_123",
        "NCBI_EMAIL": "test@example.com",
    }
    with patch.dict(os.environ, env_overrides):
        config = load_config()

    assert config["database"]["url"] == "sqlite:///test_override.db"
    assert config["pubmed"]["api_key"] == "test_key_123"
    assert config["pubmed"]["email"] == "test@example.com"
