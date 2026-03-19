"""SQLAlchemy ORM models for literature tracking."""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    ForeignKey, Table, create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
)


class Base(DeclarativeBase):
    pass


# Many-to-many: papers <-> datasets
paper_dataset = Table(
    "paper_dataset",
    Base.metadata,
    Column("paper_id", Integer, ForeignKey("papers.id")),
    Column("dataset_id", Integer, ForeignKey("datasets.id")),
)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    authors: Mapped[str] = mapped_column(Text)  # semicolon-separated
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    pmid: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, unique=True)
    biorxiv_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Review status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/approved/rejected/skipped
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source: Mapped[str] = mapped_column(String(50))  # pubmed/biorxiv/manual
    search_query: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    datasets: Mapped[list["Dataset"]] = relationship(
        secondary=paper_dataset, back_populates="papers"
    )

    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:50]}...', status='{self.status}')>"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    accession: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True)
    repository: Mapped[str] = mapped_column(String(50))  # geo/zenodo/cellxgene
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    species: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tissue: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    n_cells: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Review status
    status: Mapped[str] = mapped_column(String(20), default="pending")

    # Metadata
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    papers: Mapped[list["Paper"]] = relationship(
        secondary=paper_dataset, back_populates="datasets"
    )

    def __repr__(self):
        return f"<Dataset(id={self.id}, accession='{self.accession}', repository='{self.repository}')>"


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    query: Mapped[str] = mapped_column(String(500))
    n_results: Mapped[int] = mapped_column(Integer, default=0)
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    n_duplicates: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self):
        return f"<SearchRun(id={self.id}, source='{self.source}', query='{self.query[:30]}', n_new={self.n_new})>"
