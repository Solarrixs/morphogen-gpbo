"""SQLAlchemy ORM models for the literature scraping pipeline."""

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    pmid: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_scrna_seq: Mapped[bool] = mapped_column(Boolean, default=False)
    has_spatial: Mapped[bool] = mapped_column(Boolean, default=False)
    species: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON array stored as TEXT
    brain_regions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cell_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # JSON dict stored as TEXT
    accession_numbers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset", back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} doi={self.doi!r} title={self.title[:40]!r}>"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("papers.id"), nullable=True
    )
    accession: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    species: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cell_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    format: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    download_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    paper: Mapped[Optional["Paper"]] = relationship("Paper", back_populates="datasets")

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} accession={self.accession!r} source={self.source!r}>"


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    n_results: Mapped[int] = mapped_column(Integer, default=0)
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<SearchRun id={self.id} source={self.source!r} "
            f"query={self.query[:30]!r} n_results={self.n_results}>"
        )


# ---------------------------------------------------------------------------
# Knowledge graph entities
# ---------------------------------------------------------------------------


class CellType(Base):
    """Canonical cell type entity with aliases."""

    __tablename__ = "cell_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    # JSON list stored as TEXT, e.g. '["RGC", "radial glia"]'
    aliases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    paper_links: Mapped[List["PaperCellType"]] = relationship(
        "PaperCellType", back_populates="cell_type", cascade="all, delete-orphan"
    )
    dataset_links: Mapped[List["DatasetCellType"]] = relationship(
        "DatasetCellType", back_populates="cell_type", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<CellType id={self.id} name={self.canonical_name!r}>"


class Morphogen(Base):
    """Canonical morphogen entity."""

    __tablename__ = "morphogens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    pathway: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON list stored as TEXT
    aliases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    paper_links: Mapped[List["PaperMorphogen"]] = relationship(
        "PaperMorphogen", back_populates="morphogen", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Morphogen id={self.id} name={self.name!r}>"


class PaperCellType(Base):
    """Association: paper mentions/studies this cell type."""

    __tablename__ = "paper_cell_types"
    __table_args__ = (
        UniqueConstraint("paper_id", "cell_type_id", name="uq_paper_celltype"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id"), nullable=False)
    cell_type_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cell_types.id"), nullable=False
    )
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    paper: Mapped["Paper"] = relationship("Paper")
    cell_type: Mapped["CellType"] = relationship("CellType", back_populates="paper_links")


class PaperMorphogen(Base):
    """Association: paper uses this morphogen."""

    __tablename__ = "paper_morphogens"
    __table_args__ = (
        UniqueConstraint("paper_id", "morphogen_id", name="uq_paper_morphogen"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id"), nullable=False)
    morphogen_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("morphogens.id"), nullable=False
    )
    concentration: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    paper: Mapped["Paper"] = relationship("Paper")
    morphogen: Mapped["Morphogen"] = relationship("Morphogen", back_populates="paper_links")


class DatasetCellType(Base):
    """Association: dataset contains this cell type."""

    __tablename__ = "dataset_cell_types"
    __table_args__ = (
        UniqueConstraint("dataset_id", "cell_type_id", name="uq_dataset_celltype"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id"), nullable=False
    )
    cell_type_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cell_types.id"), nullable=False
    )
    fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset")
    cell_type: Mapped["CellType"] = relationship("CellType", back_populates="dataset_links")
