"""Database engine, session factory, and initialization helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session

from literature.models import Base

_DEFAULT_DB_PATH = str(Path(__file__).parent / "papers.db")


def get_engine(db_path: Optional[str] = None) -> Engine:
    """Return a SQLAlchemy engine.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``literature/papers.db`` relative to this package.

    Returns:
        A configured SQLAlchemy :class:`Engine`.
    """
    path = db_path or _DEFAULT_DB_PATH
    url = f"sqlite:///{path}"
    return create_engine(url, echo=False, future=True)


def init_db(db_path: Optional[str] = None) -> Engine:
    """Create all tables and return the engine.

    Args:
        db_path: Path to the SQLite database file (see :func:`get_engine`).

    Returns:
        The engine used to create the tables.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: Optional[str] = None) -> Session:
    """Return a new SQLAlchemy :class:`Session`.

    The caller is responsible for committing/rolling back and closing the
    session.

    Args:
        db_path: Path to the SQLite database file (see :func:`get_engine`).

    Returns:
        A new :class:`Session` bound to the database.
    """
    engine = init_db(db_path)
    return Session(engine)
