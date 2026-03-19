"""Database initialization and session management."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Generator

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from literature.models import Base


def load_config(config_path: Path = None) -> dict:
    """Load config from YAML, with env var overrides."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Env var overrides
    if os.environ.get("LITERATURE_DB_URL"):
        config["database"]["url"] = os.environ["LITERATURE_DB_URL"]
    if os.environ.get("NCBI_API_KEY"):
        config["pubmed"]["api_key"] = os.environ["NCBI_API_KEY"]
    if os.environ.get("NCBI_EMAIL"):
        config["pubmed"]["email"] = os.environ["NCBI_EMAIL"]

    return config


def get_engine(db_url: str = None):
    """Create SQLAlchemy engine."""
    if db_url is None:
        config = load_config()
        db_url = config["database"]["url"]
    return create_engine(db_url, echo=False)


def init_db(engine=None):
    """Create all tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None) -> Generator[Session, None, None]:
    """Yield a database session."""
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
