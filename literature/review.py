"""Interactive CLI to review pending papers in the literature database."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from literature.db import init_db
from literature.models import Paper


def review_papers(db_path: Optional[str] = None, limit: int = 50) -> None:
    """Interactive CLI to review pending papers.

    Fetches up to *limit* papers with status="pending", ordered by
    ``discovered_at`` descending (newest first), and presents each one for
    human review.  The reviewer can approve, reject, skip, or quit at each
    prompt.

    Args:
        db_path: Path to the SQLite database.  Uses the default
            ``literature/papers.db`` when not supplied.
        limit: Maximum number of papers to present in one session.
    """
    engine = init_db(db_path)

    with Session(engine) as session:
        pending = (
            session.query(Paper)
            .filter_by(status="pending")
            .order_by(Paper.discovered_at.desc())
            .limit(limit)
            .all()
        )

        if not pending:
            print("No pending papers to review.")
            return

        total = len(pending)
        for idx, paper in enumerate(pending, start=1):
            _print_paper(idx, total, paper)
            while True:
                try:
                    choice = input("  [a]pprove  [r]eject  [s]kip  [q]uit  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nQuitting review session.")
                    session.commit()
                    return

                if choice == "a":
                    paper.status = "approved"
                    paper.reviewed_at = datetime.now(timezone.utc)
                    print(f"  -> Approved.\n")
                    break
                elif choice == "r":
                    paper.status = "rejected"
                    paper.reviewed_at = datetime.now(timezone.utc)
                    print(f"  -> Rejected.\n")
                    break
                elif choice == "s":
                    print(f"  -> Skipped.\n")
                    break
                elif choice == "q":
                    print("Quitting review session.")
                    session.commit()
                    return
                else:
                    print("  Please enter a, r, s, or q.")

        session.commit()
        print(f"Review complete ({total} paper(s) presented).")


def _print_paper(idx: int, total: int, paper: Paper) -> None:
    """Print formatted paper details to stdout."""
    title = paper.title or "(no title)"
    year = f" ({paper.year})" if paper.year else ""
    authors = paper.authors or "(unknown)"
    journal = paper.journal or "(unknown)"
    doi = paper.doi or "(none)"
    abstract_snippet = ""
    if paper.abstract:
        abstract_snippet = paper.abstract[:300]
        if len(paper.abstract) > 300:
            abstract_snippet += "..."

    datasets_str = ""
    if paper.datasets:
        accessions = [d.accession for d in paper.datasets]
        datasets_str = f"\n         Datasets: {', '.join(accessions)}"

    print(
        f"\n[{idx}/{total}] \"{title}\"{year}\n"
        f"       Authors: {authors}\n"
        f"       Journal: {journal}\n"
        f"       DOI: {doi}\n"
        f"       Abstract: {abstract_snippet}"
        f"{datasets_str}\n"
    )
