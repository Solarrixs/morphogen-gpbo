"""Knowledge graph for cross-referencing literature entities.

Links papers <-> datasets <-> cell types <-> morphogens with semantic search
via TF-IDF embeddings.  No GPU required.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from literature.models import (
    Base,
    CellType,
    Dataset,
    DatasetCellType,
    Morphogen,
    Paper,
    PaperCellType,
    PaperMorphogen,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical seed data
# ---------------------------------------------------------------------------

# Morphogens derived from gopro/config.py MORPHOGEN_COLUMNS
_SEED_MORPHOGENS: List[Dict[str, Any]] = [
    {"name": "CHIR99021", "pathway": "WNT", "aliases": ["CHIR", "GSK3i"]},
    {"name": "BMP4", "pathway": "BMP", "aliases": ["bone morphogenetic protein 4"]},
    {"name": "BMP7", "pathway": "BMP", "aliases": ["bone morphogenetic protein 7"]},
    {"name": "SHH", "pathway": "Hedgehog", "aliases": ["Sonic Hedgehog", "sonic hedgehog"]},
    {"name": "SAG", "pathway": "Hedgehog", "aliases": ["Smoothened agonist", "smoothened agonist"]},
    {"name": "RA", "pathway": "Retinoic acid", "aliases": ["retinoic acid", "all-trans retinoic acid", "ATRA"]},
    {"name": "FGF8", "pathway": "FGF", "aliases": ["fibroblast growth factor 8"]},
    {"name": "FGF2", "pathway": "FGF", "aliases": ["bFGF", "basic FGF", "fibroblast growth factor 2"]},
    {"name": "FGF4", "pathway": "FGF", "aliases": ["fibroblast growth factor 4"]},
    {"name": "IWP2", "pathway": "WNT", "aliases": ["WNT inhibitor", "Porcupine inhibitor"]},
    {"name": "XAV939", "pathway": "WNT", "aliases": ["tankyrase inhibitor"]},
    {"name": "SB431542", "pathway": "TGF-beta", "aliases": ["SB", "ALK4/5/7 inhibitor"]},
    {"name": "LDN193189", "pathway": "BMP", "aliases": ["LDN", "ALK2/3 inhibitor"]},
    {"name": "DAPT", "pathway": "Notch", "aliases": ["gamma-secretase inhibitor", "Notch inhibitor"]},
    {"name": "EGF", "pathway": "EGF", "aliases": ["epidermal growth factor"]},
    {"name": "ActivinA", "pathway": "TGF-beta", "aliases": ["Activin A", "activin"]},
    {"name": "Dorsomorphin", "pathway": "BMP", "aliases": ["compound C", "BMP inhibitor"]},
    {"name": "purmorphamine", "pathway": "Hedgehog", "aliases": ["SHH agonist"]},
    {"name": "cyclopamine", "pathway": "Hedgehog", "aliases": ["SHH antagonist"]},
    {"name": "BDNF", "pathway": "neurotrophin", "aliases": ["brain-derived neurotrophic factor"]},
    {"name": "NT3", "pathway": "neurotrophin", "aliases": ["neurotrophin-3", "NTF3"]},
    {"name": "cAMP", "pathway": "cAMP", "aliases": ["dibutyryl-cAMP", "db-cAMP", "dbcAMP"]},
    {"name": "AscorbicAcid", "pathway": "antioxidant", "aliases": ["ascorbic acid", "vitamin C", "L-ascorbic acid 2-phosphate"]},
]

# Cell types from HNOCA annotation levels (representative subset)
_SEED_CELL_TYPES: List[Dict[str, Any]] = [
    # Progenitors
    {"canonical_name": "Radial glia", "category": "progenitor", "aliases": ["RG", "RGC", "radial glial cell"]},
    {"canonical_name": "Neural progenitor", "category": "progenitor", "aliases": ["NPC", "neural progenitor cell"]},
    {"canonical_name": "Intermediate progenitor", "category": "progenitor", "aliases": ["IP", "IPC", "basal progenitor"]},
    {"canonical_name": "Outer radial glia", "category": "progenitor", "aliases": ["oRG", "outer RG", "basal radial glia"]},
    {"canonical_name": "Cycling progenitor", "category": "progenitor", "aliases": ["dividing progenitor", "proliferating progenitor"]},
    # Neurons
    {"canonical_name": "Glutamatergic neuron", "category": "neuron", "aliases": ["excitatory neuron", "glutamatergic"]},
    {"canonical_name": "GABAergic neuron", "category": "neuron", "aliases": ["inhibitory neuron", "GABAergic", "interneuron"]},
    {"canonical_name": "Dopaminergic neuron", "category": "neuron", "aliases": ["DA neuron", "dopaminergic"]},
    {"canonical_name": "Serotonergic neuron", "category": "neuron", "aliases": ["5-HT neuron", "serotonergic"]},
    {"canonical_name": "Cholinergic neuron", "category": "neuron", "aliases": ["motor neuron", "cholinergic"]},
    {"canonical_name": "Cajal-Retzius cell", "category": "neuron", "aliases": ["CR cell"]},
    {"canonical_name": "Purkinje cell", "category": "neuron", "aliases": ["Purkinje neuron"]},
    {"canonical_name": "Granule cell", "category": "neuron", "aliases": ["granule neuron", "cerebellar granule"]},
    # Glia
    {"canonical_name": "Astrocyte", "category": "glia", "aliases": ["astroglia", "GFAP+"]},
    {"canonical_name": "Oligodendrocyte", "category": "glia", "aliases": ["oligo", "OL"]},
    {"canonical_name": "Oligodendrocyte precursor", "category": "glia", "aliases": ["OPC", "oligodendrocyte precursor cell"]},
    {"canonical_name": "Microglia", "category": "glia", "aliases": ["microglial cell"]},
    # Other
    {"canonical_name": "Choroid plexus", "category": "other", "aliases": ["ChP", "choroid plexus epithelium"]},
    {"canonical_name": "Mesenchymal cell", "category": "other", "aliases": ["mesenchyme"]},
    {"canonical_name": "Retinal pigment epithelium", "category": "other", "aliases": ["RPE"]},
]


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _fuzzy_match(query: str, candidates: Sequence[str], cutoff: float = 0.6) -> List[str]:
    """Return candidates matching *query* above *cutoff* via SequenceMatcher."""
    q = _normalize(query)
    matches = difflib.get_close_matches(q, [_normalize(c) for c in candidates], n=20, cutoff=cutoff)
    # Map normalized back to originals
    norm_to_orig = {_normalize(c): c for c in candidates}
    return [norm_to_orig[m] for m in matches]


def _get_aliases(entity) -> List[str]:
    """Parse JSON aliases field, returning empty list on failure."""
    if not entity.aliases:
        return []
    try:
        return json.loads(entity.aliases)
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_morphogens(session: Session) -> List[Morphogen]:
    """Insert seed morphogens if not already present. Returns all morphogens."""
    existing = {m.name for m in session.execute(select(Morphogen)).scalars().all()}
    added = []
    for spec in _SEED_MORPHOGENS:
        if spec["name"] not in existing:
            m = Morphogen(
                name=spec["name"],
                pathway=spec.get("pathway"),
                aliases=json.dumps(spec.get("aliases", [])),
            )
            session.add(m)
            added.append(m)
    if added:
        session.flush()
        logger.info("Seeded %d morphogens", len(added))
    return list(session.execute(select(Morphogen)).scalars().all())


def seed_cell_types(session: Session) -> List[CellType]:
    """Insert seed cell types if not already present. Returns all cell types."""
    existing = {
        ct.canonical_name
        for ct in session.execute(select(CellType)).scalars().all()
    }
    added = []
    for spec in _SEED_CELL_TYPES:
        if spec["canonical_name"] not in existing:
            ct = CellType(
                canonical_name=spec["canonical_name"],
                category=spec.get("category"),
                aliases=json.dumps(spec.get("aliases", [])),
            )
            session.add(ct)
            added.append(ct)
    if added:
        session.flush()
        logger.info("Seeded %d cell types", len(added))
    return list(session.execute(select(CellType)).scalars().all())


def seed_known_entities(session: Session) -> Dict[str, int]:
    """Seed all known morphogens and cell types.

    Returns:
        Dict with counts: {"morphogens": N, "cell_types": M}
    """
    morphogens = seed_morphogens(session)
    cell_types = seed_cell_types(session)
    session.commit()
    return {"morphogens": len(morphogens), "cell_types": len(cell_types)}


# ---------------------------------------------------------------------------
# Entity lookup with fuzzy matching
# ---------------------------------------------------------------------------


def _resolve_cell_type(name: str, session: Session, cutoff: float = 0.6) -> Optional[CellType]:
    """Find a CellType by canonical name or alias, with fuzzy fallback."""
    norm_name = _normalize(name)
    all_cts = list(session.execute(select(CellType)).scalars().all())

    # Exact match on canonical name
    for ct in all_cts:
        if _normalize(ct.canonical_name) == norm_name:
            return ct

    # Exact match on alias
    for ct in all_cts:
        for alias in _get_aliases(ct):
            if _normalize(alias) == norm_name:
                return ct

    # Fuzzy match on canonical names + aliases
    name_map: Dict[str, CellType] = {}
    for ct in all_cts:
        name_map[ct.canonical_name] = ct
        for alias in _get_aliases(ct):
            name_map[alias] = ct

    matches = _fuzzy_match(name, list(name_map.keys()), cutoff=cutoff)
    if matches:
        return name_map[matches[0]]
    return None


def _resolve_morphogen(name: str, session: Session, cutoff: float = 0.6) -> Optional[Morphogen]:
    """Find a Morphogen by name or alias, with fuzzy fallback."""
    norm_name = _normalize(name)
    all_morphs = list(session.execute(select(Morphogen)).scalars().all())

    # Exact match on name
    for m in all_morphs:
        if _normalize(m.name) == norm_name:
            return m

    # Exact match on alias
    for m in all_morphs:
        for alias in _get_aliases(m):
            if _normalize(alias) == norm_name:
                return m

    # Fuzzy
    name_map: Dict[str, Morphogen] = {}
    for m in all_morphs:
        name_map[m.name] = m
        for alias in _get_aliases(m):
            name_map[alias] = m

    matches = _fuzzy_match(name, list(name_map.keys()), cutoff=cutoff)
    if matches:
        return name_map[matches[0]]
    return None


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------


def query_by_cell_type(cell_type: str, session: Session) -> List[Paper]:
    """Find papers studying a given cell type (fuzzy match on aliases)."""
    ct = _resolve_cell_type(cell_type, session)
    if ct is None:
        return []
    rows = (
        session.execute(
            select(PaperCellType).where(PaperCellType.cell_type_id == ct.id)
        )
        .scalars()
        .all()
    )
    return [row.paper for row in rows]


def query_by_morphogen(morphogen: str, session: Session) -> List[Paper]:
    """Find papers using a given morphogen (fuzzy match on aliases)."""
    m = _resolve_morphogen(morphogen, session)
    if m is None:
        return []
    rows = (
        session.execute(
            select(PaperMorphogen).where(PaperMorphogen.morphogen_id == m.id)
        )
        .scalars()
        .all()
    )
    return [row.paper for row in rows]


def link_paper_cell_type(
    paper_id: int,
    cell_type_name: str,
    session: Session,
    context: Optional[str] = None,
) -> Optional[PaperCellType]:
    """Link a paper to a cell type (resolved by name/alias)."""
    ct = _resolve_cell_type(cell_type_name, session)
    if ct is None:
        logger.warning("Cell type %r not found, skipping link", cell_type_name)
        return None
    # Check for existing link
    existing = session.execute(
        select(PaperCellType).where(
            PaperCellType.paper_id == paper_id,
            PaperCellType.cell_type_id == ct.id,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    link = PaperCellType(paper_id=paper_id, cell_type_id=ct.id, context=context)
    session.add(link)
    session.flush()
    return link


def link_paper_morphogen(
    paper_id: int,
    morphogen_name: str,
    session: Session,
    concentration: Optional[str] = None,
    context: Optional[str] = None,
) -> Optional[PaperMorphogen]:
    """Link a paper to a morphogen (resolved by name/alias)."""
    m = _resolve_morphogen(morphogen_name, session)
    if m is None:
        logger.warning("Morphogen %r not found, skipping link", morphogen_name)
        return None
    existing = session.execute(
        select(PaperMorphogen).where(
            PaperMorphogen.paper_id == paper_id,
            PaperMorphogen.morphogen_id == m.id,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    link = PaperMorphogen(
        paper_id=paper_id,
        morphogen_id=m.id,
        concentration=concentration,
        context=context,
    )
    session.add(link)
    session.flush()
    return link


def link_dataset_cell_type(
    dataset_id: int,
    cell_type_name: str,
    session: Session,
    fraction: Optional[float] = None,
) -> Optional[DatasetCellType]:
    """Link a dataset to a cell type."""
    ct = _resolve_cell_type(cell_type_name, session)
    if ct is None:
        logger.warning("Cell type %r not found, skipping link", cell_type_name)
        return None
    existing = session.execute(
        select(DatasetCellType).where(
            DatasetCellType.dataset_id == dataset_id,
            DatasetCellType.cell_type_id == ct.id,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    link = DatasetCellType(
        dataset_id=dataset_id, cell_type_id=ct.id, fraction=fraction
    )
    session.add(link)
    session.flush()
    return link


# ---------------------------------------------------------------------------
# TF-IDF semantic search
# ---------------------------------------------------------------------------


class TfidfIndex:
    """In-memory TF-IDF index over paper titles and abstracts.

    Lightweight: sklearn only, no GPU needed. Papers < 10K means the
    matrix fits comfortably in RAM.
    """

    def __init__(self) -> None:
        self._paper_ids: List[int] = []
        self._vectorizer = None
        self._tfidf_matrix = None

    @property
    def is_built(self) -> bool:
        return self._tfidf_matrix is not None

    def build(self, session: Session) -> int:
        """Build TF-IDF index from all papers in the database.

        Returns:
            Number of papers indexed.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        papers = list(session.execute(select(Paper)).scalars().all())
        if not papers:
            self._paper_ids = []
            self._vectorizer = None
            self._tfidf_matrix = None
            return 0

        corpus = []
        self._paper_ids = []
        for p in papers:
            text = (p.title or "") + " " + (p.abstract or "")
            corpus.append(text.strip())
            self._paper_ids.append(p.id)

        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        logger.info("Built TF-IDF index: %d papers, %d features",
                     len(papers), self._tfidf_matrix.shape[1])
        return len(papers)

    def search(self, query: str, session: Session, top_k: int = 10) -> List[Paper]:
        """Search papers by text similarity.

        Args:
            query: Free-text search query.
            session: SQLAlchemy session for fetching Paper objects.
            top_k: Maximum number of results.

        Returns:
            List of Paper objects ordered by relevance (best first).
        """
        if not self.is_built or not self._paper_ids:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Filter out zero-score results
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            paper_id = self._paper_ids[idx]
            paper = session.get(Paper, paper_id)
            if paper is not None:
                results.append(paper)
        return results


# Module-level singleton index
_tfidf_index = TfidfIndex()


def query_semantic(
    query: str, session: Session, top_k: int = 10, rebuild: bool = False
) -> List[Paper]:
    """Semantic search using TF-IDF over paper titles and abstracts.

    Args:
        query: Free-text search query.
        session: SQLAlchemy session.
        top_k: Number of top results to return.
        rebuild: Force rebuild of the TF-IDF index.

    Returns:
        List of Paper objects ordered by relevance.
    """
    if rebuild or not _tfidf_index.is_built:
        _tfidf_index.build(session)
    return _tfidf_index.search(query, session, top_k=top_k)


# ---------------------------------------------------------------------------
# Knowledge graph summary
# ---------------------------------------------------------------------------


def build_knowledge_graph(session: Session) -> Dict[str, Any]:
    """Build a summary of the knowledge graph from existing DB entities.

    Returns:
        Dict with counts and entity lists:
            - papers: int
            - datasets: int
            - cell_types: list of canonical names
            - morphogens: list of names
            - paper_cell_type_links: int
            - paper_morphogen_links: int
            - dataset_cell_type_links: int
    """
    papers = session.execute(select(Paper)).scalars().all()
    datasets = session.execute(select(Dataset)).scalars().all()
    cell_types = session.execute(select(CellType)).scalars().all()
    morphogens = session.execute(select(Morphogen)).scalars().all()
    pct_links = session.execute(select(PaperCellType)).scalars().all()
    pm_links = session.execute(select(PaperMorphogen)).scalars().all()
    dct_links = session.execute(select(DatasetCellType)).scalars().all()

    return {
        "papers": len(list(papers)),
        "datasets": len(list(datasets)),
        "cell_types": [ct.canonical_name for ct in cell_types],
        "morphogens": [m.name for m in morphogens],
        "paper_cell_type_links": len(list(pct_links)),
        "paper_morphogen_links": len(list(pm_links)),
        "dataset_cell_type_links": len(list(dct_links)),
    }


def find_morphogen_fate_links(session: Session) -> List[Dict[str, Any]]:
    """Extract morphogen -> cell type fate links from the knowledge graph.

    Finds morphogens and cell types that are co-mentioned in the same paper,
    using the PaperMorphogen and PaperCellType association tables.

    Returns:
        List of dicts with keys: morphogen, cell_type, papers (list of titles),
        paper_count.
    """
    morphogens = list(session.execute(select(Morphogen)).scalars().all())
    cell_types = list(session.execute(select(CellType)).scalars().all())

    links: List[Dict[str, Any]] = []

    for morph in morphogens:
        # Papers linked to this morphogen
        pm_rows = (
            session.execute(
                select(PaperMorphogen).where(PaperMorphogen.morphogen_id == morph.id)
            )
            .scalars()
            .all()
        )
        morph_paper_ids = {row.paper_id for row in pm_rows}
        if not morph_paper_ids:
            continue

        for ct in cell_types:
            # Papers linked to this cell type
            pct_rows = (
                session.execute(
                    select(PaperCellType).where(PaperCellType.cell_type_id == ct.id)
                )
                .scalars()
                .all()
            )
            ct_paper_ids = {row.paper_id for row in pct_rows}

            shared = morph_paper_ids & ct_paper_ids
            if shared:
                shared_papers = [
                    session.get(Paper, pid) for pid in shared
                ]
                links.append({
                    "morphogen": morph.name,
                    "cell_type": ct.canonical_name,
                    "papers": [p.title for p in shared_papers if p],
                    "paper_count": len(shared),
                })

    # Sort by paper_count descending
    links.sort(key=lambda x: x["paper_count"], reverse=True)
    return links
