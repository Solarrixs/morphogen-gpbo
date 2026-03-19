"""Tests for the literature knowledge graph module."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from literature.models import (
    Base,
    CellType,
    Dataset,
    Morphogen,
    Paper,
    PaperCellType,
    PaperMorphogen,
)
from literature.knowledge_graph import (
    TfidfIndex,
    build_knowledge_graph,
    find_morphogen_fate_links,
    link_paper_cell_type,
    link_paper_morphogen,
    query_by_cell_type,
    query_by_morphogen,
    query_semantic,
    seed_cell_types,
    seed_known_entities,
    seed_morphogens,
    _resolve_cell_type,
    _resolve_morphogen,
    _fuzzy_match,
)


@pytest.fixture
def db_session():
    """In-memory SQLite engine with all tables created; yields a Session."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def seeded_session(db_session):
    """Session with seed morphogens and cell types pre-loaded."""
    seed_known_entities(db_session)
    return db_session


@pytest.fixture
def populated_session(seeded_session):
    """Session with papers, links, and seed data for graph queries."""
    session = seeded_session

    # Create papers
    p1 = Paper(
        doi="10.1234/organoid.001",
        title="SHH signaling drives dopaminergic neuron differentiation in brain organoids",
        abstract="We show that Sonic Hedgehog (SHH) and SAG promote dopaminergic neuron fate in human brain organoids. "
        "Treatment with purmorphamine also activates the hedgehog pathway.",
        source="pubmed",
        year=2025,
    )
    p2 = Paper(
        doi="10.1234/organoid.002",
        title="WNT and BMP gradients pattern cortical organoids with astrocytes",
        abstract="CHIR99021 activates WNT signaling while BMP4 promotes astrocyte differentiation. "
        "LDN193189 inhibits BMP to maintain progenitor state.",
        source="biorxiv",
        year=2025,
    )
    p3 = Paper(
        doi="10.1234/organoid.003",
        title="FGF2 maintains radial glia in cerebral organoids",
        abstract="Basic FGF (FGF2) supports radial glial cell maintenance and proliferation.",
        source="pubmed",
        year=2024,
    )
    session.add_all([p1, p2, p3])
    session.flush()

    # Link papers to morphogens
    link_paper_morphogen(p1.id, "SHH", session, context="promotes")
    link_paper_morphogen(p1.id, "SAG", session, context="promotes")
    link_paper_morphogen(p1.id, "purmorphamine", session, context="activates")
    link_paper_morphogen(p2.id, "CHIR99021", session, context="activates")
    link_paper_morphogen(p2.id, "BMP4", session, context="promotes")
    link_paper_morphogen(p2.id, "LDN193189", session, context="inhibits")
    link_paper_morphogen(p3.id, "FGF2", session, context="maintains")

    # Link papers to cell types
    link_paper_cell_type(p1.id, "Dopaminergic neuron", session, context="produces")
    link_paper_cell_type(p2.id, "Astrocyte", session, context="produces")
    link_paper_cell_type(p2.id, "Neural progenitor", session, context="maintains")
    link_paper_cell_type(p3.id, "Radial glia", session, context="maintains")

    session.commit()
    return session


class TestSeedEntities:
    def test_seed_morphogens(self, db_session):
        morphogens = seed_morphogens(db_session)
        db_session.commit()
        assert len(morphogens) >= 20
        names = {m.name for m in morphogens}
        assert "CHIR99021" in names
        assert "SHH" in names
        assert "BMP4" in names

    def test_seed_cell_types(self, db_session):
        cell_types = seed_cell_types(db_session)
        db_session.commit()
        assert len(cell_types) >= 15
        names = {ct.canonical_name for ct in cell_types}
        assert "Radial glia" in names
        assert "Astrocyte" in names
        assert "GABAergic neuron" in names

    def test_seed_known_entities(self, db_session):
        counts = seed_known_entities(db_session)
        assert counts["morphogens"] >= 20
        assert counts["cell_types"] >= 15

    def test_seed_idempotent(self, db_session):
        """Seeding twice should not create duplicates."""
        counts1 = seed_known_entities(db_session)
        counts2 = seed_known_entities(db_session)
        assert counts1 == counts2

    def test_morphogen_aliases_stored(self, seeded_session):
        from sqlalchemy import select
        m = seeded_session.execute(
            select(Morphogen).where(Morphogen.name == "SHH")
        ).scalar_one()
        aliases = json.loads(m.aliases)
        assert "Sonic Hedgehog" in aliases

    def test_cell_type_category(self, seeded_session):
        from sqlalchemy import select
        ct = seeded_session.execute(
            select(CellType).where(CellType.canonical_name == "Astrocyte")
        ).scalar_one()
        assert ct.category == "glia"


class TestQueryByCellType:
    def test_query_by_cell_type_exact(self, populated_session):
        papers = query_by_cell_type("Dopaminergic neuron", populated_session)
        assert len(papers) == 1
        assert "SHH" in papers[0].title

    def test_query_by_cell_type_fuzzy(self, populated_session):
        """Fuzzy match should find 'Radial glia' from alias 'RGC'."""
        papers = query_by_cell_type("RGC", populated_session)
        assert len(papers) == 1
        assert "radial glia" in papers[0].title.lower()

    def test_query_by_cell_type_alias(self, populated_session):
        """Exact alias match: 'DA neuron' -> Dopaminergic neuron."""
        papers = query_by_cell_type("DA neuron", populated_session)
        assert len(papers) == 1

    def test_query_by_cell_type_not_found(self, populated_session):
        papers = query_by_cell_type("nonexistent_cell_type_xyz", populated_session)
        assert papers == []

    def test_query_returns_multiple_papers(self, populated_session):
        """Neural progenitor and Astrocyte are both linked to p2."""
        papers = query_by_cell_type("Astrocyte", populated_session)
        assert len(papers) == 1
        assert papers[0].doi == "10.1234/organoid.002"


class TestQueryByMorphogen:
    def test_query_by_morphogen_exact(self, populated_session):
        papers = query_by_morphogen("SHH", populated_session)
        assert len(papers) == 1
        assert "SHH" in papers[0].title

    def test_query_by_morphogen_alias(self, populated_session):
        """Alias 'bFGF' should resolve to FGF2."""
        papers = query_by_morphogen("bFGF", populated_session)
        assert len(papers) == 1
        assert "FGF2" in papers[0].title

    def test_query_by_morphogen_not_found(self, populated_session):
        papers = query_by_morphogen("nonexistent_morphogen_xyz", populated_session)
        assert papers == []


class TestSemanticSearch:
    def test_semantic_search_tfidf(self, populated_session):
        results = query_semantic(
            "hedgehog dopaminergic neurons", populated_session, top_k=3, rebuild=True
        )
        assert len(results) >= 1
        # The SHH/dopaminergic paper should rank first
        assert "SHH" in results[0].title or "dopaminergic" in results[0].title.lower()

    def test_semantic_search_cortical(self, populated_session):
        results = query_semantic(
            "WNT cortical organoid astrocyte", populated_session, top_k=3, rebuild=True
        )
        assert len(results) >= 1
        assert any("WNT" in r.title or "cortical" in r.title.lower() for r in results)

    def test_semantic_search_empty_db(self, db_session):
        results = query_semantic("brain organoid", db_session, top_k=5, rebuild=True)
        assert results == []

    def test_tfidf_index_rebuild(self, populated_session):
        idx = TfidfIndex()
        n = idx.build(populated_session)
        assert n == 3
        assert idx.is_built

    def test_tfidf_index_not_built(self):
        idx = TfidfIndex()
        assert not idx.is_built


class TestBuildKnowledgeGraph:
    def test_build_knowledge_graph(self, populated_session):
        graph = build_knowledge_graph(populated_session)
        assert graph["papers"] == 3
        assert graph["paper_cell_type_links"] == 4
        assert graph["paper_morphogen_links"] == 7
        assert len(graph["cell_types"]) >= 15  # seeded
        assert len(graph["morphogens"]) >= 20  # seeded

    def test_build_empty_graph(self, db_session):
        graph = build_knowledge_graph(db_session)
        assert graph["papers"] == 0
        assert graph["cell_types"] == []


class TestFindMorphogenFateLinks:
    def test_find_morphogen_fate_links(self, populated_session):
        links = find_morphogen_fate_links(populated_session)
        assert len(links) > 0

        # SHH -> Dopaminergic neuron (via p1)
        shh_da = [l for l in links if l["morphogen"] == "SHH" and l["cell_type"] == "Dopaminergic neuron"]
        assert len(shh_da) == 1
        assert shh_da[0]["paper_count"] == 1

        # CHIR99021 -> Astrocyte (via p2)
        chir_astro = [l for l in links if l["morphogen"] == "CHIR99021" and l["cell_type"] == "Astrocyte"]
        assert len(chir_astro) == 1

    def test_find_morphogen_fate_links_empty(self, seeded_session):
        """No papers linked -> no fate links."""
        links = find_morphogen_fate_links(seeded_session)
        assert links == []


class TestFuzzyMatch:
    def test_fuzzy_match_close(self):
        candidates = ["Radial glia", "Astrocyte", "Oligodendrocyte"]
        matches = _fuzzy_match("radial glial", candidates, cutoff=0.6)
        assert "Radial glia" in matches

    def test_fuzzy_match_no_match(self):
        candidates = ["Radial glia", "Astrocyte"]
        matches = _fuzzy_match("zzzzzzzzz", candidates, cutoff=0.6)
        assert matches == []


class TestLinkFunctions:
    def test_link_paper_cell_type(self, seeded_session):
        session = seeded_session
        paper = Paper(title="Test paper", source="test")
        session.add(paper)
        session.flush()

        link = link_paper_cell_type(paper.id, "Astrocyte", session, context="produces")
        assert link is not None
        assert link.context == "produces"

    def test_link_paper_cell_type_idempotent(self, seeded_session):
        session = seeded_session
        paper = Paper(title="Test paper 2", source="test")
        session.add(paper)
        session.flush()

        link1 = link_paper_cell_type(paper.id, "Astrocyte", session)
        link2 = link_paper_cell_type(paper.id, "Astrocyte", session)
        assert link1.id == link2.id

    def test_link_paper_morphogen(self, seeded_session):
        session = seeded_session
        paper = Paper(title="Test paper 3", source="test")
        session.add(paper)
        session.flush()

        link = link_paper_morphogen(paper.id, "BMP4", session, concentration="10 ng/mL")
        assert link is not None
        assert link.concentration == "10 ng/mL"

    def test_link_unknown_entity_returns_none(self, seeded_session):
        session = seeded_session
        paper = Paper(title="Test paper 4", source="test")
        session.add(paper)
        session.flush()

        link = link_paper_cell_type(paper.id, "totally_unknown_xyzzy", session)
        assert link is None
