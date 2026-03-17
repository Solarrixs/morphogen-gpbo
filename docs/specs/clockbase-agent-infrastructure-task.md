# Task: Build ClockBase-Inspired Agent Infrastructure for GP-BO Pipeline

## Complexity: complex

## Context

Branch: ralph/agent-infrastructure (create from ralph/production-readiness-phase2)
Tests: 571 gopro tests, 0 failures
Venv: source .venv/bin/activate
Test cmd: python -m pytest gopro/tests/ -v
Anthropic API: available (user has key)

### Motivation

The ClockBase paper (Tyshkovskiy et al. 2025) demonstrates a multi-agent system for automated biological age analysis at scale (15,380 agent-study interactions, 206,543 code execution blocks). Three key architectural patterns are directly applicable to the morphogen-gpbo pipeline:

1. **Multi-agent orchestration**: ClockBase uses 3 agents (executor, interpreter, scorer) working in pipeline. We need analogous agents for GP-BO: a pipeline executor, a biological interpreter that gates/modifies recommendations, and a scoring/QC agent.
2. **RAG + knowledge graph**: ClockBase built a 6,424-paper corpus with Neo4j knowledge graph, MinIO object storage, and Redis task queues. We need a morphogen-focused version for biological reasoning.
3. **Composite scoring with plausibility filters**: ClockBase scores interventions on 5 weighted dimensions with plausibility gates. We need analogous scoring for GP-BO recommendations before wet-lab execution.

### Architecture Overview

```
gopro/agents/                    # New module
├── __init__.py
├── config.py                    # Agent config (API keys, model selection, prompts)
├── orchestrator.py              # Pipeline orchestrator agent (runs steps 00-06)
├── interpreter.py               # Biological interpreter agent (gates/modifies recs)
├── scorer.py                    # Recommendation scoring agent (multi-criteria)
├── rag/
│   ├── __init__.py
│   ├── corpus_builder.py        # Semantic Scholar API → paper ingestion
│   ├── embedder.py              # Text embedding (voyage-3 or local)
│   ├── retriever.py             # Hybrid search (semantic + keyword)
│   └── prompts.py               # RAG prompt templates
├── knowledge_graph/
│   ├── __init__.py
│   ├── schema.py                # Neo4j node/relationship definitions
│   ├── loader.py                # Populate KG from papers, pathway DBs, ChEMBL
│   ├── queries.py               # Cypher query library
│   └── reasoning.py             # KG-augmented biological reasoning
└── tests/
    ├── test_orchestrator.py
    ├── test_interpreter.py
    ├── test_scorer.py
    ├── test_rag.py
    └── test_knowledge_graph.py
```

### Existing Infrastructure to Build On

- `gopro/orchestrator.py` — Pipeline orchestrator already exists (27 tests, Phase 4B complete)
- `gopro/validation.py` — Inter-step validators (10 tests, Phase 1A complete)
- `literature/` — Scraper design spec at `docs/specs/2026-03-15-literature-scrapers-design.md` (SQLite, PubMed/bioRxiv/GEO/Zenodo/CellxGene scrapers designed but NOT built)
- `gopro/config.py` — Centralized config with env var overrides
- `MORPHOGEN_COLUMNS`, `PROTEIN_MW_KDA` — Canonical morphogen metadata in config

### Key ClockBase Design Decisions to Adapt

| ClockBase | GP-BO Adaptation |
|-----------|-----------------|
| Claude 3.7 Sonnet + GPT-4O | Claude claude-sonnet-4-6 via Anthropic SDK (user has API key) |
| 3 agents: executor, interpreter, scorer | 3 agents: pipeline executor, biological interpreter, recommendation scorer |
| Neo4j knowledge graph | Neo4j (local Docker or Aura Free) for morphogen→pathway→cell_type→paper |
| MinIO object storage | Local filesystem (data/ already gitignored, 48GB Mac) |
| Redis task queue | Python asyncio + queue (lightweight, no extra infra) |
| RAG with 6,424 aging papers | RAG with brain organoid + morphogen signaling + fetal brain atlas papers |
| Semantic Scholar API for discovery | Semantic Scholar + PubMed MCP + bioRxiv MCP (already available) |
| Composite scoring (0-100, 5 dimensions) | Adapted scoring (0-100, 4 dimensions + plausibility gate) |

## Rules

- Run tests after EVERY subtask: python -m pytest gopro/tests/ -v
- Tests must be GENUINE — test real problems, never rig to pass
- Import constants from gopro.config — never hardcode paths or columns
- Use .copy() before mutating DataFrames
- Read files BEFORE modifying them
- Keep solutions minimal — don't over-engineer
- Use `anthropic` Python SDK for all Claude API calls
- Neo4j driver: `neo4j` Python package (official driver)
- All agent prompts stored as constants in `agents/config.py`, NOT inline strings
- Agent responses must be structured (JSON mode or tool_use), never free-text parsed with regex
- Every agent action must be logged with `get_logger(__name__)` from gopro.config
- RAG retrieval must include source attribution (paper DOI + relevant passage)
- Knowledge graph schema changes require migration scripts

## Subtasks

### Phase 1: Recommendation Scoring Framework (no external deps)

- [ ] P1-1: Design scoring schema — Define `RecommendationScore` dataclass with 4 dimensions: biological_plausibility (0-25), novelty_vs_redundancy (0-25), practical_feasibility (0-25), predicted_fidelity (0-25), plus plausibility_penalty (-50 to +30) and composite (0-100). Write to `gopro/agents/scorer.py`. | Acceptance: dataclass with `compute_composite()` method; 3+ tests
- [ ] P1-2: Implement novelty scoring — Given a candidate morphogen vector and the existing training set, compute: (a) min Euclidean distance to any tested condition, (b) mean distance to k=5 nearest tested conditions, (c) number of novel non-zero morphogens not yet explored. Map to 0-25 score. | Acceptance: novelty scores for Round 1 recommendations match intuition; 4+ tests
- [ ] P1-3: Implement feasibility scoring — Score based on: (a) morphogen cost (lookup table for $/µg for each of 24 morphogens), (b) concentration vs toxicity threshold (literature LD50 or max-tested values), (c) number of morphogens in cocktail (complexity penalty), (d) availability (common vs custom synthesis). | Acceptance: feasibility scores distinguish cheap/simple from expensive/complex; 3+ tests
- [ ] P1-4: Implement predicted fidelity scoring — Extract GP posterior mean and variance for each recommendation. Score: high mean + low variance = 25, high mean + high variance = 15 (risky bet), low mean = 0-5. Normalize across batch. | Acceptance: scoring uses actual GP predictions from gp_recommendations CSV; 3+ tests
- [ ] P1-5: Implement plausibility filter — Rule-based gate using morphogen biology: (a) WNT agonist + WNT antagonist at high dose = implausible (-50), (b) SHH + cyclopamine at high dose = implausible (-50), (c) known synergistic pairs at moderate dose = bonus (+10-30). Encode rules as a YAML config. | Acceptance: filter catches 2+ known antagonistic combinations; plausibility rules are YAML-configurable; 4+ tests
- [ ] P1-6: Wire scoring into GP-BO loop — After `recommend_next_experiments()` in `04_gpbo_loop.py`, call scorer on each recommendation. Add scores to output CSV. Add `--score-recommendations` flag (default: on). | Acceptance: `gp_recommendations_round{N}.csv` includes score columns; 3+ tests
- [ ] P1-7: Add scored leaderboard to visualization — New Plotly figure in `visualize_report.py` showing recommendations ranked by composite score with dimension breakdown. Color-code plausibility flags. | Acceptance: figure renders in HTML report; 2+ tests

### Phase 2: RAG Literature Corpus (requires Anthropic API)

- [ ] P2-1: Corpus builder via Semantic Scholar — `gopro/agents/rag/corpus_builder.py`: Query Semantic Scholar API for papers matching: "brain organoid morphogen", "neural differentiation protocol", "fetal brain atlas single-cell", "WNT BMP SHH signaling neural". Paginate through results, extract: DOI, title, authors, abstract, year, venue, citation count. Store as JSON-lines in `data/literature/corpus.jsonl`. Rate-limit to 100 req/sec (S2 API limit). | Acceptance: downloads 500+ papers; deduplicates by DOI; 4+ tests (with mocked API)
- [ ] P2-2: Text embedding pipeline — `gopro/agents/rag/embedder.py`: Embed paper abstracts + titles using Claude API (or voyage-3 if available). Chunk long abstracts at 512 tokens. Store embeddings in a local vector store (ChromaDB or LanceDB — pick whichever has lighter deps for macOS). | Acceptance: 500+ papers embedded; retrieval returns relevant papers for "SHH concentration dorsal forebrain"; 3+ tests
- [ ] P2-3: Hybrid retriever — `gopro/agents/rag/retriever.py`: Combine semantic search (vector similarity) with keyword search (BM25 on title+abstract). Reciprocal rank fusion to merge results. Return top-k with scores and source attribution. | Acceptance: hybrid beats pure semantic on 5 test queries; 4+ tests
- [ ] P2-4: RAG prompt templates — `gopro/agents/rag/prompts.py`: System prompts for: (a) "Given these morphogen recommendations, retrieve relevant literature on each morphogen's role in neural development", (b) "For this signaling pathway combination, find evidence of synergy or antagonism", (c) "What concentration ranges have been used for {morphogen} in brain organoid protocols?". | Acceptance: prompts are parameterized Jinja2 templates; 2+ tests
- [ ] P2-5: CLI for corpus management — Add `python -m gopro.agents.rag {build,search,stats}` subcommands. `build` populates corpus, `search <query>` does retrieval, `stats` shows corpus size/coverage. | Acceptance: all 3 subcommands work; 2+ tests

### Phase 3: Neo4j Knowledge Graph

- [ ] P3-1: Neo4j schema design — `gopro/agents/knowledge_graph/schema.py`: Define node types: `Morphogen` (name, MW, type, pathway), `Pathway` (name, type: WNT/BMP/SHH/RA/FGF/Notch/EGF), `CellType` (name, region, layer), `Paper` (DOI, title, year), `Protocol` (condition_name, morphogen_vector, harvest_day). Relationships: `ACTIVATES`, `INHIBITS`, `PRODUCES`, `DESCRIBED_IN`, `TARGETS_REGION`, `SYNERGIZES_WITH`, `ANTAGONIZES_WITH`. Write as Python dataclasses + Cypher CREATE constraints. | Acceptance: schema creates all indexes/constraints in test Neo4j; 3+ tests
- [ ] P3-2: Seed KG from pipeline data — `gopro/agents/knowledge_graph/loader.py`: (a) Load `MORPHOGEN_COLUMNS` + `PROTEIN_MW_KDA` → create Morphogen nodes, (b) parse morphogen_parser.py condition handlers → create Protocol nodes + USES relationships, (c) load fidelity_report.csv → create CellType nodes + PRODUCES relationships, (d) hardcode core pathway relationships (WNT→dorsal telencephalon, SHH→ventral, BMP→choroid plexus, RA→hindbrain, etc.) from CLAUDE.md domain knowledge. | Acceptance: KG has 24 Morphogen + 7 Pathway + 48 Protocol + ~30 CellType nodes; 4+ tests
- [ ] P3-3: Seed KG from literature corpus — Extend loader to: (a) create Paper nodes from corpus.jsonl, (b) extract morphogen mentions from abstracts via regex/NER → MENTIONED_IN relationships, (c) extract cell type mentions → STUDIED_IN relationships. Use Claude API for NER if regex is insufficient. | Acceptance: 100+ Paper nodes linked to Morphogens and CellTypes; 3+ tests
- [ ] P3-4: Cypher query library — `gopro/agents/knowledge_graph/queries.py`: Pre-built queries for: (a) "What pathways does morphogen X activate?", (b) "Which morphogens produce cell type Y?", (c) "Find antagonistic morphogen pairs", (d) "Which papers describe protocol Z?", (e) "What concentration ranges are reported for morphogen X?", (f) "What cell types does pathway P produce in region R?". Return typed Python objects. | Acceptance: all 6 queries return correct results on seeded KG; 6+ tests
- [ ] P3-5: Docker Compose for Neo4j — `docker-compose.yml` at project root with Neo4j 5.x Community Edition. Persist data to `data/neo4j/`. Add setup instructions to README. Configure APOC plugin for path algorithms. | Acceptance: `docker compose up` starts Neo4j; loader populates KG; queries work; documented in README

### Phase 4: Biological Interpreter Agent (requires Phase 2 + 3)

- [ ] P4-1: Interpreter core — `gopro/agents/interpreter.py`: Claude API agent that receives GP-BO recommendations (24 morphogen vectors) and for each: (a) queries KG for pathway activation pattern, (b) retrieves relevant literature via RAG, (c) assesses biological plausibility, (d) generates natural language explanation. Returns structured JSON: `{recommendation_idx, plausibility: "high"|"medium"|"low", explanation: str, concerns: [str], suggested_modifications: [{morphogen, current_conc, suggested_conc, reason}]}`. | Acceptance: interpreter produces structured output for 5 test recommendations; concerns are biologically grounded; 4+ tests
- [ ] P4-2: Gate mode — Add `--gate-mode strict` to interpreter: in strict mode, recommendations scored "low" plausibility are either (a) dropped and replaced with next-best from acquisition function, or (b) modified per `suggested_modifications`. Log all modifications. Require user confirmation for modifications >20% concentration change. | Acceptance: strict mode drops/modifies at least 1 implausible recommendation in test set; 3+ tests
- [ ] P4-3: Concentration safety bounds — Build concentration safety table from KG + literature: for each morphogen, extract max concentration used in published brain organoid protocols + any reported toxicity thresholds. Interpreter flags recommendations exceeding safety bounds. Store as `data/morphogen_safety_bounds.yaml`. | Acceptance: safety table covers all 24 morphogens; interpreter flags concentrations above bounds; 3+ tests
- [ ] P4-4: Pathway conflict detection — Using KG ANTAGONIZES_WITH relationships: detect when a recommendation activates contradictory pathways at high doses (e.g., dorsalizing + ventralizing simultaneously above threshold). Distinguish intentional (known patterning strategy) from accidental. | Acceptance: detects WNT+SHH conflict in test cases; does NOT flag known valid combos like CHIR+SAG at low dose; 4+ tests
- [ ] P4-5: Integration with scoring — Wire interpreter plausibility assessment into scorer (P1-5). Replace rule-based plausibility with KG+RAG-informed assessment when agents are available, fall back to rules when offline. | Acceptance: scorer uses interpreter when available, rules when not; scores differ meaningfully; 2+ tests

### Phase 5: Pipeline Orchestrator Agent (requires Phase 1-4)

- [ ] P5-1: Extend existing orchestrator — `gopro/orchestrator.py` already handles step execution. Add Claude API agent wrapper that: (a) reads pipeline state (which steps have been run, what data exists), (b) determines next action, (c) executes step with appropriate arguments, (d) validates output via `validation.py`, (e) decides whether to proceed or retry. | Acceptance: agent can run steps 03→04→scorer→interpreter in sequence; 3+ tests
- [ ] P5-2: End-to-end autonomous mode — `--autonomous` flag: given new scRNA-seq data, agent runs the full pipeline: map → score → GP-BO → score recommendations → interpret → generate report. Checkpoint after each step. Resume from checkpoint on failure. | Acceptance: autonomous mode completes on existing data with mocked new input; 3+ tests
- [ ] P5-3: Agent action logging — Log all agent decisions, tool calls, and state transitions to `data/agent_log_round{N}.jsonl` (matching ClockBase's approach of tracking 206K+ code execution blocks). Include: action category, timestamp, input summary, output summary, tokens used, latency. | Acceptance: log file captures full session; 2+ tests
- [ ] P5-4: Agent workflow analysis — After autonomous run, generate summary statistics: (a) action category distribution (like ClockBase's 10 categories), (b) total tokens used, (c) total latency, (d) steps that required retry, (e) recommendations that were modified by interpreter. Add to visualization report. | Acceptance: workflow stats in report; 2+ tests

### Phase 6: Polish + Integration

- [ ] P6-1: Unified CLI — Add top-level `python -m gopro.agents {run,score,interpret,rag,kg}` CLI. `run` = full pipeline, `score` = score existing recommendations, `interpret` = biological interpretation of existing recommendations, `rag` = corpus management, `kg` = knowledge graph management. | Acceptance: all subcommands documented with --help; 2+ tests
- [ ] P6-2: Configuration file — `gopro/agents/config.yaml`: API keys (ref env vars), model selection, scoring weights, plausibility rules, RAG parameters, Neo4j connection, logging level. All overridable via env vars. | Acceptance: config loads with sensible defaults; env vars override; 2+ tests
- [ ] P6-3: /simplify pass — Run 3-agent code review on all Phase 1-5 changes. Fix HIGH/MEDIUM issues. | Acceptance: all tests pass after fixes
- [ ] P6-4: /bug-hunter sweep — Adversarial QA swarm across gopro/agents/. Fix confirmed criticals. | Acceptance: no confirmed critical bugs remain

## Dependencies

- Phase 1 (scoring) has NO external dependencies — can start immediately
- Phase 2 (RAG) requires: `anthropic` SDK, `chromadb` or `lancedb`, `requests` (for Semantic Scholar API)
- Phase 3 (KG) requires: `neo4j` Python driver, Docker (for Neo4j)
- Phase 4 (interpreter) depends on Phase 2 + 3
- Phase 5 (orchestrator) depends on Phase 1-4
- Phase 6 (polish) depends on Phase 1-5

## Estimated Scope

- ~25 subtasks across 6 phases
- ~15 new files in gopro/agents/
- ~80-120 new tests
- Key new dependencies: anthropic, neo4j, chromadb/lancedb, requests
- Infrastructure: Docker (Neo4j only)

## References

- ClockBase paper: Tyshkovskiy et al. 2025 (methods section provided by user)
- Existing literature scraper spec: `docs/specs/2026-03-15-literature-scrapers-design.md`
- Existing pipeline orchestrator: `gopro/orchestrator.py` (Phase 4B, 27 tests)
- Existing validation: `gopro/validation.py` (Phase 1A, 10 tests)
- Competitive landscape: `docs/plans/competitive_landscape_ideas_index.md` (52 ideas from 9 papers)
- ClockBase scoring: 5 dimensions (model quality 0-25, aging relevance 0-25, rigor 0-10, translational 0-5, novelty 0-35) + plausibility gate (-50/+30) → adapted to 4 dimensions (plausibility 0-25, novelty 0-25, feasibility 0-25, predicted fidelity 0-25) + plausibility gate
