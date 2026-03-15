# Session Handoff — 2026-03-15

## User Profile & Working Style

**Maxx** is a hands-on scientist building a real wet-lab Bayesian optimization pipeline for brain organoid morphogen protocols. He has a 48GB Mac (daily driver) and a 4090/5080 GPU desktop available. He prefers:

- **Deep multi-source research** (20+ searches) before making decisions — academic sources, forums, niche blogs
- **Autonomous execution** — "execute autonomously, iterate continuously until you achieve your goals"
- **Parallel agents** — spawn research agents in background while doing implementation work
- **Auto-detect GPU** — all model code must use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, never hardcode
- **Genuine tests** — "don't rig or fix test cases, they should test real problems"
- **Use superpowers skills** — invoke relevant skills (TDD, verification, simplify, etc.) at each step
- **Interview with AskUserQuestion** before starting ambiguous work

## What Was Accomplished This Session (Chronologically)

### 1. Phase 1 Foundations — Completed ✅

Implemented a 3-track plan from `docs/plans/task_plan.md`:

**Track 1A: Inter-step data validation**
- Created `gopro/validation.py` with 4 validators: `validate_mapped_adata`, `validate_training_csvs`, `validate_temporal_atlas`, `validate_fidelity_report`
- Wired into steps 03 (before loading 11GB Braun data), 04 (`build_training_set`), 05 (`main()`), 06 (`run_virtual_screen`)
- 10 new tests in `gopro/tests/test_validation.py`
- Fixed 1 existing test (`test_mismatched_indices`) whose fractions didn't sum to 1.0

**Track 1B: Decomposed `project_query_forward()` (step 05)**
- Promoted `LABEL_HARMONIZATION` to module-level constant
- Extracted 5 helpers: `_embed_query_in_atlas_pca`, `_resolve_target_labels`, `_compose_transport_chain`, `_project_condition_push`, `_project_condition_transport`
- Function went from 327 → ~115 lines (body)
- 8 new tests in `gopro/tests/test_phase4_5.py`

**Track 1C: Importable API surface**
- Created `gopro/__init__.py` with `__getattr__` lazy loading — `import gopro` does NOT import torch/scanpy
- Extracted `run_mapping_pipeline()` from step 02 (returns data without writing files)
- Extracted `run_fidelity_scoring()` from step 03 (returns data without writing files)
- 8 new tests in `gopro/tests/test_api_surface.py`

**Result: 220 gopro tests passing (was 194)**

### 2. Deep Research — Cross-Species & Atlas Landscape ✅

Three parallel research agents searched 60+ sources. Full report at `docs/cross_species_and_atlas_research.md`.

**Key conclusions:**
- **Do NOT use mouse data for GP training** — 2-2.5x temporal scaling (Rayon 2020, Science), FGF8 role diverges between species, human iPSC variation already exceeds cross-species signal, no equivalent mouse morphogen screen exists
- **If mouse data ever appears:** assign fidelity=0.1-0.2 in existing multi-fidelity GP framework
- **Cell type categories at HNOCA level 1/2 ARE conserved** — but proportions and dose-response curves differ
- **10+ high-value human datasets identified** (see Tier 1-3 catalog in the report)

**Tier 1 datasets to download (user hasn't done this yet):**
1. BrainSTEM (Zenodo: 13879662) — 680K fetal brain cells, mirrors pipeline's two-tier scoring
2. Velmeshev 2023 (CellxGene: bacccb91) — 700K cortical cells, prenatal-postnatal
3. BTS Atlas + CellTypist model (Zenodo: 14177002) — pre-trained annotation model
4. Brain Cell Atlas (Nature Med 2024) — 26.3M cells, consensus annotations

Download commands for all tiers are in the conversation but NOT saved to a file. User said "I will do this myself."

### 3. CellWhisperer & Tool Assessment ✅

Research agent assessed 15+ tools. Key findings:

| Tool | Best For | Install | GPU? | Status |
|------|----------|---------|------|--------|
| **scGPT brain checkpoint** | Cell annotation validation | `pip install scgpt` | Optional | **Downloaded to `data/scgpt_brain/`** (205MB model + vocab + args) |
| **GenePT** | Lightweight cell embeddings for vector DB | git clone + Zenodo embeddings | No | Not installed |
| **AnnDictionary** | LLM annotation validation (Claude/GPT) | `pip install anndict` | No | Not installed |
| **scExtract** | Auto-extract cell types from paper PDFs | `pip install -e .` | No | Not installed |
| **CellWhisperer** | NL querying of scRNA-seq results | git clone + pixi | 4GB embed | Not installed |

**Important caveat:** Benchmarks show HVG selection outperforms scGPT/Geneformer in zero-shot settings (Genome Biology 2025). Don't replace working methods without benchmarking.

### 4. Literature Scraping Pipeline (Sub-project 1) — Completed ✅

**Spec:** `docs/specs/2026-03-15-literature-scrapers-design.md`
**Plan:** `docs/plans/2026-03-15-literature-scrapers-plan.md`

Built `literature/` top-level package with:

| Component | File | Tests |
|-----------|------|-------|
| SQLAlchemy models (Paper, Dataset, SearchRun) | `literature/models.py` | 7 |
| DB session factory | `literature/db.py` | — |
| Config loader (YAML + env vars) | `literature/config.py`, `config.yaml` | — |
| Base scraper ABC + keyword detection | `literature/scrapers/base.py` | 4 |
| PubMed scraper (Entrez) | `literature/scrapers/pubmed.py` | 2 |
| bioRxiv scraper (REST API) | `literature/scrapers/biorxiv.py` | 1 |
| GEO + Zenodo + CellxGene scrapers | `literature/scrapers/dataset_sources.py` | 2 |
| Scheduler (orchestrator + Levenshtein dedup) | `literature/scheduler.py` | 7+1 |
| Review CLI (interactive approve/reject/skip) | `literature/review.py` | manual |
| Main CLI (scrape/review/status/export) | `literature/cli.py` | manual |
| Daily cron (launchd plist) | `com.maxxyung.literature-scraper.plist` | — |

**24 literature tests passing. Live PubMed scrape confirmed working.**

**Bug fixed during live testing:** PubMed Entrez `esearch` requires both `mindate` AND `maxdate` when using date filtering. Added `maxdate` param.

**Cron activation (user hasn't done this yet):**
```bash
export NCBI_API_KEY="4cf8aae981eab1a81343206114e6704a3008"  # add to ~/.zshrc
python -m literature scrape  # test first
cp literature/com.maxxyung.literature-scraper.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.maxxyung.literature-scraper.plist
```

## Current Test Counts
- **gopro/**: 220 tests passing
- **literature/**: 24 tests passing
- **Total: 244 tests**

## What's Next — Remaining Tasks

These tasks exist in the task system:

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| 3 | Paper OCR + structured knowledge extraction pipeline | Pending | — |
| 4 | Knowledge graph with vector DB and cross-reference links | Pending | Task 3 |
| 5 | Auto-discovery pipeline for new scRNA-seq datasets | Pending | Tasks 3, 4 |
| 6 | Integrate scGPT brain checkpoint for annotation validation | Pending | — |
| 7 | Integrate AnnDictionary for LLM-based annotation validation | Pending | — |
| 8 | Integrate scExtract for automated paper knowledge extraction | Pending | — |
| 9 | Integrate GenePT + CellWhisperer embeddings for knowledge base vector DB | Pending | — |

**Recommended execution order:**
1. Tasks 6+7 (annotation validation — independent, can run in parallel)
2. Task 3 (OCR/extraction — with scExtract from Task 8)
3. Task 4 (knowledge graph — with GenePT/CellWhisperer from Task 9)
4. Task 5 (auto-discovery — depends on 3+4)

**Each task needs:** brainstorming → spec → plan → implementation (follow the superpowers skill chain)

## API Keys & Credentials
- **NCBI/PubMed:** `4cf8aae981eab1a81343206114e6704a3008` — store in env var `NCBI_API_KEY`
- **OpenAI:** User said "can get one" — needed for GenePT (but pre-computed embeddings on Zenodo don't need it)
- **Anthropic/Claude:** User has access — AnnDictionary supports Claude as LLM backend

## Key User Decisions Made
1. Cross-species: **Focus on human only**, ignore mouse data
2. Literature scope: **Broad** — any brain single-cell/spatial paper, not just organoids
3. Storage: **SQLite** for scraper results
4. Review UX: **CLI tool** (approve/reject/skip)
5. Config: **YAML config file** with env var resolution
6. Scraper architecture: **Merged small scrapers** (GEO+Zenodo+CellxGene in one file)
7. Tool integration: **All in parallel** — spec them all, implement as resources allow
8. Knowledge graph: **Build from scratch** — no existing vector DB
9. Query interfaces needed: **Python API + CLI + web dashboard** (shared team resource)
10. Curation model: **Auto + human review** (flag for approval before indexing)

## Files Changed This Session

### Created
- `gopro/validation.py` — inter-step validators
- `gopro/tests/test_validation.py` — 10 tests
- `gopro/tests/test_api_surface.py` — 8 tests
- `gopro/__init__.py` — lazy loading API surface (was empty)
- `docs/cross_species_and_atlas_research.md` — full research report
- `docs/specs/2026-03-15-literature-scrapers-design.md` — scraper spec
- `docs/plans/2026-03-15-literature-scrapers-plan.md` — scraper implementation plan
- `literature/` — entire package (20 files)
- `data/scgpt_brain/` — downloaded model checkpoint (best_model.pt, vocab.json, args.json)

### Modified
- `gopro/02_map_to_hnoca.py` — extracted `run_mapping_pipeline()`
- `gopro/03_fidelity_scoring.py` — extracted `run_fidelity_scoring()`, added validation call
- `gopro/04_gpbo_loop.py` — added `validate_training_csvs()` call in `build_training_set()`
- `gopro/05_cellrank2_virtual.py` — decomposed `project_query_forward()`, added validation calls
- `gopro/06_cellflow_virtual.py` — added `validate_training_csvs()` call in `run_virtual_screen()`
- `gopro/tests/test_unit.py` — fixed `test_mismatched_indices` fractions
- `gopro/tests/test_phase4_5.py` — added 8 decomposition tests
- `.gitignore` — added literature DB/log exclusions
- Memory files updated in `.claude/projects/`

## Memory Files Updated
- `memory/user_goals.md` — full literature infrastructure requirements, cross-species strategy, research preferences
- `memory/feedback_gpu_detection.md` — always auto-detect CUDA, never hardcode device
- `memory/reference_api_keys.md` — NCBI API key reference
- `memory/MEMORY.md` — updated with Phase 1, cross-species, literature pipeline summaries
