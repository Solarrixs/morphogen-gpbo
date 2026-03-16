# Session Handoff — 2026-03-16 (Ralph Loop + Scientific Validation)

## Final Test Counts
- **gopro/**: 495 tests passing
- **literature/**: 53 tests passing
- **Total: 548 tests, 0 failures**

## Git Branch: `ralph/production-readiness-phase2`

```
5f0587f fix: scientific validation P0/P1 — Aitchison distance, moscot tau config
5d894c1 feat: GP warm-start across rounds (495 tests)
e2266a6 feat: Phase A competitive landscape — cross-fidelity validation, cost ratios, replicates
a782d11 refactor: simplify review — fix eager imports, memory, N+1 queries, dead imports
d91f048 chore: code quality polish — deduplicate md5_file, fix docstring, update progress
f7550f9 feat: test coverage push + knowledge graph (513 total tests)
7a94e59 feat: production readiness phases 1-4 + scGPT integration (403 tests)
```

## What Was Accomplished

### Production Readiness Plan — ALL 12 PHASES COMPLETE
| Phase | Description | Key Files |
|-------|-------------|-----------|
| 1A | Inter-step validation | `gopro/validation.py` |
| 1B | Decompose step 05 | `gopro/05_cellrank2_virtual.py` |
| 1C | Importable API | `gopro/__init__.py` |
| 2A | Region targeting | `gopro/region_targets.py` |
| 2B | Dynamic label maps | In `region_targets.py` |
| 3C | CellFlow heuristic | `gopro/06_cellflow_virtual.py` |
| 4A | Config-driven datasets | `gopro/datasets.py`, `gopro/datasets.yaml` |
| 4B | Pipeline orchestrator | `gopro/orchestrator.py` |
| 5A | Test coverage push | `gopro/tests/test_coverage_push.py` |
| 5B | Code quality polish | Config, download scripts |

### Competitive Landscape Phase A — ALL 4 IDEAS COMPLETE
| Idea | Source Paper | Key Change |
|------|-------------|------------|
| #1 Cross-fidelity validation gate | McDonald 2025 | `validate_fidelity_correlation()` — auto-fallback if correlation < 0.3 |
| #3 Fidelity cost ratios | McDonald 2025 | `FIDELITY_COSTS` in config.py |
| #5 GP warm-start | Narayanan 2025 | `save_gp_state()` / `load_gp_state()` |
| #7 Replicate wells | BATCHIE 2025 | `_select_replicate_conditions()`, `--n-replicates` CLI |

### Scientific Validation Sweep — 8 ISSUES FOUND, P0/P1 FIXED
| Issue | Fix |
|-------|-----|
| P0-1: Missing lengthscale prior | `_set_dim_scaled_lengthscale_prior()` — LogNormal(log(sqrt(d))+0.5, 1.0) |
| P0-2: Bad pseudo-count (1e-10) | `_multiplicative_replacement()` — CoDA-correct delta |
| P0-3: Cosine sim on compositions | `aitchison_distance()` / `aitchison_similarity()` |
| P1-1: Low acqf restarts | num_restarts=10, raw_samples=1024 |
| P1-4: Hardcoded moscot tau | Env-var configurable (GPBO_MOSCOT_TAU_A etc.) |

**Validated as correct:** Matern 5/2 + ARD, ILR transform, Helmert basis, qLogEI/qLogNEHVI, multi-fidelity GP setup, moscot for OT, KNN k=10

### Literature Intelligence
| Component | Tests |
|-----------|-------|
| Scrapers (PubMed, bioRxiv, GEO, Zenodo, CellxGene) | 24 |
| Knowledge graph + vector DB (TF-IDF search) | 29 |
| scGPT annotation validation | ~10 |
| Literature report (22 papers, 6 topics) | — |

### /simplify Fixes Applied
- Lazy loading for region_targets (prevents anndata import at `import gopro`)
- scGPT: keep sparse matrix sparse in embed_cells()
- scGPT: minimal AnnData for validation clustering
- Knowledge graph: N+1 → 2-query batch fetch
- Knowledge graph: COUNT queries instead of full table loads
- Removed 6 unused imports

## What's Left

### Competitive Landscape Phase B (Round 2 Prep)
- Idea #2: Cost-aware acquisition (TVR model)
- Idea #4: Target profile refinement from Round 1 data
- Idea #6: Train CellFlow on own data (needs GPU)
- Idea #3 (Sanchis-Calleja): Ingest 97 patterning screen conditions

### Competitive Landscape Phase C (Modeling)
- Idea #8: Additive + interaction kernel decomposition (NAIAD 2025)
- Idea #9: Adaptive model complexity schedule
- Idea #10: Morphogen timing window encoding
- Idea #11: Per-cell-type GP models (GPerturb 2025)

### Competitive Landscape Phase D (Diagnostics)
- Ideas #13-17: Monitoring, saturation detection, convergence, ensemble stability

### Deferred Scientific Fixes
- P1-2: LassoBO as SAASBO alternative (larger effort)
- P1-3: Update EC50 values from Sanchis-Calleja dose-response
- P2-1: Bootstrap uncertainty → heteroscedastic GP noise
- P2-2: Data-driven entropy center

### Infrastructure
- Phase 3A/3B: Build temporal atlas + run CellRank2 (needs 22GB data, compute time)
- Literature sub-projects 3-5: Paper OCR, auto-discovery pipeline
- Run step 02 for SAG screen (30-60 min CPU)

## Key References
- `docs/plans/competitive_landscape_ideas_index.md` — 52 ideas from 9 papers
- `docs/plans/ideas_from_*.md` — Per-paper idea files with DOIs and code repos
- `docs/plans/findings.md` — Scientific validation findings
- `docs/literature_intelligence_2026-03-15.md` — 22-paper literature report
