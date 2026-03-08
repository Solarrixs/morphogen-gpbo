# Task Plan — Bugfix Swarm

## Dependency Graph
```
Phase 1: spec-config (FIRST, blocking)
  ├── Phase 2A-1: spec-morphogen-parser ──→ Phase 3: spec-04-gpbo-loop
  ├── Phase 2A-2: spec-03-fidelity-scoring
  ├── Phase 2A-3: spec-02-map-to-hnoca
  ├── Phase 2B-1: spec-00-downloads
  └── Phase 2B-2: spec-01-load-and-convert
                                          ↓
                              Phase 4: spec-tests-coverage (LAST)
                              Phase 5: QA Verification
```

## Tasks

- [ ] Task 1: Phase 1 Foundation — create gopro/config.py, migrate all pipeline files to centralized config, replace print() with logging, replace exit(1) with SystemExit | Acceptance: all existing 124 tests pass, zero hardcoded paths
- [ ] Task 2: Phase 2A-1 Morphogen Parser — temporal encoding fix, true concentrations, _set_morphogen helper | Acceptance: 8 new tests pass, backward compatible 20-column output
- [ ] Task 3: Phase 2A-2 Fidelity Scoring — wire align_composition_to_braun into scoring, decompose main(), logging | Acceptance: 6 new tests pass
- [ ] Task 4: Phase 2A-3 Atlas Mapping — refactor prepare_query_for_scpoli, create utils.py, logging | Acceptance: 19 new tests pass
- [ ] Task 5: Phase 2B-1 Downloads — fix resume logic, sha256, logging | Acceptance: 10 new tests pass
- [ ] Task 6: Phase 2B-2 Load & Convert — type annotations, logging, config imports | Acceptance: 5 new tests pass
- [ ] Task 7: Phase 3 GP-BO Loop — zero-variance filtering, adaptive ILR pseudocount, multi-objective ref_point, refactor recommend_next_experiments | Acceptance: 14 new tests pass
- [ ] Task 8: Phase 4 Test Coverage — add ~28 new tests across unit/integration/property files | Acceptance: coverage >= 85%
- [ ] Task 9: Phase 5 QA Verification — run all checks, confirm zero print/exit/hardcoded paths | Acceptance: all 8 QA checks pass

## Errors
| Task | Attempt | Error | Resolution |
|------|---------|-------|------------|
