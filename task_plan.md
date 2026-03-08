# Task Plan — Bugfix Swarm
> Goal: Implement 38 bug fixes across 8 spec areas using agent swarm pattern
> Created: 2026-03-08
> Branch: ralph/bugfix-swarm

## Tasks

- [~] Task 1: Phase 1 Foundation — create gopro/config.py, migrate all pipeline files to centralized config, replace print() with logging, replace exit(1) with SystemExit | Acceptance: all existing 124 tests pass, zero hardcoded paths in gopro/*.py | Attempts: 1
- [ ] Task 2: Phase 2A-1 Morphogen Parser — temporal encoding fix, true concentrations, _set_morphogen helper | Acceptance: 8 new tests pass, backward compatible 20-column output
- [ ] Task 3: Phase 2A-2 Fidelity Scoring — wire align_composition_to_braun into scoring, decompose main(), logging | Acceptance: 6 new tests pass
- [ ] Task 4: Phase 2A-3 Atlas Mapping — refactor prepare_query_for_scpoli, create utils.py, logging | Acceptance: 19 new tests pass
- [ ] Task 5: Phase 2B-1 Downloads — fix resume logic, sha256, logging | Acceptance: 10 new tests pass
- [ ] Task 6: Phase 2B-2 Load & Convert — type annotations, logging, config imports | Acceptance: 5 new tests pass
- [ ] Task 7: Phase 3 GP-BO Loop — zero-variance filtering, adaptive ILR pseudocount, multi-objective ref_point, refactor | Acceptance: 14 new tests pass
- [ ] Task 8: Phase 4 Test Coverage — add ~28 new tests across unit/integration/property files | Acceptance: coverage >= 85%
- [ ] Task 9: Phase 5 QA Verification — run all checks, confirm zero print/exit/hardcoded paths | Acceptance: all 8 QA checks pass

## Dependency Graph
```
Task 1 (config) → BLOCKS all others
  ├── Task 2 (morphogen-parser) → Task 7 (gpbo-loop)
  ├── Task 3 (fidelity-scoring)
  ├── Task 4 (atlas-mapping)
  ├── Task 5 (downloads)
  └── Task 6 (load-convert)
                                → Task 8 (test-coverage) → Task 9 (QA)
```

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|
| Centralized config.py | Single source of truth for paths, constants, logging |
| Worktree isolation per agent | Prevents merge conflicts during parallel work |

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
