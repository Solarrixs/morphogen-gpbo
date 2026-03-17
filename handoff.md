# Handoff to Iteration 11

## Last Completed: Phase D Idea #17 — Ensemble disagreement diagnostic
- `compute_ensemble_disagreement()` in `04_gpbo_loop.py` (~124 lines net)
- `--ensemble-restarts N` CLI flag wired into `run_gpbo_loop()`
- 3 new tests, simplify pass cleaned 4 issues
- All 564 tests passing, 0 failures
- **All Phase D diagnostics now complete** (#13, #16, #17)

## Next Up: Remaining competitive landscape ideas

### Immediate candidates (no external blockers):
1. **Idea #14: CellFlow saturation detection** (Cosenza 2022, MEDIUM)
   - Detect if CellFlow heuristic predictions plateau across dose ranges
   - Acceptance: function returns saturation flag per morphogen dimension; logged in screening report

2. **Idea #15: Decoy robustness test** (McDonald 2025, MEDIUM)
   - Inject corrupted CellFlow predictions to measure GP resilience
   - Acceptance: test function quantifies recommendation shift under perturbation

### Blocked / deferred:
3. **Idea #3: Ingest Sanchis-Calleja 97 conditions** — needs temporal atlas (Phase 3A) + mapping run
4. **Idea #6: Train CellFlow** — requires GPU
5. **P1-2: LassoBO** — lower priority alternative to SAASBO
6. **P2-1: Bootstrap uncertainty** — lower priority
7. **P2-2: Data-driven entropy center** — small fix

### Original plan phases still open:
- Phase 3A: Build temporal atlas (data on disk, needs run)
- Phase 3B: CellRank2 virtual data generation (depends on 3A)

## Warnings
- `04_gpbo_loop.py` is ~2930 lines — modular but large; avoid adding more unless essential
- Ensemble disagreement uses `np.clip` on cosine similarity to avoid float >1.0
- Phase C/D ideas are all modeling features behind CLI flags — they compose but haven't been tested in combination (e.g., `--per-type-gp` + `--timing-windows` + `--adaptive-complexity`)
- Iterations 8-10 progress entries were logged out-of-order (8→9→10 as 10→9→8 in file); cosmetic only

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Venv: `source .venv/bin/activate`
- Test: `python -m pytest gopro/tests/ -v`
- 13/17 competitive landscape ideas implemented (Phase A: #1,5,7; Phase B: #2,4,12; Phase C: #8,9,10,11; Phase D: #13,16,17)
- All P0 scientific findings fixed; P1-1, P1-4 fixed; P1-2, P1-3 deferred

## Remaining: 6 tasks todo, 2 blocked (GPU + temporal atlas), 13 complete
