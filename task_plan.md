# Task Plan
> Goal: Pipeline Fixes & GP-BO Modeling Improvements
> Created: 2026-03-18T04:28:29Z
> Last updated: 2026-03-19

## Tasks

### Phase A: Critical Bug Fixes — COMPLETE
- [x] TODO-24: Remap fidelity encodings for MF-GP kernel
- [x] TODO-25: R²-based 3-zone fidelity routing
- [x] TODO-26: Fix CellFlow dose encoding to use log1p

### Phase B: CellFlow Integration Fixes — COMPLETE
- [x] TODO-1: Fix CellFlow JAX vs PyTorch API mismatch
- [x] TODO-3: Add Day 72 out-of-distribution warning — `_warn_ood_harvest_days()` implemented
- [x] TODO-4: Handle CellFlow conservative prediction bias — `confidence_to_noise_variance()` + `allow_fallback` gate

### Phase C–H: All pipeline TODOs complete. See docs/task_plan.md for full details (§1.3–§1.9 all marked COMPLETE).

### Paper work: See docs/task_plan.md §4 for detailed paper TODOs (fact-checks, figures, methods expansion).

### Code Review Fixes (ralph-task.md)
- [x] I-3: Add TVR gradient warning docstring + acquisition optimization test | Acceptance: docstring updated, 2 tests added, all pass
- [x] I-4: Deduplicate antagonist pairs — YAML single source of truth | Acceptance: ANTAGONIST_PAIRS removed from 04, both consumers load YAML
- [x] I-1: Extract _inflate_cellflow_variance — remove importlib usage | Acceptance: no importlib.util in 04_gpbo_loop.py
- [x] S-1: Replace np.random.RandomState with default_rng in benchmarks | Acceptance: no RandomState in benchmarks/
- [x] S-2: Optimize score_gene_signatures memory — copy obs not full AnnData | Acceptance: reduced memory
- [x] S-3: Rename run_noise_sweep → run_random_baseline_noise_sweep | Acceptance: name clarified
- [x] S-4: Fix _fidelity_to_task_idx dtype to torch.long | Acceptance: dtype is torch.long, tests pass
- [x] S-5: Add 17 unit tests for new features | Acceptance: 17 new tests added, all pass

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
