# Handoff to Iteration 26

## Last Completed: §1.9 — Ingest 98 Sanchis-Calleja conditions
Added `SanchisCallejaParser` class to `morphogen_parser.py` with:
- 98 conditions parsed via regex tokenizer (`_SC_TOKEN_RE`)
- Concentration levels A-E from Supplementary Figure 1 (CHIR, RA, SHH+PM, FGF8 early/late, BMP4, BMP7, XAV939, Cyclopamine)
- Fixed timing doses for tA-tE experiments
- SHH always paired with purmorphamine at proportional doses
- Gradient conditions (Grad_<A, Grad_C-E) approximated as scalar
- No Amin/Kelley base media (different protocol, Day 21 harvest)
- `SANCHIS_CALLEJA_CONDITIONS` exported (98 names, MOESM5 canonical)
- 10 new tests, 689 total passing

## Next Up: Wire Sanchis-Calleja into multi-fidelity merge (§1.9 follow-up)
- Assign fidelity 0.85 to Sanchis-Calleja data
- Add to `merge_multi_fidelity_data()` auto-discovery in `04_gpbo_loop.py`
- Add `--sanchis-fractions`/`--sanchis-morphogens` CLI flags
- Acceptance: 3× training data when Sanchis-Calleja CSVs present; test verifies merge; 3+ new tests

Alternative next tasks:
- TODO-12: Contextual parameter support (§1.5)
- TODO-36: Carry-forward top-K controls (§1.6)
- TODO-49: Exact v-score formula (§1.7)
- TODO-53: Domain-informed toy morphogen function (§1.8)

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- MOESM5 has 98 conditions (not 97 as originally estimated in task plan)
- `ZeroPassingKernel` created via lazy factory `_get_zero_passing_kernel_class()` — never reference global directly
- `use_alr` and `use_ilr` are mutually exclusive — ALR takes precedence when both set
- CellFlow uses JAX (`jax.random`), NOT torch
- Sanchis-Calleja parser does NOT set base media (BDNF, NT3, cAMP, AscorbicAcid) — different protocol from Amin/Kelley

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (689 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 nearly complete (11/13, remaining 2 deferred)
- §1.9 parser COMPLETE, merge follow-up pending
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~78 tasks todo, 0 blocked, ~52 complete
