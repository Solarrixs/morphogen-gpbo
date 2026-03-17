# Findings & Discoveries
> Survives context resets. Updated by every phase.

## Codebase Patterns
- KernelSpec namedtuple at module level controls kernel selection — avoids twin-variable bugs
- `_helmert_basis` is cached (computed once) — torch ILR delegates to numpy for correctness
- Sobol seed=42 used for reproducible convergence diagnostics

## Gotchas
- fidelity=1.0 collapses MF-GP inter-fidelity kernel (TODO-24, unfixed)
- CellFlow dose encoding uses raw dose×onehot instead of log1p (TODO-26, unfixed)
- Fidelity correlation threshold at Spearman 0.3 is too permissive (TODO-25, unfixed)

## Quality Issues Found
- Bug-hunter iteration 4: 76 findings (6 critical, 38 warning, 32 info). See `.bug-hunter/SUMMARY.md`
- 5 remaining criticals are test coverage gaps in `02_map_to_hnoca.py` and `05_cellrank2_virtual.py`, not code bugs
