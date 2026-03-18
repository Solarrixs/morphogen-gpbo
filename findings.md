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

## Bug Hunter Fixes (Round 7 — 2026-03-17)

### Critical Fixes
1. **[A-C-NEW-1] PCA index corruption in `_embed_query_in_atlas_pca`** (`05_cellrank2_virtual.py:303`)
   - `gene_idx` was computed relative to `hvg_genes` but used to index `pca_loadings` (indexed by full `var_names`). All CellRank2 virtual projections were silently corrupted.
   - Fix: `list(hvg_genes).index(g)` → `list(atlas_adata.var_names).index(g)`

2. **[A-C-001] Ensemble restarts overwrite warm-start checkpoint** (`04_gpbo_loop.py:1418`)
   - `compute_ensemble_disagreement` called `fit_gp_botorch(round_num=1)` which unconditionally saved state, overwriting the main model's checkpoint.
   - Fix: Added `save_state: bool = True` parameter; ensemble calls pass `save_state=False`.

3. **[A-C-004] `generate_report` hardcodes `amin_kelley` prefix** (`visualize_report.py:977-986`)
   - Three file paths hardcoded to `amin_kelley`. Any non-default `--output-prefix` caused `FileNotFoundError`.
   - Fix: Added `output_prefix` parameter, threaded through CLI.

4. **[A-C-003] Confusing error for small plate sizes** (`04_gpbo_loop.py:2840`)
   - `n_duplicates >= n_novel` produced error referencing internal variable names.
   - Fix: Early validation with user-facing message referencing original CLI arguments.

### Warning Fix
5. **[A-W-NEW-2] Zero-division guard in `compute_soft_cell_type_fractions`** (`02_map_to_hnoca.py:436`)
   - `row_sums` could be zero for edge-case conditions. NaN would propagate to GP training.
   - Fix: `row_sums.replace(0, 1)` with warning log.
