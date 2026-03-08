# Progress Log

## Session 1 — 2026-03-08

### Phase 1: Environment + Data Foundation
- [05:02] Started iteration 1. Read all pipeline scripts, checked packages.
- [05:02] Discovered scarches 0.6.1 broken with anndata 0.12.10 (imports removed `anndata.read`)
- [05:02] Confirmed all data files present. Identified .gz extension bug.
- [05:02] Listed 46 unique conditions from primary screen metadata.
- [05:02] Created planning files.
- [05:03] Fixed .gz extension bugs in step 01, MODEL_DIR paths in steps 01/02
- [05:03] Ran step 01 successfully: 53,340 cells × 58,395 genes (primary), 6,183 cells (SAG)
- [05:05] scarches fixed (background agent upgraded it)
- [05:06] Read HNOCA reference notebooks (Fig4_Amin_mapping/) — understood scPoli workflow
- [05:06] Inspected HNOCA reference: 1.77M cells, annot_level_1/2/3, X_scpoli latent
- [05:07] Rewrote step 02: scPoli mapping following reference impl exactly
- [05:08] Built morphogen_parser.py: parses all 46 conditions → concentration vectors
- [05:09] Rewrote step 04: BoTorch GP with ILR transform, multi-objective acquisition
- [05:09] Fixed ILR roundtrip bug (Helmert basis matrix approach)
- [05:10] Rewrote step 03: fidelity scoring against Braun fetal brain
- [05:11] Wrote 45 tests (20 unit + 16 integration + 9 property) — ALL PASS
- [05:12] Committed Phase 1 + partial Phase 2 work
- [05:12] BoTorch GP-BO tested end-to-end with real X, synthetic Y — works!
- [05:14] Fixed scPoli loading: map_location='cpu', column name mapping
- [05:15] Running step 02 with real data (1.77M ref + 36K query cells)

## Session 2 — 2026-03-08 (continued)

### Phase 2: Core Pipeline (continued)
- Step 02 scPoli training restarted (previous session timed out at ~40%)
- Training loss converging: 763→600, stable at ~605 by epoch 50/500
- Added 17 unit tests for step 03 fidelity scoring (ALL PASS)
- Added 2 integration tests for step 03 (ALL PASS)
- Total: 62 tests passing (35 unit + 18 integration + 9 property)
- Updated requirements.txt: uncommented botorch/gpytorch/torch, added pytest/hypothesis
- Added argparse CLI to step 04 for configurable target cell types
- Step 02 training running in background (~10% complete)

### Key Decisions
- Use CPU for BoTorch (MPS doesn't support float64)
- Map annot_level_* → snapseed_pca_rss_level_* for scPoli compatibility
- 2,431/3,000 genes shared between query and reference (569 zero-filled)
- KNN label transfer with sklearn (not cuml, since no CUDA)

## Session 3 — 2026-03-08 (continued)

### Phase 2 + 3 Completion
- Step 02 training completed: 500 epochs, val_loss converged to 627.31
- Step 02 outputs saved: amin_kelley_mapped.h5ad (36,265 cells × 3,000 genes, 14 cell types)
- Step 03 ran successfully with real mapped data:
  - 46 conditions scored against Braun fetal brain (16 regions × 12 cell classes)
  - Top conditions: LDN (0.899), IWP2 switch CHIR (0.898), IWP2 (0.895)
  - Bottom: CHIR3-SAG1000 (0.593), BMP4 CHIR (0.600), FGF2-20 (0.609)
  - Mean composite fidelity: 0.740
- Step 04 ran with real data: 46 conditions × 21 features → 24 recommendations
- All 65 tests pass (35 unit + 18 integration + 12 property)
- Phases 2 and 3 COMPLETE — all pipeline steps run end-to-end with real data
