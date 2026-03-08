# GP-BO Pipeline Build — Task Plan

## Goal
Build production-quality end-to-end GP-BO pipeline for brain organoid morphogen optimization.

## Phase 1: Environment + Data Foundation [COMPLETE]
- [x] Check installed packages
- [x] Fix scarches import (anndata API change — `read` removed)
- [x] Fix .gz file extension bugs in `gopro/01_load_and_convert_data.py`
- [x] Fix MODEL_DIR paths in steps 01/02
- [x] Inventory data files (shapes, obs columns, var)
- [x] Run `01_load_and_convert_data.py` successfully (53,340 cells × 58,395 genes)
- [x] Parse morphogen concentrations from condition names (morphogen_parser.py)
- [x] Commit: `b0e17ba feat: fix path bugs, rewrite pipeline steps 01-04, add tests`

## Phase 2: Core Pipeline Working (Steps 01-04) [COMPLETE]
- [x] Step 01 end-to-end with real data (53,340 cells primary, 6,183 SAG)
- [x] Step 02 scArches/scPoli mapping — 500 epochs, val_loss 627 → 36,265 cells mapped
- [x] Step 03 fidelity scoring — 46 conditions scored, top: LDN (0.899), IWP2 switch CHIR (0.898)
- [x] Step 04 GP-BO with BoTorch — 24 recommendations generated from real data
- [x] Run step 02 with real data → 14 cell type fractions per condition
- [x] Run step 03 with mapped data → fidelity scores (mean=0.740, range 0.593-0.899)
- [x] Run step 04 with REAL data — plate map saved to gp_recommendations_round1.csv
- [x] Validate end-to-end data flow: step 01 → 02 → 03 → 04
- [x] Write tests (65 tests: 35 unit + 18 integration + 12 property — ALL PASS)
- [x] Commit: see below

## Phase 3: Upgrade GP to BoTorch [COMPLETE]
- [x] Replace sklearn GP with BoTorch SingleTaskGP
- [x] Matérn 5/2 + ARD kernel (BoTorch default)
- [x] Multi-objective acquisition: qLogNoisyExpectedHypervolumeImprovement
- [x] Configurable target cell type (argparse CLI + target_cell_types param)
- [x] Real morphogen bounds per dimension
- [x] ILR transform for compositional Y (Helmert basis)
- [x] Plate map CSV output with well labels
- [x] Multi-fidelity GP support (SingleTaskMultiFidelityGP when fidelity varies)
- [x] Commit: see below

## Phase 4: CellRank 2 (if time) [NOT STARTED]
## Phase 5: CellFlow (if time) [NOT STARTED]

## Blocking Items
- None — Phases 1-3 complete

## Errors Encountered
| Error | Resolution |
|-------|------------|
| scarches 0.6.1 imports `anndata.read` | Upgraded scarches |
| `.gz` extension bugs in step 01 | Removed `.gz` suffixes |
| MODEL_DIR paths wrong | Fixed to use DATA_DIR |
| MPS doesn't support float64 | Use CPU for BoTorch |
| ILR roundtrip bug | Rewrote with Helmert basis |
| Multi-output acquisition error | Added GenericMCObjective scalarization |
| scPoli CUDA→CPU loading | map_location='cpu' |
| Low gene overlap (367/3000) | Mapped Ensembl IDs → gene symbols via gene_name_unique |
