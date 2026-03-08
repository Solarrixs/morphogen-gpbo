# GP-BO Pipeline Build — Task Plan

## Goal
Build production-quality end-to-end GP-BO pipeline for brain organoid morphogen optimization.

## Phase 1: Environment + Data Foundation [COMPLETE]
- [x] Check installed packages
- [x] Fix scarches import (anndata API change — `read` removed)
- [x] Fix .gz file extension bugs in `gopro/01_load_and_convert_data.py` (lines 110-124)
- [x] Fix MODEL_DIR path in `gopro/01_load_and_convert_data.py` (lines 82-84)
- [x] Fix MODEL_DIR path in `gopro/02_map_to_hnoca.py` (line 29)
- [ ] Extract patterning_screen tarball (14GB, running in background)
- [x] Inventory data files (shapes, obs columns, var)
- [x] Run `01_load_and_convert_data.py` successfully (53,340 cells × 58,395 genes)
- [x] Parse morphogen concentrations from condition names (morphogen_parser.py)
- [x] Commit pending

## Phase 2: Core Pipeline Working (Steps 01-04) [IN PROGRESS]
- [x] Step 01 end-to-end with real data (53,340 cells primary, 6,183 SAG)
- [ ] Step 02 scArches/scPoli mapping (rewritten, needs to run with real data)
- [x] Step 03 fidelity scoring (rewritten)
- [x] Step 04 GP-BO with BoTorch (working with synthetic Y, real X)
- [ ] Run step 02 with real data to generate real Y
- [ ] Run step 03 with mapped data
- [ ] Validate end-to-end with real data
- [x] Write tests (45 tests: 20 unit + 16 integration + 9 property — ALL PASS)
- [ ] Commit: "feat: core pipeline steps 01-04 working end-to-end"

## Phase 3: Upgrade GP to BoTorch [NOT STARTED]
- [ ] Replace sklearn GP with BoTorch SingleTaskMultiFidelityGP
- [ ] Matérn 5/2 + ARD kernel with additive structure
- [ ] Multi-objective acquisition: qLogNoisyExpectedHypervolumeImprovement
- [ ] Configurable target cell type
- [ ] Real morphogen bounds
- [ ] ILR transform for compositional Y
- [ ] Plate map CSV output
- [ ] Commit: "feat: upgrade to BoTorch multi-fidelity GP"

## Phase 4: CellRank 2 (if time) [NOT STARTED]
## Phase 5: CellFlow (if time) [NOT STARTED]

## Errors Encountered
| Error | Resolution |
|-------|------------|
| scarches 0.6.1 imports `anndata.read` (removed in anndata 0.12.10) | Need to patch or upgrade scarches |
