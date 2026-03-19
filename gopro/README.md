# Morphogen GP-BO Pipeline

Gaussian Process Bayesian Optimization for brain organoid morphogen protocol optimization. Uses active learning to find optimal morphogen combinations that drive brain organoids toward cell type compositions matching the human fetal brain.

## Setup

```bash
cd ..  # from gopro/ to project root
python -m venv .venv && source .venv/bin/activate
pip install -r gopro/requirements.txt
```

## Pipeline Steps

Run sequentially:

```bash
# 1. Download reference data
python 00_zenodo_download.py              # HNOCA + Braun fetal brain (14 GB)
python 00a_download_geo.py                # GEO GSE233574 raw data (MD5 verified)
python 00b_download_patterning_screen.py  # Sanchis-Calleja patterning screen (22 GB)
python 00c_build_temporal_atlas.py        # Build temporal atlas for CellRank 2

# 2. Convert and parse
python 01_load_and_convert_data.py        # GEO MTX → AnnData h5ad
python morphogen_parser.py                # Condition names → 24D µM concentration vectors

# 3. Map to reference atlas
python 02_map_to_hnoca.py                 # Primary screen (default)
python 02_map_to_hnoca.py --input data/amin_kelley_sag_screen.h5ad --output-prefix sag_screen

# 4. Score fidelity and optimize
python 03_fidelity_scoring.py             # Two-tier fidelity scoring + NEST-Score maturity
python 04_gpbo_loop.py                    # Fit GP, recommend next 24 experiments

# 5. Virtual data augmentation (optional)
python 05_cellrank2_virtual.py            # CellRank 2 temporal projection (fidelity=0.5)
python 06_cellflow_virtual.py             # CellFlow virtual screening (fidelity=0.0, gated)

# 6. Visualization
python 05_visualize.py                    # Interactive HTML report
```

See `docs/specs/pipeline-execution-spec.md` for detailed execution plan with dependencies.

## Key Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized paths, constants, 25 morphogen column names, unit conversions, cost dict |
| `morphogen_parser.py` | Parse 48+ conditions (46 Amin/Kelley + 2 SAG + 98 Sanchis-Calleja) → 24D µM vectors |
| `signature_utils.py` | NEST-Score transcriptomic fidelity, gene signature scoring, signature refinement |
| `agents/scorer.py` | Recommendation scoring: plausibility, novelty, feasibility, predicted fidelity |
| `agents/pathway_rules.yaml` | Morphogen antagonist pair rules (BMP, WNT, SHH pathways) |
| `benchmarks/toy_morphogen_function.py` | 24D Hill response synthetic test function (simplex output) |
| `benchmarks/noise_robustness.py` | Noise × batch size sweep for robustness characterization |
| `validation.py` | Inter-step data validation (mapped h5ad, CSVs, fidelity report) |
| `datasets.py` | YAML-backed dataset registry with fidelity metadata |
| `region_targets.py` | Brain region profiles, A-P axis targeting, custom target support |
| `orchestrator.py` | Run any subset of pipeline steps programmatically |

## Step 04 Key Flags

```bash
python 04_gpbo_loop.py \
  --n-recommendations 24 \
  --round 1 \
  --n-controls 2 \              # Carry-forward top-K controls (Kanda eLife 2022)
  --contextual-cols log_harvest_day \  # Fix harvest day within plates
  --cost-weight 0.1 \           # Cost-aware desirability gate
  --input-warp \                # Kumaraswamy CDF input warping
  --explicit-priors \           # Hvarfner 2024 dimensionality-scaled priors
  --mc-samples 1024 \           # Sobol QMC samples for acquisition
  --sanchis-fractions data/gp_training_labels_sanchis_calleja.csv \
  --sanchis-morphogens data/morphogen_matrix_sanchis_calleja.csv \
  --validation \                # Generate 0.3x/1x/3x dose-response plate
  --confirmation                # Generate confirmation plate with replicates
```

## Tests

```bash
python -m pytest tests/ -v  # ~860 tests
```

## Data

All data lives in `../data/` (gitignored). See `../data/README.md` for download instructions.

## Configuration

Environment variables:
- `GPBO_PROJECT_DIR` — Project root (auto-detected)
- `GPBO_DATA_DIR` — Data directory (default: `../data/`)
- `GPBO_MODEL_DIR` — scPoli model directory
- `GPBO_LOG_LEVEL` — Logging level (default: INFO)
