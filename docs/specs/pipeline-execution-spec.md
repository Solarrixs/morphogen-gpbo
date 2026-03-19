# Pipeline Execution Spec: GP-BO Real Data Jobs

This spec describes the CPU/GPU jobs required to run the full GP-BO pipeline on real data. Each job is defined with its command, prerequisites, expected outputs, estimated runtime, and validation steps. A future Claude instance should execute these jobs in dependency order.

## Dependency Graph

```
Job 1 (SAG screen mapping)  ─────────────────────────────────────────────┐
                                                                          │
Job 2 (Download patterning screen)                                        │
  └─► Job 3 (Build temporal atlas)                                        │
        ├─► Job 4 (Map Sanchis-Calleja to HNOCA)                         │
        └─► Job 5 (CellRank2 virtual data)                               │
                                                                          │
Job 6 (Train CellFlow model) ← requires Job 2 outputs                    │
                                                                          │
Job 7 (Full GP-BO Round 1) ← requires Jobs 1, 4, 5 (Job 6 optional) ────┘
```

Parallelizable:
- Jobs 1 and 2 can run in parallel (no shared dependencies).
- Jobs 4 and 5 can run in parallel once Job 3 completes.
- Job 6 is independent but requires Job 2 outputs and a GPU.

---

## Environment Setup

Before running any job, ensure the virtual environment is active and the working directory is correct:

```bash
cd /path/to/morphogen-gpbo
source .venv/bin/activate
cd gopro
```

All commands below assume `cwd = gopro/`. Estimated times assume a modern multi-core CPU (Apple M-series or equivalent). GPU jobs are explicitly marked.

---

## Job 1: Map SAG Screen to HNOCA

**Purpose:** Map the SAG secondary screen (4 conditions, 2 unique new: SAG_50nM, SAG_2uM) onto the HNOCA reference atlas via scArches/scPoli, producing cell type fraction labels for the GP.

### Prerequisites

| File | Source |
|------|--------|
| `data/amin_kelley_sag_screen.h5ad` | Step 01 (`01_load_and_convert_data.py`) |
| `data/hnoca_minimal_for_mapping.h5ad` | Step 00 (`00_zenodo_download.py`) |
| `data/neural_organoid_atlas/supplemental_files/scpoli_model_params/` | Cloned theislab HNOCA repo |

Verify prerequisites exist before starting:

```bash
python -c "
from pathlib import Path
d = Path('../data')
for f in [
    'amin_kelley_sag_screen.h5ad',
    'hnoca_minimal_for_mapping.h5ad',
    'neural_organoid_atlas/supplemental_files/scpoli_model_params/model_params.pt',
]:
    p = d / f
    print(f'  {\"OK\" if p.exists() else \"MISSING\"}: {p}')
    assert p.exists(), f'Missing prerequisite: {p}'
print('All Job 1 prerequisites met.')
"
```

### Command

```bash
python 02_map_to_hnoca.py \
  --input data/amin_kelley_sag_screen.h5ad \
  --output-prefix sag_screen
```

The script will:
1. Load the SAG screen h5ad
2. Filter cells (`ClusterLabel != 'filtered'`, handled by `filter_quality_cells()`)
3. Prepare query for scPoli (gene alignment, HVG subset)
4. Run scArches architecture surgery + fine-tuning
5. Transfer labels via KNN from the HNOCA reference
6. Compute cell type fractions per condition

### Expected Outputs

| File | Description |
|------|-------------|
| `data/sag_screen_mapped.h5ad` | Mapped AnnData with transferred labels |
| `data/gp_training_labels_sag_screen.csv` | Cell type fractions per condition (index = condition names) |
| `data/gp_training_regions_sag_screen.csv` | Brain region fractions per condition |

### Estimated Time

30-60 minutes on CPU. scPoli fine-tuning is the bottleneck. BoTorch/GP fitting is CPU-only (MPS does not support float64).

### Validation

```bash
python -c "
import pandas as pd
df = pd.read_csv('../data/gp_training_labels_sag_screen.csv', index_col=0)
print(f'Shape: {df.shape}')
print(f'Conditions: {list(df.index)}')
# Expect >= 2 conditions (SAG_50nM, SAG_2uM); SAG_250nM/SAG_1uM are excluded as near-duplicates
assert df.shape[0] >= 2, f'Expected >= 2 conditions, got {df.shape[0]}'
# Fractions should sum to ~1.0 per condition
sums = df.sum(axis=1)
assert (sums - 1.0).abs().max() < 0.05, f'Fractions do not sum to 1: {sums.values}'
print('Job 1 validation PASSED.')
"
```

Also verify the mapped h5ad:

```bash
python -c "
import anndata
a = anndata.read_h5ad('../data/sag_screen_mapped.h5ad', backed='r')
print(f'Cells: {a.n_obs:,}')
print(f'Genes: {a.n_vars:,}')
print(f'Conditions: {a.obs[\"condition\"].nunique()}')
print(f'Has scPoli labels: {\"predicted_annot_level_2\" in a.obs.columns}')
"
```

---

## Job 2: Download and Convert Patterning Screen

**Purpose:** Download the Sanchis-Calleja/Azbukina patterning screen dataset (22.8 GB) from Zenodo and convert the RDS files to h5ad format.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Disk space | ~25 GB free |
| R >= 4.2 | With Seurat and SeuratDisk packages installed |
| Network | Stable connection for 22.8 GB Zenodo download |

Verify R is available:

```bash
R --version | head -1
Rscript -e "library(Seurat); library(SeuratDisk); cat('R packages OK\n')"
```

If R packages are missing, `convert_rds_to_h5ad.py` will attempt auto-install, but manual installation is more reliable:

```r
install.packages("Seurat")
remotes::install_github("mojaveazure/seurat-disk")
```

### Commands

**Step 2a: Download from Zenodo**

```bash
python 00b_download_patterning_screen.py
```

This downloads to `data/patterning_screen/` and extracts `OSMGT_processed_files.tar.gz`. Uses `aria2c` for parallel download if available; falls back to Python streaming download.

To list files first without downloading:

```bash
python 00b_download_patterning_screen.py --list
```

**Step 2b: Convert RDS to h5ad (if needed)**

The build_temporal_atlas step (Job 3) expects `exp1_processed_8.h5ad.gz`. If the download produced `.rds.gz` files instead:

```bash
python convert_rds_to_h5ad.py data/patterning_screen/exp1_processed_8.rds.gz
```

Check metadata first with:

```bash
python convert_rds_to_h5ad.py data/patterning_screen/exp1_processed_8.rds.gz --check-only
```

### Expected Outputs

| File | Description |
|------|-------------|
| `data/patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz` | Processed patterning screen data |
| `data/patterning_screen/` (various) | Additional experiment files |

### Estimated Time

- Download: 1-3 hours depending on bandwidth (22.8 GB)
- RDS conversion: 30-60 minutes per file
- Total: 2-4 hours

### Validation

```bash
python -c "
from pathlib import Path
p = Path('../data/patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz')
assert p.exists(), f'Missing: {p}'
print(f'File size: {p.stat().st_size / 1e9:.1f} GB')
print('Job 2 validation PASSED.')
"
```

---

## Job 3: Build Temporal Atlas

**Purpose:** Convert the patterning screen h5ad into the temporal atlas format required by CellRank 2 (step 05). Adds a standardized `day` column and harmonizes cell type labels.

### Prerequisites

| File | Source |
|------|--------|
| `data/patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz` | Job 2 |

### Command

```bash
python 00c_build_temporal_atlas.py
```

To inspect the source file metadata first (does not build atlas):

```bash
python 00c_build_temporal_atlas.py --inspect-only
```

If the time column or cell type column has a non-default name:

```bash
python 00c_build_temporal_atlas.py --time-col stage --label-col cell_type
```

The script auto-detects candidate column names for timepoint (`day`, `timepoint`, `time`, `age`, `stage`) and cell type labels (`predicted_annot_level_2`, `annot_level_2`, `cell_type`).

### Expected Outputs

| File | Description |
|------|-------------|
| `data/azbukina_temporal_atlas.h5ad` | Temporal atlas with `day` column in `.obs` |

### Estimated Time

10-30 minutes (mostly I/O for decompressing and reading the large h5ad).

### Validation

```bash
python -c "
import anndata
a = anndata.read_h5ad('../data/azbukina_temporal_atlas.h5ad', backed='r')
print(f'Cells: {a.n_obs:,}')
print(f'Genes: {a.n_vars:,}')
print(f'obs columns: {list(a.obs.columns[:10])}...')
assert 'day' in a.obs.columns, 'Missing day column'
day_counts = a.obs['day'].value_counts().sort_index()
print(f'Timepoints:\\n{day_counts}')
expected = {7, 15, 30, 60, 90, 120}
actual = set(day_counts.index.astype(int))
missing = expected - actual
assert not missing, f'Missing expected timepoints: {missing}'
print('Job 3 validation PASSED.')
"
```

---

## Job 4: Map Sanchis-Calleja to HNOCA

**Purpose:** Map the Sanchis-Calleja/Azbukina temporal atlas onto HNOCA via scArches/scPoli, producing cell type fractions. This enables the `--sanchis-fractions`/`--sanchis-morphogens` flags in step 04 for multi-fidelity GP integration.

### Prerequisites

| File | Source |
|------|--------|
| `data/azbukina_temporal_atlas.h5ad` | Job 3 |
| `data/hnoca_minimal_for_mapping.h5ad` | Step 00 |
| `data/neural_organoid_atlas/supplemental_files/scpoli_model_params/` | Cloned HNOCA repo |

### Command

```bash
python 02_map_to_hnoca.py \
  --input data/azbukina_temporal_atlas.h5ad \
  --output-prefix sanchis_calleja \
  --condition-key condition \
  --batch-key sample
```

**Important:** The `--condition-key` and `--batch-key` values depend on the column names in the temporal atlas `.obs`. Run `--inspect-only` in Job 3 first to confirm the correct column names. Common alternatives: `--condition-key treatment`, `--batch-key batch`.

### Expected Outputs

| File | Description |
|------|-------------|
| `data/sanchis_calleja_mapped.h5ad` | Mapped AnnData with transferred labels |
| `data/gp_training_labels_sanchis_calleja.csv` | Cell type fractions per condition |
| `data/gp_training_regions_sanchis_calleja.csv` | Brain region fractions per condition |

### Estimated Time

1-3 hours on CPU. The Sanchis-Calleja dataset is significantly larger than the Amin/Kelley primary screen, so scPoli fine-tuning takes longer.

### Validation

```bash
python -c "
import pandas as pd
df = pd.read_csv('../data/gp_training_labels_sanchis_calleja.csv', index_col=0)
print(f'Shape: {df.shape}')
print(f'Conditions (first 10): {list(df.index[:10])}')
sums = df.sum(axis=1)
assert (sums - 1.0).abs().max() < 0.05, f'Fractions do not sum to 1'
print(f'Total conditions: {df.shape[0]}')
print('Job 4 validation PASSED.')
"
```

### Note on Morphogen Matrix

The Sanchis-Calleja morphogen matrix is needed for GP integration. This may require extending `morphogen_parser.py` with a `SanchisCallejaParser` class if one does not already exist. Check:

```bash
python -c "
from gopro.morphogen_parser import CombinedParser
# CombinedParser currently merges AminKelleyParser + SAGSecondaryParser.
# A SanchisCallejaParser may need to be written if the Sanchis-Calleja
# conditions are not yet parseable.
print('CombinedParser available')
"
```

If the morphogen matrix for Sanchis-Calleja conditions does not exist, the condition names from the patterning screen will need to be parsed into 24D morphogen vectors. This is a code-writing task, not a pipeline execution task.

---

## Job 5: Generate CellRank2 Virtual Data

**Purpose:** Use moscot optimal transport + CellRank 2 RealTimeKernel to forward-project Amin/Kelley query cells (Day 72) to Days 90 and 120, generating medium-fidelity (0.5) virtual training points for the multi-fidelity GP.

### Prerequisites

| File | Source |
|------|--------|
| `data/azbukina_temporal_atlas.h5ad` | Job 3 |
| `data/amin_kelley_mapped.h5ad` | Step 02 (primary screen mapping, already completed) |
| `data/morphogen_matrix_amin_kelley.csv` | `morphogen_parser.py` (already completed) |

Verify prerequisites:

```bash
python -c "
from pathlib import Path
d = Path('../data')
for f in [
    'azbukina_temporal_atlas.h5ad',
    'amin_kelley_mapped.h5ad',
    'morphogen_matrix_amin_kelley.csv',
]:
    p = d / f
    print(f'  {\"OK\" if p.exists() else \"MISSING\"}: {p}')
    assert p.exists(), f'Missing: {p}'
print('All Job 5 prerequisites met.')
"
```

### Command

```bash
python 05_cellrank2_virtual.py
```

The script runs 4 internal steps:
1. Load and preprocess the temporal atlas
2. Compute moscot OT transport maps (cached to `data/cellrank2_transport_maps.pkl`)
3. Load query data and project forward to Days 90 and 120
4. Save virtual training data

### Moscot Solver Tuning

Environment variables control the OT solver parameters (defaults are usually fine):

```bash
# Optional: override moscot parameters
export GPBO_MOSCOT_EPSILON=1e-3   # Sinkhorn regularization
export GPBO_MOSCOT_TAU_A=0.94     # Unbalanced transport marginal relaxation
export GPBO_MOSCOT_TAU_B=0.94
```

### Expected Outputs

| File | Description |
|------|-------------|
| `data/cellrank2_virtual_fractions.csv` | Virtual cell type fractions (index = virtual condition names) |
| `data/cellrank2_virtual_morphogens.csv` | Morphogen vectors for virtual points (same as real, duplicated for each target timepoint) |
| `data/cellrank2_transport_quality.csv` | Transport quality report with status per condition (OK, HIGH_COST, NOT_CONVERGED) |
| `data/cellrank2_transport_maps.pkl` | Cached moscot transport maps (reusable for re-runs) |

The transport quality CSV is consumed by `merge_multi_fidelity_data()` in step 04 to inflate noise variance for virtual conditions routed through HIGH_COST or NOT_CONVERGED transitions.

### Estimated Time

1-2 hours on CPU. The moscot OT solver is the bottleneck. Transport maps are cached after the first run, so re-runs skip this step.

### Validation

```bash
python -c "
import pandas as pd
fracs = pd.read_csv('../data/cellrank2_virtual_fractions.csv', index_col=0)
morphs = pd.read_csv('../data/cellrank2_virtual_morphogens.csv', index_col=0)
tq = pd.read_csv('../data/cellrank2_transport_quality.csv')
print(f'Virtual fractions shape: {fracs.shape}')
print(f'Virtual morphogens shape: {morphs.shape}')
assert fracs.shape[0] == morphs.shape[0], 'Row count mismatch'
# Expect ~92 virtual data points (46 conditions x 2 target timepoints)
assert fracs.shape[0] > 0, 'No virtual data generated'
sums = fracs.sum(axis=1)
assert (sums - 1.0).abs().max() < 0.05, f'Fractions do not sum to 1'
print(f'Transport quality statuses: {tq[\"status\"].value_counts().to_dict()}')
print(f'Virtual data points: {fracs.shape[0]}')
print('Job 5 validation PASSED.')
"
```

---

## Job 6: Train CellFlow Model (GPU Required)

**Purpose:** Train a CellFlow (Klein et al. 2025) flow-matching generative model to predict single-cell distributions from novel protocol encodings. This enables low-fidelity (0.0) virtual screening of ~23,000 morphogen combinations.

### Architecture

CellFlow uses JAX/Flax for flow matching:
- **Protocol encoding:** RDKit molecular fingerprints (small molecules) + ESM2 protein embeddings (Lin et al. 2023) for growth factors, plus concentration, timing window, and pathway annotations
- **Training data:** Sanchis-Calleja patterning screen (176 conditions) + Amin/Kelley primary screen
- **Output:** Predicted single-cell distributions per protocol, converted to cell type fractions

### Prerequisites

| Requirement | Details |
|-------------|---------|
| GPU | NVIDIA GPU with CUDA support (JAX does not use MPS) |
| JAX + Flax | `pip install jax[cuda12] flax` (version-dependent on CUDA) |
| RDKit | `pip install rdkit` for molecular fingerprints |
| ESM2 | `pip install fair-esm` for protein embeddings |
| `data/patterning_screen/` | Job 2 outputs |
| `data/amin_kelley_mapped.h5ad` | Step 02 (primary screen) |

### Training

CellFlow training is not yet fully automated in the pipeline. The general workflow:

1. Prepare training data (protocol encodings + matched scRNA-seq)
2. Train the flow-matching model (~4-8 hours on a single GPU)
3. Save the trained model to `data/cellflow_model/`

The model is loaded by `predict_cellflow()` in `gopro/06_cellflow_virtual.py`, which searches these paths in order:
- `data/cellflow_model/`
- `data/cellflow_model.pt`
- `data/patterning_screen/cellflow_model/`

### Running Virtual Screening (After Training)

```bash
python 06_cellflow_virtual.py
```

### Fallback Mode (No Trained Model)

Without a trained CellFlow model, the step is gated. To use the heuristic baseline fallback (hand-tuned sigmoid dose-response, no literature basis -- for development/testing only):

```bash
python 06_cellflow_virtual.py --use-cellflow-fallback
```

The fallback uses `_heuristic_baseline_predict()` which applies sigmoid dose-response curves with pathway antagonism scaling. This is explicitly not recommended for production use.

### Expected Outputs (When Model Exists)

| File | Description |
|------|-------------|
| `data/cellflow_model/` | Trained CellFlow model directory |
| `data/cellflow_virtual_fractions.csv` | Predicted cell type fractions (up to 5000 virtual conditions) |
| `data/cellflow_virtual_morphogens.csv` | Morphogen concentration vectors for virtual conditions |
| `data/cellflow_screening_report.csv` | Quality metrics per prediction (confidence scores) |

### Estimated Time

- Model training: 4-8 hours on GPU
- Virtual screening: 30-60 minutes on GPU (or CPU with degraded performance)

### Current Status

CellFlow integration is partially implemented. The `predict_cellflow()` function in `06_cellflow_virtual.py` checks for a trained model and falls back to the heuristic baseline only if `--use-cellflow-fallback` is passed. **For Round 1 of GP-BO, CellFlow virtual data is optional.** The multi-fidelity GP works with real data (fidelity=1.0) + CellRank2 virtual data (fidelity=0.5) alone.

---

## Job 7: Run Full GP-BO Round 1

**Purpose:** Fit a multi-fidelity Gaussian Process on all available data sources and generate 24 recommended experiments for the next wet-lab round.

### Prerequisites

All of these must exist. Jobs 1, 4, and 5 are required; Job 6 is optional.

| File | Source | Required |
|------|--------|----------|
| `data/gp_training_labels_amin_kelley.csv` | Step 02 (primary screen) | Yes |
| `data/morphogen_matrix_amin_kelley.csv` | `morphogen_parser.py` | Yes |
| `data/gp_training_labels_sag_screen.csv` | Job 1 | Yes |
| `data/morphogen_matrix_sag_screen.csv` | `morphogen_parser.py` | Yes |
| `data/gp_training_labels_sanchis_calleja.csv` | Job 4 | Yes |
| `data/morphogen_matrix_sanchis_calleja.csv` | Needs SanchisCallejaParser | Yes |
| `data/cellrank2_virtual_fractions.csv` | Job 5 | Yes |
| `data/cellrank2_virtual_morphogens.csv` | Job 5 | Yes |
| `data/cellrank2_transport_quality.csv` | Job 5 | Yes (auto-read) |
| `data/cellflow_virtual_fractions.csv` | Job 6 | No |
| `data/cellflow_virtual_morphogens.csv` | Job 6 | No |

Verify prerequisites:

```bash
python -c "
from pathlib import Path
d = Path('../data')
required = [
    'gp_training_labels_amin_kelley.csv',
    'morphogen_matrix_amin_kelley.csv',
    'gp_training_labels_sag_screen.csv',
    'morphogen_matrix_sag_screen.csv',
    'gp_training_labels_sanchis_calleja.csv',
    'morphogen_matrix_sanchis_calleja.csv',
    'cellrank2_virtual_fractions.csv',
    'cellrank2_virtual_morphogens.csv',
]
optional = [
    'cellflow_virtual_fractions.csv',
    'cellflow_virtual_morphogens.csv',
]
all_ok = True
for f in required:
    p = d / f
    ok = p.exists()
    if not ok: all_ok = False
    print(f'  {\"OK\" if ok else \"MISSING\"} (required): {f}')
for f in optional:
    p = d / f
    print(f'  {\"OK\" if p.exists() else \"ABSENT\"} (optional): {f}')
assert all_ok, 'Missing required prerequisites'
print('All Job 7 prerequisites met.')
"
```

### Full Command (All Data Sources)

```bash
python 04_gpbo_loop.py \
  --fractions data/gp_training_labels_amin_kelley.csv \
  --morphogens data/morphogen_matrix_amin_kelley.csv \
  --sag-fractions data/gp_training_labels_sag_screen.csv \
  --sag-morphogens data/morphogen_matrix_sag_screen.csv \
  --sanchis-fractions data/gp_training_labels_sanchis_calleja.csv \
  --sanchis-morphogens data/morphogen_matrix_sanchis_calleja.csv \
  --cellrank2-fractions data/cellrank2_virtual_fractions.csv \
  --cellrank2-morphogens data/cellrank2_virtual_morphogens.csv \
  --n-recommendations 24 \
  --round 1 \
  --n-controls 2 \
  --contextual-cols log_harvest_day \
  --cost-weight 0.1 \
  --input-warp \
  --explicit-priors \
  --mc-samples 1024
```

### Flag Reference

| Flag | Purpose |
|------|---------|
| `--n-recommendations 24` | Recommend 24 new experiments (one 24-well plate) |
| `--round 1` | Optimization round number (affects output filenames) |
| `--n-controls 2` | Include 2 control conditions in the plate map |
| `--contextual-cols log_harvest_day` | Treat harvest day as a contextual (non-optimized) dimension |
| `--cost-weight 0.1` | Penalize expensive morphogen combinations (cost-aware acquisition) |
| `--input-warp` | Apply input warping (Kumaraswamy CDF transform) to handle non-stationary morphogen effects |
| `--explicit-priors` | Use explicit GP hyperparameter priors (recommended for stability) |
| `--mc-samples 1024` | Monte Carlo samples for acquisition function (higher = more accurate, slower) |

### Minimal Command (Without Sanchis-Calleja or CellFlow)

If Job 4 is not complete or the morphogen parser for Sanchis-Calleja is not ready:

```bash
python 04_gpbo_loop.py \
  --fractions data/gp_training_labels_amin_kelley.csv \
  --morphogens data/morphogen_matrix_amin_kelley.csv \
  --sag-fractions data/gp_training_labels_sag_screen.csv \
  --sag-morphogens data/morphogen_matrix_sag_screen.csv \
  --cellrank2-fractions data/cellrank2_virtual_fractions.csv \
  --cellrank2-morphogens data/cellrank2_virtual_morphogens.csv \
  --n-recommendations 24 \
  --round 1 \
  --n-controls 2 \
  --input-warp \
  --explicit-priors \
  --mc-samples 1024
```

### With CellFlow Virtual Data (If Job 6 Complete)

Add these flags to either command above:

```bash
  --cellflow-fractions data/cellflow_virtual_fractions.csv \
  --cellflow-morphogens data/cellflow_virtual_morphogens.csv \
  --cellflow-variance-inflation 2.0 \
  --cellflow-relevance-check
```

### Multi-Fidelity Data Integration

Step 04 uses `merge_multi_fidelity_data()` to combine data sources at different fidelity levels:

| Source | Fidelity | Notes |
|--------|----------|-------|
| Amin/Kelley primary screen (46 conditions) | 1.0 | Real scRNA-seq data, Day 72 |
| SAG secondary screen (2 conditions) | 1.0 | Real scRNA-seq data, Day 70 |
| Sanchis-Calleja patterning screen | Configurable via `--sanchis-fidelity` (default from `SANCHIS_CALLEJA_DEFAULT_FIDELITY` in config) | Real data but different lab/protocol |
| CellRank2 virtual projections | 0.5 | OT-projected, noise inflated by transport quality |
| CellFlow virtual screening | 0.0 | Generative model predictions, lowest trust |

### Expected Outputs

| File | Description |
|------|-------------|
| `data/gp_recommendations_round1.csv` | Plate map with 24 recommended morphogen combinations |
| `data/gp_diagnostics_round1.csv` | GP kernel parameters, lengthscales, model diagnostics |
| `data/convergence_diagnostics.csv` | Convergence metrics (acquisition decay, cluster spread, posterior variance) |
| `data/gp_state/` | Saved GP model state for warm-starting future rounds |

### Estimated Time

30-90 minutes on CPU, depending on data size and `--mc-samples`. The GP fitting (BoTorch `SingleTaskGP` with Matern 5/2 + ARD) uses CPU only because MPS does not support float64.

### Validation

```bash
python -c "
import pandas as pd

# Check recommendations
recs = pd.read_csv('../data/gp_recommendations_round1.csv', index_col=0)
print(f'Recommendations shape: {recs.shape}')
assert recs.shape[0] == 24, f'Expected 24 recommendations, got {recs.shape[0]}'
print(f'Columns: {list(recs.columns[:8])}...')

# Check diagnostics
diag = pd.read_csv('../data/gp_diagnostics_round1.csv')
print(f'Diagnostics shape: {diag.shape}')

# Check convergence
conv = pd.read_csv('../data/convergence_diagnostics.csv')
print(f'Convergence metrics: {list(conv.columns)}')

print('Job 7 validation PASSED.')
"
```

### Generating the Visualization Report

After Job 7 completes:

```bash
python 05_visualize.py
```

This generates `data/report_round1.html` -- a self-contained Plotly HTML report showing:
- Morphogen PCA (recommended vs. existing conditions)
- Plate map layout
- Feature importance (ARD lengthscales)
- Condition leaderboard
- Cell type composition bar charts
- Convergence diagnostics

---

## Execution Checklist

```
[ ] Job 1: SAG screen mapping (02_map_to_hnoca.py --output-prefix sag_screen)
[ ] Job 2: Download patterning screen (00b_download_patterning_screen.py)
[ ] Job 3: Build temporal atlas (00c_build_temporal_atlas.py)
[ ] Job 4: Map Sanchis-Calleja to HNOCA (02_map_to_hnoca.py --output-prefix sanchis_calleja)
[ ] Job 5: Generate CellRank2 virtual data (05_cellrank2_virtual.py)
[ ] Job 6: Train CellFlow model (GPU, optional for Round 1)
[ ] Job 7: Run GP-BO Round 1 (04_gpbo_loop.py with all flags)
[ ] Generate visualization report (05_visualize.py)
```

---

## Troubleshooting

### Common Issues

**scPoli fine-tuning OOM:** If memory is insufficient for scPoli in Jobs 1 or 4, the script may silently fail. Monitor memory usage. The Sanchis-Calleja dataset (Job 4) is the largest.

**moscot solver not converging (Job 5):** If transport maps do not converge, try adjusting `GPBO_MOSCOT_TAU_A` and `GPBO_MOSCOT_TAU_B` downward (e.g., 0.90). Lower values allow more unbalanced transport.

**R package installation fails (Job 2):** SeuratDisk is not on CRAN. Install from GitHub: `remotes::install_github("mojaveazure/seurat-disk")`. Requires R >= 4.2.

**GP fitting fails with Cholesky error (Job 7):** This typically means the kernel matrix is near-singular. Try adding `--fixed-noise` to use fixed observation noise instead of learned noise, or reduce the number of data sources.

**Missing morphogen matrix for Sanchis-Calleja:** The `morphogen_matrix_sanchis_calleja.csv` requires parsing the patterning screen condition names into 24D morphogen vectors. If no parser exists yet, this is a prerequisite code task before Job 7 can include `--sanchis-fractions`/`--sanchis-morphogens`.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPBO_PROJECT_DIR` | Auto-detected | Project root directory |
| `GPBO_DATA_DIR` | `{project}/data` | Data directory |
| `GPBO_MODEL_DIR` | `{data}/neural_organoid_atlas/...` | scPoli model directory |
| `GPBO_LOG_LEVEL` | `INFO` | Logging verbosity |
| `GPBO_MOSCOT_EPSILON` | `1e-3` | Moscot Sinkhorn regularization |
| `GPBO_MOSCOT_TAU_A` | `0.94` | Moscot marginal relaxation (source) |
| `GPBO_MOSCOT_TAU_B` | `0.94` | Moscot marginal relaxation (target) |
