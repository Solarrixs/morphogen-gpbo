#!/usr/bin/env bash
# ==============================================================================
# pipeline_runner.sh — Run GP-BO pipeline on Lambda Labs GPU instance
#
# Called by run_on_lambda.sh via SSH. Assumes code and data are already
# uploaded to ~/morphogen-gpbo/.
#
# Pipeline phases:
#   Phase 1: SAG screen mapping + temporal atlas build (parallel)
#   Phase 2: Sanchis-Calleja mapping + CellRank2 virtual data (parallel)
#   Phase 3: GP-BO Round 1 (sequential, uses all outputs)
# ==============================================================================
set -euo pipefail

WORKDIR="${HOME}/morphogen-gpbo"
DATA_DIR="${WORKDIR}/data"
TIMING_LOG="${WORKDIR}/pipeline_timing.log"
GOPRO_DIR="${WORKDIR}/gopro"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "FATAL: $*"; exit 1; }
warn() { log "WARNING: $*"; }

# Record phase timing to log file
time_phase() {
    local phase_name="$1"
    shift
    local start end elapsed
    start=$(date +%s)
    log "START: ${phase_name}"
    set +e
    "$@"
    local rc=$?
    set -e
    end=$(date +%s)
    elapsed=$((end - start))
    printf '%s\t%dm %ds\t(exit %d)\n' "${phase_name}" $((elapsed / 60)) $((elapsed % 60)) "${rc}" >> "${TIMING_LOG}"
    log "END: ${phase_name} — ${elapsed}s (exit ${rc})"
    return ${rc}
}

validate_csv_fractions() {
    local csv_path="$1"
    local min_rows="${2:-1}"
    python3 -c "
import pandas as pd, sys
csv_path, min_rows = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(csv_path, index_col=0)
print(f'  Shape: {df.shape}')
print(f'  Conditions: {list(df.index[:5])}...')
assert df.shape[0] >= min_rows, f'Expected >= {min_rows} rows, got {df.shape[0]}'
sums = df.sum(axis=1)
assert (sums - 1.0).abs().max() < 0.05, f'Fractions do not sum to 1: max deviation {(sums - 1.0).abs().max():.4f}'
print('  Fractions OK.')
" "${csv_path}" "${min_rows}"
}

# ── Initialize timing log ────────────────────────────────────────────────────

echo "Pipeline run started: $(date)" > "${TIMING_LOG}"
echo "Instance: $(hostname)" >> "${TIMING_LOG}"
echo "---" >> "${TIMING_LOG}"

PIPELINE_START=$(date +%s)

# ── Environment setup ────────────────────────────────────────────────────────

log "Setting up Python environment..."
cd "${WORKDIR}"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip --quiet

# Install PyTorch with CUDA support first (before requirements.txt pulls CPU torch)
log "Installing PyTorch with CUDA 12.4 support..."
pip install torch --index-url https://download.pytorch.org/whl/cu124 --quiet

# Install JAX with CUDA support (needed for moscot/CellRank2)
log "Installing JAX with CUDA 12 support..."
pip install "jax[cuda12]" --quiet

# Install pipeline requirements
log "Installing pipeline requirements..."
pip install -r gopro/requirements.txt --quiet

log "Environment setup complete."

# ── Verify GPU ───────────────────────────────────────────────────────────────

log "Verifying GPU..."
nvidia-smi || warn "nvidia-smi failed — GPU may not be available."

python3 -c "
import torch
if torch.cuda.is_available():
    print(f'PyTorch CUDA: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: PyTorch CUDA not available. GP-BO uses CPU anyway, but scArches benefits from GPU.')
"

python3 -c "
try:
    import jax
    devices = jax.devices()
    print(f'JAX devices: {devices}')
except Exception as e:
    print(f'WARNING: JAX GPU check failed: {e}')
" || true

# ── Verify prerequisites ────────────────────────────────────────────────────

log "Verifying data prerequisites..."
cd "${GOPRO_DIR}"

python3 -c "
from pathlib import Path
d = Path('../data')
required = {
    'amin_kelley_sag_screen.h5ad': 'SAG screen (Step 01)',
    'hnoca_minimal_for_mapping.h5ad': 'HNOCA reference (Step 00)',
    'neural_organoid_atlas/supplemental_files/scpoli_model_params/model_params.pt': 'scPoli model',
    'gp_training_labels_amin_kelley.csv': 'Primary screen labels (Step 02)',
    'morphogen_matrix_amin_kelley.csv': 'Primary morphogen matrix',
    'morphogen_matrix_sag_screen.csv': 'SAG morphogen matrix',
    'amin_kelley_mapped.h5ad': 'Primary mapped data (Step 02)',
}
optional = {
    'patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz': 'Patterning screen (Job 2)',
}
all_ok = True
for f, desc in required.items():
    p = d / f
    ok = p.exists()
    if not ok:
        all_ok = False
    print(f'  {\"OK\" if ok else \"MISSING\"} (required): {f} — {desc}')
for f, desc in optional.items():
    p = d / f
    print(f'  {\"OK\" if p.exists() else \"ABSENT\"} (optional): {f} — {desc}')
if not all_ok:
    raise SystemExit('Missing required prerequisites. Upload data first.')
print('All prerequisites verified.')
"

# ── Phase 1: SAG screen mapping + Build temporal atlas (parallel) ────────────

log "========== PHASE 1: SAG mapping + Temporal atlas (parallel) =========="

# Job 1: Map SAG screen to HNOCA
run_job1() {
    log "[Job 1] Mapping SAG screen to HNOCA..."
    python3 02_map_to_hnoca.py \
        --input ${DATA_DIR}/amin_kelley_sag_screen.h5ad \
        --output-prefix sag_screen
}

# Job 3: Build temporal atlas (Job 2 = download, assumed already done)
run_job3() {
    if [[ -f "${DATA_DIR}/patterning_screen/OSMGT_processed_files/exp1_processed_8.h5ad.gz" ]]; then
        log "[Job 3] Building temporal atlas..."
        python3 00c_build_temporal_atlas.py
    else
        # Try downloading first
        log "[Job 2+3] Downloading patterning screen, then building atlas..."
        python3 00b_download_patterning_screen.py
        python3 00c_build_temporal_atlas.py
    fi
}

# Run in parallel
time_phase "Job 1: SAG screen mapping" run_job1 &
PID_JOB1=$!

time_phase "Job 3: Build temporal atlas" run_job3 &
PID_JOB3=$!

# Wait for both, track individual failures
JOB3_OK=true

if ! wait "${PID_JOB1}"; then
    die "Job 1 (SAG screen mapping) failed. This is required for GP-BO. Aborting."
fi

if ! wait "${PID_JOB3}"; then
    JOB3_OK=false
    warn "Job 3 (temporal atlas) failed. Continuing without Sanchis-Calleja and CellRank2 data."
fi

# Validate Job 1 output
log "Validating Job 1 outputs..."
validate_csv_fractions "${DATA_DIR}/gp_training_labels_sag_screen.csv" 2

if [[ "${JOB3_OK}" == true ]]; then
    python3 -c "
import anndata
a = anndata.read_h5ad('${DATA_DIR}/azbukina_temporal_atlas.h5ad', backed='r')
print(f'  Temporal atlas: {a.n_obs:,} cells, {a.n_vars:,} genes')
assert 'day' in a.obs.columns, 'Missing day column in temporal atlas'
print('  Job 3 validation OK.')
"
fi

log "Phase 1 complete."

# ── Phase 2: Sanchis-Calleja mapping + CellRank2 virtual data (parallel) ────

if [[ "${JOB3_OK}" == true ]]; then
    log "========== PHASE 2: Sanchis-Calleja mapping + CellRank2 (parallel) =========="

    # Job 4: Map Sanchis-Calleja to HNOCA
    run_job4() {
        log "[Job 4] Mapping Sanchis-Calleja to HNOCA..."
        python3 02_map_to_hnoca.py \
            --input ${DATA_DIR}/azbukina_temporal_atlas.h5ad \
            --output-prefix sanchis_calleja \
            --condition-key condition \
            --batch-key sample
    }

    # Job 5: CellRank2 virtual data
    run_job5() {
        log "[Job 5] Generating CellRank2 virtual data..."
        python3 05_cellrank2_virtual.py
    }

    # Run in parallel
    time_phase "Job 4: Sanchis-Calleja mapping" run_job4 &
    PID_JOB4=$!

    time_phase "Job 5: CellRank2 virtual data" run_job5 &
    PID_JOB5=$!

    JOB4_OK=true
    JOB5_OK=true

    if ! wait "${PID_JOB4}"; then
        JOB4_OK=false
        warn "Job 4 (Sanchis-Calleja mapping) failed. Continuing without Sanchis-Calleja data."
    fi

    if ! wait "${PID_JOB5}"; then
        JOB5_OK=false
        warn "Job 5 (CellRank2 virtual data) failed. Continuing without CellRank2 data."
    fi

    # Validate outputs
    if [[ "${JOB4_OK}" == true ]]; then
        log "Validating Job 4 outputs..."
        validate_csv_fractions "${DATA_DIR}/gp_training_labels_sanchis_calleja.csv" 1
    fi

    if [[ "${JOB5_OK}" == true ]]; then
        log "Validating Job 5 outputs..."
        validate_csv_fractions "${DATA_DIR}/cellrank2_virtual_fractions.csv" 1
        python3 -c "
import pandas as pd
tq = pd.read_csv('${DATA_DIR}/cellrank2_transport_quality.csv')
print(f'  Transport quality: {tq[\"status\"].value_counts().to_dict()}')
print('  Job 5 validation OK.')
"
    fi

    log "Phase 2 complete."
else
    JOB4_OK=false
    JOB5_OK=false
    warn "Skipping Phase 2 (depends on temporal atlas from Phase 1)."
fi

# ── Phase 3: GP-BO Round 1 ──────────────────────────────────────────────────

log "========== PHASE 3: GP-BO Round 1 =========="

# Build the command dynamically based on which jobs succeeded
GPBO_CMD=(
    python3 04_gpbo_loop.py
    --fractions ${DATA_DIR}/gp_training_labels_amin_kelley.csv
    --morphogens ${DATA_DIR}/morphogen_matrix_amin_kelley.csv
    --sag-fractions ${DATA_DIR}/gp_training_labels_sag_screen.csv
    --sag-morphogens ${DATA_DIR}/morphogen_matrix_sag_screen.csv
)

if [[ "${JOB4_OK}" == true ]] && [[ -f "${DATA_DIR}/gp_training_labels_sanchis_calleja.csv" ]] && [[ -f "${DATA_DIR}/morphogen_matrix_sanchis_calleja.csv" ]]; then
    log "  Including Sanchis-Calleja data (fidelity=0.7)"
    GPBO_CMD+=(
        --sanchis-fractions ${DATA_DIR}/gp_training_labels_sanchis_calleja.csv
        --sanchis-morphogens ${DATA_DIR}/morphogen_matrix_sanchis_calleja.csv
    )
else
    warn "  Sanchis-Calleja data not available — running without it."
fi

if [[ "${JOB5_OK}" == true ]] && [[ -f "${DATA_DIR}/cellrank2_virtual_fractions.csv" ]] && [[ -f "${DATA_DIR}/cellrank2_virtual_morphogens.csv" ]]; then
    log "  Including CellRank2 virtual data (fidelity=0.5)"
    GPBO_CMD+=(
        --cellrank2-fractions ${DATA_DIR}/cellrank2_virtual_fractions.csv
        --cellrank2-morphogens ${DATA_DIR}/cellrank2_virtual_morphogens.csv
    )
else
    warn "  CellRank2 virtual data not available — running without it."
fi

# Add optimization flags
GPBO_CMD+=(
    --n-recommendations 24
    --round 1
    --n-controls 2
    --input-warp
    --explicit-priors
    --sequential-batch
    --contextual-cols log_harvest_day
    --cost-weight 0.1
    --mc-samples 1024
)

log "Running GP-BO with command:"
log "  ${GPBO_CMD[*]}"

run_gpbo() {
    "${GPBO_CMD[@]}"
}

time_phase "Job 7: GP-BO Round 1" run_gpbo

# Validate GP-BO outputs
log "Validating GP-BO outputs..."
python3 -c "
import pandas as pd

recs = pd.read_csv('${DATA_DIR}/gp_recommendations_round1.csv', index_col=0)
print(f'  Recommendations: {recs.shape[0]} conditions x {recs.shape[1]} features')
assert recs.shape[0] == 24, f'Expected 24 recommendations, got {recs.shape[0]}'

diag = pd.read_csv('${DATA_DIR}/gp_diagnostics_round1.csv')
print(f'  Diagnostics: {diag.shape}')

if __import__('pathlib').Path('${DATA_DIR}/convergence_diagnostics.csv').exists():
    conv = pd.read_csv('${DATA_DIR}/convergence_diagnostics.csv')
    print(f'  Convergence metrics: {list(conv.columns)}')

print('  Job 7 validation PASSED.')
"

# ── Generate visualization report ────────────────────────────────────────────

log "Generating visualization report..."
run_viz() {
    python3 05_visualize.py || warn "Visualization generation failed (non-critical)."
}
time_phase "Visualization report" run_viz || true

# ── Summary ──────────────────────────────────────────────────────────────────

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))
echo "---" >> "${TIMING_LOG}"
printf 'TOTAL\t%dm %ds\n' $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60)) >> "${TIMING_LOG}"

log "==============================================="
log "Pipeline complete."
log "  Total time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
log ""
log "  Phase results:"
log "    Job 1 (SAG mapping):       OK"
log "    Job 3 (Temporal atlas):    $(if ${JOB3_OK}; then echo OK; else echo FAILED; fi)"
log "    Job 4 (Sanchis-Calleja):   $(if ${JOB4_OK}; then echo OK; else echo SKIPPED/FAILED; fi)"
log "    Job 5 (CellRank2 virtual): $(if ${JOB5_OK}; then echo OK; else echo SKIPPED/FAILED; fi)"
log "    Job 7 (GP-BO Round 1):     OK"
log ""
log "  Output files:"
log "    data/gp_recommendations_round1.csv"
log "    data/gp_diagnostics_round1.csv"
log "    data/convergence_diagnostics.csv"
log ""
log "  Timing log: ${TIMING_LOG}"
log "==============================================="

cat "${TIMING_LOG}"
