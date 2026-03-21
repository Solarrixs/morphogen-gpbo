#!/usr/bin/env bash
# ==============================================================================
# run_on_lambda.sh — Orchestrate GP-BO pipeline on a Lambda Labs A100 instance
#
# Usage: ./scripts/run_on_lambda.sh [--keep] [--skip-upload] [--dry-run]
#   --keep          Don't terminate instance after run (for debugging)
#   --skip-upload   Skip data upload (data already on instance)
#   --dry-run       Show what would happen without provisioning
#
# Prerequisites:
#   - LAMBDA_API_KEY env var set (https://cloud.lambdalabs.com/api-keys)
#   - SSH key at ~/.ssh/id_ed25519 (or override with LAMBDA_SSH_KEY)
#   - jq installed (brew install jq)
#   - rsync installed
#
# Estimated cost: ~$2.60 for 2 hours on gpu_1x_a100_sxm4 ($1.29/hr)
# ==============================================================================
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

LAMBDA_API_BASE="https://cloud.lambdalabs.com/api/v1"
INSTANCE_TYPE="gpu_1x_a100_sxm4"
INSTANCE_TYPE_FALLBACKS=("gpu_1x_a100" "gpu_1x_a6000" "gpu_1x_h100_sxm5")
HOURLY_RATE="1.29"  # USD for gpu_1x_a100_sxm4
ESTIMATED_HOURS="2"
SSH_KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_USER="ubuntu"
REMOTE_DIR="morphogen-gpbo"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_ID=""  # populated after launch

# ── Flags ────────────────────────────────────────────────────────────────────

FLAG_KEEP=false
FLAG_SKIP_UPLOAD=false
FLAG_DRY_RUN=false

# ── Usage ────────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
Usage: ./scripts/run_on_lambda.sh [OPTIONS]

Provision a Lambda Labs A100 80GB instance, upload data, run the GP-BO
pipeline, pull results, and terminate the instance.

Options:
  --keep          Don't terminate instance after run (for debugging)
  --skip-upload   Skip data upload (data already on instance)
  --dry-run       Show what would happen without provisioning
  -h, --help      Show this help message

Environment variables:
  LAMBDA_API_KEY    (required) Lambda Labs API key
  LAMBDA_SSH_KEY    SSH key path (default: ~/.ssh/id_ed25519)
  LAMBDA_SSH_NAME   SSH key name registered in Lambda (default: auto-detected)
  LAMBDA_REGION     Preferred region (default: auto-selected)

USAGE
    exit 0
}

# ── Argument parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)        FLAG_KEEP=true; shift ;;
        --skip-upload) FLAG_SKIP_UPLOAD=true; shift ;;
        --dry-run)     FLAG_DRY_RUN=true; shift ;;
        -h|--help)     usage ;;
        *)             echo "ERROR: Unknown option: $1"; usage ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }
warn() { log "WARNING: $*"; }

lambda_api() {
    local method="$1" endpoint="$2"
    shift 2
    # Lambda Cloud API uses HTTP Basic auth: API key as username, empty password
    curl -s -X "$method" \
        -u "${LAMBDA_API_KEY}:" \
        -H "Content-Type: application/json" \
        "${LAMBDA_API_BASE}${endpoint}" "$@"
}

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 \
        -o LogLevel=ERROR \
        -i "${SSH_KEY}" \
        "${SSH_USER}@${INSTANCE_IP}" "$@"
}

rsync_to() {
    rsync -az --progress \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i \"${SSH_KEY}\"" \
        "$@"
}

rsync_from() {
    rsync -az --progress \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i \"${SSH_KEY}\"" \
        "$@"
}

# ── Cleanup trap ─────────────────────────────────────────────────────────────

cleanup() {
    local exit_code=$?
    if [[ -n "${INSTANCE_ID}" ]] && [[ "${FLAG_KEEP}" == false ]]; then
        log "Terminating instance ${INSTANCE_ID}..."
        local resp
        resp=$(lambda_api POST "/instance-operations/terminate" \
            -d "{\"instance_ids\": [\"${INSTANCE_ID}\"]}" 2>/dev/null || true)
        if echo "${resp}" | jq -e '.terminated_instances' >/dev/null 2>&1; then
            log "Instance ${INSTANCE_ID} terminated."
        else
            warn "Failed to terminate instance ${INSTANCE_ID}. Terminate manually!"
            warn "  curl -X POST ${LAMBDA_API_BASE}/instance-operations/terminate \\"
            warn "    -u '\$LAMBDA_API_KEY:' \\"
            warn "    -H 'Content-Type: application/json' \\"
            warn "    -d '{\"instance_ids\": [\"${INSTANCE_ID}\"]}'"
        fi
    elif [[ -n "${INSTANCE_ID}" ]] && [[ "${FLAG_KEEP}" == true ]]; then
        log "Instance ${INSTANCE_ID} kept alive (--keep flag). Remember to terminate it!"
        log "  IP: ${INSTANCE_IP:-unknown}"
        log "  ssh -i ${SSH_KEY} ${SSH_USER}@${INSTANCE_IP:-unknown}"
    fi
    exit "${exit_code}"
}

trap cleanup EXIT
# Note: SIGKILL (kill -9) and OOM kills bypass all traps. If the process is
# forcibly killed or your Mac loses power, check Lambda dashboard manually:
# https://cloud.lambdalabs.com/instances
trap 'log "Interrupted — cleaning up..."' INT TERM

# ── Step 0: Check prerequisites ──────────────────────────────────────────────

log "Checking prerequisites..."

[[ -n "${LAMBDA_API_KEY:-}" ]] || die "LAMBDA_API_KEY environment variable not set.
  Get your key at: https://cloud.lambdalabs.com/api-keys
  Then: export LAMBDA_API_KEY='your-key-here'"

[[ -f "${SSH_KEY}" ]] || die "SSH key not found at ${SSH_KEY}.
  Generate one:  ssh-keygen -t ed25519 -f ${SSH_KEY}
  Or set:        export LAMBDA_SSH_KEY=/path/to/your/key"

[[ -f "${SSH_KEY}.pub" ]] || die "SSH public key not found at ${SSH_KEY}.pub"

command -v jq >/dev/null 2>&1 || die "jq not found. Install with: brew install jq"
command -v rsync >/dev/null 2>&1 || die "rsync not found."
command -v curl >/dev/null 2>&1 || die "curl not found."

# Verify API key works
log "Verifying Lambda API key..."
api_test=$(lambda_api GET "/ssh-keys")
if echo "${api_test}" | jq -e '.error' >/dev/null 2>&1; then
    die "Lambda API key invalid: $(echo "${api_test}" | jq -r '.error.message // .error')"
fi
log "API key verified."

# ── Step 1: Ensure SSH key is registered ─────────────────────────────────────

log "Checking SSH key registration..."
SSH_PUB_KEY=$(cat "${SSH_KEY}.pub")
# Fingerprint used for debug logging if needed
# shellcheck disable=SC2034
SSH_KEY_FINGERPRINT=$(ssh-keygen -lf "${SSH_KEY}.pub" | awk '{print $2}')

# List registered keys and find ours
REGISTERED_KEYS=$(lambda_api GET "/ssh-keys")
LAMBDA_SSH_NAME="${LAMBDA_SSH_NAME:-}"

if [[ -z "${LAMBDA_SSH_NAME}" ]]; then
    # Try to find our key by fingerprint match -- check if any registered key
    # has a matching public key
    LAMBDA_SSH_NAME=$(echo "${REGISTERED_KEYS}" | jq -r \
        --arg pub "${SSH_PUB_KEY}" \
        '.data[] | select(.public_key == $pub) | .name' | head -1)
fi

if [[ -z "${LAMBDA_SSH_NAME}" ]]; then
    LAMBDA_SSH_NAME="morphogen-gpbo-$(date +%Y%m%d)"
    log "Registering SSH key as '${LAMBDA_SSH_NAME}'..."
    reg_resp=$(lambda_api POST "/ssh-keys" \
        -d "{\"name\": \"${LAMBDA_SSH_NAME}\", \"public_key\": \"${SSH_PUB_KEY}\"}")
    if echo "${reg_resp}" | jq -e '.error' >/dev/null 2>&1; then
        die "Failed to register SSH key: $(echo "${reg_resp}" | jq -r '.error.message // .error')"
    fi
    log "SSH key registered."
else
    log "SSH key already registered as '${LAMBDA_SSH_NAME}'."
fi

# ── Step 2: Find available region ────────────────────────────────────────────

log "Checking A100 80GB availability..."
AVAIL=$(lambda_api GET "/instance-types")

find_available_region() {
    local itype="$1"
    echo "${AVAIL}" | jq -r \
        --arg t "${itype}" \
        '.data[$t].regions_with_capacity_available[]?.name // empty' | head -1
}

REGION=""
SELECTED_TYPE=""

# Try preferred region first
if [[ -n "${LAMBDA_REGION:-}" ]]; then
    HAS_REGION=$(echo "${AVAIL}" | jq -r \
        --arg t "${INSTANCE_TYPE}" --arg r "${LAMBDA_REGION}" \
        '[.data[$t].regions_with_capacity_available[]?.name] | map(select(. == $r)) | length')
    if [[ "${HAS_REGION}" -gt 0 ]]; then
        REGION="${LAMBDA_REGION}"
        SELECTED_TYPE="${INSTANCE_TYPE}"
    fi
fi

# Auto-select region for primary type
if [[ -z "${REGION}" ]]; then
    REGION=$(find_available_region "${INSTANCE_TYPE}")
    if [[ -n "${REGION}" ]]; then
        SELECTED_TYPE="${INSTANCE_TYPE}"
    fi
fi

# Try fallback instance types
if [[ -z "${REGION}" ]]; then
    warn "${INSTANCE_TYPE} not available in any region."
    for fallback in "${INSTANCE_TYPE_FALLBACKS[@]}"; do
        REGION=$(find_available_region "${fallback}")
        if [[ -n "${REGION}" ]]; then
            SELECTED_TYPE="${fallback}"
            warn "Using fallback instance type: ${SELECTED_TYPE} in ${REGION}"
            # Update cost estimate for fallback
            local_rate=$(echo "${AVAIL}" | jq -r \
                --arg t "${SELECTED_TYPE}" \
                '.data[$t].instance_type.price_cents_per_hour // 0' 2>/dev/null)
            if [[ "${local_rate}" -gt 0 ]]; then
                HOURLY_RATE=$(echo "scale=2; ${local_rate} / 100" | bc)
            fi
            break
        fi
    done
fi

if [[ -z "${REGION}" ]]; then
    log "No GPU instances available. Current availability:"
    echo "${AVAIL}" | jq -r '.data | to_entries[] | select(.value.regions_with_capacity_available | length > 0) | "  \(.key): \([.value.regions_with_capacity_available[].name] | join(", "))"'
    die "No suitable GPU instance available. Try again later or check https://cloud.lambdalabs.com/instances"
fi

ESTIMATED_COST=$(echo "scale=2; ${HOURLY_RATE} * ${ESTIMATED_HOURS}" | bc)
log "Selected: ${SELECTED_TYPE} in ${REGION}"
log "Estimated cost: ~\$${ESTIMATED_COST} for ${ESTIMATED_HOURS} hours at \$${HOURLY_RATE}/hr"

# ── Dry run exit ─────────────────────────────────────────────────────────────

if [[ "${FLAG_DRY_RUN}" == true ]]; then
    log "DRY RUN — would provision ${SELECTED_TYPE} in ${REGION}"
    log "DRY RUN — would upload ~15 GB of data from ${PROJECT_DIR}"
    log "DRY RUN — would run pipeline_runner.sh on remote instance"
    log "DRY RUN — would pull results back to ${PROJECT_DIR}/data/"
    log "DRY RUN — would terminate instance after completion"
    exit 0
fi

# ── Step 3: Launch instance ──────────────────────────────────────────────────

log "Launching ${SELECTED_TYPE} in ${REGION}..."
LAUNCH_RESP=$(lambda_api POST "/instance-operations/launch" \
    -d "{
        \"region_name\": \"${REGION}\",
        \"instance_type_name\": \"${SELECTED_TYPE}\",
        \"ssh_key_names\": [\"${LAMBDA_SSH_NAME}\"],
        \"quantity\": 1,
        \"name\": \"morphogen-gpbo-$(date +%Y%m%d-%H%M%S)\"
    }")

if echo "${LAUNCH_RESP}" | jq -e '.error' >/dev/null 2>&1; then
    die "Launch failed: $(echo "${LAUNCH_RESP}" | jq -r '.error.message // .error')"
fi

INSTANCE_ID=$(echo "${LAUNCH_RESP}" | jq -r '.data.instance_ids[0]')
if [[ -z "${INSTANCE_ID}" ]] || [[ "${INSTANCE_ID}" == "null" ]]; then
    die "Launch succeeded but no instance ID returned: ${LAUNCH_RESP}"
fi
log "Instance launched: ${INSTANCE_ID}"

# ── Step 4: Wait for instance to become active ──────────────────────────────

log "Waiting for instance to become active..."
INSTANCE_IP=""
MAX_WAIT=600  # 10 minutes
ELAPSED=0
POLL_INTERVAL=15

while [[ ${ELAPSED} -lt ${MAX_WAIT} ]]; do
    INSTANCE_INFO=$(lambda_api GET "/instances/${INSTANCE_ID}")
    STATUS=$(echo "${INSTANCE_INFO}" | jq -r '.data.status')
    INSTANCE_IP=$(echo "${INSTANCE_INFO}" | jq -r '.data.ip // empty')

    case "${STATUS}" in
        active)
            if [[ -n "${INSTANCE_IP}" ]]; then
                log "Instance active at ${INSTANCE_IP}"
                break
            fi
            ;;
        booting|unhealthy)
            ;;
        terminated|error)
            die "Instance entered state '${STATUS}'. Check Lambda dashboard."
            ;;
    esac

    printf '  Status: %s (%ds elapsed)\r' "${STATUS}" "${ELAPSED}"
    sleep "${POLL_INTERVAL}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [[ -z "${INSTANCE_IP}" ]]; then
    die "Instance did not become active within ${MAX_WAIT}s."
fi

# Wait a bit for SSH to be ready
log "Waiting for SSH to become available..."
SSH_READY=false
for _ in $(seq 1 12); do
    if ssh_cmd "echo ok" >/dev/null 2>&1; then
        SSH_READY=true
        break
    fi
    sleep 10
done

if [[ "${SSH_READY}" == false ]]; then
    die "SSH not available after 2 minutes. Instance IP: ${INSTANCE_IP}"
fi

log "SSH connection established."

# ── Step 5: Upload data ─────────────────────────────────────────────────────

if [[ "${FLAG_SKIP_UPLOAD}" == true ]]; then
    log "Skipping data upload (--skip-upload)."
else
    log "Uploading code and data to instance (~15 GB, this may take a while)..."

    # Create remote directory
    ssh_cmd "mkdir -p ~/${REMOTE_DIR}/data ~/${REMOTE_DIR}/scripts"

    # Upload pipeline code
    log "  Uploading gopro/ ..."
    rsync_to \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "${PROJECT_DIR}/gopro/" \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/gopro/"

    # Upload scripts
    log "  Uploading scripts/ ..."
    rsync_to \
        "${PROJECT_DIR}/scripts/pipeline_runner.sh" \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/scripts/"

    # Upload h5ad files
    log "  Uploading *.h5ad files ..."
    rsync_to \
        --include='*.h5ad' \
        --exclude='*' \
        "${PROJECT_DIR}/data/" \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/data/"

    # Upload CSV files
    log "  Uploading *.csv files ..."
    rsync_to \
        --include='*.csv' \
        --exclude='*' \
        "${PROJECT_DIR}/data/" \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/data/"

    # Upload scPoli model
    log "  Uploading scPoli model ..."
    rsync_to \
        "${PROJECT_DIR}/data/neural_organoid_atlas/" \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/data/neural_organoid_atlas/"

    # Upload patterning screen
    if [[ -d "${PROJECT_DIR}/data/patterning_screen" ]]; then
        log "  Uploading patterning screen ..."
        rsync_to \
            "${PROJECT_DIR}/data/patterning_screen/" \
            "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/data/patterning_screen/"
    else
        warn "data/patterning_screen/ not found locally. Job 2 will download it on the instance."
    fi

    log "Upload complete."
fi

# ── Step 6: Run pipeline ────────────────────────────────────────────────────

log "Starting pipeline on remote instance..."
log "  Instance: ${SELECTED_TYPE} (${INSTANCE_IP})"
log "  Pipeline log: ~/morphogen-gpbo/pipeline_run.log on remote"

ssh_cmd "chmod +x ~/${REMOTE_DIR}/scripts/pipeline_runner.sh"

# Run pipeline_runner.sh; tee to both remote log and local stdout
# Note: Job 6 (CellFlow training) is omitted — not required for Round 1.
# See docs/specs/pipeline-execution-spec.md for details.
# Timeout after 4 hours (generous for ~2hr estimated run) to prevent billing runaway.
START_TIME=$(date +%s)
ssh_cmd "cd ~/${REMOTE_DIR} && timeout 14400 bash scripts/pipeline_runner.sh 2>&1 | tee pipeline_run.log" || {
    EXIT_CODE=$?
    warn "Pipeline exited with code ${EXIT_CODE}."
    warn "Pulling partial results and logs..."
    rsync_from \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/pipeline_run.log" \
        "${PROJECT_DIR}/pipeline_run.log" 2>/dev/null || true
    rsync_from \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/pipeline_timing.log" \
        "${PROJECT_DIR}/pipeline_timing.log" 2>/dev/null || true
}
END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))
log "Pipeline finished in $((DURATION / 60))m $((DURATION % 60))s."

# ── Step 7: Pull results ────────────────────────────────────────────────────

log "Pulling results from instance..."

RESULT_FILES=(
    "data/gp_training_labels_sag_screen.csv"
    "data/gp_training_regions_sag_screen.csv"
    "data/sag_screen_mapped.h5ad"
    "data/gp_training_labels_sanchis_calleja.csv"
    "data/gp_training_regions_sanchis_calleja.csv"
    "data/sanchis_calleja_mapped.h5ad"
    "data/azbukina_temporal_atlas.h5ad"
    "data/cellrank2_virtual_fractions.csv"
    "data/cellrank2_virtual_morphogens.csv"
    "data/cellrank2_transport_quality.csv"
    "data/gp_recommendations_round1.csv"
    "data/gp_diagnostics_round1.csv"
    "data/convergence_diagnostics.csv"
    "data/fidelity_noise_round1.csv"
    "data/report_round1.html"
)

for f in "${RESULT_FILES[@]}"; do
    if ssh_cmd "test -f ~/${REMOTE_DIR}/${f}" 2>/dev/null; then
        log "  Pulling ${f} ..."
        mkdir -p "${PROJECT_DIR}/$(dirname "${f}")"
        rsync_from \
            "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/${f}" \
            "${PROJECT_DIR}/${f}"
    else
        warn "  Not found on remote: ${f}"
    fi
done

# Pull gp_state directory
if ssh_cmd "test -d ~/${REMOTE_DIR}/data/gp_state" 2>/dev/null; then
    log "  Pulling data/gp_state/ ..."
    mkdir -p "${PROJECT_DIR}/data/gp_state"
    rsync_from \
        "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/data/gp_state/" \
        "${PROJECT_DIR}/data/gp_state/"
fi

# Pull any additional round files (glob pattern)
for pattern in "gp_recommendations_round*.csv" "gp_diagnostics_round*.csv" "fidelity_noise_round*.csv"; do
    ssh_cmd "ls ~/${REMOTE_DIR}/data/${pattern} 2>/dev/null" | while read -r remote_file; do
        basename_f=$(basename "${remote_file}")
        if [[ ! -f "${PROJECT_DIR}/data/${basename_f}" ]]; then
            log "  Pulling data/${basename_f} ..."
            rsync_from \
                "${SSH_USER}@${INSTANCE_IP}:${remote_file}" \
                "${PROJECT_DIR}/data/${basename_f}"
        fi
    done
done

# Pull logs
rsync_from \
    "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/pipeline_run.log" \
    "${PROJECT_DIR}/pipeline_run.log" 2>/dev/null || true
rsync_from \
    "${SSH_USER}@${INSTANCE_IP}:~/${REMOTE_DIR}/pipeline_timing.log" \
    "${PROJECT_DIR}/pipeline_timing.log" 2>/dev/null || true

log "Results pulled to ${PROJECT_DIR}/data/"

# ── Step 8: Summary ─────────────────────────────────────────────────────────

ACTUAL_HOURS=$(echo "scale=2; ${DURATION} / 3600" | bc)
ACTUAL_COST=$(echo "scale=2; ${ACTUAL_HOURS} * ${HOURLY_RATE}" | bc)

log "==============================================="
log "Pipeline complete."
log "  Duration:  $((DURATION / 60))m $((DURATION % 60))s"
log "  Est. cost: ~\$${ACTUAL_COST} (${ACTUAL_HOURS} hrs at \$${HOURLY_RATE}/hr)"
log "  Results:   ${PROJECT_DIR}/data/"
log "  Full log:  ${PROJECT_DIR}/pipeline_run.log"
log "  Timing:    ${PROJECT_DIR}/pipeline_timing.log"
if [[ "${FLAG_KEEP}" == true ]]; then
    log "  Instance:  ${INSTANCE_IP} (kept alive — remember to terminate!)"
else
    log "  Instance:  terminated"
fi
log "==============================================="
