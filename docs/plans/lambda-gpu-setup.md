# Lambda Labs GPU Pipeline Setup Guide

> Created: 2026-03-19
> Status: Ready to use
> Cost: ~$2.60 per full pipeline run (A100 80GB @ $1.29/hr × ~2 hours)

## One-time Setup (5 minutes)

### 1. Get Lambda API Key

1. Go to https://cloud.lambdalabs.com/api-keys
2. Click "Generate API Key"
3. Add to your shell profile:
   ```bash
   echo 'export LAMBDA_API_KEY="your-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

### 2. Generate SSH Key (if you don't have one)

```bash
# Check if you already have a key
ls ~/.ssh/id_ed25519.pub

# If not, generate one
ssh-keygen -t ed25519 -C "morphogen-gpbo-lambda"
# Press Enter for default path, no passphrase (or add one)
```

The `run_on_lambda.sh` script will automatically register this key with Lambda on first run.

### 3. Install jq (JSON processor)

```bash
brew install jq
```

### 4. Verify Setup

```bash
# Test API access (Lambda uses HTTP Basic auth: API key as username, empty password)
curl -s -u "$LAMBDA_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/instance-types | jq '.data | keys'
```

You should see a list of available instance types like `gpu_1x_a100_sxm4`.

**Important**: `shutdown` commands from within the instance do NOT terminate it — they put
it in `alert` status and billing continues. Always use the API (or the script) to terminate.

## Running the Pipeline

### Full pipeline (all jobs, ~2 hours, ~$2.60)

```bash
cd /Users/maxxyung/Projects/engram-projects/morphogen-gpbo
./scripts/run_on_lambda.sh
```

This will:
1. Provision an A100 80GB instance (~1-2 min)
2. Upload code + data via rsync (~10-15 min for ~15GB)
3. Install CUDA PyTorch + JAX + dependencies (~5 min)
4. Run pipeline Jobs 1-7 with GPU acceleration (~1-1.5 hours)
5. Pull results back to your Mac (~2 min)
6. Terminate the instance (stops billing)

### Dry run (see what would happen)

```bash
./scripts/run_on_lambda.sh --dry-run
```

### Keep instance alive (for debugging)

```bash
./scripts/run_on_lambda.sh --keep
# SSH in manually:
ssh ubuntu@<IP_ADDRESS> -i ~/.ssh/id_ed25519
# When done, terminate via Lambda dashboard
```

### Skip upload (data already on instance)

```bash
./scripts/run_on_lambda.sh --skip-upload
```

## What Runs on GPU vs CPU

| Step | Device | Time (A100) | Time (M4 CPU) |
|------|--------|-------------|---------------|
| Job 1: scPoli SAG mapping | **CUDA GPU** | ~5-10 min | ~30-60 min |
| Job 3: Build temporal atlas | CPU | ~10 min | ~10-30 min |
| Job 4: scPoli Sanchis-Calleja | **CUDA GPU** | ~15-30 min | ~1-3 hr |
| Job 5: CellRank2 (moscot OT) | **CUDA GPU** | ~10-20 min | ~1-2 hr |
| Job 7: GP-BO Round 1 | CPU (float64) | ~30-60 min | ~30-90 min |
| **Total** | | **~1.5-2 hr** | **~4+ hr** |

### Pipeline execution on Lambda

```
Phase 1 (parallel):  Job 1 (scPoli SAG) ─────────┐
                     Job 3 (atlas build) ─────────┤
                                                   │
Phase 2 (parallel):  Job 4 (scPoli Sanchis) ──────┤  (after Job 3)
                     Job 5 (CellRank2) ────────────┤  (after Job 3)
                                                   │
Phase 3 (serial):    Job 7 (GP-BO Round 1) ────────┘  (after all)
```

## Error Handling

The script handles failures gracefully:

- **Job 1 fails** (SAG mapping): Pipeline aborts — SAG screen is required for GP-BO
- **Job 3/4/5 fails** (non-critical): GP-BO runs with reduced data sources
- **Any failure**: Instance is still terminated (EXIT trap), partial results pulled
- **Network interruption**: Re-run with `--skip-upload` to resume

## Output Files

After a successful run, these files will be on your Mac in `data/`:

| File | Description |
|------|-------------|
| `gp_recommendations_round1.csv` | 24 recommended morphogen cocktails |
| `gp_diagnostics_round1.csv` | GP kernel parameters and model diagnostics |
| `convergence_diagnostics.csv` | Convergence tracking metrics |
| `gp_training_labels_sag_screen.csv` | SAG screen cell type fractions |
| `sag_screen_mapped.h5ad` | SAG screen mapped AnnData |
| `gp_training_labels_sanchis_calleja.csv` | Sanchis-Calleja fractions (if Job 4 succeeded) |
| `cellrank2_virtual_fractions.csv` | CellRank2 virtual data (if Job 5 succeeded) |
| `pipeline_timing.log` | Per-phase timing report |

## Troubleshooting

### "No A100 instances available"
Lambda has limited GPU supply. The script auto-falls back to A100 (non-SXM4), A6000, or H100.
If all are unavailable, try again in 30-60 minutes or check https://cloud.lambdalabs.com/instances.

### rsync is slow
First upload is ~15GB. Subsequent runs (with `--skip-upload` or if instance is kept) skip this.
For faster transfer, consider compressing h5ad files first:
```bash
# Pre-compress large files (one-time)
pigz -k data/braun-et-al_minimal_for_mapping.h5ad  # 11GB → ~3GB
```

### Pipeline fails on remote
Check the log:
```bash
ssh ubuntu@<IP> cat ~/morphogen-gpbo/pipeline_run.log
```

### Instance not terminating
Check Lambda dashboard: https://cloud.lambdalabs.com/instances
Manually terminate any running instances to stop billing.

## Cost Management

- **Full pipeline**: ~$2.60 (2 hours × $1.29/hr)
- **Data upload only**: ~$0.20 (10 min)
- **Idle instance**: $1.29/hr — always terminate when done
- **Monthly cap**: Set a spending limit in Lambda dashboard → Settings → Billing
