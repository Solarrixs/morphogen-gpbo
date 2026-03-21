# Apple Silicon (M4) Optimization Guide

> Created: 2026-03-19
> Status: Implemented (CPU optimizations); GPU blocked by upstream bugs

## Summary

MPS GPU acceleration is **blocked** for all critical pipeline components:

| Component | MPS? | Reason |
|-----------|------|--------|
| scPoli (Jobs 1, 4) | **No** | `torch.lgamma` MPS bug → NaN in epoch 1 (pytorch#132605) |
| BoTorch GP (Job 7) | **No** | float64 unsupported on MPS (hardware limitation) |
| moscot/JAX (Job 5) | **Risky** | jax-metal abandoned since Jan 2025; jax-mps experimental |
| CellFlow (Job 6) | **Risky** | Same JAX Metal constraints; RNG perf regression |

**All gains come from CPU-side optimization.** This is already implemented.

## What's implemented

### 1. BLAS thread tuning (`gopro/config.py`)

Sets `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`, etc. to **performance core count only**
(10 for M4 Pro, 12 for M4 Max). Efficiency cores are slower and cause thread contention.

Override with: `export GPBO_PERF_CORES=12` (for M4 Max)

Must execute before `import numpy` — placed at top of `config.py`.

### 2. PyTorch thread tuning (`gopro/04_gpbo_loop.py`)

Sets `torch.set_num_threads()` and `torch.set_num_interop_threads()` to match
performance core count. Affects GP fitting, acquisition optimization.

### 3. NumPy uses Apple Accelerate (verified)

NumPy BLAS backend is already `accelerate` (Apple's optimized linear algebra).
This gives ~2-3x speedup for KNN, PCA, SVD operations in scanpy.

## How to run optimally

### Parallel job execution

```bash
cd /Users/maxxyung/Projects/engram-projects/morphogen-gpbo/gopro

# Terminal 1: Job 1 (SAG mapping, ~30-60 min)
python 02_map_to_hnoca.py \
  --input ../data/amin_kelley_sag_screen.h5ad \
  --output-prefix sag_screen

# Terminal 2: Job 3 → Job 5 (atlas + CellRank2, ~1.5 hr)
python 00c_build_temporal_atlas.py && python 05_cellrank2_virtual.py

# Terminal 3 (after Job 3 finishes): Job 4 (Sanchis mapping, ~1-3 hr)
python 02_map_to_hnoca.py \
  --input ../data/azbukina_temporal_atlas.h5ad \
  --output-prefix sanchis_calleja

# After all complete: Job 7 (GP-BO, ~30-90 min)
python 04_gpbo_loop.py \
  --fractions ../data/gp_training_labels_amin_kelley.csv \
  --morphogens ../data/morphogen_matrix_amin_kelley.csv \
  --sag-fractions ../data/gp_training_labels_sag_screen.csv \
  --sag-morphogens ../data/morphogen_matrix_sag_screen.csv \
  --cellrank2-fractions ../data/cellrank2_virtual_fractions.csv \
  --cellrank2-morphogens ../data/cellrank2_virtual_morphogens.csv \
  --n-recommendations 24 --round 1 --n-controls 2 \
  --input-warp --explicit-priors --sequential-batch --mc-samples 1024
```

### Minimum viable Round 1 (fastest path, ~1-2 hr)

Skip Sanchis-Calleja and CellRank2 — use only primary + SAG + CellFlow:

```bash
# Only need Job 1
python 02_map_to_hnoca.py \
  --input ../data/amin_kelley_sag_screen.h5ad \
  --output-prefix sag_screen

# Then GP-BO
python 04_gpbo_loop.py \
  --fractions ../data/gp_training_labels_amin_kelley.csv \
  --morphogens ../data/morphogen_matrix_amin_kelley.csv \
  --sag-fractions ../data/gp_training_labels_sag_screen.csv \
  --sag-morphogens ../data/morphogen_matrix_sag_screen.csv \
  --cellflow-fractions ../data/cellflow_virtual_fractions_200.csv \
  --cellflow-morphogens ../data/cellflow_virtual_morphogens_200.csv \
  --n-recommendations 24 --round 1 --n-controls 2 \
  --input-warp --explicit-priors --sequential-batch --mc-samples 1024
```

### Memory considerations

- Job 4 (Sanchis-Calleja) loads large h5ad files. Needs 32+ GB RAM.
- If 16 GB, use backed mode: add `backed='r'` to `anndata.read_h5ad()` calls.
- Or run Job 4 on a cloud instance with more memory.

## GPU status — revisit conditions

- **scPoli on MPS**: Blocked until pytorch#132605 is fixed. Check:
  `python -c "import torch; x=torch.lgamma(torch.tensor([1.0], device='mps').expand(2,3)); print(x)"`
  If this doesn't produce NaN, MPS may be safe to try.

- **jax-mps for moscot**: Install `pip install jax-mps`, set `JAX_PLATFORMS=mps`.
  Test: `python -c "import jax; print(jax.devices())"`
  If Metal device appears, try `python 05_cellrank2_virtual.py`.

- **BoTorch on MPS**: Will never work until Apple adds float64 to Metal.

## References

- pytorch#132605: torch.lgamma MPS broadcasting bug
- scverse discourse: MPS acceleration with scvi
- BoTorch#1444: float64 best practices
- jax-metal PyPI: last release Jan 2025
- Apple Accelerate: numpy BLAS on arm64
