# GP-BO Visualization Report Design

**Date:** 2026-03-08
**Status:** Approved
**Branch:** maxx-frontend

## Goal

Generate a self-contained interactive HTML report after each GP-BO round that communicates the optimization state to three audiences: ML engineers (model diagnostics), lab biologists (what to do next), and PIs (are we converging?).

## Format

- Static HTML file with embedded Plotly charts (no server needed)
- Generated as pipeline step 05, after `04_gpbo_loop.py`
- Clean scientific aesthetic: white background, viridis/colorbrewer palette, minimal chrome
- Future: hosted dashboard version

## Report Sections

### 1. Auto-generated text summary
Short paragraph generated from data: round number, top morphogens by importance, best condition name and fidelity score, number of recommendations. No templates to maintain.

### 2. Convergence trend chart
Line chart of best composite fidelity score per round. Single point for round 0, grows with each round. Shows whether optimization is improving.

### 3. UMAP Panel A — Morphogen space
UMAP projection of 20D morphogen input vectors. Each dot = one experimental condition. Color = GP-predicted fidelity (viridis). Stars = 24 recommended next experiments. Previous rounds dimmed, current round bright.

### 4. UMAP Panel B — Cell space
Standard scRNA-seq UMAP of individual cells from the mapped h5ad. Color = cell type (Level 2 brain regions). Hover links cells to their source condition, connecting biology to the optimization.

### 5. Plate map (4x6 grid)
Visual 24-well plate layout. Each well colored by predicted fidelity. Hover reveals full morphogen recipe (20 concentrations). Wells labeled A1-D6.

### 6. Morphogen importance ranking
Horizontal bar chart of 1/lengthscale from ARD Matern kernel, sorted descending. Shows which of the 20 morphogen dimensions the GP considers most informative.

### 7. Best conditions leaderboard
Table of top 10 conditions ranked by composite fidelity. Columns: condition name, composite fidelity, RSS score, on-target fraction, off-target fraction, entropy, and the top 3 morphogen concentrations.

### 8. Cell type composition
Stacked bar chart per condition showing Level 2 (brain region) fractions. Sorted by fidelity score. Separate smaller panel with Level 1 (broad) categories as summary view.

## Data Inputs

| File | Source step | Content |
|------|------------|---------|
| `fidelity_report.csv` | 03 | Per-condition fidelity scores and sub-metrics |
| `gp_recommendations_round{N}.csv` | 04 | 24 recommended morphogen conditions with acquisition values |
| `gp_diagnostics_round{N}.csv` | 04 | GP model diagnostics including lengthscales |
| `gp_training_labels_*.csv` | 02 | Cell type fractions per condition |
| `gp_training_regions_*.csv` | 02 | Brain region fractions per condition |
| `amin_kelley_mapped.h5ad` | 02 | Mapped cells with UMAP coordinates and labels |

## Architecture

- Single script: `gopro/05_visualize.py`
- Auto-detects number of rounds from `gp_recommendations_round*.csv` files on disk
- Designed for multi-round from day 1 (cumulative views)
- Output: `data/report_round{N}.html`

## Dependencies

New: `plotly`, `umap-learn`
Existing: `pandas`, `numpy`, `anndata`

## Future Work

- Hosted dashboard (Streamlit or similar) with live filtering
- Per-section captions for non-technical readers
- Side-by-side round comparison view
