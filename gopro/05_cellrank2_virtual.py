"""
Step 5: CellRank 2 Virtual Data Generation via Temporal Projection.

Uses moscot optimal transport + CellRank 2 RealTimeKernel to forward-predict
cell fates from early timepoint data, generating medium-fidelity virtual
training points for the multi-fidelity GP.

The Azbukina temporal atlas (Days 7, 15, 30, 60, 90, 120) provides the
transport maps. When a new experiment is mapped at Day 21, cells are projected
forward through these maps to predict cell type fractions at later timepoints.

Each real experiment generates 3 virtual data points (Day 30, 60, 90),
providing 3x data amplification per dollar spent.

Inputs:
  - data/azbukina_temporal_atlas.h5ad (temporal atlas with day annotations)
  - data/amin_kelley_mapped.h5ad (or any mapped query data from step 02)
  - data/gp_training_labels_*.csv (real cell type fractions from step 02)
  - data/morphogen_matrix_*.csv (morphogen concentrations)

Outputs:
  - data/cellrank2_transport_maps.pkl (cached moscot transport maps)
  - data/cellrank2_virtual_fractions.csv (virtual cell type fractions)
  - data/cellrank2_virtual_morphogens.csv (morphogen vectors for virtual points)
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

# Timepoints in the Azbukina temporal atlas
ATLAS_TIMEPOINTS = [7, 15, 30, 60, 90, 120]

# Virtual projection targets from Day 21 query data
PROJECTION_TARGETS = [30, 60, 90]

# Medium fidelity level for CellRank 2 virtual data
FIDELITY_LEVEL = 0.5

# Moscot OT solver parameters
MOSCOT_EPSILON = 1e-3  # Entropic regularization
MOSCOT_TAU_A = 0.94    # Unbalancedness parameter (source)
MOSCOT_TAU_B = 0.94    # Unbalancedness parameter (target)


def load_temporal_atlas(
    atlas_path: Path,
    time_key: str = "day",
) -> sc.AnnData:
    """Load the Azbukina temporal atlas with timepoint annotations.

    Args:
        atlas_path: Path to temporal atlas h5ad file.
        time_key: Column in obs containing timepoint information.

    Returns:
        AnnData with numeric time_key in obs.
    """
    logger.info("Loading temporal atlas from %s...", atlas_path.name)
    adata = sc.read_h5ad(str(atlas_path))
    logger.info("Atlas: %s cells x %s genes", f"{adata.n_obs:,}", f"{adata.n_vars:,}")

    if time_key not in adata.obs.columns:
        raise ValueError(
            f"Time key '{time_key}' not found in atlas obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Ensure time key is numeric
    adata.obs[time_key] = pd.to_numeric(adata.obs[time_key])
    timepoints = sorted(adata.obs[time_key].unique())
    logger.info("Timepoints: %s", timepoints)

    for tp in timepoints:
        n = (adata.obs[time_key] == tp).sum()
        logger.info("Day %s: %s cells", tp, f"{n:,}")

    return adata


def preprocess_for_moscot(
    adata: sc.AnnData,
    n_pcs: int = 30,
    n_neighbors: int = 30,
) -> sc.AnnData:
    """Preprocess AnnData for moscot optimal transport.

    Computes PCA and neighbor graph if not already present.

    Args:
        adata: AnnData to preprocess.
        n_pcs: Number of PCA components.
        n_neighbors: Number of neighbors for k-NN graph.

    Returns:
        Preprocessed AnnData with PCA in obsm and neighbors computed.
    """
    if "X_pca" not in adata.obsm:
        logger.info("Computing PCA...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=True)

    if "neighbors" not in adata.uns:
        logger.info("Computing neighbor graph...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    return adata


def compute_transport_maps(
    adata: sc.AnnData,
    time_key: str = "day",
    epsilon: float = MOSCOT_EPSILON,
    tau_a: float = MOSCOT_TAU_A,
    tau_b: float = MOSCOT_TAU_B,
    cache_path: Optional[Path] = None,
) -> object:
    """Compute moscot optimal transport maps between consecutive timepoints.

    Args:
        adata: Temporal atlas AnnData with time_key in obs.
        time_key: Column containing timepoint annotations.
        epsilon: Entropic regularization parameter.
        tau_a: Unbalancedness parameter for source marginal.
        tau_b: Unbalancedness parameter for target marginal.
        cache_path: If provided, cache/load transport maps.

    Returns:
        moscot TemporalProblem with solved transport maps.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached transport maps from %s...", cache_path.name)
        with open(str(cache_path), "rb") as f:
            return pickle.load(f)

    import moscot as mt

    logger.info("Setting up moscot TemporalProblem...")
    problem = mt.problems.TemporalProblem(adata)
    problem = problem.prepare(
        time_key=time_key,
        joint_attr={"attr": "obsm", "key": "X_pca"},
    )

    logger.info("Solving optimal transport...")
    problem = problem.solve(
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
    )

    if cache_path is not None:
        logger.info("Caching transport maps to %s...", cache_path.name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_path), "wb") as f:
            pickle.dump(problem, f)

    return problem


def build_cellrank_kernel(
    adata: sc.AnnData,
    problem: object,
    conn_weight: float = 0.2,
) -> object:
    """Build CellRank 2 RealTimeKernel from moscot transport maps.

    Args:
        adata: Temporal atlas AnnData.
        problem: Solved moscot TemporalProblem.
        conn_weight: Weight for connectivity kernel (graph smoothing).

    Returns:
        CellRank RealTimeKernel with computed transition matrix.
    """
    import cellrank as cr

    logger.info("Building CellRank 2 RealTimeKernel...")

    # Ensure time_key is categorical for CellRank
    if not isinstance(adata.obs["day"].dtype, pd.CategoricalDtype):
        adata.obs["day"] = adata.obs["day"].astype("category")

    rtk = cr.kernels.RealTimeKernel.from_moscot(problem)
    rtk = rtk.compute_transition_matrix(
        self_transitions="all",
        conn_weight=conn_weight,
        threshold="auto",
    )

    logger.info("Transition matrix computed.")
    return rtk


def compute_fate_probabilities(
    adata: sc.AnnData,
    kernel: object,
    n_macrostates: int = 12,
) -> tuple[object, pd.DataFrame]:
    """Compute cell fate probabilities using GPCCA estimator.

    Args:
        adata: Temporal atlas AnnData.
        kernel: CellRank kernel with transition matrix.
        n_macrostates: Number of macrostates to identify.

    Returns:
        Tuple of (GPCCA estimator, fate_probabilities DataFrame).
    """
    import cellrank as cr

    logger.info("Computing fate probabilities (n_macrostates=%d)...", n_macrostates)
    estimator = cr.estimators.GPCCA(kernel)
    estimator.compute_macrostates(n_states=n_macrostates)
    estimator.set_terminal_states()
    estimator.compute_fate_probabilities()

    fate_probs = estimator.fate_probabilities
    logger.info("Fate probabilities: %s", fate_probs.shape)
    logger.info("Terminal states: %s", list(estimator.terminal_states.cat.categories))

    return estimator, fate_probs


def project_query_forward(
    query_adata: sc.AnnData,
    atlas_adata: sc.AnnData,
    problem: object,
    query_timepoint: int = 21,
    target_timepoints: list[int] = None,
    label_key: str = "predicted_annot_level_2",
    condition_key: str = "condition",
) -> pd.DataFrame:
    """Project query cells forward in time using transport maps.

    For each query cell at query_timepoint, finds the nearest atlas cells
    and uses the transport maps to predict which cell types those cells
    would become at later timepoints.

    Args:
        query_adata: Mapped query AnnData with cell type labels.
        atlas_adata: Temporal atlas AnnData.
        problem: Solved moscot TemporalProblem.
        query_timepoint: Timepoint of query data.
        target_timepoints: Timepoints to project to.
        label_key: Cell type label column in query obs.
        condition_key: Condition column in query obs.

    Returns:
        DataFrame with virtual cell type fractions per condition per
        target timepoint. Index format: "{condition}_day{timepoint}".
    """
    if target_timepoints is None:
        target_timepoints = PROJECTION_TARGETS

    from sklearn.neighbors import NearestNeighbors

    logger.info("Projecting %s query cells from Day %d to %s...",
                f"{query_adata.n_obs:,}", query_timepoint, target_timepoints)

    # Find nearest atlas cells at the source timepoint closest to query
    # Use the atlas timepoint closest to but <= query_timepoint
    atlas_timepoints = sorted(atlas_adata.obs["day"].unique())
    source_tp = max([t for t in atlas_timepoints if t <= query_timepoint],
                    default=atlas_timepoints[0])
    logger.info("Using atlas Day %s as source for transport", source_tp)

    # Get source atlas cells
    source_mask = atlas_adata.obs["day"] == source_tp
    source_pca = atlas_adata[source_mask].obsm["X_pca"]
    source_indices = np.where(source_mask)[0]

    # Project query cells into atlas PCA space
    if "X_pca" in query_adata.obsm:
        query_embedding = query_adata.obsm["X_pca"]
        # Match dimensionality if needed
        min_dim = min(source_pca.shape[1], query_embedding.shape[1])
        source_pca = source_pca[:, :min_dim]
        query_embedding = query_embedding[:, :min_dim]
    else:
        # Re-embed query through atlas PCA loadings
        logger.info("Projecting query cells into atlas PCA space...")
        if "PCs" in atlas_adata.varm:
            pca_loadings = atlas_adata.varm["PCs"]
            hvg_mask = atlas_adata.var.get("highly_variable", pd.Series(True, index=atlas_adata.var_names))
            hvg_genes = atlas_adata.var_names[hvg_mask]
            shared_genes = query_adata.var_names.intersection(hvg_genes)
            logger.info("Shared HVGs for PCA projection: %d / %d", len(shared_genes), len(hvg_genes))
            if len(shared_genes) < 100:
                logger.warning("Low gene overlap (%d) — projection quality may be poor", len(shared_genes))

            # Get query expression for shared genes, normalize
            query_subset = query_adata[:, shared_genes].copy()
            sc.pp.normalize_total(query_subset, target_sum=1e4)
            sc.pp.log1p(query_subset)

            # Project through atlas PCA loadings
            gene_idx = [list(hvg_genes).index(g) for g in shared_genes]
            query_expr = query_subset.X.toarray() if hasattr(query_subset.X, "toarray") else np.asarray(query_subset.X)
            query_embedding = query_expr @ pca_loadings[gene_idx, :]

            # Match dimensions
            min_dim = min(source_pca.shape[1], query_embedding.shape[1])
            source_pca = source_pca[:, :min_dim]
            query_embedding = query_embedding[:, :min_dim]
        else:
            # Fallback: compute joint PCA
            logger.warning("No PCA loadings in atlas — computing joint PCA")
            combined = ad.concat([atlas_adata[source_mask], query_adata], join="inner")
            sc.pp.normalize_total(combined, target_sum=1e4)
            sc.pp.log1p(combined)
            sc.pp.highly_variable_genes(combined, n_top_genes=2000)
            sc.pp.pca(combined, n_comps=30)
            n_atlas = int(source_mask.sum())
            source_pca = combined.obsm["X_pca"][:n_atlas]
            query_embedding = combined.obsm["X_pca"][n_atlas:]

    # Find nearest atlas neighbors for each query cell
    logger.info("Finding nearest atlas neighbors for query cells...")
    nn = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=-1)
    nn.fit(source_pca)
    distances, neighbor_idx = nn.kneighbors(query_embedding)

    # Convert distances to weights (inverse distance)
    weights = 1.0 / (distances + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # For each target timepoint, use transport maps to predict cell types
    all_virtual_fractions = []
    conditions = query_adata.obs[condition_key].unique()

    for target_tp in target_timepoints:
        # Find the transport chain: source_tp -> ... -> target_tp
        intermediate_tps = [t for t in atlas_timepoints
                           if source_tp <= t <= target_tp]

        if len(intermediate_tps) < 2:
            logger.warning("No transport path from Day %s to Day %s, skipping",
                          source_tp, target_tp)
            continue

        logger.info("Projecting to Day %s (chain: %s)",
                    target_tp, " -> ".join(map(str, intermediate_tps)))

        # Get target timepoint atlas cells and their labels
        target_mask = atlas_adata.obs["day"] == target_tp
        target_obs = atlas_adata[target_mask].obs
        n_target_atlas = int(target_mask.sum())

        # Get cell type labels at target timepoint
        # Use the annotation available in the atlas
        target_label_col = None
        for candidate in [label_key, "annot_level_2", "cell_type",
                         "CellType", "celltype"]:
            if candidate in target_obs.columns:
                target_label_col = candidate
                break

        if target_label_col is None:
            logger.warning("No cell type label column found at Day %s", target_tp)
            continue

        target_labels = target_obs[target_label_col]

        # Harmonize labels if using a non-standard column
        if target_label_col != label_key:
            LABEL_HARMONIZATION = {
                "Excitatory neuron": "Cortical EN",
                "Inhibitory neuron": "Cortical IN",
                "Radial glia": "Cortical RG",
                "Intermediate progenitor": "Cortical IP",
                "Astrocyte": "Astroglia",
                "Oligodendrocyte precursor": "OPC",
                "Oligodendrocyte": "OPC",
                "Choroid plexus": "CP epithelial",
                "Microglia": "Microglia",
                "Endothelial": "Vascular endothelial",
                "Fibroblast": "Mesenchymal",
            }
            logger.info("Using atlas column '%s' — harmonizing to '%s' vocabulary", target_label_col, label_key)
            mapped = target_labels.map(LABEL_HARMONIZATION)
            n_unmapped = mapped.isna().sum()
            if n_unmapped > 0:
                unmapped_labels = target_labels[mapped.isna()].unique()
                logger.warning("%d cells (%d labels) unmapped: %s", n_unmapped, len(unmapped_labels), list(unmapped_labels)[:10])
                mapped = mapped.fillna(target_labels)  # keep unmapped as-is
            target_labels = mapped

        target_ct_fracs = target_labels.value_counts(normalize=True)

        # Compose transport across intermediate timepoints (sparse)
        # Done once per target_tp, shared across all conditions
        tp_pairs = list(zip(intermediate_tps[:-1], intermediate_tps[1:]))
        use_transport = True
        try:
            transport = problem[(tp_pairs[0][0], tp_pairs[0][1])].solution.transport_matrix
            if not sp.issparse(transport):
                transport = sp.csr_matrix(transport)
            elif not sp.isspmatrix_csr(transport):
                transport = transport.tocsr()
            for src_t, tgt_t in tp_pairs[1:]:
                t_next = problem[(src_t, tgt_t)].solution.transport_matrix
                if not sp.issparse(t_next):
                    t_next = sp.csr_matrix(t_next)
                elif not sp.isspmatrix_csr(t_next):
                    t_next = t_next.tocsr()
                transport = transport @ t_next  # sparse @ sparse = sparse

            # Dimension check: transport target dim must match atlas target cells
            n_target_transport = transport.shape[1]
            if n_target_transport != n_target_atlas:
                logger.warning(
                    "Transport target dim (%d) != atlas target cells (%d) at Day %s — "
                    "moscot may have subsampled. Using atlas average.",
                    n_target_transport, n_target_atlas, target_tp,
                )
                use_transport = False

            # Build local index lookup for vectorized neighbor mapping
            if use_transport:
                max_source_idx = int(source_indices.max()) + 1
                local_idx_map = -np.ones(max_source_idx, dtype=np.intp)
                local_idx_map[source_indices] = np.arange(len(source_indices))
                target_labels_arr = target_labels.values
        except (KeyError, AttributeError, IndexError) as e:
            logger.warning(
                "Transport composition failed at Day %s (%s), using atlas average for all conditions",
                target_tp, e,
            )
            use_transport = False

        # For each condition, compute predicted cell type fractions
        # weighted by transport probability
        for cond in conditions:
            cond_mask = query_adata.obs[condition_key] == cond
            cond_indices = np.where(cond_mask)[0]
            n_cells = len(cond_indices)

            if n_cells == 0:
                continue

            virtual_key = f"{cond}_day{target_tp}"

            if use_transport:
                try:
                    cond_nn = neighbor_idx[cond_indices]   # [n_cond_cells, k]
                    cond_w = weights[cond_indices]          # [n_cond_cells, k]

                    # Vectorized mapping of global neighbor indices to local source indices
                    flat_nn = cond_nn.ravel()
                    flat_w = cond_w.ravel()

                    # Map to local indices; out-of-range -> -1
                    in_range = flat_nn < len(local_idx_map)
                    flat_local = np.full_like(flat_nn, -1, dtype=np.intp)
                    flat_local[in_range] = local_idx_map[flat_nn[in_range]]
                    valid = (flat_local >= 0) & (flat_local < transport.shape[0])

                    if valid.any():
                        # Gather transport rows for valid neighbors (sparse row indexing)
                        t_rows = transport[flat_local[valid], :]  # CSR supports row fancy indexing
                        w_valid = flat_w[valid]

                        # Weight and sum: W @ t_rows where W is diagonal
                        W = sp.diags(w_valid)
                        weighted = W @ t_rows
                        target_dist = np.asarray(weighted.sum(axis=0)).ravel()

                        if target_dist.sum() > 0:
                            target_dist /= target_dist.sum()

                            # Compute cell type fractions from weighted target distribution
                            virtual_fracs = pd.Series(0.0, index=target_ct_fracs.index)
                            for ct in virtual_fracs.index:
                                ct_mask = (target_labels_arr == ct)
                                virtual_fracs[ct] = target_dist[ct_mask].sum()
                        else:
                            logger.warning("No valid transport for condition '%s' at Day %s, using atlas average", cond, target_tp)
                            virtual_fracs = target_ct_fracs.copy()
                    else:
                        logger.warning("No valid neighbors for condition '%s' at Day %s, using atlas average", cond, target_tp)
                        virtual_fracs = target_ct_fracs.copy()
                except (KeyError, AttributeError, IndexError) as e:
                    logger.warning("Transport lookup failed for condition '%s' at Day %s (%s), using atlas average", cond, target_tp, e)
                    virtual_fracs = target_ct_fracs.copy()
            else:
                virtual_fracs = target_ct_fracs.copy()

            all_virtual_fractions.append({
                "virtual_condition": virtual_key,
                "original_condition": cond,
                "target_day": target_tp,
                "n_source_cells": n_cells,
                **{ct: frac for ct, frac in virtual_fracs.items()},
            })

    if not all_virtual_fractions:
        logger.warning("No virtual fractions generated")
        return pd.DataFrame()

    result = pd.DataFrame(all_virtual_fractions)
    result = result.set_index("virtual_condition")

    # Fill NaN cell type columns with 0
    ct_cols = [c for c in result.columns
               if c not in ["original_condition", "target_day", "n_source_cells"]]
    result[ct_cols] = result[ct_cols].fillna(0.0)

    # Normalize rows to sum to 1
    row_sums = result[ct_cols].sum(axis=1)
    result[ct_cols] = result[ct_cols].div(row_sums, axis=0)

    logger.info("Generated %d virtual data points", len(result))
    return result


def build_virtual_morphogen_matrix(
    virtual_fractions: pd.DataFrame,
    real_morphogen_csv: Path,
) -> pd.DataFrame:
    """Build morphogen concentration matrix for virtual data points.

    Virtual points inherit the morphogen vector from their source condition,
    but with an updated log_harvest_day reflecting the target timepoint.

    Args:
        virtual_fractions: DataFrame with virtual_condition index and
            original_condition, target_day columns.
        real_morphogen_csv: Path to real morphogen matrix CSV.

    Returns:
        DataFrame with morphogen concentrations for virtual conditions.
    """
    real_morphogens = pd.read_csv(str(real_morphogen_csv), index_col=0)

    rows = []
    for virt_cond, row in virtual_fractions.iterrows():
        orig_cond = row["original_condition"]
        target_day = row["target_day"]

        if orig_cond not in real_morphogens.index:
            continue

        morph_vec = real_morphogens.loc[orig_cond].copy()
        # Update harvest day to target timepoint
        morph_vec["log_harvest_day"] = math.log(target_day)
        morph_vec.name = virt_cond
        rows.append(morph_vec)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.index.name = None
    return result


def generate_virtual_training_data(
    query_adata: sc.AnnData,
    atlas_adata: sc.AnnData,
    problem: object,
    real_morphogen_csv: Path,
    query_timepoint: int = 21,
    target_timepoints: list[int] = None,
    label_key: str = "predicted_annot_level_2",
    condition_key: str = "condition",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate virtual (X, Y) training data via CellRank 2 projection.

    This is the main entry point that combines projection + morphogen
    matrix construction.

    Args:
        query_adata: Mapped query AnnData.
        atlas_adata: Temporal atlas AnnData.
        problem: Solved moscot TemporalProblem.
        real_morphogen_csv: Path to real morphogen matrix.
        query_timepoint: Timepoint of query data.
        target_timepoints: Target projection timepoints.
        label_key: Cell type label column.
        condition_key: Condition column.

    Returns:
        Tuple of (virtual_morphogens_df, virtual_fractions_df).
        Both DataFrames have matching indices (virtual condition names).
    """
    # Project query forward
    virtual_fractions = project_query_forward(
        query_adata=query_adata,
        atlas_adata=atlas_adata,
        problem=problem,
        query_timepoint=query_timepoint,
        target_timepoints=target_timepoints,
        label_key=label_key,
        condition_key=condition_key,
    )

    if virtual_fractions.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Build morphogen matrix for virtual points
    virtual_morphogens = build_virtual_morphogen_matrix(
        virtual_fractions, real_morphogen_csv,
    )

    # Extract just the cell type fraction columns from virtual_fractions
    meta_cols = ["original_condition", "target_day", "n_source_cells"]
    ct_cols = [c for c in virtual_fractions.columns if c not in meta_cols]
    virtual_Y = virtual_fractions[ct_cols]

    # Align indices
    common = virtual_Y.index.intersection(virtual_morphogens.index)
    virtual_Y = virtual_Y.loc[common]
    virtual_morphogens = virtual_morphogens.loc[common]

    logger.info("Virtual training data summary:")
    logger.info("X (morphogens): %s", virtual_morphogens.shape)
    logger.info("Y (fractions):  %s", virtual_Y.shape)
    logger.info("Fidelity level: %s", FIDELITY_LEVEL)

    return virtual_morphogens, virtual_Y


def validate_transport_quality(
    problem: object,
    max_cost_threshold: float = 100.0,
) -> pd.DataFrame:
    """Validate transport map quality by checking transport costs.

    High transport costs indicate distribution shift — the model is
    less reliable for those transitions.

    Args:
        problem: Solved moscot TemporalProblem.
        max_cost_threshold: Flag transitions above this cost.

    Returns:
        DataFrame with transport costs per timepoint pair.
    """
    logger.info("Validating transport map quality...")
    results = []

    for key in problem.solutions:
        sol = problem.solutions[key]
        cost = float(sol.cost) if hasattr(sol, "cost") else np.nan
        converged = bool(sol.converged) if hasattr(sol, "converged") else None

        status = "OK"
        if not np.isnan(cost) and cost > max_cost_threshold:
            status = "HIGH_COST"
        if converged is False:
            status = "NOT_CONVERGED"

        results.append({
            "transition": f"{key[0]} -> {key[1]}",
            "cost": cost,
            "converged": converged,
            "status": status,
        })

    report = pd.DataFrame(results)
    for _, row in report.iterrows():
        flag = " !!!" if row["status"] != "OK" else ""
        logger.info("%s: cost=%.4f converged=%s%s",
                    row["transition"], row["cost"], row["converged"], flag)

    return report


def main() -> None:
    """Run the full CellRank 2 virtual data generation pipeline."""
    import time
    start = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 4: CellRank 2 Virtual Data Generation")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Check prerequisites
    # -----------------------------------------------------------------------
    atlas_path = DATA_DIR / "azbukina_temporal_atlas.h5ad"
    query_path = DATA_DIR / "amin_kelley_mapped.h5ad"
    morphogen_path = DATA_DIR / "morphogen_matrix_amin_kelley.csv"

    for path, name in [
        (atlas_path, "Azbukina temporal atlas"),
        (query_path, "Mapped query data (from step 02)"),
        (morphogen_path, "Morphogen matrix (from morphogen_parser)"),
    ]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            logger.error("See data/README.md for download instructions.")
            return

    # -----------------------------------------------------------------------
    # Step 1: Load temporal atlas
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Load and preprocess temporal atlas")
    logger.info("=" * 60)

    atlas = load_temporal_atlas(atlas_path, time_key="day")
    atlas = preprocess_for_moscot(atlas)

    # -----------------------------------------------------------------------
    # Step 2: Compute transport maps
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Compute moscot transport maps")
    logger.info("=" * 60)

    cache_path = DATA_DIR / "cellrank2_transport_maps.pkl"
    problem = compute_transport_maps(
        atlas,
        time_key="day",
        cache_path=cache_path,
    )

    # Validate transport quality
    quality_report = validate_transport_quality(problem)

    # -----------------------------------------------------------------------
    # Step 3: Load query data and project forward
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Project query data forward in time")
    logger.info("=" * 60)

    logger.info("Loading mapped query data...")
    query = sc.read_h5ad(str(query_path))
    logger.info("Query: %s cells, %d conditions",
                f"{query.n_obs:,}", query.obs["condition"].nunique())

    virtual_X, virtual_Y = generate_virtual_training_data(
        query_adata=query,
        atlas_adata=atlas,
        problem=problem,
        real_morphogen_csv=morphogen_path,
        query_timepoint=72,  # Amin/Kelley harvest day
        target_timepoints=[90, 120],
    )

    # -----------------------------------------------------------------------
    # Step 4: Save outputs
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Save virtual training data")
    logger.info("=" * 60)

    if not virtual_X.empty:
        virtual_Y.to_csv(str(DATA_DIR / "cellrank2_virtual_fractions.csv"))
        virtual_X.to_csv(str(DATA_DIR / "cellrank2_virtual_morphogens.csv"))
        # TODO: Wire quality scores into step 04 to filter/weight virtual data
        quality_report.to_csv(
            str(DATA_DIR / "cellrank2_transport_quality.csv"), index=False
        )
        logger.info("Virtual fractions  -> data/cellrank2_virtual_fractions.csv")
        logger.info("Virtual morphogens -> data/cellrank2_virtual_morphogens.csv")
        logger.info("Transport quality  -> data/cellrank2_transport_quality.csv")
    else:
        logger.warning("No virtual data generated.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("CELLRANK 2 VIRTUAL DATA SUMMARY")
    logger.info("=" * 60)
    logger.info("Virtual data points: %d", len(virtual_X))
    logger.info("Fidelity level:      %s", FIDELITY_LEVEL)
    logger.info("Data amplification:  %d virtual from %d real conditions",
                len(virtual_X), query.obs["condition"].nunique())
    logger.info("Time elapsed:        %.1fs", elapsed)


if __name__ == "__main__":
    main()
