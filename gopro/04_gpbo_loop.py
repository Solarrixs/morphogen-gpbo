"""
Step 4: GP-BO Active Reinforcement Learning Loop.

This is the core optimization engine:
  1. Loads GP training data (X=morphogen conditions, Y=cell type fractions)
  2. Applies ILR transform to compositional Y data
  3. Fits a multi-fidelity GP using BoTorch with Matérn 5/2 + ARD kernel
  4. Runs multi-objective acquisition function to recommend next experiments
  5. Outputs plate map CSV for wet lab execution

Inputs:
  - data/gp_training_labels_amin_kelley.csv (from step 02)
  - data/morphogen_matrix_amin_kelley.csv (from morphogen_parser)
  - data/fidelity_report.csv (from step 03, optional)

Outputs:
  - data/gp_recommendations_round{N}.csv (next experiment plate map)
  - data/gp_model_diagnostics.csv (kernel parameters, lengthscales)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from gopro.config import (
    DATA_DIR,
    MORPHOGEN_COLUMNS,
    PROTEIN_MW_KDA,
    nM_to_uM,
    ng_mL_to_uM,
    get_logger,
)

logger = get_logger(__name__)

# BoTorch requires float64, which MPS doesn't support. Use CPU for GP fitting.
# MPS can be used for neural network components (CellFlow, scPoli) separately.
DEVICE = torch.device("cpu")
DTYPE = torch.double

# Literature-based morphogen bounds (all in µM).
# These serve as fallback maxima; actual bounds used during optimization
# are computed dynamically from training data (see _compute_active_bounds).
MORPHOGEN_BOUNDS_LITERATURE = {
    "CHIR99021_uM":     (0.0, 12.0),
    "BMP4_uM":          (0.0, ng_mL_to_uM(50.0, PROTEIN_MW_KDA["BMP4"])),
    "BMP7_uM":          (0.0, ng_mL_to_uM(50.0, PROTEIN_MW_KDA["BMP7"])),
    "SHH_uM":           (0.0, ng_mL_to_uM(500.0, PROTEIN_MW_KDA["SHH"])),
    "SAG_uM":           (0.0, nM_to_uM(2000.0)),
    "RA_uM":            (0.0, nM_to_uM(1000.0)),
    "FGF8_uM":          (0.0, ng_mL_to_uM(200.0, PROTEIN_MW_KDA["FGF8"])),
    "FGF2_uM":          (0.0, ng_mL_to_uM(100.0, PROTEIN_MW_KDA["FGF2"])),
    "FGF4_uM":          (0.0, ng_mL_to_uM(200.0, PROTEIN_MW_KDA["FGF4"])),
    "IWP2_uM":          (0.0, 10.0),
    "XAV939_uM":        (0.0, 10.0),
    "SB431542_uM":      (0.0, 20.0),
    "LDN193189_uM":     (0.0, nM_to_uM(500.0)),
    "DAPT_uM":          (0.0, 10.0),
    "EGF_uM":           (0.0, ng_mL_to_uM(50.0, PROTEIN_MW_KDA["EGF"])),
    "ActivinA_uM":      (0.0, ng_mL_to_uM(100.0, PROTEIN_MW_KDA["ActivinA"])),
    "Dorsomorphin_uM":  (0.0, 5.0),
    "purmorphamine_uM": (0.0, 2.0),
    "cyclopamine_uM":   (0.0, 10.0),
    "log_harvest_day":  (np.log(7), np.log(120)),  # Day 7 to Day 120
    "BDNF_uM":          (0.0, ng_mL_to_uM(40.0, PROTEIN_MW_KDA["BDNF"])),
    "NT3_uM":           (0.0, ng_mL_to_uM(40.0, PROTEIN_MW_KDA["NT3"])),
    "cAMP_uM":          (0.0, 100.0),             # dibutyryl-cAMP 0-100 µM
    "AscorbicAcid_uM":  (0.0, 400.0),             # ascorbic acid 2-phosphate 0-400 µM
}

# Keep legacy name for backwards compatibility
MORPHOGEN_BOUNDS = MORPHOGEN_BOUNDS_LITERATURE


def _compute_active_bounds(
    X: pd.DataFrame,
    columns: list[str],
    padding: float = 0.05,
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Compute bounds from training data, dropping zero-variance columns.

    Args:
        X: Training morphogen matrix (without fidelity column).
        columns: Morphogen column names.
        padding: Fraction of training range to add as padding (default 5%).

    Returns:
        Tuple of (bounds_dict, active_columns) where zero-variance columns
        have been removed and bounds are training_max * (1 + padding).
    """
    morph_cols = [c for c in columns if c != "fidelity"]
    X_morph = X[morph_cols] if isinstance(X, pd.DataFrame) else X

    active_cols = []
    active_bounds = {}

    for col in morph_cols:
        if isinstance(X_morph, pd.DataFrame):
            col_data = X_morph[col]
        else:
            idx = morph_cols.index(col)
            col_data = X_morph[:, idx]

        col_min = float(np.min(col_data))
        col_max = float(np.max(col_data))

        # Skip zero-variance columns (no information for GP)
        if col_max == col_min:
            if col == "log_harvest_day":
                # Keep time dimension even if constant (single time point)
                active_cols.append(col)
                active_bounds[col] = (np.log(7), np.log(120))
            else:
                logger.info("Dropping zero-variance column: %s (value=%.6f)", col, col_max)
            continue

        # Bounds = training range + padding, clamped to literature max
        padded_upper = col_max * (1.0 + padding)
        lit_lo, lit_hi = MORPHOGEN_BOUNDS_LITERATURE.get(col, (0.0, padded_upper))
        upper = min(padded_upper, lit_hi)
        lower = max(0.0, col_min)  # morphogen concentrations are non-negative

        # Guarantee nonzero bound width for near-zero columns
        MIN_BOUND_WIDTH = 1e-6  # µM — minimum meaningful concentration range
        if upper - lower < MIN_BOUND_WIDTH:
            upper = lower + MIN_BOUND_WIDTH

        active_cols.append(col)
        active_bounds[col] = (lower, upper)
        logger.info("Bounds for %s: [%.6f, %.6f] (train: [%.6f, %.6f])",
                     col, lower, upper, col_min, col_max)

    # Always include fidelity if present
    if "fidelity" in columns:
        active_cols.append("fidelity")
        active_bounds["fidelity"] = (1.0, 1.0)

    logger.info("Active dimensions: %d / %d (dropped %d zero-variance)",
                len([c for c in active_cols if c != "fidelity"]),
                len(morph_cols),
                len(morph_cols) - len([c for c in active_cols if c != "fidelity"]))

    return active_bounds, active_cols


def _helmert_basis(D: int) -> np.ndarray:
    """Construct the Helmert ILR basis matrix.

    Args:
        D: Number of composition parts.

    Returns:
        Array of shape (D, D-1) — the ILR contrast matrix.
    """
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        V[:j + 1, j] = -1.0 / (j + 1)
        V[j + 1, j] = 1.0
        V[:, j] *= np.sqrt((j + 1) / (j + 2))
    return V


def ilr_transform(Y: np.ndarray) -> np.ndarray:
    """Isometric log-ratio transform for compositional data.

    Transforms D-part compositions to (D-1)-dimensional real space,
    removing the sum-to-one constraint that violates GP assumptions.

    Args:
        Y: Array of shape (N, D) with rows summing to 1.

    Returns:
        Array of shape (N, D-1) in ILR space.
    """
    D = Y.shape[1]
    Y_safe = Y + 1e-10
    Y_safe = Y_safe / Y_safe.sum(axis=1, keepdims=True)
    log_Y = np.log(Y_safe)

    V = _helmert_basis(D)
    return log_Y @ V


def ilr_inverse(Z: np.ndarray, D: int) -> np.ndarray:
    """Inverse ILR transform: (D-1) real coordinates → D-part composition.

    Args:
        Z: Array of shape (N, D-1) in ILR space.
        D: Number of composition parts.

    Returns:
        Array of shape (N, D) with rows summing to 1.
    """
    V = _helmert_basis(D)
    log_x = Z @ V.T

    # Closure (softmax) to get compositions
    log_x -= log_x.max(axis=1, keepdims=True)
    x = np.exp(log_x)
    return x / x.sum(axis=1, keepdims=True)


def build_training_set(
    fractions_csv: Path,
    morphogen_csv: Path,
    fidelity: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build GP training matrices from cell type fractions + morphogen concentrations.

    Args:
        fractions_csv: Path to cell type fractions CSV (conditions × cell types).
        morphogen_csv: Path to morphogen concentration matrix CSV.
        fidelity: Fidelity level (1.0=real, 0.5=CellRank2, 0.0=CellFlow).

    Returns:
        Tuple of (X, Y) DataFrames.
        X: (N_conditions, D_morphogens + 1_time + 1_fidelity)
        Y: (N_conditions, M_cell_types)
    """
    Y = pd.read_csv(str(fractions_csv), index_col=0)
    X = pd.read_csv(str(morphogen_csv), index_col=0)

    # Align indices
    common = Y.index.intersection(X.index)
    if len(common) < len(Y):
        logger.warning("%d conditions in Y not found in X", len(Y) - len(common))
    X = X.loc[common]
    Y = Y.loc[common]

    # Add fidelity column
    X["fidelity"] = fidelity

    logger.info("X (morphogens): %s", X.shape)
    logger.info("Y (cell type fractions): %s", Y.shape)

    return X, Y


def _extract_lengthscales(model, n_input_dims: int):
    """Extract lengthscales from a fitted GP model.

    Handles SingleTaskGP (MAP), SaasFullyBayesianSingleTaskGP,
    and ModelListGP containing SAAS models.

    Returns:
        1-D numpy array of lengthscales, or None.
    """
    try:
        # ModelListGP of SAAS models — median across sub-models
        if hasattr(model, 'models'):
            all_ls = []
            for sub_model in model.models:
                if hasattr(sub_model, 'median_lengthscale'):
                    ls = sub_model.median_lengthscale.detach().cpu().numpy()
                    all_ls.append(ls)
            if all_ls:
                return np.median(np.stack(all_ls), axis=0)

        # Single SAAS model
        if hasattr(model, 'median_lengthscale'):
            return model.median_lengthscale.detach().cpu().numpy().flatten()

        # Standard GP (MAP)
        if hasattr(model, 'covar_module'):
            if hasattr(model.covar_module, 'base_kernel') and model.covar_module.base_kernel is not None:
                ls = model.covar_module.base_kernel.lengthscale
                if ls is not None:
                    return ls.detach().cpu().numpy().flatten()
            elif hasattr(model.covar_module, 'lengthscale'):
                ls = model.covar_module.lengthscale
                if ls is not None:
                    return ls.detach().cpu().numpy().flatten()
    except (AttributeError, RuntimeError):
        pass
    return None


def fit_gp_botorch(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_cell_types: Optional[list[str]] = None,
    use_ilr: bool = True,
    use_saasbo: bool = False,
    saasbo_warmup: int = 256,
    saasbo_num_samples: int = 128,
    saasbo_thinning: int = 16,
) -> tuple:
    """Fit a GP using BoTorch (MAP, fully Bayesian SAASBO, or multi-fidelity).

    Args:
        X: Morphogen concentration matrix with fidelity column.
        Y: Cell type fraction matrix.
        target_cell_types: List of cell types to optimize. If None, uses all.
        use_ilr: Whether to apply ILR transform to Y.
        use_saasbo: Use SAASBO (fully Bayesian GP with sparsity prior).
            Ignored when multi-fidelity data is detected.
        saasbo_warmup: NUTS warmup steps for SAASBO.
        saasbo_num_samples: NUTS posterior samples for SAASBO.
        saasbo_thinning: NUTS thinning factor for SAASBO.

    Returns:
        Tuple of (model, train_X_tensor, train_Y_tensor, cell_type_columns).
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    # Select target cell types
    if target_cell_types:
        Y_selected = Y[target_cell_types]
    else:
        Y_selected = Y
    cell_type_cols = list(Y_selected.columns)

    # Prepare tensors
    train_X = torch.tensor(X.values, dtype=DTYPE, device=DEVICE)
    Y_values = Y_selected.values

    if use_ilr and Y_values.shape[1] > 1:
        logger.info("Applying ILR transform to compositional Y data...")
        Y_values = ilr_transform(Y_values)
        logger.info("ILR-transformed Y shape: %s", Y_values.shape)

    train_Y = torch.tensor(Y_values, dtype=DTYPE, device=DEVICE)

    # Fit GP
    logger.info("Fitting BoTorch GP...")
    logger.info("X: %s, Y: %s", train_X.shape, train_Y.shape)

    # Check if we have fidelity column
    has_fidelity = "fidelity" in X.columns
    fidelity_idx = list(X.columns).index("fidelity") if has_fidelity else None

    if has_fidelity and X["fidelity"].nunique() > 1:
        from botorch.models import SingleTaskMultiFidelityGP
        logger.info("Using multi-fidelity GP (fidelity column detected)")
        if use_saasbo:
            logger.info("SAASBO ignored — multi-fidelity takes priority")
        model = SingleTaskMultiFidelityGP(
            train_X,
            train_Y,
            data_fidelities=[fidelity_idx],
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    elif use_saasbo:
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        from botorch.fit import fit_fully_bayesian_model_nuts

        if train_Y.shape[1] == 1:
            logger.info("Using SAASBO (single output, fully Bayesian)")
            model = SaasFullyBayesianSingleTaskGP(
                train_X, train_Y,
                input_transform=Normalize(d=train_X.shape[1]),
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=saasbo_warmup,
                num_samples=saasbo_num_samples,
                thinning=saasbo_thinning,
            )
        else:
            from botorch.models import ModelListGP
            n_outputs = train_Y.shape[1]
            logger.info("Using SAASBO with ModelListGP (%d outputs)", n_outputs)
            saas_models = []
            for i in range(n_outputs):
                logger.info("  Fitting SAAS model %d/%d...", i + 1, n_outputs)
                m = SaasFullyBayesianSingleTaskGP(
                    train_X,
                    train_Y[:, i:i+1],
                    input_transform=Normalize(d=train_X.shape[1]),
                    outcome_transform=Standardize(m=1),
                )
                fit_fully_bayesian_model_nuts(
                    m,
                    warmup_steps=saasbo_warmup,
                    num_samples=saasbo_num_samples,
                    thinning=saasbo_thinning,
                    disable_progbar=True,
                )
                saas_models.append(m)
            model = ModelListGP(*saas_models)
    else:
        logger.info("Using single-task GP (single fidelity level)")
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    # Report kernel parameters
    logger.info("GP Kernel Parameters:")
    lengthscales = _extract_lengthscales(model, train_X.shape[1])
    if lengthscales is not None:
        importance = 1.0 / lengthscales
        morph_importance = pd.Series(importance[:len(X.columns)], index=X.columns)
        morph_importance = morph_importance.sort_values(ascending=False)
        logger.info("Morphogen importance (1/lengthscale):")
        for morph, imp in morph_importance.head(8).items():
            logger.info("  %s: %.4f", morph, imp)
    else:
        logger.info("(lengthscales not directly accessible for this model type)")

    return model, train_X, train_Y, cell_type_cols


def recommend_next_experiments(
    model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: dict[str, tuple[float, float]],
    columns: list[str],
    n_recommendations: int = 24,
    use_multi_objective: bool = False,
    ref_point: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Use acquisition function to recommend next experiments.

    Args:
        model: Fitted BoTorch GP model.
        train_X: Training X tensor.
        train_Y: Training Y tensor.
        bounds: Dict mapping column names to (lower, upper) bounds.
        columns: Column names for X.
        n_recommendations: Number of conditions to recommend.
        use_multi_objective: Whether to use multi-objective optimization.
        ref_point: Reference point for hypervolume calculation.

    Returns:
        DataFrame with recommended morphogen concentrations and predictions.
    """
    from botorch.optim import optimize_acqf
    from botorch.acquisition import qLogExpectedImprovement

    # Build bounds tensor
    lower = []
    upper = []
    for col in columns:
        if col in bounds:
            lo, hi = bounds[col]
        elif col == "fidelity":
            lo, hi = 1.0, 1.0  # always recommend at highest fidelity
        else:
            lo, hi = 0.0, 1.0
        lower.append(lo)
        upper.append(hi)

    bounds_tensor = torch.tensor(
        [lower, upper], dtype=DTYPE, device=DEVICE
    )

    if use_multi_objective and train_Y.shape[1] > 1:
        from botorch.acquisition.multi_objective import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        if ref_point is None:
            ref_point = [0.0] * train_Y.shape[1]
        ref_point_tensor = torch.tensor(ref_point, dtype=DTYPE, device=DEVICE)

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_tensor,
            X_baseline=train_X,
        )
    else:
        # For multi-output models, use a scalarization posterior transform
        if train_Y.shape[1] > 1:
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.utils.transforms import normalize, standardize

            # Default: maximize mean of all outputs (equal weight scalarization)
            weights = torch.ones(train_Y.shape[1], dtype=DTYPE, device=DEVICE)
            weights = weights / weights.sum()

            objective = GenericMCObjective(
                lambda samples, X=None: (samples * weights).sum(dim=-1)
            )
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=(train_Y @ weights.unsqueeze(1)).max(),
                objective=objective,
            )
        else:
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=train_Y.max(),
            )

    # Optimize acquisition function
    logger.info("Optimizing acquisition function for %d candidates...", n_recommendations)
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=n_recommendations,
        num_restarts=5,
        raw_samples=512,
    )

    # Get predictions at recommended points
    with torch.no_grad():
        posterior = model.posterior(candidates)
        pred_mean = posterior.mean.cpu().numpy()
        pred_var = posterior.variance.cpu().numpy()

    # Build output DataFrame
    recommendations = pd.DataFrame(
        candidates.cpu().numpy(),
        columns=columns,
    )

    # Add predictions
    for i in range(pred_mean.shape[1]):
        recommendations[f"predicted_y{i}_mean"] = pred_mean[:, i]
        recommendations[f"predicted_y{i}_std"] = np.sqrt(pred_var[:, i])

    recommendations["acquisition_value"] = acq_values.detach().cpu().numpy() if acq_values.dim() > 0 else acq_values.item()

    # Add well labels (A1-D6 for 24-well plate)
    wells = [f"{chr(65 + i // 6)}{i % 6 + 1}" for i in range(n_recommendations)]
    recommendations.index = wells[:len(recommendations)]
    recommendations.index.name = "well"

    return recommendations


def merge_multi_fidelity_data(
    data_sources: list[tuple[Path, Path, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge training data from multiple fidelity sources.

    Combines real, CellRank 2 virtual, and CellFlow virtual data into
    a single multi-fidelity training set.

    Args:
        data_sources: List of (fractions_csv, morphogen_csv, fidelity) tuples.
            Each tuple provides a data source at a given fidelity level.

    Returns:
        Tuple of merged (X, Y) DataFrames with consistent column alignment.
    """
    all_X = []
    all_Y = []

    for fractions_csv, morphogen_csv, fidelity in data_sources:
        if not fractions_csv.exists() or not morphogen_csv.exists():
            logger.info("Skipping %s (not found)", fractions_csv.name)
            continue

        X, Y = build_training_set(fractions_csv, morphogen_csv, fidelity=fidelity)
        all_X.append(X)
        all_Y.append(Y)
        logger.info("Loaded %d points at fidelity=%s", len(X), fidelity)

    if not all_X:
        raise ValueError("No valid data sources found")

    # Align Y columns (different sources may have different cell types)
    all_ct_cols = set()
    for Y in all_Y:
        all_ct_cols.update(Y.columns)
    all_ct_cols = sorted(all_ct_cols)

    aligned_Y = []
    for Y in all_Y:
        Y = Y.copy()
        for col in all_ct_cols:
            if col not in Y.columns:
                Y[col] = 0.0
        aligned_Y.append(Y[all_ct_cols])

    # Align X columns
    all_morph_cols = set()
    for X in all_X:
        all_morph_cols.update(X.columns)
    all_morph_cols = sorted(all_morph_cols)

    aligned_X = []
    for X in all_X:
        X = X.copy()
        for col in all_morph_cols:
            if col not in X.columns:
                X[col] = 0.0
        aligned_X.append(X[all_morph_cols])

    merged_X = pd.concat(aligned_X, axis=0)
    merged_Y = pd.concat(aligned_Y, axis=0)

    # Re-normalize Y rows to sum to 1 (after column alignment)
    row_sums = merged_Y.sum(axis=1)
    merged_Y = merged_Y.div(row_sums.replace(0, 1), axis=0)

    logger.info("Merged training set:")
    logger.info("  X: %s", merged_X.shape)
    logger.info("  Y: %s", merged_Y.shape)

    fidelity_counts = merged_X["fidelity"].value_counts().sort_index(ascending=False)
    for fid, count in fidelity_counts.items():
        label = {1.0: "real", 0.5: "CellRank2", 0.0: "CellFlow"}.get(fid, f"fid={fid}")
        logger.info("  %s: %d points", label, count)

    return merged_X, merged_Y


def run_gpbo_loop(
    fractions_csv: Path,
    morphogen_csv: Path,
    target_cell_types: Optional[list[str]] = None,
    n_recommendations: int = 24,
    round_num: int = 1,
    use_ilr: bool = True,
    virtual_sources: Optional[list[tuple[Path, Path, float]]] = None,
    use_saasbo: bool = False,
) -> pd.DataFrame:
    """Run one iteration of the GP-BO loop.

    Args:
        fractions_csv: Path to cell type fractions CSV.
        morphogen_csv: Path to morphogen matrix CSV.
        target_cell_types: Cell types to optimize for. None = all.
        n_recommendations: Number of experiments to recommend.
        round_num: Current optimization round number.
        use_ilr: Whether to apply ILR transform.
        virtual_sources: Optional list of (fractions_csv, morphogen_csv, fidelity)
            tuples for multi-fidelity data. Real data is always included at
            fidelity=1.0. Virtual sources add CellRank2 (0.5) or CellFlow (0.0).
        use_saasbo: Use SAASBO (fully Bayesian GP with sparsity prior).

    Returns:
        DataFrame of recommended next experiments.
    """
    logger.info("--- GP-BO ROUND %d ---", round_num)

    # Build training set (potentially multi-fidelity)
    logger.info("Building training set...")

    # Always load real data first (used for bounds and as base training set)
    X_real, Y_real = build_training_set(fractions_csv, morphogen_csv)

    if virtual_sources:
        # Multi-fidelity: combine real + virtual data
        all_sources = [(fractions_csv, morphogen_csv, 1.0)] + virtual_sources
        X, Y = merge_multi_fidelity_data(all_sources)
    else:
        X, Y = X_real, Y_real

    # Compute active bounds from REAL data only (not virtual), to prevent
    # GP from exploring far outside the experimentally tested range
    active_bounds, active_cols = _compute_active_bounds(X_real, list(X.columns))

    # Filter X to active columns only
    X_active = X[active_cols]
    logger.info("Active X shape: %s (from %s)", X_active.shape, X.shape)

    # Fit GP on active dimensions only
    model, train_X, train_Y, cell_type_cols = fit_gp_botorch(
        X_active, Y,
        target_cell_types=target_cell_types,
        use_ilr=use_ilr,
        use_saasbo=use_saasbo,
    )

    # Recommend next experiments (in active dimensions)
    recommendations = recommend_next_experiments(
        model, train_X, train_Y,
        bounds=active_bounds,
        columns=list(X_active.columns),
        n_recommendations=n_recommendations,
    )

    # Restore dropped zero-variance columns as zeros
    all_morph_cols = [c for c in X.columns if c != "fidelity"]
    for col in all_morph_cols:
        if col not in recommendations.columns:
            # Use the constant value from training data (usually 0.0)
            recommendations[col] = float(X[col].iloc[0]) if col in X.columns else 0.0
    # Reorder columns to match original ordering
    final_morph_cols = [c for c in X.columns if c in recommendations.columns]
    pred_cols = [c for c in recommendations.columns if c not in X.columns]
    recommendations = recommendations[final_morph_cols + pred_cols]

    # Save outputs
    output_path = DATA_DIR / f"gp_recommendations_round{round_num}.csv"
    recommendations.to_csv(str(output_path))
    logger.info("Plate map saved to %s", output_path)

    # Save model diagnostics
    diagnostics = {
        "round": round_num,
        "n_training_points": len(X),
        "n_morphogens": X.shape[1],
        "n_cell_types": Y.shape[1],
        "target_cell_types": str(target_cell_types or "all"),
    }

    if virtual_sources:
        diagnostics["n_real_points"] = int((X["fidelity"] == 1.0).sum())
        diagnostics["n_virtual_points"] = int((X["fidelity"] < 1.0).sum())
        diagnostics["fidelity_levels"] = str(sorted(X["fidelity"].unique().tolist()))

    try:
        ls = _extract_lengthscales(model, train_X.shape[1])
        if ls is not None:
            for i, col in enumerate(X_active.columns):
                if i < len(ls):
                    diagnostics[f"lengthscale_{col}"] = float(ls[i])
        diagnostics["model_type"] = "saasbo" if use_saasbo else "map"
    except (AttributeError, RuntimeError):
        pass

    diag_df = pd.DataFrame([diagnostics])
    diag_path = DATA_DIR / f"gp_diagnostics_round{round_num}.csv"
    diag_df.to_csv(str(diag_path), index=False)
    logger.info("Diagnostics saved to %s", diag_path)

    return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GP-BO loop for morphogen optimization")
    parser.add_argument("--fractions", type=str, default=None,
                        help="Path to cell type fractions CSV")
    parser.add_argument("--morphogens", type=str, default=None,
                        help="Path to morphogen concentration matrix CSV")
    parser.add_argument("--target-cell-types", nargs="+", default=None,
                        help="Cell types to optimize for (default: all)")
    parser.add_argument("--n-recommendations", type=int, default=24,
                        help="Number of experiments to recommend (default: 24)")
    parser.add_argument("--round", type=int, default=1,
                        help="Optimization round number (default: 1)")
    parser.add_argument("--no-ilr", action="store_true",
                        help="Disable ILR transform")
    parser.add_argument("--multi-objective", action="store_true",
                        help="Use multi-objective acquisition (qLogNEHVI)")
    parser.add_argument("--cellrank2-fractions", type=str, default=None,
                        help="Path to CellRank2 virtual fractions CSV (fidelity=0.5)")
    parser.add_argument("--cellrank2-morphogens", type=str, default=None,
                        help="Path to CellRank2 virtual morphogens CSV")
    parser.add_argument("--cellflow-fractions", type=str, default=None,
                        help="Path to CellFlow virtual fractions CSV (fidelity=0.0)")
    parser.add_argument("--cellflow-morphogens", type=str, default=None,
                        help="Path to CellFlow virtual morphogens CSV")
    parser.add_argument("--sag-fractions", type=str, default=None,
                        help="Path to SAG screen fractions CSV (fidelity=1.0)")
    parser.add_argument("--sag-morphogens", type=str, default=None,
                        help="Path to SAG screen morphogens CSV")
    parser.add_argument("--saasbo", action="store_true",
                        help="Use SAASBO (fully Bayesian GP with sparsity prior)")
    args = parser.parse_args()

    fractions_path = Path(args.fractions) if args.fractions else DATA_DIR / "gp_training_labels_amin_kelley.csv"
    morphogen_path = Path(args.morphogens) if args.morphogens else DATA_DIR / "morphogen_matrix_amin_kelley.csv"

    if not fractions_path.exists() or not morphogen_path.exists():
        logger.info("Training data not found. Running with synthetic data for demo...")

        np.random.seed(42)
        n_conditions = 46
        n_cell_types = 8

        Y_demo = pd.DataFrame(
            np.random.dirichlet(np.ones(n_cell_types), size=n_conditions),
            columns=[f"celltype_{i}" for i in range(n_cell_types)],
            index=[f"cond_{i}" for i in range(n_conditions)],
        )

        X_demo = pd.DataFrame(
            np.zeros((n_conditions, len(MORPHOGEN_COLUMNS))),
            columns=MORPHOGEN_COLUMNS,
            index=Y_demo.index,
        )
        for col in MORPHOGEN_COLUMNS:
            lo, hi = MORPHOGEN_BOUNDS.get(col, (0, 1))
            X_demo[col] = np.random.uniform(lo, hi, size=n_conditions)

        fractions_path = DATA_DIR / "gp_training_labels_demo.csv"
        morphogen_path = DATA_DIR / "morphogen_matrix_demo.csv"
        DATA_DIR.mkdir(exist_ok=True)
        Y_demo.to_csv(str(fractions_path))
        X_demo.to_csv(str(morphogen_path))
        logger.info("Synthetic data saved.")

    # Build virtual sources list
    virtual_sources = []

    # Check for CellRank2 virtual data
    cr2_frac = Path(args.cellrank2_fractions) if args.cellrank2_fractions else DATA_DIR / "cellrank2_virtual_fractions.csv"
    cr2_morph = Path(args.cellrank2_morphogens) if args.cellrank2_morphogens else DATA_DIR / "cellrank2_virtual_morphogens.csv"
    if cr2_frac.exists() and cr2_morph.exists():
        virtual_sources.append((cr2_frac, cr2_morph, 0.5))
        logger.info("Including CellRank2 virtual data (fidelity=0.5)")

    # Check for CellFlow virtual data
    cf_frac = Path(args.cellflow_fractions) if args.cellflow_fractions else DATA_DIR / "cellflow_virtual_fractions_200.csv"
    cf_morph = Path(args.cellflow_morphogens) if args.cellflow_morphogens else DATA_DIR / "cellflow_virtual_morphogens_200.csv"
    if cf_frac.exists() and cf_morph.exists():
        virtual_sources.append((cf_frac, cf_morph, 0.0))
        logger.info("Including CellFlow virtual data (fidelity=0.0)")

    # Check for SAG secondary screen real data
    sag_frac = Path(args.sag_fractions) if args.sag_fractions else DATA_DIR / "gp_training_labels_sag_screen.csv"
    sag_morph = Path(args.sag_morphogens) if args.sag_morphogens else DATA_DIR / "morphogen_matrix_sag_screen.csv"
    if sag_frac.exists() and sag_morph.exists():
        virtual_sources.append((sag_frac, sag_morph, 1.0))
        logger.info("Including SAG secondary screen real data (fidelity=1.0)")

    # Run GP-BO loop
    recs = run_gpbo_loop(
        fractions_csv=fractions_path,
        morphogen_csv=morphogen_path,
        target_cell_types=args.target_cell_types,
        n_recommendations=args.n_recommendations,
        round_num=args.round,
        use_ilr=not args.no_ilr,
        virtual_sources=virtual_sources if virtual_sources else None,
        use_saasbo=args.saasbo,
    )

    logger.info("--- NEXT EXPERIMENT RECOMMENDATIONS ---")
    morph_cols = [c for c in MORPHOGEN_COLUMNS if c in recs.columns]
    nonzero_cols = [c for c in morph_cols if recs[c].abs().sum() > 0.01]
    pred_cols = [c for c in recs.columns if "predicted" in c or "acquisition" in c]
    logger.info("\n%s", recs[nonzero_cols + pred_cols].to_string())
    logger.info("Give this plate map to the wet lab team!")
