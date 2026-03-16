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

import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from gopro.config import (
    DATA_DIR,
    FIDELITY_CORRELATION_THRESHOLD,
    FIDELITY_COSTS,
    GP_STATE_DIR,
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

        # Skip zero-variance columns (no information for GP).
        # This includes log_harvest_day when all data is at a single timepoint —
        # BoTorch's Normalize divides by (upper - lower), so equal bounds → NaN.
        # The column is restored with its constant value after optimization (line ~690).
        if col_max == col_min:
            logger.info("Dropping zero-variance column: %s (value=%.6f)", col, col_max)
            continue

        # Bounds = training range + padding, clamped to literature max
        padded_upper = col_max + padding * (col_max - col_min)
        lit_lo, lit_hi = MORPHOGEN_BOUNDS_LITERATURE.get(col, (0.0, padded_upper))
        upper = min(padded_upper, lit_hi)
        lower = 0.0 if col != "log_harvest_day" else col_min

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


def _set_dim_scaled_lengthscale_prior(model, d: int) -> None:
    """Set a dimensionality-scaled log-normal prior on kernel lengthscales.

    Following Hvarfner et al. (ICML 2024, arXiv:2402.02229), a log-normal
    prior with mode scaled by sqrt(d) prevents vanishing gradients during
    MLE in high-dimensional BO.  This improves exploration in our ~8D
    active morphogen space.

    Args:
        model: BoTorch SingleTaskGP or SingleTaskMultiFidelityGP.
        d: Number of input dimensions.
    """
    from gpytorch.priors import LogNormalPrior

    # Mode of LogNormal(loc, scale) = exp(loc - scale^2).
    # We want mode ~ sqrt(d), so loc = log(sqrt(d)) + scale^2.
    scale = 1.0
    loc = math.log(math.sqrt(d)) + scale ** 2
    prior = LogNormalPrior(loc=loc, scale=scale)

    if hasattr(model, "covar_module"):
        kernel = model.covar_module
        base = getattr(kernel, "base_kernel", kernel)
        if hasattr(base, "lengthscale"):
            try:
                base.register_prior(
                    "lengthscale_prior", prior, lambda m: m.lengthscale,
                    lambda m, v: m._set_lengthscale(v),
                )
                logger.info(
                    "Set dim-scaled LogNormal lengthscale prior (d=%d, loc=%.2f)", d, loc
                )
            except Exception as exc:
                logger.warning(
                    "Could not set lengthscale prior (d=%d): %s", d, exc
                )


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


def _multiplicative_replacement(Y: np.ndarray, delta: float | None = None) -> np.ndarray:
    """Replace zeros in compositional data using multiplicative replacement.

    Following Martin-Fernandez et al. (Statistical Modelling 2015), zeros
    are replaced with a small value *delta* and non-zero entries are scaled
    down proportionally so each row still sums to 1.  This avoids the
    extreme log-ratios produced by naive additive pseudo-counts (e.g. 1e-10).

    Args:
        Y: Array of shape (N, D) with rows summing to ~1. May contain zeros.
        delta: Replacement value for zeros.  If None, defaults to
            ``0.65 / (D * 100)`` which gives ~3.8e-4 for 17 cell types.

    Returns:
        Array of shape (N, D) with no zeros, rows summing to 1.
    """
    D = Y.shape[1]
    if delta is None:
        delta = 0.65 / (D * 100)

    Y_out = Y.copy()
    for i in range(Y_out.shape[0]):
        row = Y_out[i]
        zeros = row <= 0
        n_zeros = int(zeros.sum())
        if n_zeros == 0:
            continue
        if n_zeros == D:
            # All zeros: fall back to uniform
            Y_out[i] = 1.0 / D
            continue
        # Multiplicative replacement: zeros get delta, non-zeros are
        # scaled down so the row still sums to 1.
        total_replacement = n_zeros * delta
        nonzero_sum = row[~zeros].sum()
        row[zeros] = delta
        row[~zeros] *= (1.0 - total_replacement) / nonzero_sum

    return Y_out


def ilr_transform(Y: np.ndarray) -> np.ndarray:
    """Isometric log-ratio transform for compositional data.

    Transforms D-part compositions to (D-1)-dimensional real space,
    removing the sum-to-one constraint that violates GP assumptions.

    Uses multiplicative replacement for zeros (Martin-Fernandez et al.
    2015) instead of a naive additive pseudo-count.

    Args:
        Y: Array of shape (N, D) with rows summing to 1.

    Returns:
        Array of shape (N, D-1) in ILR space.
    """
    D = Y.shape[1]
    Y_safe = _multiplicative_replacement(Y)
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


def _helmert_basis_torch(D: int) -> torch.Tensor:
    """Construct the Helmert ILR basis matrix as a torch tensor.

    Used inside the differentiable ILR-inverse objective for scalarization.
    """
    V = torch.zeros(D, D - 1, dtype=DTYPE, device=DEVICE)
    for j in range(D - 1):
        V[:j + 1, j] = -1.0 / (j + 1)
        V[j + 1, j] = 1.0
        V[:, j] *= (float((j + 1) / (j + 2))) ** 0.5
    return V


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
    from gopro.validation import validate_training_csvs
    validate_training_csvs(fractions_csv, morphogen_csv)

    Y = pd.read_csv(str(fractions_csv), index_col=0)
    X = pd.read_csv(str(morphogen_csv), index_col=0)

    # Align indices
    common = Y.index.intersection(X.index)
    if len(common) < len(Y):
        logger.warning("%d conditions in Y not found in X", len(Y) - len(common))
    X = X.loc[common]
    Y = Y.loc[common]

    # Add fidelity column (copy to avoid mutating caller's DataFrame)
    X = X.copy()
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


def save_gp_state(model, path: Path) -> None:
    """Save GP hyperparameters for warm-starting next round.

    Saves lengthscales, outputscale, noise, and mean constant for standard
    SingleTaskGP models. For SAASBO (fully Bayesian) models, saves the
    MAP estimate (median of posterior samples) so it can be used as the
    initial position for NUTS in the next round.

    Args:
        model: Fitted BoTorch GP model.
        path: File path to save the state dict (`.pt` format).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict = {}

    # ModelListGP (e.g. SAASBO multi-output): save per-sub-model
    if hasattr(model, "models"):
        sub_states = []
        for sub in model.models:
            sub_states.append(sub.state_dict())
        state["model_list_states"] = sub_states
        state["model_type"] = "model_list"
    else:
        # Standard SingleTaskGP or SingleTaskMultiFidelityGP
        # Save selective hyperparameters (not full state_dict) for resilience
        # to dimension changes across rounds.
        try:
            if hasattr(model, "covar_module"):
                kernel = model.covar_module
                # Handle both ScaleKernel(base_kernel=...) and bare RBFKernel
                base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
                if hasattr(base, "lengthscale") and base.lengthscale is not None:
                    state["lengthscales"] = base.lengthscale.detach().cpu()
                if hasattr(kernel, "outputscale"):
                    state["outputscale"] = kernel.outputscale.detach().cpu()
            if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
                state["noise"] = model.likelihood.noise.detach().cpu()
            if hasattr(model, "mean_module") and hasattr(model.mean_module, "constant"):
                state["mean_constant"] = model.mean_module.constant.detach().cpu()
            state["model_type"] = "single_task"
        except (AttributeError, RuntimeError) as exc:
            logger.warning("Could not extract GP hyperparameters: %s", exc)
            return

    torch.save(state, str(path))
    logger.info("Saved GP state to %s", path)


def load_gp_state(model, path: Path) -> bool:
    """Load saved GP hyperparameters as initialization for fitting.

    Sets initial hyperparameter values but does NOT prevent further
    optimization (MLL fitting will continue to refine them).

    Handles dimension mismatches gracefully: if the saved lengthscales
    have a different number of dimensions than the current model, the
    state is skipped with a warning.

    Args:
        model: BoTorch GP model (pre-fitting).
        path: Path to saved state file (`.pt` format).

    Returns:
        True if warm-start was applied, False otherwise.
    """
    path = Path(path)
    if not path.exists():
        logger.info("No GP state found at %s, using default initialization", path)
        return False

    try:
        state = torch.load(str(path), weights_only=True)
    except Exception as exc:
        logger.warning("Failed to load GP state from %s: %s", path, exc)
        return False

    model_type = state.get("model_type", "single_task")

    # ModelListGP warm-start: apply per-sub-model state
    if model_type == "model_list" and hasattr(model, "models"):
        sub_states = state.get("model_list_states", [])
        if len(sub_states) != len(model.models):
            logger.warning(
                "GP warm-start skipped: saved %d sub-models but current model has %d",
                len(sub_states), len(model.models),
            )
            return False
        try:
            for sub_model, sub_state in zip(model.models, sub_states):
                sub_model.load_state_dict(sub_state, strict=False)
            logger.info("Warm-started ModelListGP from %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to warm-start ModelListGP: %s", exc)
            return False

    # Single-task GP warm-start
    try:
        if "lengthscales" in state and hasattr(model, "covar_module"):
            kernel = model.covar_module
            base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
            if hasattr(base, "lengthscale") and base.lengthscale is not None:
                saved_ls = state["lengthscales"]
                current_ls = base.lengthscale
                if saved_ls.shape != current_ls.shape:
                    logger.warning(
                        "GP warm-start dimension mismatch: saved lengthscales %s "
                        "vs current %s. Skipping warm-start.",
                        list(saved_ls.shape), list(current_ls.shape),
                    )
                    return False
                base.lengthscale = saved_ls
        if "outputscale" in state and hasattr(model, "covar_module") and hasattr(model.covar_module, "outputscale"):
            model.covar_module.outputscale = state["outputscale"]
        if "noise" in state and hasattr(model, "likelihood"):
            model.likelihood.noise = state["noise"]
        if "mean_constant" in state and hasattr(model, "mean_module") and hasattr(model.mean_module, "constant"):
            model.mean_module.constant = state["mean_constant"]
        logger.info("Warm-started GP from %s", path)
        return True
    except Exception as exc:
        logger.warning("Failed to warm-start GP: %s. Using defaults.", exc)
        return False


def fit_tvr_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_cell_types: Optional[list[str]] = None,
    use_ilr: bool = True,
) -> tuple:
    """Fit separate GPs per fidelity level for Targeted Variance Reduction.

    TVR (Fare et al. 2022, McDonald et al. 2025) fits independent GPs for
    each fidelity level and combines them by selecting the model with the
    lowest cost-scaled posterior variance at each candidate point.

    Args:
        X: Morphogen matrix with ``fidelity`` column.
        Y: Cell type fraction matrix.
        target_cell_types: Cell types to optimize for. None = all.
        use_ilr: Whether to apply ILR transform to Y.

    Returns:
        Tuple of (tvr_ensemble, train_X_tensor, train_Y_tensor, cell_type_cols)
        where tvr_ensemble is a TVRModelEnsemble instance.
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    if "fidelity" not in X.columns:
        raise ValueError("TVR requires multi-fidelity data (fidelity column missing)")

    fidelity_levels = sorted(X["fidelity"].unique())
    if len(fidelity_levels) < 2:
        raise ValueError(
            f"TVR requires at least 2 fidelity levels, got {len(fidelity_levels)}"
        )

    # Select target cell types
    if target_cell_types:
        Y_selected = Y[target_cell_types]
    else:
        Y_selected = Y
    cell_type_cols = list(Y_selected.columns)

    # Prepare Y values (ILR transform if applicable)
    Y_values = Y_selected.values
    if use_ilr and Y_values.shape[1] > 1:
        logger.info("Applying ILR transform to compositional Y data...")
        Y_values = ilr_transform(Y_values)

    # Morphogen columns (excluding fidelity)
    morph_cols = [c for c in X.columns if c != "fidelity"]

    # Fit one GP per fidelity level
    per_fidelity_models = {}
    for fid in fidelity_levels:
        mask = X["fidelity"] == fid
        X_fid = X.loc[mask, morph_cols]
        Y_fid = Y_values[mask.values] if isinstance(mask, pd.Series) else Y_values[mask]

        n_points = len(X_fid)
        logger.info("Fitting TVR GP for fidelity=%.1f (%d points, %d dims)",
                     fid, n_points, len(morph_cols))

        X_tensor = torch.tensor(X_fid.values, dtype=DTYPE, device=DEVICE)
        Y_tensor = torch.tensor(Y_fid, dtype=DTYPE, device=DEVICE)

        model = SingleTaskGP(
            X_tensor, Y_tensor,
            input_transform=Normalize(d=X_tensor.shape[1]),
            outcome_transform=Standardize(m=Y_tensor.shape[1]),
        )
        _set_dim_scaled_lengthscale_prior(model, X_tensor.shape[1])
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        per_fidelity_models[fid] = model
        noise_val = model.likelihood.noise.mean().item()
        logger.info("  Fitted GP for fidelity=%.1f (noise=%.4f)", fid, noise_val)

    # Build ensemble
    ensemble = TVRModelEnsemble(
        models=per_fidelity_models,
        cost_ratios={fid: FIDELITY_COSTS.get(fid, 1.0) for fid in fidelity_levels},
    )

    # Use full training data for train_X/train_Y (needed for acquisition function)
    train_X = torch.tensor(X[morph_cols].values, dtype=DTYPE, device=DEVICE)
    train_Y = torch.tensor(Y_values, dtype=DTYPE, device=DEVICE)

    return ensemble, train_X, train_Y, cell_type_cols


class TVRModelEnsemble:
    """Ensemble of per-fidelity GPs for Targeted Variance Reduction.

    At each candidate point, selects the model with the lowest
    cost-scaled posterior variance (variance * cost). Cheaper models
    are preferred when raw variance is similar.

    This class provides a ``posterior()`` method compatible with
    BoTorch's acquisition function interface.
    """

    def __init__(
        self,
        models: dict[float, object],
        cost_ratios: dict[float, float],
    ):
        self.models = models
        self.cost_ratios = cost_ratios
        self._fidelity_levels = sorted(models.keys())

    @property
    def num_outputs(self) -> int:
        """Number of output dimensions."""
        first_model = self.models[self._fidelity_levels[0]]
        return first_model.num_outputs

    def posterior(self, X: torch.Tensor, **kwargs):
        """Return a posterior that combines per-fidelity models via TVR.

        For each candidate point, selects the model with the lowest
        cost-scaled variance (variance * cost). Returns that model's
        posterior mean and variance.
        """
        # Collect posteriors from all models (no torch.no_grad — gradients
        # must flow for BoTorch acquisition function optimization)
        posteriors = {}
        for fid in self._fidelity_levels:
            posteriors[fid] = self.models[fid].posterior(X)

        # For each point, pick the model with lowest cost-scaled variance
        # Shape: X is (..., d), posterior mean is (..., m)
        batch_shape = X.shape[:-1]
        n_outputs = self.num_outputs

        all_means = []
        all_raw_vars = []
        all_scaled_vars = []

        for fid in self._fidelity_levels:
            post = posteriors[fid]
            mean = post.mean  # (..., m)
            var = post.variance  # (..., m)

            # Scale variance by cost (cheaper models → lower scaled variance)
            cost = self.cost_ratios.get(fid, 1.0)
            scaled_var = var * cost

            all_means.append(mean)
            all_raw_vars.append(var)
            all_scaled_vars.append(scaled_var)

        # Stack: (n_fidelities, ..., m)
        stacked_means = torch.stack(all_means, dim=0)
        stacked_raw_vars = torch.stack(all_raw_vars, dim=0)
        stacked_scaled_vars = torch.stack(all_scaled_vars, dim=0)

        # For each point and output, pick the fidelity with lowest scaled var
        # Mean of scaled_var across outputs for selection
        mean_scaled_var = stacked_scaled_vars.mean(dim=-1)  # (n_fid, ...)
        best_fid_idx = mean_scaled_var.argmin(dim=0)  # (...)

        # Gather the best mean and variance
        idx_expanded = best_fid_idx.unsqueeze(0).unsqueeze(-1).expand(
            1, *batch_shape, n_outputs
        )
        selected_mean = torch.gather(stacked_means, 0, idx_expanded).squeeze(0)
        selected_var = torch.gather(stacked_raw_vars, 0, idx_expanded).squeeze(0)

        return _TVRPosterior(selected_mean, selected_var)


class _TVRPosterior:
    """Lightweight posterior wrapper for TVR ensemble predictions.

    Provides the ``mean``, ``variance``, ``rsample()``, and ``sample()``
    interface that BoTorch acquisition functions expect.
    """

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor):
        self._mean = mean
        self._variance = variance

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        return self._variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized samples (called by BoTorch MC acquisition functions)."""
        std = self._variance.sqrt()
        eps = torch.randn(
            *sample_shape, *self._mean.shape,
            device=self._mean.device, dtype=self._mean.dtype,
        )
        return self._mean + std * eps

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw samples from the posterior."""
        return self.rsample(sample_shape)

    @property
    def batch_shape(self) -> torch.Size:
        return self._mean.shape[:-1]

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape[-1:]


def fit_gp_botorch(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_cell_types: Optional[list[str]] = None,
    use_ilr: bool = True,
    use_saasbo: bool = False,
    saasbo_warmup: int = 256,
    saasbo_num_samples: int = 128,
    saasbo_thinning: int = 16,
    warm_start: bool = False,
    round_num: int = 1,
    warm_start_state: Optional[dict] = None,
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
        warm_start: If True, load hyperparameters from previous round's
            saved state as initialization (further fitting still occurs).
        round_num: Current optimization round number (used to locate the
            previous round's state file when ``warm_start=True``).
        warm_start_state: **Deprecated**. Legacy dict-based warm-start.
            Use ``warm_start=True`` instead.

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

    # Resolve warm-start state path
    gp_state_path = GP_STATE_DIR / f"round_{round_num - 1}.pt"

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
        # Warm-start multi-fidelity GP if requested
        if warm_start:
            load_gp_state(model, gp_state_path)
        # Skip lengthscale prior for multi-fidelity GP — its kernel structure
        # wraps lengthscales in a way incompatible with register_prior closures
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
        _set_dim_scaled_lengthscale_prior(model, train_X.shape[1])
        # Warm-start: load hyperparameters from previous round
        if warm_start:
            load_gp_state(model, gp_state_path)
        elif warm_start_state is not None:
            # Legacy path: dict-based warm-start (deprecated)
            try:
                model.load_state_dict(warm_start_state, strict=False)
                logger.info("Warm-started GP from previous round hyperparameters (legacy)")
            except Exception as exc:
                logger.warning("Could not warm-start GP: %s", exc)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    # Save GP state for warm-starting future rounds
    save_path = GP_STATE_DIR / f"round_{round_num}.pt"
    save_gp_state(model, save_path)

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
    use_ilr: bool = False,
    n_composition_parts: int = 0,
    cell_type_cols: Optional[list[str]] = None,
    target_profile: Optional[pd.Series] = None,
    n_duplicates: int = 0,
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
        use_ilr: Whether ILR transform was applied to Y.
        n_composition_parts: Number of composition parts (D) for ILR inverse.
        cell_type_cols: Names of cell type columns in Y.
        target_profile: Optional target composition profile for region targeting.
            When provided, scalarization uses cosine similarity to this target
            instead of equal weighting.
        n_duplicates: Number of within-batch QC duplicate slots. The top-scoring
            new conditions are duplicated to enable observation noise estimation
            from the same round (complements n_replicates in run_gpbo_loop which
            re-runs conditions from prior rounds).

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
            # Data-driven ref_point: below observed minimum per objective
            y_min = train_Y.min(dim=0).values
            y_max = train_Y.max(dim=0).values
            margin = 0.1 * (y_max - y_min).clamp(min=1e-6)
            ref_point = (y_min - margin).tolist()
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

            if use_ilr and n_composition_parts > 0:
                D = n_composition_parts
                V_T = _helmert_basis_torch(D).T  # (D-1, D)

                # Build target weights for composition-space scalarization
                if target_profile is not None and cell_type_cols is not None:
                    # Region targeting: weight by target profile (cosine similarity)
                    target_vec = np.array([
                        target_profile.get(ct, 0.0) for ct in cell_type_cols
                    ])
                    target_norm = np.linalg.norm(target_vec)
                    if target_norm > 0:
                        target_vec = target_vec / target_norm
                    comp_weights = torch.tensor(
                        target_vec, dtype=DTYPE, device=DEVICE
                    )
                    logger.info("Using target profile for scalarization (cosine similarity)")
                else:
                    # Equal weights in composition space = maximize mean cell type fraction.
                    comp_weights = torch.ones(D, dtype=DTYPE, device=DEVICE) / D

                def _composition_scalarization(samples, X=None):
                    # samples: (..., D-1) in ILR space
                    log_x = samples @ V_T  # (..., D)
                    log_x = log_x - log_x.max(dim=-1, keepdim=True).values
                    x = torch.exp(log_x)
                    comp = x / x.sum(dim=-1, keepdim=True)  # (..., D)
                    # When target_profile is set, comp_weights is the unit target
                    # vector so this dot product is (unnormalized) cosine similarity
                    # since comp is already normalized (sums to 1).
                    return (comp * comp_weights).sum(dim=-1)

                objective = GenericMCObjective(_composition_scalarization)

                # Compute best_f in composition space from training data
                train_Y_np = train_Y.cpu().numpy()
                train_comp = ilr_inverse(train_Y_np, n_composition_parts)
                comp_weights_np = comp_weights.cpu().numpy()
                best_f = float((train_comp @ comp_weights_np).max())
            else:
                # Legacy ILR-space scalarization (when --no-ilr is used)
                weights = torch.ones(train_Y.shape[1], dtype=DTYPE, device=DEVICE)
                weights = weights / weights.sum()

                objective = GenericMCObjective(
                    lambda samples, X=None: (samples * weights).sum(dim=-1)
                )
                best_f = float((train_Y @ weights.unsqueeze(1)).max())

            acqf = qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                objective=objective,
            )
        else:
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=train_Y.max(),
            )

    # Reserve slots for QC duplicates
    n_unique = n_recommendations - n_duplicates
    if n_unique < 1:
        raise ValueError(
            f"n_duplicates ({n_duplicates}) must be less than "
            f"n_recommendations ({n_recommendations})"
        )

    # Optimize acquisition function
    logger.info("Optimizing acquisition function for %d candidates...", n_unique)
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=n_unique,
        num_restarts=10,
        raw_samples=1024,
    )

    # Append QC duplicates: copy the top-scoring unique conditions
    if n_duplicates > 0:
        logger.info("Adding %d QC duplicate(s) for noise estimation", n_duplicates)
        # Top conditions by acquisition value are first in the batch
        dup_indices = list(range(min(n_duplicates, n_unique)))
        dup_candidates = candidates[dup_indices]
        candidates = torch.cat([candidates, dup_candidates], dim=0)
        dup_acq = acq_values[dup_indices] if acq_values.dim() > 0 else acq_values
        acq_values = torch.cat([acq_values, dup_acq], dim=0) if acq_values.dim() > 0 else acq_values

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

    # Mark QC duplicates
    is_duplicate = [False] * n_unique + [True] * n_duplicates
    recommendations["is_qc_duplicate"] = is_duplicate[:len(recommendations)]

    # Add predictions: convert from ILR space back to composition space if applicable
    if use_ilr and n_composition_parts > 0 and cell_type_cols is not None:
        pred_compositions = ilr_inverse(pred_mean, n_composition_parts)
        for i, ct_name in enumerate(cell_type_cols):
            recommendations[f"predicted_{ct_name}"] = pred_compositions[:, i]
        # Note: ILR-space std cannot be meaningfully inverse-transformed to
        # composition space, so we omit per-cell-type std columns.
    else:
        for i in range(pred_mean.shape[1]):
            recommendations[f"predicted_y{i}_mean"] = pred_mean[:, i]
            recommendations[f"predicted_y{i}_std"] = np.sqrt(pred_var[:, i])

    recommendations["acquisition_value"] = acq_values.detach().cpu().numpy() if acq_values.dim() > 0 else acq_values.item()

    # Add well labels (A1-D6 for 24-well plate)
    wells = [f"{chr(65 + i // 6)}{i % 6 + 1}" for i in range(n_recommendations)]
    recommendations.index = wells[:len(recommendations)]
    recommendations.index.name = "well"

    return recommendations


def validate_fidelity_correlation(
    real_fractions: pd.DataFrame,
    virtual_fractions: pd.DataFrame,
    fidelity_label: str = "virtual",
    method: str = "spearman",
) -> dict:
    """Validate cross-fidelity correlation for overlapping conditions.

    For conditions present in both real and virtual data, computes:
    1. Per-cell-type correlation (Pearson and Spearman)
    2. Overall correlation across all cell types

    Returns dict with:
        - overall_correlation: float (Spearman across all cell types)
        - per_cell_type: dict[str, float] (per-type Spearman)
        - recommendation: str ("use_mfbo" | "single_fidelity" | "skip_mfbo_use_cheap")
        - details: str (human-readable explanation)
        - n_overlap: int (number of overlapping conditions)

    Decision gate (per McDonald et al. 2025):
        - correlation > 0.9 everywhere: skip MF-BO, just use cheap fidelity as pre-filter
        - correlation < 0.3 everywhere: skip MF-BO, use single-fidelity GP on real data only
        - correlation 0.3-0.9: MF-BO is appropriate
    """
    from scipy.stats import spearmanr, pearsonr

    real_fractions = real_fractions.copy()
    virtual_fractions = virtual_fractions.copy()

    # Find overlapping conditions
    overlap = real_fractions.index.intersection(virtual_fractions.index)
    if len(overlap) == 0:
        logger.warning(
            "No overlapping conditions between real and %s data — "
            "cannot validate fidelity correlation",
            fidelity_label,
        )
        return {
            "overall_correlation": float("nan"),
            "per_cell_type": {},
            "recommendation": "use_mfbo",
            "details": (
                f"No overlapping conditions between real and {fidelity_label} data. "
                "Cannot validate correlation; defaulting to MF-BO."
            ),
            "n_overlap": 0,
        }

    # Align columns (union of cell types)
    all_cols = sorted(set(real_fractions.columns) | set(virtual_fractions.columns))
    for col in all_cols:
        if col not in real_fractions.columns:
            real_fractions[col] = 0.0
        if col not in virtual_fractions.columns:
            virtual_fractions[col] = 0.0

    real_overlap = real_fractions.loc[overlap, all_cols]
    virtual_overlap = virtual_fractions.loc[overlap, all_cols]

    # Per-cell-type correlation
    per_cell_type = {}
    for col in all_cols:
        r_vals = real_overlap[col].values
        v_vals = virtual_overlap[col].values
        # Skip constant columns (no variance -> correlation undefined)
        if np.std(r_vals) < 1e-12 or np.std(v_vals) < 1e-12:
            continue
        if method == "spearman":
            corr, _ = spearmanr(r_vals, v_vals)
        else:
            corr, _ = pearsonr(r_vals, v_vals)
        per_cell_type[col] = float(corr)

    # Overall correlation: flatten all cell types into one vector
    r_flat = real_overlap.values.flatten()
    v_flat = virtual_overlap.values.flatten()

    if np.std(r_flat) < 1e-12 or np.std(v_flat) < 1e-12:
        overall_corr = float("nan")
    else:
        if method == "spearman":
            overall_corr, _ = spearmanr(r_flat, v_flat)
        else:
            overall_corr, _ = pearsonr(r_flat, v_flat)
        overall_corr = float(overall_corr)

    # Decision gate
    if math.isnan(overall_corr):
        recommendation = "use_mfbo"
        details = (
            "Overall correlation is NaN (constant data). "
            f"Defaulting to MF-BO for {fidelity_label} data."
        )
    elif overall_corr > 0.9:
        recommendation = "skip_mfbo_use_cheap"
        details = (
            f"Cross-fidelity correlation with {fidelity_label} is very high "
            f"({overall_corr:.3f} > 0.9). MF-BO adds overhead without benefit — "
            f"consider using {fidelity_label} data as a cheap pre-filter instead."
        )
    elif overall_corr < FIDELITY_CORRELATION_THRESHOLD:
        recommendation = "single_fidelity"
        details = (
            f"Cross-fidelity correlation with {fidelity_label} is too low "
            f"({overall_corr:.3f} < {FIDELITY_CORRELATION_THRESHOLD}). "
            f"MF-BO would be counterproductive — dropping {fidelity_label} data "
            f"and using single-fidelity GP on real data only."
        )
    else:
        recommendation = "use_mfbo"
        details = (
            f"Cross-fidelity correlation with {fidelity_label} is moderate "
            f"({overall_corr:.3f}), within [0.3, 0.9]. MF-BO is appropriate."
        )

    logger.info(
        "Fidelity validation (%s): overall_corr=%.3f, recommendation=%s",
        fidelity_label, overall_corr, recommendation,
    )
    logger.info("  %s", details)

    return {
        "overall_correlation": overall_corr,
        "per_cell_type": per_cell_type,
        "recommendation": recommendation,
        "details": details,
        "n_overlap": len(overlap),
    }


def merge_multi_fidelity_data(
    data_sources: list[tuple[Path, Path, float]],
    cellflow_confidence_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge training data from multiple fidelity sources.

    Combines real, CellRank 2 virtual, and CellFlow virtual data into
    a single multi-fidelity training set.

    For CellFlow data (fidelity=0.0), if a screening report CSV with a
    ``confidence`` column exists alongside the fractions CSV, points
    below ``cellflow_confidence_threshold`` are filtered out.

    Args:
        data_sources: List of (fractions_csv, morphogen_csv, fidelity) tuples.
            Each tuple provides a data source at a given fidelity level.
        cellflow_confidence_threshold: Minimum confidence for CellFlow
            virtual points. Points below this are dropped (default 0.3).

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

        # Filter low-confidence CellFlow virtual points
        if fidelity == 0.0:
            report_path = fractions_csv.parent / "cellflow_screening_report.csv"
            if report_path.exists():
                report = pd.read_csv(str(report_path))
                if "confidence" in report.columns and "condition" in report.columns:
                    report = report.set_index("condition")
                    # Align report to X index
                    common_idx = X.index.intersection(report.index)
                    if len(common_idx) > 0:
                        conf = report.loc[common_idx, "confidence"]
                        keep_mask = conf >= cellflow_confidence_threshold
                        n_before = len(X)
                        keep_conditions = common_idx[keep_mask]
                        # Also keep any conditions not in the report
                        extra = X.index.difference(report.index)
                        keep_all = keep_conditions.append(extra)
                        X = X.loc[X.index.intersection(keep_all)]
                        Y = Y.loc[Y.index.intersection(keep_all)]
                        n_filtered = n_before - len(X)
                        if n_filtered > 0:
                            logger.info(
                                "Filtered %d/%d CellFlow points below confidence threshold %.2f",
                                n_filtered, n_before, cellflow_confidence_threshold,
                            )

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
    total_cost = 0.0
    for fid, count in fidelity_counts.items():
        label = {1.0: "real", 0.5: "CellRank2", 0.0: "CellFlow"}.get(fid, f"fid={fid}")
        cost_per = FIDELITY_COSTS.get(fid, 1.0)
        level_cost = count * cost_per
        total_cost += level_cost
        logger.info("  %s: %d points (cost ratio %.3f, equivalent cost %.2f)",
                     label, count, cost_per, level_cost)
    logger.info("  Total equivalent cost: %.2f real-experiment units", total_cost)

    return merged_X, merged_Y




def _select_replicate_conditions(
    train_X: pd.DataFrame,
    train_Y: pd.DataFrame,
    n_replicates: int = 2,
    strategy: str = "high_variance",
    model=None,
    active_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Select conditions to replicate in the next round.

    Strategies:
      - ``high_variance``: Replicate conditions with highest GP posterior
        variance.  Requires *model* and *active_cols*.
      - ``high_value``: Replicate the best-performing conditions (highest
        mean Y across cell types).
      - ``random``: Random selection from existing conditions.

    Args:
        train_X: Training morphogen matrix (with fidelity column).
        train_Y: Training cell type fraction matrix.
        n_replicates: Number of replicate conditions to select.
        strategy: Selection strategy name.
        model: Fitted BoTorch GP model (required for ``high_variance``).
        active_cols: Active column names used in model fitting.

    Returns:
        DataFrame of replicate morphogen conditions (*n_replicates* rows).
        Columns match *train_X* (without fidelity).
    """
    if n_replicates <= 0:
        return pd.DataFrame(columns=[c for c in train_X.columns if c != "fidelity"])

    n_available = len(train_X)
    n_replicates = min(n_replicates, n_available)

    morph_cols = [c for c in train_X.columns if c != "fidelity"]

    if strategy == "high_variance" and model is not None and active_cols is not None:
        X_active = train_X[active_cols].copy()
        X_tensor = torch.tensor(X_active.values, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            posterior = model.posterior(X_tensor)
            variances = posterior.variance.cpu().numpy()
        mean_var = variances.mean(axis=1)
        top_indices = np.argsort(mean_var)[-n_replicates:][::-1]
        selected = train_X.iloc[top_indices][morph_cols].copy()
    elif strategy == "high_value":
        mean_scores = train_Y.mean(axis=1)
        top_indices = mean_scores.nlargest(n_replicates).index
        selected = train_X.loc[top_indices, morph_cols].copy()
    elif strategy == "random":
        rng = np.random.default_rng(42)
        chosen = rng.choice(n_available, size=n_replicates, replace=False)
        selected = train_X.iloc[chosen][morph_cols].copy()
    else:
        if strategy == "high_variance":
            logger.warning(
                "Model not available for high_variance strategy, "
                "falling back to high_value",
            )
            return _select_replicate_conditions(
                train_X,
                train_Y,
                n_replicates,
                strategy="high_value",
            )
        raise ValueError(f"Unknown replicate strategy: {strategy}")

    logger.info(
        "Selected %d replicate conditions (strategy=%s): %s",
        n_replicates,
        strategy,
        list(selected.index),
    )
    return selected


def _estimate_noise_from_replicates(
    train_Y: pd.DataFrame,
    replicate_groups: dict[str, list[int]],
) -> float:
    """Estimate observation noise from replicate variance.

    For each group of replicate conditions, computes the within-group
    variance across cell type fractions.  The noise estimate is the
    pooled mean variance across all groups and cell types.

    Args:
        train_Y: Cell type fraction matrix (conditions x cell types).
        replicate_groups: Mapping from group label to list of row indices
            in *train_Y* that are replicates of each other.

    Returns:
        Noise level (variance) suitable for FixedNoiseGP.  Returns 0.0
        if no replicate groups have more than one member.
    """
    group_variances: list[float] = []

    for _label, indices in replicate_groups.items():
        if len(indices) < 2:
            continue
        group_data = train_Y.iloc[indices].values
        var_per_ct = np.var(group_data, axis=0, ddof=1)
        group_variances.append(float(np.mean(var_per_ct)))

    if not group_variances:
        logger.info("No replicate groups with 2+ members; cannot estimate noise")
        return 0.0

    noise = float(np.mean(group_variances))
    logger.info(
        "Estimated observation noise from %d replicate groups: %.6f",
        len(group_variances),
        noise,
    )
    return noise


def refine_target_profile(
    fractions: pd.DataFrame,
    fidelity_scores: pd.Series,
    original_target: pd.Series,
    learning_rate: float = 0.3,
) -> pd.Series:
    """Refine a target composition profile using observed data (DeMeo 2025).

    After Round 1, update the target profile by interpolating between the
    original reference (e.g., Braun fetal brain) and a "learned" profile
    derived from correlating per-cell-type fractions with composite fidelity
    scores.  Cell types whose abundance positively correlates with fidelity
    are upweighted; noise cell types are downweighted.

    The refined target is:
        refined = (1 - lr) * original + lr * learned

    Args:
        fractions: Cell type fractions DataFrame (conditions × cell types).
        fidelity_scores: Composite fidelity score per condition (Series
            indexed by condition name, values in [0, 1]).
        original_target: Original target composition profile (Series indexed
            by cell type name).
        learning_rate: Interpolation weight toward learned profile. 0 = keep
            original, 1 = fully learned. Default 0.3 (conservative).

    Returns:
        Refined target profile (Series, sums to 1, non-negative).

    Raises:
        ValueError: If no overlapping conditions between fractions and scores,
            or if learning_rate is outside [0, 1].
    """
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError(f"learning_rate must be in [0, 1], got {learning_rate}")

    # Align conditions between fractions and fidelity scores
    common = fractions.index.intersection(fidelity_scores.index)
    if len(common) < 3:
        logger.warning(
            "Too few overlapping conditions (%d) for target refinement; "
            "returning original target unchanged",
            len(common),
        )
        return original_target.copy()

    Y = fractions.loc[common]
    scores = fidelity_scores.loc[common]

    # Compute Pearson correlation between each cell type fraction and fidelity
    correlations = Y.corrwith(scores)

    # Replace NaN correlations (constant columns) with 0
    correlations = correlations.fillna(0.0)

    # Build learned profile: softmax of correlations to get non-negative weights
    # that sum to 1.  Using softmax rather than raw correlations ensures the
    # profile is a valid composition even when some correlations are negative.
    corr_shifted = correlations - correlations.max()  # numerical stability
    exp_corr = np.exp(corr_shifted)
    learned_profile = exp_corr / exp_corr.sum()

    # Align learned profile with original target (union of cell types)
    all_types = original_target.index.union(learned_profile.index)
    orig_aligned = original_target.reindex(all_types, fill_value=0.0)
    learned_aligned = learned_profile.reindex(all_types, fill_value=0.0)

    # Re-normalize original in case it didn't sum to 1
    orig_sum = orig_aligned.sum()
    if orig_sum > 0:
        orig_aligned = orig_aligned / orig_sum

    # Interpolate
    refined = (1 - learning_rate) * orig_aligned + learning_rate * learned_aligned

    # Ensure valid composition (non-negative, sums to 1)
    refined = refined.clip(lower=0.0)
    total = refined.sum()
    if total > 0:
        refined = refined / total

    logger.info(
        "Target profile refined: lr=%.2f, %d conditions, "
        "top learned cell types: %s",
        learning_rate,
        len(common),
        ", ".join(f"{ct}={v:.3f}" for ct, v in
                  learned_profile.nlargest(3).items()),
    )

    return refined


def run_gpbo_loop(
    fractions_csv: Path,
    morphogen_csv: Path,
    target_cell_types: Optional[list[str]] = None,
    n_recommendations: int = 24,
    round_num: int = 1,
    use_ilr: bool = True,
    virtual_sources: Optional[list[tuple[Path, Path, float]]] = None,
    use_saasbo: bool = False,
    use_tvr: bool = False,
    target_profile: Optional[pd.Series] = None,
    n_replicates: int = 2,
    replicate_strategy: str = "high_variance",
    warm_start: bool = False,
    refine_target: bool = False,
    refine_lr: float = 0.3,
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
        use_tvr: Use Targeted Variance Reduction (TVR) instead of
            SingleTaskMultiFidelityGP. Fits separate GPs per fidelity level
            and selects the model with lowest cost-scaled variance at each
            candidate point. Requires multi-fidelity data.
        target_profile: Optional target cell type composition profile (Series).
            When provided, the GP objective maximizes cosine similarity between
            predicted composition and this target instead of equal weighting.
        n_replicates: Number of plate wells reserved for replicate experiments.
            Set to 0 to disable replicates. (default: 2)
        replicate_strategy: Strategy for selecting replicates. One of
            "high_variance", "high_value", or "random". (default: "high_variance")
        warm_start: If True, warm-start GP from previous round's saved
            hyperparameters. State files are stored in GP_STATE_DIR.
        refine_target: If True and target_profile is provided, refine the
            target using observed data (DeMeo 2025 interpolation). The
            refined target is used for acquisition instead of the original.
        refine_lr: Learning rate for target refinement interpolation.
            0 = keep original, 1 = fully learned. (default: 0.3)

    Returns:
        DataFrame of recommended next experiments.
    """
    logger.info("--- GP-BO ROUND %d ---", round_num)

    # Build training set (potentially multi-fidelity)
    logger.info("Building training set...")

    # Always load real data first (used for bounds and as base training set)
    X_real, Y_real = build_training_set(fractions_csv, morphogen_csv)

    if virtual_sources:
        # Cross-fidelity correlation validation gate
        # Validate each virtual source against real data before merging
        validated_virtual = []
        for v_frac_csv, v_morph_csv, v_fidelity in virtual_sources:
            if not v_frac_csv.exists():
                continue
            # Only validate non-real fidelity sources
            if v_fidelity < 1.0:
                v_Y = pd.read_csv(str(v_frac_csv), index_col=0)
                fid_label = {0.5: "CellRank2", 0.0: "CellFlow"}.get(
                    v_fidelity, f"fidelity={v_fidelity}"
                )
                val_result = validate_fidelity_correlation(
                    Y_real, v_Y, fidelity_label=fid_label,
                )
                if val_result["recommendation"] == "single_fidelity":
                    logger.warning(
                        "Dropping %s data: cross-fidelity correlation too low (%.3f)",
                        fid_label, val_result["overall_correlation"],
                    )
                    continue
            validated_virtual.append((v_frac_csv, v_morph_csv, v_fidelity))

        if validated_virtual:
            all_sources = [(fractions_csv, morphogen_csv, 1.0)] + validated_virtual
            X, Y = merge_multi_fidelity_data(all_sources)
        else:
            logger.info("All virtual sources dropped by fidelity validation; "
                        "using single-fidelity GP on real data only")
            X, Y = X_real, Y_real
    else:
        X, Y = X_real, Y_real

    # Refine target profile using observed data (DeMeo 2025)
    if refine_target and target_profile is not None:
        # Cosine similarity to target as fidelity proxy
        target_aligned = target_profile.reindex(Y_real.columns, fill_value=0.0)
        target_arr = target_aligned.values
        target_norm = np.linalg.norm(target_arr)
        if target_norm > 0:
            Y_vals = Y_real.values
            row_norms = np.linalg.norm(Y_vals, axis=1)
            cos_sims = (Y_vals @ target_arr) / (row_norms * target_norm + 1e-12)
            fidelity_proxy = pd.Series(cos_sims, index=Y_real.index)
        else:
            logger.warning("Target profile is zero after alignment; skipping refinement")
            fidelity_proxy = None
        if fidelity_proxy is not None:
            target_profile = refine_target_profile(
                Y_real, fidelity_proxy, target_profile, learning_rate=refine_lr,
            )

    # Compute active bounds: use real data for morphogen ranges, but merged X
    # when virtual sources exist so log_harvest_day variance is detected.
    bounds_X = X if virtual_sources else X_real
    active_bounds, active_cols = _compute_active_bounds(bounds_X, list(X.columns))

    # Filter X to active columns only
    X_active = X[active_cols]
    logger.info("Active X shape: %s (from %s)", X_active.shape, X.shape)

    # Fit GP on active dimensions only (save/load handled inside fit_gp_botorch)
    if use_tvr and "fidelity" in X_active.columns and X_active["fidelity"].nunique() > 1:
        logger.info("Using TVR (Targeted Variance Reduction) with %d fidelity levels",
                     X_active["fidelity"].nunique())
        model, train_X, train_Y, cell_type_cols = fit_tvr_models(
            X_active, Y,
            target_cell_types=target_cell_types,
            use_ilr=use_ilr,
        )
    else:
        if use_tvr:
            logger.warning("TVR requested but only 1 fidelity level; falling back to standard GP")
        model, train_X, train_Y, cell_type_cols = fit_gp_botorch(
            X_active, Y,
            target_cell_types=target_cell_types,
            use_ilr=use_ilr,
            use_saasbo=use_saasbo,
            warm_start=warm_start,
            round_num=round_num,
        )

    # Recommend next experiments (in active dimensions)
    # For TVR, the model doesn't have a fidelity dimension — exclude it
    # from the columns and bounds passed to the acquisition optimizer.
    if use_tvr and isinstance(model, TVRModelEnsemble):
        rec_cols = [c for c in X_active.columns if c != "fidelity"]
        rec_bounds = {k: v for k, v in active_bounds.items() if k != "fidelity"}
    else:
        rec_cols = list(X_active.columns)
        rec_bounds = active_bounds

    # Reserve wells for replicates if requested
    n_novel = max(1, n_recommendations - n_replicates)
    recommendations = recommend_next_experiments(
        model, train_X, train_Y,
        bounds=rec_bounds,
        columns=rec_cols,
        n_recommendations=n_novel,
        use_ilr=use_ilr,
        n_composition_parts=len(cell_type_cols),
        cell_type_cols=cell_type_cols,
        target_profile=target_profile,
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

    # Append replicate conditions
    if n_replicates > 0:
        replicates = _select_replicate_conditions(
            X, Y,
            n_replicates=n_replicates,
            strategy=replicate_strategy,
            model=model,
            active_cols=active_cols,
        )
        # Mark novel vs replicate rows
        recommendations["is_replicate"] = False
        rep_df = replicates.copy()
        # Add missing columns (predictions, etc.) as NaN for replicates
        for col in recommendations.columns:
            if col not in rep_df.columns:
                rep_df[col] = np.nan
        rep_df["is_replicate"] = True
        # Reorder rep_df columns to match recommendations
        rep_df = rep_df[recommendations.columns]
        recommendations = pd.concat([recommendations, rep_df], ignore_index=True)

        # Re-label wells for the full plate
        wells = [f"{chr(65 + i // 6)}{i % 6 + 1}" for i in range(len(recommendations))]
        recommendations.index = wells[:len(recommendations)]
        recommendations.index.name = "well"

        n_reps_actual = int(recommendations["is_replicate"].sum())
        logger.info(
            "Plate map: %d novel + %d replicate = %d total",
            len(recommendations) - n_reps_actual,
            n_reps_actual,
            len(recommendations),
        )

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
    parser.add_argument("--n-duplicates", type=int, default=2,
                        help="Number of QC duplicate slots for noise estimation (default: 2)")
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
    parser.add_argument("--tvr", action="store_true",
                        help="Use Targeted Variance Reduction (fit per-fidelity GPs, "
                             "select by lowest cost-scaled variance)")
    parser.add_argument("--target-region", type=str, default=None,
                        help="Named region to optimize for (e.g., dorsal_telencephalon, ap_axis)")
    parser.add_argument("--target-profile", type=str, default=None,
                        help="Path to custom target profile CSV")
    parser.add_argument("--target-fbaxis", type=float, default=None,
                        help="Target A-P axis position (0=anterior/forebrain, 1=posterior/hindbrain). "
                             "Used with --target-region ap_axis")
    parser.add_argument("--list-regions", action="store_true",
                        help="List available region profiles and exit")
    parser.add_argument("--n-replicates", type=int, default=2,
                        help="Number of wells reserved for replicate experiments (default: 2)")
    parser.add_argument("--replicate-strategy", type=str, default="high_variance",
                        choices=["high_variance", "high_value", "random"],
                        help="Strategy for selecting replicate conditions (default: high_variance)")
    parser.add_argument("--validate-fidelity", action="store_true",
                        help="Validate cross-fidelity correlation and print recommendation, then exit")
    parser.add_argument("--warm-start", action="store_true",
                        help="Warm-start GP from previous round's hyperparameters")
    parser.add_argument("--refine-target", action="store_true",
                        help="Refine target profile using observed data (DeMeo 2025 interpolation)")
    parser.add_argument("--refine-lr", type=float, default=0.3,
                        help="Learning rate for target refinement (0=keep original, 1=fully learned, default: 0.3)")
    args = parser.parse_args()

    # Handle --list-regions
    if args.list_regions:
        from gopro.region_targets import list_named_profiles, AP_AXIS_REGION
        profiles_df = list_named_profiles()
        print("\nAvailable region profiles:")
        print("=" * 80)
        for _, row in profiles_df.iterrows():
            print(f"  {row['name']:<30s} {row['description']}")
        print(f"\n  {AP_AXIS_REGION:<30s} Continuous A-P axis targeting (use with --target-fbaxis)")
        print(f"\nUse: --target-region <name>")
        print(f"  or: --target-region {AP_AXIS_REGION} --target-fbaxis 0.7  (for hindbrain)")
        raise SystemExit(0)

    # Handle --validate-fidelity
    if args.validate_fidelity:
        fractions_path = Path(args.fractions) if args.fractions else DATA_DIR / "gp_training_labels_amin_kelley.csv"
        morphogen_path = Path(args.morphogens) if args.morphogens else DATA_DIR / "morphogen_matrix_amin_kelley.csv"
        if not fractions_path.exists():
            logger.error("Fractions CSV not found: %s", fractions_path)
            raise SystemExit(1)
        Y_real = pd.read_csv(str(fractions_path), index_col=0)

        # Validate each virtual source
        virtual_pairs = []
        cr2_frac = Path(args.cellrank2_fractions) if args.cellrank2_fractions else DATA_DIR / "cellrank2_virtual_fractions.csv"
        if cr2_frac.exists():
            virtual_pairs.append((cr2_frac, 0.5, "CellRank2"))
        cf_frac = Path(args.cellflow_fractions) if args.cellflow_fractions else DATA_DIR / "cellflow_virtual_fractions_200.csv"
        if cf_frac.exists():
            virtual_pairs.append((cf_frac, 0.0, "CellFlow"))

        if not virtual_pairs:
            print("No virtual data sources found to validate.")
            raise SystemExit(0)

        import json
        for v_frac, v_fid, v_label in virtual_pairs:
            v_Y = pd.read_csv(str(v_frac), index_col=0)
            result = validate_fidelity_correlation(Y_real, v_Y, fidelity_label=v_label)
            print(f"\n=== {v_label} (fidelity={v_fid}) ===")
            print(f"  Overlapping conditions: {result['n_overlap']}")
            print(f"  Overall correlation: {result['overall_correlation']:.4f}")
            print(f"  Recommendation: {result['recommendation']}")
            print(f"  Details: {result['details']}")
            if result["per_cell_type"]:
                print(f"  Per-cell-type correlations:")
                for ct, corr in sorted(result["per_cell_type"].items(), key=lambda x: x[1]):
                    print(f"    {ct}: {corr:.4f}")
        raise SystemExit(0)

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

    # Resolve target profile
    target_profile = None
    if args.target_region and args.target_profile:
        logger.error("Cannot specify both --target-region and --target-profile")
        raise SystemExit(1)

    if args.target_region:
        from gopro.region_targets import AP_AXIS_REGION
        if args.target_region == AP_AXIS_REGION:
            from gopro.region_targets import build_ap_target_profile
            if args.target_fbaxis is None:
                logger.error("--target-region ap_axis requires --target-fbaxis (0=anterior, 1=posterior)")
                raise SystemExit(1)
            target_profile = build_ap_target_profile(args.target_fbaxis)
            logger.info("Targeting A-P axis position: %.2f", args.target_fbaxis)
        else:
            from gopro.region_targets import NAMED_REGION_PROFILES, load_region_profile
            if args.target_region not in NAMED_REGION_PROFILES:
                logger.error("Unknown region '%s'. Use --list-regions to see options.",
                             args.target_region)
                raise SystemExit(1)
            # Try to load from reference atlas; fall back to a simple placeholder
            ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"
            braun_path = DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
            for atlas_path in [ref_path, braun_path]:
                if atlas_path.exists():
                    target_profile = load_region_profile(args.target_region, atlas_path)
                    break
            if target_profile is None:
                logger.error("No reference atlas found. Cannot load region profile.")
                raise SystemExit(1)
            logger.info("Targeting region: %s", args.target_region)

    if args.target_profile:
        from gopro.region_targets import load_target_profile_csv
        target_profile = load_target_profile_csv(Path(args.target_profile))
        logger.info("Using custom target profile from %s", args.target_profile)

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
        use_tvr=args.tvr,
        target_profile=target_profile,
        n_replicates=args.n_replicates,
        replicate_strategy=args.replicate_strategy,
        warm_start=args.warm_start,
        refine_target=args.refine_target,
        refine_lr=args.refine_lr,
    )

    logger.info("--- NEXT EXPERIMENT RECOMMENDATIONS ---")
    morph_cols = [c for c in MORPHOGEN_COLUMNS if c in recs.columns]
    nonzero_cols = [c for c in morph_cols if recs[c].abs().sum() > 0.01]
    pred_cols = [c for c in recs.columns if "predicted" in c or "acquisition" in c]
    logger.info("\n%s", recs[nonzero_cols + pred_cols].to_string())
    logger.info("Give this plate map to the wet lab team!")
