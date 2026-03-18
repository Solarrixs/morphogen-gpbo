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

import functools
import math
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Any, Literal, Optional

from gopro.config import (
    CONVERGENCE_ACQUISITION_DECAY_THRESHOLD,
    CONVERGENCE_CLUSTER_SPREAD_THRESHOLD,
    CONVERGENCE_POSTERIOR_EVAL_POINTS,
    DATA_DIR,
    ENSEMBLE_DEFAULT_N_RESTARTS,
    ENSEMBLE_STABILITY_LOW_THRESHOLD,
    FIDELITY_CORRELATION_THRESHOLD,
    FIDELITY_COSTS,
    FIDELITY_LABELS,
    FIDELITY_SKIP_MFBO_THRESHOLD,
    GP_STATE_DIR,
    KERNEL_COMPLEXITY_THRESHOLDS,
    MORPHOGEN_COLUMNS,
    PROTEIN_MW_KDA,
    TIMING_FULL,
    TIMING_WINDOW_COLUMNS,
    nM_to_uM,
    ng_mL_to_uM,
    get_logger,
)

logger = get_logger(__name__)

# BoTorch requires float64, which MPS doesn't support. Use CPU for GP fitting.
# MPS can be used for neural network components (CellFlow, scPoli) separately.
DEVICE = torch.device("cpu")
DTYPE = torch.double

# --- Fidelity kernel remap ---
# SingleTaskMultiFidelityGP uses LinearTruncatedFidelityKernel which collapses
# when fidelity ∈ {0.0, 1.0} (boundary values annihilate the inter-fidelity
# kernel component). Remap to the open interval (0, 1) to avoid this.
FIDELITY_KERNEL_REMAP: dict[float, float] = {
    0.0: 1 / 3,   # CellFlow → 0.333
    0.5: 1 / 2,   # CellRank2 → 0.500
    1.0: 2 / 3,   # real → 0.667
}
FIDELITY_KERNEL_UNMAP: dict[float, float] = {v: k for k, v in FIDELITY_KERNEL_REMAP.items()}


def _remap_fidelity(fidelity_values: torch.Tensor) -> torch.Tensor:
    """Remap fidelity values from {0.0, 0.5, 1.0} → open interval (0, 1).

    Prevents boundary collapse in LinearTruncatedFidelityKernel.  Unknown
    fidelity values are linearly interpolated between 0→1/3 and 1→2/3.
    """
    out = fidelity_values.clone()
    matched = torch.zeros_like(fidelity_values, dtype=torch.bool)
    for orig, mapped in FIDELITY_KERNEL_REMAP.items():
        hits = torch.isclose(fidelity_values, torch.tensor(orig, dtype=fidelity_values.dtype))
        out[hits] = mapped
        matched |= hits
    # Fallback: linearly remap any unseen values from [0, 1] → [1/3, 2/3]
    if (~matched).any():
        logger.warning("Unknown fidelity values encountered: %s", fidelity_values[~matched].tolist())
        out[~matched] = 1 / 3 + fidelity_values[~matched] * (2 / 3 - 1 / 3)
    return out


def _unmap_fidelity(remapped_values: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_remap_fidelity``: (0, 1) → {0.0, 0.5, 1.0}.

    Note: currently unused in production — reserved for post-hoc analysis.
    """
    out = remapped_values.clone()
    matched = torch.zeros_like(remapped_values, dtype=torch.bool)
    for mapped, orig in FIDELITY_KERNEL_UNMAP.items():
        hits = torch.isclose(remapped_values, torch.tensor(mapped, dtype=remapped_values.dtype))
        out[hits] = orig
        matched |= hits
    if (~matched).any():
        logger.warning("Unmapped fidelity values (no known inverse): %s", remapped_values[~matched].tolist())
    return out

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

    # Include fidelity only when multiple fidelity levels exist; a single-fidelity
    # column has zero variance which causes NaN in BoTorch Normalize (upper-lower=0)
    if "fidelity" in columns:
        if isinstance(X, pd.DataFrame) and "fidelity" in X.columns:
            n_unique = X["fidelity"].nunique()
        else:
            n_unique = 1
        if n_unique > 1:
            fid_min = float(X["fidelity"].min())
            fid_max = float(X["fidelity"].max())
            active_cols.append("fidelity")
            active_bounds["fidelity"] = (fid_min, fid_max)
        else:
            logger.info("Dropping fidelity column (single level, zero variance)")

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


def _fit_lassobo(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    lasso_alpha: float = 0.1,
    n_steps: int = 200,
    lr: float = 0.05,
) -> "SingleTaskGP":
    """Fit a GP with Lasso-regularized lengthscale estimation.

    LassoBO (AISTATS 2025) applies an L1 penalty to the inverse lengthscales
    during MAP optimization, encouraging sparsity.  This achieves similar
    variable selection to SAASBO but via gradient-based optimization rather
    than NUTS sampling, making it much faster (~10x) while giving comparable
    variable selection.

    The penalized objective is::

        loss = -MLL + alpha * sum(1 / lengthscale_i)

    Small lengthscales → large penalties, so the optimizer drives irrelevant
    dimensions' lengthscales toward infinity (effectively removing them).

    Args:
        train_X: Training inputs tensor (N x d).
        train_Y: Training targets tensor (N x m).
        lasso_alpha: L1 penalty strength on inverse lengthscales.
            Higher values → more aggressive variable pruning. (default: 0.1)
        n_steps: Number of Adam optimization steps. (default: 200)
        lr: Learning rate for Adam optimizer. (default: 0.05)

    Returns:
        Fitted SingleTaskGP model with sparse lengthscales.
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import gpytorch

    d = train_X.shape[1]
    m = train_Y.shape[1]

    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=m),
    )
    # Remove default lengthscale and noise priors — the L1 penalty already
    # regularizes lengthscales, and LogNormalPrior's validate_args check
    # can raise during Adam optimization when parameter values temporarily
    # exit the support region.
    base_kernel = getattr(model.covar_module, "base_kernel", model.covar_module)
    for prior_name in list(base_kernel._priors.keys()):
        del base_kernel._priors[prior_name]
    for prior_name in list(model.likelihood.noise_covar._priors.keys()):
        del model.likelihood.noise_covar._priors[prior_name]

    has_lengthscale = hasattr(base_kernel, "lengthscale")
    if not has_lengthscale:
        logger.warning("LassoBO: kernel has no lengthscale attribute; L1 penalty disabled")

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    prev_loss = float("inf")
    patience_counter = 0
    convergence_tol = 1e-6
    patience = 5

    with gpytorch.settings.debug(False):
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            mll_value = mll(output, model.train_targets)
            loss = -mll_value.sum()  # sum for multi-output

            # L1 penalty on inverse lengthscales → drives irrelevant dims to ∞
            if has_lengthscale:
                inv_ls = 1.0 / base_kernel.lengthscale.clamp(min=1e-6)
                loss = loss + lasso_alpha * inv_ls.sum()

            loss.backward()
            optimizer.step()

            # Early exit on convergence
            curr_loss = loss.item()
            if abs(prev_loss - curr_loss) < convergence_tol:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_loss = curr_loss

    model.eval()
    model.likelihood.eval()

    # Log which dimensions were effectively pruned
    ls = _extract_lengthscales(model, d)
    if ls is not None:
        median_ls = float(np.median(ls))
        # Dimensions with lengthscale > 10x median are considered pruned
        pruned = np.sum(ls > 10 * median_ls)
        logger.info(
            "LassoBO: %d/%d dimensions pruned (alpha=%.3f, median_ls=%.3f)",
            pruned, d, lasso_alpha, median_ls,
        )

    return model


def _build_additive_interaction_kernel(d: int):
    """Build an additive + interaction kernel for GP fitting.

    Following NAIAD (Qin et al., ICML 2025, arXiv:2411.12010), decomposes the
    covariance function into:
      k(x, x') = k_additive(x, x') + k_interaction(x, x')

    where k_additive is a sum of 1D Matern 5/2 kernels (one per morphogen) and
    k_interaction is a full ARD Matern 5/2 over all dimensions.  The additive
    component captures independent morphogen effects with O(d) parameters,
    while the interaction component captures higher-order effects.  The
    interaction outputscale is initialized small (0.1) to encode a prior
    toward additivity.

    Args:
        d: Number of input dimensions.

    Returns:
        A GPyTorch kernel suitable for passing as ``covar_module`` to
        ``SingleTaskGP``.
    """
    from gpytorch.kernels import AdditiveKernel, MaternKernel, ScaleKernel

    # Additive: sum of d independent 1D Matern 5/2 kernels
    additive_kernels = [
        ScaleKernel(MaternKernel(nu=2.5, active_dims=(i,)))
        for i in range(d)
    ]
    # No outer ScaleKernel — per-dim ScaleKernels already provide variance params
    k_additive = AdditiveKernel(*additive_kernels)

    # Interaction: full ARD Matern 5/2 over all dimensions
    k_interaction = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=d)
    )
    # Initialize interaction scale small → prior toward additivity
    k_interaction.outputscale = 0.1

    combined = k_additive + k_interaction
    logger.info(
        "Built additive+interaction kernel (d=%d): %d additive + 1 interaction",
        d, d,
    )
    return combined


def _select_kernel_complexity(
    n_conditions: int,
    d_active: int,
    thresholds: Optional[dict[str, float]] = None,
) -> dict:
    """Auto-select GP kernel complexity based on data density.

    Implements the adaptive complexity schedule from NAIAD (Qin et al.,
    ICML 2025): start simple in sparse-data regimes to avoid overfitting,
    increase complexity as data accumulates.

    The N/d ratio (conditions per active dimension) drives the selection:
      - N/d < 8:  shared lengthscale (fewest params)
      - 8 ≤ N/d < 15: per-dim ARD (standard)
      - N/d ≥ 15: SAASBO (fully Bayesian with sparsity prior)

    Args:
        n_conditions: Number of training conditions (N).
        d_active: Number of active (non-zero-variance) input dimensions.
        thresholds: Override thresholds dict. Must have keys "shared" and
            "ard" mapping to N/d cutoffs. If None, uses
            KERNEL_COMPLEXITY_THRESHOLDS.

    Returns:
        Dict with keys:
          - "kernel_type": one of "shared", "ard", "saasbo"
          - "use_saasbo": bool
          - "n_d_ratio": float
          - "reason": human-readable explanation
    """
    if thresholds is None:
        thresholds = KERNEL_COMPLEXITY_THRESHOLDS

    shared_threshold = thresholds["shared"]
    ard_threshold = thresholds["ard"]

    d_safe = max(d_active, 1)
    ratio = n_conditions / d_safe

    if ratio < shared_threshold:
        return {
            "kernel_type": "shared",
            "use_saasbo": False,
            "n_d_ratio": ratio,
            "reason": (
                f"N/d={ratio:.1f} < {shared_threshold} → shared lengthscale "
                f"(N={n_conditions}, d={d_active})"
            ),
        }
    elif ratio < ard_threshold:
        return {
            "kernel_type": "ard",
            "use_saasbo": False,
            "n_d_ratio": ratio,
            "reason": (
                f"{shared_threshold} ≤ N/d={ratio:.1f} < {ard_threshold} → "
                f"per-dim ARD (N={n_conditions}, d={d_active})"
            ),
        }
    else:
        return {
            "kernel_type": "saasbo",
            "use_saasbo": True,
            "n_d_ratio": ratio,
            "reason": (
                f"N/d={ratio:.1f} ≥ {ard_threshold} → SAASBO fully Bayesian "
                f"(N={n_conditions}, d={d_active})"
            ),
        }


# KernelSpec bundles the two variables that must stay in sync (A-C-001 fix).
KernelSpec = namedtuple("KernelSpec", ["kernel_type", "use_saasbo"])


def _resolve_kernel_spec(
    kernel_type: str, use_saasbo: bool, adaptive_complexity: bool,
    n_conditions: int, d_active: int,
) -> KernelSpec:
    """Return a single KernelSpec, resolving adaptive complexity if enabled."""
    if not adaptive_complexity:
        return KernelSpec(kernel_type=kernel_type, use_saasbo=use_saasbo)
    complexity = _select_kernel_complexity(
        n_conditions=n_conditions, d_active=d_active,
    )
    logger.info("Adaptive complexity: %s", complexity["reason"])
    if kernel_type != "ard" or use_saasbo:
        logger.warning(
            "Adaptive complexity overriding explicit kernel_type=%r, "
            "use_saasbo=%r → %s",
            kernel_type, use_saasbo, complexity["kernel_type"],
        )
    if complexity["use_saasbo"]:
        return KernelSpec(kernel_type="ard", use_saasbo=True)
    return KernelSpec(kernel_type=complexity["kernel_type"], use_saasbo=False)


@functools.lru_cache(maxsize=8)
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
    Delegates to the cached numpy version to avoid duplicated loop logic.
    """
    return torch.tensor(_helmert_basis(D), dtype=DTYPE, device=DEVICE)


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


def _sobol_eval_points(
    d: int, n_points: int, lower: torch.Tensor, upper: torch.Tensor,
) -> torch.Tensor:
    """Generate Sobol evaluation points scaled to [lower, upper]."""
    from torch.quasirandom import SobolEngine

    sobol = SobolEngine(dimension=d, scramble=True, seed=42)
    unit = sobol.draw(n_points).to(dtype=DTYPE, device=DEVICE)
    return lower + unit * (upper - lower)


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
            covar = model.covar_module
            # Additive+interaction kernel: find the ARD sub-kernel by attribute
            if hasattr(covar, 'kernels'):
                for sub_k in covar.kernels:
                    base = getattr(sub_k, 'base_kernel', sub_k)
                    if getattr(base, 'ard_num_dims', None) is not None:
                        ls = base.lengthscale
                        if ls is not None:
                            return ls.detach().cpu().numpy().flatten()
            if hasattr(covar, 'base_kernel') and covar.base_kernel is not None:
                ls = covar.base_kernel.lengthscale
                if ls is not None:
                    return ls.detach().cpu().numpy().flatten()
            elif hasattr(covar, 'lengthscale'):
                ls = covar.lengthscale
                if ls is not None:
                    return ls.detach().cpu().numpy().flatten()
    except (AttributeError, RuntimeError):
        pass
    return None


def _extract_per_output_lengthscales(model, n_input_dims: int):
    """Extract per-output lengthscale matrix from a ModelListGP.

    Returns a (n_input_dims x n_outputs) numpy array where entry [d, k]
    is the ARD lengthscale of morphogen d for output k. Shorter lengthscales
    mean higher sensitivity. Returns None if extraction fails.

    This matrix answers "which morphogens drive which cell types?" and is
    the primary interpretability output of per-type GP models (GPerturb,
    Xing & Yau, Nature Communications 2025).
    """
    if not hasattr(model, 'models'):
        return None
    all_ls = []
    for sub_model in model.models:
        ls = _extract_lengthscales(sub_model, n_input_dims)
        if ls is None:
            return None
        all_ls.append(ls)
    # Stack: each row is a sub-model's lengthscales → transpose to (d x n_outputs)
    return np.column_stack(all_ls)


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

            # Scale variance by cost: cheaper models get lower scaled variance
            # cost=0.001 (CellFlow) → scaled_var = var * 0.001 (very small → preferred)
            # cost=1.0 (real) → scaled_var = var * 1.0 (full penalty)
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
    use_lassobo: bool = False,
    lassobo_alpha: float = 0.1,
    warm_start: bool = False,
    round_num: int = 1,
    warm_start_state: Optional[dict] = None,
    kernel_type: Literal["ard", "additive_interaction", "shared"] = "ard",
    cat_dims: Optional[list[int]] = None,
    per_type_gp: bool = False,
    noise_variance: Optional[pd.DataFrame] = None,
    save_state: bool = True,
) -> tuple:
    """Fit a GP using BoTorch (MAP, fully Bayesian SAASBO, LassoBO, or multi-fidelity).

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
        use_lassobo: Use LassoBO (L1-regularized MAP variable selection,
            AISTATS 2025). Much faster than SAASBO — gradient-based
            optimization instead of NUTS sampling. Mutually exclusive
            with use_saasbo.
        lassobo_alpha: L1 penalty strength for LassoBO. Higher values →
            more aggressive variable pruning. (default: 0.1)
        warm_start: If True, load hyperparameters from previous round's
            saved state as initialization (further fitting still occurs).
        round_num: Current optimization round number (used to locate the
            previous round's state file when ``warm_start=True``).
        warm_start_state: **Deprecated**. Legacy dict-based warm-start.
            Use ``warm_start=True`` instead.
        kernel_type: Kernel structure. ``"ard"`` (default) uses Matern 5/2 +
            ARD. ``"additive_interaction"`` uses a sum-of-1D additive kernel
            plus a full ARD interaction kernel (NAIAD, Qin et al. ICML 2025).
            ``"shared"`` uses Matern 5/2 with a single shared lengthscale
            (fewest parameters, best for sparse data regimes).
        cat_dims: List of column indices in X that are categorical (integer-
            coded). When provided, uses MixedSingleTaskGP instead of
            SingleTaskGP to handle mixed continuous+categorical inputs
            (Sanchis-Calleja timing windows).
        per_type_gp: If True, fit separate GP per output dimension (MAP path
            only). Produces a ModelListGP with per-output lengthscales for
            interpretability (GPerturb, Xing & Yau 2025). Each sub-model
            gets its own Matern 5/2 + ARD kernel.
        noise_variance: Per-condition, per-cell-type bootstrap variance
            estimates (conditions x cell types DataFrame).  When provided,
            passed as ``train_Yvar`` to ``SingleTaskGP`` for heteroscedastic
            noise modeling.  Aligned to *Y* index/columns automatically.
            Only used on the standard MAP and per-type-GP paths (ignored for
            SAASBO, LassoBO, and multi-fidelity).

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

    _VALID_KERNEL_TYPES = ("ard", "additive_interaction", "shared")
    if kernel_type not in _VALID_KERNEL_TYPES:
        raise ValueError(
            f"Unknown kernel_type={kernel_type!r}. Must be one of {_VALID_KERNEL_TYPES}"
        )

    # Prepare tensors
    train_X = torch.tensor(X.values, dtype=DTYPE, device=DEVICE)
    Y_values = Y_selected.values

    if use_ilr and Y_values.shape[1] > 1:
        logger.info("Applying ILR transform to compositional Y data...")
        Y_values = ilr_transform(Y_values)
        logger.info("ILR-transformed Y shape: %s", Y_values.shape)

    train_Y = torch.tensor(Y_values, dtype=DTYPE, device=DEVICE)

    # Build heteroscedastic noise tensor from bootstrap variance
    train_Yvar = None
    if noise_variance is not None:
        # Align columns to selected cell types (index already aligned by caller)
        nv = noise_variance.reindex(columns=Y_selected.columns).fillna(0.0)
        nv_values = nv.values
        if use_ilr and nv_values.shape[1] > 1:
            # Delta-method: Var_ilr = J @ diag(var_comp) @ J^T per row,
            # where J = diag(1/y) @ V is the Jacobian of log(y) @ V.
            D = nv_values.shape[1]
            V = _helmert_basis(D)  # (D, D-1)
            Y_safe = _multiplicative_replacement(Y_selected.values)
            ilr_var = np.empty((nv_values.shape[0], D - 1))
            for i in range(nv_values.shape[0]):
                J = np.diag(1.0 / Y_safe[i]) @ V  # (D, D-1)
                cov_ilr = J.T @ np.diag(nv_values[i]) @ J  # (D-1, D-1)
                ilr_var[i] = np.diag(cov_ilr)
            nv_values = ilr_var
        train_Yvar = torch.tensor(nv_values, dtype=DTYPE, device=DEVICE)
        train_Yvar = torch.clamp(train_Yvar, min=1e-6)
        logger.info(
            "Using heteroscedastic noise (train_Yvar): mean=%.2e, range=[%.2e, %.2e]",
            train_Yvar.mean().item(),
            train_Yvar.min().item(),
            train_Yvar.max().item(),
        )

    # Resolve warm-start state path
    gp_state_path = GP_STATE_DIR / f"round_{round_num - 1}.pt"

    # Fit GP
    logger.info("Fitting BoTorch GP...")
    logger.info("X: %s, Y: %s", train_X.shape, train_Y.shape)

    # Validate mutually exclusive fitting strategies
    if use_saasbo and use_lassobo:
        raise ValueError(
            "use_saasbo and use_lassobo are mutually exclusive — choose one variable selection method"
        )
    if per_type_gp and use_saasbo:
        raise ValueError(
            "per_type_gp is incompatible with use_saasbo — choose one fitting strategy"
        )
    if per_type_gp and use_lassobo:
        raise ValueError(
            "per_type_gp is incompatible with use_lassobo — choose one fitting strategy"
        )
    if per_type_gp and cat_dims:
        raise ValueError(
            "per_type_gp is incompatible with cat_dims (MixedSingleTaskGP) — choose one fitting strategy"
        )

    # Check if we have fidelity column
    has_fidelity = "fidelity" in X.columns
    fidelity_idx = list(X.columns).index("fidelity") if has_fidelity else None

    if has_fidelity and X["fidelity"].nunique() > 1:
        from botorch.models import SingleTaskMultiFidelityGP
        logger.info("Using multi-fidelity GP (fidelity column detected)")
        if use_saasbo:
            logger.info("SAASBO ignored — multi-fidelity takes priority")
        if use_lassobo:
            logger.info("LassoBO ignored — multi-fidelity takes priority")
        if per_type_gp:
            logger.info("per_type_gp ignored — multi-fidelity takes priority")
        # Remap fidelity values to open interval (0, 1) to prevent boundary
        # collapse in LinearTruncatedFidelityKernel (TODO-24).
        train_X = train_X.clone()
        train_X[:, fidelity_idx] = _remap_fidelity(train_X[:, fidelity_idx])
        logger.info(
            "Remapped fidelity for MF kernel: %s",
            sorted(train_X[:, fidelity_idx].unique().tolist()),
        )
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
        if cat_dims:
            logger.warning(
                "SAASBO does not support categorical dims; ignoring cat_dims=%s",
                cat_dims,
            )
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
    elif use_lassobo:
        if cat_dims:
            logger.warning(
                "LassoBO does not support categorical dims; ignoring cat_dims=%s",
                cat_dims,
            )
        logger.info("Using LassoBO (L1-regularized MAP, alpha=%.3f)", lassobo_alpha)
        model = _fit_lassobo(
            train_X, train_Y,
            lasso_alpha=lassobo_alpha,
        )
    elif cat_dims:
        # Mixed continuous+categorical GP (timing window encoding)
        from botorch.models import MixedSingleTaskGP
        logger.info(
            "Using MixedSingleTaskGP with %d categorical dims at indices %s",
            len(cat_dims), cat_dims,
        )
        model = MixedSingleTaskGP(
            train_X,
            train_Y,
            cat_dims=cat_dims,
            input_transform=Normalize(
                d=train_X.shape[1],
                indices=[i for i in range(train_X.shape[1]) if i not in cat_dims],
            ),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )
        # Warm-start if requested
        if warm_start:
            load_gp_state(model, gp_state_path)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    elif per_type_gp and train_Y.shape[1] > 1:
        # Per-cell-type GP models (GPerturb, Xing & Yau 2025):
        # Fit separate SingleTaskGP per output → ModelListGP.
        # Each sub-model has its own ARD lengthscales for interpretability.
        from botorch.models import ModelListGP
        n_outputs = train_Y.shape[1]
        logger.info("Using per-type GP (ModelListGP, %d outputs, MAP)", n_outputs)
        per_type_models = []
        for i in range(n_outputs):
            yvar_i = train_Yvar[:, i:i+1] if train_Yvar is not None else None
            m = SingleTaskGP(
                train_X,
                train_Y[:, i:i+1],
                train_Yvar=yvar_i,
                input_transform=Normalize(d=train_X.shape[1]),
                outcome_transform=Standardize(m=1),
            )
            _set_dim_scaled_lengthscale_prior(m, train_X.shape[1])
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_mll(mll)
            per_type_models.append(m)
            logger.info("  Fitted per-type GP %d/%d", i + 1, n_outputs)
        model = ModelListGP(*per_type_models)
    else:
        # Build kernel based on kernel_type
        covar_module = None
        if kernel_type == "additive_interaction":
            covar_module = _build_additive_interaction_kernel(train_X.shape[1])
            logger.info("Using single-task GP with additive+interaction kernel")
        elif kernel_type == "shared":
            from gpytorch.kernels import MaternKernel, ScaleKernel
            covar_module = ScaleKernel(MaternKernel(nu=2.5))
            logger.info("Using single-task GP with shared lengthscale (sparse-data mode)")
        else:
            logger.info("Using single-task GP with per-dim ARD")

        model = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )
        if covar_module is None:
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
    if save_state:
        save_path = GP_STATE_DIR / f"round_{round_num}.pt"
        save_gp_state(model, save_path)

    # Report kernel parameters
    logger.info("GP Kernel Parameters:")
    lengthscales = _extract_lengthscales(model, train_X.shape[1])
    if lengthscales is not None:
        importance = 1.0 / np.maximum(lengthscales, 1e-6)
        morph_importance = pd.Series(importance[:len(X.columns)], index=X.columns)
        morph_importance = morph_importance.sort_values(ascending=False)
        logger.info("Morphogen importance (1/lengthscale):")
        for morph, imp in morph_importance.head(8).items():
            logger.info("  %s: %.4f", morph, imp)
    else:
        logger.info("(lengthscales not directly accessible for this model type)")

    # Per-output lengthscale matrix for per-type GP interpretability
    if per_type_gp and hasattr(model, 'models'):
        ls_matrix = _extract_per_output_lengthscales(model, train_X.shape[1])
        if ls_matrix is not None:
            # Label columns as ILR components (not cell types) since each
            # ILR component is a linear combination of all cell types
            ilr_labels = [f"ILR_{j+1}" for j in range(ls_matrix.shape[1])]
            ls_df = pd.DataFrame(
                ls_matrix,
                index=X.columns[:ls_matrix.shape[0]] if ls_matrix.shape[0] <= len(X.columns) else None,
                columns=ilr_labels,
            )
            logger.info("Per-output lengthscale matrix (morphogen x ILR component):")
            # Log top-3 most sensitive morphogens per ILR component
            for comp in ls_df.columns:
                sensitivity = (1.0 / ls_df[comp].clip(lower=1e-6)).sort_values(ascending=False)
                top3 = ", ".join(f"{m}={v:.3f}" for m, v in sensitivity.head(3).items())
                logger.info("  %s: %s", comp, top3)

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
            # Fix fidelity to remapped highest-fidelity value (TODO-24).
            remapped_hf = FIDELITY_KERNEL_REMAP.get(1.0, 1.0)
            lo, hi = remapped_hf, remapped_hf
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
        # Guard: SAASBO ModelListGP with scalarized qLogEI is unsupported —
        # independent MCMC samples per sub-model cause batch dimension mismatch
        if (hasattr(model, 'models') and train_Y.shape[1] > 1
                and any(hasattr(m, 'median_lengthscale') for m in model.models)):
            raise NotImplementedError(
                "SAASBO (ModelListGP of fully Bayesian sub-models) is not compatible "
                "with scalarized qLogEI. Use --multi-objective to route through "
                "qLogNEHVI, or disable --saasbo for multi-output optimization."
            )
        # For multi-output models, use a scalarization posterior transform
        if train_Y.shape[1] > 1:
            from botorch.acquisition.objective import GenericMCObjective

            if use_ilr and n_composition_parts > 1:
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
    if use_ilr and n_composition_parts > 1 and cell_type_cols is not None:
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

    Decision gate (per McDonald et al. 2025, thresholds from config):
        - correlation > FIDELITY_SKIP_MFBO_THRESHOLD: skip MF-BO, use cheap fidelity as pre-filter
        - correlation < FIDELITY_CORRELATION_THRESHOLD: skip MF-BO, use single-fidelity GP on real data only
        - in between: MF-BO is appropriate
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
    elif overall_corr > FIDELITY_SKIP_MFBO_THRESHOLD:
        recommendation = "skip_mfbo_use_cheap"
        details = (
            f"Cross-fidelity correlation with {fidelity_label} is very high "
            f"({overall_corr:.3f} > {FIDELITY_SKIP_MFBO_THRESHOLD}). MF-BO adds overhead without benefit — "
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
            f"({overall_corr:.3f}), within [{FIDELITY_CORRELATION_THRESHOLD}, {FIDELITY_SKIP_MFBO_THRESHOLD}]. MF-BO is appropriate."
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


def _append_round_to_csv(
    history_path: Path,
    new_df: pd.DataFrame,
    round_num: int,
    sort_keys: list[str],
) -> pd.DataFrame:
    """Append rows for a round to a persistent CSV, replacing any existing rows
    for the same round (idempotent re-runs)."""
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = history[history["round"] != round_num]
        history = pd.concat([history, new_df], ignore_index=True)
    else:
        history = new_df
    history = history.sort_values(sort_keys).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    return history


def monitor_fidelity_per_round(
    val_results: dict[str, dict],
    round_num: int,
    history_path: Path | None = None,
    degradation_window: int = 2,
) -> dict:
    """Track cross-fidelity correlation across rounds and detect degradation.

    Appends the current round's validation results to a persistent CSV and
    checks for correlation degradation over consecutive rounds.  If correlation
    declines for ``degradation_window`` consecutive rounds, recommends
    auto-fallback to single-fidelity GP.

    Args:
        val_results: Dict mapping fidelity label (e.g. "CellRank2") to the
            dict returned by ``validate_fidelity_correlation()``.
        round_num: Current optimization round number.
        history_path: Path for the persistent monitoring CSV.
            Default: ``DATA_DIR / "fidelity_monitoring.csv"``.
        degradation_window: Number of consecutive declining rounds before
            recommending fallback. (default: 2)

    Returns:
        Dict with:
            - history: DataFrame of all rounds' monitoring data
            - degraded_sources: list of fidelity labels with sustained decline
            - auto_fallback: bool — True if any source should be dropped
    """
    if history_path is None:
        history_path = DATA_DIR / "fidelity_monitoring.csv"

    # Build rows for this round
    rows = []
    for label, res in val_results.items():
        rows.append({
            "round": round_num,
            "fidelity_label": label,
            "overall_correlation": res.get("overall_correlation", float("nan")),
            "recommendation": res.get("recommendation", "unknown"),
            "n_overlap": res.get("n_overlap", 0),
        })

    new_df = pd.DataFrame(rows)

    history = _append_round_to_csv(
        history_path, new_df, round_num,
        sort_keys=["fidelity_label", "round"],
    )
    logger.info("Fidelity monitoring history saved to %s (%d rows)", history_path, len(history))

    # Detect degradation: correlation declining for N consecutive rounds
    degraded_sources = []
    for label in history["fidelity_label"].unique():
        label_hist = history[history["fidelity_label"] == label]
        corrs = label_hist["overall_correlation"].values

        if len(corrs) < degradation_window + 1:
            # Not enough history to detect a trend
            continue

        # Check last `degradation_window` deltas
        recent = corrs[-(degradation_window + 1):]
        deltas = np.diff(recent)
        if np.all(deltas < 0) and not np.any(np.isnan(recent)):
            degraded_sources.append(label)
            logger.warning(
                "Fidelity degradation detected for %s: "
                "correlation declined for %d consecutive rounds (%.3f → %.3f)",
                label, degradation_window,
                float(recent[0]), float(recent[-1]),
            )

    auto_fallback = len(degraded_sources) > 0
    if auto_fallback:
        logger.warning(
            "Auto-fallback to single-fidelity recommended for: %s",
            degraded_sources,
        )

    return {
        "history": history,
        "degraded_sources": degraded_sources,
        "auto_fallback": auto_fallback,
    }


def compute_convergence_diagnostics(
    model,
    train_X: torch.Tensor,
    recommendations: pd.DataFrame,
    bounds_tensor: torch.Tensor,
    columns: list[str],
    round_num: int = 1,
    history_path: Path | None = None,
    n_eval_points: int = CONVERGENCE_POSTERIOR_EVAL_POINTS,
) -> dict:
    """Compute convergence diagnostics for the GP-BO loop.

    Tracks three signals of convergence (Narayanan et al. 2025):
    1. **Mean posterior std**: average posterior standard deviation across
       the design space — drops as the GP gains coverage.
    2. **Max acquisition value**: the peak acquisition function value for
       the recommended batch — decays as the GP believes less improvement
       remains.
    3. **Recommendation spread**: mean pairwise L2 distance between
       recommended morphogen vectors (normalised to [0,1]) — shrinks as
       the optimizer converges on a region.
    4. **Adaptive batch suggestion**: when the three metrics indicate
       convergence, suggests reducing the batch size.

    Results are appended to a persistent CSV so trends can be plotted
    across rounds.

    Args:
        model: Fitted BoTorch GP model.
        train_X: Training X tensor (used for dimensionality and size).
        recommendations: DataFrame from ``recommend_next_experiments``.
        bounds_tensor: (2, d) bounds tensor used for normalisation.
        columns: Column names for morphogen dimensions.
        round_num: Current round number.
        history_path: Persistent CSV path.
            Default: ``DATA_DIR / "convergence_diagnostics.csv"``.
        n_eval_points: Number of Sobol points for posterior variance
            estimation.

    Returns:
        Dict with convergence metrics and adaptive batch suggestion.
    """
    if history_path is None:
        history_path = DATA_DIR / "convergence_diagnostics.csv"

    # --- 1. Mean posterior std across design space ---
    d = train_X.shape[1]
    eval_X = _sobol_eval_points(d, n_eval_points, bounds_tensor[0], bounds_tensor[1])

    with torch.no_grad():
        posterior = model.posterior(eval_X)
        # Mean standard deviation across all eval points and outputs
        post_std = posterior.variance.sqrt().mean().item()

    # --- 2. Max acquisition value from recommendations ---
    max_acq = float("nan")
    if "acquisition_value" in recommendations.columns:
        acq_vals = recommendations["acquisition_value"].dropna()
        if len(acq_vals) > 0:
            max_acq = float(acq_vals.max())

    # --- 3. Recommendation spread (mean pairwise L2 in normalised space) ---
    morph_cols = [c for c in columns if c in recommendations.columns]
    if len(morph_cols) > 0:
        rec_vals = recommendations[morph_cols].dropna().values
        # Normalise to [0,1] using bounds
        bounds_np = bounds_tensor.cpu().numpy()
        span = bounds_np[1] - bounds_np[0]
        span[span < 1e-12] = 1.0  # avoid div-by-zero for fixed dims
        rec_norm = (rec_vals - bounds_np[0][:len(morph_cols)]) / span[:len(morph_cols)]

        if len(rec_norm) > 1:
            from scipy.spatial.distance import pdist
            pairwise = pdist(rec_norm, metric="euclidean")
            rec_spread = float(np.mean(pairwise))
        else:
            rec_spread = 0.0
    else:
        rec_spread = float("nan")

    # --- Build row and append to persistent CSV ---
    row = {
        "round": round_num,
        "mean_posterior_std": post_std,
        "max_acquisition_value": max_acq,
        "recommendation_spread": rec_spread,
        "n_training_points": int(train_X.shape[0]),
    }

    history = _append_round_to_csv(
        history_path, pd.DataFrame([row]), round_num, sort_keys=["round"],
    )
    logger.info(
        "Convergence diagnostics saved to %s — posterior_std=%.4f, "
        "max_acq=%.4f, spread=%.4f",
        history_path, post_std, max_acq, rec_spread,
    )

    # --- 4. Adaptive batch suggestion ---
    # Use absolute values for decay ratio since qLogEI returns log-space
    # (negative) acquisition values.
    suggested_batch = None
    if len(history) >= 2:
        acq_vals = history["max_acquisition_value"].dropna().values
        if len(acq_vals) >= 2 and abs(acq_vals[0]) > 1e-12:
            decay_ratio = abs(acq_vals[-1]) / abs(acq_vals[0])
            spread_vals = history["recommendation_spread"].dropna().values
            low_spread = (
                len(spread_vals) > 0
                and spread_vals[-1] < CONVERGENCE_CLUSTER_SPREAD_THRESHOLD
            )
            low_acq = decay_ratio < CONVERGENCE_ACQUISITION_DECAY_THRESHOLD

            if low_acq and low_spread:
                suggested_batch = max(4, train_X.shape[0] // 10)
                logger.info(
                    "Convergence detected: acquisition decayed to %.1f%% of "
                    "round 1, spread=%.4f. Suggested batch size: %d",
                    decay_ratio * 100, spread_vals[-1], suggested_batch,
                )

    return {
        "mean_posterior_std": post_std,
        "max_acquisition_value": max_acq,
        "recommendation_spread": rec_spread,
        "suggested_batch_size": suggested_batch,
    }


def compute_ensemble_disagreement(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_restarts: int = ENSEMBLE_DEFAULT_N_RESTARTS,
    use_ilr: bool = True,
    kernel_type: Literal["ard", "additive_interaction", "shared"] = "ard",
    cat_dims: Optional[list[int]] = None,
    per_type_gp: bool = False,
    n_eval_points: int = 128,
    existing_model: Optional[Any] = None,
) -> dict:
    """Compute ensemble disagreement by fitting N independent GPs.

    Fits *n_restarts* independent GP models from different random seeds and
    compares their posterior predictions at a common set of Sobol evaluation
    points.  High disagreement indicates the GP fit is sensitive to
    initialisation — a sign of identifiability issues (GPerturb, Xing & Yau,
    Nature Communications 2025).

    Metrics:
        - **stability_score**: mean pairwise cosine similarity of posterior
          mean vectors across restarts (1.0 = perfect agreement, 0.0 = none).
        - **lengthscale_agreement**: Kendall's tau on lengthscale importance
          rankings across restarts.
        - **mean_pred_std_across_models**: average standard deviation of
          posterior means across the ensemble at evaluation points.

    Args:
        X: Training morphogen matrix (may include fidelity column).
        Y: Training cell-type fraction matrix.
        n_restarts: Number of independent GP fits (default from config).
        use_ilr: Whether ILR transform is applied.
        kernel_type: Kernel structure for each GP.
        cat_dims: Categorical column indices (for MixedSingleTaskGP).
        per_type_gp: Fit per-output ModelListGP.
        n_eval_points: Sobol points for posterior comparison.
        existing_model: Pre-fitted model to include as ensemble member #0,
            avoiding one redundant fit. Pass the model from ``run_gpbo_loop``.

    Returns:
        Dict with stability_score, lengthscale_agreement,
        mean_pred_std_across_models, and is_stable flag.
    """
    from itertools import combinations

    from scipy.stats import kendalltau

    if n_restarts < 2:
        return {
            "stability_score": 1.0,
            "lengthscale_agreement": 1.0,
            "mean_pred_std_across_models": 0.0,
            "is_stable": True,
            "n_restarts": n_restarts,
        }

    # Generate common evaluation points in the input space
    d = X.shape[1]
    X_tensor = torch.tensor(X.values, dtype=DTYPE, device=DEVICE)
    x_min = X_tensor.min(dim=0).values
    x_max = X_tensor.max(dim=0).values
    span = x_max - x_min
    span[span < 1e-12] = 1.0
    eval_X = _sobol_eval_points(d, n_eval_points, x_min, x_min + span)

    # Collect predictions and lengthscales from ensemble members
    pred_means = []
    lengthscales_list = []

    def _collect_from_model(model_i):
        with torch.no_grad():
            posterior = model_i.posterior(eval_X)
            pred_means.append(posterior.mean.cpu().numpy())
        ls = _extract_lengthscales(model_i, d)
        if ls is not None:
            lengthscales_list.append(ls)

    # Reuse existing model as ensemble member #0
    if existing_model is not None:
        try:
            _collect_from_model(existing_model)
        except (RuntimeError, ValueError) as e:
            logger.warning("Existing model prediction failed: %s", e)

    # Save/restore RNG state to avoid corrupting downstream randomness
    torch_rng_state = torch.random.get_rng_state()
    np_rng_state = np.random.get_state()

    fits_needed = n_restarts - len(pred_means)
    try:
        for i in range(fits_needed):
            torch.manual_seed(42 + i * 7)
            np.random.seed(42 + i * 7)
            try:
                model_i, _, _, _ = fit_gp_botorch(
                    X, Y,
                    use_ilr=use_ilr,
                    kernel_type=kernel_type,
                    cat_dims=cat_dims,
                    per_type_gp=per_type_gp,
                    round_num=1,
                    save_state=False,
                )
                _collect_from_model(model_i)
            except (RuntimeError, ValueError) as e:
                logger.warning("Ensemble restart %d failed: %s", i, e)
                continue
    finally:
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

    if len(pred_means) < 2:
        logger.warning(
            "Only %d successful ensemble restarts; cannot compute disagreement",
            len(pred_means),
        )
        return {
            "stability_score": float("nan"),
            "lengthscale_agreement": float("nan"),
            "mean_pred_std_across_models": float("nan"),
            "is_stable": True,  # can't tell — assume stable
            "n_restarts": len(pred_means),
        }

    # --- Stability score: mean pairwise cosine similarity of pred vectors ---
    flat_preds = [m.flatten() for m in pred_means]
    cos_sims = []
    for a, b in combinations(flat_preds, 2):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm > 1e-12:
            cos_sims.append(np.dot(a, b) / norm)
    stability_score = float(np.clip(np.mean(cos_sims), 0.0, 1.0)) if cos_sims else 1.0

    # --- Lengthscale agreement: Kendall's tau on importance rankings ---
    ls_agreement = float("nan")
    if len(lengthscales_list) >= 2:
        rankings = [np.argsort(ls) for ls in lengthscales_list]
        taus = []
        for r_a, r_b in combinations(rankings, 2):
            tau, _ = kendalltau(r_a, r_b)
            if not np.isnan(tau):
                taus.append(tau)
        if taus:
            ls_agreement = float(np.mean(taus))

    # --- Mean prediction std across models at eval points ---
    stacked = np.stack(pred_means, axis=0)  # (n_models, n_eval, n_outputs)
    mean_pred_std = float(stacked.std(axis=0).mean())

    n_models = len(pred_means)
    is_stable = stability_score >= ENSEMBLE_STABILITY_LOW_THRESHOLD

    if not is_stable:
        logger.warning(
            "Ensemble disagreement detected: stability=%.3f (threshold=%.2f). "
            "Recommendations may be sensitive to random initialisation. "
            "Consider: (a) collecting more data, (b) using SAASBO for better "
            "uncertainty calibration, (c) increasing exploration.",
            stability_score, ENSEMBLE_STABILITY_LOW_THRESHOLD,
        )

    logger.info(
        "Ensemble disagreement (%d restarts): stability=%.3f, "
        "ls_agreement=%.3f, pred_std=%.4f, stable=%s",
        n_models, stability_score, ls_agreement, mean_pred_std, is_stable,
    )

    return {
        "stability_score": stability_score,
        "lengthscale_agreement": ls_agreement,
        "mean_pred_std_across_models": mean_pred_std,
        "is_stable": is_stable,
        "n_restarts": n_models,
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
                        keep_all = keep_conditions.union(extra)
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
        label = FIDELITY_LABELS.get(fid, f"fid={fid}")
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

    # Only select replicates from real-fidelity conditions (physically executable)
    if "fidelity" in train_X.columns:
        real_mask = train_X["fidelity"] == 1.0
        train_X = train_X.loc[real_mask].copy()
        train_Y = train_Y.loc[real_mask].copy()

    n_available = len(train_X)
    n_replicates = min(n_replicates, n_available)

    morph_cols = [c for c in train_X.columns if c != "fidelity"]

    if strategy == "high_variance" and model is not None and active_cols is not None:
        # Strip fidelity from active_cols for TVR models (sub-GPs trained without fidelity)
        if hasattr(model, '_fidelity_levels'):  # TVRModelEnsemble
            cols = [c for c in active_cols if c != "fidelity"]
        else:
            cols = active_cols
        X_active = train_X[cols].copy()
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
    use_lassobo: bool = False,
    lassobo_alpha: float = 0.1,
    use_tvr: bool = False,
    target_profile: Optional[pd.Series] = None,
    n_replicates: int = 2,
    replicate_strategy: str = "high_variance",
    warm_start: bool = False,
    refine_target: bool = False,
    refine_lr: float = 0.3,
    kernel_type: Literal["ard", "additive_interaction", "shared"] = "ard",
    multi_objective: bool = False,
    n_duplicates: int = 0,
    adaptive_complexity: bool = False,
    timing_windows: bool = False,
    per_type_gp: bool = False,
    ensemble_restarts: int = 0,
    noise_variance_csv: Optional[Path] = None,
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
        use_lassobo: Use LassoBO (L1-regularized MAP variable selection).
            Much faster than SAASBO. Mutually exclusive with use_saasbo.
        lassobo_alpha: L1 penalty strength for LassoBO (default: 0.1).
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
        kernel_type: Kernel structure for the GP. ``"ard"`` (default) uses
            Matern 5/2 + ARD. ``"additive_interaction"`` uses additive +
            interaction decomposition (NAIAD, Qin et al. ICML 2025).
            ``"shared"`` uses Matern 5/2 with shared lengthscale.
        multi_objective: If True, use multi-objective acquisition
            (qLogNoisyExpectedHypervolumeImprovement) instead of scalarized
            qLogExpectedImprovement. (default: False)
        n_duplicates: Number of QC duplicate plate positions for within-round
            noise estimation. The top-scoring new conditions are duplicated.
            Set to 0 to disable. (default: 0)
        adaptive_complexity: If True, auto-select kernel complexity based on
            the N/d ratio (NAIAD, Qin et al. ICML 2025). Overrides
            ``kernel_type`` and ``use_saasbo``. (default: False)
        timing_windows: If True, append categorical timing window columns
            to the training data and use MixedSingleTaskGP for mixed
            continuous+categorical inputs (Sanchis-Calleja et al. 2025).
        per_type_gp: If True, fit separate GP per cell type (or ILR
            component) using ModelListGP instead of a single multi-output GP.
            Produces per-output lengthscale matrix for interpretability
            (GPerturb, Xing & Yau 2025). Only applies to the standard MAP
            path (not SAASBO, multi-fidelity, or TVR). (default: False)
        ensemble_restarts: Number of independent GP restarts for ensemble
            disagreement diagnostic (GPerturb, Xing & Yau 2025). When > 1,
            fits N models from different seeds and reports a stability score.
            Set to 0 to skip. (default: 0)
        noise_variance_csv: Path to per-condition bootstrap variance CSV
            (from ``compute_bootstrap_uncertainty``). When provided, uses
            heteroscedastic noise (``train_Yvar``) in the GP. Only applies to
            the standard MAP and per-type-GP paths.

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
        round_val_results: dict[str, dict] = {}
        for v_frac_csv, v_morph_csv, v_fidelity in virtual_sources:
            if not v_frac_csv.exists():
                continue
            # Only validate non-real fidelity sources
            if v_fidelity < 1.0:
                v_Y = pd.read_csv(str(v_frac_csv), index_col=0)
                fid_label = FIDELITY_LABELS.get(
                    v_fidelity, f"fidelity={v_fidelity}"
                )
                val_result = validate_fidelity_correlation(
                    Y_real, v_Y, fidelity_label=fid_label,
                )
                round_val_results[fid_label] = val_result
                if val_result["recommendation"] == "single_fidelity":
                    logger.warning(
                        "Dropping %s data: cross-fidelity correlation too low (%.3f)",
                        fid_label, val_result["overall_correlation"],
                    )
                    continue
            validated_virtual.append((v_frac_csv, v_morph_csv, v_fidelity))

        # Per-round fidelity monitoring: track trend and detect degradation
        if round_val_results:
            monitor_result = monitor_fidelity_per_round(
                round_val_results, round_num=round_num,
            )
            # Auto-fallback: drop sources with sustained correlation decline
            if monitor_result["auto_fallback"]:
                degraded = set(monitor_result["degraded_sources"])
                before = len(validated_virtual)
                validated_virtual = [
                    (vf, vm, vfid) for vf, vm, vfid in validated_virtual
                    if FIDELITY_LABELS.get(
                        vfid, f"fidelity={vfid}"
                    ) not in degraded
                ]
                dropped = before - len(validated_virtual)
                if dropped > 0:
                    logger.warning(
                        "Fidelity monitoring auto-fallback: dropped %d degraded "
                        "source(s): %s", dropped, list(degraded),
                    )

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

    # Append categorical timing window columns if requested
    timing_cat_dims = None
    if timing_windows:
        from gopro.morphogen_parser import compute_timing_windows
        tw_df = compute_timing_windows(list(X_active.index))
        # Only keep timing columns with >1 unique value (drop constant ones)
        tw_active = tw_df.loc[:, tw_df.nunique() > 1]
        if tw_active.shape[1] > 0:
            X_active = X_active.copy()
            for col in tw_active.columns:
                X_active[col] = tw_active[col].values
                # Add bounds for categorical columns (0..TIMING_FULL)
                active_bounds[col] = (0.0, float(TIMING_FULL))
            # Record indices of categorical columns for MixedSingleTaskGP
            timing_cat_dims = [
                X_active.columns.get_loc(col) for col in tw_active.columns
            ]
            logger.info(
                "Timing windows: added %d categorical dims %s at indices %s",
                len(timing_cat_dims), list(tw_active.columns), timing_cat_dims,
            )
        else:
            logger.warning("Timing windows: all timing columns are constant; skipping")

    # Adaptive complexity: auto-select kernel based on N/d ratio
    d_morph = len([c for c in active_cols if c != "fidelity"])
    kernel_spec = _resolve_kernel_spec(
        kernel_type, use_saasbo, adaptive_complexity,
        n_conditions=X_active.shape[0], d_active=d_morph,
    )

    # Load bootstrap noise variance if provided
    noise_var_df = None
    if noise_variance_csv is not None:
        nv_path = Path(noise_variance_csv)
        try:
            noise_var_df = pd.read_csv(nv_path, index_col=0)
            # Align to Y's index (conditions in merged training set)
            noise_var_df = noise_var_df.reindex(Y.index)
            n_matched = noise_var_df.dropna(how="all").shape[0]
            logger.info(
                "Loaded bootstrap noise variance: %d/%d conditions matched",
                n_matched, len(Y),
            )
            if n_matched < len(Y) * 0.5:
                logger.warning(
                    "Low noise variance match rate (%d/%d). "
                    "Unmatched conditions will use column-mean variance.",
                    n_matched, len(Y),
                )
            # Fill unmatched conditions (e.g. virtual data) with column means
            noise_var_df = noise_var_df.fillna(noise_var_df.mean())
        except FileNotFoundError:
            logger.warning("Noise variance CSV not found: %s", nv_path)

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
            use_saasbo=kernel_spec.use_saasbo,
            use_lassobo=use_lassobo,
            lassobo_alpha=lassobo_alpha,
            warm_start=warm_start,
            round_num=round_num,
            kernel_type=kernel_spec.kernel_type,
            cat_dims=timing_cat_dims,
            per_type_gp=per_type_gp,
            noise_variance=noise_var_df,
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
    if n_duplicates >= n_novel:
        raise ValueError(
            f"Not enough wells for novel candidates: n_recommendations={n_recommendations}, "
            f"n_replicates={n_replicates}, n_duplicates={n_duplicates}. "
            f"After reserving replicates, only {n_novel} wells remain but "
            f"{n_duplicates} are needed for QC duplicates. "
            f"Increase --n-recommendations or decrease --n-replicates/--n-duplicates."
        )
    recommendations = recommend_next_experiments(
        model, train_X, train_Y,
        bounds=rec_bounds,
        columns=rec_cols,
        n_recommendations=n_novel,
        use_multi_objective=multi_objective,
        use_ilr=use_ilr,
        n_composition_parts=len(cell_type_cols),
        cell_type_cols=cell_type_cols,
        target_profile=target_profile,
        n_duplicates=n_duplicates,
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
        if timing_cat_dims:
            diagnostics["model_type"] = "mixed_categorical"
        elif kernel_spec.use_saasbo:
            diagnostics["model_type"] = "saasbo"
        elif use_lassobo:
            diagnostics["model_type"] = "lassobo"
        else:
            diagnostics["model_type"] = "map"
    except (AttributeError, RuntimeError):
        pass

    # Compute convergence diagnostics
    try:
        bounds_lower = [rec_bounds.get(c, (0.0, 1.0))[0] for c in rec_cols]
        bounds_upper = [rec_bounds.get(c, (0.0, 1.0))[1] for c in rec_cols]
        conv_bounds = torch.tensor(
            [bounds_lower, bounds_upper], dtype=DTYPE, device=DEVICE
        )
        conv_diag = compute_convergence_diagnostics(
            model=model,
            train_X=train_X,
            recommendations=recommendations,
            bounds_tensor=conv_bounds,
            columns=rec_cols,
            round_num=round_num,
        )
        diagnostics["mean_posterior_std"] = conv_diag["mean_posterior_std"]
        diagnostics["max_acquisition_value"] = conv_diag["max_acquisition_value"]
        diagnostics["recommendation_spread"] = conv_diag["recommendation_spread"]
        if conv_diag["suggested_batch_size"] is not None:
            diagnostics["suggested_batch_size"] = conv_diag["suggested_batch_size"]
    except (RuntimeError, ValueError) as e:
        logger.warning("Convergence diagnostics failed: %s", e, exc_info=True)

    # Ensemble disagreement diagnostic (GPerturb, Xing & Yau 2025)
    if ensemble_restarts >= 2:
        try:
            ens_diag = compute_ensemble_disagreement(
                X_active, Y,
                n_restarts=ensemble_restarts,
                use_ilr=use_ilr,
                kernel_type=kernel_spec.kernel_type,
                cat_dims=timing_cat_dims,
                per_type_gp=per_type_gp,
                existing_model=model,
            )
            diagnostics["ensemble_stability_score"] = ens_diag["stability_score"]
            diagnostics["ensemble_ls_agreement"] = ens_diag["lengthscale_agreement"]
            diagnostics["ensemble_pred_std"] = ens_diag["mean_pred_std_across_models"]
            diagnostics["ensemble_is_stable"] = ens_diag["is_stable"]
            diagnostics["ensemble_n_restarts"] = ens_diag["n_restarts"]
        except (RuntimeError, ValueError) as e:
            logger.warning("Ensemble disagreement failed: %s", e, exc_info=True)

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
    parser.add_argument("--lassobo", action="store_true",
                        help="Use LassoBO (L1-regularized MAP variable selection, AISTATS 2025). "
                             "Faster alternative to SAASBO for automatic variable selection.")
    parser.add_argument("--lassobo-alpha", type=float, default=0.1,
                        help="L1 penalty strength for LassoBO (default: 0.1). "
                             "Higher → more aggressive pruning.")
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
    parser.add_argument("--kernel", type=str, default="ard",
                        choices=["ard", "additive_interaction", "shared"],
                        help="GP kernel structure: 'ard' (Matern 5/2 + ARD, default), "
                             "'additive_interaction' (NAIAD 2025), or 'shared' (single lengthscale)")
    parser.add_argument("--adaptive-complexity", action="store_true",
                        help="Auto-select kernel complexity based on N/d ratio (NAIAD 2025). "
                             "Overrides --kernel and --saasbo.")
    parser.add_argument("--timing-windows", action="store_true",
                        help="Append categorical timing window columns (early/mid/late) to "
                             "morphogen matrix and use MixedSingleTaskGP (Sanchis-Calleja 2025)")
    parser.add_argument("--per-type-gp", action="store_true",
                        help="Fit separate GP per cell type (ModelListGP) instead of single "
                             "multi-output GP. Produces per-output lengthscale matrix for "
                             "interpretability (GPerturb, Xing & Yau 2025)")
    parser.add_argument("--ensemble-restarts", type=int, default=0,
                        help="Number of independent GP restarts for ensemble disagreement "
                             "diagnostic (GPerturb 2025). Set to 0 to skip. (default: 0)")
    parser.add_argument("--bootstrap-noise", type=str, default=None,
                        help="Path to per-condition bootstrap variance CSV "
                             "(from compute_bootstrap_uncertainty). Enables "
                             "heteroscedastic noise modeling via train_Yvar.")
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
        use_lassobo=args.lassobo,
        lassobo_alpha=args.lassobo_alpha,
        use_tvr=args.tvr,
        target_profile=target_profile,
        n_replicates=args.n_replicates,
        replicate_strategy=args.replicate_strategy,
        warm_start=args.warm_start,
        refine_target=args.refine_target,
        refine_lr=args.refine_lr,
        kernel_type=args.kernel,
        multi_objective=args.multi_objective,
        n_duplicates=args.n_duplicates,
        adaptive_complexity=args.adaptive_complexity,
        timing_windows=args.timing_windows,
        per_type_gp=args.per_type_gp,
        ensemble_restarts=args.ensemble_restarts,
        noise_variance_csv=Path(args.bootstrap_noise) if args.bootstrap_noise else None,
    )

    logger.info("--- NEXT EXPERIMENT RECOMMENDATIONS ---")
    morph_cols = [c for c in MORPHOGEN_COLUMNS if c in recs.columns]
    nonzero_cols = [c for c in morph_cols if recs[c].abs().sum() > 0.01]
    pred_cols = [c for c in recs.columns if "predicted" in c or "acquisition" in c]
    logger.info("\n%s", recs[nonzero_cols + pred_cols].to_string())
    logger.info("Give this plate map to the wet lab team!")
