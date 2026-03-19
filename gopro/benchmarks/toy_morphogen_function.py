"""Domain-informed synthetic test function for GP-BO benchmarking.

Simulates a 24-dimensional morphogen response surface that outputs
compositional cell type fractions (simplex), suitable for testing
the full ILR-transform-to-GP pipeline.

Each morphogen's effect is modeled as a Hill function, with pathway
interactions creating combinatorial effects on cell type proportions.
"""

from __future__ import annotations
import numpy as np
from gopro.config import MORPHOGEN_COLUMNS, get_logger

logger = get_logger(__name__)

# Simplified pathway effects: which morphogens push which cell types
# Based on known developmental biology (Amin & Kelley 2024)
CELL_TYPES = ["Neuron", "NPC", "IP", "Neuroepithelium", "CP", "Glia"]

# Sparse pathway effect matrix: (n_morphogens, n_cell_types)
# Rows correspond to MORPHOGEN_COLUMNS, columns to CELL_TYPES.
# Nonzero entries encode which morphogens affect which cell types.
# Signs: positive = promotes, negative = inhibits.
# fmt: off
PATHWAY_EFFECTS = np.array([
    # Neuron  NPC     IP    Neuroepi  CP    Glia
    [ 0.0,   1.5,   0.0,   2.0,    0.0,   0.0],   # CHIR99021 (WNT agonist)
    [ 0.0,   0.0,   0.0,   0.0,    1.2,   0.0],   # BMP4
    [ 0.0,   0.0,   0.0,   0.0,    1.0,   0.0],   # BMP7
    [ 1.8,   0.0,   0.5,   0.0,    0.0,  -0.5],   # SHH
    [ 1.5,   0.0,   0.3,   0.0,    0.0,  -0.3],   # SAG (Smoothened agonist)
    [ 0.0,   0.0,   0.8,   0.0,    0.0,   1.0],   # RA
    [ 0.0,   0.0,   0.5,   0.0,    0.0,   0.7],   # SR11237
    [ 0.0,   0.5,   0.0,   1.0,    0.0,   0.0],   # FGF8
    [ 0.0,   0.8,   0.0,   0.5,    0.0,   0.3],   # FGF2
    [ 0.0,   0.6,   0.0,   0.4,    0.0,   0.0],   # FGF4
    [ 0.0,  -0.5,   0.0,  -1.0,    0.0,   0.0],   # IWP2 (WNT inhibitor)
    [ 0.0,  -0.5,   0.0,  -1.0,    0.0,   0.0],   # XAV939 (WNT inhibitor)
    [ 1.0,   0.0,   0.0,   0.0,   -0.5,   0.0],   # SB431542 (TGF-beta inhib)
    [ 0.5,   0.0,   0.0,   0.0,   -0.8,   0.0],   # LDN193189 (BMP inhib)
    [ 1.5,  -0.5,   0.0,   0.0,    0.0,   0.0],   # DAPT (Notch inhib)
    [ 0.0,   0.0,   0.0,   0.0,    0.0,   1.2],   # EGF
    [ 0.0,   0.0,   0.0,   0.0,    0.5,   0.0],   # ActivinA
    [ 0.3,   0.0,   0.0,   0.0,   -0.5,   0.0],   # Dorsomorphin
    [ 1.2,   0.0,   0.2,   0.0,    0.0,  -0.2],   # purmorphamine
    [-0.8,   0.0,  -0.2,   0.0,    0.0,   0.2],   # cyclopamine
    [ 0.5,  -0.3,   0.0,   0.0,    0.0,   0.0],   # log_harvest_day
    [ 0.3,   0.0,   0.0,   0.0,    0.0,   0.1],   # BDNF
    [ 0.2,   0.0,   0.0,   0.0,    0.0,   0.1],   # NT3
    [ 0.1,   0.0,   0.0,   0.0,    0.0,   0.0],   # cAMP
    [ 0.0,   0.1,   0.0,   0.0,    0.0,   0.0],   # AscorbicAcid
], dtype=np.float64)
# fmt: on

# Default EC50 ranges per morphogen (in uM) for realistic dose-response
# Small molecules: 0.01-10 uM; recombinant proteins: 0.0001-0.01 uM
_DEFAULT_EC50_RANGES = {
    "CHIR99021_uM": (0.5, 5.0),
    "BMP4_uM": (0.0001, 0.005),
    "BMP7_uM": (0.0001, 0.005),
    "SHH_uM": (0.001, 0.05),
    "SAG_uM": (0.01, 1.0),
    "RA_uM": (0.01, 1.0),
    "SR11237_uM": (0.01, 1.0),
    "FGF8_uM": (0.0001, 0.01),
    "FGF2_uM": (0.0001, 0.01),
    "FGF4_uM": (0.0001, 0.01),
    "IWP2_uM": (0.1, 5.0),
    "XAV939_uM": (0.1, 5.0),
    "SB431542_uM": (0.5, 20.0),
    "LDN193189_uM": (0.01, 1.0),
    "DAPT_uM": (0.5, 10.0),
    "EGF_uM": (0.0001, 0.01),
    "ActivinA_uM": (0.0001, 0.01),
    "Dorsomorphin_uM": (0.1, 5.0),
    "purmorphamine_uM": (0.1, 5.0),
    "cyclopamine_uM": (0.1, 5.0),
    "log_harvest_day": (3.0, 5.0),
    "BDNF_uM": (0.0001, 0.01),
    "NT3_uM": (0.0001, 0.01),
    "cAMP_uM": (10.0, 500.0),
    "AscorbicAcid_uM": (10.0, 500.0),
}


def hill_response(concentration: float, ec50: float, hill_n: float = 2.0) -> float:
    """Standard Hill equation dose-response."""
    if concentration <= 0 or ec50 <= 0:
        return 0.0
    return concentration**hill_n / (ec50**hill_n + concentration**hill_n)


class ToyMorphogenFunction:
    """Synthetic morphogen response function outputting cell type fractions.

    Args:
        seed: Random seed for reproducible parameter initialization.
        n_cell_types: Number of cell types in the output simplex.
        noise_std: Additive Gaussian noise (in logit space) per evaluation.
    """

    def __init__(self, seed: int = 42, n_cell_types: int = 6, noise_std: float = 0.0):
        self.n_cell_types = n_cell_types
        self.noise_std = noise_std
        self.n_morphogens = len(MORPHOGEN_COLUMNS)
        self.rng = np.random.RandomState(seed)

        # Initialize EC50 values from realistic ranges
        self.ec50 = np.zeros(self.n_morphogens)
        for i, col in enumerate(MORPHOGEN_COLUMNS):
            lo, hi = _DEFAULT_EC50_RANGES.get(col, (0.01, 1.0))
            self.ec50[i] = self.rng.uniform(lo, hi)

        # Use the predefined pathway effects (truncated to n_cell_types)
        self.pathway_effects = PATHWAY_EFFECTS[:, :n_cell_types].copy()

        # Precompute the optimum
        self._optimum = self._compute_optimum()

        logger.debug(
            "ToyMorphogenFunction initialized: %d morphogens, %d cell types, noise=%.3f",
            self.n_morphogens,
            self.n_cell_types,
            self.noise_std,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function at morphogen concentrations.

        Args:
            x: Array of shape (n_points, n_morphogens) -- morphogen concentrations
                where n_morphogens = len(MORPHOGEN_COLUMNS).

        Returns:
            Array of shape (n_points, n_cell_types) -- cell type fractions
            summing to 1.0 per row.
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.n_morphogens:
            raise ValueError(
                f"Expected {self.n_morphogens} morphogen dimensions, got {x.shape[1]}"
            )

        n_points = x.shape[0]

        # Vectorized Hill response: h(c, ec50) = c^n / (ec50^n + c^n)
        # (scalar hill_response() function retained for external callers)
        ec50 = self.ec50.reshape(1, -1)
        safe_x = np.maximum(x, 0.0)
        safe_ec50 = np.maximum(ec50, 1e-20)
        hill_responses = safe_x**2 / (safe_ec50**2 + safe_x**2)
        # Zero out where input concentration is zero
        hill_responses[x <= 0] = 0.0

        # Logit-space cell type scores: (n_points, n_cell_types)
        logits = hill_responses @ self.pathway_effects

        # Add noise in logit space
        if self.noise_std > 0:
            logits += self.rng.normal(0, self.noise_std, logits.shape)

        # Softmax to convert to simplex
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        fractions = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return fractions

    def _compute_optimum(self) -> np.ndarray:
        """Compute the morphogen vector that maximizes the first cell type (Neuron).

        For each morphogen with a positive effect on Neuron, set concentration
        to EC50 (Hill response = 0.5, giving substantial signal). For morphogens
        with negative effect, set to 0. For morphogens with zero effect on
        Neuron, set to 0 to avoid boosting competing cell types.
        """
        opt = np.zeros(self.n_morphogens)
        neuron_effects = self.pathway_effects[:, 0]
        for d in range(self.n_morphogens):
            if neuron_effects[d] > 0:
                # At EC50, Hill response = 0.5 -- good activation
                opt[d] = self.ec50[d]
            # else: leave at 0
        return opt

    @property
    def optimum(self) -> np.ndarray:
        """Known global optimum morphogen concentrations (24D)."""
        return self._optimum.copy()
