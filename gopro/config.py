"""Centralized configuration for the GP-BO pipeline.

This module provides all shared constants, paths, and logging utilities
used across the pipeline. Import from here instead of defining locally.

IMPORTANT: This module must NOT import from any other gopro module
to avoid circular imports.
"""

import logging
import os
from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(os.environ.get("GPBO_PROJECT_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.environ.get("GPBO_DATA_DIR", str(PROJECT_DIR / "data")))
MODEL_DIR = Path(os.environ.get(
    "GPBO_MODEL_DIR",
    str(PROJECT_DIR / "data" / "neural_organoid_atlas" / "supplemental_files" / "scpoli_model_params"),
))
GP_STATE_DIR = Path(os.environ.get("GPBO_GP_STATE_DIR", str(DATA_DIR / "gp_state")))

# --- Morphogen columns (canonical ordering, all concentrations in µM) ---
MORPHOGEN_COLUMNS: list[str] = [
    "CHIR99021_uM",       # 0 - WNT agonist
    "BMP4_uM",            # 1 - BMP signaling
    "BMP7_uM",            # 2 - BMP signaling
    "SHH_uM",             # 3 - Sonic hedgehog
    "SAG_uM",             # 4 - Smoothened agonist
    "RA_uM",              # 5 - Retinoic acid
    "FGF8_uM",            # 6 - FGF8
    "FGF2_uM",            # 7 - FGF2
    "FGF4_uM",            # 8 - FGF4
    "IWP2_uM",            # 9 - WNT inhibitor
    "XAV939_uM",          # 10 - WNT inhibitor
    "SB431542_uM",        # 11 - TGF-beta inhibitor
    "LDN193189_uM",       # 12 - BMP inhibitor
    "DAPT_uM",            # 13 - Notch inhibitor
    "EGF_uM",             # 14 - EGF
    "ActivinA_uM",        # 15 - Activin
    "Dorsomorphin_uM",    # 16 - BMP inhibitor
    "purmorphamine_uM",   # 17 - SHH agonist
    "cyclopamine_uM",     # 18 - SHH antagonist
    "log_harvest_day",    # 19 - Time dimension
    "BDNF_uM",            # 20 - Brain-derived neurotrophic factor
    "NT3_uM",             # 21 - Neurotrophin-3
    "cAMP_uM",            # 22 - Dibutyryl-cAMP
    "AscorbicAcid_uM",   # 23 - L-Ascorbic acid 2-phosphate
]

# --- Molecular weights (kDa) for recombinant protein morphogens ---
# Used to convert ng/mL → µM: µM = (ng/mL) / (MW_kDa × 1000)
PROTEIN_MW_KDA: dict[str, float] = {
    "BMP4":     13.0,   # Mature monomer (recombinant human BMP4)
    "BMP7":     15.7,   # Mature monomer
    "SHH":      19.6,   # N-terminal signaling domain
    "FGF2":     17.2,   # 154-aa isoform (bFGF)
    "FGF4":     19.2,
    "FGF8":     22.5,   # FGF8b isoform
    "EGF":       6.2,
    "ActivinA": 26.0,   # Homodimer
    "BDNF":     13.5,   # Mature monomer (recombinant human BDNF)
    "NT3":      13.6,   # Mature monomer (neurotrophin-3)
}


def ng_mL_to_uM(ng_per_mL: float, mw_kda: float) -> float:
    """Convert ng/mL to µM using molecular weight.

    Formula: µM = (ng/mL) / (MW_Da) = (ng/mL) / (MW_kDa × 1000)
    """
    return ng_per_mL / (mw_kda * 1000.0)


def nM_to_uM(nM: float) -> float:
    """Convert nM to µM."""
    return nM / 1000.0

# --- HNOCA annotation column names ---
ANNOT_LEVEL_1 = "annot_level_1"
ANNOT_LEVEL_2 = "annot_level_2"
ANNOT_REGION = "annot_region_rev2"
ANNOT_LEVEL_3 = "annot_level_3_rev2"

# --- Logging ---
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    level = os.environ.get("GPBO_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger

# --- File hashing ---

def md5_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 hash of a file without loading it all into memory.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size in bytes (default 8 MiB).

    Returns:
        Hex-encoded MD5 digest string.
    """
    import hashlib
    from pathlib import Path as _Path

    path = _Path(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Fidelity R² thresholds (3-zone routing, Sabanza-Gil 2025) ---
# R² measures how well low-fidelity data predicts high-fidelity outcomes.
# Three zones:
#   R² > skip  → fidelities are redundant, skip MF-BO overhead
#   R² < drop  → fidelities too divergent, drop low-fidelity source
#   drop ≤ R² ≤ skip → MF-BO is appropriate
FIDELITY_R2_THRESHOLDS: dict[str, float] = {
    "drop": 0.80,   # Below this R², fall back to single-fidelity GP
    "skip": 0.90,   # Above this R², MF-BO adds no benefit
}

# Explicit aliases for clarity (R²-based, not Spearman)
FIDELITY_DROP_R2_THRESHOLD = FIDELITY_R2_THRESHOLDS["drop"]
FIDELITY_SKIP_R2_THRESHOLD = FIDELITY_R2_THRESHOLDS["skip"]

# Human-readable labels for fidelity levels
FIDELITY_LABELS: dict[float, str] = {
    1.0: "real",
    0.5: "CellRank2",
    0.0: "CellFlow",
}

# --- Cost ratios for fidelity levels (relative to real experiment = 1.0) ---
# Real scRNA-seq: ~$2,000 + 72 days; CellRank2: ~2 hours compute; CellFlow: ~10 min
FIDELITY_COSTS: dict[float, float] = {
    1.0: 1.0,     # real experiment
    0.5: 0.005,   # CellRank2 forward projection
    0.0: 0.001,   # CellFlow generative prediction
}

# --- CellFlow training domain limit ---
# CellFlow (Klein et al. 2025) was trained on days 1-36 only.
# Predictions for harvest days beyond this are out-of-distribution.
CELLFLOW_MAX_TRAINING_DAY: int = 36

# --- CellFlow variance inflation ---
# CellFlow predictions tend to be conservative (clustered near the mean
# composition).  Variance inflation amplifies deviations from the global
# mean before the predictions are fed into the multi-fidelity GP, so the
# GP can learn from more spread-out low-fidelity signal.
CELLFLOW_DEFAULT_VARIANCE_INFLATION: float = 2.0

# --- Adaptive kernel complexity schedule (NAIAD, Qin et al. ICML 2025) ---
# N/d ratio thresholds for auto-selecting GP kernel complexity.
# Below SHARED: use shared lengthscale (fewest params, avoids overfitting).
# Between SHARED and ARD: use per-dim ARD (standard BoTorch default).
# Above ARD: use SAASBO (fully Bayesian with sparsity prior).
KERNEL_COMPLEXITY_THRESHOLDS: dict[str, float] = {
    "shared": 8.0,   # N/d < 8  → shared lengthscale
    "ard": 15.0,     # 8 ≤ N/d < 15 → ARD
                      # N/d ≥ 15 → SAASBO
}

# --- Morphogen timing window encoding (Sanchis-Calleja et al. 2025) ---
# Categorical encoding for when each morphogen is applied during the
# standard patterning window (Day 6-21).
TIMING_NOT_APPLIED = 0   # morphogen not used in this condition
TIMING_EARLY = 1          # Day 6-11 only
TIMING_MID = 2            # Day 11-16 only
TIMING_LATE = 3            # Day 16-21 only
TIMING_FULL = 4            # Full window (Day 6-21)

# Morphogens with observed timing variation in the Amin/Kelley dataset.
# These get categorical timing columns when --timing-windows is enabled.
TIMING_WINDOW_MORPHOGENS: list[str] = ["CHIR99021", "SAG", "BMP4"]

# Column names for timing window categoricals
TIMING_WINDOW_COLUMNS: list[str] = [
    f"{m}_window" for m in TIMING_WINDOW_MORPHOGENS
]

# --- Convergence diagnostics (Narayanan et al. 2025) ---
# Acquisition decay: if max acqf value falls below this fraction of round 1, converging
CONVERGENCE_ACQUISITION_DECAY_THRESHOLD = 0.1
# Recommendation spread: pairwise L2 distance threshold (in normalised morphogen space)
CONVERGENCE_CLUSTER_SPREAD_THRESHOLD = 0.05
# Number of Sobol candidates to evaluate for mean posterior variance
CONVERGENCE_POSTERIOR_EVAL_POINTS = 512

# --- Ensemble disagreement (GPerturb, Xing & Yau 2025) ---
ENSEMBLE_DEFAULT_N_RESTARTS = 5
ENSEMBLE_STABILITY_LOW_THRESHOLD = 0.5  # below this → unstable recommendations

# --- Gruffi stress-filtering defaults ---
GRUFFI_DEFAULT_THRESHOLD = 0.15
GRUFFI_DEFAULT_RESOLUTION = 2.0
GRUFFI_MIN_CELLS_PER_CONDITION = 50
