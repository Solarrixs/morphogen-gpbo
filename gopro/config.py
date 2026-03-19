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
    "RA_uM",              # 5 - Retinoic acid (RAR agonist)
    "SR11237_uM",         # 6 - RXR agonist (NOT equivalent to RA; see morphogen_parser.py)
    "FGF8_uM",            # 7 - FGF8
    "FGF2_uM",            # 8 - FGF2
    "FGF4_uM",            # 9 - FGF4
    "IWP2_uM",            # 10 - WNT inhibitor
    "XAV939_uM",          # 11 - WNT inhibitor
    "SB431542_uM",        # 12 - TGF-beta inhibitor
    "LDN193189_uM",       # 13 - BMP inhibitor
    "DAPT_uM",            # 14 - Notch inhibitor
    "EGF_uM",             # 15 - EGF
    "ActivinA_uM",        # 16 - Activin
    "Dorsomorphin_uM",    # 17 - BMP inhibitor
    "purmorphamine_uM",   # 18 - SHH agonist
    "cyclopamine_uM",     # 19 - SHH antagonist
    "log_harvest_day",    # 20 - Time dimension
    "BDNF_uM",            # 21 - Brain-derived neurotrophic factor
    "NT3_uM",             # 22 - Neurotrophin-3
    "cAMP_uM",            # 23 - Dibutyryl-cAMP
    "AscorbicAcid_uM",   # 24 - L-Ascorbic acid 2-phosphate
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


# --- Fidelity R² thresholds (3-zone routing) ---
# Inspired by Sabanza-Gil et al. 2025 (DOI:10.1038/s43588-025-00822-9),
# who propose R²-based routing between single- and multi-fidelity BO.
# Specific thresholds (0.80/0.90) are pipeline heuristics, not exact
# values from the paper.
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
# CellFlow (Klein et al. 2025, DOI:10.1101/2025.04.11.648220) was trained on
# the Sanchis-Calleja et al. 2025 patterning screen (DOI:10.1038/s41592-025-02927-5)
# covering days 1-36 only. Predictions for harvest days beyond this are
# out-of-distribution temporal extrapolations and may be unreliable.
CELLFLOW_MAX_TRAINING_DAY: int = 36

# --- CellFlow variance inflation ---
# CellFlow predictions tend to be conservative (clustered near the mean
# composition).  Variance inflation amplifies deviations from the global
# mean before the predictions are fed into the multi-fidelity GP, so the
# GP can learn from more spread-out low-fidelity signal.
#
# HEURISTIC: The default factor of 2.0 is an engineering choice, not derived
# from calibration data.  CellFlow's training range (Day 1-36) does not cover
# the pipeline's target harvest days (Day 70-72), making these predictions
# out-of-distribution extrapolations (Klein et al. 2025).  Per MFBO best
# practices (Sabanza-Gil et al. 2025, DOI:10.1038/s43588-025-00822-9), LF
# sources only help when R² > 0.9 and cost ratio ~0.1.  When in doubt, use
# --cellflow-relevance-check to validate CellFlow's contribution via LOO-CV
# (following rMFBO, Mikkola et al. 2023, arXiv:2210.13937).
CELLFLOW_DEFAULT_VARIANCE_INFLATION: float = 2.0

# --- Fixed-noise GP minimum variance (Cosenza 2022) ---
# Minimum per-observation noise variance for FixedNoiseGP / heteroscedastic
# noise modeling.  Clamping at 0.02 prevents the GP from treating any
# observation as noise-free, which would cause numerical instability.
FIXED_NOISE_MIN_VARIANCE: float = 0.02

# --- Adaptive kernel complexity schedule ---
# N/d ratio thresholds for auto-selecting GP kernel complexity.
# ARD with proper priors (Hvarfner et al., ICML 2024, arXiv:2402.02229) is
# feasible even in moderate-data regimes; SAASBO (Eriksson & Jankowiak,
# UAI 2021, arXiv:2103.00349) adds fully Bayesian sparsity for high N/d.
# Below SHARED: use shared lengthscale (fewest params, avoids overfitting).
# Between SHARED and ARD: use per-dim ARD (standard BoTorch default).
# Above ARD: use SAASBO (fully Bayesian with sparsity prior).
KERNEL_COMPLEXITY_THRESHOLDS: dict[str, float] = {
    "shared": 3.0,   # N/d < 3  → shared lengthscale
    "ard": 15.0,     # 3 ≤ N/d < 15 → ARD
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

# --- Temporal bin encoding (alternative to time-fraction scaling) ---
# Instead of concentration * (exposure_days / total_window_days), encode
# each temporally-varying morphogen as 3 separate bins: early/mid/late.
# Each bin holds the FULL concentration if the morphogen was applied during
# that sub-window, or 0 if not.
#
# Literature basis:
#   - Sorre et al. 2014 (DOI:10.1016/j.devcel.2014.05.022): TGF-beta
#     response depends on speed of concentration change, not just AUC
#   - Sasai, Kutejova & Briscoe 2014 (DOI:10.1371/journal.pbio.1001907):
#     competence changes over time create non-interchangeable windows
#   - Amin & Kelley 2024 (DOI:10.1016/j.stem.2024.10.016): found narrow
#     critical timing windows for morphogen action
#   - Dessaud et al. 2007 (DOI:10.1038/nature06347): SHH concentration-
#     duration equivalence (partial, pathway-specific)
#
# The time-fraction encoding collapses temporal and concentration info:
# CHIR at 1.5 uM for days 11-16 becomes 1.5 * 5/15 = 0.5, which is
# indistinguishable from CHIR at 0.5 uM for the full window. The bin
# encoding preserves both: [0, 1.5, 0] vs [0.5, 0.5, 0.5].
TEMPORAL_BIN_EDGES: list[tuple[int, int]] = [
    (6, 11),    # early
    (11, 16),   # mid
    (16, 21),   # late
]
TEMPORAL_BIN_NAMES: list[str] = ["early", "mid", "late"]

# Morphogens with observed timing variation across Amin/Kelley conditions.
# These are eligible for temporal bin expansion. IWP2 also has timing
# variation (CHIR switch IWP2, IWP2 switch CHIR) so it is included.
TEMPORAL_BIN_MORPHOGENS: list[str] = ["CHIR99021", "SAG", "IWP2", "BMP4"]

# Column names for the temporal bin encoding, e.g. CHIR99021_early_uM
TEMPORAL_BIN_COLUMNS: list[str] = [
    f"{morph}_{bin_name}_uM"
    for morph in TEMPORAL_BIN_MORPHOGENS
    for bin_name in TEMPORAL_BIN_NAMES
]

# Expanded column list when temporal bins are enabled: standard columns with
# binned morphogens replaced by their 3 temporal bin columns.
# This is the column order produced by build_morphogen_matrix(temporal_bins=True).
_BINNED_COLS = {f"{m}_uM" for m in TEMPORAL_BIN_MORPHOGENS}
MORPHOGEN_COLUMNS_TEMPORAL: list[str] = [
    c for c in MORPHOGEN_COLUMNS if c not in _BINNED_COLS
] + TEMPORAL_BIN_COLUMNS

# --- Convergence diagnostics ---
# Pipeline-specific heuristics for convergence detection. No published BO
# convergence diagnostic paper prescribes these specific values.
# Acquisition decay: if max acqf value falls below this fraction of round 1, converging
CONVERGENCE_ACQUISITION_DECAY_THRESHOLD = 0.1
# Recommendation spread: pairwise L2 distance threshold (in normalised morphogen space)
CONVERGENCE_CLUSTER_SPREAD_THRESHOLD = 0.05
# Number of Sobol candidates to evaluate for mean posterior variance
CONVERGENCE_POSTERIOR_EVAL_POINTS = 512

# --- Ensemble disagreement ---
# Ensemble restart count and stability threshold are pipeline heuristics.
# The zero-passing kernel concept is inspired by GPerturb (Xing & Yau 2025,
# Nature Communications 16), but these specific constants are not from that paper.
ENSEMBLE_DEFAULT_N_RESTARTS = 5
ENSEMBLE_STABILITY_LOW_THRESHOLD = 0.5  # below this → unstable recommendations

# --- Selective log-scaling for concentration dimensions ---
# Log-scale morphogen concentrations follow pharmacological convention
# (Hill equation log-linear dose-response). Kanda et al. 2022
# (DOI:10.7554/eLife.77007) demonstrated robotic BO for cell culture
# optimization.
# Excludes log_harvest_day (already log-scaled) and base media columns
# (usually constant across conditions → zero-variance → auto-dropped).
LOG_SCALE_COLUMNS: list[str] = [
    col for col in MORPHOGEN_COLUMNS
    if col.endswith("_uM")
]

# Per-morphogen activity thresholds (µM) based on published EC50/IC50.
# Below this concentration, morphogen is considered inactive for antagonism detection.
MORPHOGEN_ACTIVITY_THRESHOLDS: dict[str, float] = {
    "LDN193189_uM": 0.005,    # IC50 ~5 nM (ALK2/ALK3)
    "SAG_uM": 0.003,           # EC50 ~3 nM (Chen et al. 2002, DOI:10.1073/pnas.212323999)
    "SB431542_uM": 0.05,       # IC50 ~94 nM (Inman et al. 2002)
    "DAPT_uM": 0.5,            # IC50 ~115 nM (gamma-secretase)
    "Dorsomorphin_uM": 0.5,    # IC50 ~500 nM (ALK2/3/6)
    "IWP2_uM": 0.1,            # IC50 ~27 nM (Porcupine)
}
# Default 0.1 µM for morphogens not listed
MORPHOGEN_ACTIVITY_THRESHOLD_DEFAULT: float = 0.1

# --- Gruffi stress-filtering defaults ---
# Pipeline-specific defaults; Gruffi (Vertesy et al. 2022,
# DOI:10.15252/embj.2022111118) recommends visual inspection of score
# distributions. These values are suitable for brain organoid scRNA-seq
# but not direct paper defaults.
GRUFFI_DEFAULT_THRESHOLD = 0.15
GRUFFI_DEFAULT_RESOLUTION = 2.0
GRUFFI_MIN_CELLS_PER_CONDITION = 50
