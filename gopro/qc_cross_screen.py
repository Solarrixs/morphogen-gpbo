"""Cross-screen QC validation for overlapping morphogen conditions.

Compares cell type fractions between screens for conditions that share
the same morphogen vector, flagging batch effect concerns.

Uses Aitchison similarity (Euclidean distance in CLR space) which is the
correct metric for compositional data (Quinn et al., *Bioinformatics* 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gopro.config import get_logger

logger = get_logger(__name__)


def _aitchison_distance_vec(a: np.ndarray, b: np.ndarray) -> float:
    """Aitchison distance between two composition vectors."""
    eps = 1e-6
    a_safe = np.maximum(a.ravel(), eps)
    b_safe = np.maximum(b.ravel(), eps)
    a_safe = a_safe / a_safe.sum()
    b_safe = b_safe / b_safe.sum()
    clr_a = np.log(a_safe) - np.mean(np.log(a_safe))
    clr_b = np.log(b_safe) - np.mean(np.log(b_safe))
    return float(np.linalg.norm(clr_a - clr_b))


# Fallback scale when fewer than 2 pairwise distances are available.
_AITCHISON_FALLBACK_SCALE = 2.0


def compute_cross_screen_similarity(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
) -> dict[str, dict]:
    """Compute Aitchison similarity between overlapping conditions across screens.

    Args:
        fracs_a: Cell type fractions from screen A (conditions x cell types).
        fracs_b: Cell type fractions from screen B (conditions x cell types).
        condition_mapping: Dict mapping screen_a condition -> screen_b condition.

    Returns:
        Dict mapping screen_a condition -> {similarity, screen_b_condition}.
    """
    fracs_a = fracs_a.copy()
    fracs_b = fracs_b.copy()

    # Align columns (union)
    all_cols = sorted(set(fracs_a.columns) | set(fracs_b.columns))
    for col in all_cols:
        if col not in fracs_a.columns:
            fracs_a[col] = 0.0
        if col not in fracs_b.columns:
            fracs_b[col] = 0.0

    # First pass: compute all pairwise Aitchison distances.
    dist_records: list[tuple[str, str, float]] = []
    for cond_a, cond_b in condition_mapping.items():
        if cond_a not in fracs_a.index or cond_b not in fracs_b.index:
            logger.warning("Condition not found: %s or %s", cond_a, cond_b)
            continue

        vec_a = fracs_a.loc[cond_a, all_cols].values
        vec_b = fracs_b.loc[cond_b, all_cols].values
        dist_records.append((cond_a, cond_b, _aitchison_distance_vec(vec_a, vec_b)))

    # Median heuristic for kernel bandwidth (Garreau et al., 2017).
    if len(dist_records) >= 2:
        scale = float(np.median([d for _, _, d in dist_records]))
        if scale < 1e-8:
            scale = _AITCHISON_FALLBACK_SCALE
    else:
        scale = _AITCHISON_FALLBACK_SCALE

    # Second pass: convert distances to similarities.
    results = {}
    for cond_a, cond_b, dist in dist_records:
        sim = float(np.exp(-dist / scale))
        results[cond_a] = {
            "similarity": sim,
            "screen_b_condition": cond_b,
        }
        logger.info("QC: %s vs %s — Aitchison similarity = %.3f", cond_a, cond_b, sim)

    return results


def validate_cross_screen(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
    threshold: float = 0.5,
) -> list[str]:
    """Validate overlapping conditions and return flagged ones.

    Args:
        fracs_a: Cell type fractions from screen A.
        fracs_b: Cell type fractions from screen B.
        condition_mapping: Dict mapping screen_a -> screen_b conditions.
        threshold: Minimum Aitchison similarity (below = flagged).
            Default 0.5 (corresponding to Aitchison distance ~1.4).

    Returns:
        List of screen_a condition names that failed QC.
    """
    similarities = compute_cross_screen_similarity(fracs_a, fracs_b, condition_mapping)

    flagged = []
    for cond, result in similarities.items():
        if result["similarity"] < threshold:
            logger.warning(
                "QC FLAG: %s vs %s Aitchison similarity %.3f < %.3f threshold",
                cond, result["screen_b_condition"],
                result["similarity"], threshold,
            )
            flagged.append(cond)

    if not flagged:
        logger.info("Cross-screen QC passed: all overlapping conditions above %.2f threshold", threshold)

    return flagged
