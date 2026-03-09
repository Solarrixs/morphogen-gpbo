"""Cross-screen QC validation for overlapping morphogen conditions.

Compares cell type fractions between screens for conditions that share
the same morphogen vector, flagging batch effect concerns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from gopro.config import get_logger

logger = get_logger(__name__)


def compute_cross_screen_similarity(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
) -> dict[str, dict]:
    """Compute cosine similarity between overlapping conditions across screens.

    Args:
        fracs_a: Cell type fractions from screen A (conditions x cell types).
        fracs_b: Cell type fractions from screen B (conditions x cell types).
        condition_mapping: Dict mapping screen_a condition -> screen_b condition.

    Returns:
        Dict mapping screen_a condition -> {cosine_similarity, screen_b_condition}.
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

    results = {}
    for cond_a, cond_b in condition_mapping.items():
        if cond_a not in fracs_a.index or cond_b not in fracs_b.index:
            logger.warning("Condition not found: %s or %s", cond_a, cond_b)
            continue

        vec_a = fracs_a.loc[cond_a, all_cols].values.reshape(1, -1)
        vec_b = fracs_b.loc[cond_b, all_cols].values.reshape(1, -1)

        sim = float(sklearn_cosine(vec_a, vec_b)[0, 0])
        results[cond_a] = {
            "cosine_similarity": sim,
            "screen_b_condition": cond_b,
        }
        logger.info("QC: %s vs %s — cosine similarity = %.3f", cond_a, cond_b, sim)

    return results


def validate_cross_screen(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
    threshold: float = 0.8,
) -> list[str]:
    """Validate overlapping conditions and return flagged ones.

    Args:
        fracs_a: Cell type fractions from screen A.
        fracs_b: Cell type fractions from screen B.
        condition_mapping: Dict mapping screen_a -> screen_b conditions.
        threshold: Minimum cosine similarity (below = flagged).

    Returns:
        List of screen_a condition names that failed QC.
    """
    similarities = compute_cross_screen_similarity(fracs_a, fracs_b, condition_mapping)

    flagged = []
    for cond, result in similarities.items():
        if result["cosine_similarity"] < threshold:
            logger.warning(
                "QC FLAG: %s vs %s similarity %.3f < %.3f threshold",
                cond, result["screen_b_condition"],
                result["cosine_similarity"], threshold,
            )
            flagged.append(cond)

    if not flagged:
        logger.info("Cross-screen QC passed: all overlapping conditions above %.2f threshold", threshold)

    return flagged
