"""Recommendation scoring framework for GP-BO pipeline.

Scores BO recommendations on four dimensions:
1. Plausibility (0-25): biological feasibility, pathway antagonism penalty
2. Novelty (0-25): distance from training data in morphogen space
3. Feasibility (0-25): morphogen cost and availability
4. Predicted fidelity (0-25): GP posterior mean

Composite score (0-100) = sum of four dimensions.

References:
- PrBO (Souza et al., NeurIPS 2020) for prior-guided BO
- Kanda et al. (eLife 2022) for robotic BO with domain constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from gopro.config import MORPHOGEN_COST_PER_UM, get_logger

logger = get_logger(__name__)

RULES_PATH = Path(__file__).parent / "pathway_rules.yaml"


@dataclass
class RecommendationScore:
    """Multi-dimensional score for a single BO recommendation."""

    condition: str
    plausibility: float = 25.0  # 0-25, 25 = no antagonisms
    novelty: float = 0.0  # 0-25, 25 = maximally novel
    feasibility: float = 25.0  # 0-25, 25 = cheapest
    predicted_fidelity: float = 0.0  # 0-25, 25 = highest predicted
    antagonism_penalties: list[str] = field(default_factory=list)

    @property
    def composite(self) -> float:
        return self.plausibility + self.novelty + self.feasibility + self.predicted_fidelity


def load_pathway_rules(path: Optional[Path] = None) -> dict:
    """Load antagonist pair rules from YAML."""
    path = path or RULES_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def score_plausibility(
    recommendation: pd.Series,
    rules: Optional[dict] = None,
) -> tuple[float, list[str]]:
    """Score biological plausibility (0-25).

    Penalizes pathway antagonisms: each active antagonist pair
    deducts 10 points (clamped to 0).
    """
    if rules is None:
        rules = load_pathway_rules()

    threshold = rules.get("absence_threshold_uM", 0.001)
    penalties: list[str] = []
    score = 25.0

    for pair_rule in rules.get("antagonist_pairs", []):
        col_a, col_b = pair_rule["pair"]
        if col_a in recommendation.index and col_b in recommendation.index:
            conc_a = float(recommendation.get(col_a, 0.0))
            conc_b = float(recommendation.get(col_b, 0.0))
            if conc_a > threshold and conc_b > threshold:
                penalties.append(
                    f"{col_a}={conc_a:.4f} + {col_b}={conc_b:.4f}: {pair_rule['reason']}"
                )
                score -= 10.0

    return max(0.0, score), penalties


def score_novelty(
    recommendation: pd.Series,
    training_X: pd.DataFrame,
    k: int = 5,
) -> float:
    """Score novelty (0-25) based on distance from training data.

    Uses mean distance to k-nearest training points, normalized
    to [0, 25] by the max pairwise distance in training data.
    """
    from scipy.spatial.distance import pdist

    # Get shared morphogen columns (exclude fidelity, metadata)
    shared_cols = [
        c for c in recommendation.index if c in training_X.columns and c != "fidelity"
    ]
    if not shared_cols:
        return 12.5  # neutral

    rec_vec = recommendation[shared_cols].values.astype(float).reshape(1, -1)
    train_vecs = training_X[shared_cols].values.astype(float)

    # Distances to all training points
    dists = np.sqrt(((train_vecs - rec_vec) ** 2).sum(axis=1))
    k_actual = min(k, len(dists))
    knn_dists = np.sort(dists)[:k_actual]
    mean_knn_dist = float(knn_dists.mean())

    # Normalize by max pairwise distance in training data
    if len(train_vecs) > 1:
        max_pairwise = float(pdist(train_vecs).max())
        if max_pairwise > 0:
            return min(25.0, 25.0 * mean_knn_dist / max_pairwise)
        else:
            # All training points are identical; novelty is purely based
            # on whether the recommendation matches them.
            if mean_knn_dist == 0.0:
                return 0.0
            return 25.0

    return 12.5


def score_feasibility(
    recommendation: pd.Series,
    cost_dict: Optional[dict[str, float]] = None,
    max_budget: float = 50.0,
) -> float:
    """Score feasibility (0-25) based on morphogen cost.

    Cheaper cocktails score higher. Linear mapping from
    [0, max_budget] to [25, 0].
    """
    if cost_dict is None:
        cost_dict = dict(MORPHOGEN_COST_PER_UM)

    total_cost = sum(
        float(recommendation.get(col, 0.0)) * cost_dict.get(col, 0.0)
        for col in recommendation.index
        if col in cost_dict
    )

    return max(0.0, 25.0 * (1.0 - total_cost / max_budget))


def score_predicted_fidelity(
    predicted_mean: float,
    predicted_std: float = 0.0,
    max_fidelity: float = 1.0,
) -> float:
    """Score predicted fidelity (0-25).

    Maps GP posterior mean to [0, 25], with a small bonus
    for low uncertainty (exploitation-aligned).
    """
    base = 25.0 * min(1.0, predicted_mean / max_fidelity)
    # Small uncertainty bonus: up to 2.5 points for low std
    if predicted_std > 0:
        uncertainty_bonus = 2.5 * max(0.0, 1.0 - predicted_std)
    else:
        uncertainty_bonus = 0.0
    return min(25.0, base + uncertainty_bonus)


def score_recommendations(
    recommendations: pd.DataFrame,
    training_X: pd.DataFrame,
    predicted_means: Optional[pd.Series] = None,
    predicted_stds: Optional[pd.Series] = None,
    rules: Optional[dict] = None,
) -> pd.DataFrame:
    """Score all recommendations on all four dimensions.

    Args:
        recommendations: BO recommendations (morphogen concentrations).
        training_X: Training data morphogen concentrations.
        predicted_means: GP posterior means per recommendation.
        predicted_stds: GP posterior stds per recommendation.
        rules: Pathway antagonism rules.

    Returns:
        DataFrame with scoring columns added, sorted by composite score descending.
    """
    if rules is None:
        rules = load_pathway_rules()

    scores: list[RecommendationScore] = []
    for idx, row in recommendations.iterrows():
        plaus, penalties = score_plausibility(row, rules)
        nov = score_novelty(row, training_X)
        feas = score_feasibility(row)

        pred_mean = (
            float(predicted_means.get(idx, 0.5))
            if predicted_means is not None
            else 12.5
        )
        pred_std = (
            float(predicted_stds.get(idx, 0.0))
            if predicted_stds is not None
            else 0.0
        )
        fid = score_predicted_fidelity(pred_mean, pred_std)

        s = RecommendationScore(
            condition=str(idx),
            plausibility=plaus,
            novelty=nov,
            feasibility=feas,
            predicted_fidelity=fid,
            antagonism_penalties=penalties,
        )
        scores.append(s)

    result = recommendations.copy()
    result["plausibility_score"] = [s.plausibility for s in scores]
    result["novelty_score"] = [s.novelty for s in scores]
    result["feasibility_score"] = [s.feasibility for s in scores]
    result["predicted_fidelity_score"] = [s.predicted_fidelity for s in scores]
    result["composite_score"] = [s.composite for s in scores]
    result["antagonism_penalties"] = [
        "; ".join(s.antagonism_penalties) if s.antagonism_penalties else ""
        for s in scores
    ]

    return result.sort_values("composite_score", ascending=False)
