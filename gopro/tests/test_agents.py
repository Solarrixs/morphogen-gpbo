"""Tests for gopro.agents.scorer — recommendation scoring framework."""

import numpy as np
import pandas as pd
import pytest

from gopro.agents.scorer import (
    RecommendationScore,
    load_pathway_rules,
    score_feasibility,
    score_novelty,
    score_plausibility,
    score_predicted_fidelity,
    score_recommendations,
)
from gopro.config import MORPHOGEN_COLUMNS


def _make_morphogen_series(**overrides) -> pd.Series:
    """Helper: create a morphogen Series with zeros, overriding specified columns."""
    data = {col: 0.0 for col in MORPHOGEN_COLUMNS}
    data.update(overrides)
    return pd.Series(data)


def _make_training_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Helper: create a small synthetic training DataFrame."""
    rng = np.random.RandomState(seed)
    data = {col: rng.uniform(0, 1, size=n) for col in MORPHOGEN_COLUMNS}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# RecommendationScore
# ---------------------------------------------------------------------------


class TestRecommendationScore:
    def test_composite_score_sum(self):
        """Composite should be sum of 4 dimensions."""
        s = RecommendationScore(
            condition="test",
            plausibility=20.0,
            novelty=15.0,
            feasibility=10.0,
            predicted_fidelity=5.0,
        )
        assert s.composite == pytest.approx(50.0)

    def test_max_composite_is_100(self):
        """Perfect scores should give composite=100."""
        s = RecommendationScore(
            condition="perfect",
            plausibility=25.0,
            novelty=25.0,
            feasibility=25.0,
            predicted_fidelity=25.0,
        )
        assert s.composite == pytest.approx(100.0)

    def test_default_antagonism_penalties_empty(self):
        """Default antagonism_penalties should be empty list."""
        s = RecommendationScore(condition="test")
        assert s.antagonism_penalties == []


# ---------------------------------------------------------------------------
# Plausibility
# ---------------------------------------------------------------------------


class TestScorePlausibility:
    def test_no_antagonisms_full_score(self):
        """No active antagonist pairs -> 25."""
        rec = _make_morphogen_series(CHIR99021_uM=3.0)
        score, penalties = score_plausibility(rec)
        assert score == pytest.approx(25.0)
        assert penalties == []

    def test_bmp_ldn_antagonism_detected(self):
        """BMP4 + LDN193189 both active -> penalty."""
        rec = _make_morphogen_series(BMP4_uM=0.01, LDN193189_uM=0.1)
        score, penalties = score_plausibility(rec)
        assert score < 25.0
        assert len(penalties) >= 1
        assert "BMP4_uM" in penalties[0]

    def test_below_threshold_not_penalized(self):
        """Concentrations below absence_threshold not flagged."""
        rec = _make_morphogen_series(BMP4_uM=0.0001, LDN193189_uM=0.0005)
        score, penalties = score_plausibility(rec)
        assert score == pytest.approx(25.0)
        assert penalties == []

    def test_multiple_antagonisms_cumulative(self):
        """Multiple active pairs -> cumulative penalty (clamped to 0)."""
        rec = _make_morphogen_series(
            BMP4_uM=1.0,
            LDN193189_uM=0.1,
            CHIR99021_uM=3.0,
            IWP2_uM=1.0,
            SAG_uM=0.5,
            cyclopamine_uM=1.0,
        )
        score, penalties = score_plausibility(rec)
        # At least 3 antagonisms: BMP4/LDN, CHIR/IWP2, SAG/cyclopamine
        assert len(penalties) >= 3
        assert score == pytest.approx(0.0)

    def test_custom_rules(self):
        """Custom rules dict is respected."""
        custom_rules = {
            "antagonist_pairs": [
                {"pair": ["CHIR99021_uM", "IWP2_uM"], "reason": "test"},
            ],
            "absence_threshold_uM": 0.001,
        }
        rec = _make_morphogen_series(CHIR99021_uM=3.0, IWP2_uM=1.0)
        score, penalties = score_plausibility(rec, rules=custom_rules)
        assert score == pytest.approx(15.0)
        assert len(penalties) == 1


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


class TestScoreNovelty:
    def test_near_training_low_novelty(self):
        """Point near training data -> low novelty score."""
        # Create training data where all points are identical -> knn dist = 0
        train = pd.DataFrame(
            {col: [1.0] * 10 for col in MORPHOGEN_COLUMNS},
        )
        rec = _make_morphogen_series(**{col: 1.0 for col in MORPHOGEN_COLUMNS})
        nov = score_novelty(rec, train)
        assert nov < 1.0  # should be ~0 since rec == all training points

    def test_far_from_training_high_novelty(self):
        """Point far from training data -> high novelty score."""
        train = _make_training_df(n=10, seed=42)
        # Extreme point far from [0,1] range training data
        rec = _make_morphogen_series(**{col: 100.0 for col in MORPHOGEN_COLUMNS})
        nov = score_novelty(rec, train)
        assert nov > 15.0  # should be high

    def test_empty_shared_cols_neutral(self):
        """No shared columns -> neutral score 12.5."""
        train = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        rec = _make_morphogen_series(CHIR99021_uM=1.0)
        nov = score_novelty(rec, train)
        assert nov == pytest.approx(12.5)

    def test_single_training_point(self):
        """Single training point should still return a score."""
        train = _make_training_df(n=1, seed=42)
        rec = _make_morphogen_series(CHIR99021_uM=5.0)
        nov = score_novelty(rec, train)
        # With only 1 training point, pdist can't normalize -> 12.5
        assert 0.0 <= nov <= 25.0


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


class TestScoreFeasibility:
    def test_cheap_cocktail_high_score(self):
        """All-small-molecule cocktail -> high feasibility."""
        rec = _make_morphogen_series(CHIR99021_uM=3.0, RA_uM=1.0)
        feas = score_feasibility(rec)
        # Cost: 3*0.01 + 1*0.01 = 0.04, very cheap
        assert feas > 24.0

    def test_expensive_cocktail_low_score(self):
        """Recombinant protein cocktail -> low feasibility."""
        rec = _make_morphogen_series(SHH_uM=5.0, BMP4_uM=5.0, FGF8_uM=3.0)
        feas = score_feasibility(rec)
        # Cost: 5*5 + 5*5 + 3*3 = 59.0, exceeds max_budget of 50
        assert feas == pytest.approx(0.0)

    def test_custom_cost_dict(self):
        """Custom cost dict is respected."""
        rec = _make_morphogen_series(CHIR99021_uM=1.0)
        feas = score_feasibility(rec, cost_dict={"CHIR99021_uM": 25.0}, max_budget=50.0)
        # Cost = 25, so score = 25 * (1 - 25/50) = 12.5
        assert feas == pytest.approx(12.5)

    def test_zero_cost_full_score(self):
        """All-zero recommendation -> full feasibility."""
        rec = _make_morphogen_series()
        feas = score_feasibility(rec)
        assert feas == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Predicted fidelity
# ---------------------------------------------------------------------------


class TestScorePredictedFidelity:
    def test_high_mean_high_score(self):
        """High predicted mean -> high score."""
        score = score_predicted_fidelity(0.9, 0.1)
        assert score > 20.0

    def test_zero_mean_zero_score(self):
        """Zero predicted mean -> zero base score."""
        score = score_predicted_fidelity(0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_low_uncertainty_bonus(self):
        """Low std gives small bonus over no-std case."""
        score_with_std = score_predicted_fidelity(0.5, 0.1)
        score_no_std = score_predicted_fidelity(0.5, 0.0)
        assert score_with_std > score_no_std

    def test_capped_at_25(self):
        """Score never exceeds 25."""
        score = score_predicted_fidelity(2.0, 0.0, max_fidelity=1.0)
        assert score <= 25.0


# ---------------------------------------------------------------------------
# End-to-end: score_recommendations
# ---------------------------------------------------------------------------


class TestScoreRecommendations:
    def test_full_scoring_pipeline(self):
        """End-to-end scoring produces all expected columns."""
        train = _make_training_df(n=10)
        recs = pd.DataFrame(
            {col: [0.0, 1.0] for col in MORPHOGEN_COLUMNS},
            index=["rec_0", "rec_1"],
        )
        result = score_recommendations(recs, train)

        expected_cols = [
            "plausibility_score",
            "novelty_score",
            "feasibility_score",
            "predicted_fidelity_score",
            "composite_score",
            "antagonism_penalties",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Composite scores should be in [0, 100]
        assert (result["composite_score"] >= 0).all()
        assert (result["composite_score"] <= 100).all()

    def test_antagonistic_cocktail_ranked_lower(self):
        """Cocktail with antagonisms should rank below clean cocktail."""
        train = _make_training_df(n=10)

        clean = {col: 0.0 for col in MORPHOGEN_COLUMNS}
        clean["CHIR99021_uM"] = 3.0

        antagonistic = {col: 0.0 for col in MORPHOGEN_COLUMNS}
        antagonistic["BMP4_uM"] = 1.0
        antagonistic["LDN193189_uM"] = 0.5
        antagonistic["CHIR99021_uM"] = 3.0
        antagonistic["IWP2_uM"] = 1.0

        recs = pd.DataFrame([clean, antagonistic], index=["clean", "antagonistic"])
        result = score_recommendations(recs, train)

        clean_score = result.loc["clean", "composite_score"]
        antag_score = result.loc["antagonistic", "composite_score"]
        assert clean_score > antag_score

    def test_sorted_descending(self):
        """Results should be sorted by composite_score descending."""
        train = _make_training_df(n=10)
        recs = pd.DataFrame(
            {col: np.random.uniform(0, 2, size=5) for col in MORPHOGEN_COLUMNS},
            index=[f"rec_{i}" for i in range(5)],
        )
        result = score_recommendations(recs, train)
        scores = result["composite_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_with_predicted_means_and_stds(self):
        """Passing predicted_means and predicted_stds is used in scoring."""
        train = _make_training_df(n=5)
        recs = pd.DataFrame(
            {col: [0.5] for col in MORPHOGEN_COLUMNS},
            index=["rec_0"],
        )
        means = pd.Series({"rec_0": 0.8})
        stds = pd.Series({"rec_0": 0.1})
        result = score_recommendations(recs, train, predicted_means=means, predicted_stds=stds)
        fid_score = result.loc["rec_0", "predicted_fidelity_score"]
        # 0.8 / 1.0 * 25 = 20.0 base + uncertainty bonus
        assert fid_score > 20.0


# ---------------------------------------------------------------------------
# load_pathway_rules
# ---------------------------------------------------------------------------


class TestLoadPathwayRules:
    def test_loads_default_rules(self):
        """Default rules file loads successfully."""
        rules = load_pathway_rules()
        assert "antagonist_pairs" in rules
        assert "absence_threshold_uM" in rules
        assert len(rules["antagonist_pairs"]) >= 7

    def test_all_rule_columns_in_morphogen_columns(self):
        """All column names in rules should be valid MORPHOGEN_COLUMNS."""
        rules = load_pathway_rules()
        for pair_rule in rules["antagonist_pairs"]:
            for col in pair_rule["pair"]:
                assert col in MORPHOGEN_COLUMNS, (
                    f"Rule references '{col}' which is not in MORPHOGEN_COLUMNS"
                )
