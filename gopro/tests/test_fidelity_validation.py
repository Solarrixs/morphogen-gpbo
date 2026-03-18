"""Tests for cross-fidelity correlation validation gate."""

import math
import numpy as np
import pandas as pd
import pytest

from conftest import _import_pipeline_module

step04 = _import_pipeline_module("04_gpbo_loop")


def _make_fractions(data: dict, index: list[str]) -> pd.DataFrame:
    """Helper to create a cell type fractions DataFrame."""
    return pd.DataFrame(data, index=index)


class TestComputeRSquared:
    """Tests for _compute_r_squared() helper."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert step04._compute_r_squared(y, y) == pytest.approx(1.0)

    def test_zero_prediction(self):
        """Predicting the mean gives R²=0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert step04._compute_r_squared(y_true, y_pred) == pytest.approx(0.0)

    def test_negative_r_squared(self):
        """Predictions worse than mean give R² < 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 2.0])  # anti-correlated
        assert step04._compute_r_squared(y_true, y_pred) < 0.0

    def test_constant_true_returns_nan(self):
        """Zero variance in y_true gives NaN."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert math.isnan(step04._compute_r_squared(y_true, y_pred))


class TestValidateFidelityCorrelation:
    """Tests for validate_fidelity_correlation()."""

    def test_perfect_correlation_recommends_skip(self):
        """Identical data should give R²~1.0 and 'skip_mfbo_use_cheap'."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1], "ct_B": [0.4, 0.7, 0.9]},
            index=["c1", "c2", "c3"],
        )
        virtual = real.copy()

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert result["overall_correlation"] > 0.99
        assert result["recommendation"] == "skip_mfbo_use_cheap"
        assert result["n_overlap"] == 3

    def test_no_correlation_recommends_single_fidelity(self):
        """Uncorrelated data should give low R² and 'single_fidelity'."""
        np.random.seed(42)
        n = 20
        real = _make_fractions(
            {
                "ct_A": np.random.dirichlet([1, 1, 1], n)[:, 0],
                "ct_B": np.random.dirichlet([1, 1, 1], n)[:, 1],
                "ct_C": np.random.dirichlet([1, 1, 1], n)[:, 2],
            },
            index=[f"c{i}" for i in range(n)],
        )
        # Create uncorrelated virtual data (independent random)
        np.random.seed(999)
        virtual = _make_fractions(
            {
                "ct_A": np.random.dirichlet([1, 1, 1], n)[:, 0],
                "ct_B": np.random.dirichlet([1, 1, 1], n)[:, 1],
                "ct_C": np.random.dirichlet([1, 1, 1], n)[:, 2],
            },
            index=[f"c{i}" for i in range(n)],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert result["overall_correlation"] < 0.80
        assert result["recommendation"] == "single_fidelity"
        assert result["n_overlap"] == n

    def test_moderate_correlation_recommends_mfbo(self):
        """Data with R² in [0.80, 0.90] should give 'use_mfbo'."""
        np.random.seed(42)
        n = 50
        base = np.random.dirichlet([2, 3, 5], n)
        # Add Gaussian noise to target R² ~ 0.85
        np.random.seed(100)
        noise = np.random.normal(0, 0.065, base.shape)
        virtual_vals = base + noise
        virtual_vals = np.clip(virtual_vals, 0.001, None)
        virtual_vals = virtual_vals / virtual_vals.sum(axis=1, keepdims=True)

        real = _make_fractions(
            {"ct_A": base[:, 0], "ct_B": base[:, 1], "ct_C": base[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )
        virtual = _make_fractions(
            {"ct_A": virtual_vals[:, 0], "ct_B": virtual_vals[:, 1], "ct_C": virtual_vals[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert 0.80 <= result["overall_correlation"] <= 0.90, (
            f"Expected R² in [0.80, 0.90], got {result['overall_correlation']:.3f}"
        )
        assert result["recommendation"] == "use_mfbo"

    def test_no_overlap_returns_nan(self):
        """No overlapping conditions should return NaN."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["c1", "c2"],
        )
        virtual = _make_fractions(
            {"ct_A": [0.5, 0.5], "ct_B": [0.5, 0.5]},
            index=["c3", "c4"],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert math.isnan(result["overall_correlation"])
        assert result["n_overlap"] == 0
        assert result["recommendation"] == "use_mfbo"

    def test_per_cell_type_r_squared(self):
        """Verify per-cell-type R² is computed for variable columns."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1, 0.5], "ct_B": [0.4, 0.7, 0.9, 0.5]},
            index=["c1", "c2", "c3", "c4"],
        )
        virtual = _make_fractions(
            {"ct_A": [0.55, 0.35, 0.15, 0.45], "ct_B": [0.45, 0.65, 0.85, 0.55]},
            index=["c1", "c2", "c3", "c4"],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert "ct_A" in result["per_cell_type"]
        assert "ct_B" in result["per_cell_type"]
        # Both should have high R² since virtual ~ real + small offset
        for ct, r2 in result["per_cell_type"].items():
            assert r2 > 0.8, f"Expected high R² for {ct}, got {r2}"

    def test_partial_overlap(self):
        """Only overlapping conditions should be used."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1], "ct_B": [0.4, 0.7, 0.9]},
            index=["c1", "c2", "c3"],
        )
        virtual = _make_fractions(
            {"ct_A": [0.6, 0.5], "ct_B": [0.4, 0.5]},
            index=["c1", "c99"],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert result["n_overlap"] == 1

    def test_different_columns_aligned(self):
        """Virtual data with different cell types should be zero-filled and aligned."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["c1", "c2"],
        )
        virtual = _make_fractions(
            {"ct_A": [0.5, 0.4], "ct_C": [0.5, 0.6]},
            index=["c1", "c2"],
        )

        result = step04.validate_fidelity_correlation(
            real, virtual, fidelity_label="test"
        )

        assert result["n_overlap"] == 2
        assert not math.isnan(result["overall_correlation"])

    def test_does_not_mutate_input(self):
        """Verify input DataFrames are not mutated."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["c1", "c2"],
        )
        virtual = _make_fractions(
            {"ct_A": [0.5, 0.4], "ct_B": [0.5, 0.6]},
            index=["c1", "c2"],
        )
        real_orig = real.copy()
        virtual_orig = virtual.copy()

        step04.validate_fidelity_correlation(real, virtual)

        pd.testing.assert_frame_equal(real, real_orig)
        pd.testing.assert_frame_equal(virtual, virtual_orig)

    def test_legacy_spearman_method(self):
        """Verify legacy spearman method still works."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1], "ct_B": [0.4, 0.7, 0.9]},
            index=["c1", "c2", "c3"],
        )
        virtual = real.copy()

        result = step04.validate_fidelity_correlation(
            real, virtual, method="spearman"
        )

        assert result["overall_correlation"] > 0.99
        assert result["recommendation"] == "skip_mfbo_use_cheap"

    def test_legacy_pearson_method(self):
        """Verify legacy pearson method still works."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1], "ct_B": [0.4, 0.7, 0.9]},
            index=["c1", "c2", "c3"],
        )
        virtual = real.copy()

        result = step04.validate_fidelity_correlation(
            real, virtual, method="pearson"
        )

        assert result["overall_correlation"] > 0.99
        assert result["recommendation"] == "skip_mfbo_use_cheap"

    def test_default_method_is_r_squared(self):
        """Default method should be 'r_squared', not 'spearman'."""
        import inspect
        sig = inspect.signature(step04.validate_fidelity_correlation)
        assert sig.parameters["method"].default == "r_squared"


class TestFidelityR2Thresholds:
    """Tests for FIDELITY_R2_THRESHOLDS config constant."""

    def test_thresholds_dict_exists(self):
        from gopro.config import FIDELITY_R2_THRESHOLDS
        assert isinstance(FIDELITY_R2_THRESHOLDS, dict)
        assert "drop" in FIDELITY_R2_THRESHOLDS
        assert "skip" in FIDELITY_R2_THRESHOLDS

    def test_thresholds_values(self):
        from gopro.config import FIDELITY_R2_THRESHOLDS
        assert FIDELITY_R2_THRESHOLDS["drop"] == 0.80
        assert FIDELITY_R2_THRESHOLDS["skip"] == 0.90

    def test_drop_less_than_skip(self):
        from gopro.config import FIDELITY_R2_THRESHOLDS
        assert FIDELITY_R2_THRESHOLDS["drop"] < FIDELITY_R2_THRESHOLDS["skip"]

    def test_thresholds_in_valid_range(self):
        from gopro.config import FIDELITY_R2_THRESHOLDS
        for key, val in FIDELITY_R2_THRESHOLDS.items():
            assert 0.0 < val < 1.0, f"{key} threshold {val} out of (0,1)"

    def test_legacy_aliases_match(self):
        """Legacy constants should match new thresholds dict."""
        from gopro.config import (
            FIDELITY_CORRELATION_THRESHOLD,
            FIDELITY_R2_THRESHOLDS,
            FIDELITY_SKIP_MFBO_THRESHOLD,
        )
        assert FIDELITY_CORRELATION_THRESHOLD == FIDELITY_R2_THRESHOLDS["drop"]
        assert FIDELITY_SKIP_MFBO_THRESHOLD == FIDELITY_R2_THRESHOLDS["skip"]


class TestThreeZoneRouting:
    """Tests for the 3-zone R²-based routing logic with synthetic data."""

    def test_zone_skip_high_r2(self):
        """R² > 0.90: recommend skip_mfbo_use_cheap."""
        np.random.seed(42)
        n = 30
        base = np.random.dirichlet([5, 3, 2], n)
        # Very small noise → R² > 0.90
        noise = np.random.normal(0, 0.005, base.shape)
        virtual_vals = base + noise
        virtual_vals = np.clip(virtual_vals, 0.001, None)
        virtual_vals = virtual_vals / virtual_vals.sum(axis=1, keepdims=True)

        real = _make_fractions(
            {"ct_A": base[:, 0], "ct_B": base[:, 1], "ct_C": base[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )
        virtual = _make_fractions(
            {"ct_A": virtual_vals[:, 0], "ct_B": virtual_vals[:, 1], "ct_C": virtual_vals[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )

        result = step04.validate_fidelity_correlation(real, virtual)
        assert result["overall_correlation"] > 0.90
        assert result["recommendation"] == "skip_mfbo_use_cheap"

    def test_zone_drop_low_r2(self):
        """R² < 0.80: recommend single_fidelity."""
        np.random.seed(42)
        n = 30
        real = _make_fractions(
            {
                "ct_A": np.random.dirichlet([1, 1, 1], n)[:, 0],
                "ct_B": np.random.dirichlet([1, 1, 1], n)[:, 1],
                "ct_C": np.random.dirichlet([1, 1, 1], n)[:, 2],
            },
            index=[f"c{i}" for i in range(n)],
        )
        np.random.seed(999)
        virtual = _make_fractions(
            {
                "ct_A": np.random.dirichlet([1, 1, 1], n)[:, 0],
                "ct_B": np.random.dirichlet([1, 1, 1], n)[:, 1],
                "ct_C": np.random.dirichlet([1, 1, 1], n)[:, 2],
            },
            index=[f"c{i}" for i in range(n)],
        )

        result = step04.validate_fidelity_correlation(real, virtual)
        assert result["overall_correlation"] < 0.80
        assert result["recommendation"] == "single_fidelity"

    def test_zone_mfbo_moderate_r2(self):
        """0.80 ≤ R² ≤ 0.90: recommend use_mfbo."""
        np.random.seed(42)
        n = 50
        base = np.random.dirichlet([2, 3, 5], n)
        np.random.seed(100)
        noise = np.random.normal(0, 0.065, base.shape)
        virtual_vals = base + noise
        virtual_vals = np.clip(virtual_vals, 0.001, None)
        virtual_vals = virtual_vals / virtual_vals.sum(axis=1, keepdims=True)

        real = _make_fractions(
            {"ct_A": base[:, 0], "ct_B": base[:, 1], "ct_C": base[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )
        virtual = _make_fractions(
            {"ct_A": virtual_vals[:, 0], "ct_B": virtual_vals[:, 1], "ct_C": virtual_vals[:, 2]},
            index=[f"c{i}" for i in range(n)],
        )

        result = step04.validate_fidelity_correlation(real, virtual)
        assert 0.80 <= result["overall_correlation"] <= 0.90, (
            f"Expected R² in [0.80, 0.90], got {result['overall_correlation']:.3f}"
        )
        assert result["recommendation"] == "use_mfbo"

    def test_r_squared_details_mention_r2(self):
        """Details string should mention R² when using default method."""
        real = _make_fractions(
            {"ct_A": [0.6, 0.3, 0.1], "ct_B": [0.4, 0.7, 0.9]},
            index=["c1", "c2", "c3"],
        )
        virtual = real.copy()

        result = step04.validate_fidelity_correlation(real, virtual)
        assert "R²" in result["details"]


class TestMonitorFidelityPerRound:
    """Tests for monitor_fidelity_per_round()."""

    def test_single_round_no_degradation(self, tmp_path):
        """First round should never trigger degradation (not enough history)."""
        val_results = {
            "CellRank2": {
                "overall_correlation": 0.75,
                "recommendation": "use_mfbo",
                "n_overlap": 5,
            },
        }
        history_path = tmp_path / "fidelity_monitoring.csv"
        result = step04.monitor_fidelity_per_round(
            val_results, round_num=1, history_path=history_path,
        )
        assert history_path.exists()
        assert len(result["history"]) == 1
        assert result["degraded_sources"] == []
        assert result["auto_fallback"] is False

    def test_stable_correlation_no_fallback(self, tmp_path):
        """Stable or improving correlation should not trigger fallback."""
        history_path = tmp_path / "fidelity_monitoring.csv"
        # Round 1: moderate correlation
        step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.6, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=1, history_path=history_path,
        )
        # Round 2: improved
        step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.7, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=2, history_path=history_path,
        )
        # Round 3: still good
        result = step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.72, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=3, history_path=history_path,
        )
        assert len(result["history"]) == 3
        assert result["degraded_sources"] == []
        assert result["auto_fallback"] is False

    def test_degrading_correlation_triggers_fallback(self, tmp_path):
        """Two consecutive declining rounds should trigger auto-fallback."""
        history_path = tmp_path / "fidelity_monitoring.csv"
        # Round 1: good
        step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.8, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=1, history_path=history_path,
        )
        # Round 2: declining
        step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.65, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=2, history_path=history_path,
        )
        # Round 3: declining further
        result = step04.monitor_fidelity_per_round(
            {"CellRank2": {"overall_correlation": 0.5, "recommendation": "use_mfbo", "n_overlap": 5}},
            round_num=3, history_path=history_path,
        )
        assert "CellRank2" in result["degraded_sources"]
        assert result["auto_fallback"] is True

    def test_idempotent_rerun(self, tmp_path):
        """Re-running same round should replace, not duplicate rows."""
        history_path = tmp_path / "fidelity_monitoring.csv"
        val = {"CellRank2": {"overall_correlation": 0.7, "recommendation": "use_mfbo", "n_overlap": 5}}
        step04.monitor_fidelity_per_round(val, round_num=1, history_path=history_path)
        result = step04.monitor_fidelity_per_round(val, round_num=1, history_path=history_path)
        assert len(result["history"]) == 1  # Not 2

    def test_multiple_sources_tracked_independently(self, tmp_path):
        """Each fidelity source is tracked and evaluated independently."""
        history_path = tmp_path / "fidelity_monitoring.csv"
        # Both sources start good
        step04.monitor_fidelity_per_round(
            {
                "CellRank2": {"overall_correlation": 0.8, "recommendation": "use_mfbo", "n_overlap": 5},
                "CellFlow": {"overall_correlation": 0.6, "recommendation": "use_mfbo", "n_overlap": 3},
            },
            round_num=1, history_path=history_path,
        )
        # CellRank2 stable, CellFlow declining
        step04.monitor_fidelity_per_round(
            {
                "CellRank2": {"overall_correlation": 0.82, "recommendation": "use_mfbo", "n_overlap": 5},
                "CellFlow": {"overall_correlation": 0.45, "recommendation": "use_mfbo", "n_overlap": 3},
            },
            round_num=2, history_path=history_path,
        )
        # CellFlow declining further
        result = step04.monitor_fidelity_per_round(
            {
                "CellRank2": {"overall_correlation": 0.80, "recommendation": "use_mfbo", "n_overlap": 5},
                "CellFlow": {"overall_correlation": 0.35, "recommendation": "use_mfbo", "n_overlap": 3},
            },
            round_num=3, history_path=history_path,
        )
        # Only CellFlow should be degraded
        assert "CellFlow" in result["degraded_sources"]
        assert "CellRank2" not in result["degraded_sources"]
