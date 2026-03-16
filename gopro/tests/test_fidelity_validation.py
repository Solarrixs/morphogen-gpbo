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


class TestValidateFidelityCorrelation:
    """Tests for validate_fidelity_correlation()."""

    def test_perfect_correlation_recommends_skip(self):
        """Identical data should give correlation ~1.0 and 'skip_mfbo_use_cheap'."""
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
        """Shuffled/uncorrelated data should give low correlation and 'single_fidelity'."""
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

        assert result["overall_correlation"] < 0.3
        assert result["recommendation"] == "single_fidelity"
        assert result["n_overlap"] == n

    def test_moderate_correlation_recommends_mfbo(self):
        """Partially correlated data should give 'use_mfbo'."""
        np.random.seed(42)
        n = 30
        base = np.random.dirichlet([2, 3, 5], n)
        noise = np.random.dirichlet([2, 3, 5], n)
        # Mix: 40% signal + 60% noise -> moderate correlation
        virtual_vals = 0.4 * base + 0.6 * noise
        # Re-normalize
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

        assert 0.3 <= result["overall_correlation"] <= 0.9
        assert result["recommendation"] == "use_mfbo"

    def test_no_overlap_returns_nan(self):
        """No overlapping conditions should return NaN correlation."""
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

    def test_per_cell_type_correlations(self):
        """Verify per-cell-type correlations are computed for variable columns."""
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
        # Both should have high correlation since virtual ~ real + small noise
        for ct, corr in result["per_cell_type"].items():
            assert corr > 0.8, f"Expected high correlation for {ct}, got {corr}"

    def test_partial_overlap(self):
        """Only overlapping conditions should be used for correlation."""
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

    def test_pearson_method(self):
        """Verify pearson method works."""
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


class TestFidelityCorrelationThreshold:
    """Tests for FIDELITY_CORRELATION_THRESHOLD config constant."""

    def test_threshold_exists(self):
        from gopro.config import FIDELITY_CORRELATION_THRESHOLD
        assert isinstance(FIDELITY_CORRELATION_THRESHOLD, float)
        assert 0.0 < FIDELITY_CORRELATION_THRESHOLD < 1.0

    def test_threshold_default_value(self):
        from gopro.config import FIDELITY_CORRELATION_THRESHOLD
        assert FIDELITY_CORRELATION_THRESHOLD == 0.3
