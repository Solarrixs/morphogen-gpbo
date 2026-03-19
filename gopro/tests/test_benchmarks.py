"""Tests for toy morphogen benchmark function and ARD Lipschitz diagnostic."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

from gopro.benchmarks.toy_morphogen_function import (
    ToyMorphogenFunction,
    hill_response,
    CELL_TYPES,
    MORPHOGEN_COLUMNS,
)
from conftest import _import_pipeline_module

step04 = _import_pipeline_module("04_gpbo_loop")


class TestHillResponse:
    def test_zero_concentration(self):
        assert hill_response(0.0, 1.0) == 0.0

    def test_zero_ec50(self):
        assert hill_response(1.0, 0.0) == 0.0

    def test_at_ec50(self):
        """At concentration == EC50, response should be 0.5."""
        result = hill_response(1.0, 1.0, hill_n=2.0)
        assert abs(result - 0.5) < 1e-10

    def test_high_concentration(self):
        """At very high concentration, response should approach 1.0."""
        result = hill_response(1000.0, 1.0, hill_n=2.0)
        assert result > 0.999

    def test_negative_concentration(self):
        assert hill_response(-1.0, 1.0) == 0.0


N_MORPHOGENS = len(MORPHOGEN_COLUMNS)


class TestToyMorphogenFunction:
    def test_output_shape(self):
        """Output should be (n_points, n_cell_types)."""
        fn = ToyMorphogenFunction(seed=42)
        x = np.random.rand(10, N_MORPHOGENS)
        y = fn.evaluate(x)
        assert y.shape == (10, 6)

    def test_output_is_simplex(self):
        """Each row should sum to 1.0."""
        fn = ToyMorphogenFunction(seed=42)
        x = np.random.rand(20, N_MORPHOGENS) * 10
        y = fn.evaluate(x)
        np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-10)

    def test_all_positive(self):
        """All fractions should be positive."""
        fn = ToyMorphogenFunction(seed=42)
        x = np.random.rand(20, N_MORPHOGENS) * 10
        y = fn.evaluate(x)
        assert np.all(y > 0)

    def test_deterministic_without_noise(self):
        """Same input should produce same output when noise_std=0."""
        fn = ToyMorphogenFunction(seed=42, noise_std=0.0)
        x = np.random.rand(5, N_MORPHOGENS) * 5
        y1 = fn.evaluate(x)
        y2 = fn.evaluate(x)
        np.testing.assert_array_equal(y1, y2)

    def test_noise_adds_variation(self):
        """With noise_std > 0, repeated evals should differ."""
        fn = ToyMorphogenFunction(seed=42, noise_std=1.0)
        x = np.random.rand(5, N_MORPHOGENS) * 5
        y1 = fn.evaluate(x)
        y2 = fn.evaluate(x)
        # Noise is drawn from rng state, so sequential calls differ
        assert not np.allclose(y1, y2)

    def test_single_point(self):
        """Should handle a single point (1D input promoted to 2D)."""
        fn = ToyMorphogenFunction(seed=42)
        x = np.random.rand(N_MORPHOGENS) * 5
        y = fn.evaluate(x)
        assert y.shape == (1, 6)
        np.testing.assert_allclose(y.sum(), 1.0, atol=1e-10)

    def test_zero_input(self):
        """Zero morphogens should return uniform-ish distribution (no signal)."""
        fn = ToyMorphogenFunction(seed=42, noise_std=0.0)
        x = np.zeros((1, N_MORPHOGENS))
        y = fn.evaluate(x)
        assert y.shape == (1, 6)
        np.testing.assert_allclose(y.sum(), 1.0, atol=1e-10)
        # With zero input, all Hill responses are 0, logits are 0, so softmax = uniform
        np.testing.assert_allclose(y, 1.0 / 6, atol=1e-10)

    def test_optimum_shape(self):
        """Optimum should have the same dimensionality as MORPHOGEN_COLUMNS."""
        fn = ToyMorphogenFunction(seed=42)
        assert fn.optimum.shape == (N_MORPHOGENS,)

    def test_optimum_returns_copy(self):
        """Optimum property should return a copy, not the internal array."""
        fn = ToyMorphogenFunction(seed=42)
        opt1 = fn.optimum
        opt1[:] = 0
        opt2 = fn.optimum
        assert not np.allclose(opt2, 0)

    def test_optimum_boosts_neuron(self):
        """Evaluating at optimum should give higher Neuron fraction than zero input."""
        fn = ToyMorphogenFunction(seed=42, noise_std=0.0)
        y_opt = fn.evaluate(fn.optimum.reshape(1, -1))
        y_zero = fn.evaluate(np.zeros((1, N_MORPHOGENS)))
        assert y_opt[0, 0] > y_zero[0, 0]

    def test_custom_n_cell_types(self):
        """Should work with fewer cell types."""
        fn = ToyMorphogenFunction(seed=42, n_cell_types=3)
        x = np.random.rand(5, N_MORPHOGENS) * 5
        y = fn.evaluate(x)
        assert y.shape == (5, 3)
        np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-10)

    def test_wrong_input_dims(self):
        """Should raise ValueError for wrong number of input dimensions."""
        fn = ToyMorphogenFunction(seed=42)
        with pytest.raises(ValueError, match=f"Expected {N_MORPHOGENS}"):
            fn.evaluate(np.random.rand(5, 10))

    def test_reproducible_with_same_seed(self):
        """Two instances with same seed should produce identical results."""
        fn1 = ToyMorphogenFunction(seed=123)
        fn2 = ToyMorphogenFunction(seed=123)
        np.testing.assert_array_equal(fn1.ec50, fn2.ec50)
        np.testing.assert_array_equal(fn1.optimum, fn2.optimum)


class TestARDLipschitz:
    """Tests for compute_ard_lipschitz in 04_gpbo_loop."""

    def _make_mock_model(self, lengthscales, outputscale=1.0):
        """Create a mock GP model with known kernel parameters."""
        model = MagicMock()
        # Set up covar_module with base_kernel
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [lengthscales], dtype=torch.float64
        )
        model.covar_module.outputscale = torch.tensor(
            outputscale, dtype=torch.float64
        )
        model.covar_module.base_kernel.ard_num_dims = len(lengthscales)
        # No sub-kernels (not additive)
        del model.covar_module.kernels
        # Not a ModelListGP
        del model.models
        # Not SAASBO
        del model.median_lengthscale
        return model

    def test_basic_computation(self):
        """Should return DataFrame with correct columns."""
        cols = ["dim_a", "dim_b", "dim_c"]
        model = self._make_mock_model([1.0, 2.0, 3.0], outputscale=1.0)
        df = step04.compute_ard_lipschitz(model, cols)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"dimension", "lengthscale", "lipschitz_estimate"}
        assert len(df) == 3

    def test_short_lengthscale_high_lipschitz(self):
        """Dimension with shorter lengthscale should have higher Lipschitz."""
        cols = ["short", "long"]
        model = self._make_mock_model([0.1, 10.0], outputscale=1.0)
        df = step04.compute_ard_lipschitz(model, cols)
        # Sorted descending by lipschitz_estimate
        assert df.iloc[0]["dimension"] == "short"
        assert df.iloc[1]["dimension"] == "long"
        assert df.iloc[0]["lipschitz_estimate"] > df.iloc[1]["lipschitz_estimate"]

    def test_outputscale_scales_lipschitz(self):
        """Doubling outputscale should double all Lipschitz estimates."""
        cols = ["a", "b"]
        model1 = self._make_mock_model([1.0, 2.0], outputscale=1.0)
        model2 = self._make_mock_model([1.0, 2.0], outputscale=2.0)
        df1 = step04.compute_ard_lipschitz(model1, cols)
        df2 = step04.compute_ard_lipschitz(model2, cols)
        # Sort both by dimension for comparison
        df1 = df1.sort_values("dimension").reset_index(drop=True)
        df2 = df2.sort_values("dimension").reset_index(drop=True)
        np.testing.assert_allclose(
            df2["lipschitz_estimate"].values,
            df1["lipschitz_estimate"].values * 2.0,
            rtol=1e-10,
        )

    def test_sorted_descending(self):
        """Output should be sorted by lipschitz_estimate descending."""
        cols = ["a", "b", "c", "d"]
        model = self._make_mock_model([5.0, 0.1, 1.0, 10.0], outputscale=1.0)
        df = step04.compute_ard_lipschitz(model, cols)
        lipschitz_vals = df["lipschitz_estimate"].values
        assert all(
            lipschitz_vals[i] >= lipschitz_vals[i + 1]
            for i in range(len(lipschitz_vals) - 1)
        )

    def test_dimension_mismatch_raises(self):
        """Should raise ValueError if columns don't match lengthscale dims."""
        cols = ["a", "b"]  # 2 columns
        model = self._make_mock_model([1.0, 2.0, 3.0], outputscale=1.0)  # 3 dims
        with pytest.raises(ValueError, match="mismatch"):
            step04.compute_ard_lipschitz(model, cols)
