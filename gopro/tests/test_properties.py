"""Property-based tests for GP-BO pipeline invariants."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pathlib import Path
import importlib.util

GOPRO_DIR = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(GOPRO_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step02 = _load("02_map_to_hnoca")
step03 = _load("03_fidelity_scoring")
step04 = _load("04_gpbo_loop")


class TestCompositionProperties:
    """Property tests for compositional data handling."""

    @given(st.integers(min_value=3, max_value=10), st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=5000)
    def test_ilr_roundtrip(self, D, N):
        """ILR transform and inverse should be identity."""
        Y = np.random.dirichlet(np.ones(D), size=N)
        Z = step04.ilr_transform(Y)
        Y_recovered = step04.ilr_inverse(Z, D)
        np.testing.assert_allclose(Y, Y_recovered, atol=1e-3)

    @given(st.integers(min_value=3, max_value=10), st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=5000)
    def test_ilr_output_dimension(self, D, N):
        """ILR should reduce dimension by 1."""
        Y = np.random.dirichlet(np.ones(D), size=N)
        Z = step04.ilr_transform(Y)
        assert Z.shape == (N, D - 1)

    @given(st.integers(min_value=3, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_ilr_finite(self, D):
        """ILR output should always be finite."""
        Y = np.random.dirichlet(np.ones(D) * 0.1, size=5)
        Z = step04.ilr_transform(Y)
        assert np.all(np.isfinite(Z))


class TestMorphogenBoundsProperties:
    """Property tests for morphogen bounds."""

    def test_all_bounds_non_negative(self):
        for col, (lo, hi) in step04.MORPHOGEN_BOUNDS.items():
            assert lo >= 0, f"{col} has negative lower bound"

    def test_all_bounds_ordered(self):
        for col, (lo, hi) in step04.MORPHOGEN_BOUNDS.items():
            assert lo < hi, f"{col} bounds not ordered: {lo} >= {hi}"

    def test_morphogen_columns_match_bounds(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert col in step04.MORPHOGEN_BOUNDS, f"Missing bounds for {col}"


class TestCellTypeFractionProperties:
    """Property tests for cell type fraction computation."""

    @given(st.integers(min_value=2, max_value=5), st.integers(min_value=10, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_fractions_sum_to_one(self, n_types, n_cells):
        types = [f"type_{i}" for i in range(n_types)]
        obs = pd.DataFrame({
            "condition": np.random.choice(["A", "B", "C"], size=n_cells),
            "label": np.random.choice(types, size=n_cells),
            "quality": "keep",
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label")
        np.testing.assert_allclose(fracs.sum(axis=1), 1.0, atol=1e-6)

    @given(st.integers(min_value=2, max_value=5), st.integers(min_value=10, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_fractions_non_negative(self, n_types, n_cells):
        types = [f"type_{i}" for i in range(n_types)]
        obs = pd.DataFrame({
            "condition": np.random.choice(["A", "B"], size=n_cells),
            "label": np.random.choice(types, size=n_cells),
            "quality": "keep",
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label")
        assert (fracs >= 0).all().all()

    @given(st.integers(min_value=2, max_value=5), st.integers(min_value=10, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_fractions_at_most_one(self, n_types, n_cells):
        types = [f"type_{i}" for i in range(n_types)]
        obs = pd.DataFrame({
            "condition": np.random.choice(["A", "B"], size=n_cells),
            "label": np.random.choice(types, size=n_cells),
            "quality": "keep",
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label")
        assert (fracs <= 1).all().all()


class TestFidelityScoringProperties:
    """Property tests for fidelity scoring invariants."""

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_composite_fidelity_in_range(self, rss, on_target, off_target, entropy):
        score = step03.compute_composite_fidelity(rss, on_target, off_target, entropy)
        assert 0.0 <= score <= 1.0

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_normalized_entropy_in_range(self, n):
        p = np.random.dirichlet(np.ones(n))
        h = step03.normalized_entropy(p)
        assert 0.0 <= h <= 1.0

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_cosine_similarity_self_is_one(self, n):
        a = np.random.rand(n)
        a = a / a.sum()  # normalize
        sim = step03.cosine_similarity(a, a)
        assert sim == pytest.approx(1.0, abs=1e-6)
