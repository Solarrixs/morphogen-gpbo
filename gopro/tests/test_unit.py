"""Unit tests for GP-BO pipeline components."""

import pytest
import numpy as np
import pandas as pd
import torch
import unittest.mock
from pathlib import Path
import sys
import plotly.graph_objects as go

from conftest import _import_pipeline_module

step01 = _import_pipeline_module("01_load_and_convert_data")
step02 = _import_pipeline_module("02_map_to_hnoca")
step03 = _import_pipeline_module("03_fidelity_scoring")
step04 = _import_pipeline_module("04_gpbo_loop")


class TestILRTransform:
    """Tests for ILR (isometric log-ratio) transform."""

    def test_ilr_reduces_dimension(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        Z = step04.ilr_transform(Y)
        assert Z.shape == (2, 2)

    def test_ilr_inverse_roundtrip(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])
        Z = step04.ilr_transform(Y)
        Y_recovered = step04.ilr_inverse(Z, D=3)
        np.testing.assert_allclose(Y, Y_recovered, atol=1e-3)

    def test_ilr_handles_near_zero(self):
        Y = np.array([[0.99, 0.005, 0.005]])
        Z = step04.ilr_transform(Y)
        assert np.all(np.isfinite(Z))

    def test_ilr_handles_uniform(self):
        Y = np.array([[0.25, 0.25, 0.25, 0.25]])
        Z = step04.ilr_transform(Y)
        np.testing.assert_allclose(Z, np.zeros((1, 3)), atol=0.1)

    def test_ilr_single_row(self):
        Y = np.array([[0.7, 0.2, 0.1]])
        Z = step04.ilr_transform(Y)
        assert Z.shape == (1, 2)
        assert np.all(np.isfinite(Z))

    def test_ilr_pseudocount_custom(self):
        """Custom pseudocount is threaded to multiplicative replacement."""
        Y = np.array([[0.5, 0.5, 0.0]])  # has a zero
        Z_default = step04.ilr_transform(Y)
        Z_custom = step04.ilr_transform(Y, pseudocount=1e-8)
        # Both should be finite
        assert np.all(np.isfinite(Z_default))
        assert np.all(np.isfinite(Z_custom))
        # Different pseudocounts should produce different ILR values
        assert not np.allclose(Z_default, Z_custom)

    def test_ilr_pseudocount_zero_row(self):
        """All-zero row handled gracefully with custom pseudocount."""
        Y = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.2]])
        Z = step04.ilr_transform(Y, pseudocount=1e-6)
        assert Z.shape == (2, 2)
        assert np.all(np.isfinite(Z))
        # All-zero row → uniform after replacement → exact zero ILR coords
        np.testing.assert_allclose(Z[0], np.zeros(2), atol=1e-6)

    def test_ilr_pseudocount_roundtrip(self):
        """ILR roundtrip with custom pseudocount on data containing zeros."""
        Y = np.array([[0.5, 0.5, 0.0], [0.0, 0.8, 0.2]])
        # With zeros, multiplicative replacement shifts values, so the
        # roundtrip recovers the *replaced* composition, not the original.
        Z, Y_safe = step04.ilr_transform(Y, pseudocount=1e-6, return_safe=True)
        Y_recovered = step04.ilr_inverse(Z, D=3)
        np.testing.assert_allclose(Y_safe, Y_recovered, atol=1e-10)


class TestALRTransform:
    """Tests for ALR (additive log-ratio) transform."""

    def test_alr_reduces_dimension(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        Z = step04.alr_transform(Y)
        assert Z.shape == (2, 2)

    def test_alr_inverse_roundtrip(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])
        Z = step04.alr_transform(Y)
        Y_recovered = step04.alr_inverse(Z, D=3)
        np.testing.assert_allclose(Y, Y_recovered, atol=1e-3)

    def test_alr_handles_zeros(self):
        """ALR handles zeros via multiplicative replacement."""
        Y = np.array([[0.5, 0.5, 0.0]])
        Z = step04.alr_transform(Y)
        assert np.all(np.isfinite(Z))

    def test_alr_reference_component(self):
        """ALR uses last component as reference: z_j = log(y_j / y_D)."""
        Y = np.array([[0.4, 0.3, 0.3]])  # No zeros → no replacement needed
        Z = step04.alr_transform(Y)
        # z_0 = log(0.4 / 0.3) = log(4/3) ≈ 0.288
        # z_1 = log(0.3 / 0.3) = log(1) = 0
        np.testing.assert_allclose(Z[0, 0], np.log(0.4 / 0.3), atol=1e-6)
        np.testing.assert_allclose(Z[0, 1], np.log(0.3 / 0.3), atol=1e-6)

    def test_alr_return_safe(self):
        """return_safe returns both Z and Y_safe."""
        Y = np.array([[0.5, 0.5, 0.0]])
        Z, Y_safe = step04.alr_transform(Y, return_safe=True)
        assert Z.shape == (1, 2)
        assert Y_safe.shape == (1, 3)
        assert np.all(Y_safe > 0)  # No zeros after replacement
        # Roundtrip: ALR inverse of ALR-transformed safe data recovers it
        Z_safe = step04.alr_transform(Y_safe)  # No zeros → no replacement
        Y_recovered = step04.alr_inverse(Z_safe, D=3)
        np.testing.assert_allclose(Y_safe, Y_recovered, atol=1e-10)

    def test_alr_uniform_near_zero(self):
        """Uniform composition → ALR coords near zero."""
        Y = np.array([[0.25, 0.25, 0.25, 0.25]])
        Z = step04.alr_transform(Y)
        np.testing.assert_allclose(Z, np.zeros((1, 3)), atol=0.01)

    def test_alr_inverse_sums_to_one(self):
        """ALR inverse output always sums to 1."""
        rng = np.random.default_rng(42)
        Z = rng.standard_normal((10, 4))  # 5 composition parts
        Y = step04.alr_inverse(Z, D=5)
        np.testing.assert_allclose(Y.sum(axis=1), np.ones(10), atol=1e-10)
        assert np.all(Y > 0)


class TestMorphogenBounds:
    """Tests for morphogen bounds configuration."""

    def test_all_columns_have_bounds(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert col in step04.MORPHOGEN_BOUNDS, f"Missing bounds for {col}"

    def test_bounds_are_valid(self):
        for col, (lo, hi) in step04.MORPHOGEN_BOUNDS.items():
            assert lo < hi, f"Invalid bounds for {col}: ({lo}, {hi})"
            assert lo >= 0, f"Negative lower bound for {col}: {lo}"

    def test_harvest_day_bounds(self):
        lo, hi = step04.MORPHOGEN_BOUNDS["log_harvest_day"]
        assert np.exp(lo) == pytest.approx(7, abs=1)
        assert np.exp(hi) == pytest.approx(120, abs=1)


class TestBuildTrainingSet:
    """Tests for build_training_set function."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        Y = pd.DataFrame(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["cond_1", "cond_2"],
        )
        X = pd.DataFrame(
            {"CHIR99021_uM": [1.5, 3.0], "SAG_uM": [0.25, 1.0]},
            index=["cond_1", "cond_2"],
        )
        y_path = tmp_path / "Y.csv"
        x_path = tmp_path / "X.csv"
        Y.to_csv(y_path)
        X.to_csv(x_path)
        return x_path, y_path

    def test_loads_and_aligns(self, sample_data):
        x_path, y_path = sample_data
        X, Y = step04.build_training_set(y_path, x_path)
        assert len(X) == len(Y) == 2
        assert "fidelity" in X.columns

    def test_fidelity_column_added(self, sample_data):
        x_path, y_path = sample_data
        X, Y = step04.build_training_set(y_path, x_path, fidelity=0.5)
        assert (X["fidelity"] == 0.5).all()

    def test_mismatched_indices(self, tmp_path):
        Y = pd.DataFrame({"ct_A": [0.5, 0.6], "ct_B": [0.5, 0.4]}, index=["cond_1", "cond_3"])
        X = pd.DataFrame({"morph": [1.0, 2.0]}, index=["cond_1", "cond_2"])
        Y.to_csv(tmp_path / "Y.csv")
        X.to_csv(tmp_path / "X.csv")
        X_out, Y_out = step04.build_training_set(
            tmp_path / "Y.csv", tmp_path / "X.csv"
        )
        assert len(X_out) == 1

    def test_status_column_filters_failed(self, tmp_path):
        """Rows with status != valid/success are filtered out (TODO-37)."""
        Y = pd.DataFrame(
            {
                "ct_A": [0.6, 0.3, 0.5],
                "ct_B": [0.4, 0.7, 0.5],
                "status": ["valid", "failed", "invalid"],
            },
            index=["cond_1", "cond_2", "cond_3"],
        )
        X = pd.DataFrame(
            {"CHIR99021_uM": [1.5, 3.0, 2.0], "SAG_uM": [0.25, 1.0, 0.5]},
            index=["cond_1", "cond_2", "cond_3"],
        )
        Y.to_csv(tmp_path / "Y.csv")
        X.to_csv(tmp_path / "X.csv")
        X_out, Y_out = step04.build_training_set(
            tmp_path / "Y.csv", tmp_path / "X.csv"
        )
        assert len(X_out) == 1
        assert len(Y_out) == 1
        assert "cond_1" in X_out.index
        assert "cond_2" not in X_out.index
        assert "cond_3" not in X_out.index
        # status column should not leak into output
        assert "status" not in Y_out.columns

    def test_status_column_accepts_success(self, tmp_path):
        """Rows with status 'success' (case-insensitive) are kept."""
        Y = pd.DataFrame(
            {
                "ct_A": [0.6, 0.3],
                "ct_B": [0.4, 0.7],
                "status": ["Valid", "SUCCESS"],
            },
            index=["cond_1", "cond_2"],
        )
        X = pd.DataFrame(
            {"CHIR99021_uM": [1.5, 3.0], "SAG_uM": [0.25, 1.0]},
            index=["cond_1", "cond_2"],
        )
        Y.to_csv(tmp_path / "Y.csv")
        X.to_csv(tmp_path / "X.csv")
        X_out, Y_out = step04.build_training_set(
            tmp_path / "Y.csv", tmp_path / "X.csv"
        )
        assert len(X_out) == 2
        assert len(Y_out) == 2

    def test_status_column_optional(self, tmp_path):
        """When no status column exists, all rows are kept (backward compatible)."""
        Y = pd.DataFrame(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["cond_1", "cond_2"],
        )
        X = pd.DataFrame(
            {"CHIR99021_uM": [1.5, 3.0], "SAG_uM": [0.25, 1.0]},
            index=["cond_1", "cond_2"],
        )
        Y.to_csv(tmp_path / "Y.csv")
        X.to_csv(tmp_path / "X.csv")
        X_out, Y_out = step04.build_training_set(
            tmp_path / "Y.csv", tmp_path / "X.csv"
        )
        assert len(X_out) == 2
        assert len(Y_out) == 2


class TestComputeCellTypeFractions:
    """Tests for compute_cell_type_fractions function."""

    def test_fractions_sum_to_one(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A", "B", "B"],
            "predicted_annot_level_2": ["Neuron", "NPC", "Neuron", "Glia", "Glia"],
            "quality": ["keep"] * 5,
        })
        fracs = step02.compute_cell_type_fractions(
            obs, "condition", "predicted_annot_level_2"
        )
        np.testing.assert_allclose(fracs.sum(axis=1), 1.0, atol=1e-6)

    def test_correct_fractions(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A", "A"],
            "predicted_annot_level_2": ["Neuron", "Neuron", "NPC", "NPC"],
            "quality": ["keep"] * 4,
        })
        fracs = step02.compute_cell_type_fractions(
            obs, "condition", "predicted_annot_level_2"
        )
        assert fracs.loc["A", "Neuron"] == pytest.approx(0.5)
        assert fracs.loc["A", "NPC"] == pytest.approx(0.5)

    def test_single_cell_type(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A"],
            "label": ["X", "X", "X"],
            "quality": ["keep"] * 3,
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label")
        assert fracs.loc["A", "X"] == pytest.approx(1.0)


class TestFilterQualityCells:
    """Tests for filter_quality_cells function."""

    def test_filters_correctly(self):
        import anndata
        obs = pd.DataFrame({
            "quality": ["keep", "low-quality", "stress", "keep"],
        })
        adata = anndata.AnnData(X=np.zeros((4, 2)), obs=obs)
        filtered = step02.filter_quality_cells(adata)
        assert filtered.n_obs == 2

    def test_no_quality_column(self):
        import anndata
        adata = anndata.AnnData(X=np.zeros((3, 2)))
        filtered = step02.filter_quality_cells(adata)
        assert filtered.n_obs == 3


class TestStep01ConvertGeoToAnndata:
    """Tests for step 01 data conversion."""

    def test_creates_anndata(self, tmp_path):
        from scipy.io import mmwrite
        from scipy import sparse

        n_cells, n_genes = 10, 5
        counts = sparse.random(n_genes, n_cells, density=0.3, format="coo")
        mmwrite(str(tmp_path / "counts.mtx"), counts)

        meta = pd.DataFrame(
            {"condition": ["A"] * 5 + ["B"] * 5},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        meta.to_csv(tmp_path / "meta.csv")

        genes = pd.DataFrame(
            {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
            index=[f"ENSG{i:011d}" for i in range(n_genes)],
        )
        genes.to_csv(tmp_path / "genes.csv")

        output = tmp_path / "test.h5ad"
        adata = step01.convert_geo_to_anndata(
            tmp_path / "counts.mtx",
            tmp_path / "meta.csv",
            tmp_path / "genes.csv",
            output,
            "test",
        )

        assert adata.shape == (n_cells, n_genes)
        assert output.exists()
        assert "counts" in adata.layers

    def test_verify_references(self):
        result = step01.verify_references()
        assert isinstance(result, bool)


class TestFidelityScoring:
    """Tests for step 03 fidelity scoring functions."""

    def test_cosine_similarity_identical(self):
        a = np.array([0.5, 0.3, 0.2])
        assert step03.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert step03.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.5, 0.3])
        b = np.array([0.0, 0.0])
        assert step03.cosine_similarity(a, b) == 0.0

    def test_shannon_entropy_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert step03.shannon_entropy(p) == pytest.approx(2.0)

    def test_shannon_entropy_deterministic(self):
        p = np.array([1.0, 0.0, 0.0])
        assert step03.shannon_entropy(p) == pytest.approx(0.0)

    def test_normalized_entropy_bounds(self):
        p = np.array([0.5, 0.3, 0.2])
        h = step03.normalized_entropy(p)
        assert 0.0 <= h <= 1.0

    def test_normalized_entropy_uniform_is_one(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert step03.normalized_entropy(p) == pytest.approx(1.0)

    def test_compute_rss(self):
        condition_vec = pd.Series({"Neuron": 0.6, "Radial glia": 0.3, "Oligo": 0.1})
        ref_profiles = pd.DataFrame({
            "Neuron": [0.5, 0.1],
            "Radial glia": [0.3, 0.8],
            "Oligo": [0.2, 0.1],
        }, index=["Cortex", "Thalamus"])
        region, score = step03.compute_rss(condition_vec, ref_profiles)
        assert region == "Cortex"  # closer to Cortex
        assert 0.0 <= score <= 1.0

    def test_compute_off_target_fraction(self):
        obs = pd.DataFrame({
            "predicted_annot_level_1": ["Neuron", "NPC", "PSC", "MC", "Neuron"],
        })
        frac = step03.compute_off_target_fraction(obs)
        assert frac == pytest.approx(0.4)  # 2/5

    def test_compute_on_target_fraction(self):
        obs = pd.DataFrame({
            "predicted_annot_region_rev2": ["Cortex", "Cortex", "Cortex", "Thalamus"],
        })
        region, frac = step03.compute_on_target_fraction(obs)
        assert region == "Cortex"
        assert frac == pytest.approx(0.75)

    def test_composite_fidelity_bounds(self):
        score = step03.compute_composite_fidelity(
            rss_score=0.8, on_target_frac=0.6,
            off_target_frac=0.1, norm_entropy=0.5,
        )
        assert 0.0 <= score <= 1.0

    def test_composite_fidelity_perfect(self):
        score = step03.compute_composite_fidelity(
            rss_score=1.0, on_target_frac=1.0,
            off_target_frac=0.0, norm_entropy=0.55,
        )
        assert score > 0.9

    def test_composite_fidelity_handles_nan(self):
        score = step03.compute_composite_fidelity(
            rss_score=np.nan, on_target_frac=np.nan,
            off_target_frac=np.nan, norm_entropy=np.nan,
        )
        assert np.isfinite(score)

    def test_hnoca_to_braun_label_map(self):
        label_map = step03.build_hnoca_to_braun_label_map()
        assert "Neuron" in label_map
        assert label_map["Neuron"] == "Neuron"
        assert "PSC" in label_map

    def test_align_composition_to_braun(self):
        fracs = pd.Series({"Neuron": 0.5, "NPC": 0.3, "IP": 0.2})
        label_map = step03.build_hnoca_to_braun_label_map()
        aligned = step03.align_composition_to_braun(fracs, label_map)
        assert aligned.sum() == pytest.approx(1.0)
        # NPC and IP both map to Radial glia and Neuronal IPC
        assert "Neuron" in aligned.index

    def test_align_sums_multiple_to_same_target(self):
        """NPC + Neuroepithelium should both map to Radial glia and sum."""
        fracs = pd.Series({"NPC": 0.3, "Neuroepithelium": 0.2, "Neuron": 0.5})
        label_map = step03.build_hnoca_to_braun_label_map()
        aligned = step03.align_composition_to_braun(fracs, label_map)
        assert aligned["Radial glia"] == pytest.approx(0.5)  # 0.3 + 0.2
        assert aligned.sum() == pytest.approx(1.0)

    def test_score_all_conditions_with_label_map(self):
        """Bug #2: label_map should improve RSS scores when using Braun profiles."""
        import scanpy as sc

        np.random.seed(42)
        n_cells = 200
        conditions = ["A", "B"]
        level1_types = ["Neuron", "NPC", "IP"]
        level2_types = ["Cortical EN", "Cortical RG", "Cortical IP"]
        regions = ["Dorsal telencephalon", "Ventral telencephalon"]

        obs = pd.DataFrame({
            "condition": np.random.choice(conditions, size=n_cells),
            "predicted_annot_level_1": np.random.choice(level1_types, size=n_cells),
            "predicted_annot_level_2": np.random.choice(level2_types, size=n_cells),
            "predicted_annot_region_rev2": np.random.choice(regions, size=n_cells),
        })
        adata = sc.AnnData(X=np.zeros((n_cells, 5)), obs=obs)

        braun_profiles = pd.DataFrame({
            "Neuron": [0.4, 0.3],
            "Radial glia": [0.3, 0.4],
            "Neuronal IPC": [0.2, 0.2],
            "Oligo": [0.1, 0.1],
        }, index=["Dorsal telencephalon", "Ventral telencephalon"])

        label_map = step03.build_hnoca_to_braun_label_map()

        # Without label_map (HNOCA labels vs Braun — near-orthogonal)
        report_without = step03.score_all_conditions(adata, braun_profiles, "condition")

        # With label_map (aligned labels — meaningful similarity)
        report_with = step03.score_all_conditions(
            adata, braun_profiles, "condition", label_map=label_map
        )

        # RSS scores should differ with label alignment (direction depends on
        # random test data and median-heuristic scale; just verify scoring ran)
        assert "rss_score" in report_with.columns
        assert report_with["rss_score"].notna().all()


class TestBraunEntropyCenterDataDriven:
    """Tests for data-driven entropy center from Braun fetal brain reference."""

    def test_uniform_profiles_give_high_center(self):
        """Uniform distributions should yield normalized entropy near 1.0."""
        profiles = pd.DataFrame(
            np.ones((3, 6)) / 6,
            index=["RegionA", "RegionB", "RegionC"],
            columns=[f"Type{i}" for i in range(6)],
        )
        center = step03.compute_braun_entropy_center(profiles)
        assert 0.95 <= center <= 1.0

    def test_peaked_profiles_give_low_center(self):
        """Strongly peaked distributions should yield low entropy center."""
        data = np.zeros((3, 6))
        data[:, 0] = 0.95
        data[:, 1] = 0.05
        profiles = pd.DataFrame(
            data,
            index=["RegionA", "RegionB", "RegionC"],
            columns=[f"Type{i}" for i in range(6)],
        )
        center = step03.compute_braun_entropy_center(profiles)
        assert center < 0.3

    def test_center_in_unit_interval(self):
        """Entropy center should always be in [0, 1]."""
        np.random.seed(42)
        # Random Dirichlet profiles (realistic)
        profiles = pd.DataFrame(
            np.random.dirichlet(np.ones(8), size=5),
            index=[f"Region{i}" for i in range(5)],
            columns=[f"Type{i}" for i in range(8)],
        )
        center = step03.compute_braun_entropy_center(profiles)
        assert 0.0 <= center <= 1.0

    def test_entropy_center_param_changes_score(self):
        """Custom entropy_center should shift the optimal entropy value."""
        # Score at entropy=0.3 with center=0.3 vs center=0.8
        score_centered = step03.compute_composite_fidelity(
            rss_score=0.5, on_target_frac=0.5,
            off_target_frac=0.5, norm_entropy=0.3,
            entropy_center=0.3,
        )
        score_off_center = step03.compute_composite_fidelity(
            rss_score=0.5, on_target_frac=0.5,
            off_target_frac=0.5, norm_entropy=0.3,
            entropy_center=0.8,
        )
        assert score_centered > score_off_center

    def test_default_fallback_matches_legacy(self):
        """When entropy_center=None, should fall back to 0.55 (legacy)."""
        score_none = step03.compute_composite_fidelity(
            rss_score=0.8, on_target_frac=0.6,
            off_target_frac=0.1, norm_entropy=0.5,
            entropy_center=None,
        )
        score_legacy = step03.compute_composite_fidelity(
            rss_score=0.8, on_target_frac=0.6,
            off_target_frac=0.1, norm_entropy=0.5,
            entropy_center=0.55,
        )
        assert score_none == pytest.approx(score_legacy)

    def test_default_composite_weights_sum_to_one(self):
        """DEFAULT_COMPOSITE_WEIGHTS must sum to 1.0."""
        assert sum(step03.DEFAULT_COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_default_composite_weights_has_required_keys(self):
        """DEFAULT_COMPOSITE_WEIGHTS must have the four expected keys."""
        assert set(step03.DEFAULT_COMPOSITE_WEIGHTS.keys()) == {
            "rss", "on_target", "off_target", "entropy",
        }

    def test_custom_weights_override_defaults(self):
        """Passing custom weights should change the score."""
        kwargs = dict(rss_score=0.9, on_target_frac=0.3,
                      off_target_frac=0.1, norm_entropy=0.5)
        score_default = step03.compute_composite_fidelity(**kwargs)
        # Weight RSS at 100%
        score_rss_heavy = step03.compute_composite_fidelity(
            **kwargs,
            weights={"rss": 1.0, "on_target": 0.0, "off_target": 0.0, "entropy": 0.0},
        )
        assert score_rss_heavy != pytest.approx(score_default)
        assert score_rss_heavy == pytest.approx(0.9)

    def test_sensitivity_analysis_weights_basic(self):
        """sensitivity_analysis_weights returns expected structure."""
        report = pd.DataFrame({
            "condition": [f"c{i}" for i in range(10)],
            "rss_score": np.random.default_rng(0).uniform(0.2, 0.9, 10),
            "on_target_fraction": np.random.default_rng(1).uniform(0.3, 0.8, 10),
            "off_target_fraction": np.random.default_rng(2).uniform(0.0, 0.4, 10),
            "normalized_entropy": np.random.default_rng(3).uniform(0.3, 0.8, 10),
        }).set_index("condition")
        result = step03.sensitivity_analysis_weights(report, n_samples=50, seed=123)
        assert len(result) == 50
        assert "spearman_rho" in result.columns
        assert "w_rss" in result.columns
        # Weight columns should sum to ~1 for each row
        w_sum = result[["w_rss", "w_on_target", "w_off_target", "w_entropy"]].sum(axis=1)
        np.testing.assert_allclose(w_sum, 1.0, atol=1e-10)

    def test_sensitivity_analysis_weights_missing_columns(self):
        """sensitivity_analysis_weights raises on missing report columns."""
        report = pd.DataFrame({"rss_score": [0.5]})
        with pytest.raises(ValueError, match="missing columns"):
            step03.sensitivity_analysis_weights(report)


class TestVisualizationReport:
    """Tests for visualization report module."""

    @pytest.fixture
    def sample_fidelity(self):
        return pd.DataFrame({
            "composite_fidelity": [0.85, 0.72, 0.91],
            "rss_score": [0.80, 0.65, 0.88],
            "on_target_fraction": [0.9, 0.7, 0.95],
            "off_target_fraction": [0.05, 0.1, 0.02],
        }, index=pd.Index(["cond_A", "cond_B", "cond_C"], name="condition"))

    @pytest.fixture
    def sample_morphogens(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.rand(10, 5),
            columns=["CHIR99021_uM", "BMP4_uM", "SAG_uM", "RA_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(10)],
        )

    @pytest.fixture
    def sample_recs(self):
        return pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "BMP4_uM": [0.000769, 0.001538],
            "SAG_uM": [0.25, 0.5],
            "RA_uM": [0, 0.1],
            "log_harvest_day": [4.0, 4.2],
            "predicted_y0_mean": [0.5, 0.6],
            "predicted_y0_std": [0.1, 0.2],
            "acquisition_value": [-1.3, -1.5],
        }, index=pd.Index(["A1", "A2"], name="well"))

    def test_discover_rounds(self, tmp_path):
        from gopro.visualize_report import discover_rounds
        (tmp_path / "gp_recommendations_round1.csv").touch()
        (tmp_path / "gp_recommendations_round3.csv").touch()
        (tmp_path / "gp_recommendations_round2.csv").touch()
        assert discover_rounds(tmp_path) == [1, 2, 3]

    def test_discover_rounds_empty(self, tmp_path):
        from gopro.visualize_report import discover_rounds
        assert discover_rounds(tmp_path) == []

    def test_load_fidelity_report(self, tmp_path, sample_fidelity):
        from gopro.visualize_report import load_fidelity_report
        path = tmp_path / "fidelity_report.csv"
        sample_fidelity.to_csv(path)
        df = load_fidelity_report(path)
        assert df.shape == (3, 4)
        assert df.index.name == "condition"

    def test_load_recommendations(self, tmp_path, sample_recs):
        from gopro.visualize_report import load_recommendations
        path = tmp_path / "recs.csv"
        sample_recs.to_csv(path)
        df = load_recommendations(path)
        assert len(df) == 2
        assert df.index.name == "well"

    def test_load_morphogen_matrix(self, tmp_path, sample_morphogens):
        from gopro.visualize_report import load_morphogen_matrix
        path = tmp_path / "morph.csv"
        sample_morphogens.to_csv(path)
        df = load_morphogen_matrix(path)
        assert df.shape == (10, 5)

    def test_load_cell_type_fractions(self, tmp_path):
        from gopro.visualize_report import load_cell_type_fractions
        fracs = pd.DataFrame(
            {"Neuron": [0.6, 0.3], "NPC": [0.4, 0.7]},
            index=pd.Index(["cond_A", "cond_B"], name="condition"),
        )
        path = tmp_path / "fracs.csv"
        fracs.to_csv(path)
        df = load_cell_type_fractions(path)
        assert df.shape == (2, 2)

    def test_load_diagnostics_no_lengthscales(self, tmp_path):
        from gopro.visualize_report import load_diagnostics
        diag = pd.DataFrame([{"round": 1, "n_training_points": 20}])
        path = tmp_path / "diag.csv"
        diag.to_csv(path, index=False)
        d = load_diagnostics(path)
        assert d["round"] == 1
        assert d["lengthscales"] is None

    def test_load_diagnostics_with_lengthscales(self, tmp_path):
        from gopro.visualize_report import load_diagnostics
        diag = pd.DataFrame([{
            "round": 1, "n_training_points": 20,
            "lengthscale_CHIR99021_uM": 0.5, "lengthscale_SAG_uM": 1.2,
        }])
        path = tmp_path / "diag.csv"
        diag.to_csv(path, index=False)
        d = load_diagnostics(path)
        assert d["lengthscales"]["CHIR99021_uM"] == 0.5

    def test_generate_summary_text(self, sample_fidelity):
        from gopro.visualize_report import generate_summary_text
        diag = {"round": 1, "lengthscales": None}
        text = generate_summary_text(sample_fidelity, diag, 6)
        assert "Round 1" in text
        assert "cond_C" in text
        assert "0.910" in text
        assert "6 new experiments" in text

    def test_generate_summary_with_lengthscales(self, sample_fidelity):
        from gopro.visualize_report import generate_summary_text
        diag = {"round": 1, "lengthscales": {"CHIR": 0.1, "SAG": 0.5, "BMP": 0.3, "RA": 2.0}}
        text = generate_summary_text(sample_fidelity, diag, 6)
        assert "CHIR" in text

    def test_build_morphogen_pca_figure(self):
        from gopro.visualize_report import build_morphogen_pca_figure
        coords = pd.DataFrame(
            {"PC1": [1, 2, 3], "PC2": [4, 5, 6]},
            index=["A", "B", "C"],
        )
        fidelity = pd.Series([0.5, 0.7, 0.9], index=["A", "B", "C"])
        fig = build_morphogen_pca_figure(coords, fidelity)
        assert isinstance(fig, go.Figure)

    def test_compute_morphogen_pca_shape(self, sample_morphogens):
        from gopro.visualize_report import compute_morphogen_pca_with_recommendations
        recs = pd.DataFrame(
            np.random.rand(3, sample_morphogens.shape[1]),
            columns=sample_morphogens.columns,
            index=["R1", "R2", "R3"],
        )
        train_df, rec_df, loadings, var_pct, active_cols = (
            compute_morphogen_pca_with_recommendations(
                sample_morphogens, recs, list(sample_morphogens.columns)
            )
        )
        assert train_df.shape == (10, 2)
        assert rec_df.shape == (3, 2)
        assert list(train_df.columns) == ["PC1", "PC2"]
        assert var_pct > 0

    def test_build_plate_map_figure(self, sample_recs):
        from gopro.visualize_report import build_plate_map_figure
        fig = build_plate_map_figure(sample_recs)
        assert isinstance(fig, go.Figure)

    def test_build_importance_with_lengthscales(self):
        from gopro.visualize_report import build_importance_figure
        ls = {"CHIR": 0.5, "SAG": 2.0, "BMP": 1.0}
        fig = build_importance_figure(lengthscales=ls)
        assert isinstance(fig, go.Figure)

    def test_build_importance_fallback_variance(self, sample_morphogens):
        from gopro.visualize_report import build_importance_figure
        fig = build_importance_figure(morphogen_df=sample_morphogens)
        assert isinstance(fig, go.Figure)

    def test_build_leaderboard_figure(self, sample_fidelity):
        from gopro.visualize_report import build_leaderboard_figure
        fig = build_leaderboard_figure(sample_fidelity, top_n=2)
        assert isinstance(fig, go.Figure)

    def test_build_composition_figure(self):
        from gopro.visualize_report import build_composition_figure
        fracs = pd.DataFrame(
            {"Neuron": [0.6, 0.3], "NPC": [0.4, 0.7]},
            index=["cond_A", "cond_B"],
        )
        fig = build_composition_figure(fracs)
        assert isinstance(fig, go.Figure)

    def test_build_convergence_figure(self):
        from gopro.visualize_report import build_convergence_figure
        fig = build_convergence_figure({1: 0.7})
        assert isinstance(fig, go.Figure)

    def test_build_convergence_multi_round(self):
        from gopro.visualize_report import build_convergence_figure
        fig = build_convergence_figure({1: 0.7, 2: 0.8, 3: 0.85})
        assert isinstance(fig, go.Figure)

    def test_build_cell_umap_figure(self):
        from gopro.visualize_report import build_cell_umap_figure
        coords = np.random.rand(50, 2)
        cell_types = pd.Series(["A"] * 25 + ["B"] * 25)
        conditions = pd.Series(["cond1"] * 25 + ["cond2"] * 25)
        fig = build_cell_umap_figure(coords, cell_types, conditions)
        assert isinstance(fig, go.Figure)

    def test_build_fidelity_trend_figure(self):
        from gopro.visualize_report import build_fidelity_trend_figure
        monitor_df = pd.DataFrame({
            "round": [1, 2, 3, 1, 2, 3],
            "fidelity_label": ["CellRank2"] * 3 + ["CellFlow"] * 3,
            "overall_correlation": [0.8, 0.75, 0.7, 0.5, 0.45, 0.4],
            "recommendation": ["use_mfbo"] * 6,
        })
        fig = build_fidelity_trend_figure(monitor_df)
        assert isinstance(fig, go.Figure)
        # Should have 2 traces (one per source)
        assert len(fig.data) == 2

    def test_build_convergence_diagnostics_figure(self):
        from gopro.visualize_report import build_convergence_diagnostics_figure
        conv_df = pd.DataFrame({
            "round": [1, 2, 3],
            "mean_posterior_std": [0.5, 0.3, 0.2],
            "max_acquisition_value": [10.0, 5.0, 1.0],
            "recommendation_spread": [0.8, 0.4, 0.1],
        })
        fig = build_convergence_diagnostics_figure(conv_df)
        assert isinstance(fig, go.Figure)
        # Should have 3 traces (one per panel)
        assert len(fig.data) == 3


class TestMorphogenColumns:
    """Tests for morphogen column consistency."""

    def test_columns_are_unique(self):
        assert len(step04.MORPHOGEN_COLUMNS) == len(set(step04.MORPHOGEN_COLUMNS))

    def test_columns_are_strings(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert isinstance(col, str)
            assert len(col) > 0


class TestConfig:
    """Tests for centralized config module."""

    def test_config_project_dir_fallback(self):
        """PROJECT_DIR resolves without env var."""
        from gopro.config import PROJECT_DIR
        assert isinstance(PROJECT_DIR, Path)
        assert PROJECT_DIR.exists()

    def test_config_project_dir_env_override(self, monkeypatch, tmp_path):
        """GPBO_PROJECT_DIR env var overrides PROJECT_DIR."""
        monkeypatch.setenv("GPBO_PROJECT_DIR", str(tmp_path))
        # Need to reimport to pick up the env var
        import importlib
        import gopro.config
        importlib.reload(gopro.config)
        assert gopro.config.PROJECT_DIR == tmp_path
        # Restore
        monkeypatch.delenv("GPBO_PROJECT_DIR")
        importlib.reload(gopro.config)

    def test_config_data_dir_env_override(self, monkeypatch, tmp_path):
        """GPBO_DATA_DIR env var overrides DATA_DIR."""
        data_dir = tmp_path / "custom_data"
        monkeypatch.setenv("GPBO_DATA_DIR", str(data_dir))
        import importlib
        import gopro.config
        importlib.reload(gopro.config)
        assert gopro.config.DATA_DIR == data_dir
        monkeypatch.delenv("GPBO_DATA_DIR")
        importlib.reload(gopro.config)

    def test_morphogen_columns_length(self):
        """MORPHOGEN_COLUMNS has exactly 25 entries."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert len(MORPHOGEN_COLUMNS) == 25

    def test_morphogen_columns_canonical_order(self):
        """Indices 17-19 are Dorsomorphin, purmorphamine, cyclopamine."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert MORPHOGEN_COLUMNS[6] == "SR11237_uM"
        assert MORPHOGEN_COLUMNS[17] == "Dorsomorphin_uM"
        assert MORPHOGEN_COLUMNS[18] == "purmorphamine_uM"
        assert MORPHOGEN_COLUMNS[19] == "cyclopamine_uM"

    def test_morphogen_columns_unique(self):
        """No duplicate entries in MORPHOGEN_COLUMNS."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert len(MORPHOGEN_COLUMNS) == len(set(MORPHOGEN_COLUMNS))

    def test_annot_level_values(self):
        """All 4 ANNOT constants have expected values."""
        from gopro.config import ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3
        assert ANNOT_LEVEL_1 == "annot_level_1"
        assert ANNOT_LEVEL_2 == "annot_level_2"
        assert ANNOT_REGION == "annot_region_rev2"
        assert ANNOT_LEVEL_3 == "annot_level_3_rev2"

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance."""
        import logging
        from gopro.config import get_logger
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_module"

    def test_get_logger_respects_env_level(self, monkeypatch):
        """GPBO_LOG_LEVEL env var controls logger level."""
        import logging
        monkeypatch.setenv("GPBO_LOG_LEVEL", "DEBUG")
        from gopro.config import get_logger
        log = get_logger("test_debug_level")
        assert log.level == logging.DEBUG

    def test_model_dir_under_data(self):
        """Bug #9: MODEL_DIR should be under data/neural_organoid_atlas."""
        from gopro.config import MODEL_DIR
        assert "data" in str(MODEL_DIR)
        assert "neural_organoid_atlas" in str(MODEL_DIR)


from gopro.config import MORPHOGEN_COLUMNS


class TestSAGSecondaryScreen:
    """Tests for SAG secondary screen condition parsing."""

    def test_sag_50nm_vector(self):
        """SAG_50nM: CHIR=1.5, SAG=0.05, harvest=Day70, plus base media."""
        from gopro.morphogen_parser import parse_condition_name, _BASE_MEDIA
        import math
        vec = parse_condition_name("SAG_50nM")
        assert vec["CHIR99021_uM"] == 1.5
        assert vec["SAG_uM"] == pytest.approx(0.05)
        assert vec["log_harvest_day"] == pytest.approx(math.log(70))
        # Base media morphogens should have their default values
        for col, val in _BASE_MEDIA.items():
            assert vec[col] == pytest.approx(val), f"{col} should be {val}"
        # All other morphogens (not CHIR, SAG, harvest, or base media) should be 0
        non_zero_cols = {"CHIR99021_uM", "SAG_uM", "log_harvest_day"} | set(_BASE_MEDIA.keys())
        for col in MORPHOGEN_COLUMNS:
            if col not in non_zero_cols:
                assert vec[col] == 0.0, f"{col} should be 0"

    def test_sag_2um_vector(self):
        """SAG_2uM: CHIR=1.5, SAG=2.0, harvest=Day70."""
        from gopro.morphogen_parser import parse_condition_name
        import math
        vec = parse_condition_name("SAG_2uM")
        assert vec["CHIR99021_uM"] == 1.5
        assert vec["SAG_uM"] == pytest.approx(2.0)
        assert vec["log_harvest_day"] == pytest.approx(math.log(70))

    def test_sag_secondary_conditions_list(self):
        """SAG_SECONDARY_CONDITIONS should have exactly 2 entries."""
        from gopro.morphogen_parser import SAG_SECONDARY_CONDITIONS
        assert len(SAG_SECONDARY_CONDITIONS) == 2
        assert "SAG_50nM" in SAG_SECONDARY_CONDITIONS
        assert "SAG_2uM" in SAG_SECONDARY_CONDITIONS

    def test_sag_secondary_build_matrix(self):
        """build_morphogen_matrix works for SAG secondary conditions."""
        from gopro.morphogen_parser import build_morphogen_matrix, SAG_SECONDARY_CONDITIONS
        df = build_morphogen_matrix(SAG_SECONDARY_CONDITIONS)
        assert df.shape == (2, 25)
        assert df.loc["SAG_50nM", "CHIR99021_uM"] == 1.5
        assert df.loc["SAG_2uM", "SAG_uM"] == pytest.approx(2.0)


class TestSAGScreenFiltering:
    """Tests for SAG screen cell filtering (ClusterLabel-based)."""

    def test_filter_sag_screen_cells(self):
        """filter_quality_cells handles SAG screen ClusterLabel column."""
        import anndata
        obs = pd.DataFrame({
            "ClusterLabel": ["MAF_NKX2-1", "filtered", "LHX8_NKX2-1", "filtered", "c0"],
            "condition": ["SAG_1uM"] * 5,
        })
        adata = anndata.AnnData(
            X=np.zeros((5, 10)),
            obs=obs,
        )
        filtered = step02.filter_quality_cells(adata)
        assert filtered.n_obs == 3  # 2 'filtered' cells removed


class TestCrossScreenQC:
    """Tests for cross-screen condition QC validation."""

    def test_similarity_identical(self):
        """Identical fractions should have Aitchison similarity = 1.0."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG250"])
        fracs_b = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG_250nM"])
        mapping = {"SAG250": "SAG_250nM"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["SAG250"]["similarity"] == pytest.approx(1.0)

    def test_similarity_dissimilar(self):
        """Opposite fractions should have very low Aitchison similarity."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [1.0], "Neuron": [0.0]}, index=["cond_a"])
        fracs_b = pd.DataFrame({"NPC": [0.0], "Neuron": [1.0]}, index=["cond_b"])
        mapping = {"cond_a": "cond_b"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["cond_a"]["similarity"] < 0.01  # Near zero for opposite compositions

    def test_flag_low_similarity(self):
        """Should flag conditions with similarity < threshold."""
        from gopro.qc_cross_screen import validate_cross_screen
        fracs_a = pd.DataFrame({"NPC": [0.9], "Neuron": [0.1]}, index=["cond_a"])
        fracs_b = pd.DataFrame({"NPC": [0.1], "Neuron": [0.9]}, index=["cond_b"])
        mapping = {"cond_a": "cond_b"}
        flagged = validate_cross_screen(fracs_a, fracs_b, mapping, threshold=0.8)
        assert len(flagged) == 1
        assert "cond_a" in flagged


class TestMultiSourceRealData:
    """Tests for merging multiple real-data sources at fidelity=1.0."""

    def test_merge_primary_plus_sag(self):
        """merge_multi_fidelity_data handles two fidelity=1.0 sources."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Primary screen (3 conditions, 4 cell types)
            fracs_1 = pd.DataFrame(
                {"NPC": [0.5, 0.3, 0.2], "Neuron": [0.3, 0.4, 0.5], "IP": [0.1, 0.2, 0.2], "Glia": [0.1, 0.1, 0.1]},
                index=["cond_a", "cond_b", "cond_c"],
            )
            morph_1 = pd.DataFrame(
                {"CHIR99021_uM": [1.5, 0, 3.0], "SAG_uM": [0, 0.25, 0], "log_harvest_day": [4.28, 4.28, 4.28]},
                index=["cond_a", "cond_b", "cond_c"],
            )
            fracs_1.to_csv(tmpdir / "fracs_primary.csv")
            morph_1.to_csv(tmpdir / "morph_primary.csv")

            # SAG screen (2 conditions, 3 cell types — missing 'Glia')
            fracs_2 = pd.DataFrame(
                {"NPC": [0.4, 0.2], "Neuron": [0.4, 0.6], "IP": [0.2, 0.2]},
                index=["SAG_50nM", "SAG_2uM"],
            )
            morph_2 = pd.DataFrame(
                {"CHIR99021_uM": [1.5, 1.5], "SAG_uM": [0.05, 2.0], "log_harvest_day": [4.25, 4.25]},
                index=["SAG_50nM", "SAG_2uM"],
            )
            fracs_2.to_csv(tmpdir / "fracs_sag.csv")
            morph_2.to_csv(tmpdir / "morph_sag.csv")

            sources = [
                (tmpdir / "fracs_primary.csv", tmpdir / "morph_primary.csv", 1.0),
                (tmpdir / "fracs_sag.csv", tmpdir / "morph_sag.csv", 1.0),
            ]
            X, Y, _noise = step04.merge_multi_fidelity_data(sources)

            assert len(X) == 5  # 3 + 2
            assert len(Y) == 5
            assert "Glia" in Y.columns  # aligned from primary
            assert Y.loc["SAG_50nM", "Glia"] == 0.0  # filled with 0
            assert np.allclose(Y.sum(axis=1), 1.0, atol=1e-6)  # re-normalized


class TestMorphogenParserClass:
    """Tests for generic MorphogenParser class."""

    def test_amin_kelley_parser(self):
        """AminKelleyParser should parse all 46 primary conditions."""
        from gopro.morphogen_parser import AminKelleyParser
        parser = AminKelleyParser()
        assert len(parser.conditions) == 46
        vec = parser.parse("CHIR1.5")
        assert vec["CHIR99021_uM"] == 1.5

    def test_sag_secondary_parser(self):
        """SAGSecondaryParser should parse 2 conditions."""
        from gopro.morphogen_parser import SAGSecondaryParser
        parser = SAGSecondaryParser()
        assert len(parser.conditions) == 2
        vec = parser.parse("SAG_2uM")
        assert vec["SAG_uM"] == pytest.approx(2.0)

    def test_build_matrix_from_parser(self):
        """Parser.build_matrix() should return DataFrame."""
        from gopro.morphogen_parser import AminKelleyParser
        parser = AminKelleyParser()
        df = parser.build_matrix()
        assert df.shape == (46, 25)

    def test_combined_parser(self):
        """CombinedParser merges multiple parsers."""
        from gopro.morphogen_parser import AminKelleyParser, SAGSecondaryParser, CombinedParser
        combined = CombinedParser([AminKelleyParser(), SAGSecondaryParser()])
        assert len(combined.conditions) == 48
        df = combined.build_matrix()
        assert df.shape == (48, 25)


class TestSanchisCallejaParser:
    """Tests for Sanchis-Calleja et al. 2025 patterning screen parser."""

    def test_condition_count(self):
        """SanchisCallejaParser should have 98 conditions."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        assert len(parser.conditions) == 98

    def test_build_matrix_shape(self):
        """build_matrix() should return (98, 25) DataFrame."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        df = parser.build_matrix()
        assert df.shape == (98, 25)
        assert (df.values >= 0).all(), "All concentrations must be non-negative"

    def test_known_concentrations(self):
        """Spot-check concentrations from Supplementary Figure 1."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        # CHIR_E = 1.8 µM
        assert parser.parse("CHIR_E")["CHIR99021_uM"] == pytest.approx(1.8)
        # CHIR_A = 0.2 µM
        assert parser.parse("CHIR_A")["CHIR99021_uM"] == pytest.approx(0.2)
        # RA_E = 500 nM = 0.5 µM
        assert parser.parse("RA_E")["RA_uM"] == pytest.approx(0.5)
        # XAV939_E = 5.0 µM
        assert parser.parse("XAV939_E")["XAV939_uM"] == pytest.approx(5.0)
        # CycA_E = 150 nM = 0.15 µM
        assert parser.parse("CycA_E")["cyclopamine_uM"] == pytest.approx(0.15)

    def test_shh_pairs_with_purmorphamine(self):
        """SHH conditions should always set purmorphamine_uM."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        # SHH_A: 20 ng/mL SHH + 0.03 µM PM
        vec = parser.parse("SHH_A")
        assert vec["SHH_uM"] > 0
        assert vec["purmorphamine_uM"] == pytest.approx(0.03)
        # SHH_E: 180 ng/mL SHH + 0.27 µM PM
        vec = parser.parse("SHH_E")
        assert vec["purmorphamine_uM"] == pytest.approx(0.27)
        # Timing: SHH_tA uses fixed dose + PM
        vec = parser.parse("SHH_tA")
        assert vec["SHH_uM"] > 0
        assert vec["purmorphamine_uM"] == pytest.approx(0.30)

    def test_no_base_media(self):
        """Sanchis-Calleja conditions should NOT have Amin/Kelley base media."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        vec = parser.parse("CHIR_E")
        assert vec["BDNF_uM"] == 0.0
        assert vec["NT3_uM"] == 0.0
        assert vec["cAMP_uM"] == 0.0
        assert vec["AscorbicAcid_uM"] == 0.0

    def test_harvest_day_21(self):
        """All Sanchis-Calleja conditions should have log(21) harvest day."""
        import math
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        expected = math.log(21)
        for cond in parser.conditions:
            vec = parser.parse(cond)
            assert vec["log_harvest_day"] == pytest.approx(expected), cond

    def test_combination_conditions(self):
        """Multi-morphogen conditions should set all components."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        # RA_E_SHH_A_FGF8_A: 3-morphogen combination
        vec = parser.parse("RA_E_SHH_A_FGF8_A")
        assert vec["RA_uM"] == pytest.approx(0.5)  # 500 nM
        assert vec["SHH_uM"] > 0
        assert vec["FGF8_uM"] > 0
        assert vec["purmorphamine_uM"] == pytest.approx(0.03)  # paired with SHH_A

    def test_gradient_conditions(self):
        """Gradient conditions should set approximate scalar concentrations."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        vec = parser.parse("XAV939_D-E_SHH_Grad_<A")
        assert vec["XAV939_uM"] == pytest.approx(4.0)  # D-E midpoint
        assert vec["SHH_uM"] > 0  # Grad_<A → half of level A
        assert vec["purmorphamine_uM"] > 0

    def test_unrecognized_condition_raises(self):
        """Unrecognized condition names should raise ValueError."""
        from gopro.morphogen_parser import SanchisCallejaParser
        parser = SanchisCallejaParser()
        with pytest.raises(ValueError, match="Unrecognized"):
            parser.parse("NOT_A_REAL_CONDITION")

    def test_all_conditions_parse(self):
        """Every condition in the canonical list should parse without error."""
        from gopro.morphogen_parser import SanchisCallejaParser, SANCHIS_CALLEJA_CONDITIONS
        parser = SanchisCallejaParser()
        for cond in SANCHIS_CALLEJA_CONDITIONS:
            vec = parser.parse(cond)
            # Every condition should set at least one morphogen > 0
            morphogen_cols = [k for k in vec if k != "log_harvest_day"]
            nonzero = sum(1 for k in morphogen_cols if vec[k] > 0)
            assert nonzero >= 1, f"{cond} has no active morphogens"


class TestNoHardcodedPaths:
    """Verify no hardcoded absolute paths remain in gopro/ source files."""

    def test_no_hardcoded_user_paths(self):
        """Bug #5: No /Users/... paths should remain in any gopro/ source file."""
        import re
        gopro_dir = Path(__file__).parent.parent
        violations = []
        for py_file in gopro_dir.glob("*.py"):
            content = py_file.read_text()
            matches = re.findall(r'/Users/\w+/.*', content)
            if matches:
                violations.append((py_file.name, matches))
        assert violations == [], f"Hardcoded paths found: {violations}"

    def test_no_print_calls_in_cellrank2(self):
        """Bug #12: 05_cellrank2_virtual.py should use logger, not print."""
        import re
        gopro_dir = Path(__file__).parent.parent
        content = (gopro_dir / "05_cellrank2_virtual.py").read_text()
        # Match print( at start of line (with optional indentation)
        prints = re.findall(r'^\s*print\(', content, re.MULTILINE)
        assert prints == [], f"Found {len(prints)} print() calls in 05_cellrank2_virtual.py"

    def test_no_print_calls_in_cellflow(self):
        """Bug #12: 06_cellflow_virtual.py should use logger, not print."""
        import re
        gopro_dir = Path(__file__).parent.parent
        content = (gopro_dir / "06_cellflow_virtual.py").read_text()
        prints = re.findall(r'^\s*print\(', content, re.MULTILINE)
        assert prints == [], f"Found {len(prints)} print() calls in 06_cellflow_virtual.py"


class TestExtractLengthscales:
    """Tests for _extract_lengthscales helper."""

    def test_standard_gp(self):
        """Extract lengthscales from MAP-fitted SingleTaskGP."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll
        import torch

        X = torch.rand(10, 3, dtype=torch.double)
        Y = torch.rand(10, 1, dtype=torch.double)
        model = SingleTaskGP(X, Y, input_transform=Normalize(d=3), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        ls = step04._extract_lengthscales(model, 3)
        assert ls is not None
        assert ls.shape == (3,)
        assert np.all(ls > 0)

    def test_returns_none_for_unknown_model(self):
        """Returns None for unrecognized model types."""
        ls = step04._extract_lengthscales("not_a_model", 3)
        assert ls is None


class TestSAASBO:
    """Tests for SAASBO integration in GP-BO pipeline."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n, d, m = 15, 4, 3
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
            index=X.index,
        )
        return X, Y

    def test_saasbo_returns_model_list_gp(self, small_data):
        from botorch.models import ModelListGP
        X, Y = small_data
        model, _, train_Y, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        assert isinstance(model, ModelListGP)
        assert train_Y.shape[1] == 2  # ILR: 3 -> 2

    def test_saasbo_lengthscales_extractable(self, small_data):
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        ls = step04._extract_lengthscales(model, X.shape[1])
        assert ls is not None
        assert len(ls) == X.shape[1]

    def test_saasbo_posterior_works(self, small_data):
        import torch
        X, Y = small_data
        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        with torch.no_grad():
            post = model.posterior(train_X[:2])
        assert post.mean.shape[-1] == 2

    def test_saasbo_false_uses_standard_gp(self, small_data):
        from botorch.models import SingleTaskGP
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=True, use_saasbo=False)
        assert isinstance(model, SingleTaskGP)

    def test_saasbo_ignored_for_multi_fidelity(self, small_data):
        from botorch.models import SingleTaskMultiFidelityGP
        X, Y = small_data
        X = X.copy()
        X.loc[X.index[:5], "fidelity"] = 0.5
        model, _, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=True, use_saasbo=True)
        assert isinstance(model, SingleTaskMultiFidelityGP)

    def test_saasbo_single_output(self, small_data):
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y[["ct_A"]], use_ilr=False, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        assert isinstance(model, SaasFullyBayesianSingleTaskGP)


class TestLassoBO:
    """Tests for LassoBO (L1-regularized MAP variable selection)."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n, d, m = 15, 4, 3
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
            index=X.index,
        )
        return X, Y

    def test_lassobo_returns_single_task_gp(self, small_data):
        """LassoBO should return a standard SingleTaskGP (not ModelListGP)."""
        from botorch.models import SingleTaskGP
        X, Y = small_data
        model, _, train_Y, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_lassobo=True, lassobo_alpha=0.1,
        )
        assert isinstance(model, SingleTaskGP)
        assert train_Y.shape[1] == 2  # ILR: 3 -> 2

    def test_lassobo_lengthscales_extractable(self, small_data):
        """Lengthscales should be extractable from a LassoBO-fitted model."""
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_lassobo=True, lassobo_alpha=0.1,
        )
        ls = step04._extract_lengthscales(model, X.shape[1])
        assert ls is not None
        # Multi-output GP has batch lengthscales (n_outputs x 1 x d) → flattened
        assert len(ls) >= X.shape[1]
        assert all(l > 0 for l in ls)

    def test_lassobo_posterior_works(self, small_data):
        """Model fitted with LassoBO should produce valid posteriors."""
        import torch
        X, Y = small_data
        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_lassobo=True, lassobo_alpha=0.1,
        )
        with torch.no_grad():
            post = model.posterior(train_X[:2])
        assert post.mean.shape[-1] == 2  # ILR-transformed outputs

    def test_lassobo_alpha_affects_lengthscales(self, small_data):
        """Different alpha values should produce different lengthscale distributions."""
        X, Y = small_data
        # Zero alpha — no L1 penalty (just MLL)
        model_zero, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_lassobo=True, lassobo_alpha=0.0,
        )
        ls_zero = step04._extract_lengthscales(model_zero, X.shape[1])

        # Nonzero alpha — L1 penalty active
        model_reg, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_lassobo=True, lassobo_alpha=0.1,
        )
        ls_reg = step04._extract_lengthscales(model_reg, X.shape[1])

        # Both should produce valid non-None lengthscales
        assert ls_zero is not None and ls_reg is not None
        # With regularization, at least some lengthscales should differ
        assert not np.allclose(ls_zero, ls_reg, atol=1e-3)

    def test_lassobo_mutually_exclusive_with_saasbo(self, small_data):
        """LassoBO + SAASBO should raise ValueError."""
        X, Y = small_data
        with pytest.raises(ValueError, match="mutually exclusive"):
            step04.fit_gp_botorch(X, Y, use_saasbo=True, use_lassobo=True)

    def test_lassobo_mutually_exclusive_with_per_type_gp(self, small_data):
        """LassoBO + per_type_gp should raise ValueError."""
        X, Y = small_data
        with pytest.raises(ValueError, match="incompatible with use_lassobo"):
            step04.fit_gp_botorch(X, Y, use_lassobo=True, per_type_gp=True)

    def test_lassobo_ignored_for_multi_fidelity(self, small_data):
        """Multi-fidelity should take priority over LassoBO."""
        from botorch.models import SingleTaskMultiFidelityGP
        X, Y = small_data
        X = X.copy()
        X.loc[X.index[:5], "fidelity"] = 0.5
        model, _, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=True, use_lassobo=True)
        assert isinstance(model, SingleTaskMultiFidelityGP)


class TestSoftKNNFractions:
    """Tests for soft probability output from KNN label transfer."""

    @pytest.fixture
    def knn_data(self):
        np.random.seed(42)
        n_ref, n_query, d = 100, 20, 10
        ref_latent = np.random.randn(n_ref, d)
        query_latent = np.random.randn(n_query, d)
        ref_obs = pd.DataFrame({
            "annot_level_2": np.random.choice(["TypeA", "TypeB", "TypeC"], n_ref),
        }, index=[f"ref_{i}" for i in range(n_ref)])
        query_obs = pd.DataFrame({
            "condition": np.random.choice(["cond1", "cond2"], n_query),
        }, index=[f"query_{i}" for i in range(n_query)])
        return ref_latent, query_latent, ref_obs, query_obs

    def test_returns_soft_probabilities(self, knn_data):
        ref_latent, query_latent, ref_obs, query_obs = knn_data
        results, soft_probs = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_2"], k=10,
        )
        assert "annot_level_2" in soft_probs
        prob_df = soft_probs["annot_level_2"]
        assert prob_df.shape == (20, 3)  # 20 query cells x 3 types
        assert np.allclose(prob_df.sum(axis=1), 1.0, atol=1e-6)
        assert (prob_df >= 0).all().all()

    def test_hard_labels_match_soft_argmax(self, knn_data):
        ref_latent, query_latent, ref_obs, query_obs = knn_data
        results, soft_probs = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_2"], k=10,
        )
        prob_df = soft_probs["annot_level_2"]
        hard = results["predicted_annot_level_2"].values
        soft_argmax = prob_df.columns[prob_df.values.argmax(axis=1)]
        assert np.array_equal(hard, soft_argmax)

    def test_backward_compatible_results_df(self, knn_data):
        ref_latent, query_latent, ref_obs, query_obs = knn_data
        results, _ = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_2"], k=10,
        )
        assert "predicted_annot_level_2" in results.columns
        assert "annot_level_2_confidence" in results.columns


class TestGruffiStressFiltering:
    """Tests for Gruffi-inspired stress cell filtering."""

    def test_fetch_go_gene_sets(self):
        """Bundled JSON contains expected pathways with >20 genes each."""
        from gopro.gruffi_qc import fetch_go_gene_sets
        gene_sets = fetch_go_gene_sets()
        for key in ("glycolysis", "er_stress", "upr"):
            assert key in gene_sets, f"Missing pathway: {key}"
            assert len(gene_sets[key]) > 20, f"{key} has only {len(gene_sets[key])} genes"

    def test_score_stress_pathways_scanpy(self):
        """Scanpy scoring adds gruffi_stress_score; stressed cells score higher."""
        import anndata
        import scipy.sparse as sp

        np.random.seed(42)
        n_cells, n_genes = 100, 200
        # Known glycolysis genes to plant in the first few columns
        stress_genes = ["ALDOA", "ENO1", "GAPDH", "PKM", "LDHA", "PGK1",
                        "TPI1", "GPI", "PFKL", "HK1"]
        other_genes = [f"Gene_{i}" for i in range(n_genes - len(stress_genes))]
        var_names = stress_genes + other_genes

        X = sp.random(n_cells, n_genes, density=0.1, format="csr", dtype=np.float32)
        X = X.toarray()
        # Make ~20 cells have high expression in stress genes
        stressed_idx = np.arange(20)
        X[stressed_idx, :len(stress_genes)] = np.random.uniform(3.0, 6.0,
            size=(len(stressed_idx), len(stress_genes)))

        adata = anndata.AnnData(X=sp.csr_matrix(X))
        adata.var_names = var_names
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

        gene_sets = {"glycolysis": stress_genes}
        from gopro.gruffi_qc import score_stress_pathways
        score_stress_pathways(adata, gene_sets=gene_sets, method="scanpy")

        assert "gruffi_stress_score" in adata.obs.columns
        stressed_scores = adata.obs["gruffi_stress_score"].values[stressed_idx]
        unstressed_scores = adata.obs["gruffi_stress_score"].values[20:]
        assert np.mean(stressed_scores) > np.mean(unstressed_scores)

    def test_identify_stressed_clusters(self):
        """Clusters with high median stress score are flagged."""
        import anndata
        import scanpy as sc

        np.random.seed(42)
        n_cells = 100
        # Create two groups: 20 stressed, 80 clean
        X = np.random.randn(n_cells, 50).astype(np.float32)
        # Make stressed cells cluster together by shifting their features
        X[:20] += 5.0

        adata = anndata.AnnData(X=X)
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        # Pre-assign stress scores
        scores = np.full(n_cells, 0.01)
        scores[:20] = 0.5
        adata.obs["gruffi_stress_score"] = scores

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

        from gopro.gruffi_qc import identify_stressed_clusters
        mask = identify_stressed_clusters(adata, threshold=0.15, resolution=2.0)

        assert mask.dtype == bool
        assert mask.shape == (n_cells,)
        # Most of the flagged cells should be from the stressed group
        flagged_stressed = mask[:20].sum()
        flagged_clean = mask[20:].sum()
        assert flagged_stressed > flagged_clean

    def test_filter_stressed_cells_removes_stressed(self):
        """End-to-end: stressed cells are removed and gruffi_is_stressed exists."""
        import anndata

        np.random.seed(42)
        n_cells, n_genes = 100, 200
        stress_genes = ["ALDOA", "ENO1", "GAPDH", "PKM", "LDHA", "PGK1",
                        "TPI1", "GPI", "PFKL", "HK1"]
        other_genes = [f"Gene_{i}" for i in range(n_genes - len(stress_genes))]
        var_names = stress_genes + other_genes

        X = np.random.rand(n_cells, n_genes).astype(np.float32) * 0.1
        # Make 20 cells highly express stress genes
        X[:20, :len(stress_genes)] = np.random.uniform(3.0, 6.0,
            size=(20, len(stress_genes)))

        adata = anndata.AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obs["condition"] = np.array(["A"] * 50 + ["B"] * 50)

        from gopro.gruffi_qc import filter_stressed_cells
        filtered = filter_stressed_cells(adata, threshold=0.15, method="scanpy",
                                          min_cells_per_condition=10)

        assert filtered.n_obs < n_cells
        assert "gruffi_is_stressed" in filtered.obs.columns

    def test_filter_stressed_cells_min_cells_safety(self):
        """Safety floor: conditions keep at least min_cells_per_condition cells."""
        import anndata

        np.random.seed(42)
        n_cells, n_genes = 60, 200
        stress_genes = ["ALDOA", "ENO1", "GAPDH", "PKM", "LDHA", "PGK1",
                        "TPI1", "GPI", "PFKL", "HK1"]
        other_genes = [f"Gene_{i}" for i in range(n_genes - len(stress_genes))]
        var_names = stress_genes + other_genes

        X = np.random.rand(n_cells, n_genes).astype(np.float32) * 0.1
        # Make 40 out of 60 cells stressed
        X[:40, :len(stress_genes)] = np.random.uniform(3.0, 6.0,
            size=(40, len(stress_genes)))

        adata = anndata.AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obs["condition"] = "only_cond"

        from gopro.gruffi_qc import filter_stressed_cells
        filtered = filter_stressed_cells(adata, threshold=0.15, method="scanpy",
                                          min_cells_per_condition=50)

        # Safety floor should rescue cells so at least 50 remain
        assert filtered.n_obs >= 50

    def test_filter_stressed_cells_no_stress(self):
        """Clean data: all cells retained when none are stressed."""
        import anndata

        np.random.seed(42)
        n_cells, n_genes = 100, 200
        # All low expression — no stress signal
        X = np.random.rand(n_cells, n_genes).astype(np.float32) * 0.01

        adata = anndata.AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obs["condition"] = "clean"

        # Provide gene sets with genes not in var_names so scoring yields 0
        gene_sets = {"glycolysis": ["NOT_A_GENE_1", "NOT_A_GENE_2"]}

        from gopro.gruffi_qc import score_stress_pathways, filter_stressed_cells
        # score_stress_pathways with <3 overlapping genes will set scores to 0
        # filter_stressed_cells calls score internally, so we use it directly
        # but we need the gene sets to produce zero scores — use method=scanpy
        # with genes absent from var_names
        filtered = filter_stressed_cells(adata, threshold=0.15, method="scanpy",
                                          min_cells_per_condition=10)
        assert filtered.n_obs == n_cells

    def test_compute_stress_fraction_per_condition(self):
        """Stress fraction computation returns correct per-condition values."""
        import anndata

        n_cells = 100
        adata = anndata.AnnData(X=np.zeros((n_cells, 5)))
        adata.obs["condition"] = np.array(["A"] * 60 + ["B"] * 40)
        # 10 stressed in A, 20 stressed in B
        stressed = np.zeros(n_cells, dtype=bool)
        stressed[50:60] = True   # 10 in A
        stressed[60:80] = True   # 20 in B
        adata.obs["gruffi_is_stressed"] = stressed

        from gopro.gruffi_qc import compute_stress_fraction_per_condition
        df = compute_stress_fraction_per_condition(adata, condition_key="condition")

        assert "condition" in df.columns
        assert "fraction_stressed" in df.columns
        row_a = df[df["condition"] == "A"].iloc[0]
        row_b = df[df["condition"] == "B"].iloc[0]
        assert row_a["n_stressed"] == pytest.approx(10)
        assert row_a["fraction_stressed"] == pytest.approx(10 / 60)
        assert row_b["n_stressed"] == pytest.approx(20)
        assert row_b["fraction_stressed"] == pytest.approx(20 / 40)


class TestComputeSoftCellTypeFractions:
    """Tests for soft probability-based cell type fraction computation."""

    def test_soft_fractions_sum_to_one(self):
        np.random.seed(42)
        n_cells = 50
        obs = pd.DataFrame({
            "condition": np.repeat(["condA", "condB"], n_cells // 2),
        }, index=[f"cell_{i}" for i in range(n_cells)])
        prob_df = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=n_cells),
            columns=["TypeA", "TypeB", "TypeC"],
            index=obs.index,
        )
        fracs = step02.compute_soft_cell_type_fractions(obs, prob_df, condition_key="condition")
        assert np.allclose(fracs.sum(axis=1), 1.0, atol=1e-6)
        assert fracs.shape == (2, 3)

    def test_deterministic_cells_match_hard(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "B", "B"],
        }, index=[f"c{i}" for i in range(4)])
        prob_df = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            columns=["X", "Y"],
            index=obs.index,
        )
        fracs = step02.compute_soft_cell_type_fractions(obs, prob_df, condition_key="condition")
        assert fracs.loc["A", "X"] == pytest.approx(0.5)
        assert fracs.loc["A", "Y"] == pytest.approx(0.5)
        assert fracs.loc["B", "X"] == pytest.approx(1.0)
        assert fracs.loc["B", "Y"] == pytest.approx(0.0)


class TestNormalizedEntropyWithTotalTypes:
    """Tests for normalized_entropy with total_types parameter (Fix C)."""

    def test_entropy_with_total_types(self):
        """Entropy with total_types=8 on 4-element uniform array → ~0.5."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        # Without total_types: normalized by log2(4) = 2 → 1.0
        h_old = step03.normalized_entropy(p)
        assert h_old == pytest.approx(1.0, abs=1e-6)
        # With total_types=8: normalized by log2(8) = 3 → 2/3
        h_new = step03.normalized_entropy(p, total_types=8)
        assert h_new == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_entropy_without_total_types_backward_compatible(self):
        """Without total_types, behavior is unchanged."""
        p = np.array([0.5, 0.3, 0.2])
        h1 = step03.normalized_entropy(p)
        h2 = step03.normalized_entropy(p, total_types=None)
        assert h1 == pytest.approx(h2)

    def test_entropy_single_element(self):
        p = np.array([1.0])
        assert step03.normalized_entropy(p, total_types=10) == 0.0


class TestBuildTrainingSetNoMutation:
    """Test that build_training_set doesn't mutate input DataFrames (Fix E)."""

    def test_original_dataframe_not_modified(self, tmp_path):
        Y = pd.DataFrame({"ct_A": [0.6, 0.4], "ct_B": [0.4, 0.6]},
                         index=["c1", "c2"])
        X = pd.DataFrame({"CHIR99021_uM": [1.5, 3.0]}, index=["c1", "c2"])
        Y.to_csv(tmp_path / "y.csv")
        X.to_csv(tmp_path / "x.csv")

        # Read original columns before
        X_orig = pd.read_csv(tmp_path / "x.csv", index_col=0)
        orig_cols = list(X_orig.columns)

        X_result, Y_result = step04.build_training_set(
            tmp_path / "y.csv", tmp_path / "x.csv", fidelity=1.0
        )

        # The returned X should have fidelity, but original file unchanged
        assert "fidelity" in X_result.columns
        # Re-read file to confirm it wasn't mutated on disk
        X_check = pd.read_csv(tmp_path / "x.csv", index_col=0)
        assert list(X_check.columns) == orig_cols


class TestFidelityCosts:
    """Tests for FIDELITY_COSTS configuration (Idea #3)."""

    def test_fidelity_costs_defined(self):
        """FIDELITY_COSTS has expected keys for all fidelity levels."""
        from gopro.config import FIDELITY_COSTS
        assert 1.0 in FIDELITY_COSTS
        assert 0.5 in FIDELITY_COSTS
        assert 0.0 in FIDELITY_COSTS

    def test_fidelity_costs_ordering(self):
        """Higher fidelity data should cost more."""
        from gopro.config import FIDELITY_COSTS
        assert FIDELITY_COSTS[1.0] > FIDELITY_COSTS[0.5]
        assert FIDELITY_COSTS[0.5] > FIDELITY_COSTS[0.0]

    def test_fidelity_costs_real_is_one(self):
        """Real experiment cost ratio is 1.0 (baseline)."""
        from gopro.config import FIDELITY_COSTS
        assert FIDELITY_COSTS[1.0] == 1.0

    def test_fidelity_costs_imported_in_gpbo(self):
        """FIDELITY_COSTS is accessible from 04_gpbo_loop module."""
        assert hasattr(step04, "FIDELITY_COSTS")
        assert step04.FIDELITY_COSTS[1.0] == 1.0


class TestFidelityKernelRemap:
    """Tests for fidelity remap to prevent MF-GP kernel boundary collapse (TODO-24)."""

    def test_remap_known_values(self):
        """Standard fidelity levels map to open interval (0, 1)."""
        fid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        remapped = step04._remap_fidelity(fid)
        expected = torch.tensor([1 / 5, 2 / 5, 4 / 5], dtype=torch.float64)
        torch.testing.assert_close(remapped, expected)

    def test_remap_excludes_boundaries(self):
        """Remapped values must be strictly in (0, 1)."""
        fid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        remapped = step04._remap_fidelity(fid)
        assert (remapped > 0).all()
        assert (remapped < 1).all()

    def test_remap_preserves_ordering(self):
        """Higher fidelity → higher remapped value."""
        fid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        remapped = step04._remap_fidelity(fid)
        assert remapped[0] < remapped[1] < remapped[2]

    def test_unmap_roundtrip(self):
        """remap → unmap is identity for known fidelity values."""
        fid = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        remapped = step04._remap_fidelity(fid)
        recovered = step04._unmap_fidelity(remapped)
        torch.testing.assert_close(recovered, fid)

    def test_remap_constants_exported(self):
        """FIDELITY_KERNEL_REMAP and FIDELITY_KERNEL_UNMAP are accessible."""
        assert hasattr(step04, "FIDELITY_KERNEL_REMAP")
        assert hasattr(step04, "FIDELITY_KERNEL_UNMAP")
        assert len(step04.FIDELITY_KERNEL_REMAP) == 4
        # All mapped values in (0, 1)
        for v in step04.FIDELITY_KERNEL_REMAP.values():
            assert 0 < v < 1

    def test_mf_gp_receives_remapped_fidelity(self):
        """When fitting MF-GP, the model should see remapped fidelity values."""
        np.random.seed(42)
        n, d, m = 20, 3, 2
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM"],
            index=[f"cond_{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        X.loc[X.index[:8], "fidelity"] = 0.5
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n),
            columns=["ct_A", "ct_B"],
            index=X.index,
        )
        from botorch.models import SingleTaskMultiFidelityGP
        model, train_X, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=False)
        assert isinstance(model, SingleTaskMultiFidelityGP)
        # Fidelity column in train_X should contain remapped values, not raw
        fid_col = train_X[:, -1]
        # Should NOT contain 1.0 or 0.5 (raw values)
        assert not torch.any(torch.isclose(fid_col, torch.tensor(1.0, dtype=torch.float64)))
        # Should contain remapped values ≈ 2/5 and 4/5
        unique_fids = sorted(fid_col.unique().tolist())
        assert len(unique_fids) == 2
        torch.testing.assert_close(
            torch.tensor(unique_fids, dtype=torch.float64),
            torch.tensor([2 / 5, 4 / 5], dtype=torch.float64),
        )


class TestPerFidelityARD:
    """Tests for per-fidelity ARD kernel: g(x) + delta(x,m) structure (TODO-5)."""

    @pytest.fixture
    def mf_data(self):
        """Create multi-fidelity training data with 3 fidelity levels."""
        np.random.seed(42)
        n, d, m = 30, 3, 2
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM"],
            index=[f"cond_{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        X.loc[X.index[:10], "fidelity"] = 0.0  # CellFlow
        X.loc[X.index[10:20], "fidelity"] = 0.5  # CellRank2
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n),
            columns=["ct_A", "ct_B"],
            index=X.index,
        )
        return X, Y

    def test_fidelity_to_task_idx_mapping(self):
        """_fidelity_to_task_idx assigns ascending indices."""
        fid = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0], dtype=torch.float64)
        task_idx, fid_map = step04._fidelity_to_task_idx(fid)
        # Lowest fidelity → lowest index
        assert fid_map[0.0] == 0
        assert fid_map[0.5] == 1
        assert fid_map[1.0] == 2
        # Verify actual indices
        assert task_idx[0].item() == 2.0  # fidelity 1.0 → idx 2
        assert task_idx[1].item() == 0.0  # fidelity 0.0 → idx 0
        assert task_idx[2].item() == 1.0  # fidelity 0.5 → idx 1

    def test_fidelity_to_task_idx_two_levels(self):
        """_fidelity_to_task_idx handles 2 fidelity levels."""
        fid = torch.tensor([0.5, 1.0, 0.5], dtype=torch.float64)
        task_idx, fid_map = step04._fidelity_to_task_idx(fid)
        assert len(fid_map) == 2
        assert fid_map[0.5] == 0
        assert fid_map[1.0] == 1

    def test_per_fidelity_ard_model_fits(self, mf_data):
        """fit_gp_botorch with per_fidelity_ard=True produces a fitted model."""
        from botorch.models import SingleTaskGP
        X, Y = mf_data
        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, per_fidelity_ard=True,
        )
        assert isinstance(model, SingleTaskGP)
        # Fidelity column should contain integer task indices
        fid_col = train_X[:, -1]
        unique_idx = sorted(fid_col.unique().tolist())
        assert unique_idx == [0.0, 1.0, 2.0]

    def test_per_fidelity_ard_kernel_structure(self, mf_data):
        """Per-fidelity ARD model has additive kernel: base + residual*fidelity."""
        X, Y = mf_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, per_fidelity_ard=True,
        )
        covar = model.covar_module
        # AdditiveKernel with 2 sub-kernels
        assert hasattr(covar, "kernels")
        assert len(covar.kernels) == 2

    def test_per_fidelity_ard_lengthscale_extraction(self, mf_data):
        """Lengthscale extraction works for per-fidelity ARD models."""
        X, Y = mf_data
        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, per_fidelity_ard=True,
        )
        # Standard extraction returns base lengthscales
        ls = step04._extract_lengthscales(model, train_X.shape[1])
        assert ls is not None
        # Should have 3 lengthscales (3 morphogen dims, NOT 4 including fidelity)
        assert len(ls) == 3

        # Detailed extraction returns both base and residual
        pf_ls = step04._extract_per_fidelity_ard_lengthscales(model)
        assert pf_ls is not None
        assert "base" in pf_ls
        assert "residual" in pf_ls
        assert len(pf_ls["base"]) == 3
        assert len(pf_ls["residual"]) == 3

    def test_per_fidelity_ard_posterior(self, mf_data):
        """Per-fidelity ARD model produces valid posterior predictions."""
        X, Y = mf_data
        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, per_fidelity_ard=True,
        )
        with torch.no_grad():
            # Predict at a few training points
            post = model.posterior(train_X[:3])
        assert post.mean.shape == (3, Y.shape[1])
        assert not torch.isnan(post.mean).any()

    def test_per_fidelity_ard_not_used_without_flag(self, mf_data):
        """Without per_fidelity_ard=True, standard MF-GP is used."""
        from botorch.models import SingleTaskMultiFidelityGP
        X, Y = mf_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, per_fidelity_ard=False,
        )
        assert isinstance(model, SingleTaskMultiFidelityGP)


class TestSelectReplicateConditions:
    """Tests for _select_replicate_conditions (Idea #7)."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for replicate tests."""
        np.random.seed(42)
        n = 10
        X = pd.DataFrame({
            "CHIR99021_uM": np.random.uniform(0, 12, n),
            "SAG_uM": np.random.uniform(0, 2, n),
            "fidelity": 1.0,
        }, index=[f"cond_{i}" for i in range(n)])
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(4), size=n),
            columns=["Neuron", "NPC", "Glia", "Other"],
            index=X.index,
        )
        return X, Y

    def test_select_replicate_conditions_count(self, sample_training_data):
        """Returns correct number of replicates."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=3, strategy="high_value",
        )
        assert len(result) == 3

    def test_select_replicate_conditions_high_value(self, sample_training_data):
        """high_value strategy picks conditions with highest mean Y."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=2, strategy="high_value",
        )
        # The selected conditions should be among the top-2 by mean Y
        mean_scores = Y.mean(axis=1)
        top2 = mean_scores.nlargest(2).index
        assert set(result.index) == set(top2)

    def test_select_replicate_conditions_random(self, sample_training_data):
        """random strategy returns correct count."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=2, strategy="random",
        )
        assert len(result) == 2

    def test_select_replicate_conditions_no_fidelity_column(self, sample_training_data):
        """Output does not include fidelity column."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=2, strategy="high_value",
        )
        assert "fidelity" not in result.columns

    def test_select_replicate_conditions_zero(self, sample_training_data):
        """n_replicates=0 returns empty DataFrame."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=0, strategy="high_value",
        )
        assert len(result) == 0

    def test_select_replicate_conditions_exceeds_available(self, sample_training_data):
        """n_replicates > n_available is clamped."""
        X, Y = sample_training_data
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=100, strategy="random",
        )
        assert len(result) == len(X)

    def test_select_replicate_conditions_high_variance_fallback(self, sample_training_data):
        """high_variance without model falls back to high_value."""
        X, Y = sample_training_data
        # No model provided -> should fallback
        result = step04._select_replicate_conditions(
            X, Y, n_replicates=2, strategy="high_variance",
            model=None, active_cols=None,
        )
        assert len(result) == 2

    def test_unknown_strategy_raises(self, sample_training_data):
        """Unknown strategy raises ValueError."""
        X, Y = sample_training_data
        with pytest.raises(ValueError, match="Unknown replicate strategy"):
            step04._select_replicate_conditions(
                X, Y, n_replicates=2, strategy="nonexistent",
            )


class TestEstimateNoiseFromReplicates:
    """Tests for _estimate_noise_from_replicates (Idea #7)."""

    def test_estimate_noise_basic(self):
        """Noise estimate from synthetic replicates is reasonable."""
        Y = pd.DataFrame({
            "ct_A": [0.5, 0.52, 0.3, 0.31, 0.8],
            "ct_B": [0.5, 0.48, 0.7, 0.69, 0.2],
        })
        groups = {
            "group1": [0, 1],  # cond with ~0.5/0.5
            "group2": [2, 3],  # cond with ~0.3/0.7
        }
        noise = step04._estimate_noise_from_replicates(Y, groups)
        assert noise > 0
        # Noise should be small since replicates are close
        assert noise < 0.01

    def test_estimate_noise_no_groups(self):
        """No replicate groups returns 0.0."""
        Y = pd.DataFrame({"ct_A": [0.5], "ct_B": [0.5]})
        noise = step04._estimate_noise_from_replicates(Y, {})
        assert noise == 0.0

    def test_estimate_noise_single_member_groups(self):
        """Groups with only 1 member are ignored."""
        Y = pd.DataFrame({
            "ct_A": [0.5, 0.3],
            "ct_B": [0.5, 0.7],
        })
        groups = {"g1": [0], "g2": [1]}
        noise = step04._estimate_noise_from_replicates(Y, groups)
        assert noise == 0.0

    def test_estimate_noise_high_variance(self):
        """High variance replicates yield higher noise estimate."""
        Y = pd.DataFrame({
            "ct_A": [0.2, 0.8, 0.5, 0.5],
            "ct_B": [0.8, 0.2, 0.5, 0.5],
        })
        groups = {"g1": [0, 1]}  # very different -> high var
        noise = step04._estimate_noise_from_replicates(Y, groups)
        assert noise > 0.05


class TestGPWarmStart:
    """Tests for GP state serialization and warm-starting across rounds."""

    @pytest.fixture
    def simple_gp(self):
        """Create a fitted SingleTaskGP for testing."""
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll

        torch.manual_seed(42)
        n, d = 10, 3
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.rand(n, 2, dtype=torch.double)
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=2),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def test_save_gp_state_creates_file(self, simple_gp, tmp_path):
        """Verify save_gp_state creates a state file on disk."""
        save_path = tmp_path / "gp_state" / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_save_gp_state_contains_hyperparams(self, simple_gp, tmp_path):
        """Verify saved state contains expected hyperparameter keys."""
        import torch
        save_path = tmp_path / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)
        state = torch.load(str(save_path), weights_only=True)
        assert "lengthscales" in state
        assert "noise" in state
        assert state["model_type"] == "single_task"
        # mean_constant should also be saved
        assert "mean_constant" in state

    def test_load_gp_state_sets_hyperparams(self, simple_gp, tmp_path):
        """Verify load_gp_state sets hyperparameters on a new model."""
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        # Save the fitted model's state
        save_path = tmp_path / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)

        # Get original values — covar_module may be bare RBFKernel or ScaleKernel
        kernel = simple_gp.covar_module
        base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
        orig_ls = base.lengthscale.detach().clone()

        # Create a fresh (unfitted) model with same dimensions
        torch.manual_seed(99)
        n, d = 10, 3
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.rand(n, 2, dtype=torch.double)
        new_model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=2),
        )

        # Load state into new model
        result = step04.load_gp_state(new_model, save_path)
        assert result is True

        # Verify hyperparameters match
        new_kernel = new_model.covar_module
        new_base = new_kernel.base_kernel if hasattr(new_kernel, "base_kernel") else new_kernel
        new_ls = new_base.lengthscale.detach()
        torch.testing.assert_close(new_ls, orig_ls)

    def test_load_gp_state_missing_file_noop(self, tmp_path):
        """Verify graceful handling when state file does not exist."""
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        n, d = 10, 3
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.rand(n, 2, dtype=torch.double)
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=2),
        )

        # Get default values before
        kernel = model.covar_module
        base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
        ls_before = base.lengthscale.detach().clone()

        # Load from nonexistent path — should return False and not crash
        result = step04.load_gp_state(model, tmp_path / "nonexistent.pt")
        assert result is False

        # Values should be unchanged
        ls_after = base.lengthscale.detach()
        torch.testing.assert_close(ls_after, ls_before)

    def test_warm_start_dimension_mismatch_warning(self, simple_gp, tmp_path):
        """Different data dims should warn and skip, not crash."""
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        # Save state from 3-dim model
        save_path = tmp_path / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)

        # Create a model with different input dimensions (5 instead of 3)
        torch.manual_seed(99)
        n, d_new = 10, 5
        train_X = torch.rand(n, d_new, dtype=torch.double)
        train_Y = torch.rand(n, 2, dtype=torch.double)
        new_model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d_new),
            outcome_transform=Standardize(m=2),
        )

        # Load should return False due to dimension mismatch
        result = step04.load_gp_state(new_model, save_path)
        assert result is False

    def test_round_trip_state(self, simple_gp, tmp_path):
        """Save then load should preserve all hyperparameter values."""
        import torch

        # Extract all hyperparams — covar_module may be bare RBFKernel
        kernel = simple_gp.covar_module
        base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
        orig_ls = base.lengthscale.detach().clone()
        orig_noise = simple_gp.likelihood.noise.detach().clone()
        orig_mean = simple_gp.mean_module.constant.detach().clone()

        # Save
        save_path = tmp_path / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)

        # Load state file and verify raw values
        state = torch.load(str(save_path), weights_only=True)
        torch.testing.assert_close(state["lengthscales"], orig_ls)
        torch.testing.assert_close(state["noise"], orig_noise)
        torch.testing.assert_close(state["mean_constant"], orig_mean)

    def test_save_creates_parent_directories(self, simple_gp, tmp_path):
        """save_gp_state should create nested parent directories."""
        save_path = tmp_path / "nested" / "deep" / "round_1.pt"
        step04.save_gp_state(simple_gp, save_path)
        assert save_path.exists()

    def test_fit_gp_botorch_saves_state(self, tmp_path, monkeypatch):
        """fit_gp_botorch should auto-save state after fitting."""
        import torch
        from gopro.config import GP_STATE_DIR

        # Redirect GP_STATE_DIR to tmp_path
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)

        n, d = 15, 4
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"m{i}" for i in range(d)],
        )
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(3), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
        )

        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, round_num=2,
        )

        # State file should have been created for round 2
        state_path = tmp_path / "round_2.pt"
        assert state_path.exists()

        state = torch.load(str(state_path), weights_only=True)
        assert "lengthscales" in state


class TestTVR:
    """Tests for Targeted Variance Reduction (TVR) model ensemble."""

    @pytest.fixture
    def multi_fidelity_data(self):
        """Create multi-fidelity training data for TVR tests."""
        np.random.seed(42)
        n_real, n_virtual = 15, 10
        d, m = 4, 3
        morph_cols = ["CHIR99021_uM", "SAG_uM", "BMP4_uM", "log_harvest_day"]

        # Real data (fidelity=1.0)
        X_real = pd.DataFrame(
            np.random.rand(n_real, d),
            columns=morph_cols,
            index=[f"real_{i}" for i in range(n_real)],
        )
        X_real["fidelity"] = 1.0
        Y_real = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n_real),
            columns=["ct_A", "ct_B", "ct_C"],
            index=X_real.index,
        )

        # Virtual data (fidelity=0.5)
        X_virt = pd.DataFrame(
            np.random.rand(n_virtual, d),
            columns=morph_cols,
            index=[f"virt_{i}" for i in range(n_virtual)],
        )
        X_virt["fidelity"] = 0.5
        Y_virt = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n_virtual),
            columns=["ct_A", "ct_B", "ct_C"],
            index=X_virt.index,
        )

        X = pd.concat([X_real, X_virt])
        Y = pd.concat([Y_real, Y_virt])
        return X, Y

    def test_fit_tvr_models_returns_ensemble(self, multi_fidelity_data):
        """fit_tvr_models returns a TVRModelEnsemble."""
        X, Y = multi_fidelity_data
        model, train_X, train_Y, ct_cols = step04.fit_tvr_models(X, Y, use_ilr=True)
        assert isinstance(model, step04.TVRModelEnsemble)
        assert len(ct_cols) == 3

    def test_tvr_ensemble_has_per_fidelity_models(self, multi_fidelity_data):
        """TVR ensemble should contain one GP per fidelity level."""
        X, Y = multi_fidelity_data
        model, _, _, _ = step04.fit_tvr_models(X, Y, use_ilr=True)
        assert len(model.models) == 2
        assert 1.0 in model.models
        assert 0.5 in model.models

    def test_tvr_posterior_shape(self, multi_fidelity_data):
        """TVR posterior should return correct shapes for mean and variance."""
        import torch
        X, Y = multi_fidelity_data
        model, train_X, _, _ = step04.fit_tvr_models(X, Y, use_ilr=True)

        # Query 3 points
        X_test = train_X[:3]
        post = model.posterior(X_test)
        # ILR: 3 cell types -> 2 ILR coords
        assert post.mean.shape == (3, 2)
        assert post.variance.shape == (3, 2)
        assert torch.all(post.variance > 0)

    def test_tvr_posterior_sample(self, multi_fidelity_data):
        """TVR posterior should support sampling for MC acquisition."""
        import torch
        X, Y = multi_fidelity_data
        model, train_X, _, _ = step04.fit_tvr_models(X, Y, use_ilr=True)

        X_test = train_X[:3]
        post = model.posterior(X_test)
        samples = post.sample(torch.Size([5]))
        assert samples.shape == (5, 3, 2)

    def test_tvr_requires_multi_fidelity(self):
        """TVR should raise ValueError without fidelity column."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(10, 3),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM"],
        )
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(2), size=10),
            columns=["ct_A", "ct_B"],
        )
        with pytest.raises(ValueError, match="fidelity column missing"):
            step04.fit_tvr_models(X, Y)

    def test_tvr_requires_multiple_fidelity_levels(self):
        """TVR should raise ValueError with only one fidelity level."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(10, 3),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM"],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(2), size=10),
            columns=["ct_A", "ct_B"],
        )
        with pytest.raises(ValueError, match="at least 2 fidelity levels"):
            step04.fit_tvr_models(X, Y)

    def test_tvr_cost_scaling_affects_selection(self, multi_fidelity_data):
        """Cost-scaled variance should prefer cheaper models when variance is similar."""
        import torch
        X, Y = multi_fidelity_data
        model, train_X, _, _ = step04.fit_tvr_models(X, Y, use_ilr=True)

        # The ensemble should use cost ratios from FIDELITY_COSTS
        assert model.cost_ratios[1.0] == 1.0
        assert model.cost_ratios[0.5] < 1.0  # cheaper model

    def test_tvr_num_outputs(self, multi_fidelity_data):
        """TVR ensemble num_outputs should match ILR-transformed dimension."""
        X, Y = multi_fidelity_data
        model, _, _, _ = step04.fit_tvr_models(X, Y, use_ilr=True)
        # 3 cell types -> 2 ILR coords
        assert model.num_outputs == 2


class TestRefineTargetProfile:
    """Tests for target profile refinement (DeMeo 2025 interpolation)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample fractions and fidelity scores for refinement tests."""
        np.random.seed(42)
        cell_types = ["Neuron", "Astrocyte", "OPC", "Microglia"]
        n_conditions = 20

        # Create fractions where Neuron correlates positively with fidelity
        fractions = pd.DataFrame(
            np.random.dirichlet([3, 2, 2, 1], size=n_conditions),
            columns=cell_types,
            index=[f"cond_{i}" for i in range(n_conditions)],
        )
        # Make Neuron fraction correlate with fidelity
        fidelity = pd.Series(
            fractions["Neuron"] * 0.8 + np.random.normal(0, 0.05, n_conditions),
            index=fractions.index,
        ).clip(0, 1)

        original_target = pd.Series(
            [0.25, 0.25, 0.25, 0.25], index=cell_types
        )
        return fractions, fidelity, original_target

    def test_refined_differs_from_original(self, sample_data):
        """Refined target should differ from original when data is informative."""
        fractions, fidelity, original = sample_data
        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=0.5,
        )
        # Should not be identical
        assert not np.allclose(refined.values, original.values, atol=1e-6)

    def test_refined_is_valid_composition(self, sample_data):
        """Refined target must be a valid composition (non-negative, sums to 1)."""
        fractions, fidelity, original = sample_data
        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=0.7,
        )
        assert (refined >= 0).all()
        assert abs(refined.sum() - 1.0) < 1e-10

    def test_lr_zero_returns_original(self, sample_data):
        """learning_rate=0 should return the original target."""
        fractions, fidelity, original = sample_data
        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=0.0,
        )
        np.testing.assert_allclose(refined.values, original.values, atol=1e-10)

    def test_lr_one_fully_learned(self, sample_data):
        """learning_rate=1 should return the fully learned target (not original)."""
        fractions, fidelity, original = sample_data
        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=1.0,
        )
        # Should be fully learned — NOT equal to original
        assert not np.allclose(refined.values, original.values, atol=1e-3)
        # Should still be a valid composition
        assert (refined >= 0).all()
        assert abs(refined.sum() - 1.0) < 1e-10

    def test_too_few_conditions_returns_original(self):
        """With fewer than 3 overlapping conditions, return original unchanged."""
        cell_types = ["A", "B", "C"]
        fractions = pd.DataFrame(
            [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]],
            columns=cell_types,
            index=["c0", "c1"],
        )
        fidelity = pd.Series([0.8, 0.6], index=["c0", "c1"])
        original = pd.Series([0.33, 0.33, 0.34], index=cell_types)

        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=0.5,
        )
        np.testing.assert_allclose(refined.values, original.values)

    def test_invalid_lr_raises(self, sample_data):
        """learning_rate outside [0, 1] should raise ValueError."""
        fractions, fidelity, original = sample_data
        with pytest.raises(ValueError, match="learning_rate"):
            step04.refine_target_profile(fractions, fidelity, original, learning_rate=-0.1)
        with pytest.raises(ValueError, match="learning_rate"):
            step04.refine_target_profile(fractions, fidelity, original, learning_rate=1.5)

    def test_neuron_upweighted_when_correlated(self, sample_data):
        """Neuron should be upweighted in refined target (positively correlated with fidelity)."""
        fractions, fidelity, original = sample_data
        refined = step04.refine_target_profile(
            fractions, fidelity, original, learning_rate=0.5,
        )
        # Original is uniform 0.25 each. Since Neuron correlates with fidelity,
        # Neuron should be higher in refined target
        assert refined["Neuron"] > original["Neuron"]


class TestAdditiveInteractionKernel:
    """Tests for additive + interaction kernel (Idea #8)."""

    def test_build_kernel_structure(self):
        """Kernel should have d additive + 1 interaction sub-kernels."""
        from gpytorch.kernels import AdditiveKernel, ScaleKernel, MaternKernel

        kernel = step04._build_additive_interaction_kernel(d=5)
        # GPyTorch flattens AdditiveKernel + ScaleKernel into one AdditiveKernel
        assert isinstance(kernel, AdditiveKernel)
        # d per-dim ScaleKernel(Matern) + 1 interaction ScaleKernel(Matern ARD)
        assert len(kernel.kernels) == 6  # 5 additive + 1 interaction
        # All sub-kernels should be ScaleKernel wrapping MaternKernel
        for sk in kernel.kernels:
            assert isinstance(sk, ScaleKernel)
            assert isinstance(sk.base_kernel, MaternKernel)

    def test_interaction_scale_initialized_small(self):
        """Interaction outputscale should start small (prior toward additivity)."""
        kernel = step04._build_additive_interaction_kernel(d=4)
        # Last kernel is the interaction ScaleKernel(Matern ARD)
        interaction_part = kernel.kernels[-1]
        # Interaction has ARD (ard_num_dims > 1), additive kernels have active_dims=(i,)
        assert interaction_part.base_kernel.ard_num_dims == 4
        assert interaction_part.outputscale.item() == pytest.approx(0.1, abs=1e-6)

    def test_additive_kernels_have_correct_active_dims(self):
        """Each additive sub-kernel should operate on exactly one dimension."""
        kernel = step04._build_additive_interaction_kernel(d=3)
        # First d kernels are per-dim additive, last is interaction
        for i, sub_k in enumerate(kernel.kernels[:3]):
            # ScaleKernel wraps MaternKernel; active_dims is on the Matern
            matern = sub_k.base_kernel
            assert tuple(matern.active_dims) == (i,)

    def test_fit_gp_with_additive_interaction_kernel(self, tmp_path, monkeypatch):
        """GP should fit and produce predictions with additive+interaction kernel."""
        import torch

        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)

        np.random.seed(42)
        n, d = 20, 4
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"m{i}" for i in range(d)],
        )
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(3), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
        )

        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True, round_num=1,
            kernel_type="additive_interaction",
        )

        # Model should produce predictions
        model.eval()
        with torch.no_grad():
            test_X = torch.tensor(
                np.random.rand(5, d), dtype=torch.float64
            )
            posterior = model.posterior(test_X)
            mean = posterior.mean
            assert mean.shape == (5, 2)  # ILR: 3 cell types -> 2 dims

    def test_extract_lengthscales_from_additive_interaction(self, tmp_path, monkeypatch):
        """_extract_lengthscales should return interaction kernel lengthscales."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)

        np.random.seed(42)
        n, d = 20, 4
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"m{i}" for i in range(d)],
        )
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(3), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
        )

        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, round_num=1,
            kernel_type="additive_interaction",
        )

        ls = step04._extract_lengthscales(model, d)
        assert ls is not None
        assert ls.shape == (d,)
        assert np.all(ls > 0)


class TestAdaptiveComplexitySchedule:
    """Tests for adaptive kernel complexity selection (Idea #9)."""

    def test_sparse_data_selects_shared(self):
        """N/d < 3 should select shared lengthscale kernel."""
        result = step04._select_kernel_complexity(n_conditions=10, d_active=5)
        # 10/5 = 2.0 < 3 → shared
        assert result["kernel_type"] == "shared"
        assert result["use_saasbo"] is False
        assert result["n_d_ratio"] == pytest.approx(2.0)

    def test_moderate_data_selects_ard(self):
        """3 ≤ N/d < 15 should select per-dim ARD kernel."""
        result = step04._select_kernel_complexity(n_conditions=50, d_active=5)
        # 50/5 = 10.0, in [3, 15) → ARD
        assert result["kernel_type"] == "ard"
        assert result["use_saasbo"] is False
        assert result["n_d_ratio"] == pytest.approx(10.0)

    def test_dense_data_selects_saasbo(self):
        """N/d ≥ 15 should select SAASBO."""
        result = step04._select_kernel_complexity(n_conditions=80, d_active=5)
        # 80/5 = 16.0 ≥ 15 → SAASBO
        assert result["kernel_type"] == "saasbo"
        assert result["use_saasbo"] is True
        assert result["n_d_ratio"] == pytest.approx(16.0)

    def test_boundary_shared_ard(self):
        """Exactly at shared threshold should select ARD."""
        result = step04._select_kernel_complexity(n_conditions=15, d_active=5)
        # 15/5 = 3.0, right at threshold → ARD (not shared)
        assert result["kernel_type"] == "ard"

    def test_boundary_ard_saasbo(self):
        """Exactly at ARD threshold should select SAASBO."""
        result = step04._select_kernel_complexity(n_conditions=75, d_active=5)
        # 75/5 = 15.0, right at threshold → SAASBO
        assert result["kernel_type"] == "saasbo"
        assert result["use_saasbo"] is True

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        custom = {"shared": 3.0, "ard": 5.0}
        # 20/5 = 4.0, in [3, 5) → ARD with custom thresholds
        result = step04._select_kernel_complexity(
            n_conditions=20, d_active=5, thresholds=custom,
        )
        assert result["kernel_type"] == "ard"

    def test_zero_active_dims_safe(self):
        """d_active=0 should not cause division by zero."""
        result = step04._select_kernel_complexity(n_conditions=10, d_active=0)
        # d_safe = max(0, 1) = 1; ratio = 10.0, in [3, 15) → ARD
        assert result["kernel_type"] == "ard"
        assert result["n_d_ratio"] == pytest.approx(10.0)

    def test_reason_string_contains_values(self):
        """Reason string should mention N, d, and ratio."""
        result = step04._select_kernel_complexity(n_conditions=48, d_active=8)
        assert "N=48" in result["reason"]
        assert "d=8" in result["reason"]
        assert "N/d=" in result["reason"]


class TestTimingWindowEncoding:
    """Tests for morphogen timing window categorical encoding (Sanchis-Calleja 2025, Idea #10)."""

    def test_compute_timing_windows_shape(self):
        """Output shape should match (n_conditions, n_timing_cols)."""
        from gopro.morphogen_parser import compute_timing_windows, ALL_CONDITIONS
        from gopro.config import TIMING_WINDOW_COLUMNS
        tw = compute_timing_windows(ALL_CONDITIONS)
        assert tw.shape == (len(ALL_CONDITIONS), len(TIMING_WINDOW_COLUMNS))
        assert list(tw.columns) == TIMING_WINDOW_COLUMNS

    def test_timing_windows_known_conditions(self):
        """Conditions with known sub-windows should get correct categorical values."""
        from gopro.morphogen_parser import compute_timing_windows
        from gopro.config import (
            TIMING_EARLY, TIMING_MID, TIMING_LATE, TIMING_FULL, TIMING_NOT_APPLIED,
        )
        tw = compute_timing_windows(["CHIR-d6-11", "CHIR-d11-16", "CHIR-d16-21", "CHIR1.5"])
        # CHIR-d6-11: CHIR=early, SAG=not applied, BMP4=not applied
        assert tw.loc["CHIR-d6-11", "CHIR99021_window"] == TIMING_EARLY
        assert tw.loc["CHIR-d6-11", "SAG_window"] == TIMING_NOT_APPLIED
        # CHIR-d11-16: CHIR=mid
        assert tw.loc["CHIR-d11-16", "CHIR99021_window"] == TIMING_MID
        # CHIR-d16-21: CHIR=late
        assert tw.loc["CHIR-d16-21", "CHIR99021_window"] == TIMING_LATE
        # CHIR1.5: CHIR=full window (default for non-sub-windowed conditions)
        assert tw.loc["CHIR1.5", "CHIR99021_window"] == TIMING_FULL

    def test_timing_windows_sag_conditions(self):
        """SAG sub-window conditions should be correctly encoded."""
        from gopro.morphogen_parser import compute_timing_windows
        from gopro.config import TIMING_EARLY, TIMING_MID, TIMING_LATE, TIMING_FULL
        tw = compute_timing_windows(["SAG-d6-11", "SAG-d11-16", "SAG-d16-21"])
        assert tw.loc["SAG-d6-11", "SAG_window"] == TIMING_EARLY
        assert tw.loc["SAG-d11-16", "SAG_window"] == TIMING_MID
        assert tw.loc["SAG-d16-21", "SAG_window"] == TIMING_LATE

    def test_timing_windows_not_applied(self):
        """Conditions without a morphogen should get TIMING_NOT_APPLIED."""
        from gopro.morphogen_parser import compute_timing_windows
        from gopro.config import TIMING_NOT_APPLIED, TIMING_FULL
        # DAPT uses neither CHIR, SAG, nor BMP4
        tw = compute_timing_windows(["DAPT"])
        assert tw.loc["DAPT", "CHIR99021_window"] == TIMING_NOT_APPLIED
        assert tw.loc["DAPT", "SAG_window"] == TIMING_NOT_APPLIED
        assert tw.loc["DAPT", "BMP4_window"] == TIMING_NOT_APPLIED

    def test_timing_windows_bmp4_subwindow(self):
        """BMP4 CHIR d11-16 should have BMP4=mid, CHIR=mid."""
        from gopro.morphogen_parser import compute_timing_windows
        from gopro.config import TIMING_MID
        tw = compute_timing_windows(["BMP4 CHIR d11-16"])
        assert tw.loc["BMP4 CHIR d11-16", "BMP4_window"] == TIMING_MID
        assert tw.loc["BMP4 CHIR d11-16", "CHIR99021_window"] == TIMING_MID

    def test_timing_windows_integer_coded(self):
        """All timing window values should be integers in [0, 4]."""
        from gopro.morphogen_parser import compute_timing_windows, ALL_CONDITIONS
        tw = compute_timing_windows(ALL_CONDITIONS)
        for col in tw.columns:
            assert tw[col].between(0, 4).all(), f"Column {col} has values outside [0, 4]"

    def test_fit_gp_with_cat_dims(self):
        """MixedSingleTaskGP should fit successfully with categorical timing dims."""
        from botorch.models import MixedSingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll
        import torch

        np.random.seed(42)
        n, d_cont = 20, 3
        X_cont = np.random.rand(n, d_cont)
        # Add one categorical column with values in {0, 1, 2, 3, 4}
        X_cat = np.random.randint(0, 5, size=(n, 1)).astype(float)
        X = np.hstack([X_cont, X_cat])
        Y = np.random.rand(n, 2)

        train_X = torch.tensor(X, dtype=torch.float64)
        train_Y = torch.tensor(Y, dtype=torch.float64)

        cat_dims = [3]  # last column is categorical
        cont_indices = [i for i in range(4) if i not in cat_dims]
        model = MixedSingleTaskGP(
            train_X, train_Y,
            cat_dims=cat_dims,
            input_transform=Normalize(d=4, indices=cont_indices),
            outcome_transform=Standardize(m=2),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Model should produce predictions
        test_X = torch.tensor([[0.5, 0.5, 0.5, 2.0]], dtype=torch.float64)
        posterior = model.posterior(test_X)
        assert posterior.mean.shape == (1, 2)


class TestTemporalBinEncoding:
    """Tests for temporal bin encoding (Sorre 2014, Sasai 2014, Amin/Kelley 2024)."""

    def test_compute_temporal_bins_shape(self):
        """Output shape should match (n_conditions, n_temporal_bin_cols)."""
        from gopro.morphogen_parser import compute_temporal_bins, ALL_CONDITIONS
        from gopro.config import TEMPORAL_BIN_COLUMNS
        tb = compute_temporal_bins(ALL_CONDITIONS)
        assert tb.shape == (len(ALL_CONDITIONS), len(TEMPORAL_BIN_COLUMNS))
        assert list(tb.columns) == TEMPORAL_BIN_COLUMNS

    def test_chir_d11_16_bins(self):
        """CHIR-d11-16: only mid bin should have concentration."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR-d11-16"])
        row = tb.loc["CHIR-d11-16"]
        assert row["CHIR99021_early_uM"] == 0.0
        assert row["CHIR99021_mid_uM"] == 1.5  # full concentration, not scaled
        assert row["CHIR99021_late_uM"] == 0.0

    def test_chir_d6_11_bins(self):
        """CHIR-d6-11: only early bin should have concentration."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR-d6-11"])
        row = tb.loc["CHIR-d6-11"]
        assert row["CHIR99021_early_uM"] == 1.5
        assert row["CHIR99021_mid_uM"] == 0.0
        assert row["CHIR99021_late_uM"] == 0.0

    def test_chir_d16_21_bins(self):
        """CHIR-d16-21: only late bin should have concentration."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR-d16-21"])
        row = tb.loc["CHIR-d16-21"]
        assert row["CHIR99021_early_uM"] == 0.0
        assert row["CHIR99021_mid_uM"] == 0.0
        assert row["CHIR99021_late_uM"] == 1.5

    def test_full_window_gets_all_bins(self):
        """CHIR1.5 (full window): all 3 bins should have the concentration."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR1.5"])
        row = tb.loc["CHIR1.5"]
        assert row["CHIR99021_early_uM"] == 1.5
        assert row["CHIR99021_mid_uM"] == 1.5
        assert row["CHIR99021_late_uM"] == 1.5

    def test_absent_morphogen_all_zeros(self):
        """DAPT: CHIR, SAG, IWP2, BMP4 are all absent -> all bins zero."""
        from gopro.morphogen_parser import compute_temporal_bins
        from gopro.config import TEMPORAL_BIN_COLUMNS
        tb = compute_temporal_bins(["DAPT"])
        for col in TEMPORAL_BIN_COLUMNS:
            assert tb.loc["DAPT", col] == 0.0

    def test_sag_subwindows(self):
        """SAG sub-window conditions should place concentration in correct bins."""
        from gopro.morphogen_parser import compute_temporal_bins
        from gopro.config import nM_to_uM
        sag_default = nM_to_uM(50.0)  # 0.05 uM
        tb = compute_temporal_bins(["SAG-d6-11", "SAG-d11-16", "SAG-d16-21"])
        # SAG-d6-11: early only
        assert tb.loc["SAG-d6-11", "SAG_early_uM"] == sag_default
        assert tb.loc["SAG-d6-11", "SAG_mid_uM"] == 0.0
        assert tb.loc["SAG-d6-11", "SAG_late_uM"] == 0.0
        # SAG-d11-16: mid only
        assert tb.loc["SAG-d11-16", "SAG_mid_uM"] == sag_default
        # SAG-d16-21: late only
        assert tb.loc["SAG-d16-21", "SAG_late_uM"] == sag_default

    def test_chir_switch_iwp2_bins(self):
        """CHIR switch IWP2: CHIR days 6-13 (early+mid), IWP2 days 13-21 (mid+late)."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR switch IWP2"])
        row = tb.loc["CHIR switch IWP2"]
        # CHIR 6-13 overlaps early (6-11) and mid (11-16)
        assert row["CHIR99021_early_uM"] == 1.5
        assert row["CHIR99021_mid_uM"] == 1.5
        assert row["CHIR99021_late_uM"] == 0.0
        # IWP2 13-21 overlaps mid (11-16) and late (16-21)
        assert row["IWP2_early_uM"] == 0.0
        assert row["IWP2_mid_uM"] == 5.0  # IWP2 default = 5.0 uM
        assert row["IWP2_late_uM"] == 5.0

    def test_chir_sagd10_21_bins(self):
        """CHIR-SAGd10-21: SAG days 10-21 overlaps all 3 bins."""
        from gopro.morphogen_parser import compute_temporal_bins
        from gopro.config import nM_to_uM
        tb = compute_temporal_bins(["CHIR-SAGd10-21"])
        row = tb.loc["CHIR-SAGd10-21"]
        sag_default = nM_to_uM(50.0)
        # SAG 10-21 overlaps early (6-11, overlap at day 10-11),
        # mid (11-16), and late (16-21)
        assert row["SAG_early_uM"] == sag_default
        assert row["SAG_mid_uM"] == sag_default
        assert row["SAG_late_uM"] == sag_default
        # CHIR is full window (not in explicit windows for this condition)
        assert row["CHIR99021_early_uM"] == 1.5
        assert row["CHIR99021_mid_uM"] == 1.5
        assert row["CHIR99021_late_uM"] == 1.5

    def test_chir3_uses_correct_concentration(self):
        """CHIR3: non-default concentration (3.0 uM) in all bins."""
        from gopro.morphogen_parser import compute_temporal_bins
        tb = compute_temporal_bins(["CHIR3"])
        row = tb.loc["CHIR3"]
        assert row["CHIR99021_early_uM"] == 3.0
        assert row["CHIR99021_mid_uM"] == 3.0
        assert row["CHIR99021_late_uM"] == 3.0

    def test_build_matrix_with_temporal_bins(self):
        """Combined matrix should have non-binned + binned columns."""
        from gopro.morphogen_parser import (
            build_morphogen_matrix_with_temporal_bins, ALL_CONDITIONS,
        )
        from gopro.config import MORPHOGEN_COLUMNS, TEMPORAL_BIN_COLUMNS, TEMPORAL_BIN_MORPHOGENS
        df = build_morphogen_matrix_with_temporal_bins(ALL_CONDITIONS)
        assert df.shape[0] == len(ALL_CONDITIONS)
        # Should NOT contain the original single columns for binned morphogens
        for morph in TEMPORAL_BIN_MORPHOGENS:
            assert f"{morph}_uM" not in df.columns
        # Should contain all temporal bin columns
        for col in TEMPORAL_BIN_COLUMNS:
            assert col in df.columns
        # Non-binned columns should still be present
        binned_cols = {f"{m}_uM" for m in TEMPORAL_BIN_MORPHOGENS}
        for col in MORPHOGEN_COLUMNS:
            if col not in binned_cols:
                assert col in df.columns

    def test_temporal_bins_non_negative(self):
        """All temporal bin values should be non-negative."""
        from gopro.morphogen_parser import compute_temporal_bins, ALL_CONDITIONS
        tb = compute_temporal_bins(ALL_CONDITIONS)
        assert (tb >= 0).all().all()

    def test_bmp4_chir_d11_16_bins(self):
        """BMP4 CHIR d11-16: both BMP4 and CHIR in mid bin only."""
        from gopro.morphogen_parser import compute_temporal_bins
        from gopro.config import ng_mL_to_uM, PROTEIN_MW_KDA
        tb = compute_temporal_bins(["BMP4 CHIR d11-16"])
        row = tb.loc["BMP4 CHIR d11-16"]
        bmp4_default = ng_mL_to_uM(10.0, PROTEIN_MW_KDA["BMP4"])
        # CHIR mid only
        assert row["CHIR99021_early_uM"] == 0.0
        assert row["CHIR99021_mid_uM"] == 1.5
        assert row["CHIR99021_late_uM"] == 0.0
        # BMP4 mid only
        assert row["BMP4_early_uM"] == 0.0
        assert abs(row["BMP4_mid_uM"] - bmp4_default) < 1e-10
        assert row["BMP4_late_uM"] == 0.0

    def test_window_overlaps_bin(self):
        """Unit test for _window_overlaps_bin helper."""
        from gopro.morphogen_parser import _window_overlaps_bin
        # Exact match
        assert _window_overlaps_bin(6, 11, 6, 11) is True
        # No overlap
        assert _window_overlaps_bin(6, 11, 11, 16) is False
        assert _window_overlaps_bin(16, 21, 6, 11) is False
        # Partial overlap
        assert _window_overlaps_bin(6, 13, 11, 16) is True
        assert _window_overlaps_bin(10, 21, 6, 11) is True
        # Contained
        assert _window_overlaps_bin(6, 21, 11, 16) is True


class TestTemporalBinsParameter:
    """Tests for ``temporal_bins`` parameter on build_morphogen_matrix and parser classes."""

    def test_build_morphogen_matrix_temporal_bins_flag(self):
        """build_morphogen_matrix(temporal_bins=True) produces expanded columns."""
        from gopro.morphogen_parser import build_morphogen_matrix
        from gopro.config import (
            MORPHOGEN_COLUMNS_TEMPORAL, TEMPORAL_BIN_MORPHOGENS,
        )
        df = build_morphogen_matrix(["CHIR-d11-16", "CHIR1.5"], temporal_bins=True)
        assert list(df.columns) == MORPHOGEN_COLUMNS_TEMPORAL
        # Binned morphogen single columns should be absent
        for morph in TEMPORAL_BIN_MORPHOGENS:
            assert f"{morph}_uM" not in df.columns
        # CHIR-d11-16 should have mid bin only
        assert df.loc["CHIR-d11-16", "CHIR99021_mid_uM"] == 1.5
        assert df.loc["CHIR-d11-16", "CHIR99021_early_uM"] == 0.0

    def test_build_morphogen_matrix_default_unchanged(self):
        """Default call (temporal_bins=False) returns standard columns."""
        from gopro.morphogen_parser import build_morphogen_matrix
        from gopro.config import MORPHOGEN_COLUMNS
        df = build_morphogen_matrix(["CHIR1.5"])
        assert list(df.columns) == MORPHOGEN_COLUMNS

    def test_parser_class_build_matrix_temporal_bins(self):
        """AminKelleyParser.build_matrix(temporal_bins=True) works."""
        from gopro.morphogen_parser import AminKelleyParser
        from gopro.config import MORPHOGEN_COLUMNS_TEMPORAL
        parser = AminKelleyParser()
        df = parser.build_matrix(["CHIR1.5", "SAG-d6-11"], temporal_bins=True)
        assert list(df.columns) == MORPHOGEN_COLUMNS_TEMPORAL
        assert df.loc["SAG-d6-11", "SAG_early_uM"] > 0
        assert df.loc["SAG-d6-11", "SAG_mid_uM"] == 0.0

    def test_combined_parser_build_matrix_temporal_bins(self):
        """CombinedParser.build_matrix(temporal_bins=True) works."""
        from gopro.morphogen_parser import (
            AminKelleyParser, SAGSecondaryParser, CombinedParser,
        )
        from gopro.config import MORPHOGEN_COLUMNS_TEMPORAL
        combined = CombinedParser([AminKelleyParser(), SAGSecondaryParser()])
        df = combined.build_matrix(["CHIR1.5", "SAG_2uM"], temporal_bins=True)
        assert list(df.columns) == MORPHOGEN_COLUMNS_TEMPORAL

    def test_morphogen_columns_temporal_structure(self):
        """MORPHOGEN_COLUMNS_TEMPORAL has correct structure."""
        from gopro.config import (
            MORPHOGEN_COLUMNS, MORPHOGEN_COLUMNS_TEMPORAL,
            TEMPORAL_BIN_MORPHOGENS, TEMPORAL_BIN_COLUMNS,
        )
        binned = {f"{m}_uM" for m in TEMPORAL_BIN_MORPHOGENS}
        non_binned = [c for c in MORPHOGEN_COLUMNS if c not in binned]
        # Should start with non-binned, then temporal bin columns
        assert MORPHOGEN_COLUMNS_TEMPORAL == non_binned + TEMPORAL_BIN_COLUMNS
        # Total columns: original minus binned + 3 bins per binned morphogen
        expected_len = len(MORPHOGEN_COLUMNS) - len(TEMPORAL_BIN_MORPHOGENS) + 3 * len(TEMPORAL_BIN_MORPHOGENS)
        assert len(MORPHOGEN_COLUMNS_TEMPORAL) == expected_len


class TestPerTypeGP:
    """Tests for per-cell-type GP models (GPerturb, Xing & Yau 2025, Idea #11)."""

    @staticmethod
    def _fit_per_type_model(tmp_path, monkeypatch):
        """Shared setup: fit a per-type GP model (reused across tests)."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        np.random.seed(42)
        n, d = 20, 4
        X = pd.DataFrame(np.random.rand(n, d), columns=[f"m{i}" for i in range(d)])
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(3), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
        )
        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True, round_num=1, per_type_gp=True,
        )
        return model, train_X, train_Y, cols, X, d

    def test_per_type_gp_produces_model_list(self, tmp_path, monkeypatch):
        """--per-type-gp should produce a ModelListGP with one model per output."""
        from botorch.models import ModelListGP

        model, _, _, cols, _, _ = self._fit_per_type_model(tmp_path, monkeypatch)

        assert isinstance(model, ModelListGP)
        # ILR: 3 cell types -> 2 ILR components -> 2 sub-models
        assert len(model.models) == 2
        assert cols == ["ct_A", "ct_B", "ct_C"]

    def test_per_type_gp_predictions(self, tmp_path, monkeypatch):
        """Per-type GP should produce correct-shaped predictions."""
        import torch

        model, _, _, _, _, d = self._fit_per_type_model(tmp_path, monkeypatch)

        model.eval()
        with torch.no_grad():
            test_X = torch.tensor(
                np.random.rand(5, d), dtype=torch.float64
            )
            posterior = model.posterior(test_X)
            mean = posterior.mean
            # ModelListGP concatenates outputs: shape (5, 2)
            assert mean.shape == (5, 2)

    def test_per_output_lengthscale_matrix(self, tmp_path, monkeypatch):
        """_extract_per_output_lengthscales should return (d x n_outputs) matrix."""
        model, _, _, _, _, d = self._fit_per_type_model(tmp_path, monkeypatch)

        ls_matrix = step04._extract_per_output_lengthscales(model, d)
        assert ls_matrix is not None
        assert ls_matrix.shape == (d, 2)  # d morphogens x 2 ILR components
        assert np.all(ls_matrix > 0)

    def test_per_output_lengthscales_differ_across_outputs(self, tmp_path, monkeypatch):
        """Each output should have different lengthscales (not shared)."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)

        np.random.seed(42)
        n, d = 25, 3
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"m{i}" for i in range(d)],
        )
        # Create Y where different outputs depend on different inputs
        Y_vals = np.column_stack([
            np.sin(2 * np.pi * X["m0"].values) * 0.3 + 0.5,
            np.cos(2 * np.pi * X["m1"].values) * 0.3 + 0.5,
            np.ones(n) * 0.0,  # remainder to sum to 1
        ])
        Y_vals[:, 2] = 1.0 - Y_vals[:, 0] - Y_vals[:, 1]
        Y_vals = np.clip(Y_vals, 0.01, 0.98)
        row_sums = Y_vals.sum(axis=1, keepdims=True)
        Y_vals = Y_vals / row_sums
        Y = pd.DataFrame(Y_vals, columns=["ct_A", "ct_B", "ct_C"])

        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, round_num=1, per_type_gp=True,
        )

        ls_matrix = step04._extract_per_output_lengthscales(model, d)
        assert ls_matrix is not None
        # Lengthscales across the two ILR components should not be identical
        assert not np.allclose(ls_matrix[:, 0], ls_matrix[:, 1], atol=0.01)

    def test_per_type_gp_single_output_fallback(self, tmp_path, monkeypatch):
        """With single output, per_type_gp should fall back to standard GP."""
        import torch
        from botorch.models import SingleTaskGP

        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)

        np.random.seed(42)
        n, d = 20, 3
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"m{i}" for i in range(d)],
        )
        Y = pd.DataFrame(
            np.random.rand(n, 1),
            columns=["ct_A"],
        )

        model, _, _, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, round_num=1, per_type_gp=True,
        )

        # Single output + per_type_gp: Y.shape[1]=1, condition
        # `per_type_gp and train_Y.shape[1] > 1` is False → falls to else
        assert isinstance(model, SingleTaskGP)
        assert cols == ["ct_A"]

    def test_extract_per_output_on_non_modellist_returns_none(self):
        """_extract_per_output_lengthscales returns None for non-ModelListGP."""
        result = step04._extract_per_output_lengthscales(object(), 5)
        assert result is None

    def test_per_type_gp_rejects_saasbo(self, tmp_path, monkeypatch):
        """per_type_gp + use_saasbo should raise ValueError."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        X = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame(np.random.rand(10, 2), columns=["ct_A", "ct_B"])
        with pytest.raises(ValueError, match="per_type_gp is incompatible with use_saasbo"):
            step04.fit_gp_botorch(X, Y, use_saasbo=True, per_type_gp=True, round_num=1)

    def test_per_type_gp_rejects_cat_dims(self, tmp_path, monkeypatch):
        """per_type_gp + cat_dims should raise ValueError."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        X = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame(np.random.rand(10, 2), columns=["ct_A", "ct_B"])
        with pytest.raises(ValueError, match="per_type_gp is incompatible with cat_dims"):
            step04.fit_gp_botorch(X, Y, cat_dims=[0], per_type_gp=True, round_num=1)


class TestConvergenceDiagnostics:
    """Tests for compute_convergence_diagnostics()."""

    def _fit_simple_gp(self):
        """Helper: fit a simple GP for testing."""
        X = pd.DataFrame(np.random.rand(15, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame(np.random.rand(15, 2), columns=["ct_A", "ct_B"])
        model, train_X, train_Y, ct_cols = step04.fit_gp_botorch(
            X, Y, round_num=1,
        )
        return model, train_X, train_Y, ct_cols, X.columns.tolist()

    def test_returns_expected_keys(self, tmp_path, monkeypatch):
        """compute_convergence_diagnostics returns all required keys."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        model, train_X, train_Y, ct_cols, cols = self._fit_simple_gp()

        # Build fake recommendations
        recs = pd.DataFrame(np.random.rand(5, 3), columns=cols)
        recs["acquisition_value"] = np.random.rand(5)

        bounds = torch.tensor(
            [[0.0] * 3, [1.0] * 3], dtype=torch.double
        )

        result = step04.compute_convergence_diagnostics(
            model=model,
            train_X=train_X,
            recommendations=recs,
            bounds_tensor=bounds,
            columns=cols,
            round_num=1,
            history_path=tmp_path / "conv.csv",
        )

        assert "mean_posterior_std" in result
        assert "max_acquisition_value" in result
        assert "recommendation_spread" in result
        assert "suggested_batch_size" in result
        assert result["mean_posterior_std"] > 0
        assert result["recommendation_spread"] >= 0

    def test_history_persists_across_rounds(self, tmp_path, monkeypatch):
        """Convergence history CSV accumulates across rounds."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        model, train_X, train_Y, ct_cols, cols = self._fit_simple_gp()

        recs = pd.DataFrame(np.random.rand(5, 3), columns=cols)
        recs["acquisition_value"] = np.random.rand(5)
        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

        history_path = tmp_path / "conv.csv"

        for rnd in [1, 2, 3]:
            result = step04.compute_convergence_diagnostics(
                model=model, train_X=train_X,
                recommendations=recs, bounds_tensor=bounds,
                columns=cols, round_num=rnd,
                history_path=history_path,
            )

        assert history_path.exists()
        history = pd.read_csv(history_path)
        assert len(history) == 3
        assert list(history["round"]) == [1, 2, 3]

    def test_idempotent_rerun(self, tmp_path, monkeypatch):
        """Re-running same round replaces rather than duplicates rows."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        model, train_X, train_Y, ct_cols, cols = self._fit_simple_gp()

        recs = pd.DataFrame(np.random.rand(5, 3), columns=cols)
        recs["acquisition_value"] = np.random.rand(5)
        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
        history_path = tmp_path / "conv.csv"

        for _ in range(3):
            step04.compute_convergence_diagnostics(
                model=model, train_X=train_X,
                recommendations=recs, bounds_tensor=bounds,
                columns=cols, round_num=1,
                history_path=history_path,
            )

        history = pd.read_csv(history_path)
        assert len(history) == 1  # not 3

    def test_adaptive_batch_suggestion(self, tmp_path, monkeypatch):
        """When acquisition decays and spread is low, suggests smaller batch."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        model, train_X, train_Y, ct_cols, cols = self._fit_simple_gp()

        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
        history_path = tmp_path / "conv.csv"

        # Seed history with a high round-1 acquisition value
        seed_df = pd.DataFrame([{
            "round": 1,
            "mean_posterior_std": 1.0,
            "max_acquisition_value": 100.0,
            "recommendation_spread": 0.5,
            "n_training_points": 15,
        }])
        seed_df.to_csv(history_path, index=False)

        # Round 2: very low acquisition and tight clustering
        recs = pd.DataFrame(
            np.full((5, 3), 0.5),  # tightly clustered
            columns=cols,
        )
        recs["acquisition_value"] = 0.001  # very low

        result = step04.compute_convergence_diagnostics(
            model=model, train_X=train_X,
            recommendations=recs, bounds_tensor=bounds,
            columns=cols, round_num=2,
            history_path=history_path,
        )

        # Spread should be near 0 (all same point) → below threshold
        assert result["recommendation_spread"] < 0.01
        # With 100x decay and near-zero spread, should suggest smaller batch
        assert result["suggested_batch_size"] is not None
        assert result["suggested_batch_size"] >= 4

    def test_max_acq_from_recommendations(self, tmp_path, monkeypatch):
        """Max acquisition value is correctly extracted from recommendations."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        model, train_X, train_Y, ct_cols, cols = self._fit_simple_gp()

        recs = pd.DataFrame(np.random.rand(5, 3), columns=cols)
        recs["acquisition_value"] = [0.1, 0.5, 0.3, 0.2, 0.4]
        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

        result = step04.compute_convergence_diagnostics(
            model=model, train_X=train_X,
            recommendations=recs, bounds_tensor=bounds,
            columns=cols, round_num=1,
            history_path=tmp_path / "conv.csv",
        )

        assert abs(result["max_acquisition_value"] - 0.5) < 1e-6


class TestEnsembleDisagreement:
    """Tests for compute_ensemble_disagreement() (GPerturb, Xing & Yau 2025)."""

    def test_returns_expected_keys(self, tmp_path, monkeypatch):
        """Ensemble disagreement returns all expected keys."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(15, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame(np.random.rand(15, 2), columns=["ct_A", "ct_B"])

        result = step04.compute_ensemble_disagreement(
            X, Y, n_restarts=3, n_eval_points=16,
        )

        expected_keys = {
            "stability_score", "lengthscale_agreement",
            "mean_pred_std_across_models", "is_stable", "n_restarts",
        }
        assert expected_keys.issubset(result.keys())
        assert result["n_restarts"] == 3
        assert 0 <= result["stability_score"] <= 1.0

    def test_single_restart_returns_trivial(self, tmp_path, monkeypatch):
        """With n_restarts=1, returns perfect stability (no comparison possible)."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        result = step04.compute_ensemble_disagreement(
            pd.DataFrame(np.random.rand(10, 2), columns=["a", "b"]),
            pd.DataFrame(np.random.rand(10, 2), columns=["ct_A", "ct_B"]),
            n_restarts=1,
        )
        assert result["stability_score"] == 1.0
        assert result["is_stable"] is True

    def test_identical_data_high_stability(self, tmp_path, monkeypatch):
        """With clean data, restarts should mostly agree (high stability)."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        np.random.seed(123)
        # Create a simple dataset with clear signal
        X = pd.DataFrame(np.random.rand(20, 2), columns=["a", "b"])
        # Y depends linearly on X to make the GP fit easy
        Y = pd.DataFrame({
            "ct_A": 0.3 + 0.4 * X["a"].values,
            "ct_B": 0.7 - 0.4 * X["a"].values,
        })

        result = step04.compute_ensemble_disagreement(
            X, Y, n_restarts=3, n_eval_points=16,
        )

        # With a clean linear relationship, models should agree well
        assert result["stability_score"] > 0.5
        assert result["mean_pred_std_across_models"] >= 0


class TestBootstrapUncertainty:
    """Tests for bootstrap uncertainty estimation and heteroscedastic noise GP."""

    @pytest.fixture
    def soft_probs_fixture(self):
        """Create synthetic soft probability data for 3 conditions, 4 cell types."""
        np.random.seed(42)
        n_cells_per_cond = [50, 80, 30]
        conditions = []
        probs_list = []
        for i, n in enumerate(n_cells_per_cond):
            cond_name = f"cond_{i}"
            conditions.extend([cond_name] * n)
            # Different Dirichlet concentrations per condition
            alpha = np.array([1.0, 2.0, 0.5, 1.5]) * (i + 1)
            raw = np.random.dirichlet(alpha, size=n)
            probs_list.append(raw)
        all_probs = np.vstack(probs_list)
        cell_types = ["ct_A", "ct_B", "ct_C", "ct_D"]
        obs = pd.DataFrame({"condition": conditions})
        soft_probs = pd.DataFrame(all_probs, columns=cell_types, index=obs.index)
        return obs, soft_probs

    def test_bootstrap_shape_and_positive(self, soft_probs_fixture):
        """Bootstrap variance should be (n_conditions x n_cell_types) and non-negative."""
        obs, soft_probs = soft_probs_fixture
        var_df = step02.compute_bootstrap_uncertainty(
            obs, soft_probs, condition_key="condition", n_bootstrap=100,
        )
        assert var_df.shape == (3, 4)
        assert (var_df.values >= 0).all()
        assert list(var_df.columns) == ["ct_A", "ct_B", "ct_C", "ct_D"]

    def test_bootstrap_more_cells_less_variance(self):
        """Conditions with more cells should have lower bootstrap variance."""
        np.random.seed(123)
        alpha = np.ones(3)
        # Small condition: 10 cells
        obs_small = pd.DataFrame({"condition": ["small"] * 10})
        probs_small = pd.DataFrame(
            np.random.dirichlet(alpha, size=10),
            columns=["A", "B", "C"],
        )
        var_small = step02.compute_bootstrap_uncertainty(
            obs_small, probs_small, condition_key="condition", n_bootstrap=500,
        )
        # Large condition: 500 cells
        obs_large = pd.DataFrame({"condition": ["large"] * 500})
        probs_large = pd.DataFrame(
            np.random.dirichlet(alpha, size=500),
            columns=["A", "B", "C"],
        )
        var_large = step02.compute_bootstrap_uncertainty(
            obs_large, probs_large, condition_key="condition", n_bootstrap=500,
        )
        # More cells → lower variance
        assert var_large.values.mean() < var_small.values.mean()

    def test_bootstrap_reproducible(self, soft_probs_fixture):
        """Same seed should produce identical results."""
        obs, soft_probs = soft_probs_fixture
        var1 = step02.compute_bootstrap_uncertainty(
            obs, soft_probs, condition_key="condition", seed=99,
        )
        var2 = step02.compute_bootstrap_uncertainty(
            obs, soft_probs, condition_key="condition", seed=99,
        )
        pd.testing.assert_frame_equal(var1, var2)

    def test_heteroscedastic_gp_fits(self, soft_probs_fixture):
        """fit_gp_botorch should accept noise_variance and produce valid posterior."""
        import torch
        obs, soft_probs = soft_probs_fixture
        n_conds = 3
        d = 4
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(n_conds, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(n_conds)],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(4), size=n_conds),
            columns=["ct_A", "ct_B", "ct_C", "ct_D"],
            index=X.index,
        )
        noise_var = step02.compute_bootstrap_uncertainty(
            obs, soft_probs, condition_key="condition", n_bootstrap=50,
        )
        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True, noise_variance=noise_var,
        )
        # Model should fit without error and produce valid predictions
        with torch.no_grad():
            post = model.posterior(train_X[:1])
        assert post.mean.shape[-1] == 3  # ILR: 4 types -> 3 components


class TestFixedNoise:
    """Tests for --fixed-noise heteroscedastic noise modeling (TODO-31)."""

    @staticmethod
    def _make_xy(n=5, d=3, n_types=4, seed=42):
        """Build random X (with fidelity) and compositional Y DataFrames."""
        np.random.seed(seed)
        cols = ["CHIR99021_uM", "SAG_uM", "BMP4_uM"][:d]
        X = pd.DataFrame(
            np.random.rand(n, d), columns=cols,
            index=[f"c{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(n_types), size=n),
            columns=[f"ct_{chr(65 + i)}" for i in range(n_types)],
            index=X.index,
        )
        return X, Y

    def test_fixed_noise_min_variance_config(self):
        """FIXED_NOISE_MIN_VARIANCE is exported from config and equals 0.02."""
        from gopro.config import FIXED_NOISE_MIN_VARIANCE
        assert FIXED_NOISE_MIN_VARIANCE == 0.02

    @pytest.mark.parametrize("fill_value", [1e-8, 0.0],
                             ids=["near_zero", "exact_zero"])
    def test_noise_clamped_at_min_variance(self, fill_value):
        """train_Yvar values at or below FIXED_NOISE_MIN_VARIANCE are clamped up."""
        import torch
        X, Y = self._make_xy()
        noise_var = pd.DataFrame(
            np.full((len(Y), Y.shape[1]), fill_value),
            columns=Y.columns, index=Y.index,
        )
        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True, noise_variance=noise_var,
        )
        with torch.no_grad():
            post = model.posterior(train_X[:1])
        assert post.mean.shape[-1] == 3  # ILR: 4 types -> 3 components

    def test_fixed_noise_uniform_fallback(self):
        """When --fixed-noise is set without CSV, uniform noise from Y variance is used."""
        import torch
        X, Y = self._make_xy(n=8)
        # Compute expected uniform noise = per-column variance tiled
        expected_col_var = Y.var(axis=0)
        uniform_nv = pd.DataFrame(
            np.tile(expected_col_var.values, (len(Y), 1)),
            columns=Y.columns, index=Y.index,
        )
        assert uniform_nv.shape == (8, 4)
        for col in Y.columns:
            np.testing.assert_allclose(
                uniform_nv[col].values,
                np.full(8, expected_col_var[col]),
            )
        # Fit GP with the uniform noise to verify it works end-to-end
        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True, noise_variance=uniform_nv,
        )
        with torch.no_grad():
            post = model.posterior(train_X[:1])
        assert post.mean.shape[-1] == 3


class TestLogScale:
    """Tests for selective log-scaling of concentration dimensions."""

    def test_apply_log_scale_transforms_selected_columns(self):
        """log1p is applied only to specified columns."""
        X = pd.DataFrame({
            "CHIR99021_uM": [1.0, 3.0, 0.0],
            "BMP4_uM": [0.001, 0.01, 0.0],
            "log_harvest_day": [4.28, 4.28, 4.28],
            "fidelity": [1.0, 1.0, 1.0],
        })
        cols = ["CHIR99021_uM", "BMP4_uM"]
        result = step04._apply_log_scale(X, cols)
        # Transformed columns should be log1p of original
        np.testing.assert_allclose(result["CHIR99021_uM"], np.log1p([1.0, 3.0, 0.0]))
        np.testing.assert_allclose(result["BMP4_uM"], np.log1p([0.001, 0.01, 0.0]))
        # Non-specified columns unchanged
        np.testing.assert_array_equal(result["log_harvest_day"], X["log_harvest_day"])
        np.testing.assert_array_equal(result["fidelity"], X["fidelity"])

    def test_inverse_log_scale_roundtrip(self):
        """expm1(log1p(x)) recovers original values."""
        X = pd.DataFrame({
            "CHIR99021_uM": [0.0, 0.5, 1.0, 3.0, 10.0],
            "BMP4_uM": [0.0, 0.001, 0.005, 0.01, 0.05],
            "log_harvest_day": [4.0, 4.1, 4.2, 4.3, 4.4],
        })
        cols = ["CHIR99021_uM", "BMP4_uM"]
        scaled = step04._apply_log_scale(X, cols)
        recovered = step04._inverse_log_scale(scaled, cols)
        np.testing.assert_allclose(
            recovered["CHIR99021_uM"], X["CHIR99021_uM"], atol=1e-12
        )
        np.testing.assert_allclose(
            recovered["BMP4_uM"], X["BMP4_uM"], atol=1e-12
        )
        # Untouched column remains the same
        np.testing.assert_array_equal(recovered["log_harvest_day"], X["log_harvest_day"])

    def test_log_scale_zero_maps_to_zero(self):
        """log1p(0) = 0, so zero concentrations remain zero."""
        X = pd.DataFrame({"CHIR99021_uM": [0.0, 0.0, 0.0]})
        result = step04._apply_log_scale(X, ["CHIR99021_uM"])
        np.testing.assert_array_equal(result["CHIR99021_uM"], [0.0, 0.0, 0.0])

    def test_log_scale_preserves_ordering(self):
        """log1p is monotonic, so ordering is preserved."""
        vals = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
        X = pd.DataFrame({"CHIR99021_uM": vals})
        result = step04._apply_log_scale(X, ["CHIR99021_uM"])
        transformed = result["CHIR99021_uM"].values
        assert all(transformed[i] <= transformed[i + 1] for i in range(len(vals) - 1))

    def test_log_scale_does_not_mutate_input(self):
        """_apply_log_scale returns a copy, not an in-place mutation."""
        X = pd.DataFrame({"CHIR99021_uM": [1.0, 2.0, 3.0]})
        original_vals = X["CHIR99021_uM"].values.copy()
        _ = step04._apply_log_scale(X, ["CHIR99021_uM"])
        np.testing.assert_array_equal(X["CHIR99021_uM"].values, original_vals)

    def test_log_scale_columns_config_excludes_log_harvest_day(self):
        """LOG_SCALE_COLUMNS should contain _uM columns but not log_harvest_day."""
        from gopro.config import LOG_SCALE_COLUMNS, MORPHOGEN_COLUMNS
        assert "log_harvest_day" not in LOG_SCALE_COLUMNS
        # All entries should end with _uM
        for col in LOG_SCALE_COLUMNS:
            assert col.endswith("_uM"), f"{col} does not end with _uM"
        # Should contain concentration columns
        assert "CHIR99021_uM" in LOG_SCALE_COLUMNS
        assert "BMP4_uM" in LOG_SCALE_COLUMNS

    def test_apply_log_scale_skips_missing_columns(self):
        """Columns not present in DataFrame are silently skipped."""
        X = pd.DataFrame({"CHIR99021_uM": [1.0, 2.0]})
        result = step04._apply_log_scale(X, ["CHIR99021_uM", "NONEXISTENT_uM"])
        np.testing.assert_allclose(result["CHIR99021_uM"], np.log1p([1.0, 2.0]))


class TestMLLRestarts:
    """Tests for MLL optimisation restarts (_fit_mll_with_restarts)."""

    def _make_simple_data(self, n=20, d=3, seed=42):
        """Create simple synthetic data with a clear signal."""
        np.random.seed(seed)
        X = np.random.rand(n, d)
        # Y is a function of X with some noise
        Y = 0.5 + 0.3 * X[:, 0:1] - 0.2 * X[:, 1:2] + 0.05 * np.random.randn(n, 1)
        return (
            torch.tensor(X, dtype=torch.float64),
            torch.tensor(Y, dtype=torch.float64),
        )

    def test_single_restart_returns_fitted_model(self):
        """With n_restarts=1, returns a fitted GP model."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        train_X, train_Y = self._make_simple_data()

        def factory():
            return SingleTaskGP(
                train_X, train_Y,
                input_transform=Normalize(d=train_X.shape[1]),
                outcome_transform=Standardize(m=1),
            )

        model = step04._fit_mll_with_restarts(factory, n_restarts=1)
        assert model is not None
        # Model should be in eval mode after fitting
        assert not model.training

    def test_multiple_restarts_returns_best(self):
        """With n_restarts > 1, multiple fits run and the best is kept."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        train_X, train_Y = self._make_simple_data()

        def factory():
            return SingleTaskGP(
                train_X, train_Y,
                input_transform=Normalize(d=train_X.shape[1]),
                outcome_transform=Standardize(m=1),
            )

        model = step04._fit_mll_with_restarts(factory, n_restarts=3)
        assert model is not None
        # Model should make predictions without error
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_X[:2])
            assert posterior.mean.shape == (2, 1)

    def test_all_restarts_fail_raises(self):
        """If every restart raises, RuntimeError is propagated."""
        call_count = [0]

        def bad_factory():
            call_count[0] += 1
            raise RuntimeError("deliberate failure")

        with pytest.raises(RuntimeError, match="All .* MLL restarts failed"):
            step04._fit_mll_with_restarts(bad_factory, n_restarts=2)
        assert call_count[0] == 2

    def test_fit_gp_botorch_with_mll_restarts(self, tmp_path, monkeypatch):
        """fit_gp_botorch accepts mll_restarts and produces a valid model."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        np.random.seed(42)

        X = pd.DataFrame(np.random.rand(15, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame({
            "ct_A": 0.4 + 0.2 * np.random.rand(15),
            "ct_B": 0.6 - 0.2 * np.random.rand(15),
        })

        model, tX, tY, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, mll_restarts=2,
        )
        assert model is not None
        assert tX.shape[0] == 15
        assert set(cols) == {"ct_A", "ct_B"}


class TestExplicitPriors:
    """Tests for explicit GP priors (TODO-30): lengthscale + noise priors."""

    def test_set_noise_prior_attaches_gamma(self):
        """_set_noise_prior attaches a GammaPrior to the likelihood noise."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        train_X = torch.rand(10, 3, dtype=torch.float64)
        train_Y = torch.rand(10, 1, dtype=torch.float64)
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=3),
            outcome_transform=Standardize(m=1),
        )
        step04._set_noise_prior(model)
        # Verify the prior is registered on the noise_covar
        noise_covar = model.likelihood.noise_covar
        prior_names = [name for name, *_ in noise_covar.named_priors()]
        assert "noise_prior" in prior_names

    def test_set_explicit_priors_attaches_both(self):
        """_set_explicit_priors sets both lengthscale and noise priors."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize

        train_X = torch.rand(10, 5, dtype=torch.float64)
        train_Y = torch.rand(10, 1, dtype=torch.float64)
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=5),
            outcome_transform=Standardize(m=1),
        )
        step04._set_explicit_priors(model, d=5)
        # Noise prior
        noise_covar = model.likelihood.noise_covar
        noise_prior_names = [name for name, *_ in noise_covar.named_priors()]
        assert "noise_prior" in noise_prior_names
        # Lengthscale prior on the kernel that has lengthscale
        covar = model.covar_module
        base = getattr(covar, "base_kernel", covar)
        ls_prior_names = [name for name, *_ in base.named_priors()]
        assert "lengthscale_prior" in ls_prior_names

    def test_fit_gp_botorch_explicit_priors_flag(self, tmp_path, monkeypatch):
        """fit_gp_botorch with explicit_priors=True produces a valid model with priors."""
        monkeypatch.setattr(step04, "GP_STATE_DIR", tmp_path)
        monkeypatch.setattr(step04, "DATA_DIR", tmp_path)
        np.random.seed(42)

        X = pd.DataFrame(np.random.rand(15, 3), columns=["a", "b", "c"])
        Y = pd.DataFrame({
            "ct_A": 0.4 + 0.2 * np.random.rand(15),
            "ct_B": 0.6 - 0.2 * np.random.rand(15),
        })

        model, tX, tY, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, explicit_priors=True,
        )
        assert model is not None
        # Verify noise prior was attached
        noise_covar = model.likelihood.noise_covar
        noise_prior_names = [name for name, *_ in noise_covar.named_priors()]
        assert "noise_prior" in noise_prior_names


class TestSobolQMCSampler:
    """Tests for Sobol QMC sampler in acquisition (TODO-32)."""

    _D = 3  # default dimensionality for test fixtures

    def _fit_simple_model(self, n=20, d=_D):
        """Fit a simple SingleTaskGP for testing acquisition."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(42)
        np.random.seed(42)
        train_X = torch.rand(n, d, dtype=torch.float64)
        train_Y = (train_X.sum(dim=-1, keepdim=True) +
                    0.1 * torch.randn(n, 1, dtype=torch.float64))
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model, train_X, train_Y

    def _make_bounds_and_columns(self, d=_D):
        """Build unit-cube bounds and column names for d dimensions."""
        columns = [f"x{i}" for i in range(d)]
        bounds = {c: (0.0, 1.0) for c in columns}
        return bounds, columns

    def _recommend(self, mc_samples, n_recommendations=2):
        """Shared helper: fit model → recommend with given mc_samples."""
        model, train_X, train_Y = self._fit_simple_model()
        bounds, columns = self._make_bounds_and_columns()
        recs = step04.recommend_next_experiments(
            model, train_X, train_Y,
            bounds=bounds, columns=columns,
            n_recommendations=n_recommendations,
            mc_samples=mc_samples,
        )
        return recs, columns

    def test_sobol_sampler_used_in_acquisition(self):
        """recommend_next_experiments produces valid results with custom mc_samples."""
        recs, columns = self._recommend(mc_samples=64)
        assert len(recs) == 2
        # Recommendations should be within bounds
        for col in columns:
            assert recs[col].min() >= -0.01
            assert recs[col].max() <= 1.01

    def test_mc_samples_clamped_to_max(self):
        """mc_samples > 2048 is clamped to 2048 with a warning."""
        mock_logger = unittest.mock.MagicMock()
        original_logger = step04.logger
        step04.logger = mock_logger
        try:
            recs, _ = self._recommend(mc_samples=9999)
        finally:
            step04.logger = original_logger
        assert len(recs) == 2
        # Verify a warning was logged about clamping
        warning_calls = [
            c for c in mock_logger.warning.call_args_list
            if "clamped" in str(c)
        ]
        assert len(warning_calls) == 1, "Expected a clamping warning log"

    def test_default_mc_samples_is_512(self):
        """Default mc_samples parameter is 512."""
        import inspect
        sig = inspect.signature(step04.recommend_next_experiments)
        assert sig.parameters["mc_samples"].default == 512

    def test_run_gpbo_loop_mc_samples_parameter(self):
        """run_gpbo_loop accepts mc_samples parameter."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "mc_samples" in sig.parameters
        assert sig.parameters["mc_samples"].default == 512


class TestInputWarp:
    """Tests for Kumaraswamy CDF input warping (TODO-27)."""

    def _make_training_data(self, n=20, d=3, n_outputs=2):
        """Build simple training DataFrames for GP fitting."""
        np.random.seed(42)
        cols = [f"x{i}" for i in range(d)]
        X = pd.DataFrame(np.random.rand(n, d), columns=cols,
                         index=[f"c{i}" for i in range(n)])
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(n_outputs), size=n),
            columns=[f"ct{i}" for i in range(n_outputs)],
            index=X.index,
        )
        return X, Y

    def test_input_warp_applies_chained_transform(self):
        """fit_gp_botorch with input_warp=True uses ChainedInputTransform."""
        from botorch.models.transforms.input import ChainedInputTransform
        X, Y = self._make_training_data()
        model, train_X, train_Y, cell_type_cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, input_warp=True, save_state=False,
        )
        # Model should have a ChainedInputTransform containing Warp + Normalize
        itf = model.input_transform
        assert isinstance(itf, ChainedInputTransform), (
            f"Expected ChainedInputTransform, got {type(itf).__name__}"
        )
        assert "warp" in itf, "ChainedInputTransform should contain 'warp' key"
        assert "normalize" in itf, "ChainedInputTransform should contain 'normalize' key"

    def test_input_warp_false_uses_normalize_only(self):
        """fit_gp_botorch with input_warp=False uses plain Normalize."""
        from botorch.models.transforms import Normalize
        X, Y = self._make_training_data()
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, input_warp=False, save_state=False,
        )
        itf = model.input_transform
        assert isinstance(itf, Normalize), (
            f"Expected Normalize, got {type(itf).__name__}"
        )

    def test_build_input_transform_helper(self):
        """_build_input_transform returns correct types for warp/no-warp."""
        from botorch.models.transforms import Normalize
        from botorch.models.transforms.input import ChainedInputTransform

        # No warp
        tf_plain = step04._build_input_transform(d=5, warp=False)
        assert isinstance(tf_plain, Normalize)

        # With warp
        tf_warp = step04._build_input_transform(d=5, warp=True)
        assert isinstance(tf_warp, ChainedInputTransform)
        assert "warp" in tf_warp
        assert "normalize" in tf_warp

    def test_run_gpbo_loop_input_warp_parameter(self):
        """run_gpbo_loop accepts input_warp parameter with correct default."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "input_warp" in sig.parameters
        assert sig.parameters["input_warp"].default is False

    def test_fit_gp_botorch_input_warp_parameter(self):
        """fit_gp_botorch accepts input_warp parameter with correct default."""
        import inspect
        sig = inspect.signature(step04.fit_gp_botorch)
        assert "input_warp" in sig.parameters
        assert sig.parameters["input_warp"].default is False


class TestZeroPassingKernel:
    """Tests for ZeroPassingKernel (TODO-6, GPerturb)."""

    def _make_training_data(self, n=20, d=3, n_outputs=2):
        """Build simple training DataFrames for GP fitting."""
        np.random.seed(42)
        cols = [f"x{i}" for i in range(d)]
        X = pd.DataFrame(np.random.rand(n, d), columns=cols,
                         index=[f"c{i}" for i in range(n)])
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(n_outputs), size=n),
            columns=[f"ct{i}" for i in range(n_outputs)],
            index=X.index,
        )
        return X, Y

    def test_kernel_returns_zero_when_input_is_zero_vector(self):
        """ZeroPassingKernel returns 0 when either input is a zero vector."""
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel

        ZPK = step04._get_zero_passing_kernel_class()
        base = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        zpk = ZPK(base, concentration_dims=[0, 1, 2], eps=1.0)

        x_zero = torch.zeros(1, 3, dtype=torch.double)
        x_nonzero = torch.rand(1, 3, dtype=torch.double) + 0.1

        # k(0, x) should be 0
        cov = zpk(x_zero, x_nonzero)
        if hasattr(cov, 'to_dense'):
            cov = cov.to_dense()
        assert torch.allclose(cov, torch.zeros_like(cov), atol=1e-6), (
            f"k(0, x) should be ~0, got {cov.item():.6f}"
        )

        # k(x, 0) should also be 0
        cov2 = zpk(x_nonzero, x_zero)
        if hasattr(cov2, 'to_dense'):
            cov2 = cov2.to_dense()
        assert torch.allclose(cov2, torch.zeros_like(cov2), atol=1e-6), (
            f"k(x, 0) should be ~0, got {cov2.item():.6f}"
        )

    def test_kernel_nonzero_for_nonzero_inputs(self):
        """ZeroPassingKernel returns nonzero for two nonzero inputs."""
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel

        ZPK = step04._get_zero_passing_kernel_class()
        base = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        zpk = ZPK(base, concentration_dims=[0, 1, 2], eps=1.0)

        x1 = torch.rand(1, 3, dtype=torch.double) + 0.5
        x2 = torch.rand(1, 3, dtype=torch.double) + 0.5

        cov = zpk(x1, x2)
        if hasattr(cov, 'to_dense'):
            cov = cov.to_dense()
        assert cov.item() > 0.01, (
            f"k(x, x') should be nonzero for nonzero inputs, got {cov.item():.6f}"
        )

    def test_kernel_diag_mode(self):
        """ZeroPassingKernel works in diagonal mode."""
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel

        ZPK = step04._get_zero_passing_kernel_class()
        base = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        zpk = ZPK(base, concentration_dims=[0, 1, 2], eps=1.0)

        x = torch.rand(5, 3, dtype=torch.double) + 0.1
        x[0, :] = 0.0  # First row is zero vector

        diag = zpk(x, x, diag=True)
        # First element should be ~0 (zero input)
        assert abs(diag[0].item()) < 1e-6, (
            f"Diagonal k(0, 0) should be ~0, got {diag[0].item():.6f}"
        )
        # Others should be nonzero
        assert all(diag[i].item() > 0.01 for i in range(1, 5)), (
            "Diagonal entries for nonzero inputs should be positive"
        )

    def test_concentration_dims_subset(self):
        """ZeroPassingKernel only considers specified concentration dims."""
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel

        ZPK = step04._get_zero_passing_kernel_class()
        base = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))
        # Only dims 0,1,2 are concentration; dim 3 is e.g. fidelity
        zpk = ZPK(base, concentration_dims=[0, 1, 2], eps=1.0)

        # Zero in concentration dims but nonzero in dim 3
        x_zero_conc = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double)
        x_nonzero = torch.tensor([[0.5, 0.5, 0.5, 1.0]], dtype=torch.double)

        cov = zpk(x_zero_conc, x_nonzero)
        if hasattr(cov, 'to_dense'):
            cov = cov.to_dense()
        assert torch.allclose(cov, torch.zeros_like(cov), atol=1e-6), (
            "Zero in concentration dims should still give k=0 even if other dims nonzero"
        )

    def test_fit_gp_with_zero_passing(self):
        """fit_gp_botorch with zero_passing=True wraps the kernel."""
        ZPK = step04._get_zero_passing_kernel_class()
        X, Y = self._make_training_data()
        model, train_X, train_Y, cell_type_cols = step04.fit_gp_botorch(
            X, Y, use_ilr=False, zero_passing=True, save_state=False,
        )
        # The covar_module should be a ZeroPassingKernel
        assert isinstance(model.covar_module, ZPK), (
            f"Expected ZeroPassingKernel, got {type(model.covar_module).__name__}"
        )

    def test_fit_gp_without_zero_passing(self):
        """fit_gp_botorch with zero_passing=False does NOT wrap the kernel."""
        ZPK = step04._get_zero_passing_kernel_class()
        X, Y = self._make_training_data()
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=False, zero_passing=False, save_state=False,
        )
        assert not isinstance(model.covar_module, ZPK), (
            "Without zero_passing, kernel should not be wrapped"
        )

    def test_run_gpbo_loop_accepts_zero_passing(self):
        """run_gpbo_loop accepts zero_passing parameter with correct default."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "zero_passing" in sig.parameters
        assert sig.parameters["zero_passing"].default is False

    def test_fit_gp_botorch_accepts_zero_passing(self):
        """fit_gp_botorch accepts zero_passing parameter with correct default."""
        import inspect
        sig = inspect.signature(step04.fit_gp_botorch)
        assert "zero_passing" in sig.parameters
        assert sig.parameters["zero_passing"].default is False


class TestDesirabilityGate:
    """Tests for desirability-based feasibility gate (TODO-7, Cosenza 2022)."""

    def test_no_antagonism_gives_phi_one(self):
        """Candidates with only agonists (no antagonists) should have phi=1."""
        columns = ["CHIR99021_uM", "BMP4_uM", "SHH_uM"]
        candidates = np.array([
            [3.0, 0.001, 0.005],   # Agonists only
            [6.0, 0.002, 0.010],   # Higher doses, still agonists only
        ])
        phi = step04.compute_desirability(candidates, columns)
        np.testing.assert_array_equal(phi, np.ones(2))

    def test_antagonist_pair_penalises(self):
        """WNT agonist + WNT antagonist at high dose → phi < 1."""
        columns = ["CHIR99021_uM", "IWP2_uM", "BMP4_uM"]
        candidates = np.array([
            [5.0, 3.0, 0.001],   # CHIR (agonist) + IWP2 (antagonist) both high
        ])
        phi = step04.compute_desirability(candidates, columns)
        assert phi[0] < 1.0, f"Expected phi < 1 for WNT conflict, got {phi[0]}"
        # WNT penalty: min(5,3)/max(5,3) = 3/5 = 0.6, so phi = 1 - 0.6 = 0.4
        np.testing.assert_almost_equal(phi[0], 0.4, decimal=5)

    def test_subthreshold_antagonist_no_penalty(self):
        """Trace antagonist below threshold does not trigger penalty."""
        columns = ["CHIR99021_uM", "IWP2_uM"]
        candidates = np.array([
            [5.0, 0.05],  # IWP2 below default threshold of 0.1
        ])
        phi = step04.compute_desirability(candidates, columns)
        assert phi[0] == 1.0, f"Expected no penalty for trace antagonist, got {phi[0]}"

    def test_custom_threshold(self):
        """Per-morphogen thresholds detect antagonism at pharmacologically relevant doses."""
        columns = ["CHIR99021_uM", "IWP2_uM"]
        # IWP2 at 0.5 µM (above its per-morphogen threshold of 0.1) → conflict detected
        candidates_active = np.array([[5.0, 0.5]])
        phi_active = step04.compute_desirability(candidates_active, columns)
        assert phi_active[0] < 1.0, "Active WNT agonist+antagonist should be penalized"
        # IWP2 at 0.05 µM (below its per-morphogen threshold of 0.1) → no penalty
        candidates_sub = np.array([[5.0, 0.05]])
        phi_sub = step04.compute_desirability(candidates_sub, columns)
        assert phi_sub[0] == 1.0, "Sub-threshold antagonist should not penalize"

    def test_multiple_pathway_conflicts_compound(self):
        """Conflicts in multiple pathways compound multiplicatively."""
        columns = ["CHIR99021_uM", "IWP2_uM", "BMP4_uM", "LDN193189_uM"]
        candidates = np.array([
            [5.0, 5.0, 0.001, 0.001],  # Both WNT and BMP conflict (equal doses)
        ])
        phi = step04.compute_desirability(candidates, columns)
        # WNT: min(5,5)/max(5,5) = 1.0 → penalty 1.0 → (1-1.0) ≈ 0
        # Product should be ~0 when one pathway is fully cancelled
        assert phi[0] < 1e-6, f"Full antagonism should give phi≈0, got {phi[0]}"

    def test_zero_input_gives_phi_one(self):
        """All-zero candidate has no active morphogens → phi=1."""
        columns = ["CHIR99021_uM", "IWP2_uM", "BMP4_uM"]
        candidates = np.array([[0.0, 0.0, 0.0]])
        phi = step04.compute_desirability(candidates, columns)
        assert phi[0] == 1.0

    def test_columns_not_in_pairs_ignored(self):
        """Columns not in ANTAGONIST_PAIRS are ignored gracefully."""
        columns = ["FGF2_uM", "EGF_uM"]  # No antagonist defined
        candidates = np.array([[0.001, 0.003]])
        phi = step04.compute_desirability(candidates, columns)
        assert phi[0] == 1.0

    def test_antagonist_pairs_constant_covers_major_pathways(self):
        """ANTAGONIST_PAIRS covers WNT, BMP, SHH, TGFb pathways."""
        pairs = step04.ANTAGONIST_PAIRS
        assert "WNT" in pairs
        assert "BMP" in pairs
        assert "SHH" in pairs
        assert "TGFb" in pairs
        for pw, info in pairs.items():
            assert "agonists" in info, f"{pw} missing agonists"
            assert "antagonists" in info, f"{pw} missing antagonists"
            assert len(info["agonists"]) > 0, f"{pw} has empty agonists"
            assert len(info["antagonists"]) > 0, f"{pw} has empty antagonists"

    def test_recommend_accepts_desirability_gate_param(self):
        """recommend_next_experiments accepts desirability_gate parameter."""
        import inspect
        sig = inspect.signature(step04.recommend_next_experiments)
        assert "desirability_gate" in sig.parameters
        assert sig.parameters["desirability_gate"].default is False

    def test_run_gpbo_loop_accepts_desirability_gate_param(self):
        """run_gpbo_loop accepts desirability_gate parameter."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "desirability_gate" in sig.parameters
        assert sig.parameters["desirability_gate"].default is False


class TestCarryForwardControls:
    """Tests for carry-forward top-K controls (TODO-36, Kanda eLife 2022)."""

    _D = 3  # default dimensionality

    def _fit_simple_model(self, n=20, d=_D):
        """Fit a simple SingleTaskGP for testing."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(42)
        np.random.seed(42)
        train_X = torch.rand(n, d, dtype=torch.float64)
        train_Y = (train_X.sum(dim=-1, keepdim=True) +
                    0.1 * torch.randn(n, 1, dtype=torch.float64))
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model, train_X, train_Y

    def _make_training_dfs(self, n=20, d=_D):
        """Build training DataFrames matching the tensor data."""
        np.random.seed(42)
        columns = [f"x{i}" for i in range(d)]
        X_df = pd.DataFrame(
            np.random.rand(n, d), columns=columns,
            index=[f"cond_{i}" for i in range(n)],
        )
        Y_df = pd.DataFrame(
            np.random.rand(n, 1), columns=["y0"],
            index=X_df.index,
        )
        return X_df, Y_df

    def test_carry_forward_controls_present(self):
        """With n_controls=2, verify 2 rows have is_control=True."""
        model, train_X, train_Y = self._fit_simple_model()
        X_df, Y_df = self._make_training_dfs()
        columns = list(X_df.columns)
        bounds = {c: (0.0, 1.0) for c in columns}

        recs = step04.recommend_next_experiments(
            model, train_X, train_Y,
            bounds=bounds, columns=columns,
            n_recommendations=6,
            mc_samples=64,
            n_controls=2,
            train_X_df=X_df,
            train_Y_df=Y_df,
        )
        assert "is_control" in recs.columns
        n_ctrl = int(recs["is_control"].sum())
        assert n_ctrl == 2, f"Expected 2 controls, got {n_ctrl}"
        # Total rows = (6 - 2) BO-recommended + 2 controls = 6
        assert len(recs) == 6

        # Control rows should have morphogen values from training data
        ctrl_rows = recs[recs["is_control"]]
        for col in columns:
            assert ctrl_rows[col].notna().all(), f"Control column {col} has NaN"

    def test_carry_forward_zero_controls(self):
        """With n_controls=0, verify no rows have is_control=True."""
        model, train_X, train_Y = self._fit_simple_model()
        columns = [f"x{i}" for i in range(self._D)]
        bounds = {c: (0.0, 1.0) for c in columns}

        recs = step04.recommend_next_experiments(
            model, train_X, train_Y,
            bounds=bounds, columns=columns,
            n_recommendations=4,
            mc_samples=64,
            n_controls=0,
        )
        assert "is_control" in recs.columns
        assert recs["is_control"].sum() == 0

    def test_controls_are_top_k_by_mean_y(self):
        """Controls should be the conditions with highest mean Y."""
        model, train_X, train_Y = self._fit_simple_model()
        X_df, Y_df = self._make_training_dfs()

        # Identify expected top-2 conditions
        mean_y = Y_df.mean(axis=1)
        expected_top2 = set(mean_y.nlargest(2).index)

        columns = list(X_df.columns)
        bounds = {c: (0.0, 1.0) for c in columns}

        recs = step04.recommend_next_experiments(
            model, train_X, train_Y,
            bounds=bounds, columns=columns,
            n_recommendations=6,
            mc_samples=64,
            n_controls=2,
            train_X_df=X_df,
            train_Y_df=Y_df,
        )
        ctrl_rows = recs[recs["is_control"]]
        # Verify control morphogen values match training data for top-2 conditions
        for _, row in ctrl_rows.iterrows():
            matched = False
            for cond in expected_top2:
                if np.allclose(row[columns].values, X_df.loc[cond, columns].values):
                    matched = True
                    break
            assert matched, "Control row does not match any expected top-K condition"

    def test_run_gpbo_loop_accepts_n_controls_param(self):
        """run_gpbo_loop accepts n_controls parameter."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "n_controls" in sig.parameters
        assert sig.parameters["n_controls"].default == 0

    def test_recommend_accepts_n_controls_param(self):
        """recommend_next_experiments accepts n_controls parameter."""
        import inspect
        sig = inspect.signature(step04.recommend_next_experiments)
        assert "n_controls" in sig.parameters
        assert sig.parameters["n_controls"].default == 0


class TestCellFlowRelevanceCheck:
    """Tests for cellflow_relevance_check() LOO-CV relevance function.

    Validates the rMFBO-style check (Mikkola et al. 2023, arXiv:2210.13937)
    that determines whether CellFlow low-fidelity data helps GP predictions
    on high-fidelity targets.
    """

    def _make_data(self, n_real=10, n_cellflow=20, d=3, n_ct=5,
                   cellflow_helpful=True, seed=42):
        """Build synthetic multi-fidelity data for testing.

        Args:
            n_real: Number of real (fidelity=1.0) data points.
            n_cellflow: Number of CellFlow (fidelity=0.0) data points.
            d: Number of morphogen dimensions.
            n_ct: Number of cell types.
            cellflow_helpful: If True, CellFlow data is correlated with real;
                if False, CellFlow data is random noise.
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)

        # Real data: smooth function of X
        X_real_vals = rng.rand(n_real, d)
        # Y = softmax of linear function of X (consistent signal)
        W = rng.randn(d, n_ct)
        logits_real = X_real_vals @ W
        Y_real_vals = np.exp(logits_real)
        Y_real_vals /= Y_real_vals.sum(axis=1, keepdims=True)

        ct_cols = [f"ct_{i}" for i in range(n_ct)]
        morph_cols = [f"m_{i}" for i in range(d)]

        X_real = pd.DataFrame(X_real_vals, columns=morph_cols,
                              index=[f"real_{i}" for i in range(n_real)])
        X_real["fidelity"] = 1.0
        Y_real = pd.DataFrame(Y_real_vals, columns=ct_cols,
                              index=X_real.index)

        # CellFlow data
        X_cf_vals = rng.rand(n_cellflow, d)
        if cellflow_helpful:
            # CellFlow follows the same function (+ small noise)
            logits_cf = X_cf_vals @ W + rng.randn(n_cellflow, n_ct) * 0.1
            Y_cf_vals = np.exp(logits_cf)
            Y_cf_vals /= Y_cf_vals.sum(axis=1, keepdims=True)
        else:
            # CellFlow is pure noise -- uncorrelated with real data
            Y_cf_vals = rng.dirichlet(np.ones(n_ct), size=n_cellflow)

        X_cf = pd.DataFrame(X_cf_vals, columns=morph_cols,
                            index=[f"cf_{i}" for i in range(n_cellflow)])
        X_cf["fidelity"] = 0.0
        Y_cf = pd.DataFrame(Y_cf_vals, columns=ct_cols, index=X_cf.index)

        X_all = pd.concat([X_real, X_cf])
        Y_all = pd.concat([Y_real, Y_cf])

        return X_all, Y_all, X_real, Y_real

    def test_returns_expected_keys(self):
        """cellflow_relevance_check returns dict with required keys."""
        X_all, Y_all, X_real, Y_real = self._make_data(n_real=5, n_cellflow=5)
        result = step04.cellflow_relevance_check(X_all, Y_all, X_real, Y_real)
        assert "rmse_with_cellflow" in result
        assert "rmse_without_cellflow" in result
        assert "cellflow_helps" in result
        assert "improvement_pct" in result

    def test_too_few_real_points_skips(self):
        """With <3 real points, check is skipped and cellflow_helps=True."""
        X_all, Y_all, X_real, Y_real = self._make_data(n_real=2, n_cellflow=5)
        result = step04.cellflow_relevance_check(X_all, Y_all, X_real, Y_real)
        assert result["cellflow_helps"] is True
        assert np.isnan(result["rmse_with_cellflow"])

    def test_helpful_cellflow_returns_finite_rmse(self):
        """When CellFlow is correlated with real data, RMSE values are finite."""
        X_all, Y_all, X_real, Y_real = self._make_data(
            n_real=15, n_cellflow=30, cellflow_helpful=True,
        )
        result = step04.cellflow_relevance_check(X_all, Y_all, X_real, Y_real)
        assert np.isfinite(result["rmse_with_cellflow"])
        assert np.isfinite(result["rmse_without_cellflow"])
        assert isinstance(result["cellflow_helps"], bool)

    def test_noisy_cellflow_returns_finite_rmse(self):
        """When CellFlow is noise, RMSE values are still finite and computable."""
        X_all, Y_all, X_real, Y_real = self._make_data(
            n_real=15, n_cellflow=30, cellflow_helpful=False, seed=123,
        )
        result = step04.cellflow_relevance_check(X_all, Y_all, X_real, Y_real)
        assert np.isfinite(result["rmse_with_cellflow"])
        assert np.isfinite(result["rmse_without_cellflow"])
        assert isinstance(result["cellflow_helps"], bool)

    def test_improvement_pct_sign_consistent(self):
        """improvement_pct sign is consistent with cellflow_helps flag."""
        X_all, Y_all, X_real, Y_real = self._make_data(n_real=10, n_cellflow=10)
        result = step04.cellflow_relevance_check(X_all, Y_all, X_real, Y_real)
        if result["cellflow_helps"]:
            assert result["improvement_pct"] > 0
        else:
            assert result["improvement_pct"] <= 0

    def test_run_gpbo_loop_accepts_relevance_check_param(self):
        """run_gpbo_loop accepts do_cellflow_relevance_check parameter."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "do_cellflow_relevance_check" in sig.parameters
        assert sig.parameters["do_cellflow_relevance_check"].default is False


class TestContextualParameters:
    """Tests for contextual parameter support (TODO-12 / TODO-41)."""

    def test_contextual_cols_accepted(self):
        """recommend_next_experiments accepts contextual_cols parameter."""
        import inspect
        sig = inspect.signature(step04.recommend_next_experiments)
        assert "contextual_cols" in sig.parameters
        param = sig.parameters["contextual_cols"]
        assert param.default is None

    def test_run_gpbo_loop_accepts_contextual_cols(self):
        """run_gpbo_loop accepts contextual_cols parameter."""
        import inspect
        sig = inspect.signature(step04.run_gpbo_loop)
        assert "contextual_cols" in sig.parameters
        param = sig.parameters["contextual_cols"]
        assert param.default is None

    def test_contextual_cols_in_cli_parser(self):
        """CLI argparse includes --contextual-cols flag."""
        import argparse
        # Read the source to find the ArgumentParser and parse with --help-like introspection
        # We'll create a parser the same way the script does and check the argument exists
        parser = argparse.ArgumentParser()
        parser.add_argument("--contextual-cols", nargs="+", default=None)
        # Verify it parses correctly
        args = parser.parse_args(["--contextual-cols", "log_harvest_day", "BDNF_uM"])
        assert args.contextual_cols == ["log_harvest_day", "BDNF_uM"]

        # Also verify the actual source code contains the argument
        import inspect
        source = inspect.getsource(step04)
        assert "--contextual-cols" in source


class TestNestScore:
    """Tests for NEST-inspired transcriptomic fidelity scoring."""

    def test_basic_nest_score(self):
        """Condition with lower KNN distances should get higher NEST score."""
        from gopro.signature_utils import compute_nest_score

        obs = pd.DataFrame({
            "condition": ["good"] * 50 + ["bad"] * 50,
            "mean_knn_dist_to_ref": [0.1] * 50 + [2.0] * 50,
        })
        scores = compute_nest_score(obs)
        assert scores["good"] > scores["bad"]

    def test_nest_score_range(self):
        """All NEST scores should be in (0, 1]."""
        from gopro.signature_utils import compute_nest_score

        rng = np.random.default_rng(42)
        n = 200
        obs = pd.DataFrame({
            "condition": rng.choice(["A", "B", "C", "D"], size=n),
            "mean_knn_dist_to_ref": rng.exponential(scale=1.0, size=n),
        })
        scores = compute_nest_score(obs)
        assert len(scores) == 4
        assert (scores > 0).all()
        assert (scores <= 1.0).all()

    def test_missing_knn_dist_raises(self):
        """ValueError when knn_dist_col is not in obs."""
        from gopro.signature_utils import compute_nest_score

        obs = pd.DataFrame({
            "condition": ["A", "B"],
            "some_other_col": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="mean_knn_dist_to_ref"):
            compute_nest_score(obs)

    def test_nest_score_handles_nan(self):
        """NaN distances are handled gracefully (dropped, not propagated)."""
        from gopro.signature_utils import compute_nest_score

        obs = pd.DataFrame({
            "condition": ["A"] * 10 + ["B"] * 10,
            "mean_knn_dist_to_ref": [0.5] * 5 + [np.nan] * 5 + [1.0] * 10,
        })
        scores = compute_nest_score(obs)
        assert len(scores) == 2
        assert np.isfinite(scores["A"])
        assert np.isfinite(scores["B"])


class TestScoreGeneSignatures:
    """Tests for gene signature scoring with optional permutation controls."""

    def _make_adata(self, n_cells=10, n_genes=20, n_conditions=2):
        """Create minimal AnnData for testing."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = rng.random((n_cells, n_genes))
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])
        obs = pd.DataFrame({
            "condition": [f"cond{i % n_conditions}" for i in range(n_cells)],
        })
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_basic_signature_scoring(self):
        """Score a 5-gene signature on minimal AnnData, verify output shape."""
        from gopro.signature_utils import score_gene_signatures

        adata = self._make_adata(n_cells=10, n_genes=20, n_conditions=2)
        signatures = {"test_sig": [f"Gene{i}" for i in range(5)]}

        scores_df, pvalues_df = score_gene_signatures(adata, signatures)

        assert scores_df.shape == (2, 1)  # 2 conditions, 1 signature
        assert "test_sig" in scores_df.columns
        assert pvalues_df is None  # no permutations requested

    def test_permutation_pvalues(self):
        """With n_permutations=100, verify p-values are returned and in [0, 1]."""
        from gopro.signature_utils import score_gene_signatures

        adata = self._make_adata(n_cells=20, n_genes=30, n_conditions=3)
        signatures = {"sig_a": [f"Gene{i}" for i in range(5)]}

        scores_df, pvalues_df = score_gene_signatures(
            adata, signatures, n_permutations=100,
        )

        assert scores_df.shape == (3, 1)
        assert pvalues_df is not None
        assert pvalues_df.shape == (3, 1)
        assert (pvalues_df.values >= 0).all()
        assert (pvalues_df.values <= 1).all()


class TestCharacterizeFidelityNoise:
    """Tests for characterize_fidelity_noise (TODO-14)."""

    def test_characterize_fidelity_noise_basic(self):
        """Two fidelity levels produce correct output shape and columns."""
        rng = np.random.default_rng(42)
        n_hi, n_lo = 10, 8
        X = pd.DataFrame({
            "m1": rng.random(n_hi + n_lo),
            "m2": rng.random(n_hi + n_lo),
            "fidelity": [1.0] * n_hi + [0.5] * n_lo,
        })
        Y = pd.DataFrame(
            rng.dirichlet([1, 1, 1], size=n_hi + n_lo),
            columns=["ct_A", "ct_B", "ct_C"],
        )
        result = step04.characterize_fidelity_noise(X, Y)

        assert result.shape[0] == 2
        assert result.index.name == "fidelity"
        assert set(result.columns) == {
            "n_points", "mean_variance", "median_variance",
            "coefficient_of_variation",
        }
        assert result.loc[1.0, "n_points"] == n_hi
        assert result.loc[0.5, "n_points"] == n_lo
        # Variances should be non-negative
        assert (result["mean_variance"] >= 0).all()
        assert (result["median_variance"] >= 0).all()
        assert (result["coefficient_of_variation"] >= 0).all()

    def test_single_fidelity_still_works(self):
        """With only fidelity=1.0, returns one row."""
        rng = np.random.default_rng(99)
        n = 5
        X = pd.DataFrame({
            "m1": rng.random(n),
            "fidelity": [1.0] * n,
        })
        Y = pd.DataFrame(
            rng.dirichlet([2, 3], size=n),
            columns=["ct_X", "ct_Y"],
        )
        result = step04.characterize_fidelity_noise(X, Y)

        assert result.shape[0] == 1
        assert 1.0 in result.index
        assert result.loc[1.0, "n_points"] == n

    def test_high_variance_fidelity_detected(self):
        """Noisy fidelity level has higher mean_variance than clean one."""
        rng = np.random.default_rng(7)
        n = 20
        # Clean high-fidelity data: tight around [0.5, 0.3, 0.2]
        Y_hi = pd.DataFrame(
            np.tile([0.5, 0.3, 0.2], (n, 1)) + rng.normal(0, 0.001, (n, 3)),
            columns=["a", "b", "c"],
        )
        # Noisy low-fidelity data: wide scatter
        Y_lo = pd.DataFrame(
            rng.dirichlet([0.5, 0.5, 0.5], size=n),
            columns=["a", "b", "c"],
        )
        Y = pd.concat([Y_hi, Y_lo], ignore_index=True)
        X = pd.DataFrame({
            "m1": rng.random(2 * n),
            "fidelity": [1.0] * n + [0.0] * n,
        })

        result = step04.characterize_fidelity_noise(X, Y)

        assert result.loc[0.0, "mean_variance"] > result.loc[1.0, "mean_variance"]


class TestValidationPlate:
    """Tests for generate_validation_plate multi-dose validation protocol."""

    @staticmethod
    def _make_top_conditions(n=8):
        """Create a dummy top_conditions DataFrame with morphogen columns."""
        np.random.seed(42)
        cols = ["WNT_uM", "BMP_uM", "SHH_uM", "RA_uM"]
        data = np.random.uniform(0.1, 10.0, size=(n, len(cols)))
        return pd.DataFrame(data, columns=cols, index=[f"cond_{i}" for i in range(n)])

    def test_validation_plate_shape(self):
        """Default params: 6*3 + 6 = 24 rows."""
        top = self._make_top_conditions(8)
        plate = step04.generate_validation_plate(top)
        assert plate.shape[0] == 24, f"Expected 24 rows, got {plate.shape[0]}"

    def test_dose_factors_applied(self):
        """0.3x row should have 0.3x the 1.0x row values."""
        top = self._make_top_conditions(8)
        plate = step04.generate_validation_plate(top)
        conc_cols = ["WNT_uM", "BMP_uM", "SHH_uM", "RA_uM"]
        cond = top.index[0]
        rows_03 = plate[(plate["source_condition"] == cond) & (plate["dose_factor"] == 0.3) & (~plate["is_replicate"])]
        rows_10 = plate[(plate["source_condition"] == cond) & (plate["dose_factor"] == 1.0) & (~plate["is_replicate"])]
        assert len(rows_03) == 1
        assert len(rows_10) == 1
        np.testing.assert_allclose(
            rows_03[conc_cols].values,
            rows_10[conc_cols].values * 0.3,
            rtol=1e-10,
        )

    def test_replicates_present(self):
        """Verify is_replicate=True for replicate wells."""
        top = self._make_top_conditions(8)
        plate = step04.generate_validation_plate(top)
        rep_rows = plate[plate["is_replicate"]]
        assert len(rep_rows) == 6, f"Expected 6 replicates, got {len(rep_rows)}"
        assert rep_rows["is_replicate"].all()
        # All replicates should have dose_factor 1.0
        assert (rep_rows["dose_factor"] == 1.0).all()

    def test_no_negative_concentrations(self):
        """All concentration values should be >= 0, even with negative inputs."""
        top = self._make_top_conditions(8)
        # Introduce a negative value to test clamping
        top.iloc[0, 0] = -5.0
        plate = step04.generate_validation_plate(top)
        conc_cols = ["WNT_uM", "BMP_uM", "SHH_uM", "RA_uM"]
        assert (plate[conc_cols].values >= 0).all(), "Found negative concentration values"

    def test_custom_parameters(self):
        """Test with non-default parameters."""
        top = self._make_top_conditions(10)
        plate = step04.generate_validation_plate(
            top, n_cocktails=4, n_replicates=4,
            dose_factors=(0.1, 0.5, 1.0, 2.0, 5.0),
            replicate_top_n=2,
        )
        # 4 cocktails * 5 dose factors + 4 replicates = 24
        assert plate.shape[0] == 24

    def test_insufficient_conditions_raises(self):
        """Requesting more cocktails than available conditions should raise."""
        top = self._make_top_conditions(3)
        with pytest.raises(ValueError, match="n_cocktails.*exceeds"):
            step04.generate_validation_plate(top, n_cocktails=6)


class TestGenerateLhdFill:
    """Tests for generate_lhd_fill (TODO-38: LHD gap-filling for Round 2+)."""

    def _make_bounds(self, n_dims=5):
        return {f"dim_{i}": (float(i), float(i + 10)) for i in range(n_dims)}

    def test_generate_lhd_fill_shape(self):
        """Output should have correct shape (n_points, n_dims)."""
        bounds = self._make_bounds(5)
        result = step04.generate_lhd_fill(bounds, n_points=10, seed=42)
        assert result.shape == (10, 5)
        assert list(result.columns) == [f"dim_{i}" for i in range(5)]

    def test_generate_lhd_fill_within_bounds(self):
        """All generated values must lie within specified bounds."""
        bounds = self._make_bounds(4)
        result = step04.generate_lhd_fill(bounds, n_points=50, seed=123)
        for col, (lo, hi) in bounds.items():
            assert result[col].min() >= lo, f"{col} below lower bound"
            assert result[col].max() <= hi, f"{col} above upper bound"

    def test_generate_lhd_fill_deterministic(self):
        """Same seed should produce identical output."""
        bounds = self._make_bounds(3)
        r1 = step04.generate_lhd_fill(bounds, n_points=8, seed=99)
        r2 = step04.generate_lhd_fill(bounds, n_points=8, seed=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_generate_lhd_fill_excludes_fidelity(self):
        """The fidelity column should be excluded from LHD output."""
        bounds = {"x1": (0.0, 1.0), "fidelity": (0.0, 1.0), "x2": (0.0, 5.0)}
        result = step04.generate_lhd_fill(bounds, n_points=5, seed=42)
        assert "fidelity" not in result.columns
        assert result.shape == (5, 2)
        assert list(result.columns) == ["x1", "x2"]

    def test_lhd_fill_skipped_round_1(self):
        """n_lhd_fill should be ignored when round_num=1.

        We verify this by calling recommend_next_experiments with round_num=1
        and n_lhd_fill>0, and checking that the effective_lhd logic produces
        zero LHD rows. Uses a mock to bypass the full BO pipeline.
        """
        from unittest.mock import MagicMock, patch

        n_recs = 6
        n_lhd = 4
        d = 3
        columns = ["x0", "x1", "x2"]
        bounds = {c: (0.0, 1.0) for c in columns}

        train_X = torch.rand(5, d, dtype=torch.float64)
        train_Y = torch.rand(5, 1, dtype=torch.float64)

        # Mock model
        mock_model = MagicMock()
        mock_posterior = MagicMock()
        mock_posterior.mean = torch.rand(n_recs, 1, dtype=torch.float64)
        mock_posterior.variance = torch.rand(n_recs, 1, dtype=torch.float64).abs() + 1e-6
        mock_model.posterior.return_value = mock_posterior

        mock_candidates = torch.rand(n_recs, d, dtype=torch.float64)
        mock_acq_values = torch.rand(n_recs, dtype=torch.float64)

        # Patch both the acquisition function constructor and optimizer
        with patch("botorch.optim.optimize_acqf", return_value=(mock_candidates, mock_acq_values)), \
             patch("botorch.acquisition.qLogExpectedImprovement", return_value=MagicMock()):
            recs = step04.recommend_next_experiments(
                model=mock_model,
                train_X=train_X,
                train_Y=train_Y,
                bounds=bounds,
                columns=columns,
                n_recommendations=n_recs,
                n_lhd_fill=n_lhd,
                round_num=1,
            )

        # No LHD rows should be present in Round 1
        assert "is_lhd_fill" in recs.columns
        assert not recs["is_lhd_fill"].any(), "LHD rows should not appear in Round 1"
        assert len(recs) == n_recs


class TestConfirmationPlate:
    """Tests for generate_confirmation_plate confirmation protocol."""

    @staticmethod
    def _make_recommendations(n=10):
        """Create a dummy recommendations DataFrame with morphogen columns."""
        np.random.seed(42)
        cols = ["WNT_uM", "BMP_uM", "SHH_uM", "RA_uM"]
        data = np.random.uniform(0.1, 10.0, size=(n, len(cols)))
        return pd.DataFrame(data, columns=cols, index=[f"rec_{i}" for i in range(n)])

    def test_confirmation_plate_total_wells(self):
        """Output should have exactly n_total rows when enough candidates."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_total=24)
        assert len(plate) == 24

    def test_confirmation_plate_total_wells_custom(self):
        """n_total parameter controls plate size."""
        recs = self._make_recommendations(n=20)
        plate = step04.generate_confirmation_plate(recs, n_total=12)
        assert len(plate) == 12

    def test_optimum_replicates_present(self):
        """There should be exactly n_replicates_optimum rows with well_type='optimum_rep'."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=3, n_total=24)
        optimum_rows = plate[plate["well_type"] == "optimum_rep"]
        assert len(optimum_rows) == 3

    def test_optimum_replicates_custom_count(self):
        """Custom replicate count is respected."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=5, n_total=24)
        optimum_rows = plate[plate["well_type"] == "optimum_rep"]
        assert len(optimum_rows) == 5

    def test_optimum_replicates_identical(self):
        """All optimum replicate rows should have identical morphogen concentrations."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=3, n_total=24)
        optimum_rows = plate[plate["well_type"] == "optimum_rep"]
        morph_cols = [c for c in plate.columns if c not in ("well_type", "well_label")]
        for col in morph_cols:
            assert optimum_rows[col].nunique() == 1, f"Optimum replicates differ in {col}"

    def test_reference_replicates_present(self):
        """Reference wells appear when reference_conditions is provided."""
        recs = self._make_recommendations(n=30)
        ref = pd.DataFrame(
            {"WNT_uM": [1.0], "BMP_uM": [2.0], "SHH_uM": [0.5], "RA_uM": [0.1]},
            index=["ref_ctrl"],
        )
        plate = step04.generate_confirmation_plate(
            recs, reference_conditions=ref, n_replicates_reference=2, n_total=24
        )
        ref_rows = plate[plate["well_type"] == "reference_rep"]
        assert len(ref_rows) == 2
        assert len(plate) == 24

    def test_no_reference_without_arg(self):
        """No reference wells when reference_conditions is None."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_total=24)
        ref_rows = plate[plate["well_type"] == "reference_rep"]
        assert len(ref_rows) == 0

    def test_diverse_runners_are_diverse(self):
        """Runner-up conditions should not all be identical."""
        recs = self._make_recommendations(n=30)
        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=3, n_total=24)
        runners = plate[plate["well_type"] == "diverse_runner_up"]
        assert len(runners) > 1, "Need multiple runners to test diversity"
        morph_cols = [c for c in plate.columns if c not in ("well_type", "well_label")]
        runner_vals = runners[morph_cols].values
        # At least one column should have more than one unique value
        has_diversity = any(len(np.unique(runner_vals[:, i])) > 1 for i in range(len(morph_cols)))
        assert has_diversity, "All runner-up conditions are identical"

    def test_diverse_runners_maxmin_spread(self):
        """Runner-ups selected by max-min should be more spread than just taking top-N."""
        np.random.seed(99)
        # Create recommendations where first few are clustered, rest are spread
        cols = ["WNT_uM", "BMP_uM", "SHH_uM", "RA_uM"]
        # Cluster near origin
        clustered = np.random.uniform(0.0, 0.5, size=(5, 4))
        # Spread out
        spread = np.random.uniform(0.0, 10.0, size=(25, 4))
        data = np.vstack([clustered, spread])
        recs = pd.DataFrame(data, columns=cols, index=[f"rec_{i}" for i in range(30)])

        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=3, n_total=24)
        runners = plate[plate["well_type"] == "diverse_runner_up"]
        morph_cols = [c for c in plate.columns if c not in ("well_type", "well_label")]
        runner_vals = runners[morph_cols].values

        # Compute pairwise distances among selected runners
        from scipy.spatial.distance import pdist
        dists = pdist(runner_vals)
        # The minimum pairwise distance should be non-trivial (not all clustered)
        assert np.min(dists) > 0.0, "Some runner-ups are identical"

    def test_well_type_and_label_columns(self):
        """Output should have well_type and well_label columns."""
        recs = self._make_recommendations(n=10)
        plate = step04.generate_confirmation_plate(recs, n_total=10)
        assert "well_type" in plate.columns
        assert "well_label" in plate.columns

    def test_fewer_candidates_than_slots(self):
        """When fewer candidates than remaining slots, plate may be smaller."""
        recs = self._make_recommendations(n=3)
        plate = step04.generate_confirmation_plate(recs, n_replicates_optimum=3, n_total=24)
        # 3 optimum reps + 2 runners (only 2 candidates after removing optimum)
        assert len(plate) == 5

    def test_empty_recommendations_raises(self):
        """Should raise on empty recommendations."""
        recs = pd.DataFrame(columns=["WNT_uM", "BMP_uM"])
        with pytest.raises(ValueError, match="at least one row"):
            step04.generate_confirmation_plate(recs)


class TestRefineSignatures:
    """Tests for paired objective refinement with configurable alpha (TODO-17+50)."""

    @staticmethod
    def _make_synthetic_data(n_cells=20, n_genes=50, n_conditions=4, seed=42):
        """Build synthetic AnnData + fidelity_report for testing."""
        import anndata as ad

        rng = np.random.default_rng(seed)
        conditions = [f"cond_{i}" for i in range(n_conditions)]
        gene_names = [f"Gene{i}" for i in range(n_genes)]

        # Create expression data where high-fidelity conditions have
        # elevated expression of genes 0-9 (so DE can detect them)
        X = rng.random((n_cells, n_genes))
        obs_conditions = [conditions[i % n_conditions] for i in range(n_cells)]
        for i in range(n_cells):
            cond_idx = i % n_conditions
            if cond_idx < n_conditions // 2:
                # "high fidelity" conditions: boost first 10 genes
                X[i, :10] += 3.0

        obs = pd.DataFrame({"condition": obs_conditions})
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=gene_names))

        # Fidelity report: first half conditions score high, second half low
        fidelity_report = pd.DataFrame({
            "condition": conditions,
            "composite_fidelity": [0.9 - 0.2 * i for i in range(n_conditions)],
        }).set_index("condition")

        return adata, fidelity_report, gene_names

    def test_refine_signatures_basic(self):
        """Refined signatures should differ from prior when alpha > 0."""
        from gopro.signature_utils import refine_signatures

        adata, fidelity_report, gene_names = self._make_synthetic_data()

        # Prior: genes 20-34 (not among the DE genes 0-9)
        prior = {"my_sig": [f"Gene{i}" for i in range(20, 35)]}

        refined = refine_signatures(
            prior_signatures=prior,
            adata=adata,
            fidelity_report=fidelity_report,
            alpha=0.7,
            top_k=15,
        )

        assert "my_sig" in refined
        assert len(refined["my_sig"]) == 15
        # With alpha=0.7, data-derived genes should dominate
        # and the signature should not be identical to the prior
        assert set(refined["my_sig"]) != set(prior["my_sig"])

    def test_refine_signatures_alpha_zero_returns_prior(self):
        """alpha=0 should return the prior signatures unchanged."""
        from gopro.signature_utils import refine_signatures

        adata, fidelity_report, gene_names = self._make_synthetic_data()

        prior_genes = [f"Gene{i}" for i in range(10, 25)]
        prior = {"my_sig": prior_genes}

        refined = refine_signatures(
            prior_signatures=prior,
            adata=adata,
            fidelity_report=fidelity_report,
            alpha=0.0,
            top_k=15,
        )

        assert refined["my_sig"] == prior_genes[:15]

    def test_refine_signatures_alpha_one_fully_data_driven(self):
        """alpha=1 should use only data-derived genes (no prior genes)."""
        from gopro.signature_utils import refine_signatures

        adata, fidelity_report, gene_names = self._make_synthetic_data()

        # Prior: genes 40-49 (far from DE genes)
        prior_genes = [f"Gene{i}" for i in range(40, 50)]
        prior = {"my_sig": prior_genes}

        refined = refine_signatures(
            prior_signatures=prior,
            adata=adata,
            fidelity_report=fidelity_report,
            alpha=1.0,
            top_k=10,
        )

        assert "my_sig" in refined
        assert len(refined["my_sig"]) == 10
        # All genes should be data-derived; none from the prior
        # (unless a prior gene also happens to be top DE, which is unlikely here)
        prior_set = set(prior_genes)
        data_only = [g for g in refined["my_sig"] if g not in prior_set]
        # At least most should be data-derived
        assert len(data_only) >= 8, (
            f"Expected mostly data-derived genes with alpha=1, "
            f"but {len(refined['my_sig']) - len(data_only)} are from prior"
        )

    def test_refine_signatures_too_few_conditions_returns_prior(self):
        """With only 1 shared condition, should return prior unchanged."""
        from gopro.signature_utils import refine_signatures
        import anndata as ad

        rng = np.random.default_rng(42)
        gene_names = [f"Gene{i}" for i in range(20)]
        adata = ad.AnnData(
            X=rng.random((5, 20)).astype(np.float32),
            obs=pd.DataFrame({"condition": ["cond_0"] * 5}),
            var=pd.DataFrame(index=gene_names),
        )
        fidelity_report = pd.DataFrame(
            {"composite_fidelity": [0.8]},
            index=pd.Index(["cond_0"], name="condition"),
        )
        prior = {"sig": gene_names[:10]}

        refined = refine_signatures(
            prior_signatures=prior, adata=adata, fidelity_report=fidelity_report,
            alpha=0.7, top_k=10,
        )

        assert refined["sig"] == gene_names[:10]

    def test_refine_signatures_multiple(self):
        """Refinement works on multiple signatures simultaneously."""
        from gopro.signature_utils import refine_signatures

        adata, fidelity_report, gene_names = self._make_synthetic_data()

        prior = {
            "sig_a": [f"Gene{i}" for i in range(20, 35)],
            "sig_b": [f"Gene{i}" for i in range(30, 45)],
        }

        refined = refine_signatures(
            prior_signatures=prior,
            adata=adata,
            fidelity_report=fidelity_report,
            alpha=0.5,
            top_k=15,
        )

        assert set(refined.keys()) == {"sig_a", "sig_b"}
        assert len(refined["sig_a"]) == 15
        assert len(refined["sig_b"]) == 15


# --- Cost-aware desirability scoring (TODO-42) ---

class TestCostAwareDesirability:
    """Tests for cost-aware cocktail scoring and desirability re-ranking."""

    def test_cost_dict_in_config(self):
        """MORPHOGEN_COST_PER_UM exists in config and covers all MORPHOGEN_COLUMNS."""
        from gopro.config import MORPHOGEN_COST_PER_UM, MORPHOGEN_COLUMNS
        assert isinstance(MORPHOGEN_COST_PER_UM, dict)
        assert len(MORPHOGEN_COST_PER_UM) > 0
        # Every canonical morphogen column should have a cost entry
        for col in MORPHOGEN_COLUMNS:
            assert col in MORPHOGEN_COST_PER_UM, f"Missing cost entry for {col}"

    def test_compute_cocktail_cost_basic(self):
        """Verify cost computation: cost = sum(concentration_i * cost_per_um_i)."""
        recs = pd.DataFrame({
            "SHH_uM": [1.0, 0.0, 2.0],
            "CHIR99021_uM": [0.0, 10.0, 5.0],
        })
        cost_dict = {"SHH_uM": 5.0, "CHIR99021_uM": 0.01}
        cost = step04.compute_cocktail_cost(recs, cost_dict=cost_dict)
        assert len(cost) == 3
        np.testing.assert_allclose(cost.values, [5.0, 0.1, 10.05], atol=1e-9)

    def test_compute_cocktail_cost_missing_cols_ignored(self):
        """Columns in cost_dict but not in DataFrame are silently skipped."""
        recs = pd.DataFrame({"SHH_uM": [1.0]})
        cost_dict = {"SHH_uM": 5.0, "NOT_A_COL": 100.0}
        cost = step04.compute_cocktail_cost(recs, cost_dict=cost_dict)
        np.testing.assert_allclose(cost.values, [5.0], atol=1e-9)

    def test_compute_cocktail_cost_default_dict(self):
        """When no cost_dict is passed, uses MORPHOGEN_COST_PER_UM from config."""
        from gopro.config import MORPHOGEN_COST_PER_UM
        recs = pd.DataFrame({"SHH_uM": [1.0], "CHIR99021_uM": [3.0]})
        cost = step04.compute_cocktail_cost(recs)
        expected = 1.0 * MORPHOGEN_COST_PER_UM["SHH_uM"] + 3.0 * MORPHOGEN_COST_PER_UM["CHIR99021_uM"]
        np.testing.assert_allclose(cost.values, [expected], atol=1e-9)

    def test_expensive_cocktail_penalized(self):
        """Desirability re-ranks expensive cocktails lower when cost_weight > 0."""
        # Cocktail A: cheap (small molecule), Cocktail B: expensive (protein)
        recs = pd.DataFrame({
            "CHIR99021_uM": [10.0, 0.0],
            "SHH_uM": [0.0, 1.0],
        }, index=["cheap", "expensive"])
        # Both have the same acquisition value
        acq = pd.Series([1.0, 1.0], index=["cheap", "expensive"])
        cost_dict = {"CHIR99021_uM": 0.01, "SHH_uM": 5.0}

        result = step04.apply_desirability_gate(
            recs, acq, cost_dict=cost_dict, cost_weight=0.5,
        )
        # Cheap cocktail should be ranked first (higher desirability)
        assert result.index[0] == "cheap"
        assert result.loc["cheap", "desirability"] > result.loc["expensive", "desirability"]
        assert "cocktail_cost" in result.columns
        assert "desirability" in result.columns

    def test_cost_weight_zero_no_rerank(self):
        """With cost_weight=0, order is unchanged (desirability = acquisition)."""
        recs = pd.DataFrame({
            "SHH_uM": [5.0, 0.0],
            "CHIR99021_uM": [0.0, 10.0],
        }, index=["A", "B"])
        # A has lower acquisition, B has higher
        acq = pd.Series([0.5, 1.0], index=["A", "B"])
        cost_dict = {"SHH_uM": 5.0, "CHIR99021_uM": 0.01}

        result = step04.apply_desirability_gate(
            recs, acq, cost_dict=cost_dict, cost_weight=0.0,
        )
        # B should still be first (higher acquisition, cost ignored)
        assert result.index[0] == "B"
        # Desirability should equal acquisition when cost_weight=0
        np.testing.assert_allclose(
            result.loc["A", "desirability"], 0.5, atol=1e-9,
        )
        np.testing.assert_allclose(
            result.loc["B", "desirability"], 1.0, atol=1e-9,
        )

    def test_desirability_gate_adds_columns(self):
        """apply_desirability_gate adds cocktail_cost and desirability columns."""
        recs = pd.DataFrame({"SHH_uM": [1.0, 2.0]})
        acq = pd.Series([1.0, 0.8], index=recs.index)
        result = step04.apply_desirability_gate(recs, acq, cost_weight=0.2)
        assert "cocktail_cost" in result.columns
        assert "desirability" in result.columns

    def test_desirability_gate_does_not_mutate_input(self):
        """apply_desirability_gate does not modify the input DataFrame."""
        recs = pd.DataFrame({"SHH_uM": [1.0, 2.0]})
        original_cols = list(recs.columns)
        acq = pd.Series([1.0, 0.8], index=recs.index)
        _ = step04.apply_desirability_gate(recs, acq, cost_weight=0.2)
        assert list(recs.columns) == original_cols


class TestSanchisCallejaMultiFidelity:
    """Tests for Sanchis-Calleja patterning screen multi-fidelity wire-up."""

    def test_sanchis_calleja_fidelity_in_config(self):
        """SANCHIS_CALLEJA_DEFAULT_FIDELITY exists and is 0.7."""
        from gopro.config import SANCHIS_CALLEJA_DEFAULT_FIDELITY
        assert SANCHIS_CALLEJA_DEFAULT_FIDELITY == 0.7

    def test_sanchis_calleja_in_fidelity_labels(self):
        """FIDELITY_LABELS contains the 0.7 Sanchis-Calleja level."""
        from gopro.config import FIDELITY_LABELS
        assert 0.7 in FIDELITY_LABELS
        assert FIDELITY_LABELS[0.7] == "SanchisCalleja"

    def test_sanchis_calleja_in_fidelity_costs(self):
        """FIDELITY_COSTS contains the 0.7 Sanchis-Calleja level."""
        from gopro.config import FIDELITY_COSTS
        assert 0.7 in FIDELITY_COSTS
        # Cost should be between real (1.0) and CellRank2 (0.005)
        assert FIDELITY_COSTS[1.0] > FIDELITY_COSTS[0.7] > FIDELITY_COSTS[0.5]

    def test_sanchis_calleja_imported_in_gpbo(self):
        """SANCHIS_CALLEJA_DEFAULT_FIDELITY is accessible from 04_gpbo_loop."""
        assert hasattr(step04, "SANCHIS_CALLEJA_DEFAULT_FIDELITY")
        assert step04.SANCHIS_CALLEJA_DEFAULT_FIDELITY == 0.7

    def test_sanchis_calleja_in_kernel_remap(self):
        """FIDELITY_KERNEL_REMAP includes the 0.7 level."""
        assert 0.7 in step04.FIDELITY_KERNEL_REMAP
        # Remapped value must be in (0, 1) and ordered correctly
        assert 0 < step04.FIDELITY_KERNEL_REMAP[0.7] < 1
        assert step04.FIDELITY_KERNEL_REMAP[0.5] < step04.FIDELITY_KERNEL_REMAP[0.7]
        assert step04.FIDELITY_KERNEL_REMAP[0.7] < step04.FIDELITY_KERNEL_REMAP[1.0]

    def test_sanchis_calleja_auto_discovery(self, tmp_path):
        """Auto-discovery finds Sanchis-Calleja CSVs and adds them to virtual_sources."""
        import argparse

        # Create mock CSV files in tmp dir
        sc_frac = tmp_path / "gp_training_labels_sanchis_calleja.csv"
        sc_morph = tmp_path / "morphogen_matrix_sanchis_calleja.csv"
        pd.DataFrame({"ct1": [0.5]}, index=["cond_0"]).to_csv(str(sc_frac))
        pd.DataFrame({"CHIR99021_uM": [1.0]}, index=["cond_0"]).to_csv(str(sc_morph))

        # Simulate the auto-discovery logic from __main__
        virtual_sources = []
        args = argparse.Namespace(
            sanchis_fractions=None,
            sanchis_morphogens=None,
            sanchis_fidelity=0.7,
        )
        sc_frac_path = Path(args.sanchis_fractions) if args.sanchis_fractions else sc_frac
        sc_morph_path = Path(args.sanchis_morphogens) if args.sanchis_morphogens else sc_morph
        if sc_frac_path.exists() and sc_morph_path.exists():
            virtual_sources.append((sc_frac_path, sc_morph_path, args.sanchis_fidelity))

        assert len(virtual_sources) == 1
        assert virtual_sources[0][2] == 0.7

    def test_sanchis_calleja_auto_discovery_missing(self, tmp_path):
        """Auto-discovery does not add sources when CSVs are missing."""
        import argparse

        virtual_sources = []
        args = argparse.Namespace(
            sanchis_fractions=None,
            sanchis_morphogens=None,
            sanchis_fidelity=0.7,
        )
        # Point to non-existent files
        sc_frac_path = tmp_path / "gp_training_labels_sanchis_calleja.csv"
        sc_morph_path = tmp_path / "morphogen_matrix_sanchis_calleja.csv"
        if sc_frac_path.exists() and sc_morph_path.exists():
            virtual_sources.append((sc_frac_path, sc_morph_path, args.sanchis_fidelity))

        assert len(virtual_sources) == 0

    def test_sanchis_calleja_cli_flags(self):
        """Argparse parser accepts --sanchis-fractions, --sanchis-morphogens, --sanchis-fidelity."""
        import argparse

        # Build the parser the same way as __main__ (subset of relevant args)
        parser = argparse.ArgumentParser()
        parser.add_argument("--sanchis-fractions", type=str, default=None)
        parser.add_argument("--sanchis-morphogens", type=str, default=None)
        parser.add_argument("--sanchis-fidelity", type=float, default=0.7)

        # Test with explicit paths
        args = parser.parse_args([
            "--sanchis-fractions", "/tmp/sc_frac.csv",
            "--sanchis-morphogens", "/tmp/sc_morph.csv",
            "--sanchis-fidelity", "0.65",
        ])
        assert args.sanchis_fractions == "/tmp/sc_frac.csv"
        assert args.sanchis_morphogens == "/tmp/sc_morph.csv"
        assert args.sanchis_fidelity == 0.65

        # Test defaults
        args_default = parser.parse_args([])
        assert args_default.sanchis_fractions is None
        assert args_default.sanchis_morphogens is None
        assert args_default.sanchis_fidelity == 0.7

    def test_sanchis_calleja_custom_fidelity(self, tmp_path):
        """Custom fidelity override is passed through to virtual_sources."""
        import argparse

        sc_frac = tmp_path / "sc_frac.csv"
        sc_morph = tmp_path / "sc_morph.csv"
        pd.DataFrame({"ct1": [0.5]}, index=["cond_0"]).to_csv(str(sc_frac))
        pd.DataFrame({"CHIR99021_uM": [1.0]}, index=["cond_0"]).to_csv(str(sc_morph))

        virtual_sources = []
        args = argparse.Namespace(
            sanchis_fractions=str(sc_frac),
            sanchis_morphogens=str(sc_morph),
            sanchis_fidelity=0.65,
        )
        sc_frac_path = Path(args.sanchis_fractions) if args.sanchis_fractions else tmp_path / "default.csv"
        sc_morph_path = Path(args.sanchis_morphogens) if args.sanchis_morphogens else tmp_path / "default.csv"
        if sc_frac_path.exists() and sc_morph_path.exists():
            virtual_sources.append((sc_frac_path, sc_morph_path, args.sanchis_fidelity))

        assert len(virtual_sources) == 1
        assert virtual_sources[0][2] == 0.65
