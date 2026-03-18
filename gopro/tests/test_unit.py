"""Unit tests for GP-BO pipeline components."""

import pytest
import numpy as np
import pandas as pd
import torch
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

        # RSS scores should be higher with label alignment
        assert report_with["rss_score"].mean() > report_without["rss_score"].mean()


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
        """MORPHOGEN_COLUMNS has exactly 24 entries."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert len(MORPHOGEN_COLUMNS) == 24

    def test_morphogen_columns_canonical_order(self):
        """Indices 16-18 are Dorsomorphin, purmorphamine, cyclopamine."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert MORPHOGEN_COLUMNS[16] == "Dorsomorphin_uM"
        assert MORPHOGEN_COLUMNS[17] == "purmorphamine_uM"
        assert MORPHOGEN_COLUMNS[18] == "cyclopamine_uM"

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
        assert df.shape == (2, 24)
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
            X, Y = step04.merge_multi_fidelity_data(sources)

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
        assert df.shape == (46, 24)

    def test_combined_parser(self):
        """CombinedParser merges multiple parsers."""
        from gopro.morphogen_parser import AminKelleyParser, SAGSecondaryParser, CombinedParser
        combined = CombinedParser([AminKelleyParser(), SAGSecondaryParser()])
        assert len(combined.conditions) == 48
        df = combined.build_matrix()
        assert df.shape == (48, 24)


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
        expected = torch.tensor([1 / 3, 1 / 2, 2 / 3], dtype=torch.float64)
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
        assert len(step04.FIDELITY_KERNEL_REMAP) == 3
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
        # Should contain remapped values ≈ 1/2 and 2/3
        unique_fids = sorted(fid_col.unique().tolist())
        assert len(unique_fids) == 2
        torch.testing.assert_close(
            torch.tensor(unique_fids, dtype=torch.float64),
            torch.tensor([1 / 2, 2 / 3], dtype=torch.float64),
        )


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
    """Tests for additive + interaction kernel (NAIAD 2025, Idea #8)."""

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
    """Tests for adaptive kernel complexity selection (NAIAD 2025, Idea #9)."""

    def test_sparse_data_selects_shared(self):
        """N/d < 8 should select shared lengthscale kernel."""
        result = step04._select_kernel_complexity(n_conditions=20, d_active=5)
        # 20/5 = 4.0 < 8 → shared
        assert result["kernel_type"] == "shared"
        assert result["use_saasbo"] is False
        assert result["n_d_ratio"] == pytest.approx(4.0)

    def test_moderate_data_selects_ard(self):
        """8 ≤ N/d < 15 should select per-dim ARD kernel."""
        result = step04._select_kernel_complexity(n_conditions=50, d_active=5)
        # 50/5 = 10.0, in [8, 15) → ARD
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
        result = step04._select_kernel_complexity(n_conditions=40, d_active=5)
        # 40/5 = 8.0, right at threshold → ARD (not shared)
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
        # d_safe = max(0, 1) = 1; ratio = 10.0, in [8, 15) → ARD
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
