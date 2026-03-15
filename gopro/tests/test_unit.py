"""Unit tests for GP-BO pipeline components."""

import pytest
import numpy as np
import pandas as pd
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

    def test_cosine_similarity_identical(self):
        """Identical fractions should have cosine similarity = 1.0."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG250"])
        fracs_b = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG_250nM"])
        mapping = {"SAG250": "SAG_250nM"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["SAG250"]["cosine_similarity"] == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal fractions should have cosine similarity = 0.0."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [1.0], "Neuron": [0.0]}, index=["cond_a"])
        fracs_b = pd.DataFrame({"NPC": [0.0], "Neuron": [1.0]}, index=["cond_b"])
        mapping = {"cond_a": "cond_b"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["cond_a"]["cosine_similarity"] == pytest.approx(0.0)

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
