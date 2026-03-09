"""Integration tests for the GP-BO pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from conftest import _import_pipeline_module

step01 = _import_pipeline_module("01_load_and_convert_data")
step02 = _import_pipeline_module("02_map_to_hnoca")
step03 = _import_pipeline_module("03_fidelity_scoring")
step04 = _import_pipeline_module("04_gpbo_loop")
morphogen_parser = _import_pipeline_module("morphogen_parser")


class TestMorphogenParserIntegration:
    """Test morphogen parser with all 46 real conditions."""

    def test_all_conditions_parse(self):
        conditions = [
            "BMP4 CHIR", "BMP4 CHIR d11-16", "BMP4 SAG", "BMP7", "BMP7 CHIR",
            "BMP7 SAG", "C/L/S/FGF8", "C/S/BMP7/D", "C/S/D/FGF4", "C/S/R/E/FGF2/D",
            "CHIR SAG FGF4", "CHIR SAG FGF8", "CHIR switch IWP2", "CHIR-SAG-LDN",
            "CHIR-SAG-d16-21", "CHIR-SAGd10-21", "CHIR-d11-16", "CHIR-d16-21",
            "CHIR-d6-11", "CHIR1.5", "CHIR1.5-SAG1000", "CHIR1.5-SAG250",
            "CHIR3", "CHIR3-SAG1000", "CHIR3-SAG250", "DAPT", "FGF-20/EGF",
            "FGF2-20", "FGF2-50", "FGF4", "FGF8", "I/Activin/DAPT/SR11",
            "IWP2", "IWP2 switch CHIR", "IWP2-SAG", "LDN", "RA10", "RA100",
            "S/I/E/FGF2", "SAG-CHIR-d16-21", "SAG-CHIRd10-21", "SAG-d11-16",
            "SAG-d16-21", "SAG-d6-11", "SAG1000", "SAG250",
        ]
        matrix = morphogen_parser.build_morphogen_matrix(conditions)
        assert matrix.shape == (46, 24)
        assert not matrix.isnull().any().any()

    def test_all_conditions_have_harvest_day(self):
        conditions = ["CHIR1.5", "SAG250", "BMP4 CHIR"]
        matrix = morphogen_parser.build_morphogen_matrix(conditions)
        assert (matrix["log_harvest_day"] > 0).all()

    def test_sag_concentrations(self):
        conditions = ["SAG250", "SAG1000"]
        matrix = morphogen_parser.build_morphogen_matrix(conditions)
        assert matrix.loc["SAG250", "SAG_uM"] == pytest.approx(0.25)
        assert matrix.loc["SAG1000", "SAG_uM"] == pytest.approx(1.0)

    def test_chir_concentrations(self):
        conditions = ["CHIR1.5", "CHIR3"]
        matrix = morphogen_parser.build_morphogen_matrix(conditions)
        assert matrix.loc["CHIR1.5", "CHIR99021_uM"] == 1.5
        assert matrix.loc["CHIR3", "CHIR99021_uM"] == 3.0

    def test_morphogen_columns_match_step04(self):
        """Verify morphogen parser columns align with step 04."""
        parser_cols = set(morphogen_parser.MORPHOGEN_COLUMNS)
        step04_cols = set(step04.MORPHOGEN_COLUMNS)
        assert parser_cols == step04_cols, (
            f"Mismatch: parser has {parser_cols - step04_cols}, "
            f"step04 has {step04_cols - parser_cols}"
        )


class TestGPBOIntegration:
    """Test the GP-BO pipeline end-to-end with synthetic data."""

    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic training data."""
        np.random.seed(42)
        n_conditions = 20
        n_cell_types = 5

        conditions = [f"cond_{i}" for i in range(n_conditions)]

        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(n_cell_types), size=n_conditions),
            columns=[f"ct_{i}" for i in range(n_cell_types)],
            index=conditions,
        )

        X = pd.DataFrame(
            np.random.rand(n_conditions, 4) * np.array([3, 0.004, 1.0, 4.5]),
            columns=["CHIR99021_uM", "BMP4_uM", "SAG_uM", "log_harvest_day"],
            index=conditions,
        )

        y_path = tmp_path / "Y.csv"
        x_path = tmp_path / "X.csv"
        Y.to_csv(y_path)
        X.to_csv(x_path)
        return x_path, y_path

    def test_build_and_fit(self, synthetic_data):
        """Test that we can build training set and fit GP."""
        x_path, y_path = synthetic_data
        X, Y = step04.build_training_set(y_path, x_path)

        model, train_X, train_Y, cols = step04.fit_gp_botorch(
            X, Y, use_ilr=True
        )
        assert train_X.shape[0] == 20
        assert train_Y.shape[1] == 4  # 5 cell types - 1 (ILR)

    def test_full_loop(self, synthetic_data):
        """Test the full GP-BO loop with synthetic data."""
        x_path, y_path = synthetic_data
        recs = step04.run_gpbo_loop(
            fractions_csv=y_path,
            morphogen_csv=x_path,
            n_recommendations=6,
            round_num=1,
            use_ilr=True,
        )
        assert len(recs) == 6
        assert "acquisition_value" in recs.columns

    def test_recommendations_within_bounds(self, synthetic_data):
        """Test that recommendations are within specified bounds."""
        x_path, y_path = synthetic_data

        # Read X to get column list
        X = pd.read_csv(str(x_path), index_col=0)

        recs = step04.run_gpbo_loop(
            fractions_csv=y_path,
            morphogen_csv=x_path,
            n_recommendations=6,
            round_num=1,
        )
        # Check bounds: use dynamic bounds (training range + padding)
        active_bounds, _ = step04._compute_active_bounds(X, list(X.columns))
        for col in recs.columns:
            if col in active_bounds:
                lo, hi = active_bounds[col]
                assert recs[col].min() >= lo - 0.01, f"{col} below lower bound"
                assert recs[col].max() <= hi + 0.01, f"{col} above upper bound"

    def test_predictions_are_finite(self, synthetic_data):
        """Test that GP predictions are all finite."""
        x_path, y_path = synthetic_data
        recs = step04.run_gpbo_loop(
            fractions_csv=y_path,
            morphogen_csv=x_path,
            n_recommendations=6,
            round_num=1,
        )
        pred_cols = [c for c in recs.columns if "predicted" in c]
        for col in pred_cols:
            assert np.all(np.isfinite(recs[col])), f"Non-finite values in {col}"

    def test_uncertainty_non_negative(self, synthetic_data):
        """Test that GP uncertainty (std) is non-negative."""
        x_path, y_path = synthetic_data
        recs = step04.run_gpbo_loop(
            fractions_csv=y_path,
            morphogen_csv=x_path,
            n_recommendations=6,
            round_num=1,
        )
        std_cols = [c for c in recs.columns if "_std" in c]
        for col in std_cols:
            assert (recs[col] >= 0).all(), f"Negative uncertainty in {col}"


class TestCellTypeFractionsIntegration:
    """Test cell type fraction computation with realistic data."""

    def test_multiple_conditions(self):
        np.random.seed(42)
        types = ["Neuron", "NPC", "Glia", "CP", "MC"]
        conditions = ["BMP4 CHIR", "SAG250", "CHIR1.5"]

        obs = pd.DataFrame({
            "condition": np.random.choice(conditions, size=500),
            "predicted_annot_level_2": np.random.choice(types, size=500),
            "quality": "keep",
        })

        fracs = step02.compute_cell_type_fractions(
            obs, "condition", "predicted_annot_level_2"
        )

        # Check shape
        assert fracs.shape[0] == 3
        assert fracs.shape[1] == 5

        # Check fractions sum to 1
        np.testing.assert_allclose(fracs.sum(axis=1), 1.0, atol=1e-6)

        # Check all values in [0, 1]
        assert (fracs >= 0).all().all()
        assert (fracs <= 1).all().all()


class TestFidelityScoringIntegration:
    """Test fidelity scoring pipeline with synthetic data."""

    def test_score_all_conditions(self):
        """Test full condition scoring with synthetic mapped data."""
        import scanpy as sc

        np.random.seed(42)
        n_cells = 300
        conditions = ["BMP4 CHIR", "SAG250", "CHIR3"]
        level1_types = ["Neuron", "NPC", "IP", "PSC", "MC"]
        level2_types = ["Dorsal Telencephalic Neuron", "NPC", "Glioblast", "IP", "OPC"]
        regions = ["Dorsal telencephalon", "Ventral telencephalon", "Thalamus"]

        obs = pd.DataFrame({
            "condition": np.random.choice(conditions, size=n_cells),
            "predicted_annot_level_1": np.random.choice(level1_types, size=n_cells),
            "predicted_annot_level_2": np.random.choice(level2_types, size=n_cells),
            "predicted_annot_region_rev2": np.random.choice(regions, size=n_cells),
        })

        adata = sc.AnnData(X=np.zeros((n_cells, 10)), obs=obs)

        # Create synthetic Braun profiles
        braun_profiles = pd.DataFrame({
            "Neuron": [0.4, 0.3, 0.5],
            "Radial glia": [0.3, 0.4, 0.2],
            "Neuronal IPC": [0.2, 0.2, 0.2],
            "Oligo": [0.1, 0.1, 0.1],
        }, index=["Dorsal telencephalon", "Ventral telencephalon", "Thalamus"])

        report = step03.score_all_conditions(adata, braun_profiles, "condition")

        assert len(report) == 3
        assert "composite_fidelity" in report.columns
        assert "rss_score" in report.columns
        assert (report["composite_fidelity"] >= 0).all()
        assert (report["composite_fidelity"] <= 1).all()

    def test_assign_cell_level_fidelity(self):
        """Test cell-level fidelity assignment."""
        import scanpy as sc

        obs = pd.DataFrame({
            "condition": ["A", "A", "B", "B"],
            "predicted_annot_level_1": ["Neuron", "PSC", "NPC", "Neuron"],
        })
        adata = sc.AnnData(X=np.zeros((4, 2)), obs=obs)

        report = pd.DataFrame({
            "composite_fidelity": [0.8, 0.5],
            "rss_score": [0.9, 0.6],
            "on_target_fraction": [0.7, 0.4],
        }, index=["A", "B"])

        result = step03.assign_cell_level_fidelity(adata, report, "condition")
        assert "fidelity_score" in result.obs.columns
        assert "is_off_target" in result.obs.columns
        assert result.obs.loc[result.obs["condition"] == "A", "fidelity_score"].iloc[0] == 0.8
        assert result.obs["is_off_target"].iloc[1] == True  # PSC is off-target


class TestVisualizationIntegration:
    """Integration tests for visualization report."""

    @pytest.fixture
    def report_data(self, tmp_path):
        """Create a complete set of mock data files for report generation."""
        np.random.seed(42)
        n_cond = 10
        conditions = [f"cond_{i}" for i in range(n_cond)]

        # fidelity_report.csv
        fidelity = pd.DataFrame({
            "composite_fidelity": np.random.uniform(0.5, 0.95, n_cond),
            "rss_score": np.random.uniform(0.5, 0.9, n_cond),
            "on_target_fraction": np.random.uniform(0.6, 1.0, n_cond),
            "off_target_fraction": np.random.uniform(0.0, 0.15, n_cond),
            "dominant_region": ["Cortex"] * n_cond,
        }, index=pd.Index(conditions, name="condition"))
        fidelity.to_csv(tmp_path / "fidelity_report.csv")

        # morphogen_matrix_amin_kelley.csv
        morph_cols = ["CHIR99021_uM", "BMP4_uM", "SAG_uM", "log_harvest_day"]
        morphogens = pd.DataFrame(
            np.random.rand(n_cond, 4) * [3, 0.004, 1.0, 4.5],
            columns=morph_cols, index=conditions,
        )
        morphogens.to_csv(tmp_path / "morphogen_matrix_amin_kelley.csv")

        # gp_recommendations_round1.csv
        wells = [f"A{i+1}" for i in range(6)]
        recs = pd.DataFrame(
            np.random.rand(6, 4) * [3, 0.004, 1.0, 4.5],
            columns=morph_cols, index=pd.Index(wells, name="well"),
        )
        recs["fidelity"] = 1.0
        recs["predicted_y0_mean"] = np.random.randn(6)
        recs["predicted_y0_std"] = np.abs(np.random.randn(6))
        recs["acquisition_value"] = np.random.randn(6)
        recs.to_csv(tmp_path / "gp_recommendations_round1.csv")

        # gp_diagnostics_round1.csv (no lengthscales)
        diag = pd.DataFrame([{"round": 1, "n_training_points": n_cond,
                              "n_morphogens": 4, "n_cell_types": 5,
                              "target_cell_types": "all"}])
        diag.to_csv(tmp_path / "gp_diagnostics_round1.csv", index=False)

        # gp_training_labels_amin_kelley.csv
        ct_cols = ["Neuron", "NPC", "Glia", "CP", "MC"]
        labels = pd.DataFrame(
            np.random.dirichlet(np.ones(5), size=n_cond),
            columns=ct_cols, index=pd.Index(conditions, name="condition"),
        )
        labels.to_csv(tmp_path / "gp_training_labels_amin_kelley.csv")

        # gp_training_regions_amin_kelley.csv
        region_cols = ["Cortex", "Thalamus", "Midbrain"]
        regions = pd.DataFrame(
            np.random.dirichlet(np.ones(3), size=n_cond),
            columns=region_cols, index=pd.Index(conditions, name="condition"),
        )
        regions.to_csv(tmp_path / "gp_training_regions_amin_kelley.csv")

        return tmp_path

    def test_generate_report(self, report_data):
        """Test full report generation with mock data."""
        from gopro.visualize_report import generate_report
        report_path = generate_report(report_data)
        assert report_path.exists()
        html = report_path.read_text()
        assert "GP-BO Visualization Report" in html
        assert "plotly" in html.lower()
        assert "Summary" in html
        assert "Leaderboard" in html

    def test_assemble_html_report(self, tmp_path):
        """Test HTML assembly with dummy figures."""
        from gopro.visualize_report import assemble_html_report, _placeholder_figure
        sections = {
            "Test Summary": ("Summary description", "This is a test summary."),
            "Test Chart": ("Chart description", _placeholder_figure("test placeholder")),
        }
        out = assemble_html_report(sections, tmp_path / "test.html")
        assert out.exists()
        html = out.read_text()
        assert "Test Summary" in html
        assert "test summary" in html
        assert "Test Chart" in html


class TestRealDataExists:
    """Tests that real data files are accessible (smoke tests)."""

    from gopro.config import DATA_DIR

    def test_amin_kelley_converted(self):
        path = self.DATA_DIR / "amin_kelley_2024.h5ad"
        assert path.exists(), "Run step 01 first"

    def test_morphogen_matrix_exists(self):
        path = self.DATA_DIR / "morphogen_matrix_amin_kelley.csv"
        assert path.exists(), "Run morphogen parser first"

    def test_hnoca_reference_exists(self):
        path = self.DATA_DIR / "hnoca_minimal_for_mapping.h5ad"
        assert path.exists()

    def test_braun_reference_exists(self):
        path = self.DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
        assert path.exists()

    def test_scpoli_model_exists(self):
        model_dir = self.DATA_DIR / "neural_organoid_atlas/supplemental_files/scpoli_model_params"
        assert model_dir.exists()
        assert (model_dir / "model_params.pt").exists()
