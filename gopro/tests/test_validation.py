"""Tests for inter-step data validation (Track 1A)."""

import numpy as np
import pandas as pd
import pytest

from gopro.validation import (
    ValidationError,
    validate_mapped_adata,
    validate_training_csvs,
    validate_temporal_atlas,
    validate_fidelity_report,
)


class TestValidateMappedAdata:
    """Tests for validate_mapped_adata."""

    @pytest.fixture
    def valid_h5ad(self, tmp_path):
        """Create a valid mapped h5ad file."""
        import anndata as ad

        obs = pd.DataFrame({
            "condition": ["A", "B", "C"],
            "predicted_annot_level_1": ["Neuron", "NPC", "Glia"],
            "predicted_annot_level_2": ["Cortical EN", "Cortical RG", "OPC"],
            "predicted_annot_region_rev2": ["Dorsal telencephalon"] * 3,
        })
        adata = ad.AnnData(
            X=np.random.rand(3, 5).astype(np.float32),
            obs=obs,
        )
        path = tmp_path / "valid_mapped.h5ad"
        adata.write(str(path))
        return path

    def test_valid_mapped_adata(self, valid_h5ad):
        warnings = validate_mapped_adata(valid_h5ad)
        assert isinstance(warnings, list)

    def test_missing_predicted_columns_raises(self, tmp_path):
        import anndata as ad

        obs = pd.DataFrame({"condition": ["A", "B"]})
        adata = ad.AnnData(
            X=np.random.rand(2, 5).astype(np.float32),
            obs=obs,
        )
        path = tmp_path / "bad_mapped.h5ad"
        adata.write(str(path))

        with pytest.raises(ValidationError, match="Missing predicted label columns"):
            validate_mapped_adata(path)

    def test_missing_condition_key_raises(self, tmp_path):
        import anndata as ad

        obs = pd.DataFrame({
            "predicted_annot_level_1": ["Neuron"],
            "predicted_annot_level_2": ["Cortical EN"],
            "predicted_annot_region_rev2": ["Dorsal telencephalon"],
        })
        adata = ad.AnnData(
            X=np.random.rand(1, 5).astype(np.float32),
            obs=obs,
        )
        path = tmp_path / "no_condition.h5ad"
        adata.write(str(path))

        with pytest.raises(ValidationError, match="Condition column"):
            validate_mapped_adata(path)

    def test_warns_missing_counts_layer(self, valid_h5ad):
        warnings = validate_mapped_adata(valid_h5ad, require_counts_layer=False)
        assert any("counts" in w for w in warnings)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_mapped_adata(tmp_path / "nonexistent.h5ad")


class TestValidateTrainingCsvs:
    """Tests for validate_training_csvs."""

    @pytest.fixture
    def valid_csvs(self, tmp_path):
        """Create valid fractions + morphogen CSV pair."""
        fractions = pd.DataFrame({
            "Neuron": [0.6, 0.3],
            "NPC": [0.4, 0.7],
        }, index=["cond_A", "cond_B"])
        morphogens = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "SAG_uM": [0.0, 0.25],
        }, index=["cond_A", "cond_B"])

        frac_path = tmp_path / "fractions.csv"
        morph_path = tmp_path / "morphogens.csv"
        fractions.to_csv(frac_path)
        morphogens.to_csv(morph_path)
        return frac_path, morph_path

    def test_valid_training_csvs(self, valid_csvs):
        frac_path, morph_path = valid_csvs
        warnings = validate_training_csvs(frac_path, morph_path)
        assert isinstance(warnings, list)

    def test_zero_overlap_raises(self, tmp_path):
        fractions = pd.DataFrame({"Neuron": [0.5]}, index=["cond_X"])
        morphogens = pd.DataFrame({"CHIR99021_uM": [1.5]}, index=["cond_Y"])

        frac_path = tmp_path / "frac.csv"
        morph_path = tmp_path / "morph.csv"
        fractions.to_csv(frac_path)
        morphogens.to_csv(morph_path)

        with pytest.raises(ValidationError, match="Zero overlap"):
            validate_training_csvs(frac_path, morph_path)

    def test_fraction_nan_raises(self, tmp_path):
        fractions = pd.DataFrame({"Neuron": [0.5, np.nan]}, index=["A", "B"])
        morphogens = pd.DataFrame({"CHIR99021_uM": [1.5, 3.0]}, index=["A", "B"])

        frac_path = tmp_path / "frac.csv"
        morph_path = tmp_path / "morph.csv"
        fractions.to_csv(frac_path)
        morphogens.to_csv(morph_path)

        with pytest.raises(ValidationError, match="NaN"):
            validate_training_csvs(frac_path, morph_path)

    def test_fraction_row_sum_violation(self, tmp_path):
        fractions = pd.DataFrame({
            "Neuron": [0.5, 0.1],
            "NPC": [0.5, 0.1],
        }, index=["A", "B"])  # B sums to 0.2
        morphogens = pd.DataFrame({"CHIR99021_uM": [1.5, 3.0]}, index=["A", "B"])

        frac_path = tmp_path / "frac.csv"
        morph_path = tmp_path / "morph.csv"
        fractions.to_csv(frac_path)
        morphogens.to_csv(morph_path)

        with pytest.raises(ValidationError, match="don't sum to"):
            validate_training_csvs(frac_path, morph_path)

    def test_unknown_morphogen_columns_warns(self, tmp_path):
        fractions = pd.DataFrame({"Neuron": [1.0]}, index=["A"])
        morphogens = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "MYSTERY_DRUG": [99.0],
        }, index=["A"])

        frac_path = tmp_path / "frac.csv"
        morph_path = tmp_path / "morph.csv"
        fractions.to_csv(frac_path)
        morphogens.to_csv(morph_path)

        warnings = validate_training_csvs(frac_path, morph_path)
        assert any("Unrecognized" in w for w in warnings)


class TestValidateFidelityReport:
    """Tests for validate_fidelity_report."""

    @pytest.fixture
    def valid_report(self, tmp_path):
        """Create a valid fidelity report CSV."""
        report = pd.DataFrame({
            "composite_fidelity": [0.85, 0.72],
            "rss_score": [0.9, 0.7],
            "dominant_region": ["Dorsal telencephalon", "Ventral telencephalon"],
            "maturity_score": [0.8, 0.6],
        }, index=["cond_A", "cond_B"])
        path = tmp_path / "fidelity_report.csv"
        report.to_csv(path)
        return path

    def test_valid_report_passes(self, valid_report):
        warnings = validate_fidelity_report(valid_report)
        assert isinstance(warnings, list)
        # No maturity warning since column is present
        assert not any("maturity_score" in w for w in warnings)

    def test_missing_composite_fidelity_raises(self, tmp_path):
        report = pd.DataFrame({
            "rss_score": [0.9],
            "dominant_region": ["Dorsal"],
        }, index=["cond_A"])
        path = tmp_path / "bad_report.csv"
        report.to_csv(path)
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_fidelity_report(path)

    def test_out_of_range_score_warns(self, tmp_path):
        report = pd.DataFrame({
            "composite_fidelity": [1.5],  # out of range
            "rss_score": [0.9],
            "dominant_region": ["Dorsal"],
            "maturity_score": [0.8],
        }, index=["cond_A"])
        path = tmp_path / "range_report.csv"
        report.to_csv(path)
        warnings = validate_fidelity_report(path)
        assert any("outside [0, 1]" in w for w in warnings)

    def test_missing_maturity_warns(self, tmp_path):
        report = pd.DataFrame({
            "composite_fidelity": [0.85],
            "rss_score": [0.9],
            "dominant_region": ["Dorsal"],
        }, index=["cond_A"])
        path = tmp_path / "no_maturity.csv"
        report.to_csv(path)
        warnings = validate_fidelity_report(path)
        assert any("maturity_score" in w for w in warnings)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_fidelity_report(tmp_path / "nonexistent.csv")
