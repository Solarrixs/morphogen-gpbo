"""Tests for scGPT brain checkpoint integration."""

import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from gopro.scgpt_integration import (
    VocabDict,
    load_scgpt_brain,
    embed_cells,
    add_scgpt_embeddings,
    validate_annotations_scgpt,
    compute_annotation_confidence,
    SCGPT_BRAIN_DIR,
)


# ---------------------------------------------------------------------------
# VocabDict tests
# ---------------------------------------------------------------------------

class TestVocabDict:
    def test_from_dict(self):
        v = VocabDict({"<pad>": 0, "<cls>": 1, "TP53": 2, "BRCA1": 3})
        assert len(v) == 4
        assert v["TP53"] == 2
        assert "BRCA1" in v
        assert "NOTHERE" not in v

    def test_getitem_raises_on_missing(self):
        v = VocabDict({"a": 0})
        with pytest.raises(KeyError):
            v["missing"]

    def test_default_index(self):
        v = VocabDict({"<pad>": 0, "GENE1": 1})
        v.set_default_index(0)
        assert v["unknown_gene"] == 0

    def test_call_batch_lookup(self):
        v = VocabDict({"A": 0, "B": 1, "C": 2})
        assert v(["A", "C", "B"]) == [0, 2, 1]

    def test_from_file(self, tmp_path):
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(json.dumps({"<pad>": 0, "<cls>": 1, "TP53": 2}))
        v = VocabDict.from_file(vocab_path)
        assert len(v) == 3
        assert v["TP53"] == 2

    def test_get_stoi(self):
        v = VocabDict({"A": 0, "B": 1})
        stoi = v.get_stoi()
        assert stoi == {"A": 0, "B": 1}

    def test_get_itos(self):
        v = VocabDict({"A": 0, "B": 1})
        itos = v.get_itos()
        assert itos[0] == "A"
        assert itos[1] == "B"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_vocab():
    """Vocab with a few genes + special tokens."""
    tokens = {"<pad>": 0, "<cls>": 1, "<eoc>": 2}
    # Add some real gene names
    for i, gene in enumerate(["TP53", "BRCA1", "MYC", "EGFR", "KRAS"], start=3):
        tokens[gene] = i
    v = VocabDict(tokens)
    v.set_default_index(0)
    return v


@pytest.fixture
def mock_adata():
    """Create a minimal AnnData for testing."""
    import anndata as ad
    n_cells = 50
    n_genes = 5
    gene_names = ["TP53", "BRCA1", "MYC", "EGFR", "KRAS"]
    np.random.seed(42)
    X = np.random.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        var={"gene_symbol": gene_names},
    )
    adata.var_names = gene_names
    # Add cell type labels for validation tests
    cell_types = np.random.choice(["neuron", "astrocyte", "OPC"], size=n_cells)
    adata.obs["cell_type_predicted"] = cell_types
    return adata


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing scGPT checkpoint"):
            load_scgpt_brain(model_dir=tmp_path)

    @pytest.mark.skipif(
        not (SCGPT_BRAIN_DIR / "best_model.pt").exists(),
        reason="scGPT brain checkpoint not downloaded",
    )
    def test_load_real_checkpoint(self):
        model, vocab, configs = load_scgpt_brain(device="cpu")
        assert model is not None
        assert len(vocab) > 60000
        assert configs["embsize"] == 512
        assert configs["nlayers"] == 12
        # Model should be in eval mode
        assert not model.training


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbedCells:
    @pytest.mark.skipif(
        not (SCGPT_BRAIN_DIR / "best_model.pt").exists(),
        reason="scGPT brain checkpoint not downloaded",
    )
    def test_embed_real_model(self, mock_adata):
        """Test embedding with the real scGPT brain model."""
        embeddings = embed_cells(
            mock_adata,
            gene_col="index",
            batch_size=16,
            device="cpu",
        )
        assert embeddings.shape == (50, 512)
        # Check L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.skipif(
        not (SCGPT_BRAIN_DIR / "best_model.pt").exists(),
        reason="scGPT brain checkpoint not downloaded",
    )
    def test_add_scgpt_embeddings(self, mock_adata):
        """Test adding embeddings to adata.obsm."""
        result = add_scgpt_embeddings(
            mock_adata,
            gene_col="index",
            batch_size=16,
            device="cpu",
        )
        assert "X_scGPT" in result.obsm
        assert result.obsm["X_scGPT"].shape[1] == 512

    def test_low_overlap_warning(self, mock_adata, small_vocab):
        """Test that low gene overlap produces a warning."""
        import warnings as _warnings

        # Rename genes so none match
        mock_adata.var_names = [f"FAKE_{i}" for i in range(5)]

        mock_model = MagicMock()
        mock_configs = {"pad_token": "<pad>", "pad_value": -2, "embsize": 8}

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            try:
                embed_cells(
                    mock_adata,
                    model=mock_model,
                    vocab=small_vocab,
                    model_configs=mock_configs,
                    gene_col="index",
                )
            except Exception:
                pass  # May fail on actual forward pass, that's ok
            warning_messages = [str(x.message) for x in w]
            assert any("Very low gene overlap" in msg for msg in warning_messages), (
                f"Expected 'Very low gene overlap' warning, got: {warning_messages}"
            )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateAnnotations:
    @pytest.mark.skipif(
        not (SCGPT_BRAIN_DIR / "best_model.pt").exists(),
        reason="scGPT brain checkpoint not downloaded",
    )
    def test_validate_real(self, mock_adata):
        """End-to-end validation with real model."""
        result = validate_annotations_scgpt(
            mock_adata,
            label_col="cell_type_predicted",
            batch_size=16,
            device="cpu",
        )
        assert "ari" in result
        assert "per_type_purity" in result
        assert "n_scgpt_clusters" in result
        assert "validated" in result
        assert "flagged_types" in result
        assert isinstance(result["ari"], float)
        assert isinstance(result["validated"], bool)

    def test_validate_with_precomputed_embeddings(self, mock_adata):
        """Test validation when embeddings are already computed."""
        n_cells = mock_adata.n_obs
        # Add fake embeddings with structure matching cell types
        embeddings = np.zeros((n_cells, 16), dtype=np.float32)
        labels = mock_adata.obs["cell_type_predicted"].values
        for i, ct in enumerate(["neuron", "astrocyte", "OPC"]):
            mask = labels == ct
            embeddings[mask] = np.random.randn(mask.sum(), 16) + i * 5
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)
        mock_adata.obsm["X_scGPT"] = embeddings

        result = validate_annotations_scgpt(
            mock_adata,
            label_col="cell_type_predicted",
            obsm_key="X_scGPT",
        )
        # With well-separated clusters, ARI should be high
        assert result["ari"] > 0.3
        assert result["n_scgpt_clusters"] >= 2

    def test_missing_label_col_raises(self, mock_adata):
        with pytest.raises(ValueError, match="Label column"):
            validate_annotations_scgpt(mock_adata, label_col="nonexistent")


class TestAnnotationConfidence:
    def test_confidence_with_clear_clusters(self, mock_adata):
        """Test confidence scoring with well-separated embeddings."""
        n_cells = mock_adata.n_obs
        embeddings = np.zeros((n_cells, 16), dtype=np.float32)
        labels = mock_adata.obs["cell_type_predicted"].values
        for i, ct in enumerate(["neuron", "astrocyte", "OPC"]):
            mask = labels == ct
            embeddings[mask] = np.random.randn(mask.sum(), 16) * 0.1 + i * 10
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)
        mock_adata.obsm["X_scGPT"] = embeddings

        confidence = compute_annotation_confidence(
            mock_adata,
            label_col="cell_type_predicted",
        )
        assert confidence.shape == (n_cells,)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
        # With very well separated clusters, most cells should have high confidence
        assert np.mean(confidence > 0.5) > 0.5

    def test_confidence_missing_embeddings_raises(self, mock_adata):
        with pytest.raises(ValueError, match="Run embed_cells first"):
            compute_annotation_confidence(mock_adata)
