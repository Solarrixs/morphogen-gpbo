"""Coverage push tests for 02_map_to_hnoca.py and 05_cellrank2_virtual.py.

These tests target the least-covered code paths in both modules to push
coverage above 60%. Heavy external dependencies (scarches, moscot, cellrank)
are mocked where needed.
"""

import math
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import _import_pipeline_module

step02 = _import_pipeline_module("02_map_to_hnoca")
step05 = _import_pipeline_module("05_cellrank2_virtual")


# =============================================================================
# 02_map_to_hnoca.py — prepare_query_for_scpoli
# =============================================================================

class TestPrepareQueryForScpoli:
    """Tests for prepare_query_for_scpoli — gene alignment and batch setup."""

    @pytest.fixture
    def ref_adata(self):
        """Minimal reference AnnData with 5 genes."""
        ref = ad.AnnData(
            X=sp.csr_matrix(np.ones((10, 5))),
            var=pd.DataFrame({"highly_variable": [True] * 5},
                             index=["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]),
            obs=pd.DataFrame({"batch": ["ref"] * 10},
                             index=[f"ref_{i}" for i in range(10)]),
        )
        ref.layers["counts"] = ref.X.copy()
        return ref

    @pytest.fixture
    def query_adata(self):
        """Minimal query AnnData with overlapping genes."""
        n_cells = 20
        X = sp.csr_matrix(np.random.rand(n_cells, 6))
        query = ad.AnnData(
            X=X,
            var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneX", "GeneY", "GeneZ"]),
            obs=pd.DataFrame({
                "sample": [f"s{i % 3}" for i in range(n_cells)],
                "condition": ["cond_A"] * 10 + ["cond_B"] * 10,
            }, index=[f"q_{i}" for i in range(n_cells)]),
        )
        query.layers["counts"] = X.copy()
        return query

    def test_shared_genes_subset(self, query_adata, ref_adata):
        """Output should have same var_names as reference."""
        result = step02.prepare_query_for_scpoli(query_adata, ref_adata)
        assert list(result.var_names) == list(ref_adata.var_names)

    def test_batch_column_created(self, query_adata, ref_adata):
        """Batch column should be created from sample column."""
        result = step02.prepare_query_for_scpoli(query_adata, ref_adata, batch_column="sample")
        assert "batch" in result.obs.columns
        assert result.obs["batch"].nunique() == 3

    def test_batch_column_fallback(self, ref_adata):
        """When batch column is missing, default to 'query'."""
        query = ad.AnnData(
            X=sp.csr_matrix(np.ones((5, 3))),
            var=pd.DataFrame(index=["GeneA", "GeneB", "GeneX"]),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]),
        )
        result = step02.prepare_query_for_scpoli(query, ref_adata, batch_column="nonexistent")
        assert (result.obs["batch"] == "query").all()

    def test_scpoli_label_cols_added(self, query_adata, ref_adata):
        """scPoli placeholder label columns should be added."""
        result = step02.prepare_query_for_scpoli(query_adata, ref_adata)
        for col in ["snapseed_pca_rss_level_1", "snapseed_pca_rss_level_12",
                     "snapseed_pca_rss_level_123"]:
            assert col in result.obs.columns
            assert (result.obs[col] == "unknown").all()

    def test_obsm_varm_cleared(self, query_adata, ref_adata):
        """obsm and varm should be cleared."""
        query_adata.obsm["X_pca"] = np.zeros((20, 10))
        result = step02.prepare_query_for_scpoli(query_adata, ref_adata)
        assert len(result.obsm) == 0
        assert len(result.varm) == 0

    def test_gene_symbol_mapping(self, ref_adata):
        """Query with gene_name_unique column should remap var_names."""
        query = ad.AnnData(
            X=sp.csr_matrix(np.ones((5, 4))),
            var=pd.DataFrame({
                "gene_name_unique": ["GeneA", "GeneB", "GeneD", "GeneZ"],
            }, index=["ENSG001", "ENSG002", "ENSG003", "ENSG004"]),
            obs=pd.DataFrame({"sample": ["s1"] * 5},
                             index=[f"c{i}" for i in range(5)]),
        )
        query.layers["counts"] = query.X.copy()
        result = step02.prepare_query_for_scpoli(query, ref_adata)
        assert list(result.var_names) == list(ref_adata.var_names)

    def test_zero_filled_for_missing_genes(self, query_adata, ref_adata):
        """Genes not in query should be zero-filled in output."""
        result = step02.prepare_query_for_scpoli(query_adata, ref_adata)
        X = result.X.toarray() if sp.issparse(result.X) else result.X
        # GeneD and GeneE are not in query, should be zero
        gene_d_idx = list(result.var_names).index("GeneD")
        gene_e_idx = list(result.var_names).index("GeneE")
        assert np.all(X[:, gene_d_idx] == 0)
        assert np.all(X[:, gene_e_idx] == 0)

    def test_dense_x_input(self, ref_adata):
        """Should handle dense X input (no counts layer)."""
        query = ad.AnnData(
            X=np.ones((5, 3)),
            var=pd.DataFrame(index=["GeneA", "GeneC", "GeneX"]),
            obs=pd.DataFrame({"sample": ["s1"] * 5},
                             index=[f"c{i}" for i in range(5)]),
        )
        result = step02.prepare_query_for_scpoli(query, ref_adata)
        assert result.shape == (5, 5)


# =============================================================================
# 02_map_to_hnoca.py — filter_quality_cells edge cases
# =============================================================================

class TestFilterQualityCellsEdgeCases:
    """Edge case tests for filter_quality_cells."""

    def test_no_qc_columns(self):
        """When no QC columns exist, return adata unchanged."""
        adata = ad.AnnData(
            X=sp.csr_matrix((10, 5)),
            obs=pd.DataFrame({"gene_count": range(10)},
                             index=[f"c{i}" for i in range(10)]),
        )
        result = step02.filter_quality_cells(adata)
        assert result.n_obs == 10

    def test_all_filtered_quality(self):
        """When all cells are filtered, return empty adata."""
        adata = ad.AnnData(
            X=sp.csr_matrix((5, 3)),
            obs=pd.DataFrame({"quality": ["remove"] * 5},
                             index=[f"c{i}" for i in range(5)]),
        )
        result = step02.filter_quality_cells(adata)
        assert result.n_obs == 0

    def test_cluster_label_filter(self):
        """ClusterLabel == 'filtered' should be removed."""
        adata = ad.AnnData(
            X=sp.csr_matrix((6, 3)),
            obs=pd.DataFrame({
                "ClusterLabel": ["Neuron", "filtered", "Glia", "filtered", "NPC", "Neuron"],
            }, index=[f"c{i}" for i in range(6)]),
        )
        result = step02.filter_quality_cells(adata)
        assert result.n_obs == 4
        assert "filtered" not in result.obs["ClusterLabel"].values

    def test_quality_column_takes_priority(self):
        """When both quality and ClusterLabel exist, quality is used."""
        adata = ad.AnnData(
            X=sp.csr_matrix((4, 3)),
            obs=pd.DataFrame({
                "quality": ["keep", "remove", "keep", "keep"],
                "ClusterLabel": ["Neuron", "Neuron", "filtered", "Neuron"],
            }, index=[f"c{i}" for i in range(4)]),
        )
        result = step02.filter_quality_cells(adata)
        # quality filter: keep rows 0, 2, 3 (ignores ClusterLabel)
        assert result.n_obs == 3


# =============================================================================
# 02_map_to_hnoca.py — transfer_labels_knn edge cases
# =============================================================================

class TestTransferLabelsKnnEdgeCases:
    """Edge case tests for transfer_labels_knn."""

    @pytest.fixture
    def knn_fixture(self):
        """Standard KNN fixture with 3 types."""
        np.random.seed(42)
        n_ref, n_query, d = 60, 15, 8
        # Create clustered reference data
        ref_latent = np.vstack([
            np.random.randn(20, d) + np.array([2, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(20, d) + np.array([0, 2, 0, 0, 0, 0, 0, 0]),
            np.random.randn(20, d) + np.array([0, 0, 2, 0, 0, 0, 0, 0]),
        ])
        query_latent = np.random.randn(n_query, d)
        ref_obs = pd.DataFrame({
            "annot_level_1": ["TypeA"] * 20 + ["TypeB"] * 20 + ["TypeC"] * 20,
            "annot_level_2": ["SubA1"] * 10 + ["SubA2"] * 10 + ["SubB1"] * 20 + ["SubC1"] * 20,
        }, index=[f"r{i}" for i in range(n_ref)])
        query_obs = pd.DataFrame(
            index=[f"q{i}" for i in range(n_query)],
        )
        return ref_latent, query_latent, ref_obs, query_obs

    def test_class_balanced_false(self, knn_fixture):
        """class_balanced=False should still produce valid results."""
        ref_latent, query_latent, ref_obs, query_obs = knn_fixture
        results, soft = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_1"], k=10,
            class_balanced=False,
        )
        assert "predicted_annot_level_1" in results.columns
        prob_df = soft["annot_level_1"]
        assert np.allclose(prob_df.sum(axis=1), 1.0, atol=1e-6)

    def test_multiple_label_columns(self, knn_fixture):
        """Should transfer labels for all specified columns."""
        ref_latent, query_latent, ref_obs, query_obs = knn_fixture
        results, soft = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_1", "annot_level_2"], k=10,
        )
        assert "predicted_annot_level_1" in results.columns
        assert "predicted_annot_level_2" in results.columns
        assert "annot_level_1" in soft
        assert "annot_level_2" in soft

    def test_missing_label_column_skipped(self, knn_fixture):
        """Missing label columns should be skipped gracefully."""
        ref_latent, query_latent, ref_obs, query_obs = knn_fixture
        results, soft = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_1", "nonexistent_col"], k=10,
        )
        assert "predicted_annot_level_1" in results.columns
        assert "predicted_nonexistent_col" not in results.columns
        assert "nonexistent_col" not in soft

    def test_confidence_in_range(self, knn_fixture):
        """Confidence scores should be in [0, 1]."""
        ref_latent, query_latent, ref_obs, query_obs = knn_fixture
        results, _ = step02.transfer_labels_knn(
            ref_latent, query_latent, ref_obs, query_obs,
            label_columns=["annot_level_1"], k=10,
        )
        conf = results["annot_level_1_confidence"]
        assert (conf >= 0).all()
        assert (conf <= 1).all()


# =============================================================================
# 02_map_to_hnoca.py — compute_cell_type_fractions edge cases
# =============================================================================

class TestComputeFractionsEdgeCases:
    """Edge case tests for compute_cell_type_fractions."""

    def test_single_condition(self):
        """Single condition should still produce valid fractions."""
        obs = pd.DataFrame({
            "condition": ["A"] * 10,
            "label": ["X"] * 6 + ["Y"] * 4,
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label", quality_filter=False)
        assert len(fracs) == 1
        assert fracs.loc["A", "X"] == pytest.approx(0.6)

    def test_quality_filter_off(self):
        """quality_filter=False should skip quality column filtering."""
        obs = pd.DataFrame({
            "condition": ["A"] * 4,
            "label": ["X", "X", "Y", "Y"],
            "quality": ["keep", "remove", "keep", "remove"],
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label", quality_filter=False)
        assert fracs.loc["A"].sum() == pytest.approx(1.0)

    def test_quality_filter_on(self):
        """quality_filter=True should filter to 'keep' rows."""
        obs = pd.DataFrame({
            "condition": ["A"] * 4,
            "label": ["X", "X", "Y", "Y"],
            "quality": ["keep", "remove", "keep", "remove"],
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label", quality_filter=True)
        assert fracs.loc["A", "X"] == pytest.approx(0.5)
        assert fracs.loc["A", "Y"] == pytest.approx(0.5)


# =============================================================================
# 05_cellrank2_virtual.py — load_temporal_atlas
# =============================================================================

class TestLoadTemporalAtlas:
    """Tests for load_temporal_atlas."""

    @pytest.fixture
    def atlas_h5ad(self, tmp_path):
        """Create a minimal temporal atlas h5ad."""
        n = 60
        obs = pd.DataFrame({
            "day": [7] * 20 + [15] * 20 + [30] * 20,
            "cell_type": ["Neuron"] * 30 + ["Glia"] * 30,
        }, index=[f"c{i}" for i in range(n)])
        adata = ad.AnnData(X=sp.csr_matrix((n, 10)), obs=obs)
        path = tmp_path / "atlas.h5ad"
        adata.write(str(path))
        return path

    def test_loads_and_returns_anndata(self, atlas_h5ad):
        result = step05.load_temporal_atlas(atlas_h5ad, time_key="day")
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 60

    def test_time_key_numeric(self, atlas_h5ad):
        result = step05.load_temporal_atlas(atlas_h5ad, time_key="day")
        assert pd.api.types.is_numeric_dtype(result.obs["day"])

    def test_missing_time_key_raises(self, atlas_h5ad):
        with pytest.raises(ValueError, match="Time key"):
            step05.load_temporal_atlas(atlas_h5ad, time_key="nonexistent")


# =============================================================================
# 05_cellrank2_virtual.py — preprocess_for_moscot
# =============================================================================

class TestPreprocessForMoscot:
    """Tests for preprocess_for_moscot."""

    def test_computes_pca_and_neighbors(self):
        np.random.seed(42)
        adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(50, 200)))
        result = step05.preprocess_for_moscot(adata, n_pcs=10, n_neighbors=10)
        assert "X_pca" in result.obsm
        assert "neighbors" in result.uns

    def test_does_not_mutate_input(self):
        np.random.seed(42)
        adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(50, 200)))
        original_shape = adata.shape
        step05.preprocess_for_moscot(adata, n_pcs=10, n_neighbors=10)
        assert adata.shape == original_shape
        assert "X_pca" not in adata.obsm  # original should not be modified

    def test_skips_if_already_preprocessed(self):
        np.random.seed(42)
        adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(50, 200)))
        adata.obsm["X_pca"] = np.random.rand(50, 10)
        adata.uns["neighbors"] = {"params": {}}
        result = step05.preprocess_for_moscot(adata, n_pcs=10, n_neighbors=10)
        # Should preserve existing PCA
        assert result.obsm["X_pca"].shape == (50, 10)


# =============================================================================
# 05_cellrank2_virtual.py — _compose_transport_chain
# =============================================================================

class TestComposeTransportChain:
    """Tests for _compose_transport_chain."""

    def test_single_step_transport(self):
        """Single step transport should return the matrix directly."""
        n_source, n_target = 20, 30
        T = sp.csr_matrix(np.random.rand(n_source, n_target))

        mock_problem = MagicMock()
        mock_solution = MagicMock()
        mock_solution.transport_matrix = T
        mock_problem.__getitem__ = MagicMock(return_value=MagicMock(solution=mock_solution))

        source_indices = np.arange(n_source)
        transport, local_idx, use = step05._compose_transport_chain(
            mock_problem, [7, 15], n_target, source_indices,
        )
        assert use is True
        assert transport.shape == (n_source, n_target)
        assert sp.issparse(transport)

    def test_multi_step_chain(self):
        """Multi-step transport should compose matrices."""
        T1 = sp.csr_matrix(np.eye(10, 15))
        T2 = sp.csr_matrix(np.eye(15, 20))

        mock_problem = MagicMock()
        solutions = {
            (7, 15): MagicMock(solution=MagicMock(transport_matrix=T1)),
            (15, 30): MagicMock(solution=MagicMock(transport_matrix=T2)),
        }
        mock_problem.__getitem__ = lambda self, key: solutions[key]

        source_indices = np.arange(10)
        transport, local_idx, use = step05._compose_transport_chain(
            mock_problem, [7, 15, 30], 20, source_indices,
        )
        assert use is True
        assert transport.shape == (10, 20)

    def test_dimension_mismatch_returns_false(self):
        """When transport target dim != atlas target cells, should fail."""
        T = sp.csr_matrix(np.eye(10, 15))

        mock_problem = MagicMock()
        mock_problem.__getitem__ = MagicMock(
            return_value=MagicMock(solution=MagicMock(transport_matrix=T)),
        )

        source_indices = np.arange(10)
        transport, local_idx, use = step05._compose_transport_chain(
            mock_problem, [7, 15], 20,  # 20 != 15
            source_indices,
        )
        assert use is False

    def test_key_error_returns_false(self):
        """KeyError from missing timepoint pair should return False."""
        mock_problem = MagicMock()
        mock_problem.__getitem__ = MagicMock(side_effect=KeyError("missing"))

        source_indices = np.arange(10)
        transport, local_idx, use = step05._compose_transport_chain(
            mock_problem, [7, 15], 20, source_indices,
        )
        assert use is False


# =============================================================================
# 05_cellrank2_virtual.py — _project_condition_transport edge cases
# =============================================================================

class TestProjectConditionTransport:
    """Tests for _project_condition_transport."""

    def test_valid_projection(self):
        """Should produce valid fractions from transport matrix."""
        n_source, n_target = 10, 20
        transport = sp.csr_matrix(np.random.rand(n_source, n_target))
        local_idx_map = np.arange(n_source)
        neighbor_idx = np.array([[0, 1, 2], [3, 4, 5]])
        weights = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        cond_indices = np.array([0, 1])
        target_labels_arr = np.array(["TypeA"] * 10 + ["TypeB"] * 10)
        target_ct_fracs = pd.Series({"TypeA": 0.5, "TypeB": 0.5})

        result = step05._project_condition_transport(
            transport, local_idx_map, neighbor_idx, weights,
            cond_indices, target_labels_arr, target_ct_fracs,
        )
        assert isinstance(result, pd.Series)
        assert result.sum() > 0

    def test_no_valid_neighbors_returns_atlas_avg(self):
        """When all neighbors are out of range, return atlas average."""
        transport = sp.csr_matrix(np.eye(5, 10))
        local_idx_map = np.array([0, 1, 2, 3, 4])
        # All neighbor indices are out of range
        neighbor_idx = np.array([[100, 200, 300]])
        weights = np.array([[0.5, 0.3, 0.2]])
        cond_indices = np.array([0])
        target_labels_arr = np.array(["A"] * 5 + ["B"] * 5)
        target_ct_fracs = pd.Series({"A": 0.5, "B": 0.5})

        result = step05._project_condition_transport(
            transport, local_idx_map, neighbor_idx, weights,
            cond_indices, target_labels_arr, target_ct_fracs,
        )
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)


# =============================================================================
# 05_cellrank2_virtual.py — _project_condition_push
# =============================================================================

class TestProjectConditionPush:
    """Tests for _project_condition_push."""

    def test_push_api_success(self):
        """Successful push() call should return valid fractions."""
        n_source = 20
        n_target = 30
        source_mask = np.array([True] * n_source + [False] * 10)
        neighbor_idx = np.arange(6).reshape(3, 2)
        weights = np.ones((3, 2)) * 0.5
        cond_indices = np.array([0, 1])

        target_labels = pd.Series(["TypeA"] * 15 + ["TypeB"] * 15)
        target_ct_fracs = pd.Series({"TypeA": 0.5, "TypeB": 0.5})

        # Mock problem with push() returning valid distribution
        mock_problem = MagicMock()
        target_dist = np.zeros(n_target)
        target_dist[:15] = 1.0 / 15  # All mass on TypeA
        mock_problem.push.return_value = target_dist

        result = step05._project_condition_push(
            mock_problem, 7, 30, source_mask, neighbor_idx, weights,
            cond_indices, target_labels, target_ct_fracs, n_target,
        )
        assert isinstance(result, pd.Series)
        assert result["TypeA"] > result["TypeB"]

    def test_push_api_failure_returns_atlas_avg(self):
        """When push() raises, should fall back to atlas average."""
        source_mask = np.array([True] * 10)
        neighbor_idx = np.arange(4).reshape(2, 2)
        weights = np.ones((2, 2)) * 0.5
        cond_indices = np.array([0])

        target_labels = pd.Series(["A"] * 5 + ["B"] * 5)
        target_ct_fracs = pd.Series({"A": 0.6, "B": 0.4})

        mock_problem = MagicMock()
        mock_problem.push.side_effect = RuntimeError("moscot error")

        result = step05._project_condition_push(
            mock_problem, 7, 30, source_mask, neighbor_idx, weights,
            cond_indices, target_labels, target_ct_fracs, 10,
        )
        assert result["A"] == pytest.approx(0.6)
        assert result["B"] == pytest.approx(0.4)


# =============================================================================
# 05_cellrank2_virtual.py — _resolve_target_labels
# =============================================================================

class TestResolveTargetLabelsExtended:
    """Extended tests for _resolve_target_labels."""

    def test_harmonization_applied(self):
        """Non-standard labels should be harmonized via LABEL_HARMONIZATION."""
        obs = pd.DataFrame({
            "cell_type": ["Excitatory neuron"] * 5 + ["Radial glia"] * 5,
        })
        labels, fracs, col = step05._resolve_target_labels(obs, "nonexistent")
        assert col == "cell_type"
        # Should be harmonized
        assert "Cortical EN" in labels.values
        assert "Cortical RG" in labels.values

    def test_standard_column_no_harmonization(self):
        """When preferred column exists, no harmonization should occur."""
        obs = pd.DataFrame({
            "predicted_annot_level_2": ["ExN"] * 3 + ["InN"] * 7,
        })
        labels, fracs, col = step05._resolve_target_labels(obs, "predicted_annot_level_2")
        assert col == "predicted_annot_level_2"
        assert "ExN" in labels.values

    def test_fracs_sum_to_one(self):
        """Returned fractions should sum to 1.0."""
        obs = pd.DataFrame({
            "cell_type": ["A"] * 3 + ["B"] * 7,
        })
        _, fracs, _ = step05._resolve_target_labels(obs, "nonexistent")
        assert fracs.sum() == pytest.approx(1.0)

    def test_unmapped_labels_kept(self):
        """Labels not in LABEL_HARMONIZATION should be kept as-is."""
        obs = pd.DataFrame({
            "cell_type": ["UnknownType"] * 10,
        })
        labels, fracs, _ = step05._resolve_target_labels(obs, "nonexistent")
        assert "UnknownType" in labels.values


# =============================================================================
# 05_cellrank2_virtual.py — _embed_query_in_atlas_pca extended
# =============================================================================

class TestEmbedQueryInAtlasPcaExtended:
    """Extended tests for _embed_query_in_atlas_pca."""

    def test_with_pca_loadings_in_atlas(self):
        """Should project query through atlas PCA loadings when available."""
        n_genes = 50
        n_query = 10
        n_source = 20
        n_pcs = 15

        gene_names = [f"Gene{i}" for i in range(n_genes)]

        atlas = ad.AnnData(
            X=sp.csr_matrix((30, n_genes)),
            var=pd.DataFrame({
                "highly_variable": [True] * 30 + [False] * (n_genes - 30),
            }, index=gene_names),
        )
        atlas.varm["PCs"] = np.random.rand(n_genes, n_pcs)

        query = ad.AnnData(
            X=sp.csr_matrix(np.random.rand(n_query, n_genes)),
            var=pd.DataFrame(index=gene_names),
        )

        source_mask = np.array([True] * n_source + [False] * 10)
        source_pca = np.random.rand(n_source, n_pcs)

        q_emb, s_pca = step05._embed_query_in_atlas_pca(
            query, atlas, source_mask, source_pca,
        )
        assert q_emb.shape[0] == n_query
        assert q_emb.shape[1] == s_pca.shape[1]

    def test_fallback_to_joint_pca(self):
        """When no PCA loadings exist, should compute joint PCA."""
        np.random.seed(42)
        n_genes = 200
        n_query = 20
        n_source = 40

        gene_names = [f"Gene{i}" for i in range(n_genes)]
        atlas = ad.AnnData(
            X=sp.csr_matrix(np.random.rand(60, n_genes)),
            obs=pd.DataFrame({"day": [7] * 60}, index=[f"a{i}" for i in range(60)]),
            var=pd.DataFrame(index=gene_names),
        )
        query = ad.AnnData(
            X=sp.csr_matrix(np.random.rand(n_query, n_genes)),
            var=pd.DataFrame(index=gene_names),
        )

        source_mask = np.array([True] * n_source + [False] * 20)
        source_pca = np.random.rand(n_source, 30)

        q_emb, s_pca = step05._embed_query_in_atlas_pca(
            query, atlas, source_mask, source_pca,
        )
        assert q_emb.shape[0] == n_query
        assert s_pca.shape[0] == n_source


# =============================================================================
# 05_cellrank2_virtual.py — build_virtual_morphogen_matrix edge cases
# =============================================================================

class TestBuildVirtualMorphogenMatrixEdgeCases:
    """Edge case tests for build_virtual_morphogen_matrix."""

    def test_missing_condition_skipped(self, tmp_path):
        """Conditions not in real morphogens should be skipped."""
        morph = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "log_harvest_day": [math.log(72)],
        }, index=["cond_A"])
        morph_path = tmp_path / "morph.csv"
        morph.to_csv(str(morph_path))

        virtual_fracs = pd.DataFrame({
            "original_condition": ["cond_A", "nonexistent"],
            "target_day": [90, 90],
            "TypeA": [0.5, 0.5],
        }, index=["cond_A_day90", "nonexistent_day90"])

        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert len(result) == 1
        assert "cond_A_day90" in result.index

    def test_empty_input(self, tmp_path):
        """Empty virtual fractions should return empty DataFrame."""
        morph = pd.DataFrame({"CHIR99021_uM": [1.5]}, index=["cond_A"])
        morph_path = tmp_path / "morph.csv"
        morph.to_csv(str(morph_path))

        virtual_fracs = pd.DataFrame({
            "original_condition": pd.Series([], dtype=str),
            "target_day": pd.Series([], dtype=int),
        })
        virtual_fracs.index.name = "virtual_condition"

        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert result.empty

    def test_harvest_day_updated(self, tmp_path):
        """log_harvest_day should be updated to target timepoint."""
        morph = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "log_harvest_day": [math.log(72)],
        }, index=["cond_A"])
        morph_path = tmp_path / "morph.csv"
        morph.to_csv(str(morph_path))

        virtual_fracs = pd.DataFrame({
            "original_condition": ["cond_A"],
            "target_day": [90],
            "TypeA": [0.5],
        }, index=["cond_A_day90"])

        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert result.loc["cond_A_day90", "log_harvest_day"] == pytest.approx(math.log(90))


# =============================================================================
# 05_cellrank2_virtual.py — validate_transport_quality
# =============================================================================

class TestValidateTransportQuality:
    """Tests for validate_transport_quality."""

    def test_valid_transport(self):
        """Should report OK for converged, low-cost solutions."""
        mock_problem = MagicMock()
        mock_sol1 = MagicMock()
        mock_sol1.cost = 10.0
        mock_sol1.converged = True
        mock_sol2 = MagicMock()
        mock_sol2.cost = 20.0
        mock_sol2.converged = True
        mock_problem.solutions = {(7, 15): mock_sol1, (15, 30): mock_sol2}

        result = step05.validate_transport_quality(mock_problem)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert (result["status"] == "OK").all()

    def test_high_cost_flagged(self):
        """Solutions with cost > threshold should be flagged."""
        mock_problem = MagicMock()
        mock_sol = MagicMock()
        mock_sol.cost = 200.0
        mock_sol.converged = True
        mock_problem.solutions = {(7, 15): mock_sol}

        result = step05.validate_transport_quality(mock_problem, max_cost_threshold=100.0)
        assert result.iloc[0]["status"] == "HIGH_COST"

    def test_not_converged_flagged(self):
        """Non-converged solutions should be flagged."""
        mock_problem = MagicMock()
        mock_sol = MagicMock()
        mock_sol.cost = 10.0
        mock_sol.converged = False
        mock_problem.solutions = {(7, 15): mock_sol}

        result = step05.validate_transport_quality(mock_problem)
        assert result.iloc[0]["status"] == "NOT_CONVERGED"


# =============================================================================
# 05_cellrank2_virtual.py — LABEL_HARMONIZATION
# =============================================================================

class TestLabelHarmonization:
    """Tests for the LABEL_HARMONIZATION mapping."""

    def test_all_values_are_strings(self):
        for k, v in step05.LABEL_HARMONIZATION.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_no_empty_keys_or_values(self):
        for k, v in step05.LABEL_HARMONIZATION.items():
            assert len(k) > 0
            assert len(v) > 0

    def test_key_cell_types_present(self):
        """Core cell types should be in harmonization map."""
        expected_keys = [
            "Excitatory neuron", "Inhibitory neuron", "Radial glia",
            "Astrocyte", "Oligodendrocyte precursor",
        ]
        for key in expected_keys:
            assert key in step05.LABEL_HARMONIZATION


# =============================================================================
# 05_cellrank2_virtual.py — generate_virtual_training_data (mocked)
# =============================================================================

class TestGenerateVirtualTrainingData:
    """Tests for generate_virtual_training_data with mocked transport."""

    @pytest.fixture
    def mock_setup(self, tmp_path):
        """Create query adata, atlas, mock problem, and morphogen CSV."""
        np.random.seed(42)
        n_query = 30
        n_atlas = 60

        # Atlas with days 7, 15, 30
        atlas_obs = pd.DataFrame({
            "day": [7] * 20 + [15] * 20 + [30] * 20,
            "annot_level_2": (["Neuron"] * 10 + ["NPC"] * 10) * 3,
        }, index=[f"a{i}" for i in range(n_atlas)])
        atlas = ad.AnnData(X=sp.csr_matrix(np.random.rand(n_atlas, 50)), obs=atlas_obs)
        atlas.obsm["X_pca"] = np.random.rand(n_atlas, 10)

        # Query with conditions
        query_obs = pd.DataFrame({
            "condition": ["cond_A"] * 15 + ["cond_B"] * 15,
            "predicted_annot_level_2": (["Neuron"] * 8 + ["NPC"] * 7) * 2,
        }, index=[f"q{i}" for i in range(n_query)])
        query = ad.AnnData(X=sp.csr_matrix(np.random.rand(n_query, 50)), obs=query_obs)
        query.obsm["X_pca"] = np.random.rand(n_query, 10)

        # Morphogen CSV
        morph = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "log_harvest_day": [math.log(21), math.log(21)],
        }, index=["cond_A", "cond_B"])
        morph_path = tmp_path / "morph.csv"
        morph.to_csv(str(morph_path))

        # Mock problem — no push() API, use transport chain
        # Use a regular class instance instead of MagicMock(spec=[])
        # because MagicMock with spec won't allow setting __getitem__
        class FakeProblem:
            def __getitem__(self, key):
                T = sp.csr_matrix(np.random.rand(20, 20))
                sol = MagicMock()
                sol.transport_matrix = T
                return MagicMock(solution=sol)
        mock_problem = FakeProblem()

        return query, atlas, mock_problem, morph_path

    def test_returns_tuple_of_dataframes(self, mock_setup):
        query, atlas, problem, morph_path = mock_setup
        virtual_X, virtual_Y = step05.generate_virtual_training_data(
            query_adata=query,
            atlas_adata=atlas,
            problem=problem,
            real_morphogen_csv=morph_path,
            query_timepoint=7,
            target_timepoints=[15, 30],
            condition_key="condition",
        )
        assert isinstance(virtual_X, pd.DataFrame)
        assert isinstance(virtual_Y, pd.DataFrame)

    def test_matching_indices(self, mock_setup):
        query, atlas, problem, morph_path = mock_setup
        virtual_X, virtual_Y = step05.generate_virtual_training_data(
            query_adata=query,
            atlas_adata=atlas,
            problem=problem,
            real_morphogen_csv=morph_path,
            query_timepoint=7,
            target_timepoints=[15, 30],
            condition_key="condition",
        )
        if not virtual_X.empty:
            assert list(virtual_X.index) == list(virtual_Y.index)
