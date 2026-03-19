"""scGPT brain checkpoint integration for annotation validation.

Loads the pre-trained scGPT brain model and extracts cell embeddings.
These embeddings enable:
  1. Independent validation of scPoli cell type annotations (cross-check)
  2. Semantic similarity between organoid cells and reference brain cell types
  3. Discovery of cell states not captured by discrete label transfer

The brain checkpoint (CellxGene Census brain subset) lives at:
  data/scgpt_brain/{best_model.pt, vocab.json, args.json}

Requires: scgpt>=0.2.0, torch
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

SCGPT_BRAIN_DIR = DATA_DIR / "scgpt_brain"


def _import_transformer_model():
    """Import scGPT's TransformerModel without triggering scgpt.__init__.

    scgpt.__init__ imports scbank which imports HuggingFace 'datasets',
    but gopro/datasets.py shadows that package. We import the model module
    directly via importlib.
    """
    import importlib
    import importlib.util

    # Find the scgpt package location
    scgpt_spec = importlib.util.find_spec("scgpt")
    if scgpt_spec is None:
        raise ImportError("scgpt package not found. Install with: pip install scgpt")
    scgpt_dir = Path(scgpt_spec.submodule_search_locations[0])

    # Import prerequisite submodules that model.model needs
    for submod_name, submod_file in [
        ("scgpt.model.dsbn", scgpt_dir / "model" / "dsbn.py"),
        ("scgpt.model.grad_reverse", scgpt_dir / "model" / "grad_reverse.py"),
    ]:
        if submod_name not in importlib.import_module("sys").modules:
            spec = importlib.util.spec_from_file_location(submod_name, submod_file)
            mod = importlib.util.module_from_spec(spec)
            importlib.import_module("sys").modules[submod_name] = mod
            spec.loader.exec_module(mod)

    # Import the model module
    model_file = scgpt_dir / "model" / "model.py"
    spec = importlib.util.spec_from_file_location("scgpt.model.model", model_file)
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return model_mod.TransformerModel


def _binning(row: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """Bin expression values into n_bins quantile-based bins.

    Reimplemented from scgpt.preprocess.binning to avoid import chain issues.
    """
    if row.max() == 0:
        return np.zeros_like(row, dtype=np.int64)
    non_zero_ids = row.nonzero()[0]
    non_zero_row = row[non_zero_ids]
    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
    non_zero_digits = np.digitize(non_zero_row, bins)
    binned_row = np.zeros_like(row, dtype=np.int64)
    binned_row[non_zero_ids] = non_zero_digits
    return binned_row


# ---------------------------------------------------------------------------
# Lightweight vocab wrapper (bypasses broken torchtext dependency)
# ---------------------------------------------------------------------------

class VocabDict:
    """Minimal vocabulary that mimics the torchtext Vocab interface.

    Loads token->index mapping from JSON and supports __getitem__,
    __contains__, __len__, and __call__ (batch lookup).
    """

    def __init__(self, token2idx: Dict[str, int]):
        self._token2idx = dict(token2idx)
        self._idx2token = {v: k for k, v in self._token2idx.items()}
        self._default_index: Optional[int] = None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "VocabDict":
        with open(path) as f:
            return cls(json.load(f))

    def __getitem__(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __call__(self, tokens):
        """Batch lookup: list of tokens -> list of indices."""
        return [self[t] for t in tokens]

    def set_default_index(self, idx: int) -> None:
        self._default_index = idx

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._token2idx)

    def get_itos(self) -> list:
        return [self._idx2token.get(i, "") for i in range(len(self._idx2token))]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_scgpt_brain(
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> tuple:
    """Load the scGPT brain checkpoint.

    Reference: Cui et al. (2024), "scGPT: toward building a foundation model
    for single-cell multi-omics using generative AI", Nature Methods
    21:1470-1480, DOI:10.1038/s41592-024-02201-0.

    Args:
        model_dir: Path to checkpoint directory. Defaults to data/scgpt_brain/.
        device: 'cpu' or 'cuda'. Defaults to CPU (safe for macOS).

    Returns:
        (model, vocab, model_configs) tuple.
    """
    model_dir = Path(model_dir or SCGPT_BRAIN_DIR)
    if device is None:
        device = "cpu"

    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    for f in (vocab_file, config_file, model_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing scGPT checkpoint file: {f}. "
                f"Download the brain checkpoint to {model_dir}/"
            )

    vocab = VocabDict.from_file(vocab_file)
    logger.info("Loaded scGPT vocab with %d tokens", len(vocab))

    with open(config_file) as f:
        model_configs = json.load(f)

    # Import TransformerModel directly to avoid scgpt.__init__ importing
    # scbank->datasets which is shadowed by gopro/datasets.py
    TransformerModel = _import_transformer_model()

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            logger.warning("Special token %s not in vocab, skipping", s)

    pad_token = model_configs.get("pad_token", "<pad>")
    pad_value = model_configs.get("pad_value", -2)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=model_configs.get("dropout", 0.2),
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )

    # SECURITY NOTE: weights_only=False allows arbitrary code execution from
    # malicious checkpoint files. Only load models from trusted sources
    # (e.g., CellxGene Census, official scGPT releases).
    state_dict = torch.load(model_file, map_location=device, weights_only=False)
    # Remap flash-attn keys to standard PyTorch transformer keys
    state_dict = {
        k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded scGPT brain model: %d layers, %d-dim embeddings, device=%s",
        model_configs["nlayers"],
        model_configs["embsize"],
        device,
    )
    return model, vocab, model_configs


# ---------------------------------------------------------------------------
# Cell embedding extraction
# ---------------------------------------------------------------------------

class _CellDataset(torch.utils.data.Dataset):
    """Dataset that yields tokenized cells for scGPT."""

    def __init__(self, count_matrix, gene_ids, cls_id, pad_value):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.cls_id = cls_id
        self.pad_value = pad_value

    def __len__(self):
        return self.count_matrix.shape[0]

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        if hasattr(row, "toarray"):
            row = row.toarray().squeeze()
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # prepend <cls> token
        genes = np.insert(genes, 0, self.cls_id)
        values = np.insert(values, 0, self.pad_value)
        return {
            "id": idx,
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
        }


def embed_cells(
    adata,
    model=None,
    vocab=None,
    model_configs=None,
    model_dir: Optional[Union[str, Path]] = None,
    gene_col: str = "index",
    max_length: int = 1200,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract scGPT cell embeddings from an AnnData object.

    Reference: Cui et al. (2024), "scGPT: toward building a foundation model
    for single-cell multi-omics using generative AI", Nature Methods
    21:1470-1480, DOI:10.1038/s41592-024-02201-0.

    Args:
        adata: AnnData with raw counts in X. Gene names in var.index or var[gene_col].
        model: Pre-loaded TransformerModel (or None to load from model_dir).
        vocab: Pre-loaded VocabDict (or None to load from model_dir).
        model_configs: Pre-loaded config dict (or None to load from model_dir).
        model_dir: Path to scGPT checkpoint (used if model/vocab/model_configs are None).
        gene_col: Column in adata.var containing gene symbols, or 'index'.
        max_length: Maximum sequence length for the transformer.
        batch_size: Inference batch size.
        device: Device string ('cpu' or 'cuda').

    Returns:
        np.ndarray of shape (n_cells, embsize) with L2-normalized embeddings.
    """
    from scipy import sparse

    if model is None or vocab is None or model_configs is None:
        model, vocab, model_configs = load_scgpt_brain(model_dir, device)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # Map genes to vocab IDs
    if gene_col == "index":
        gene_names = adata.var_names.tolist()
    else:
        gene_names = adata.var[gene_col].tolist()

    id_in_vocab = np.array([vocab[g] if g in vocab else -1 for g in gene_names])
    n_matched = np.sum(id_in_vocab >= 0)
    logger.info(
        "Gene overlap: %d / %d genes matched scGPT vocab (%.1f%%)",
        n_matched, len(gene_names), 100 * n_matched / len(gene_names),
    )
    if n_matched < 100:
        warnings.warn(
            f"Very low gene overlap ({n_matched} genes) with scGPT vocab. "
            "Embeddings may be unreliable."
        )

    # Subset to matched genes
    mask = id_in_vocab >= 0
    adata_sub = adata[:, mask].copy()
    gene_ids = id_in_vocab[mask]

    # Keep sparse if possible — _CellDataset handles sparse rows in __getitem__
    X = adata_sub.X
    if not sparse.issparse(X):
        X = np.asarray(X, dtype=np.float32)
    count_matrix = X

    pad_token = model_configs.get("pad_token", "<pad>")
    pad_value = model_configs.get("pad_value", -2)
    embsize = model_configs["embsize"]
    cls_id = vocab["<cls>"]
    pad_id = vocab[pad_token]

    dataset = _CellDataset(count_matrix, gene_ids, cls_id, pad_value)

    def _collate_fn(examples):
        """Collate, bin, pad/sample to max_length. No MLM masking for inference."""
        max_ori_len = max(len(ex["genes"]) for ex in examples)
        _max_len = min(max_length, max_ori_len)
        padded_genes = []
        padded_exprs = []
        for ex in examples:
            genes = ex["genes"]
            exprs = ex["expressions"]
            # Bin expression values (skip CLS at position 0)
            binned = _binning(exprs[1:].numpy())
            exprs = torch.cat([exprs[:1], torch.from_numpy(binned).float()])
            # Sample/truncate or pad
            if len(genes) > _max_len:
                idx = torch.randperm(len(genes) - 1)[:_max_len - 1]
                idx = torch.cat([torch.tensor([0]), idx + 1])
                genes, exprs = genes[idx], exprs[idx]
            elif len(genes) < _max_len:
                pad_g = torch.full((_max_len - len(genes),), pad_id, dtype=genes.dtype)
                pad_e = torch.full((_max_len - len(exprs),), pad_value, dtype=exprs.dtype)
                genes = torch.cat([genes, pad_g])
                exprs = torch.cat([exprs, pad_e])
            padded_genes.append(genes)
            padded_exprs.append(exprs)
        return {
            "gene": torch.stack(padded_genes),
            "expr": torch.stack(padded_exprs),
        }

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=_collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    cell_embeddings = np.zeros((len(dataset), embsize), dtype=np.float32)
    count = 0

    with torch.no_grad():
        for data_dict in data_loader:
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(pad_id)
            embeddings = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )
            # CLS token embedding at position 0
            embeddings = embeddings[:, 0, :].cpu().numpy()
            cell_embeddings[count: count + len(embeddings)] = embeddings
            count += len(embeddings)

    # L2 normalize
    norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    cell_embeddings = cell_embeddings / norms

    logger.info("Extracted scGPT embeddings: shape %s", cell_embeddings.shape)
    return cell_embeddings


def add_scgpt_embeddings(
    adata,
    obsm_key: str = "X_scGPT",
    **kwargs,
):
    """Add scGPT cell embeddings to adata.obsm[obsm_key] in-place.

    All keyword arguments are forwarded to :func:`embed_cells`.

    Returns:
        The same adata object (modified in place).
    """
    adata.obsm[obsm_key] = embed_cells(adata, **kwargs)
    return adata


# ---------------------------------------------------------------------------
# Annotation validation
# ---------------------------------------------------------------------------

def validate_annotations_scgpt(
    adata,
    label_col: str = "cell_type_predicted",
    obsm_key: str = "X_scGPT",
    n_neighbors: int = 15,
    min_agreement: float = 0.5,
    model=None,
    vocab=None,
    model_configs=None,
    **kwargs,
) -> dict:
    """Validate cell type annotations by comparing scPoli labels to scGPT embedding clusters.

    Computes scGPT embeddings (if not already in adata.obsm), runs Leiden clustering
    on the scGPT embedding space, and measures agreement between scPoli-assigned labels
    and scGPT-derived clusters via Adjusted Rand Index (ARI; Hubert & Arabie 1985,
    "Comparing Partitions", Journal of Classification 2:193-218) and per-cell-type
    purity.

    Args:
        adata: AnnData with cell type labels in obs[label_col].
        label_col: Column in obs with scPoli-transferred labels.
        obsm_key: Key in obsm for scGPT embeddings.
        n_neighbors: Number of neighbors for the embedding KNN graph.
        min_agreement: Minimum ARI to consider annotations validated.
        model, vocab, model_configs: Pre-loaded scGPT model (optional).

    Returns:
        Dict with keys:
            - 'ari': Adjusted Rand Index between scPoli labels and scGPT clusters
            - 'per_type_purity': Dict[str, float] mapping cell type -> cluster purity
            - 'n_scgpt_clusters': Number of Leiden clusters found
            - 'validated': bool, True if ARI >= min_agreement
            - 'flagged_types': List of cell types with purity < 0.5
    """
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    if label_col not in adata.obs.columns:
        raise ValueError(f"Label column '{label_col}' not found in adata.obs")

    # Compute embeddings if needed
    if obsm_key not in adata.obsm:
        logger.info("Computing scGPT embeddings for validation...")
        embed_cells_result = embed_cells(
            adata, model=model, vocab=vocab, model_configs=model_configs, **kwargs
        )
        adata.obsm[obsm_key] = embed_cells_result

    # Build KNN graph on scGPT embeddings and cluster (minimal AnnData to save memory)
    import anndata as ad
    adata_tmp = ad.AnnData(
        obs=adata.obs[[label_col]].copy(),
        obsm={obsm_key: adata.obsm[obsm_key]},
    )
    sc.pp.neighbors(adata_tmp, use_rep=obsm_key, n_neighbors=n_neighbors)
    sc.tl.leiden(adata_tmp, resolution=1.0, key_added="scgpt_leiden")

    labels_scpoli = adata_tmp.obs[label_col].astype(str).values
    labels_scgpt = adata_tmp.obs["scgpt_leiden"].astype(str).values

    ari = adjusted_rand_score(labels_scpoli, labels_scgpt)
    n_clusters = len(set(labels_scgpt))

    # Per-type purity: for each scPoli cell type, what fraction of cells in the
    # dominant scGPT cluster actually belong to that type?
    per_type_purity = {}
    for ct in sorted(set(labels_scpoli)):
        mask = labels_scpoli == ct
        if mask.sum() == 0:
            continue
        clusters_for_type = labels_scgpt[mask]
        # Find the most common cluster for this cell type
        unique, counts = np.unique(clusters_for_type, return_counts=True)
        dominant_cluster = unique[np.argmax(counts)]
        # Purity: of all cells in the dominant cluster, what fraction has this label?
        cluster_mask = labels_scgpt == dominant_cluster
        purity = np.sum(labels_scpoli[cluster_mask] == ct) / cluster_mask.sum()
        per_type_purity[ct] = float(purity)

    flagged = [ct for ct, p in per_type_purity.items() if p < 0.5]

    result = {
        "ari": float(ari),
        "per_type_purity": per_type_purity,
        "n_scgpt_clusters": n_clusters,
        "validated": ari >= min_agreement,
        "flagged_types": flagged,
    }

    logger.info(
        "Annotation validation: ARI=%.3f, %d scGPT clusters, %d/%d types flagged",
        ari, n_clusters, len(flagged), len(per_type_purity),
    )
    if flagged:
        logger.warning("Low-purity cell types: %s", flagged)

    return result


def compute_annotation_confidence(
    adata,
    label_col: str = "cell_type_predicted",
    obsm_key: str = "X_scGPT",
) -> np.ndarray:
    """Compute per-cell annotation confidence using scGPT embeddings.

    For each cell, measures how well it clusters with other cells of the same
    predicted type in scGPT embedding space. Returns a confidence score in [0, 1].

    Args:
        adata: AnnData with scGPT embeddings in obsm and labels in obs.
        label_col: Column with cell type predictions.
        obsm_key: Key in obsm for scGPT embeddings.

    Returns:
        np.ndarray of shape (n_cells,) with confidence scores.
    """
    if obsm_key not in adata.obsm:
        raise ValueError(f"Run embed_cells first: {obsm_key} not in adata.obsm")

    embeddings = adata.obsm[obsm_key]
    labels = adata.obs[label_col].astype(str).values
    n_cells = len(labels)
    confidence = np.zeros(n_cells, dtype=np.float32)

    # For each cell type, compute mean embedding and measure each cell's
    # cosine similarity to its type centroid vs the global centroid
    global_centroid = embeddings.mean(axis=0)
    global_centroid = global_centroid / (np.linalg.norm(global_centroid) + 1e-8)

    for ct in np.unique(labels):
        mask = labels == ct
        if mask.sum() < 2:
            confidence[mask] = 0.5
            continue
        type_centroid = embeddings[mask].mean(axis=0)
        type_centroid = type_centroid / (np.linalg.norm(type_centroid) + 1e-8)

        # Cosine similarity to type centroid
        sim_to_type = embeddings[mask] @ type_centroid
        # Cosine similarity to global centroid
        sim_to_global = embeddings[mask] @ global_centroid

        # Confidence = how much closer the cell is to its type vs global
        # Rescale so 0.5 = equal distance, 1 = very close to type
        diff = sim_to_type - sim_to_global
        # Sigmoid-like rescaling to [0, 1]
        confidence[mask] = 1.0 / (1.0 + np.exp(-5 * diff))

    return confidence
