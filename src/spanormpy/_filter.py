"""Gene filtering (port of ``R/filterGenes.R``)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _check_counts(Y) -> None:
    """Port of R ``checkSPE`` applied to a genes x cells matrix."""
    if Y.shape[0] == 0 or Y.shape[1] == 0:
        raise ValueError("'counts' must have nrow > 0 and ncol > 0")
    vals = Y.data if sp.issparse(Y) else np.asarray(Y, dtype=float).ravel()
    if np.any(vals % 1 != 0):
        raise ValueError("'counts' has non-integer values")
    if np.any(vals < 0):
        raise ValueError("'counts' has negative values")


def get_counts_genes_by_cells(adata: AnnData, layer: str | None = None):
    """Counts of ``adata`` (``X`` or ``layer``) in R orientation (genes x cells)."""
    X = adata.X if layer is None else adata.layers[layer]
    return X.T.tocsr() if sp.issparse(X) else np.asarray(X).T


def _filter_genes(Y, prop: float) -> np.ndarray:
    """``rowMeans(Y > 0) >= prop`` for a genes x cells matrix (dense or sparse)."""
    if Y.shape[1] == 0:
        raise ValueError("'counts' must have nrow > 0 and ncol > 0")
    if sp.issparse(Y):
        nnz = np.asarray((Y > 0).sum(axis=1)).ravel()
    else:
        nnz = (np.asarray(Y) > 0).sum(axis=1)
    return nnz / Y.shape[1] >= prop


def filter_genes(
    adata: AnnData, prop: float = 0.1, layer: str | None = None, inplace: bool = True
) -> np.ndarray | None:
    """Flag genes expressed in at least ``prop`` of the spots.

    Mirrors R ``filterGenes``: validates ``0 <= prop <= 1`` and the counts, then computes
    ``rowMeans(counts > 0) >= prop``. With ``inplace=False`` the boolean mask (R's return
    value) is returned. With ``inplace=True`` (scanpy ``filter_genes`` convention) the genes
    failing the filter are removed from ``adata`` and ``None`` is returned.

    Examples
    --------
    >>> import spanormpy as sn
    >>> # adata: the HumanDLPFC AnnData (raw counts in X, coordinates in obsm["spatial"])
    >>> keep = sn.filter_genes(adata, prop=0.2, inplace=False)  # genes expressed in >= 20% of spots
    >>> int(keep.sum()), keep.size
    """
    if not (0 <= prop <= 1):
        raise ValueError("'prop' must be between 0 and 1")
    Y = get_counts_genes_by_cells(adata, layer)
    _check_counts(Y)
    keep = _filter_genes(Y, prop)
    if not inplace:
        return keep
    adata._inplace_subset_var(keep)
    return None
