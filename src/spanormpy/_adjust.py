"""Adjustments applied to a fitted SpaNorm model (``normalise*`` in ``R/fitSpaNormNB.R``).

Every function takes counts ``Y`` (genes x cells; dense, scipy sparse or any
row-sliceable array), a ``scale_factor`` and a fit, and returns a dense
genes x cells ``numpy`` array. ``backend`` selects where the maths runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from ._backend import (
    DEFAULT_WINSOR,
    as_tensor,
    gene_block_indices,
    get_backend_dtype,
    resolve_device,
)
from ._fitnb import calculate_mu
from ._nbdist import compute_psi_max, dnbinom, pnbinom, qnbinom, winsorise_psi

_GENE_MODELS = ("nb",)


def _t(x, dev) -> torch.Tensor:
    return as_tensor(x, dev, get_backend_dtype() if dev.type != "cpu" else torch.float64)


def _out(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _dense(Y) -> np.ndarray:
    if sp.issparse(Y):
        return Y.toarray()
    return np.asarray(Y)


def _cols(fit, kind: str) -> np.ndarray:
    return np.asarray(fit.wtype) == kind


def _mu(fit, cols, backend: str):
    return calculate_mu(
        fit.gmean, np.asarray(fit.alpha)[:, cols], np.asarray(fit.W)[:, cols], backend=backend
    )


def _psi(fit):
    return winsorise_psi(fit.psi, DEFAULT_WINSOR, psi_max=getattr(fit, "psi_max", None))


def _log2(x):
    return torch.log2(x) if isinstance(x, torch.Tensor) else np.log2(x)


def normalise_logpac(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """logPAC: percentile-adjusted counts mapped onto the biology-only NB (R ``normaliseLogPAC``)."""
    dev = resolve_device(backend)
    allcols = np.ones(len(fit.wtype), dtype=bool)
    mu = _t(_mu(fit, allcols, backend), dev)
    mu2 = _t(_mu(fit, _cols(fit, "biology"), backend), dev)
    psi = _t(_psi(fit), dev).to(mu.dtype)
    size = 1.0 / psi
    Yd = _t(_dense(Y), dev).to(mu.dtype)
    lb = pnbinom(Yd - 1, mu, size)
    ub = dnbinom(Yd, mu, size) + lb
    p = ((lb + ub) / 2).clamp(0.001, 0.999)
    return _out(torch.log2(qnbinom(p, scale_factor * mu2, size) + 1))


def normalise_meanbio(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """log2 of the biology-only fitted mean (R ``normaliseMeanBio``)."""
    return _out(_log2(_mu(fit, _cols(fit, "biology"), backend)))


def normalise_meanbatch(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """log2 of the batch-only fitted mean (R ``normaliseMeanBatch``)."""
    return _out(_log2(_mu(fit, _cols(fit, "batch"), backend)))


def normalise_meanls(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """log2 of the library-size-only fitted mean at the median library size (R ``normaliseMeanLS``)."""
    cols = _cols(fit, "ls")
    W = np.asarray(fit.W)[:, cols]
    W = W / W[:, [0]] * np.median(W[:, 0])
    return _out(_log2(calculate_mu(fit.gmean, np.asarray(fit.alpha)[:, cols], W, backend=backend)))


def normalise_medbio(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """Median of the biology-only NB (R ``normaliseMedianBio``)."""
    dev = resolve_device(backend)
    mu2 = _t(_mu(fit, _cols(fit, "biology"), backend), dev)
    size = 1.0 / _t(_psi(fit), dev).to(mu2.dtype)
    return _out(torch.log2(qnbinom(0.5, scale_factor * mu2, size) + 1))


def normalise_pearson(Y, scale_factor, fit, backend: str = "cpu") -> np.ndarray:
    """Pearson residuals with the biology effect kept in the baseline (R ``normalisePearson``)."""
    dev = resolve_device(backend)
    allcols = np.ones(len(fit.wtype), dtype=bool)
    mu = _t(_mu(fit, allcols, backend), dev)
    psi = _t(_psi(fit), dev).to(mu.dtype)
    base = _t(_mu(fit, ~_cols(fit, "biology"), backend), dev)
    Yd = _t(_dense(Y), dev).to(mu.dtype)
    return _out((Yd - base) / torch.sqrt(mu + mu**2 * psi.unsqueeze(1)))


_ADJ = {
    "auto": normalise_logpac,
    "logpac": normalise_logpac,
    "pearson": normalise_pearson,
    "medbio": normalise_medbio,
    "meanbio": normalise_meanbio,
}


def get_gene_models() -> tuple[str, ...]:
    return _GENE_MODELS


def get_adjustment_fun(gene_model: str, adj_method: str) -> Callable:
    """Resolve an adjustment function (R ``getAdjustmentFun``); ``"auto"`` is logPAC."""
    if gene_model != "nb":
        raise ValueError(f"'{gene_model}' gene model not supported")
    try:
        return _ADJ[adj_method]
    except KeyError:
        raise ValueError("invalid argument for 'adj.method'") from None


def global_psi_max(psi) -> float:
    """The whole-fit dispersion cap (finite log-dispersions only; ``inf`` if none are finite)."""
    with np.errstate(all="ignore"):
        return compute_psi_max(psi, DEFAULT_WINSOR)


def normalise_block_size(Y) -> float:
    """Element budget per block for ``Y`` (R ``.normaliseBlockSize``).

    Disk-backed or lazy input (h5py dataset, backed AnnData sparse dataset,
    dask array) gives R's DelayedArray budget, ``max(1, 1e8 / 8)`` elements
    (default auto block size of 100 MB in doubles); in-memory input gives ``inf``.
    """
    cls = type(Y)
    root = cls.__module__.split(".")[0]
    lazy = root in ("h5py", "dask", "zarr") or (
        root == "anndata"
        and cls.__name__ in ("CSRDataset", "CSCDataset", "BaseCompressedSparseDataset")
    )
    return max(1.0, 1e8 / 8) if lazy else math.inf


def subset_fit_genes(fit, rows) -> SimpleNamespace:
    """Gene subset of a fit that keeps the global dispersion threshold (R ``subsetFitGenes``)."""
    psi = np.asarray(fit.psi, dtype=float)
    # a fit that is already a gene subset carries the global threshold of the full fit
    psi_max = getattr(fit, "psi_max", None)
    if psi_max is None:
        psi_max = global_psi_max(psi)
    return SimpleNamespace(
        gmean=np.asarray(fit.gmean)[rows],
        alpha=np.asarray(fit.alpha)[rows, :],
        psi=psi[rows],
        psi_max=psi_max,
        W=fit.W,
        wtype=fit.wtype,
    )


def normalise_blocked(
    adj_fun: Callable,
    Y,
    scale_factor,
    fit,
    n_jobs: int = 1,
    block_size: float | None = None,
    backend: str = "cpu",
) -> np.ndarray:
    """Apply ``adj_fun`` over contiguous gene blocks and stack them (R ``normaliseBlocked``).

    ``block_size`` is an element budget per block (``None`` = :func:`normalise_block_size`); with ``n_jobs > 1`` and at least 1e6 elements, at least
    ``2 * n_jobs`` blocks are run on a thread pool.
    """
    ng, nc = Y.shape
    total = float(ng) * nc
    parallelise = n_jobs > 1 and total >= 1e6
    bs = normalise_block_size(Y) if block_size is None else block_size
    nblocks = math.ceil(total / bs) if total > bs else 1
    if parallelise:
        nblocks = max(nblocks, 2 * n_jobs)
    nblocks = min(ng, nblocks)

    def call(*args):
        return adj_fun(*args, backend=backend) if backend != "cpu" else adj_fun(*args)

    if nblocks <= 1:
        return call(Y, scale_factor, fit)

    if sp.issparse(Y) and not sp.isspmatrix_csr(Y) and not isinstance(Y, sp.csr_array):
        Y = Y.tocsr()
    blocks = gene_block_indices(ng, nblocks)

    def onblock(r):
        sl = slice(int(r[0]), int(r[-1]) + 1)
        return call(_dense(Y[sl, :]), scale_factor, subset_fit_genes(fit, sl))

    if parallelise:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            res = list(ex.map(onblock, blocks))
    else:
        res = [onblock(r) for r in blocks]
    return np.vstack(res)
