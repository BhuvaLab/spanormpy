"""Gene-blocked driver of the polish engines (port of ``R/polishNB.R``).

:func:`polish_nb` converges every gene of a shared NB fit to that gene's own
penalised-NB optimum; :func:`nb_profile_psi` re-profiles the dispersion at fixed
coefficients. Both take and return arrays in R's orientation (genes x cells). R's
``BPPARAM`` becomes ``n_jobs`` (a thread pool; BLAS threads are limited to one per
worker through ``threadpoolctl`` when it is installed), and
``options(SpaNorm.polish.mem.budget)`` becomes ``spanormpy.settings.polish_mem_budget``.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ._backend import as_tensor, check_gpu, get_backend_device, resolve_device
from ._block_design import (
    _block_expand,
    _block_no_device,
    _is_block_design,
    _newton_solver_compact,
)
from ._polish_batch import (
    POLISH_GENE_CELL_MATS,
    mu_batch,
    offset_rows,
    polish_batch,
    polish_batch_size,
    psi_profile_batch,
    require_float64,
)
from ._polish_engine import nb_mu_floor, polish_gene
from ._solver import nb_newton_solver

__all__ = ["nb_mu_floor", "nb_profile_psi", "polish_nb"]


# --------------------------------------------------------------------------------------
# Gene blocking and dispatch
# --------------------------------------------------------------------------------------


def chunk_genes(n: int, block_size: int | None = None) -> list[np.ndarray]:
    """Partition ``range(n)`` into blocks of ``block_size`` (R ``.chunkGenes``)."""
    if block_size is None or block_size >= n:
        return [np.arange(n)]
    block_size = max(1, int(block_size))
    idx = np.arange(n)
    return [idx[i : i + block_size] for i in range(0, n, block_size)]


def _has_threadpoolctl() -> bool:
    """Is ``threadpoolctl`` installed? A function so a test can mock the probe."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        return False
    return True


def _single_blas(n_jobs: int) -> bool:
    """Should multi-worker dispatch run each worker's BLAS single-threaded?"""
    return n_jobs > 1 and _has_threadpoolctl()


def bplapply_single_blas(items, fun, n_jobs: int = 1) -> list:
    """Map ``fun`` over ``items``, on ``n_jobs`` threads with single-threaded BLAS.

    Workers share the process, so the BLAS/OpenMP thread count is limited once around
    the whole dispatch (``threadpoolctl``) rather than per worker; without
    ``threadpoolctl`` the workers keep the thread count they inherit. Results keep the
    order of ``items``; serial dispatch is a plain loop.
    """
    items = list(items)
    if n_jobs <= 1:
        return [fun(x) for x in items]
    ctx = nullcontext()
    if _single_blas(n_jobs):
        from threadpoolctl import threadpool_limits

        ctx = threadpool_limits(limits=1)
    with ctx, ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(fun, items))


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------


def _polish_shapes(Y, W, alpha, psi) -> None:
    """Check that counts, design, coefficients and dispersions agree in shape.

    A mismatch does not always fail by itself (a design with more rows than ``Y`` has
    columns recycles each gene's counts against the longer mean), so the check is up
    front and names the argument and both sizes.
    """
    if W.shape[0] != Y.shape[1]:
        raise ValueError(
            "'W' must have one row per cell (column of Y): "
            f"nrow(W) = {W.shape[0]}, ncol(Y) = {Y.shape[1]}"
        )
    if alpha.shape[0] != Y.shape[0]:
        raise ValueError(
            "'alpha' must have one row per gene (row of Y): "
            f"nrow(alpha) = {alpha.shape[0]}, nrow(Y) = {Y.shape[0]}"
        )
    if alpha.shape[1] != W.shape[1]:
        raise ValueError(
            "'alpha' must have one column per column of W: "
            f"ncol(alpha) = {alpha.shape[1]}, ncol(W) = {W.shape[1]}"
        )
    if np.size(psi) not in (1, Y.shape[0]):
        raise ValueError(
            "'psi' must be one value or one per gene: "
            f"length(psi) = {np.size(psi)}, nrow(Y) = {Y.shape[0]}"
        )


def _polish_offset(offset, Y):
    """Check and normalise the offset: ``None``, a per-cell vector or a genes x cells matrix.

    One place decides what an offset may be, for :func:`polish_nb` and
    :func:`nb_profile_psi` alike. Sparse offsets are made dense; a torch tensor is
    refused (the engines take a host offset and move it themselves).
    """
    if offset is None:
        return None
    ng, nc = Y.shape
    shape = (
        f"NULL, a numeric vector with one value per cell ({nc}), "
        f"or a numeric genes x cells matrix ({ng} x {nc})"
    )
    shape = shape.replace("NULL", "None")
    if isinstance(offset, torch.Tensor):
        raise ValueError(  # noqa: TRY004
            "'offset' must be a numpy vector or matrix, not a torch tensor: the polish "
            f"moves it to the device itself. Expected {shape}."
        )
    if sp.issparse(offset):
        offset = offset.toarray()
    if isinstance(offset, pd.DataFrame | pd.Series):
        offset = offset.to_numpy()
    off = np.asarray(offset)
    numeric = off.dtype.kind in "iuf"
    if off.ndim <= 1:
        ok = numeric and off.size == nc
    else:
        ok = numeric and off.ndim == 2 and off.shape == (ng, nc)
    if not ok:
        raise ValueError(f"'offset' must be {shape}.")
    off = off.astype(np.float64)
    if not np.isfinite(off).all():
        raise ValueError(f"'offset' must be finite (no NA, NaN or Inf): expected {shape}.")
    return off.reshape(-1) if off.ndim <= 1 else off


def _check_psi_range(psi_range) -> None:
    ok = (
        np.size(psi_range) == 2
        and np.all(np.isfinite(np.asarray(psi_range, dtype=float)))
        and psi_range[0] > 0
        and psi_range[0] < psi_range[1]
    )
    if not ok:
        raise ValueError("'psi_range' must be two increasing positive numbers")


def _dense_rows(Y, gi: np.ndarray) -> np.ndarray:
    """Densify the genes ``gi`` of ``Y`` (genes x cells; dense, sparse or tensor) as float64."""
    if sp.issparse(Y):
        return np.asarray(Y[gi].toarray(), dtype=np.float64)
    if isinstance(Y, torch.Tensor):
        return Y[torch.as_tensor(gi)].cpu().numpy().astype(np.float64)
    return np.asarray(Y[gi], dtype=np.float64)


def _as_matrix(x, nm: str):
    """Counts-like input as an object with ``.shape`` and row indexing (no copy)."""
    if sp.issparse(x):
        return x.tocsr()
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"'{nm}' must be a 2-D matrix")
    return arr


def _design(W, device):
    """The design on ``device``: an ``NBBlockDesign`` as is, anything else a float tensor."""
    if _is_block_design(W):
        return W
    if isinstance(W, pd.DataFrame):
        W = W.to_numpy()
    if sp.issparse(W):
        W = W.toarray()
    return as_tensor(W, device, torch.float64)


def _is_logical(x) -> bool:
    return x is not None and np.asarray(x).dtype == bool


# --------------------------------------------------------------------------------------
# polish_nb
# --------------------------------------------------------------------------------------


def polish_nb(
    Y,
    W,
    alpha,
    psi,
    lambda_a=0.0,
    offset=None,
    absorb=None,
    absorb_batch=None,
    start_cols=None,
    psi_method: str = "profile",
    psi_range=(1e-3, 1e3),
    warm: bool = False,
    maxit: int = 50,
    tol: float = 1e-8,
    engine: str = "batch",
    batch_size: int | None = None,
    block_size: int | None = None,
    backend: str = "cpu",
    n_jobs: int = 1,
    verbose: bool = False,
) -> dict:
    r"""Converge every gene of a shared NB fit to its own optimum (R ``polishNB``).

    ``fit_nb`` fits every gene in one IRLS loop with a shared cell weight vector and an
    aggregate convergence test; for a bright, cell-type-restricted gene that leaves it
    well short of its own optimum with an inflated dispersion. This takes such a fit
    and converges each gene separately by damped Newton on that gene's penalised NB
    log-likelihood

    .. math:: \sum_i \log f_{NB}(y_i; \mu_i, \psi) - \tfrac12 \sum_j \lambda_j \alpha_j^2,
              \quad \log\mu = W\alpha + \text{offset}.

    With ``psi_method="profile"`` the dispersion is then re-estimated by profile ML at
    the converged mean and the mean re-polished, twice; a gene whose dispersion optimum
    sits on the search bound keeps its input dispersion (``psi_bound``). A gene whose
    start is degenerate (a fitted log-mean below -10 at a cell with a positive count) or
    whose Newton diverges restarts from a sane point. A gene that cannot be polished from
    either start keeps its input coefficients and dispersion with ``polished=False``.

    The integer-count check reads only the first 20 genes: a cheap guard against an
    assay that is non-integer throughout, not a scan of every value.

    Parameters
    ----------
    Y
        ``genes x cells`` integer counts (dense, scipy sparse or tensor; densified one
        gene block at a time).
    W
        ``cells x p`` design, or an :func:`~spanormpy._block_design.nb_block_design`
        (CPU only; its groups' blocks are absorbed by construction).
    alpha
        ``genes x p`` starting coefficients, typically ``fit_nb()`` output.
    psi
        starting per-gene dispersions (length ``nrow(Y)``, or one value).
    lambda_a
        ridge penalty: one value or one per column of ``W`` (for a block design, also
        one per column of ``[X | Z]``). Applied as given, with no scaling.
    offset
        ``None``, a log-scale offset with one value per cell, or a genes x cells matrix
        (finite), added to every linear predictor with its coefficient fixed at 1.
    absorb
        columns the per-gene solver absorbs by a Schur complement: ``None``, a boolean
        vector (each marked column its own 1x1 block; 0/1 indicators partitioning the
        cells), or a per-column grouping (``NaN``/``None`` for a dense column).
    absorb_batch
        the absorption for the shared-factor batched solver that runs on a device:
        ``None`` or a boolean vector (1x1 blocks only). ``None`` passes a boolean
        ``absorb`` through and sends a grouping to the dense batched solver.
    start_cols
        boolean vector over the columns of ``W`` marking indicator columns (such as
        cell-type intercepts) that the sane start fills with the gene's log mean over
        that column's cells, or ``None`` for the overall log mean on the first column.
    psi_method
        ``"profile"`` or ``"fixed"`` (keep the input ``psi``, converge only the mean).
    psi_range
        search interval for the profile dispersion; a gene whose optimum falls on either
        end keeps its input dispersion.
    warm
        ``alpha``/``psi`` are an already converged fit at a nearby penalty: a few Newton
        steps at the held dispersion, no dispersion search and no restart check.
    maxit, tol
        Newton iteration cap and relative log-likelihood tolerance, per gene.
    engine
        ``"batch"`` (default) runs a batch of genes through each Newton step; ``"gene"``
        is the per-gene reference. The batched profile dispersion is a bisection and the
        per-gene one Brent's method, so they agree on ``psi`` to ``optimize()``'s tolerance.
    batch_size
        genes per batched Newton, or ``None`` to size it from
        ``spanormpy.settings.polish_mem_budget`` (bytes per worker, default 1e9).
    block_size
        genes per block (the unit of densification and dispatch), or ``None`` for at
        least one block per worker and at most 2,000 genes.
    backend
        ``"cpu"`` (default), ``"auto"`` or ``"gpu"``. A device needs ``engine="batch"``
        and float64, and forces serial dispatch.
    n_jobs
        threads over gene blocks.
    verbose
        report progress on stderr.

    Returns
    -------
    dict
        ``alpha`` (genes x p), ``psi`` and ``loglik`` (the penalised log-likelihood at the
        returned fit, NaN for a gene that was not polished), and ``polish``, a DataFrame
        with one row per gene: ``iterations``, ``psi_fitnb``, ``restarted``, ``capped``,
        ``singular``, ``psi_bound`` and ``polished``.
    """
    if psi_method not in ("profile", "fixed"):
        raise ValueError("'psi_method' should be one of 'profile', 'fixed'")
    if engine not in ("batch", "gene"):
        raise ValueError("'engine' should be one of 'batch', 'gene'")
    if backend not in ("cpu", "auto", "gpu"):
        raise ValueError("'backend' should be one of 'cpu', 'auto', 'gpu'")
    n_jobs = max(1, int(n_jobs))
    # short-circuits before check_gpu() probes, as R does
    gpu_active = backend in ("gpu", "auto") and check_gpu()
    if gpu_active:
        require_float64()
        if engine != "batch":
            raise ValueError(
                f'backend = "{backend}" needs engine = "batch": the per-gene engine keeps one '
                "factorisation object per gene, which is what cannot go to a device."
            )
        if n_jobs > 1:
            warnings.warn(
                f"GPU backend active ({get_backend_device()}); forcing serial dispatch for "
                "the polish, since threads contend for one device",
                stacklevel=2,
            )
            n_jobs = 1
    device = resolve_device(backend) if gpu_active else torch.device("cpu")
    compact = _is_block_design(W)
    if compact:
        if gpu_active:
            _block_no_device()
        if absorb is not None or absorb_batch is not None:
            raise ValueError(
                "'absorb' and 'absorb_batch' must be None for an nbBlockDesign: its groups' "
                "blocks are absorbed by construction, and any other column belongs in X"
            )
    Y = _as_matrix(Y, "Y")
    index = alpha.index if isinstance(alpha, pd.DataFrame) else None
    alpha = np.array(alpha, dtype=np.float64)
    if alpha.ndim != 2:
        raise ValueError("'alpha' must be a genes x p matrix")
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    _polish_shapes(Y, W, alpha, psi)
    ng, p = alpha.shape
    pen = lambda_a
    if compact:
        pen = _block_expand(W, pen, "lambda_a")
        if start_cols is not None:
            start_cols = _block_expand(W, start_cols, "start_cols")
    else:
        pen = np.atleast_1d(np.asarray(pen, dtype=np.float64))
        if pen.size not in (1, p):
            raise ValueError(
                f"'lambda_a' must be a single value or one per column of W ({p} here, "
                f"{pen.size} supplied)."
            )
    pen = np.asarray(pen, dtype=np.float64)
    if pen.size == 1:
        pen = np.repeat(pen, p)
    if psi.size == 1:
        psi = np.repeat(psi, ng)
    # The NB likelihood is undefined on non-integer values: dnbinom returns -Inf for
    # every cell, every step is rejected and every gene's dispersion runs to its upper
    # bound, silently. Refuse such an assay (e.g. a back-transform 2**logcounts - 1).
    chk = _dense_rows(Y, np.arange(min(ng, 20))).reshape(-1)
    chk = chk[np.isfinite(chk)]
    if chk.size and np.max(np.abs(chk - np.round(chk))) > 1e-8:
        raise ValueError(
            "the polish stage needs integer counts: the negative binomial likelihood is "
            "undefined otherwise, and every gene's dispersion would silently collapse to "
            "its upper bound.\n  The assay passed is not integer-valued (e.g. a "
            "back-transform such as 2**logcounts - 1).\n  Use the raw counts."
        )
    if absorb_batch is not None:
        ab = np.asarray(absorb_batch)
        if ab.dtype != bool or ab.ndim != 1 or ab.size != p:
            raise ValueError(
                "'absorb_batch' must be None or a boolean vector with one value per column "
                f"of W ({p} here)"
            )
    _check_psi_range(psi_range)
    # None, a per-cell vector or a genes x cells matrix, finite, from here on
    offset = _polish_offset(offset, Y)
    # `absorb` is the per-gene solver's absorption: the nested indicators as a boolean
    # vector, or the per-sample grouping of a random-slope fit's whole random block.
    nested = absorb if absorb is not None else np.zeros(p, dtype=bool)
    if _is_logical(nested):
        nested = np.asarray(nested, dtype=bool)
    # the design goes to the device once, outside the gene blocks
    W_use = _design(W, device)
    pen_t = as_tensor(pen, device, torch.float64)
    # R builds the per-gene solver from the host design; on a device path the batched engine
    # never uses it, so it must not be built from the device tensor.
    W_host = W_use if W_use.device.type == "cpu" else _design(W, torch.device("cpu"))
    pen_host = pen_t if pen_t.device.type == "cpu" else pen_t.cpu()
    solver = (
        _newton_solver_compact(W_host, pen_host)
        if compact
        else nb_newton_solver(W_host, pen_host, nested)
    )
    # The shared-factor batched solver absorbs 1x1 blocks only, so it gets `absorb_batch`
    # (for a slope fit, the nested indicators inside the per-sample blocks the per-gene
    # solver absorbs whole). Both are exact; this costs the device path a wider dense
    # block and moves no result. A grouping cannot be reduced to its 1x1 blocks
    # generically, so without `absorb_batch` it gets the dense batched solver.
    if absorb_batch is not None:
        nested_batch = np.asarray(absorb_batch, dtype=bool)
    elif _is_logical(nested):
        nested_batch = nested
    else:
        nested_batch = np.zeros(p, dtype=bool)

    # Absent an explicit block size, split at least one block per worker, and never
    # densify more than 2000 genes at a time (this stage is exact per gene, so blocking
    # never changes the answer beyond rounding).
    if block_size is None:
        block_size = max(1, min(2000, math.ceil(ng / n_jobs)))
    blocks = chunk_genes(ng, block_size)
    single_blas = _single_blas(n_jobs)
    ncell = W.shape[0]
    bsize = None
    if engine == "batch":
        # An offset adds genes x cells matrices to the batched working set: the expansion
        # offset_rows() builds at every mean evaluation and, for a per-gene matrix, the
        # batch's own rows held for the whole batch.
        mats = POLISH_GENE_CELL_MATS + (0 if offset is None else 1 + (1 if offset.ndim == 2 else 0))
        bsize = polish_batch_size(ncell, mats=mats) if batch_size is None else int(batch_size)
    if verbose:
        _msg(
            "  {} {} genes {} ({} block{}{})".format(
                "re-polishing" if warm else "converging",
                ng,
                f"in batches of {bsize}" if engine == "batch" else "per gene",
                len(blocks),
                "" if len(blocks) == 1 else "s",
                ", one BLAS thread per worker" if single_blas else "",
            )
        )
    t0 = time.time()
    nb = len(blocks)
    step = max(1, nb // 20)

    def run_block(b):
        gi = blocks[b]
        # densify the whole block once: Y may be sparse, where a per-gene read costs a
        # round trip each time (the invariant is that the WHOLE matrix is never densified)
        Yb = _dense_rows(Y, gi)
        m = len(gi)
        out = {
            "alpha": np.empty((m, p)),
            "psi": np.empty(m),
            "loglik": np.empty(m),
            "iterations": np.empty(m, dtype=np.int64),
            "restarted": np.empty(m, dtype=bool),
            "capped": np.empty(m, dtype=bool),
            "singular": np.empty(m, dtype=bool),
            "psi_bound": np.empty(m, dtype=bool),
            "polished": np.empty(m, dtype=bool),
        }
        if engine == "batch":
            # a block bounds densification of the counts; the batched Newton's working set
            # is gene x cell and is bounded separately, so a block is walked in sub-batches
            for ii in chunk_genes(m, bsize):
                # a per-gene offset matrix is sliced to exactly these genes; a per-cell
                # vector is the same for every gene
                Oblk = offset[gi[ii]] if (offset is not None and offset.ndim == 2) else offset
                r = polish_batch(
                    as_tensor(Yb[ii], device, torch.float64),
                    W_use,
                    alpha[gi[ii]],
                    psi[gi[ii]],
                    pen_t,
                    solver,
                    maxit=maxit,
                    tol=tol,
                    start_cols=start_cols,
                    psi_method=psi_method,
                    warm=warm,
                    shared_factor=gpu_active,
                    nested=nested_batch,
                    psi_range=psi_range,
                    offset=Oblk,
                )
                for k, v in out.items():
                    v[ii] = r[k]
        else:
            for i, g in enumerate(gi):
                og = offset[g] if (offset is not None and offset.ndim == 2) else offset
                r = polish_gene(
                    Yb[i],
                    W_use,
                    alpha[g],
                    psi[g],
                    pen_t,
                    solver,
                    maxit=maxit,
                    tol=tol,
                    start_cols=start_cols,
                    psi_method=psi_method,
                    warm=warm,
                    psi_range=psi_range,
                    offset=og,
                )
                for k, v in out.items():
                    v[i] = r[k]
        if verbose and ((b + 1) % step == 0 or b + 1 == nb):
            _msg(f"    block {b + 1}/{nb} ({(time.time() - t0) / 60:.1f} min elapsed)")
        return out

    res = bplapply_single_blas(range(nb), run_block, n_jobs)
    cat = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    if verbose:
        _msg(
            "  polished {}/{} genes ({} restarted, {} not converged, {} singular, "
            "{} dispersion at a bound)".format(
                int(cat["polished"].sum()),
                ng,
                int(cat["restarted"].sum()),
                int(cat["capped"].sum()),
                int(cat["singular"].sum()),
                int(cat["psi_bound"].sum()),
            )
        )
    polish = pd.DataFrame(
        {
            "iterations": cat["iterations"],
            # the dispersion optimum sat on its search bound, so the input psi was kept
            "psi_fitnb": psi,
            "restarted": cat["restarted"],
            "capped": cat["capped"],
            "singular": cat["singular"],
            "psi_bound": cat["psi_bound"],
            # False when the gene fell back to the input fit entirely
            "polished": cat["polished"],
        },
        index=index,
    )
    return {
        "alpha": cat["alpha"],
        "psi": cat["psi"],
        "loglik": cat["loglik"],
        "polish": polish,
    }


def _msg(text: str) -> None:
    print(text, file=sys.stderr)


# --------------------------------------------------------------------------------------
# nb_profile_psi
# --------------------------------------------------------------------------------------


def nb_profile_psi(
    Y,
    W,
    alpha,
    psi,
    psi_range=(1e-3, 1e3),
    block_size: int | None = None,
    n_jobs: int = 1,
    offset=None,
) -> np.ndarray:
    """Profile-ML dispersion at fixed coefficients, blocked over genes (R ``nbProfilePsi``).

    For each gene, the NB dispersion that maximises the likelihood at the mean
    ``exp(W alpha + offset)`` with the coefficients held fixed: one profile search per
    gene and no Newton, by the same bisection on the score that the batched engine uses.
    A gene whose optimum sits on a bound of ``psi_range`` (an under-dispersed or
    near-empty gene), or whose search is not finite, keeps its input dispersion. A caller
    that re-polishes the mean at a held dispersion (``polish_nb(warm=True)``) uses this to
    put the dispersion back at the reported mean.

    Parameters
    ----------
    Y
        ``genes x cells`` counts (dense, sparse or tensor).
    W
        ``cells x p`` design or an ``NBBlockDesign``.
    alpha
        ``genes x p`` coefficients.
    psi
        per-gene dispersions (length ``nrow(Y)``, or one value), kept for a gene whose
        optimum is on a bound.
    psi_range
        search interval for the dispersion.
    block_size, n_jobs
        gene blocking and dispatch, as in :func:`polish_nb`.
    offset
        ``None``, a per-cell log-scale offset or a genes x cells matrix. The dispersion
        is profiled at the mean it gives, so it must be the offset the coefficients were
        fitted with.

    Returns
    -------
    numpy.ndarray
        one dispersion per gene.
    """
    Y = _as_matrix(Y, "Y")
    alpha = np.asarray(alpha, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    n_jobs = max(1, int(n_jobs))
    _polish_shapes(Y, W, alpha, psi)
    # a single value is every gene's, as in polish_nb(); left scalar, `psi[gi]` would
    # be wrong for every at-bound gene after the first
    if psi.size == 1:
        psi = np.repeat(psi, Y.shape[0])
    offset = _polish_offset(offset, Y)
    ng = alpha.shape[0]
    W_use = _design(W, torch.device("cpu"))
    if block_size is None:
        block_size = max(1, min(2000, math.ceil(ng / n_jobs)))
    blocks = chunk_genes(ng, block_size)

    # One dispersion optimiser in the package, not two: the same kernel and at_bound rule
    # as the batched engine; an at-bound gene keeps the dispersion it came in with.
    def run(gi):
        Yb = as_tensor(_dense_rows(Y, gi), "cpu", torch.float64)
        Ab = as_tensor(alpha[gi], "cpu", torch.float64)
        # the offset rows of exactly these genes (None stays None, so the no-offset mean
        # is the pre-offset one bit for bit)
        Mu = mu_batch(Ab, W_use, offset_rows(offset, gi, Ab))
        pm, at_bound = psi_profile_batch(Yb, Mu, psi_range)
        out = pm.cpu().numpy().copy()
        keep = at_bound | ~np.isfinite(out)
        out[keep] = psi[gi][keep]
        return out

    return np.concatenate(bplapply_single_blas(blocks, run, n_jobs))
