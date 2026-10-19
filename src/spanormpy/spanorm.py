"""``spanorm()``: spatially-dependent normalisation (R ``SpaNorm()``, ``R/mainSpaNorm.R``).

Also holds the ports of ``.spaNormCore``, ``fitSpaNorm``, ``sampleRandom``, ``matchDftps``
and ``matchLambda``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.sparse as sp

from ._adjust import get_adjustment_fun, get_gene_models, normalise_block_size, normalise_blocked
from ._fit import SpaNormFit, _as_actual, load_fit, store_fit
from ._fitnb import _message, _quiet, fit_spanorm_nb
from ._io import batch_design, get_coords, get_counts, resolve_input
from ._tps import bs_tps
from .size_factors import quick_cluster, sum_factors

_ADJ_METHODS = ("auto", "logpac", "pearson", "medbio", "meanbio")
_BACKENDS = ("auto", "cpu", "gpu")
_FIT_PASSTHROUGH = {"gpu_mem_budget", "winsor"}  # R `...` keys forwarded to fitSpaNormNB


def _match_arg(value, choices, name: str) -> str:
    """R ``match.arg`` for a single string (exact match only)."""
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"'{name}' should be one of: " + ", ".join(f"'{c}'" for c in choices)
        )
    return value


def _sample_random(coords: np.ndarray, nsub: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``nsub`` cell indices for the GLM fit (R ``sampleRandom``, as 0-based indices).

    This is the only place the cell subset is drawn, so tests can inject R's draw.
    """
    return rng.choice(coords.shape[0], size=int(nsub), replace=False)


def match_df_tps(df_fit, df_par) -> bool:
    """Does the stored 4-vector ``df_fit`` agree with the requested ``df_par`` (R ``matchDftps``)?"""
    df_fit = np.atleast_1d(df_fit)
    df_par = np.atleast_1d(df_par)
    if len(df_par) not in (1, 2):
        raise ValueError("length(df.par) %in% c(1, 2) is not TRUE")
    if len(df_fit) != 4:
        raise ValueError("length(df.fit) == 4 is not TRUE")
    bio, ls = df_fit[:2].max(), df_fit[2:].max()
    if len(df_par) == 2:
        return bool(df_par[0] == bio and df_par[1] == ls)
    return bool(df_par[0] == bio and math.ceil(df_par[0] / 2) == ls)


def match_lambda(lambda_fit, lambda_par) -> bool:
    """Does the stored ``lambda_fit`` equal the requested ``lambda_par`` (R ``matchLambda``)?"""
    lambda_fit = np.atleast_1d(lambda_fit)
    lambda_par = np.atleast_1d(lambda_par)
    if len(lambda_par) not in (1, 2):
        raise ValueError("length(lambda.par) %in% c(1, 2) is not TRUE")
    if len(lambda_fit) != 2:
        raise ValueError("length(lambda.fit) == 2 is not TRUE")
    return bool(np.all(lambda_fit == lambda_par))


def _same_batch(old, new) -> bool:
    """``isTRUE(all.equal(old, new))`` for the stored and the requested batch."""
    if old is None or new is None:
        return old is None and new is None
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape:
        return False
    num_old, num_new = old.dtype.kind in "iufb", new.dtype.kind in "iufb"
    if num_old != num_new:
        return False
    if num_old:
        a, b = old.astype(float), new.astype(float)
        denom = np.abs(a).sum()
        diff = np.abs(a - b).sum()
        return bool(diff == 0 or (diff / denom if denom > 0 else diff) < 1.5e-8)
    return bool(np.array_equal(old.astype(str), new.astype(str)))


def _batch_for_storage(batch):
    """The batch as stored in the fit (a plain array, like R's ``batch`` slot)."""
    if batch is None:
        return None
    if hasattr(batch, "to_numpy"):
        batch = batch.to_numpy()
    return np.asarray(batch)


def fit_spanorm(
    Y,
    coords,
    sample_p,
    gene_model,
    df_tps=6,
    lambda_a=1e-4,
    batch=None,
    size_factors=None,
    msgfun=_message,
    rng=None,
    backend="auto",
    **kwargs,
) -> tuple[SpaNormFit, np.ndarray | None]:
    """Fit the SpaNorm model (R ``fitSpaNorm``).

    Returns the fit and the size factors computed here (``None`` when ``size_factors`` was
    given). ``kwargs`` (``tol``, ``step_factor``, ``maxit_nb``, ``maxit_psi``, ``maxn_psi``)
    go to :func:`spanormpy._fitnb.fit_spanorm_nb`.
    """
    if gene_model not in get_gene_models():
        raise ValueError(f"'gene_model' should be one of: {', '.join(get_gene_models())}")
    if sample_p <= 0 or sample_p > 1:
        raise ValueError("'sample_p' should be in the interval (0,1]")
    lambda_a = np.atleast_1d(np.asarray(lambda_a, dtype=float))
    if np.any(lambda_a < 0):
        raise ValueError("'lambda_a' should be positive")
    if len(lambda_a) not in (1, 2):
        raise ValueError("'lambda_a' should be a single value or a vector of length 2")
    if len(lambda_a) == 1:
        lambda_a = np.repeat(lambda_a, 2)
    if rng is None:
        rng = np.random.default_rng(0)

    ncells = Y.shape[1]
    coords = np.asarray(coords, dtype=float)
    with np.errstate(all="ignore"):
        rngs = coords.max(axis=0) - coords.min(axis=0)
        coords = (coords - coords.min(axis=0)) / rngs - 0.5

    df_tps = np.atleast_1d(df_tps)
    if len(df_tps) == 1:
        df_bio = df_tps[0]
        df_ls = math.ceil(df_bio / 2)
    elif len(df_tps) == 2:
        df_bio, df_ls = df_tps
    else:
        raise ValueError("'df_tps' should be a single integer or a vector of length 2")
    bs_bio, dfs_bio = bs_tps(coords[:, 0], coords[:, 1], df_tps=df_bio)
    bs_ls, dfs_ls = bs_tps(coords[:, 0], coords[:, 1], df_tps=df_ls)
    df_tps = np.array([*dfs_bio, *dfs_ls], dtype=int)

    computed_sf = None
    if size_factors is None:
        computed_sf = sum_factors(Y, quick_cluster(Y))
        ls = computed_sf
    else:
        ls = np.asarray(size_factors, dtype=float)
        if ls.shape != (ncells,):
            raise ValueError("'size_factors' should have one value per cell/spot")
    log_ls = np.log(np.maximum(1e-08, ls))

    # W = model.matrix(~ logLS + bs.xy.bio + logLS:bs.xy.ls)[, -1]
    cols = [log_ls[:, None], bs_bio, log_ls[:, None] * bs_ls]
    def term_names(prefix, k):  # model.matrix names matrix columns <term><j>, or <term> if k == 1
        return [prefix] if k == 1 else [f"{prefix}{i + 1}" for i in range(k)]

    names = (
        ["logLS"]
        + term_names("bs.xy.bio", bs_bio.shape[1])
        + term_names("logLS:bs.xy.ls", bs_ls.shape[1])
    )
    design, bnames = batch_design(batch, ncells)
    if design is not None:
        cols.append(design)
        names += bnames
    W = np.hstack(cols)

    # mark factors representing biology of interest
    nbio, nls = int(np.prod(df_tps[:2])), int(np.prod(df_tps[2:]))
    wtype = np.full(W.shape[1], "batch", dtype=object)
    wtype[1 : nbio + 1] = "biology"
    wtype[[0, *range(nbio + 1, nbio + nls + 1)]] = "ls"
    wtype = wtype.astype(str)

    lambda_vec = np.zeros(len(wtype))
    lambda_vec[wtype == "biology"] = lambda_a[0]
    lambda_vec[wtype == "ls"] = lambda_a[1]
    lambda_vec = lambda_vec[1:]

    # sample data for computational efficiency
    maxn = 3000
    nsub = int(round(sample_p * ncells))
    msgfun(f"{nsub} cells/spots sampled to fit model")
    if nsub > maxn:
        warnings.warn(
            "consider reducing 'sample_p' to "
            f"{max(math.floor(maxn / ncells * 100) / 100, 0.01):.2f} to increase "
            "computational efficiency",
            stacklevel=3,
        )
    elif nsub == 0:
        raise ValueError(
            "'sample_p' is too small, consider using "
            f"{min(1.0, math.floor(maxn / ncells * 100) / 100):.2f}"
        )
    idx = np.zeros(ncells, dtype=bool)
    idx[_sample_random(coords, nsub, rng)] = True

    res = fit_spanorm_nb(
        Y,
        W,
        idx,
        is_spanorm=True,
        lambda_a=lambda_vec * ncells,
        msgfun=msgfun,
        backend=backend,
        rng=rng,
        **kwargs,
    )
    fit = SpaNormFit(
        ngenes=Y.shape[0],
        ncells=ncells,
        gene_model=gene_model,
        df_tps=df_tps,
        sample_p=float(sample_p),
        lambda_a=lambda_a,
        batch=_batch_for_storage(batch),
        W=W,
        alpha=res["alpha"],
        gmean=res["gmean"],
        psi=res["psi"],
        wtype=wtype,
        loglik=np.atleast_1d(res["loglik"]),
        sampling=res["sampling"],
        colnames=np.array(names, dtype=str),
    )
    return fit, computed_sf


def _resolve_size_factors(adata, size_factors):
    if size_factors is None:
        return None
    if isinstance(size_factors, str):
        if size_factors not in adata.obs:
            raise ValueError(f"'size_factors' column '{size_factors}' not found in adata.obs")
        return adata.obs[size_factors].to_numpy(dtype=float)
    return np.asarray(size_factors, dtype=float)


def _resolve_batch(adata, batch_key, batch):
    if batch_key is not None and batch is not None:
        raise ValueError("specify only one of 'batch_key' and 'batch'")
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise ValueError(f"'batch_key' column '{batch_key}' not found in adata.obs")
        col = adata.obs[batch_key]
        return col.array if hasattr(col.dtype, "categories") else col.to_numpy()
    return batch


def spanorm(
    data,
    sample_p: float = 0.25,
    gene_model: str = "nb",
    adj_method: str = "auto",
    scale_factor: float = 1.0,
    df_tps=6,
    lambda_a=1e-4,
    batch_key: str | None = None,
    batch=None,
    size_factors=None,
    tol: float = 1e-4,
    step_factor: float = 0.5,
    maxit_nb: int = 50,
    maxit_psi: int = 25,
    maxn_psi: int = 500,
    overwrite: bool = False,
    backend: str = "auto",
    n_jobs: int = 1,
    verbose: bool = True,
    layer: str | None = None,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    random_state: int = 0,
    copy: bool = False,
    **kwargs,
):
    """Spatially-dependent normalisation of spatial transcriptomics counts (R ``SpaNorm``).

    Fits (or reuses) a spatial regression model for library size and biology on the raw
    counts, then writes the adjusted data to ``adata.layers["logcounts"]``. The fit is stored
    with :func:`spanormpy._fit.store_fit` under the name ``"spanorm"`` and the adjustment
    method in ``adata.uns["spanorm"]["adj_method"]``.

    Parameters
    ----------
    data
        An ``AnnData`` (raw counts in ``X`` or ``layer``, coordinates in ``obsm[spatial_key]``)
        or a ``SpatialData`` object (its table, chosen with ``table_key``).
    sample_p
        Maximum proportion of cells/spots sampled for model fitting.
    gene_model
        Gene model; ``"nb"`` for counts.
    adj_method
        ``"auto"`` (= ``"logpac"``), ``"logpac"``, ``"pearson"``, ``"medbio"`` or ``"meanbio"``.
    scale_factor
        Sample-specific scaling factor for the adjusted counts.
    df_tps
        Degrees of freedom of the thin-plate spline, length 1 (biology; library size gets
        half) or 2 (biology, library size). Truncated to integers like R's ``as.integer``.
    lambda_a
        Smoothing parameter for the regression coefficients, length 1 or 2 (biology, library
        size). The value used is ``lambda_a * n_cells``.
    batch_key, batch
        Batch to regress out: an ``obs`` column name, or a vector / numeric design matrix
        (see :func:`spanormpy._io.check_batch`). Strings and categoricals are batches; a
        numeric vector is one covariate column (R ``model.matrix`` semantics).
    size_factors
        Precomputed size factors (array, or an ``obs`` column name). If ``None`` they are
        computed (scran-style pooled factors) when a fit is needed, and stored in
        ``obs["spanorm_size_factors"]``. ``obs["size_factors"]`` is never read.
    tol, step_factor, maxit_nb, maxit_psi, maxn_psi
        Fitting controls, as in R.
    overwrite
        Force a refit even if a matching fit is stored. A stored fit is also refitted if
        ``gene_model``, ``df_tps``, ``lambda_a`` or ``batch`` changed.
    backend
        ``"auto"``, ``"cpu"`` or ``"gpu"``.
    n_jobs
        Threads used to normalise gene blocks in parallel (R ``BPPARAM``).
    verbose
        Print progress messages (to stderr, like R's ``message``).
    random_state
        Seed of the numpy generator behind the cell and dispersion subsampling.
    copy
        Return a modified copy instead of modifying ``data`` in place.
    **kwargs
        R's ``...``: ``gpu_mem_budget`` and ``winsor`` are passed to the NB fit; any other
        key is ignored with a ``UserWarning``.

    Returns
    -------
    ``None``, or with ``copy=True`` a modified copy of the same container type (an
    ``AnnData``, or a new ``SpatialData`` whose table is the result; the input is untouched).
    """
    unknown = sorted(set(kwargs) - _FIT_PASSTHROUGH)
    if unknown:
        warnings.warn(f"unrecognised arguments ignored: {', '.join(unknown)}", stacklevel=2)
    passthrough = {k: v for k, v in kwargs.items() if k in _FIT_PASSTHROUGH}
    adata, write_back = resolve_input(data, layer, spatial_key, table_key)
    if copy:
        adata = adata.copy()
    adj_method = _match_arg(adj_method, _ADJ_METHODS, "adj_method")
    gene_model = _match_arg(gene_model, get_gene_models(), "gene_model")
    backend = _match_arg(backend, _BACKENDS, "backend")
    df_tps = np.trunc(np.atleast_1d(np.asarray(df_tps, dtype=float))).astype(int)
    msgfun = _message if verbose else _quiet

    Y = get_counts(adata, layer)
    coords = get_coords(adata, spatial_key)
    ls = _resolve_size_factors(adata, size_factors)
    batch = _resolve_batch(adata, batch_key, batch)

    # ---- .spaNormCore: reuse a stored fit or (re)fit
    existing = load_fit(adata, "spanorm", validate=False)
    # R refits whenever the stored fit's ncells differs from the data (a spot subset of an
    # object carrying a fit). AnnData slices W/sampling with the spots, so the stored ncells
    # is the only trace of the original size. Genes are different: a gene subset reuses the
    # fit (spec Review Focus 2), see docs/r-discrepancies.md.
    fit_ncells = (adata.uns.get("spanorm", {}).get("spanorm", {}) or {}).get("ncells")
    spots_subset = fit_ncells is not None and int(fit_ncells) != adata.n_obs
    refit = True
    if (
        not overwrite
        and existing is not None
        and not spots_subset
        and existing.ngenes == Y.shape[0]
        and existing.ncells == Y.shape[1]
        and existing.gene_model == gene_model
        and match_df_tps(existing.df_tps, df_tps)
        and match_lambda(existing.lambda_a, lambda_a)
        and _same_batch(existing.batch, batch)
    ):
        msgfun("(1/2) Retrieve precomputed SpaNorm model")
        fit = existing
        refit = False
        computed_sf = None
    else:
        msgfun("(1/2) Fitting SpaNorm model")
        fit, computed_sf = fit_spanorm(
            Y,
            coords,
            sample_p,
            gene_model,
            df_tps=df_tps,
            lambda_a=lambda_a,
            batch=batch,
            size_factors=ls,
            msgfun=msgfun,
            rng=np.random.default_rng(random_state),
            backend=backend,
            tol=tol,
            step_factor=step_factor,
            maxit_nb=maxit_nb,
            maxit_psi=maxit_psi,
            maxn_psi=maxn_psi,
            **passthrough,
        )

    msgfun("(2/2) Normalising data")
    if not np.any(np.asarray(fit.wtype) == "biology"):
        raise ValueError("'SpaNorm' fit should have at least one column representing 'biology'")
    adj_fun = get_adjustment_fun(gene_model, adj_method)
    normmat = normalise_blocked(
        adj_fun,
        Y,
        scale_factor,
        fit,
        n_jobs=n_jobs,
        block_size=normalise_block_size(Y),
        backend=backend,
    )

    # ---- write results (a view first becomes an actual AnnData, no warning)
    _as_actual(adata)
    if refit:
        store_fit(adata, fit, "spanorm")
    if computed_sf is not None:
        adata.obs["spanorm_size_factors"] = computed_sf
    elif refit and "spanorm_size_factors" in adata.obs:
        del adata.obs["spanorm_size_factors"]  # stale: this fit used user-supplied factors
    if "logcounts" in adata.layers:
        warnings.warn("'logcounts' exists and will be overwritten", stacklevel=2)
    out = normmat.T
    adata.layers["logcounts"] = sp.csr_matrix(out) if sp.issparse(Y) else np.ascontiguousarray(out)
    adata.uns.setdefault("spanorm", {})["adj_method"] = adj_method
    result = write_back(adata, copy=copy)
    return result if copy else None
