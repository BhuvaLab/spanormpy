"""Negative-binomial model fitting core.

Port of ``R/fitSpaNormNB.R`` (everything except the ``normalise*`` functions,
which live in ``_adjust.py``): ``fitNB``, ``fitSpaNormNB``, ``fitNBGivenPsi``,
``calculateMu`` and the gene-blocked IRLS helpers.

All arrays use R's orientation (genes x cells for ``Y``/``offset``, cells x
covariates for ``W``). As in R, the counts ``Ysub`` stay host-resident for the
whole fit and only one gene-block of the big genes x cells intermediates is
moved to the compute device at a time; ``alpha``/``gmean``/``psi`` are small
and live on the device. edgeR's dispersion estimation always runs on host
numpy arrays.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import scipy.sparse as sp
import torch

from ._backend import (
    DEFAULT_WINSOR,
    as_tensor,
    check_gpu,
    check_winsor,
    gene_block_count,
    gene_block_indices,
    get_gpu_memory_budget,
    has_bad_values,
    resolve_device,
    winsorise_cols,
)
from ._edger import estimate_disp_tagwise
from ._nbdist import dnbinom
from ._settings import _match_arg

_MAD_CONSTANT = 1.4826


def _message(msg: str) -> None:
    """Default ``msgfun`` (R's ``message``): write to stderr."""
    print(msg, file=sys.stderr)


def _quiet(msg: str) -> None:
    return None


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
def _check_psi(psi, ngenes: int, argname: str = "psi"):
    """NULL, or a finite strictly positive length-``ngenes`` numeric vector."""
    if psi is None:
        return None
    if isinstance(psi, torch.Tensor):
        psi = psi.detach().cpu().numpy()
    psi = np.asarray(psi).ravel()
    if psi.dtype.kind not in "fiu" or psi.size != ngenes:
        raise ValueError(
            f"'{argname}' should be a numeric vector of length {ngenes} "
            "(one dispersion per gene), or NULL"
        )
    psi = psi.astype(float)
    if not np.all(np.isfinite(psi)) or np.any(psi <= 0):
        raise ValueError(f"'{argname}' should contain finite, strictly positive values only")
    return psi


def _check_offset(
    offset, ngenes: int, ncells: int, argname: str = "offset", allow_tensor: bool = False
):
    """NULL, or a finite numeric genes x cells matrix conforming to the counts."""
    if offset is None:
        return None
    shape = tuple(offset.shape) if hasattr(offset, "shape") else np.shape(offset)
    if len(shape) != 2 or shape[0] != ngenes or shape[1] != ncells:
        raise ValueError(
            f"'{argname}' should be a {ngenes} x {ncells} (genes x cells) matrix, or NULL"
        )
    if isinstance(offset, torch.Tensor):
        if allow_tensor:
            return offset
        raise ValueError(f"'{argname}' should be a base R matrix, not a torch tensor")
    if sp.issparse(offset):
        offset = offset.toarray()
    offset = np.asarray(offset)
    if offset.dtype.kind not in "fiu":
        raise ValueError(f"'{argname}' should contain finite numeric values only")
    offset = offset.astype(float, copy=False)
    # cheap screen first (a sum is non-finite whenever any element is)
    if not math.isfinite(offset.sum()) and not np.all(np.isfinite(offset)):
        raise ValueError(f"'{argname}' should contain finite numeric values only")
    return offset


def _check_int_positive(value, name: str) -> None:
    if value <= 0:
        raise ValueError(f"'{name}' should be greater than 0")
    if value - int(value) != 0:
        raise ValueError(f"'{name}' should be integers")


def _to_host(x) -> torch.Tensor:
    """Dense float64 CPU tensor (sparse and numpy accepted)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.float64)
    if sp.issparse(x):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _sl(blk) -> slice:
    """Contiguous gene block as a slice (blocks are contiguous by construction)."""
    blk = np.asarray(blk)
    return slice(int(blk[0]), int(blk[-1]) + 1)


def _nblocks_for(ngenes, ncells, ncov, backend, gpu_mem_budget) -> int:
    """R's ``geneBlockCount(..., budget.bytes = getGPUMemoryBudget(gpu.mem.budget))``.

    The budget is evaluated lazily, as in R: it is never touched (nor validated)
    when ``backend == "cpu"`` or no accelerator is in use.
    """
    backend = _match_arg(backend, ("auto", "cpu", "gpu"), "backend")
    if backend == "cpu" or not check_gpu():
        return 1
    return gene_block_count(
        ngenes, ncells, ncov, backend=backend, budget_bytes=get_gpu_memory_budget(gpu_mem_budget)
    )


# --------------------------------------------------------------------------
# gene-blocked helpers
# --------------------------------------------------------------------------
def _block_offset(offset, sl: slice, dev):
    """One gene-block's slice of the (host) offset, on the device (None if no offset)."""
    if offset is None:
        return None
    return as_tensor(offset[sl], dev)


def _block_lmu_hat(alpha, gmean, sl, Wsub, offset_blk=None):
    """gmean[blk] + alpha[blk] %*% t(Wsub) (+ offset[blk, ])."""
    lmu = gmean[sl].unsqueeze(1) + alpha[sl] @ Wsub.T
    return lmu if offset_blk is None else lmu + offset_blk


def _block_working_response(Ysub_blk, lmu_hat_blk, step):
    zr = (Ysub_blk + 0.01) / (torch.exp(lmu_hat_blk) + 0.01) - 1
    return lmu_hat_blk + step.unsqueeze(0) * zr


def _blocked_nb_loglik(alpha, gmean, Ysub, Wsub, psi, blocks, offset=None):
    """Per-cell NB log-likelihood summed over genes, one gene block at a time."""
    dev = Wsub.device
    total = None
    for blk in blocks:
        sl = _sl(blk)
        lmu = _block_lmu_hat(alpha, gmean, sl, Wsub, _block_offset(offset, sl, dev))
        Y = as_tensor(Ysub[sl], dev, Wsub.dtype)
        ll = dnbinom(Y, mu=torch.exp(lmu), size=(1.0 / psi[sl]).unsqueeze(1), log=True).sum(dim=0)
        total = ll if total is None else total + ll
    return total


def _blocked_offs_psi(alpha, Wsub_psi, blocks, offset_psi=None):
    """``tcrossprod(alpha, Wsub.psi)`` (+ fixed offset) assembled block by block (host numpy)."""
    out = np.zeros((alpha.shape[0], Wsub_psi.shape[0]))
    for blk in blocks:
        sl = _sl(blk)
        o = alpha[sl] @ Wsub_psi.T
        o = o.detach().cpu().numpy().astype(float)
        if offset_psi is not None:
            o = o + offset_psi[sl]
        out[sl] = o
    return out


def _irls_blocked_alpha_numerator(alpha, gmean, Ysub, Wsub, step, Wsub_wt, blocks, offset=None):
    """Pass B: the alpha-update numerator ``b`` (genes x ncov)."""
    dev = Wsub.device
    b = torch.zeros((alpha.shape[0], Wsub.shape[1]), dtype=Wsub.dtype, device=dev)
    for blk in blocks:
        sl = _sl(blk)
        O = _block_offset(offset, sl, dev)
        lmu = _block_lmu_hat(alpha, gmean, sl, Wsub, O)
        Y = as_tensor(Ysub[sl], dev, Wsub.dtype)
        Z = _block_working_response(Y, lmu, step)
        Zc = Z - gmean[sl].unsqueeze(1)
        # only the part of the working response NOT explained by the fixed offset is regressed on W
        if O is not None:
            Zc = Zc - O
        b[sl] = Zc @ Wsub_wt
    return b


def _invert_mat(mat: torch.Tensor) -> torch.Tensor:
    """Cholesky inverse, general inverse when not SPD (R ``invert_mat``)."""
    work = mat.cpu() if mat.device.type == "mps" else mat
    try:
        res = torch.cholesky_inverse(torch.linalg.cholesky(work))
    except Exception:  # noqa: BLE001 - mirror R's tryCatch fallback to solve()
        res = torch.linalg.inv(work)
    return res.to(device=mat.device, dtype=mat.dtype)


def _irls_solve_alpha_from_b(b, Wsub, Wsub_wt, lambda_a, is_spanorm):
    """Small-matrix solve of the new alpha from the assembled numerator ``b``."""
    if is_spanorm:
        ainv = _invert_mat(Wsub_wt.T @ Wsub)
        alpha_unreg = b @ ainv
        a1_mean = alpha_unreg[:, 0].mean()
        W_first = Wsub[:, :1]
        W_rest = Wsub[:, 1:]
        W_rest_wt = Wsub_wt[:, 1:]
        const_row = (W_first.T @ W_rest_wt).reshape(-1)
        b_other = b[:, 1:] - a1_mean * const_row.unsqueeze(0)
        arest_inv = _invert_mat(W_rest_wt.T @ W_rest + lambda_a)
        alpha_other = b_other @ arest_inv
        return torch.cat([a1_mean.expand(b.shape[0]).unsqueeze(1), alpha_other], dim=1)
    areg_inv = _invert_mat(Wsub_wt.T @ Wsub + lambda_a)
    return b @ areg_inv


def _irls_blocked_wt_cell(alpha, gmean, Ysub, Wsub, psi, step, is_spanorm, blocks, offset=None):
    """Pass A: clipped cell weights plus (is_spanorm) the gmean-fold reductions."""
    dev = Wsub.device
    ngenes, ncov = alpha.shape[0], Wsub.shape[1]
    wtcell_num = None
    rowsum_Zsig = rowsum_sig = M = None
    if is_spanorm:
        rowsum_Zsig = torch.zeros(ngenes, dtype=Wsub.dtype, device=dev)
        rowsum_sig = torch.zeros(ngenes, dtype=Wsub.dtype, device=dev)
        M = torch.zeros((ngenes, ncov), dtype=Wsub.dtype, device=dev)
    for blk in blocks:
        sl = _sl(blk)
        O = _block_offset(offset, sl, dev)
        lmu = _block_lmu_hat(alpha, gmean, sl, Wsub, O)
        sig_inv = 1.0 / (psi[sl].unsqueeze(1) + torch.exp(-lmu))
        colsum = sig_inv.sum(dim=0)
        wtcell_num = colsum if wtcell_num is None else wtcell_num + colsum
        if is_spanorm:
            Y = as_tensor(Ysub[sl], dev, Wsub.dtype)
            Z = _block_working_response(Y, lmu, step)
            # the offset is a fixed part of the linear predictor, so it comes off the working response
            if O is not None:
                Z = Z - O
            rowsum_Zsig[sl] = (Z * sig_inv).sum(dim=1)
            M[sl] = sig_inv @ Wsub
            rowsum_sig[sl] = sig_inv.sum(dim=1)
    wt_cell = wtcell_num / ngenes
    # R: pmin(wt.cell, quantile(wt.cell, probs = 0.98)) (type 7 == numpy "linear")
    q = float(np.quantile(wt_cell.detach().cpu().numpy(), 0.98))
    wt_cell = torch.clamp(wt_cell, max=q)
    return wt_cell, rowsum_Zsig, M, rowsum_sig


def _fit_nb_given_psi_iter(
    alpha, gmean, Ysub, Wsub, psi, step, lambda_a, is_spanorm, winsor, blocks, offset=None
):
    """One full inner IRLS iteration (Pass A, small stage, Pass B, finalise, Pass C)."""
    wt_cell, rowsum_Zsig, M, rowsum_sig = _irls_blocked_wt_cell(
        alpha, gmean, Ysub, Wsub, psi, step, is_spanorm, blocks, offset
    )
    Wsub_wt = wt_cell.unsqueeze(1) * Wsub
    b = _irls_blocked_alpha_numerator(alpha, gmean, Ysub, Wsub, step, Wsub_wt, blocks, offset)

    alpha_new = _irls_solve_alpha_from_b(b, Wsub, Wsub_wt, lambda_a, is_spanorm)
    if has_bad_values(alpha_new):
        alpha_new = alpha
    alpha_new = winsorise_cols(alpha_new, k=winsor)

    if is_spanorm:
        gmean_new = (rowsum_Zsig - (alpha_new * M).sum(dim=1)) / rowsum_sig
    else:
        gmean_new = gmean
    loglik_new = _blocked_nb_loglik(alpha_new, gmean_new, Ysub, Wsub, psi, blocks, offset)
    return alpha_new, gmean_new, loglik_new


# --------------------------------------------------------------------------
# public fitting functions
# --------------------------------------------------------------------------
def fit_nb_given_psi(
    Ysub,
    Wsub,
    psi,
    lambda_a,
    gmean=None,
    alpha=None,
    offset=None,
    step_factor: float = 0.5,
    maxit_nb: int = 50,
    tol: float = 1e-4,
    loglik=None,
    backend: str = "auto",
    is_spanorm: bool = False,
    winsor: float = DEFAULT_WINSOR,
    blocks=None,
    msgfun=_message,
) -> dict:
    """Inner IRLS fit of the NB mean model at fixed dispersions (R ``fitNBGivenPsi``).

    Returns ``gmean``, ``alpha`` (numpy), ``loglik`` (per-cell, numpy) and
    ``loglik_iter`` (the log-likelihood trajectory, newest first as in R).
    ``blocks`` is a list of contiguous 0-based gene-index arrays (default: from
    the memory budget).
    """
    dev = resolve_device(backend)
    lam = np.atleast_1d(np.asarray(lambda_a, dtype=float))
    if np.any(lam < 0):
        raise ValueError("'lambda_a' should be positive")
    Wsub = as_tensor(Wsub, dev)
    ncol_w = Wsub.shape[1]
    npen = ncol_w - 1 if is_spanorm else ncol_w
    if lam.size not in (1, npen):
        raise ValueError(
            "'lambda_a' should either be a single value or a vector of length equal "
            "to the number of penalised columns in W"
        )
    if lam.size == 1:
        lam = np.repeat(lam, npen)
    if step_factor <= 0 or step_factor >= 1:
        raise ValueError("'step_factor' should be in the interval (0,1)")
    _check_int_positive(maxit_nb, "maxit_nb")

    Ysub = _to_host(Ysub)
    ngenes, nsub = Ysub.shape
    offset = _check_offset(offset, ngenes, nsub)
    lambda_mat = torch.diag(as_tensor(lam, dev, Wsub.dtype))
    psi = as_tensor(psi, dev, Wsub.dtype)

    if gmean is None:
        gmean = (
            torch.log(Ysub + 1).mean(dim=1)
            if is_spanorm
            else torch.zeros(ngenes, dtype=torch.float64)
        )
    gmean = as_tensor(gmean, dev, Wsub.dtype)
    if alpha is None:
        alpha = torch.zeros((ngenes, ncol_w), dtype=Wsub.dtype, device=dev)
        alpha[:, 0] = 1
    alpha = as_tensor(alpha, dev, Wsub.dtype)

    if blocks is None:
        blocks = gene_block_indices(ngenes, _nblocks_for(ngenes, nsub, ncol_w, backend, None))

    step = torch.ones(nsub, dtype=Wsub.dtype, device=dev)
    if loglik is None:
        loglik = _blocked_nb_loglik(alpha, gmean, Ysub, Wsub, psi, blocks, offset)
    loglik = as_tensor(loglik, dev, Wsub.dtype)

    conv = False
    logl_beta = [float(loglik.sum())]  # stack; newest first
    it = 1
    halving = 0

    while not conv:
        msgfun(f"iter:{it:3d}, log-likelihood: {logl_beta[0]:f}")
        best_gmean, best_a = gmean, alpha

        alpha, gmean, loglik_tmp = _fit_nb_given_psi_iter(
            alpha, gmean, Ysub, Wsub, psi, step, lambda_mat, is_spanorm, winsor, blocks, offset
        )

        s_old, s_new = float(loglik.sum()), float(loglik_tmp.sum())
        degener = (s_old > s_new) or math.isnan(s_old) or math.isnan(s_new)
        if degener:
            # alter step size of the cells whose likelihood dropped
            check = loglik > loglik_tmp
            step = torch.where(check, step * step_factor, step)
            # revert to previous best
            gmean, alpha = best_gmean, best_a
            halving += 1
            if halving >= 3:
                loglik_tmp = loglik
                degener = not degener

        if not degener:
            loglik = loglik_tmp
            logl_beta.insert(0, float(loglik.sum()))
            it += 1
            halving = 0

        conv_logl = False
        if it > 2:
            conv_logl = (logl_beta[0] - logl_beta[1]) / abs(logl_beta[1]) < tol
        conv = conv_logl or it > maxit_nb

    if it > maxit_nb:
        msgfun(f"iter:{it:3d}, log-likelihood: {logl_beta[0]:f} (did not converged)")
    else:
        msgfun(f"iter:{it:3d}, log-likelihood: {logl_beta[0]:f} (converged)")

    return {
        "gmean": gmean.detach().cpu().numpy().astype(float),
        "alpha": alpha.detach().cpu().numpy().astype(float),
        "loglik": loglik.detach().cpu().numpy().astype(float),
        "loglik_iter": np.array(logl_beta),
    }


def _sample_psi_idx(idx: np.ndarray, maxn_psi: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the dispersion-estimation subset (R's ``sample.int(nsub, size = nsub.psi)``).

    ``idx`` is the boolean mask of cells used for the fit; the returned boolean
    mask is over those ``sum(idx)`` fitted cells (R's ``psi.idx``). This is the
    only place the subset is drawn, so tests can monkeypatch it with R's draw.
    """
    nsub = int(np.sum(idx))
    nsub_psi = min(int(maxn_psi), nsub)
    out = np.zeros(nsub, dtype=bool)
    out[rng.choice(nsub, size=nsub_psi, replace=False)] = True
    return out


def fit_spanorm_nb(
    Y,
    W,
    idx,
    maxit_psi: float = 25,
    tol: float = 1e-4,
    maxn_psi: int = 500,
    offset=None,
    psi=None,
    is_spanorm: bool = False,
    winsor: float = DEFAULT_WINSOR,
    backend: str = "auto",
    gpu_mem_budget=None,
    msgfun=_message,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> dict:
    """Outer dispersion / inner IRLS loop (R ``fitSpaNormNB``).

    ``idx`` is a boolean mask over cells. Extra ``kwargs`` (``lambda_a``,
    ``maxit_nb``, ``step_factor``) go to :func:`fit_nb_given_psi`. Returns
    ``gmean``, ``alpha``, ``psi``, ``sampling`` (array of ``"all"``/``"glm"``/
    ``"dispersion"``) and ``loglik`` (outer-iteration trajectory; a scalar when
    ``psi`` is supplied).
    """
    backend = _match_arg(backend, ("auto", "cpu", "gpu"), "backend")
    _check_int_positive(maxit_psi, "maxit_psi")
    if msgfun is None:
        msgfun = _message

    def msgfun_nb(prefix):
        return lambda msg: msgfun(prefix + msg)

    dev = resolve_device(backend)
    idx = np.asarray(idx, dtype=bool)
    ngenes_all, ncells_all = Y.shape
    Ysub = _to_host(Y[:, idx])
    offset = _check_offset(offset, ngenes_all, ncells_all)
    Osub = None if offset is None else offset[:, idx]
    Wsub = as_tensor(np.asarray(W)[idx, :], dev)
    nW = Wsub.shape[1]
    nsub = int(idx.sum())
    ngenes = Ysub.shape[0]
    dtype = Wsub.dtype

    nblocks = _nblocks_for(ngenes, nsub, nW, backend, gpu_mem_budget)
    blocks = gene_block_indices(ngenes, nblocks)

    gmean = (
        torch.log(Ysub + 1).mean(dim=1).to(device=dev, dtype=dtype)
        if is_spanorm
        else torch.zeros(ngenes, dtype=dtype, device=dev)
    )
    alpha = torch.zeros((ngenes, nW), dtype=dtype, device=dev)
    alpha[:, 0] = 1

    psi = _check_psi(psi, ngenes_all)
    if psi is not None:
        msgfun("using supplied gene-wise dispersions")
        psi_t = as_tensor(psi, dev, dtype)
        loglik = _blocked_nb_loglik(alpha, gmean, Ysub, Wsub, psi_t, blocks, Osub)
        msgfun(f"initial log-likelihood: {float(loglik.sum()):f}")
        msgfun("fitting NB model")
        fit_nb = fit_nb_given_psi(
            Ysub,
            Wsub,
            psi,
            gmean=gmean,
            alpha=alpha,
            offset=Osub,
            loglik=loglik,
            is_spanorm=is_spanorm,
            winsor=winsor,
            backend=backend,
            blocks=blocks,
            msgfun=msgfun_nb(""),
            **kwargs,
        )
        msgfun(f"final log-likelihood: {float(fit_nb['loglik'].sum()):f}")
        sampling = np.where(idx, "glm", "all")
        return {
            "gmean": np.asarray(fit_nb["gmean"], dtype=float),
            "alpha": np.asarray(fit_nb["alpha"], dtype=float),
            "psi": psi.copy(),
            "sampling": sampling,
            "loglik": float(fit_nb["loglik"].sum()),
        }

    # subset of cells for dispersion estimation (speed-up)
    nsub_psi = min(maxn_psi, nsub)
    if rng is None:
        rng = np.random.default_rng()
    psi_idx = np.asarray(_sample_psi_idx(idx, maxn_psi, rng), dtype=bool)
    psi_cols = np.flatnonzero(psi_idx)
    Ysub_psi = Ysub[:, psi_cols].numpy()
    Wsub_psi = Wsub[torch.as_tensor(psi_cols, device=dev)]
    Osub_psi = None if Osub is None else Osub[:, psi_cols]
    nblocks_psi = _nblocks_for(ngenes, nsub_psi, nW, backend, gpu_mem_budget)
    blocks_psi = gene_block_indices(ngenes, nblocks_psi)
    psi = np.zeros(ngenes)

    conv = False
    logl_psi: list[float] = []
    it = 1

    code = idx.astype(int)  # 0 = all, 1 = glm, 2 = dispersion
    fitted_pos = np.flatnonzero(idx)
    code[fitted_pos[psi_idx]] = 2
    sampling = np.array(["all", "glm", "dispersion"])[code]

    while not conv:
        msgfun(f"iter:{it:3d}, estimating gene-wise dispersion")
        offs_psi = _blocked_offs_psi(alpha, Wsub_psi, blocks_psi, Osub_psi)
        try:
            psi_tmp = np.asarray(
                estimate_disp_tagwise(Ysub_psi, offs_psi, robust=True), dtype=float
            )
        except Exception:  # noqa: BLE001 - R: tryCatch(error = ) catches any error
            psi_tmp = np.full(ngenes, np.nan)
        valid = ~np.isnan(psi_tmp) & ~np.isinf(psi_tmp)
        psi[valid] = psi_tmp[valid]
        psi_t = as_tensor(psi, dev, dtype)

        loglik = _blocked_nb_loglik(alpha, gmean, Ysub, Wsub, psi_t, blocks, Osub)
        msgfun(f"iter:{it:3d}, log-likelihood: {float(loglik.sum()):f}")

        msgfun(f"iter:{it:3d}, fitting NB model")
        fit_nb = fit_nb_given_psi(
            Ysub,
            Wsub,
            psi,
            gmean=gmean,
            alpha=alpha,
            offset=Osub,
            loglik=loglik,
            is_spanorm=is_spanorm,
            winsor=winsor,
            backend=backend,
            blocks=blocks,
            msgfun=msgfun_nb(f"iter:{it:3d}, "),
            **kwargs,
        )
        gmean = as_tensor(fit_nb["gmean"], dev, dtype)
        alpha = as_tensor(fit_nb["alpha"], dev, dtype)
        loglik = fit_nb["loglik"]

        logl_psi.insert(0, float(np.sum(loglik)))
        conv_logl = False
        it += 1
        if it > 2:
            conv_logl = (logl_psi[0] - logl_psi[1]) / abs(logl_psi[1]) < tol
        conv = conv_logl or it > maxit_psi

    if it > maxit_psi:
        msgfun(f"iter:{it:3d}, log-likelihood: {logl_psi[0]:f} (did not converged)")
    else:
        msgfun(f"iter:{it:3d}, log-likelihood: {logl_psi[0]:f} (converged)")

    return {
        "gmean": gmean.detach().cpu().numpy().astype(float),
        "alpha": alpha.detach().cpu().numpy().astype(float),
        "psi": psi,
        "sampling": sampling,
        "loglik": np.array(logl_psi[::-1]),
    }


def fit_nb(
    Y,
    W,
    idx=None,
    lambda_a=0.0,
    winsor: float = DEFAULT_WINSOR,
    maxit_psi: float = 25,
    maxit_nb: int = 50,
    tol: float = 1e-4,
    offset=None,
    psi=None,
    backend: str = "auto",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Fit a per-gene NB GLM over an arbitrary design (R ``fitNB``).

    ``log(mu) = W @ alpha.T (+ offset)`` with no built-in intercept (``gmean``
    is zero). ``Y`` is genes x cells, ``W`` cells x covariates. Returns
    ``alpha``, ``gmean``, ``psi``, ``loglik``, ``sampling``.

    Examples
    --------
    >>> import numpy as np
    >>> import spanormpy as sn
    >>> rng = np.random.default_rng(1)
    >>> Y = rng.poisson(5, size=(20, 50))  # counts, genes x cells
    >>> W = np.column_stack([np.ones(50), np.linspace(-1, 1, 50)])  # intercept + one covariate
    >>> fit = sn.fit_nb(Y, W, verbose=False, backend="cpu")
    >>> fit["alpha"].shape  # genes x covariates

    >>> fit["psi"][:3]  # dispersions
    """
    backend = _match_arg(backend, ("auto", "cpu", "gpu"), "backend")
    if idx is None:
        idx = np.ones(Y.shape[1], dtype=bool)
    return fit_spanorm_nb(
        Y,
        W,
        idx,
        lambda_a=lambda_a,
        is_spanorm=False,
        winsor=winsor,
        maxit_psi=maxit_psi,
        maxit_nb=maxit_nb,
        tol=tol,
        offset=offset,
        psi=psi,
        backend=backend,
        msgfun=_message if verbose else _quiet,
        **kwargs,
    )


def _winsorise_rows(lmu: torch.Tensor, k: float) -> torch.Tensor:
    """Upper-tail winsorise each row at ``median + k * MAD`` (matrixStats::rowMads)."""
    n = lmu.shape[1]

    def med(x):
        s = torch.sort(x, dim=1).values
        return 0.5 * (s[:, (n - 1) // 2] + s[:, n // 2])

    m = med(lmu)
    mad = _MAD_CONSTANT * med((lmu - m.unsqueeze(1)).abs())
    return torch.minimum(lmu, (m + k * mad).unsqueeze(1))


def calculate_mu(
    gmean, alpha, W, winsor: float = DEFAULT_WINSOR, backend: str = "cpu", offset=None
):
    """Fitted means ``exp(gmean + alpha @ W.T + offset)`` with optional row winsorisation.

    Returns a numpy array when ``backend`` resolves to the CPU and a torch
    tensor on the accelerator otherwise (R's ``calculateMu``).

    Examples
    --------
    >>> import numpy as np
    >>> import spanormpy as sn
    >>> rng = np.random.default_rng(1)
    >>> Y = rng.poisson(5, size=(20, 50))  # counts, genes x cells
    >>> W = np.column_stack([np.ones(50), np.linspace(-1, 1, 50)])  # intercept + one covariate
    >>> fit = sn.fit_nb(Y, W, verbose=False, backend="cpu")
    >>> mu = sn.calculate_mu(fit["gmean"], fit["alpha"], W)
    >>> mu.shape  # genes x cells
    """
    check_winsor(winsor)
    backend = _match_arg(backend, ("auto", "cpu", "gpu"), "backend")
    dev = resolve_device(backend)
    alpha_t = as_tensor(alpha, dev)
    ng = alpha_t.shape[0]
    Wn = W if isinstance(W, torch.Tensor) else np.asarray(W)
    offset = _check_offset(offset, ng, Wn.shape[0], allow_tensor=True)
    W_t = as_tensor(Wn, dev)
    gm = np.asarray(gmean, dtype=float).ravel() if not isinstance(gmean, torch.Tensor) else gmean
    gm = as_tensor(gm, dev)
    if gm.numel() == 1 and ng != 1:
        gm = gm.expand(ng)
    elif gm.numel() != ng:
        raise ValueError("Length of vec does not match n.")

    lmu = gm.unsqueeze(1) + alpha_t @ W_t.T
    if offset is not None:
        lmu = lmu + as_tensor(offset, dev)
    if not math.isinf(winsor):
        lmu = _winsorise_rows(lmu, winsor)
    mu = torch.exp(lmu)
    return mu if dev.type != "cpu" else mu.numpy()
