"""Penalised NB Newton solvers (dense, blocked, compact, batched) and batched grams.

Port of ``R/nbSolver.R`` and ``R/nbSolverBatch.R``. The solvers form the
penalised information ``W' diag(w) W + diag(pen)`` and solve it, absorbing a
nested indicator (or per-sample block-diagonal random) block by a Schur
complement. Everything is torch, runs on the device of ``W``, and uses 0-based
column indices (R ``NA`` block ids become ``-1`` internally).

Call contract (preserved from R's returned list of closures)
-----------------------------------------------------------
:func:`nb_newton_solver` returns a :class:`NewtonSolver` whose three callables
mirror R's ``factor`` / ``solve`` / ``xcov``, one gene at a time:

* ``factor(w)``: ``w`` is a length-n weight vector; returns an opaque state
  (:class:`NFac`) holding the penalised information (``info`` for the dense path;
  ``S``, ``B``, ``cvec`` for the nested path; ``S``, ``B``, ``fac``, ``cols`` for the
  blocked path; ``S``, ``B``, ``H``, ``rank`` for a block design), or ``None`` when
  a block of ``C`` is not positive definite.
* ``solve(w_or_state, s)``: Newton step (length-p tensor) for the score ``s`` over
  all columns, or ``None`` on a singular system.
* ``xcov(w_or_state)``: covariance of the dense columns (``px x px`` tensor; the
  full ``p x p`` on the dense path), or ``None`` on a singular system.

:func:`nb_newton_solver_batch` returns a :class:`NewtonSolver` for a block of genes:

* ``factor(wt_block, cell_tile=None)``: ``wt_block`` is ``batch x n``; returns a
  :class:`NFac` with ``S``, ``L`` (lower Cholesky factors), ``ok`` (host bool
  array, one verdict per gene), ``xi``, ``zi`` (0-based numpy column indices) and,
  with a nested block, ``B`` (``batch x G x px``) and ``cvec`` (``batch x G``).
* ``solve(state, Sc)``: ``Sc`` is ``batch x p``; returns the ``batch x p`` steps,
  rows of singular genes are NaN (R: NA).
* ``xcov(state)``: ``batch x px x px`` covariance; singular genes are NaN.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ._backend import as_tensor
from ._block_design import (
    _block_expand,
    _is_block_design,
    _newton_solver_compact,
)


class NFac(SimpleNamespace):
    """Opaque factorisation state (R class ``spiDE_nfac``); fields are attributes."""


@dataclass(frozen=True)
class NewtonSolver:
    """The ``factor`` / ``solve`` / ``xcov`` closures of R's solver list."""

    factor: Callable
    solve: Callable
    xcov: Callable

    def __getitem__(self, key: str) -> Callable:  # R's `solver$solve` / `solver[["solve"]]`
        return getattr(self, key)


def _lu_checked(A: torch.Tensor):
    """LU of ``A`` and its inverse, or ``None`` where R's ``solve`` would error.

    R's ``solve.default`` raises on an exactly singular matrix (LAPACK ``dgesv``
    info > 0) and when the reciprocal condition number is below machine epsilon.
    The 1-norm condition number is exact here (LAPACK ``dgecon`` estimates it).
    """
    LU, piv, info = torch.linalg.lu_factor_ex(A)
    if int(info) != 0:
        return None
    inv = torch.linalg.lu_solve(LU, piv, torch.eye(A.shape[0], dtype=A.dtype, device=A.device))
    rcond = 1.0 / (torch.linalg.matrix_norm(A, ord=1) * torch.linalg.matrix_norm(inv, ord=1))
    if float(rcond) < torch.finfo(torch.float64).eps:
        return None
    return LU, piv, inv


def _solve_or_none(A: torch.Tensor, b: torch.Tensor):
    """``solve(A, b)`` for a vector ``b``, or ``None`` on a singular system."""
    f = _lu_checked(A)
    return None if f is None else torch.linalg.lu_solve(f[0], f[1], b[:, None])[:, 0]


def _inverse_or_none(A: torch.Tensor):
    """``solve(A)`` (the inverse), or ``None`` on a singular system."""
    f = _lu_checked(A)
    return None if f is None else f[2]


def _na_mask(x) -> np.ndarray:
    return np.asarray(pd.isna(np.asarray(x, dtype=object)), dtype=bool)


def _absorb_blocks(nested, p: int) -> np.ndarray:
    """Normalise an absorption specification into a per-column block id.

    Returns an int array of length ``p``: ``-1`` (R ``NA``) for a column that stays
    in the dense block, otherwise the 0-based id of the block of ``C = Z'diag(w)Z``
    the column belongs to (ids ordered like R's ``factor(as.character(.))``).
    """
    if nested is None:
        return np.full(p, -1, dtype=np.int64)
    if isinstance(nested, torch.Tensor):
        nested = nested.cpu().numpy()
    arr = (
        np.asarray(nested)
        if not isinstance(nested, pd.Categorical)
        else np.asarray(nested, dtype=object)
    )
    if arr.ndim != 1 or arr.shape[0] != p:
        raise ValueError("`nested` must have one entry per column of W")
    out = np.full(p, -1, dtype=np.int64)
    if arr.dtype == bool:
        idx = np.nonzero(arr)[0]
        out[idx] = idx  # each absorbed column its own block
        return out
    keep = ~_na_mask(arr)
    if keep.any():
        vals = arr[keep]
        if vals.dtype.kind == "f" and np.all(vals == np.floor(vals)):
            vals = vals.astype(np.int64)
        strs = np.array([str(v) for v in vals])
        _, inv = np.unique(strs, return_inverse=True)
        out[keep] = inv
    return out


def _to_pen(pen, p: int, W) -> torch.Tensor:
    pen = as_tensor(pen, W.device, W.dtype).reshape(-1)
    if pen.numel() == 1:
        pen = pen.expand(p)
    if pen.numel() != p:
        raise ValueError(f"'pen' must have one value per column of W ({p}); {pen.numel()} supplied")
    return pen


def _as_dense(W) -> torch.Tensor:
    if isinstance(W, torch.Tensor):
        return W
    return as_tensor(W, "cpu")


def _newton_solver_blocked(W, pen, blk, xi, zi, X, pen_x, pen_z) -> NewtonSolver:
    """Schur absorption of a random block that is block-diagonal, not diagonal."""
    p = W.shape[1]
    bid = blk[zi]
    cols = [torch.as_tensor(np.nonzero(bid == b)[0], device=W.device) for b in np.unique(bid)]
    xi_t = torch.as_tensor(xi, device=W.device)
    zi_t = torch.as_tensor(zi, device=W.device)
    Z = W[:, zi_t]

    # No cell may load on two blocks (done once per solver).
    hits = torch.zeros(W.shape[0], dtype=torch.long, device=W.device)
    for cc in cols:
        hits = hits + (Z[:, cc].abs().sum(1) > 0).long()
    if int(hits.max()) > 1:
        raise ValueError(
            "the absorbed random-effect columns are not block-orthogonal: "
            f"{int((hits > 1).sum())} cells load on more than one block, so "
            "C = Z'WZ is not block-diagonal and .newtonSolver() cannot absorb it"
        )

    def parts(w):
        w = as_tensor(w, W.device, W.dtype).reshape(-1)
        sw = torch.sqrt(w)[:, None]
        Xw = X * sw
        A = Xw.T @ Xw + torch.diag(pen_x)
        Zw = Z * sw
        B = Xw.T @ Zw  # px x p_z
        # one dense C block per group, Cholesky-factorised once per weight vector
        fac = []
        for cc in cols:
            Zc = Zw[:, cc]
            Cb = Zc.T @ Zc + torch.diag(pen_z[cc])
            L, info = torch.linalg.cholesky_ex(Cb)
            if int(info) != 0:
                return None
            fac.append(L)  # lower factor (R keeps the upper one)
        S = A
        for k, cc in enumerate(cols):
            Bb = B[:, cc]
            S = S - Bb @ torch.cholesky_inverse(fac[k]) @ Bb.T
        return NFac(S=S, B=B, fac=fac, cols=cols)

    def as_fac(x):
        return x if (x is None or isinstance(x, NFac)) else parts(x)

    def cinv(st, v):  # C^-1 v, block by block
        out = torch.zeros_like(v)
        for k, cc in enumerate(st.cols):
            out[cc] = torch.cholesky_solve(v[cc][:, None], st.fac[k])[:, 0]
        return out

    def solve(w, s):
        st = as_fac(w)
        if st is None:
            return None
        s = as_tensor(s, W.device, W.dtype).reshape(-1)
        rhs = s[xi_t] - st.B @ cinv(st, s[zi_t])
        dx = _solve_or_none(st.S, rhs)
        if dx is None:
            return None
        dz = cinv(st, s[zi_t] - st.B.T @ dx)
        out = torch.zeros(p, dtype=W.dtype, device=W.device)
        out[xi_t] = dx
        out[zi_t] = dz
        return out

    def xcov(w):
        st = as_fac(w)
        if st is None:
            return None
        return _inverse_or_none(st.S)

    return NewtonSolver(factor=parts, solve=solve, xcov=xcov)


def _newton_solver(W, pen, nested=None) -> NewtonSolver:
    """See :func:`nb_newton_solver`."""
    if _is_block_design(W):
        absorbs = nested is not None and bool((_absorb_blocks(nested, W.shape[1]) >= 0).any())
        if absorbs:
            raise ValueError(
                "an nbBlockDesign absorbs its groups' blocks by construction; `absorb` must be None"
            )
        return _newton_solver_compact(W, _block_expand(W, pen, "pen"))
    W = _as_dense(W)
    p = W.shape[1]
    pen = _to_pen(pen, p, W)
    blk = _absorb_blocks(nested, p)
    is_nested = blk >= 0
    if not is_nested.any():

        def factor_dense(w):
            w = as_tensor(w, W.device, W.dtype).reshape(-1)
            Ws = W * torch.sqrt(w)[:, None]
            return NFac(info=Ws.T @ Ws + torch.diag(pen))

        def as_fac(x):
            return x if isinstance(x, NFac) else factor_dense(x)

        return NewtonSolver(
            factor=factor_dense,
            solve=lambda w, s: _solve_or_none(
                as_fac(w).info, as_tensor(s, W.device, W.dtype).reshape(-1)
            ),
            xcov=lambda w: _inverse_or_none(as_fac(w).info),
        )

    xi = np.nonzero(~is_nested)[0]
    zi = np.nonzero(is_nested)[0]
    xi_t = torch.as_tensor(xi, device=W.device)
    zi_t = torch.as_tensor(zi, device=W.device)
    X = W[:, xi_t]
    pen_x = pen[xi_t]
    pen_z = pen[zi_t]
    # A block carrying more than one column cannot use the reciprocal path.
    if len(np.unique(blk[zi])) < len(zi):
        return _newton_solver_blocked(W, pen, blk, xi, zi, X, pen_x, pen_z)
    gidx, G = _indicator_groups(W[:, zi_t], ".newtonSolver()")

    def parts(w):
        # gram formed symmetrically, crossprod(X * sqrt(w)), as R does
        w = as_tensor(w, W.device, W.dtype).reshape(-1)
        sw = torch.sqrt(w)[:, None]
        Xw = X * sw
        A = Xw.T @ Xw + torch.diag(pen_x)
        Xw = Xw * sw  # now X * w
        cvec = _segment_sum(w[:, None], gidx, G)[:, 0] + pen_z
        B = _segment_sum(Xw, gidx, G).T  # px x G
        return NFac(S=A - B @ (B.T / cvec[:, None]), B=B, cvec=cvec)

    def as_fac(x):
        return x if isinstance(x, NFac) else parts(x)

    def solve(w, s):
        p_ = as_fac(w)
        s = as_tensor(s, W.device, W.dtype).reshape(-1)
        rhs = s[xi_t] - p_.B @ (s[zi_t] / p_.cvec)
        dx = _solve_or_none(p_.S, rhs)
        if dx is None:
            return None
        dz = (s[zi_t] - p_.B.T @ dx) / p_.cvec
        out = torch.zeros(p, dtype=W.dtype, device=W.device)
        out[xi_t] = dx
        out[zi_t] = dz
        return out

    def xcov(w):
        return _inverse_or_none(as_fac(w).S)

    return NewtonSolver(factor=parts, solve=solve, xcov=xcov)


def _indicator_groups(Zblk: torch.Tensor, who: str):
    """Group index (0-based) of each cell from a 0/1 partition block, with a check.

    Exact only if every cell belongs to exactly one group; the index is recovered
    by ``round()`` of a dot product, not truncation (a float product of 7 can come
    back as 6.9999999).
    """
    rs = Zblk.sum(1)
    if bool(torch.isnan(rs).any()) or float((rs - 1).abs().max()) > 1e-8:
        raise ValueError(
            "the nested random-effect columns are not 0/1 indicators partitioning the cells; "
            f"{who} cannot absorb them"
        )
    G = Zblk.shape[1]
    seq = torch.arange(G, dtype=Zblk.dtype, device=Zblk.device)
    return torch.round(Zblk @ seq).long(), G


def nb_newton_solver(W, pen, absorb=None) -> NewtonSolver:
    """Newton solver for the penalised NB information (R ``nbNewtonSolver``).

    Parameters
    ----------
    W
        cells x p design: a torch tensor (numpy accepted, moved to the CPU), or an
        :class:`~spanormpy._block_design.NBBlockDesign`, whose groups' blocks are
        absorbed by construction (then ``absorb`` must be ``None``).
    pen
        length-p ridge penalty (a single value is recycled; for a block design,
        also one per column of ``[X | Z]``).
    absorb
        ``None`` (dense); a boolean vector over the columns of ``W`` (each marked
        column its own 1x1 block, ``C`` diagonal); or a per-column block id (int, str
        or categorical, ``None``/NaN for a dense column) whose columns sharing an
        id form one dense block of ``C``. All-False or all-NA selects the dense path.

    Returns
    -------
    NewtonSolver
        with callables ``factor(w)``, ``solve(w_or_state, s)`` and
        ``xcov(w_or_state)``; see the module docstring for their shapes. ``solve``
        and ``xcov`` return ``None`` on a singular system.
    """
    return _newton_solver(W, pen, absorb)


def _dense_only(fn: str):
    raise ValueError(
        f"{fn}() takes a dense design matrix, not an nbBlockDesign; use nb_newton_solver() "
        "for a block design, or .to_dense() it"
    )


def nb_newton_solver_batch(W, pen, absorb=None) -> NewtonSolver:
    """Batched Newton solver: one factorisation object for a block of genes.

    ``W`` is a dense cells x p tensor, ``pen`` a length-p penalty and ``absorb`` a
    boolean vector marking the nested indicator columns (1x1 blocks only; a
    grouping with a multi-column block raises). See the module docstring for the
    ``factor`` / ``solve`` / ``xcov`` shapes.
    """
    if _is_block_design(W):
        _dense_only("nb_newton_solver_batch")
    return _newton_solver_batch(W, pen, absorb)


def nb_gram_batch(W, wt_block, penalty_diag=None, backend: str = "cpu", cell_tile=None):
    """Batched per-gene weighted gram ``W' diag(wt_g) W``, shape ``(batch, p, p)``.

    ``wt_block`` is ``batch x cells``; ``penalty_diag`` (length p) is added to every
    slice's diagonal. ``backend`` is unused (kept for R's signature). ``cell_tile``
    accumulates over tiles of cells to bound the ``(batch, tile, p)`` intermediate;
    every tiling gives the same result.
    """
    if _is_block_design(W):
        _dense_only("nb_gram_batch")
    return _gram_batch(W, wt_block, penalty_diag, backend, cell_tile)


def nb_absorb_gram_batch(W, pen, absorb, wt_block, cell_tile=None, parts: bool = False):
    """Batched Schur absorption of a nested indicator block.

    Returns the ``(batch, px, px)`` Schur complements, or with ``parts=True`` a dict
    with ``S``, ``B`` (``batch x G x px``), ``cvec`` (``batch x G``) and 0-based
    ``xi`` / ``zi`` column index arrays. ``absorb`` is a boolean vector over the
    columns of ``W`` marking the 0/1 indicator columns that partition the cells.
    """
    if _is_block_design(W):
        _dense_only("nb_absorb_gram_batch")
    return _absorb_batch(W, pen, absorb, wt_block, cell_tile, parts)


# ---- batched helpers ----------------------------------------------------------


def _tile_starts(n: int, cell_tile):
    tile = n if cell_tile is None else max(1, min(int(cell_tile), n))
    return [(s, min(s + tile, n)) for s in range(0, n, tile)]


def _gram_batch(W, wt_block, penalty_diag=None, backend="cpu", cell_tile=None) -> torch.Tensor:
    W = _as_dense(W)
    wt_block = as_tensor(wt_block, W.device, W.dtype)
    n, _ = W.shape
    info = None
    for s, e in _tile_starts(n, cell_tile):
        # Weighting both factors by sqrt(wt) keeps each slice exactly symmetric,
        # which the Cholesky downstream relies on.
        Wg = torch.sqrt(wt_block[:, s:e])[:, :, None] * W[s:e][None]
        part = Wg.transpose(1, 2) @ Wg
        info = part if info is None else info + part
    if penalty_diag is not None:
        info = info + torch.diag(as_tensor(penalty_diag, W.device, W.dtype).reshape(-1))[None]
    return info


def _segment_sum(M: torch.Tensor, gidx: torch.Tensor, G: int) -> torch.Tensor:
    """Sum the rows of ``M`` within groups; absent groups come back as zero rows."""
    out = torch.zeros(G, M.shape[1], dtype=M.dtype, device=M.device)
    return out.index_add(0, gidx, M)


def _absorb_batch(W, pen, nested, wt_block, cell_tile=None, parts=False):
    W = _as_dense(W)
    wt_block = as_tensor(wt_block, W.device, W.dtype)
    n, p = W.shape
    pen = _to_pen(pen, p, W)
    nested = np.zeros(p, dtype=bool) if nested is None else np.asarray(nested, dtype=bool)
    xi = np.nonzero(~nested)[0]
    zi = np.nonzero(nested)[0]
    xi_t = torch.as_tensor(xi, device=W.device)
    zi_t = torch.as_tensor(zi, device=W.device)
    px, G, b = len(xi), len(zi), wt_block.shape[0]
    pen_x, pen_z = pen[xi_t], pen[zi_t]
    X = W[:, xi_t]
    gidx, _ = _indicator_groups(W[:, zi_t], ".absorbBatch()") if G else (None, 0)
    A = torch.zeros(b, px, px, dtype=W.dtype, device=W.device)
    Bacc = torch.zeros(b, G, px, dtype=W.dtype, device=W.device)
    cvec = torch.zeros(b, G, dtype=W.dtype, device=W.device)
    for s, e in _tile_starts(n, cell_tile):
        Xt, wt = X[s:e], wt_block[:, s:e]  # wt: batch x tile
        A = A + _gram_batch(Xt, wt)
        if G:
            gt = gidx[s:e]
            cvec = cvec + _segment_sum(wt.T, gt, G).T
            # (batch, tile, px): bounded by the tile
            Bacc = Bacc.index_add(1, gt, wt[:, :, None] * Xt[None])
    A = A + torch.diag(pen_x)[None]
    cvec = cvec + pen_z[None]
    S = A - Bacc.transpose(1, 2) @ (Bacc / cvec[:, :, None])
    if not parts:
        return S
    return {"S": S, "B": Bacc, "cvec": cvec, "xi": xi, "zi": zi}


def _chol_batch(S: torch.Tensor):
    """Per-slice Cholesky of a ``(batch, p, p)`` stack with a per-slice verdict.

    A singular slice must not fail its neighbours. Returns ``(L, ok)``: ``L`` the
    lower factors with ``S = L L'`` and ``ok`` a host bool array (a slice is ok when
    the factorisation succeeded and is finite).
    """
    L, info = torch.linalg.cholesky_ex(S)
    ok = (info == 0) & torch.isfinite(L).flatten(1).all(1)
    return L, ok.cpu().numpy()


def _safe_factors(L: torch.Tensor, ok: np.ndarray):
    """Replace failed slices by the identity so batched LAPACK calls cannot raise."""
    if ok.all():
        return L, None
    bad = torch.as_tensor(~ok, device=L.device)
    eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
    return torch.where(bad[:, None, None], eye, L), bad


def _chol_solve_batch(L: torch.Tensor, ok: np.ndarray, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a batch of Cholesky-factorised systems; NaN rows where the slice is singular."""
    L, bad = _safe_factors(L, ok)
    out = torch.cholesky_solve(rhs[:, :, None], L)[:, :, 0]
    if bad is not None:
        out = out.masked_fill(bad[:, None], math.nan)
    return out


def _subset_state(st: NFac, ii) -> NFac:
    """Restrict a batched factorisation state to a subset of its genes (``xi``/``zi`` shared)."""
    ii = np.asarray(ii)
    idx = torch.as_tensor(ii, device=st.S.device)

    def sub(x):
        return None if x is None else x[idx]

    return NFac(
        S=sub(st.S), L=sub(st.L), B=sub(st.B), cvec=sub(st.cvec), ok=st.ok[ii], xi=st.xi, zi=st.zi
    )


def _newton_solver_batch(W, pen, nested=None) -> NewtonSolver:
    W = _as_dense(W)
    p = W.shape[1]
    pen = _to_pen(pen, p, W)
    blk = _absorb_blocks(nested, p)
    kept = blk[blk >= 0]
    if len(np.unique(kept)) < len(kept):
        raise ValueError(
            "the batched Newton solver absorbs 1x1 blocks only (one column per group, C "
            "diagonal); this grouping has a block with more than one column, which needs a "
            "per-block Cholesky. Use nb_newton_solver() on the CPU, or pass a boolean `absorb`."
        )
    is_nested = blk >= 0
    has_nested = bool(is_nested.any())
    xi = np.nonzero(~is_nested)[0]
    zi = np.nonzero(is_nested)[0]
    xi_t = torch.as_tensor(xi, device=W.device)
    zi_t = torch.as_tensor(zi, device=W.device)

    def factor(wt_block, cell_tile=None):
        if not has_nested:
            S = _gram_batch(W, wt_block, penalty_diag=pen, cell_tile=cell_tile)
            L, ok = _chol_batch(S)
            return NFac(S=S, L=L, ok=ok, xi=xi, zi=zi, B=None, cvec=None)
        pr = _absorb_batch(W, pen, is_nested, wt_block, cell_tile=cell_tile, parts=True)
        L, ok = _chol_batch(pr["S"])
        return NFac(L=L, ok=ok, **pr)

    def solve(state, Sc):
        Sc = as_tensor(Sc, W.device, W.dtype)
        if not len(state.zi):
            return _chol_solve_batch(state.L, state.ok, Sc)
        s_x, s_z = Sc[:, xi_t], Sc[:, zi_t]
        sc = s_z / state.cvec
        rhs = s_x - (state.B.transpose(1, 2) @ sc[:, :, None])[:, :, 0]
        dx = _chol_solve_batch(state.L, state.ok, rhs)
        dz = (s_z - (state.B @ dx[:, :, None])[:, :, 0]) / state.cvec
        out = torch.zeros_like(Sc)
        out[:, xi_t] = dx
        out[:, zi_t] = dz
        return out

    def xcov(state):
        L, bad = _safe_factors(state.L, state.ok)
        out = torch.cholesky_inverse(L)
        return out if bad is None else out.masked_fill(bad[:, None, None], math.nan)

    return NewtonSolver(factor=factor, solve=solve, xcov=xcov)
