"""Compact per-group block design and its Newton solver.

Port of ``R/nbBlockDesign.R``. The design

    W = [X | Z_1 | Z_2 | ... | Z_G]

has dense columns ``X`` and, for every group g, a copy of the ``q`` columns of
``Z`` that is non-zero only on the cells of group g. It is held as the two
``n x ...`` matrices and a group id per cell, never as the dense
``n x (p_x + G q)`` matrix. All columns are 0-based here; everything is torch
and runs on the device of ``X``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ._backend import as_tensor

# Above this many genes the batched products run group by group through dense
# matmuls rather than through a gather over cells (R's BLOCK_BATCH_DENSE_MIN).
BLOCK_BATCH_DENSE_MIN = 5


def _colnames(M):
    """Column names of a DataFrame-like input (``None`` otherwise)."""
    cols = getattr(M, "columns", None)
    return [str(c) for c in cols] if cols is not None else None


def _as_numeric_matrix(M, nm: str) -> torch.Tensor:
    """A finite floating-point 2-D tensor, or an error naming the argument."""
    if isinstance(M, torch.Tensor):
        ok_kind = M.dtype.is_floating_point or (not M.dtype.is_complex and M.dtype != torch.bool)
        T = M if ok_kind else None
    else:
        if hasattr(M, "toarray"):  # scipy sparse
            M = M.toarray()
        elif hasattr(M, "to_numpy"):  # DataFrame / Series
            M = M.to_numpy()
        arr = np.asarray(M)
        T = None
        if arr.dtype.kind in "iuf":
            T = torch.as_tensor(arr.astype(np.float64))
    if T is None:
        raise ValueError(f"'{nm}' must be a numeric matrix")
    if T.ndim == 0:
        T = T.reshape(1, 1)
    elif T.ndim == 1:
        T = T.reshape(-1, 1)
    elif T.ndim != 2:
        raise ValueError(f"'{nm}' must be a numeric matrix")
    if not bool(torch.isfinite(T).all()):
        raise ValueError(f"'{nm}' must be finite (no NA, NaN or Inf)")
    return T.to(torch.float64) if not T.dtype.is_floating_point else T


class NBBlockDesign:
    """A design with a per-group block, in compact form (R ``nbBlockDesign``).

    Build with :func:`nb_block_design`. Attributes (all 0-based):
    ``X`` (n x px), ``Z`` (n x q), ``levels`` (group labels), ``bid`` (group
    index per cell), ``cells`` / ``Xl`` / ``Zl`` (per-group cell indices and
    pre-split rows), ``n``, ``px``, ``q``, ``G``, ``xi`` / ``zi`` (column
    positions of the dense and block parts), ``colnames`` and ``rank_tol``.
    """

    def __init__(self, X, Z, levels, bid, cells, colnames, rank_tol):
        self.X = X
        self.Z = Z
        self.levels = list(levels)
        self.bid = bid
        self.cells = cells
        self.Xl = [X[i] for i in cells]
        self.Zl = [Z[i] for i in cells]
        self.n, self.px = X.shape
        self.q = Z.shape[1]
        self.G = len(self.levels)
        self.xi = np.arange(self.px)
        self.zi = self.px + np.arange(self.G * self.q)
        self.colnames = list(colnames)
        self.rank_tol = rank_tol

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the implied dense design: ``(n, px + G * q)``."""
        return (self.n, self.px + self.G * self.q)

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def dtype(self) -> torch.dtype:
        return self.X.dtype

    def to_dense(self) -> np.ndarray:
        """The implied dense ``n x (px + G q)`` design as a numpy array."""
        out = torch.zeros(self.shape, dtype=self.X.dtype, device=self.X.device)
        out[:, : self.px] = self.X
        for g in range(self.G):
            if len(self.cells[g]):
                c0 = self.px + g * self.q
                out[self.cells[g], c0 : c0 + self.q] = self.Zl[g]
        return out.cpu().numpy()

    def __repr__(self) -> str:
        sz = np.array([len(c) for c in self.cells])
        n_empty = int((sz == 0).sum())
        return (
            f"nbBlockDesign: {self.n} cells x {self.shape[1]} columns "
            f"({self.px} dense + {self.G} groups x {self.q} block columns)\n"
            f"  cells per group: min {sz.min()}, median {np.median(sz):g}, max {sz.max()}"
            + (f" ({n_empty} empty)" if n_empty else "")
        )


def nb_block_design(X, Z, block, rank_tol: float = 1e-10, device=None) -> NBBlockDesign:
    """Describe the design ``[X | Z_1 | ... | Z_G]`` in compact form.

    Parameters
    ----------
    X
        cells x ``p_x`` array of the non-absorbed columns (``p_x >= 1``).
    Z
        cells x ``q`` array: each cell's values of its own group's block columns.
    block
        the group of each cell (a pandas Categorical or any vector, coerced like
        R's ``factor()``), without missing values. Every level is a group,
        including a level with no cells.
    rank_tol
        tolerance in (0, 1) below which a group's block is treated as
        rank-deficient and inverted by a truncated eigendecomposition (a
        generalised inverse; see R's ``nbBlockDesign`` "Degenerate blocks").
    device
        optional torch device to place the design on (default: where ``X`` is).

    Returns
    -------
    NBBlockDesign
        with ``.shape``, ``.colnames`` and ``.to_dense()``. Column order is the
        columns of ``X``, then group 1's ``q`` columns, group 2's, and so on
        (levels in category order).
    """
    xnames, znames = _colnames(X), _colnames(Z)
    X = _as_numeric_matrix(X, "X")
    Z = _as_numeric_matrix(Z, "Z")
    if device is not None:
        X = X.to(device)
    Z = Z.to(device=X.device, dtype=X.dtype)
    n = X.shape[0]
    if X.shape[1] < 1:
        raise ValueError("'X' must have at least one column")
    if Z.shape[1] < 1:
        raise ValueError("'Z' must have at least one column")
    if Z.shape[0] != n:
        raise ValueError(
            f"'X' and 'Z' must have one row per cell: nrow(X) = {n}, nrow(Z) = {Z.shape[0]}"
        )
    if len(block) != n:
        raise ValueError(
            f"'block' must have one value per cell: length(block) = {len(block)}, nrow(X) = {n}"
        )
    cat = pd.Categorical(block)
    if bool((cat.codes < 0).any()):
        raise ValueError("'block' must not contain missing values")
    if not (
        isinstance(rank_tol, (int, float, np.integer, np.floating))
        and not isinstance(rank_tol, (bool, np.bool_))
        and np.isfinite(rank_tol)
        and 0 < rank_tol < 1
    ):
        raise ValueError("'rank_tol' must be a single number in (0, 1)")
    levels = [str(x) for x in cat.categories]
    bid_np = np.asarray(cat.codes, dtype=np.int64)
    bid = torch.as_tensor(bid_np, device=X.device)
    # each group's cells (ascending): stable sort of the group codes
    order = np.argsort(bid_np, kind="stable")
    counts = np.bincount(bid_np, minlength=len(levels))
    splits = np.split(order, np.cumsum(counts)[:-1])
    cells = [torch.as_tensor(s, device=X.device) for s in splits]
    px, q = X.shape[1], Z.shape[1]
    xn = xnames or [f"X{j + 1}" for j in range(px)]
    zn = znames or [f"Z{j + 1}" for j in range(q)]
    cn = list(xn) + [f"{lev}:{z}" for lev in levels for z in zn]
    return NBBlockDesign(X, Z, levels, bid, cells, cn, float(rank_tol))


def _is_block_design(W) -> bool:
    return isinstance(W, NBBlockDesign)


def _block_dense_absorb(D: NBBlockDesign) -> np.ndarray:
    """Absorbed grouping of the dense design a compact one stands for.

    Per-column group id: ``-1`` (R ``NA``) for the dense columns, group ``g``
    (0-based) for each block column.
    """
    return np.concatenate([np.full(D.px, -1), np.repeat(np.arange(D.G), D.q)])


def _block_expand(D: NBBlockDesign, v, what: str):
    """Expand a per-column vector given for ``X`` and ``Z`` to the implied design.

    Length ``p_x + G q`` is returned as is; ``p_x + q`` has its ``Z`` part
    repeated for every group; a single value is recycled. A torch tensor stays a
    tensor on its own device; anything else is returned as a numpy array.
    """
    if isinstance(v, torch.Tensor):
        v = v.reshape(-1)
        n = v.numel()
        p = D.px + D.G * D.q
        if n == 1:
            return v.expand(p).clone()
        if n == p:
            return v
        if n == D.px + D.q:
            return torch.cat([v[: D.px], v[D.px :].repeat(D.G)])
    else:
        v = np.atleast_1d(np.asarray(v))
        n = v.size
        p = D.px + D.G * D.q
        if n == 1:
            return np.repeat(v, p)
        if n == p:
            return v
        if n == D.px + D.q:
            return np.concatenate([v[: D.px], np.tile(v[D.px :], D.G)])
    raise ValueError(
        f"'{what}' must be a single value, one per column of the block design ({p}), "
        f"or one per column of [X | Z] ({D.px + D.q}); {n} supplied"
    )


def _block_col_cells(D: NBBlockDesign, j: int) -> torch.Tensor:
    """Cells (ascending, 0-based) on which column ``j`` (0-based) is non-zero."""
    if j < D.px:
        return torch.nonzero(D.X[:, j] != 0).reshape(-1)
    g, k = divmod(j - D.px, D.q)
    return D.cells[g][D.Zl[g][:, k] != 0]


def _block_eta_vec(D: NBBlockDesign, a) -> torch.Tensor:
    """Linear predictor of one gene over the implied design (no offset)."""
    a = as_tensor(a, D.device, D.dtype).reshape(-1)
    Az = a[D.px :].reshape(D.G, D.q)
    return D.X @ a[: D.px] + (D.Z * Az[D.bid]).sum(1)


def _block_score_vec(D: NBBlockDesign, r) -> torch.Tensor:
    """``t(W) r`` over the implied design, for one gene."""
    r = as_tensor(r, D.device, D.dtype).reshape(-1)
    Sz = torch.zeros(D.G, D.q, dtype=D.dtype, device=D.device)
    Sz.index_add_(0, D.bid, D.Z * r[:, None])
    return torch.cat([D.X.T @ r, Sz.reshape(-1)])


def _block_eta_batch(D: NBBlockDesign, A) -> torch.Tensor:
    """Linear predictors of a block of genes (genes x cells, no offset)."""
    A = as_tensor(A, D.device, D.dtype)
    E = A[:, : D.px] @ D.X.T
    Az = A[:, D.px :].reshape(A.shape[0], D.G, D.q)
    if A.shape[0] < BLOCK_BATCH_DENSE_MIN:
        return E + (Az[:, D.bid, :] * D.Z[None]).sum(2)
    for g in range(D.G):
        cg = D.cells[g]
        if len(cg):
            E[:, cg] = E[:, cg] + Az[:, g, :] @ D.Zl[g].T
    return E


def _block_score_batch(R, D: NBBlockDesign) -> torch.Tensor:
    """``R W`` over the implied design for a block of genes (``R`` genes x cells)."""
    R = as_tensor(R, D.device, D.dtype)
    Sx = R @ D.X
    b = R.shape[0]
    Sz = torch.zeros(b, D.G, D.q, dtype=D.dtype, device=D.device)
    if b < BLOCK_BATCH_DENSE_MIN:
        Sz.index_add_(1, D.bid, R[:, :, None] * D.Z[None])
    else:
        for g in range(D.G):
            cg = D.cells[g]
            if len(cg):
                Sz[:, g, :] = R[:, cg] @ D.Zl[g]
    return torch.cat([Sx, Sz.reshape(b, D.G * D.q)], dim=1)


def _block_no_device():
    raise ValueError(
        'an nbBlockDesign runs on the CPU only (backend = "cpu"); the device path takes a '
        "dense design matrix"
    )


def _design_eta(W, a) -> torch.Tensor:
    """Linear predictor of one gene: ``W @ a`` for a dense tensor, compact for a block design."""
    if _is_block_design(W):
        return _block_eta_vec(W, a)
    return W @ as_tensor(a, W.device, W.dtype).reshape(-1)


def _design_cross(W, r) -> torch.Tensor:
    """``t(W) r``, dense or compact."""
    if _is_block_design(W):
        return _block_score_vec(W, r)
    return W.T @ as_tensor(r, W.device, W.dtype).reshape(-1)


def _design_col_nonzero(W, j: int) -> torch.Tensor:
    """Boolean mask over cells of where column ``j`` (0-based) is non-zero."""
    if not _is_block_design(W):
        return W[:, j] != 0
    out = torch.zeros(W.n, dtype=torch.bool, device=W.device)
    out[_block_col_cells(W, j)] = True
    return out


def _block_half_inverse(Cb: torch.Tensor, rank_tol: float):
    """A generalised-inverse half ``H`` (``q x r``) of one group's block, ``C^- = H H'``.

    Exact inverse through a Cholesky factor of the unit-diagonal-scaled block when
    it is well conditioned, otherwise a truncated eigendecomposition. ``None`` when
    the block is not finite.
    """
    q = Cb.shape[0]
    if not bool(torch.isfinite(Cb).all()):
        return None
    d2 = torch.diagonal(Cb)
    keep = d2 > 0
    if not bool(keep.any()):
        return Cb.new_zeros(q, 0)
    d = torch.sqrt(d2[keep])
    k = int(keep.sum())
    Cn = Cb[keep][:, keep] / torch.outer(d, d)
    L, info = torch.linalg.cholesky_ex(Cn)
    if (
        int(info) == 0
        and bool(torch.isfinite(L).all())
        and float(torch.diagonal(L).min()) ** 2 > rank_tol
    ):
        # Cb = D Cn D, Cn = R'R  =>  Cb^-1 = (D^-1 R^-1)(D^-1 R^-1)'
        Rinv = torch.linalg.solve_triangular(
            L.T, torch.eye(k, dtype=Cb.dtype, device=Cb.device), upper=True
        )
        Hk = Rinv / d[:, None]
    else:
        vals, vecs = torch.linalg.eigh(Cn)
        ok = vals > rank_tol * vals.max()
        Hk = (vecs[:, ok] / torch.sqrt(vals[ok])) / d[:, None]
    H = Cb.new_zeros(q, Hk.shape[1])
    H[keep] = Hk
    return H


def _new_state(**kw):
    from ._solver import NFac

    return NFac(**kw)


def _newton_solver_compact(D: NBBlockDesign, pen):
    """Newton solver for the penalised NB information of a compact block design.

    Same contract as :func:`spanormpy._solver._newton_solver`. ``pen`` must already
    be expanded to ``D.shape[1]`` columns (see :func:`_block_expand`). The factor
    state carries ``S`` (px x px), ``B`` (px x Gq, groups side by side), ``H`` (list
    of ``q x r_g`` matrices with ``C_g^- = H_g H_g'``) and ``rank`` (``r_g`` per
    group, numpy int array).
    """
    from ._solver import NewtonSolver, _inverse_or_none, _solve_or_none

    px, q, G = D.px, D.q, D.G
    p = px + G * q
    pen = as_tensor(pen, D.device, D.dtype).reshape(-1)
    if pen.numel() != p:
        raise ValueError("internal: `pen` must be expanded to ncol(W)")
    pen_x = pen[:px]
    pen_z = pen[px:].reshape(G, q)
    rank_tol = D.rank_tol

    def parts(w):
        w = as_tensor(w, D.device, D.dtype).reshape(-1)
        if not bool(torch.isfinite(w).all()):
            return None
        # One pass over each group's rows, scaled once by sqrt(w): every gram is
        # formed symmetrically. A and the Schur correction are accumulated
        # separately and differenced once.
        A = torch.diag(pen_x).clone()
        corr = torch.zeros(px, px, dtype=D.dtype, device=D.device)
        Bm = torch.zeros(px, G * q, dtype=D.dtype, device=D.device)
        H = []
        rk = np.zeros(G, dtype=np.int64)
        for g in range(G):
            if len(D.cells[g]):
                sw = torch.sqrt(w[D.cells[g]])[:, None]
                Xg = D.Xl[g] * sw
                Zg = D.Zl[g] * sw
                A = A + Xg.T @ Xg
                Bg = Xg.T @ Zg
                Cb = Zg.T @ Zg
            else:
                Bg = torch.zeros(px, q, dtype=D.dtype, device=D.device)
                Cb = torch.zeros(q, q, dtype=D.dtype, device=D.device)
            Cb = Cb + torch.diag(pen_z[g])
            Hg = _block_half_inverse(Cb, rank_tol)
            if Hg is None:
                return None
            if Hg.shape[1]:
                BH = Bg @ Hg
                corr = corr + BH @ BH.T
            Bm[:, g * q : (g + 1) * q] = Bg
            H.append(Hg)
            rk[g] = Hg.shape[1]
        return _new_state(S=A - corr, B=Bm, H=H, rank=rk)

    def as_fac(x):
        from ._solver import NFac

        return x if (x is None or isinstance(x, NFac)) else parts(x)

    def cinv(st, V):  # V is G x q
        out = torch.zeros(G, q, dtype=D.dtype, device=D.device)
        for g in range(G):
            Hg = st.H[g]
            if Hg.shape[1]:
                out[g] = Hg @ (Hg.T @ V[g])
        return out

    def solve(w, s):
        st = as_fac(w)
        if st is None:
            return None
        s = as_tensor(s, D.device, D.dtype).reshape(-1)
        sz = s[px:].reshape(G, q)
        rhs = s[:px] - st.B @ cinv(st, sz).reshape(-1)
        dx = _solve_or_none(st.S, rhs)
        if dx is None:
            return None
        dz = cinv(st, sz - (st.B.T @ dx).reshape(G, q))
        return torch.cat([dx, dz.reshape(-1)])

    def xcov(w):
        st = as_fac(w)
        if st is None:
            return None
        return _inverse_or_none(st.S)

    return NewtonSolver(factor=parts, solve=solve, xcov=xcov)
