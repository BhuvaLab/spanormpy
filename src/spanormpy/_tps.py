"""Thin-plate-style tensor spline basis (R ``tpsBasis.R`` and ``bs.tps``).

The per-axis basis is a port of ``splines::ns(x, df)``: a cubic B-spline basis
(``splineDesign``) with the intercept column dropped and the natural boundary
constraints (zero second derivative at both boundary knots) removed by
projecting onto the null space of the constraint matrix.  R does that with
``qr()`` (LINPACK Householder) followed by ``qr.qty``; ``numpy.linalg.qr``
(LAPACK Householder) uses the same sign convention, so the columns agree with
R column-for-column, not only in span.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["tps_basis", "bs_tps"]

_DEGREE = 3


def _design(knots: np.ndarray, x: np.ndarray, deriv: int = 0) -> np.ndarray:
    """``splines::splineDesign(knots, x, ord = 4, derivs = deriv)`` for x in the base interval."""
    n = len(knots) - _DEGREE - 1
    spl = BSpline(knots, np.eye(n), _DEGREE, extrapolate=True)
    if deriv:
        spl = spl.derivative(deriv)
    return np.asarray(spl(x)).reshape(len(x), n)


def _ns_knots(ref: np.ndarray, df: int) -> tuple[np.ndarray, np.ndarray]:
    """Interior and boundary knots exactly as ``ns(ref, df = df)`` places them."""
    bk = np.array([ref.min(), ref.max()])
    n_int = df - 1
    if n_int <= 0:
        return np.empty(0), bk
    probs = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
    knots = np.quantile(ref, probs)  # R quantile type 7 == numpy "linear"
    left_eq, right_eq = knots.min() == bk[0], knots.max() == bk[1]
    if left_eq or right_eq:
        if left_eq:
            i = knots == bk[0]
            if i.all():
                raise ValueError("all interior knots match left boundary knot")
            knots[i] += (knots[knots > bk[0]].min() - bk[0]) / 8
        if right_eq:
            i = knots == bk[1]
            if i.all():
                raise ValueError("all interior knots match right boundary knot")
            knots[i] -= (bk[1] - knots[knots < bk[1]].max()) / 8
        warnings.warn("shoving 'interior' knots matching boundary knots to inside", stacklevel=3)
    return knots, bk


def _ns_eval(x: np.ndarray, knots: np.ndarray, bk: np.ndarray) -> np.ndarray:
    """``ns(x, knots = knots, Boundary.knots = bk)``: natural cubic spline basis, no intercept."""
    aknots = np.sort(np.concatenate([np.repeat(bk, 4), knots]))
    n_basis = len(knots) + 4
    ol, orr = x < bk[0], x > bk[1]
    inside = ~(ol | orr)
    basis = np.zeros((len(x), n_basis))
    if ol.any():
        tt = np.vstack([_design(aknots, bk[:1], 0), _design(aknots, bk[:1], 1)])
        basis[ol] = np.column_stack([np.ones(ol.sum()), x[ol] - bk[0]]) @ tt
    if orr.any():
        tt = np.vstack([_design(aknots, bk[1:], 0), _design(aknots, bk[1:], 1)])
        basis[orr] = np.column_stack([np.ones(orr.sum()), x[orr] - bk[1]]) @ tt
    if inside.any():
        basis[inside] = _design(aknots, x[inside], 0)
    const = _design(aknots, bk, 2)[:, 1:]
    basis = basis[:, 1:]
    q, _ = np.linalg.qr(const.T, mode="complete")
    return basis @ q[:, 2:]


def _tensor(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    # column (i - 1) * ncol(by) + j is bx[:, i] * by[:, j] (R 1-based; here 0-based)
    return (bx[:, :, None] * by[:, None, :]).reshape(bx.shape[0], -1)


def _check_vec(v, name: str) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype.kind not in "iuf" or a.ndim == 0 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"'{name}' must be a non-empty finite numeric vector")
    return a.astype(np.float64).ravel()


def tps_basis(x, y, df=6, ref_x=None, ref_y=None, center: bool = True) -> np.ndarray:
    """Tensor natural-spline basis over spatial coordinates (R ``tpsBasis``).

    Parameters follow R: ``df`` is one positive integer (aspect-ratio rule on the
    reference ranges) or two (x and y degrees of freedom).  Knots, per-axis df
    and centring come from the reference coordinates (default ``x``, ``y``).
    Returns an (len(x), df_x * df_y) float64 matrix.  The per-axis degrees of
    freedom and the subtracted reference column means are available through
    :func:`_tps_basis_full`.
    """
    return _tps_basis_full(x, y, df, ref_x, ref_y, center)[0]


def _tps_basis_full(x, y, df, ref_x, ref_y, center):
    xa, ya = _check_vec(x, "x"), _check_vec(y, "y")
    rx = xa if ref_x is None else _check_vec(ref_x, "ref_x")
    ry = ya if ref_y is None else _check_vec(ref_y, "ref_y")
    if xa.size != ya.size:
        raise ValueError("'x' and 'y' must have the same length")
    if rx.size != ry.size:
        raise ValueError("'ref_x' and 'ref_y' must have the same length")
    dfa = np.atleast_1d(np.asarray(df))
    if (
        dfa.dtype.kind not in "iuf"
        or dfa.ndim != 1
        or dfa.size not in (1, 2)
        or not np.all(np.isfinite(dfa))
        or np.any(dfa <= 0)
        or np.any(dfa != np.round(dfa))
    ):
        raise ValueError("'df' must be one or two positive integers")
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("'center' must be True or False")

    if dfa.size == 1:
        xr, yr = np.ptp(rx), np.ptp(ry)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = max(xr, yr) / dfa[0]
            dx, dy = np.ceil(xr / gap), np.ceil(yr / gap)
    else:
        dx, dy = float(dfa[0]), float(dfa[1])
    if not (np.isfinite(dx) and np.isfinite(dy)) or dx < 1 or dy < 1:
        raise ValueError(
            "the reference coordinates span no range along an axis, so no basis can be "
            "built along it; pass df = (df_x, df_y) or other reference coordinates"
        )
    dx, dy = int(dx), int(dy)

    kx, bkx = _ns_knots(rx, dx)
    ky, bky = _ns_knots(ry, dy)
    ref_tensor = _tensor(_ns_eval(rx, kx, bkx), _ns_eval(ry, ky, bky))
    at_ref = np.array_equal(xa, rx) and np.array_equal(ya, ry)
    out = ref_tensor if at_ref else _tensor(_ns_eval(xa, kx, bkx), _ns_eval(ya, ky, bky))
    cm = None
    if center:
        cm = ref_tensor.mean(axis=0)
        out = out - cm
    return out, (dx, dy), cm


def bs_tps(x, y, df_tps: int = 6) -> tuple[np.ndarray, tuple[int, int]]:
    """Port of R ``bs.tps``: centred tensor basis and ``attr(, "df.tps")``.

    Returns ``(basis, (df_x, df_y))``; R's attribute is the per-axis pair, not a scalar.
    """
    if np.ndim(df_tps) != 0 and np.size(df_tps) != 1:
        raise ValueError("length(df_tps) == 1 is not TRUE")
    d = float(np.asarray(df_tps).ravel()[0])
    if np.size(x) != np.size(y):
        raise ValueError("length(x) == length(y) is not TRUE")
    if d <= 0:
        raise ValueError("'df.tps' should be greater than 0")
    if d != int(d):
        raise ValueError("'df.tps' should be an integer")
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    out, dfs, _ = _tps_basis_full(xa, ya, int(d), None, None, True)
    return out, dfs
