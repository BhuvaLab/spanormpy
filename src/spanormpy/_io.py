"""Input glue: counts, coordinates and batch covariates from AnnData / SpatialData.

Ports the argument checks of R ``checkSPE`` (counts validity), ``checkBatch`` and the
coordinate handling of the SpaNorm entry points. Everything here converts to R's
orientation (genes x cells) or to plain numpy arrays; nothing else in the package
touches ``AnnData`` layout details.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._filter import _check_counts, get_counts_genes_by_cells


def resolve_input(
    data, layer: str | None = None, spatial_key: str = "spatial", table_key: str | None = None
) -> tuple[AnnData, Callable[[AnnData], Any]]:
    """Return ``(table, write_back)`` for an AnnData or a SpatialData object.

    ``write_back(adata)`` puts a (possibly replaced) table back into the container; for
    AnnData it is a no-op because the object is modified in place. ``write_back(adata,
    copy=True)`` returns a new container holding ``adata`` and leaves the original untouched.
    """
    if isinstance(data, AnnData):
        return data, lambda a, copy=False: a if copy else None
    if type(data).__module__.split(".")[0] == "spatialdata":
        tables = data.tables
        if len(tables) == 0:
            raise ValueError("the SpatialData object has no tables")
        if table_key is None:
            if len(tables) > 1:
                raise ValueError(
                    f"the SpatialData object has several tables ({sorted(tables)}); "
                    "choose one with 'table_key'"
                )
            table_key = next(iter(tables))
        elif table_key not in tables:
            raise ValueError(f"table '{table_key}' not found in the SpatialData object")
        key = table_key

        def write_back(a: AnnData, copy: bool = False):
            if not copy:
                if tables[key] is not a:
                    tables[key] = a
                return data
            # new container with the result table; other elements are shared by reference
            return type(data)(
                images=dict(data.images),
                labels=dict(data.labels),
                points=dict(data.points),
                shapes=dict(data.shapes),
                tables={**{k: v for k, v in tables.items() if k != key}, key: a},
            )

        return tables[key], write_back
    raise TypeError("'data' must be an AnnData or a SpatialData object")


def get_counts(adata: AnnData, layer: str | None = None):
    """Validated raw counts, genes x cells (dense ndarray or CSR), from ``X`` or ``layer``."""
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"layer '{layer}' not found in 'adata'")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("'counts' must have nrow > 0 and ncol > 0")
    Y = get_counts_genes_by_cells(adata, layer)
    _check_counts(Y)
    if not sp.issparse(Y) and not Y.flags.writeable:  # AnnData views are read-only
        Y = Y.copy()
    return Y


def get_coords(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """First two columns of ``obsm[spatial_key]`` as a float64 (cells x 2) array."""
    if spatial_key not in adata.obsm:
        raise ValueError(f"spatial coordinates not found in adata.obsm['{spatial_key}']")
    xy = adata.obsm[spatial_key]
    xy = xy.to_numpy() if isinstance(xy, pd.DataFrame) else np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must have at least 2 columns (x and y coordinates)"
        )
    try:
        xy = xy[:, :2].astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"adata.obsm['{spatial_key}'] must be numeric") from None
    if not np.all(np.isfinite(xy)):
        raise ValueError("spatial coordinates contain non-finite values")
    return xy


# ------------------------------------------------------------------------------ batch


def _collation_key(s: str):
    """Approximate glibc ``en_US.UTF-8`` collation used by R's ``factor()`` / ``sort()``.

    Punctuation < digits < letters; letters compare case-insensitively first and ties put
    lower case before upper case ("a" < "A" < "ab" < "b" < "B").
    """

    def primary(c: str):
        if c.isalpha():
            return (2, c.lower())
        if c.isdigit():
            return (1, c)
        return (0, c)

    return (tuple(primary(c) for c in s), tuple(c.isupper() for c in s))


def _factor_levels(batch: np.ndarray | pd.Categorical) -> tuple[list, np.ndarray]:
    """Levels (R ``levels(factor(x))``) and the level index of each element."""
    if isinstance(batch, pd.Categorical):
        levels = list(batch.categories)
        return levels, np.asarray(batch.codes)
    vals = np.asarray(batch)
    if vals.dtype.kind == "b":
        levels = [False, True]
        return levels, vals.astype(int)
    strs = vals.astype(str)
    levels_s = sorted(set(strs.tolist()), key=_collation_key)
    lookup = {s: i for i, s in enumerate(levels_s)}
    return levels_s, np.array([lookup[s] for s in strs.tolist()])


def _is_missing(batch) -> bool:
    if isinstance(batch, pd.DataFrame):
        return bool(batch.isna().to_numpy().any())
    if isinstance(batch, (pd.Series, pd.Categorical)):
        return bool(pd.isna(batch).any())
    a = np.asarray(batch)
    if a.dtype.kind in "fc":
        return bool(np.isnan(a).any())
    if a.dtype.kind == "O":
        return bool(pd.isna(a).any())
    return False


def batch_design(batch, nobs: int) -> tuple[np.ndarray | None, list[str] | None]:
    """:func:`check_batch` plus the column names R's ``model.matrix``/``cbind`` would give."""
    if batch is None:
        return None, None
    if isinstance(batch, str):
        raise TypeError("'batch' should be an array, a matrix or a categorical vector")
    if _is_missing(batch):
        raise ValueError("'batch' cannot have missing values")

    if isinstance(batch, pd.Series):
        batch = batch.array if isinstance(batch.dtype, pd.CategoricalDtype) else batch.to_numpy()
        if isinstance(batch, pd.api.extensions.ExtensionArray) and not isinstance(
            batch, pd.Categorical
        ):
            batch = np.asarray(batch)

    # ---- matrix
    if isinstance(batch, pd.DataFrame) or (
        not isinstance(batch, pd.Categorical) and np.ndim(batch) == 2
    ):
        names = [str(c) for c in batch.columns] if isinstance(batch, pd.DataFrame) else None
        m = batch.to_numpy() if isinstance(batch, pd.DataFrame) else np.asarray(batch)
        if m.shape[0] != nobs:
            raise ValueError(
                "number of rows in the 'batch' matrix do not match number of cells/spots"
            )
        numeric = (
            all(pd.api.types.is_numeric_dtype(t) for t in batch.dtypes)
            if isinstance(batch, pd.DataFrame)
            else m.dtype.kind in "iufb"
        )
        if not numeric:
            raise ValueError(
                "'batch' should be a numeric matrix (consider using 'model.matrix' to "
                "define the design)"
            )
        m = m.astype(np.float64)
        is_int = np.array([bool(n) and "intercept" in n.lower() for n in names]) if names else (
            np.zeros(m.shape[1], dtype=bool)
        )
        is_int = is_int | np.all(m == 1, axis=0)
        if is_int.any():
            warnings.warn("'intercept' term detected and will be removed", stacklevel=3)
            m = m[:, ~is_int]
            if names is not None:
                names = [n for n, k in zip(names, is_int, strict=True) if not k]
        if names is None:
            names = [""] * m.shape[1]
        return m, names

    # ---- vector / factor
    if np.ndim(batch) != 1:
        raise ValueError("'batch' should be a vector or a matrix")
    if len(batch) != nobs:
        raise ValueError("length of 'batch' vector does not match number of cells/spots")
    if not isinstance(batch, pd.Categorical):
        arr = np.asarray(batch)
        if arr.dtype.kind in "iuf":  # numeric vector: a single covariate column (model.matrix)
            return arr.astype(np.float64)[:, None], ["batch"]
        batch = arr
    levels, codes = _factor_levels(batch)
    if len(levels) < 2:
        raise ValueError("contrasts can be applied only to factors with 2 or more levels")
    # treatment contrasts; unused levels keep an all-zero column, as in R's model.matrix
    design = (codes[:, None] == np.arange(1, len(levels))[None, :]).astype(np.float64)
    return design, [f"batch{lv}" for lv in levels[1:]]


def check_batch(batch, nobs: int) -> np.ndarray | None:
    """Port of R ``checkBatch``: a numeric design matrix (cells x k) or ``None``.

    ``batch`` is ``None``, a 1-d array / Series / Categorical (strings, booleans and
    categoricals are factors -> treatment-contrast columns; numeric vectors are one
    covariate column, as in ``model.matrix(~batch)``) or a numeric 2-d array / DataFrame
    (a design; intercept columns are dropped with a warning).
    """
    return batch_design(batch, nobs)[0]
