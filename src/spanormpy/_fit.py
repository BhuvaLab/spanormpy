"""The ``SpaNormFit`` container (R class ``SpaNormFit``, ``R/AllClasses.R``)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ._adjust import get_gene_models, global_psi_max


@dataclass
class SpaNormFit:
    """A fitted SpaNorm model. Matrices keep R's orientation (alpha: genes x p, W: cells x p)."""

    ngenes: int
    ncells: int
    gene_model: str
    df_tps: np.ndarray
    sample_p: float
    lambda_a: float | np.ndarray
    batch: np.ndarray | None
    W: np.ndarray
    alpha: np.ndarray
    gmean: np.ndarray
    psi: np.ndarray
    wtype: np.ndarray
    loglik: np.ndarray
    sampling: np.ndarray
    polish: dict = field(default_factory=dict)
    # Global dispersion-winsorisation threshold carried by a gene subset (R's
    # 'psi.max' attribute set by subsetFitGenes). None means "compute from psi".
    psi_max: float | None = None

    def validate(self) -> bool:
        """Check the fit's consistency (R ``validSpaNormFit``); raise ``ValueError``."""
        return validate_fit(self)


_WTYPES = ("biology", "ls", "batch")
_SAMPLING = ("all", "glm", "dispersion")
_BATCH_KINDS = ("none", "vector", "matrix")


def _na(x) -> bool:
    return bool(np.any(pd.isna(np.asarray(x))))


def validate_fit(fit: SpaNormFit) -> bool:
    """Port of ``validSpaNormFit``; messages keep R's wording. Returns True."""
    # class checks (R enforces these when the slots are assigned)
    for nm in ("W", "alpha"):
        if np.ndim(getattr(fit, nm)) != 2:
            raise ValueError(f"'{nm}' has the wrong class: expected a 2-d matrix")
    W, alpha = np.asarray(fit.W), np.asarray(fit.alpha)
    gmean, psi = np.atleast_1d(fit.gmean), np.atleast_1d(fit.psi)
    wtype, sampling = np.atleast_1d(fit.wtype), np.atleast_1d(fit.sampling)
    loglik = np.atleast_1d(fit.loglik)

    if np.any(np.asarray(fit.df_tps) <= 0):
        raise ValueError("'df_tps' should be greater than 0")
    if fit.gene_model not in get_gene_models():
        raise ValueError(f"'gene_model' should be one of: {', '.join(get_gene_models())}")
    if fit.sample_p <= 0 or fit.sample_p > 1:
        raise ValueError("'sample_p' should be in the interval (0,1]")
    if np.any(np.asarray(fit.lambda_a) < 0):
        raise ValueError("'lambda_a' should be positive")
    if np.any(loglik > 0):
        raise ValueError("'loglik' should be less than or equal to 0")
    if _na(sampling) or not set(sampling.tolist()) <= set(_SAMPLING):
        raise ValueError("'sampling' should contain 'all', 'glm', or 'dispersion'")

    if len({alpha.shape[1], W.shape[1], len(wtype)}) > 1:
        raise ValueError("ncol of 'alpha', ncol of 'W', and length of 'wtype' do not match")
    if alpha.shape[0] != fit.ngenes:
        raise ValueError("nrow of 'alpha' does not match 'ngenes'")
    if len(gmean) != fit.ngenes:
        raise ValueError("length of 'gmean' does not match 'ngenes'")
    if len(psi) != fit.ngenes:
        raise ValueError("length of 'psi' does not match 'ngenes'")
    if W.shape[0] != fit.ncells:
        raise ValueError("nrow of 'W' does not match 'ncells'")
    batch = None if fit.batch is None else np.asarray(fit.batch)
    if batch is not None and batch.ndim <= 1 and len(np.atleast_1d(batch)) != fit.ncells:
        raise ValueError("length of 'batch' does not match 'ncells'")
    if len(sampling) != fit.ncells:
        raise ValueError("length of 'sampling' does not match 'ncells'")
    if batch is not None and batch.ndim == 2 and batch.shape[0] != fit.ncells:
        raise ValueError("nrow of 'batch' does not match 'ncells'")

    if _na(W):
        raise ValueError("'W' cannot have missing values")
    if _na(alpha):
        raise ValueError("'alpha' cannot have missing values")
    if _na(gmean):
        raise ValueError("'gmean' cannot have missing values")
    if fit.gene_model == "nb" and _na(psi):
        raise ValueError("'psi' cannot have missing values")
    if _na(wtype):
        raise ValueError("'wtype' cannot have missing values")
    if batch is not None and _na(batch):
        raise ValueError("'batch' cannot have missing values")

    if not set(wtype.tolist()) <= set(_WTYPES):
        raise ValueError("'wtype' should have values that are either 'biology', 'ls', or 'batch'.")
    if len(fit.polish) > 0 and not {"settings", "genes"} <= set(fit.polish):
        raise ValueError(
            "'polish' should be an empty list, or a list with 'settings' and 'genes' elements"
        )
    return True


def is_polished(fit) -> bool:
    """Does the fit carry a per-gene polish (R ``isPolished``)?"""
    return len(getattr(fit, "polish", None) or {}) > 0


def _rows(mask, n: int) -> np.ndarray:
    mask = np.asarray(mask)
    return np.flatnonzero(mask) if mask.dtype == bool else mask.astype(int)


def subset_genes(fit: SpaNormFit, mask) -> SpaNormFit:
    """Gene subset of a fit (R ``subsetFitGenes``), keeping the full fit's dispersion threshold.

    ``mask`` is a boolean mask or integer positions. Per-gene polish diagnostics are
    subset with the genes. The result's ``psi_max`` is the whole-fit threshold, so
    winsorisation of a subset reproduces the whole-matrix result.
    """
    rows = _rows(mask, fit.ngenes)
    psi_max = fit.psi_max if fit.psi_max is not None else global_psi_max(fit.psi)
    polish = dict(fit.polish)
    if isinstance(polish.get("genes"), pd.DataFrame):
        polish["genes"] = polish["genes"].iloc[rows]
    return replace(
        fit,
        ngenes=len(rows),
        alpha=np.asarray(fit.alpha)[rows, :],
        gmean=np.asarray(fit.gmean)[rows],
        psi=np.asarray(fit.psi)[rows],
        polish=polish,
        psi_max=float(psi_max),
    )


# ---------------------------------------------------------------- AnnData storage


def _clean(x):
    """Make a value h5ad-serialisable: drop None, lists -> arrays, nested dicts recursed."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, np.ndarray) and x.dtype == object:
        return x.astype(str)
    return x


def _restore(x):
    """Inverse of ``_clean`` after an h5ad round trip: numpy scalars -> Python, object -> str."""
    if isinstance(x, dict):
        return {k: _restore(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x.astype(str) if x.dtype == object else x
    if isinstance(x, np.generic):
        return x.item()
    return x


def _as_actual(adata) -> None:
    if adata.is_view:
        adata._init_as_actual(adata.copy())


def _drop_prefixed(adata, prefix: str) -> None:
    for attr in ("var", "obs"):
        df = getattr(adata, attr)
        stale = [c for c in df.columns if c.startswith(prefix)]
        if stale:
            df.drop(columns=stale, inplace=True)
    for attr in ("varm", "obsm"):
        m = getattr(adata, attr)
        for k in [k for k in m.keys() if k.startswith(prefix)]:
            del m[k]


def _route(adata, key: str, value: np.ndarray, n_vars: int, n_obs: int) -> str:
    """Store one polish array on the axis its shape matches; return the slot kind."""
    value = np.asarray(value)
    on_var, on_obs = value.shape[:1] == (n_vars,), value.shape[:1] == (n_obs,)
    if on_var and on_obs:
        raise ValueError(f"polish element '{key}' is ambiguous: n_vars == n_obs")
    if value.ndim not in (1, 2) or not (on_var or on_obs):
        raise ValueError(f"polish element '{key}' has shape {value.shape}, matching neither axis")
    if on_var:
        if value.ndim == 1:
            adata.var[key] = value
            return "var"
        adata.varm[key] = value
        return "varm"
    if value.ndim == 1:
        adata.obs[key] = value
        return "obs"
    adata.obsm[key] = value
    return "obsm"


def _read_slot(adata, kind: str, key: str) -> np.ndarray:
    if kind == "var":
        return adata.var[key].to_numpy()
    if kind == "obs":
        return adata.obs[key].to_numpy()
    return np.asarray(getattr(adata, kind)[key])


def store_fit(adata, fit: SpaNormFit, name: str = "spanorm") -> None:
    """Write ``fit`` into ``adata`` (axis-aligned slots; only scalars and labels in ``uns``).

    ``name`` is the prefix (``spanorm``, ``spanorm_null``, ``spanorm_unpolished``, ...).
    A view is first converted to an actual AnnData. Anything stored earlier under
    ``name`` (including polish columns) is replaced.
    """
    fit.validate()
    if fit.ngenes != adata.n_vars or fit.ncells != adata.n_obs:
        raise ValueError(
            f"the fit ({fit.ngenes} genes x {fit.ncells} cells/spots) does not match "
            f"'adata' ({adata.n_vars} genes x {adata.n_obs} cells/spots)"
        )
    _as_actual(adata)
    p = f"{name}_"
    _drop_prefixed(adata, f"{p}polish_")

    adata.varm[f"{p}alpha"] = np.asarray(fit.alpha)
    adata.var[f"{p}gmean"] = np.asarray(fit.gmean)
    adata.var[f"{p}psi"] = np.asarray(fit.psi, dtype=float)
    adata.obsm[f"{p}W"] = np.asarray(fit.W)
    adata.obs[f"{p}sampling"] = pd.Categorical(np.asarray(fit.sampling).astype(str))

    # batch: per-cell, so it lives in obs / obsm; uns keeps the kind and levels
    for stale in (f"{p}batch",):
        adata.obs.drop(columns=[stale], errors="ignore", inplace=True)
        if stale in adata.obsm:
            del adata.obsm[stale]
    bmeta: dict = {"kind": "none"}
    if fit.batch is not None:
        b = np.asarray(fit.batch)
        is_str = b.dtype.kind in "USO"
        bmeta = {"kind": "vector" if b.ndim <= 1 else "matrix", "is_str": bool(is_str)}
        if b.ndim <= 1:
            adata.obs[f"{p}batch"] = b.astype(str) if is_str else b
            bmeta["levels"] = np.unique(b.astype(str) if is_str else b)
        elif is_str:
            adata.obsm[f"{p}batch"] = pd.DataFrame(
                b.astype(str), index=adata.obs_names, columns=[str(j) for j in range(b.shape[1])]
            )
        else:
            adata.obsm[f"{p}batch"] = b

    psi_max = fit.psi_max if fit.psi_max is not None else global_psi_max(fit.psi)
    meta: dict = {
        "ngenes": int(fit.ngenes),  # informational only; never used for validation
        "ncells": int(fit.ncells),
        "gene_model": str(fit.gene_model),
        "df_tps": np.atleast_1d(np.asarray(fit.df_tps)).astype(int),
        "sample_p": float(fit.sample_p),
        "lambda_a": float(fit.lambda_a) if np.ndim(fit.lambda_a) == 0 else np.asarray(fit.lambda_a),
        "wtype": np.atleast_1d(np.asarray(fit.wtype)).astype(str),
        "loglik": np.atleast_1d(np.asarray(fit.loglik, dtype=float)),
        "psi_max": float(psi_max),
        "batch": bmeta,
    }

    if is_polished(fit):
        pol = fit.polish
        pmeta: dict = {"settings": _clean(pol["settings"])}
        genes = pol["genes"]
        if len(genes) != adata.n_vars:
            raise ValueError("'polish[\"genes\"]' must have one row per gene")
        pmeta["gene_cols"] = np.asarray([str(c) for c in genes.columns], dtype=str)
        for c in genes.columns:
            adata.var[f"{p}polish_{c}"] = genes[c].to_numpy()
        extras, scalars = {}, {}
        for k, v in pol.items():
            if k in ("settings", "genes") or v is None:
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                extras[str(k)] = _route(adata, f"{p}polish_{k}", v, adata.n_vars, adata.n_obs)
            else:
                scalars[str(k)] = _clean(v)
        pmeta["extras"] = extras
        pmeta["scalars"] = scalars
        meta["polish"] = pmeta

    adata.uns.setdefault("spanorm", {})[name] = meta


def load_fit(adata, name: str = "spanorm", validate: bool = True) -> SpaNormFit | None:
    """Rebuild a fit from whatever genes and spots ``adata`` currently holds.

    Returns None when no fit is stored under ``name``. Per-gene and per-spot data are
    read from the axis-aligned slots, so gene and spot subsets are consistent by
    construction (a gene subset keeps the whole fit's ``psi_max``, as R's
    ``subsetFitGenes`` does).
    """
    meta = adata.uns.get("spanorm", {}).get(name) if "spanorm" in adata.uns else None
    if meta is None:
        return None
    p = f"{name}_"
    try:
        alpha = np.asarray(adata.varm[f"{p}alpha"])
        gmean = adata.var[f"{p}gmean"].to_numpy(dtype=float)
        psi = adata.var[f"{p}psi"].to_numpy(dtype=float)
        W = np.asarray(adata.obsm[f"{p}W"])
        sampling = np.asarray(adata.obs[f"{p}sampling"].astype(str))
    except KeyError as e:
        raise ValueError(
            f"the stored '{name}' fit is incomplete: {e} is missing from 'adata'. "
            "Please rerun the fitting function."
        ) from None

    meta = _restore(meta)
    bm = meta["batch"]
    batch = None
    if bm["kind"] == "vector":
        col = adata.obs[f"{p}batch"]
        batch = np.asarray(col.astype(str)) if bm["is_str"] else col.to_numpy()
    elif bm["kind"] == "matrix":
        m = adata.obsm[f"{p}batch"]
        m = m.to_numpy() if isinstance(m, pd.DataFrame) else np.asarray(m)
        batch = m.astype(str) if bm["is_str"] else m

    polish: dict = {}
    if "polish" in meta:
        pm = meta["polish"]
        genes = pd.DataFrame(index=adata.var_names)
        for c in np.atleast_1d(pm.get("gene_cols", [])):
            genes[str(c)] = adata.var[f"{p}polish_{c}"].to_numpy()
        polish = {"settings": pm["settings"], "genes": genes}
        for k, kind in pm.get("extras", {}).items():
            polish[k] = _read_slot(adata, kind, f"{p}polish_{k}")
        polish.update(pm.get("scalars", {}))

    lam = meta["lambda_a"]
    fit = SpaNormFit(
        ngenes=adata.n_vars,
        ncells=adata.n_obs,
        gene_model=meta["gene_model"],
        df_tps=np.atleast_1d(meta["df_tps"]).astype(int),
        sample_p=float(meta["sample_p"]),
        lambda_a=lam if isinstance(lam, np.ndarray) else float(lam),
        batch=batch,
        W=W,
        alpha=alpha,
        gmean=gmean,
        psi=psi,
        wtype=np.atleast_1d(meta["wtype"]).astype(str),
        loglik=np.atleast_1d(meta["loglik"]).astype(float),
        sampling=sampling,
        polish=polish,
    )
    # psi_max is only carried when the genes are a subset of the stored fit's
    stored = float(meta["psi_max"])
    own = global_psi_max(psi)
    if not (stored == own or (np.isnan(stored) and np.isnan(own))):
        fit.psi_max = stored
    if validate:
        fit.validate()
    return fit


def get_spanorm_fit(adata, null: bool = False, validate: bool = True) -> SpaNormFit | None:
    """The stored fit (R ``getSpaNormFit``): ``null=True`` selects the technical (SVG) null fit."""
    name = "spanorm_null" if null else "spanorm"
    fit = load_fit(adata, name, validate=False)
    if validate:
        if fit is None:
            raise ValueError(
                f"{name} model not found in 'adata'. Please run "
                f"'{'spanorm_svg' if null else 'spanorm'}' on the object first."
            )
        fit.validate()
    return fit
