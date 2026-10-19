"""Input robustness of the AnnData API (Review Focus 1-5).

Already covered elsewhere (not repeated here): sparse CSR float vs dense (test_spanorm.py::
test_sparse_matches_dense), counts in a layer (test_layer_counts_and_size_factor_obs_key), view
conversion without warning (test_view_is_converted_without_warning), gene-subset reuse vs the
full run (test_gene_subset_reuses_fit_with_global_psi_max), coordinate forms and batch cases
(tests/test_io.py).
"""

import importlib
import warnings

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

import spanormpy._fitnb as fitnb_mod
from spanormpy import spanorm
from spanormpy._fit import load_fit
from tests._ref import REF_DIR, load_ref

sn = importlib.import_module("spanormpy.spanorm")

RTOL = 1e-6  # iterative fits given R's inputs and sampling
FAST = dict(df_tps=2, sample_p=0.8, tol=1e-1, maxit_psi=3, maxit_nb=5, verbose=False)


def small_adata(seed=0, n=120, g=10):
    rng = np.random.default_rng(seed)
    a = anndata.AnnData(rng.poisson(8, size=(n, g)).astype(float))
    a.obsm["spatial"] = rng.uniform(size=(n, 2))
    a.obs_names = [f"s{i}" for i in range(n)]
    return a


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["int64", "float32"])
def test_sparse_equals_dense(fmt, dtype):
    dense = small_adata()
    dense.X = dense.X.astype(dtype)
    sparse = dense.copy()
    sparse.X = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(dense.X)
    spanorm(dense, **FAST)
    spanorm(sparse, **FAST)
    np.testing.assert_allclose(
        sparse.layers["logcounts"].toarray(), dense.layers["logcounts"], rtol=1e-10
    )
    for k in ("spanorm_alpha", "spanorm_W"):
        a, b = (dense.varm, sparse.varm) if k.endswith("alpha") else (dense.obsm, sparse.obsm)
        np.testing.assert_allclose(b[k], a[k], rtol=1e-10)
    np.testing.assert_allclose(sparse.var["spanorm_psi"], dense.var["spanorm_psi"], rtol=1e-10)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_layer_counts(fmt):
    a = small_adata()
    a.layers["raw"] = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(a.X.astype(np.int64))
    b = a.copy()
    spanorm(a, layer="raw", **FAST)
    spanorm(b, **FAST)
    np.testing.assert_allclose(
        a.layers["logcounts"].toarray(), b.layers["logcounts"], rtol=1e-10
    )


# ------------------------------------------------------------------ R-compared scenarios
# tests/ref/make_robustness.R runs R SpaNorm on 60 well-expressed DLPFC genes (size factors
# injected as in test_spanorm.py) and records what R does, errors included.

COMMON = dict(df_tps=2, tol=1e-3, sample_p=0.25)


@pytest.fixture(scope="module")
def rob():
    return load_ref("robustness")


def _inject(monkeypatch, sampling):
    sampling = np.asarray(sampling).astype(str)
    glm = np.flatnonzero(sampling != "all")
    psi_mask = sampling[glm] == "dispersion"
    monkeypatch.setattr(sn, "_sample_random", lambda coords, nsub, rng: glm)
    monkeypatch.setattr(fitnb_mod, "_sample_psi_idx", lambda idx, maxn, rng: psi_mask)


def _adata(rob, counts=None):
    full = anndata.read_h5ad(REF_DIR / "dlpfc_small.h5ad")
    gidx = np.asarray(rob["gidx"]).astype(int) - 1
    a = full[:, gidx].copy()
    if counts is not None:
        a.X = counts
    return a


def _msgs(r):
    return [m for m in np.atleast_1d(r["msgs"]) if m.startswith("(")]


def _flags(err):
    return [ln for ln in err.splitlines() if ln.startswith("(")]


@pytest.fixture
def fitted(rob, monkeypatch):
    a = _adata(rob)
    _inject(monkeypatch, rob["base"]["sampling"])
    spanorm(a, size_factors=rob["sf"], verbose=False, **COMMON)
    monkeypatch.undo()
    np.testing.assert_allclose(a.layers["logcounts"], rob["base"]["logcounts"].T, rtol=RTOL)
    return a


def _sub(rob, key):
    return np.asarray(rob[key]).astype(int) - 1


def test_rerun_after_gene_subset_reuses_fit_matches_r_full_fit(fitted, rob, capsys):
    keep = _sub(rob, "sub_g")
    sub = fitted[:, keep].copy()
    assert list(load_fit(sub).psi) == pytest.approx(list(np.asarray(rob["base"]["psi"])[keep]))
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(sub, adj_method="pearson", df_tps=2, tol=1e-3, sample_p=0.25, verbose=True)
    assert _flags(capsys.readouterr().err) == _msgs(rob["full_pearson"]) == [
        "(1/2) Retrieve precomputed SpaNorm model",
        "(2/2) Normalising data",
    ]
    np.testing.assert_allclose(
        sub.layers["logcounts"], rob["full_pearson"]["logcounts"][keep].T, rtol=RTOL
    )


def test_rerun_after_gene_subset_on_a_view(fitted, rob):
    keep = _sub(rob, "sub_g")
    view = fitted[:, keep]
    assert view.is_view
    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        with pytest.warns(UserWarning, match="overwritten"):
            spanorm(view, adj_method="pearson", verbose=False, **COMMON)
    assert not view.is_view
    np.testing.assert_allclose(
        view.layers["logcounts"], rob["full_pearson"]["logcounts"][keep].T, rtol=RTOL
    )
    # the parent object is untouched
    assert fitted.layers["logcounts"].shape == (fitted.n_obs, fitted.n_vars)


def test_rerun_after_gene_subset_overwrite_matches_r_refit(fitted, rob, monkeypatch, capsys):
    """What R does on the gene-subsetted SPE (a refit) is reachable with overwrite=True."""
    r = rob["gene_subset"]
    assert r["status"] == 0 and _msgs(r)[0] == "(1/2) Fitting SpaNorm model"
    keep = _sub(rob, "sub_g")
    sub = fitted[:, keep].copy()
    _inject(monkeypatch, r["sampling"])
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(
            sub, adj_method="pearson", overwrite=True, size_factors=rob["sf"],
            verbose=True, **COMMON,
        )
    assert _flags(capsys.readouterr().err) == _msgs(r)
    np.testing.assert_allclose(sub.varm["spanorm_alpha"], r["alpha"], rtol=RTOL)
    np.testing.assert_allclose(sub.var["spanorm_psi"], r["psi"], rtol=RTOL)
    np.testing.assert_allclose(sub.layers["logcounts"], r["logcounts"].T, rtol=RTOL)


def test_rerun_after_spot_subset_matches_r(fitted, rob, monkeypatch, capsys):
    """R refits on a spot-subsetted SPE (fit$ncells != ncol) with a fresh sample; so do we."""
    r = rob["spot_subset"]
    assert r["status"] == 0 and _msgs(r)[0] == "(1/2) Fitting SpaNorm model"
    keep = _sub(rob, "sub_s")
    sub = fitted[keep, :].copy()
    assert sub.n_obs == r["logcounts"].shape[1]
    _inject(monkeypatch, r["sampling"])
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(
            sub, adj_method="pearson", size_factors=np.asarray(rob["sf"])[keep],
            verbose=True, **COMMON,
        )
    assert _flags(capsys.readouterr().err) == _msgs(r)
    assert load_fit(sub).ncells == sub.n_obs
    np.testing.assert_allclose(sub.varm["spanorm_alpha"], r["alpha"], rtol=RTOL)
    np.testing.assert_allclose(sub.var["spanorm_psi"], r["psi"], rtol=RTOL)
    np.testing.assert_allclose(sub.obsm["spanorm_W"], r["W"], rtol=RTOL)
    np.testing.assert_allclose(sub.layers["logcounts"], r["logcounts"].T, rtol=RTOL)


def test_spot_subset_view_is_refitted_not_written_into_parent(fitted, rob):
    keep = _sub(rob, "sub_s")
    view = fitted[keep, :]
    before = fitted.varm["spanorm_alpha"].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        with pytest.warns(UserWarning, match="overwritten"):
            spanorm(view, size_factors=np.asarray(rob["sf"])[keep], verbose=False, **COMMON)
    assert not view.is_view and load_fit(view).ncells == len(keep)
    np.testing.assert_array_equal(fitted.varm["spanorm_alpha"], before)


def test_all_zero_gene_matches_r(rob, monkeypatch):
    """R handles an all-zero gene end to end (finite output); so do we, with no NaN."""
    r = rob["zero_gene"]
    assert r["status"] == 0 and r["nan"] == 0
    j = int(rob["zero_gene_idx"]) - 1
    a = _adata(rob)
    X = a.X.copy()
    X[:, j] = 0
    a.X = X
    _inject(monkeypatch, r["sampling"])
    spanorm(a, size_factors=rob["sf"], verbose=False, **COMMON)
    lc = a.layers["logcounts"]
    assert np.isfinite(lc).all()
    np.testing.assert_allclose(a.var["spanorm_psi"], r["psi"], rtol=RTOL)
    np.testing.assert_allclose(a.varm["spanorm_alpha"], r["alpha"], rtol=RTOL, atol=1e-8)
    np.testing.assert_allclose(lc, r["logcounts"].T, rtol=RTOL, atol=1e-8)


@pytest.mark.parametrize("name", ["zero_spot_sf", "zero_spot_sf0"])
def test_zero_total_spot_with_size_factors_matches_r(rob, monkeypatch, name):
    """With supplied size factors R handles a zero-total spot (a factor of 0 is floored at
    1e-8 by ``pmax``); output is finite and equal to R's."""
    r = rob[name]
    assert r["status"] == 0 and r["nan"] == 0
    i = int(rob["zero_spot_idx"]) - 1
    a = _adata(rob)
    X = a.X.copy()
    X[i, :] = 0
    a.X = X
    sf = np.asarray(rob["sf"]).copy()
    if name == "zero_spot_sf0":
        sf[i] = 0
    _inject(monkeypatch, r["sampling"])
    spanorm(a, size_factors=sf, verbose=False, **COMMON)
    assert np.isfinite(a.layers["logcounts"]).all()
    np.testing.assert_allclose(a.varm["spanorm_alpha"], r["alpha"], rtol=RTOL, atol=1e-8)
    np.testing.assert_allclose(a.layers["logcounts"], r["logcounts"].T, rtol=RTOL, atol=1e-8)


def test_zero_total_spot_without_size_factors_errors_like_r(rob):
    """R's own size factors fail on a zero-total spot (scuttle: 'size factors should be
    positive'); the error is raised before anything is written to the object."""
    r = rob["zero_spot_nosf"]
    assert r["status"] == 1 and "size factors should be positive" in r["error"]
    i = int(rob["zero_spot_idx"]) - 1
    a = _adata(rob)
    X = a.X.copy()
    X[i, :] = 0
    a.X = X
    with pytest.raises(ValueError, match="size factors should be positive"):
        spanorm(a, verbose=False, **COMMON)
    assert "logcounts" not in a.layers and "spanorm_alpha" not in a.varm


# ------------------------------------------------------------------ batch through the API


def test_batch_key_edge_cases_through_spanorm():
    a = small_adata()
    a.obs["b"] = np.tile(["x", "y"], a.n_obs // 2).astype(object)
    a.obs.loc[a.obs_names[3], "b"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        spanorm(a, batch_key="b", **FAST)
    a.obs["b"] = "x"
    with pytest.raises(ValueError, match="2 or more levels"):
        spanorm(a, batch_key="b", **FAST)
    # unused categorical level: model.matrix keeps an all-zero, unregularised column, so R
    # fails with "Lapack routine dgesv: system is exactly singular"; torch raises its
    # LinAlgError ("... singular") -- same failure, different exception type. Nothing is written.
    a.obs["b"] = np.tile(["x", "y"], a.n_obs // 2)
    a.obs["b"] = a.obs["b"].astype("category").cat.add_categories("z")
    with pytest.raises(Exception, match="singular"):
        spanorm(a, batch_key="b", **FAST)
    assert "logcounts" not in a.layers and "spanorm_alpha" not in a.varm
