"""spanorm_pca: R parity (tests/ref/make_pca.R) and the ported testthat case (test-mainSpaNormPCA.R)."""

import warnings
from types import SimpleNamespace

import anndata
import numpy as np
import pytest
import scipy.sparse as sp
from tests._ref import fit_from_ref, load_ref
from tests.test_polish_spanorm import make_adata

from spanormpy import spanorm_pca
from spanormpy._fit import store_fit
from spanormpy.pca import calculate_pca, deviance_residuals, get_residuals


@pytest.fixture(scope="module")
def ref():
    return load_ref("pca")


def seeded(ref):
    a = make_adata(ref["Y"], ref["coords"], ref["genes"])
    store_fit(a, fit_from_ref(ref["input"]), "spanorm")
    store_fit(a, fit_from_ref(ref["null"]), "spanorm_null")
    a.var["svg_F"] = ref["svg_F"]
    a.var["svg_p"] = ref["svg_p"]
    a.var["svg_fdr"] = ref["svg_fdr"]
    return a


def align_signs(x, rot, rx, rrot):
    """Flip each component of (x, rot) to R's sign via the largest-magnitude loading."""
    out_x, out_r = x.copy(), rot.copy()
    for j in range(rot.shape[1]):
        i = np.argmax(np.abs(rrot[:, j]))
        sgn = np.sign(rot[i, j]) * np.sign(rrot[i, j])
        out_x[:, j] *= sgn
        out_r[:, j] *= sgn
    return out_x, out_r


def test_residuals_match_r(ref):
    ft = fit_from_ref(ref["null"])
    Y = np.asarray(ref["Y"])
    np.testing.assert_allclose(
        get_residuals(Y, ft, "deviance"), ref["dev_all"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        get_residuals(Y, ft, "pearson"), ref["pearson_all"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(  # sparse counts give the same residuals
        get_residuals(sp.csr_matrix(Y), ft, "deviance"), ref["dev_all"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "case,res", [("dev", "deviance"), ("pearson", "pearson"), ("fdr", "deviance")]
)
def test_pca_matches_r_up_to_sign(ref, case, res):
    r = ref[case]
    a = seeded(ref)
    spanorm_pca(
        a,
        nsvgs=int(r["nsvgs"]),
        ncomponents=int(r["ncomp"]),
        svg_fdr=float(r["fdr"]),
        residuals=res,
    )
    svgs = list(np.atleast_1d(r["svgs"]))
    assert list(a.uns["X_spanorm_pca"]["svgs"]) == svgs
    load = a.varm["X_spanorm_pca_loadings"]
    rows = [list(a.var_names).index(g) for g in svgs]
    rot = load[rows]
    x, rot = align_signs(a.obsm["X_spanorm_pca"], rot, r["x"], r["rotation"])
    np.testing.assert_allclose(x, r["x"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rot, r["rotation"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a.uns["X_spanorm_pca"]["variance"], r["varExplained"], rtol=1e-6)
    np.testing.assert_allclose(a.uns["X_spanorm_pca"]["percent_var"], r["percentVar"], rtol=1e-6)
    # genes not used have zero loadings
    unused = np.setdiff1d(np.arange(a.n_vars), rows)
    assert not load[unused].any()


def test_deviance_residuals_match_dense_reference():
    """test-mainSpaNormPCA.R: devianceResiduals on a sparse Y matches a dense base-R reference."""
    rng = np.random.default_rng(7)
    ng, nc, ncov = 8, 12, 3
    W = rng.standard_normal((nc, ncov))
    alpha = rng.standard_normal((ng, ncov)) * 0.1
    gmean = rng.standard_normal(ng)
    psi = rng.uniform(0.1, 0.5, ng)
    Y = rng.poisson(5, (ng, nc)).astype(float)
    Y.flat[rng.choice(Y.size, 10, replace=False)] = 0
    ft = SimpleNamespace(gmean=gmean, alpha=alpha, W=W, psi=psi, wtype=np.array(["ls"] * ncov))
    mu = np.exp(gmean[:, None] + alpha @ W.T)
    lp = np.log(psi)
    med = np.median(lp)
    pp = np.minimum(psi, np.exp(med + 3 * 1.4826 * np.median(np.abs(lp - med))))[:, None]
    with np.errstate(all="ignore"):
        d = Y * np.log(Y / mu)
    d[np.isnan(d)] = 0
    d = np.maximum(d - (Y + 1 / pp) * np.log((1 + Y * pp) / (1 + mu * pp)), 0)
    ref = np.sign(Y - mu) * np.sqrt(2 * d)
    got = deviance_residuals(sp.csr_matrix(Y), ft)
    np.testing.assert_allclose(got, ref, rtol=1e-8)
    assert not np.isnan(got).any()


def test_sparse_counts_and_layer_match_dense(ref):
    a, b = seeded(ref), seeded(ref)
    b.layers["c"] = sp.csr_matrix(np.asarray(b.X))
    spanorm_pca(a, nsvgs=25, ncomponents=5)
    spanorm_pca(b, nsvgs=25, ncomponents=5, layer="c")
    np.testing.assert_allclose(
        a.obsm["X_spanorm_pca"], b.obsm["X_spanorm_pca"], rtol=1e-10, atol=1e-12
    )


def test_copy_view_and_key(ref):
    a = seeded(ref)
    out = spanorm_pca(a, nsvgs=25, ncomponents=3, key_added="pc", copy=True)
    assert (
        isinstance(out, anndata.AnnData) and "pc" not in a.obsm and out.obsm["pc"].shape == (500, 3)
    )
    assert out.varm["pc_loadings"].shape == (40, 3)
    v = a[:, :]  # a view of the object: converted to actual, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spanorm_pca(v, nsvgs=25, ncomponents=3)
    assert "X_spanorm_pca" in v.obsm and not v.is_view


def test_gene_subset_reuses_fit(ref):
    a = seeded(ref)
    sub = a[:, :30].copy()
    spanorm_pca(sub, nsvgs=10, ncomponents=3, residuals="pearson")
    assert sub.varm["X_spanorm_pca_loadings"].shape == (30, 3)


def test_randomized_close_to_exact(ref):
    a, b = seeded(ref), seeded(ref)
    spanorm_pca(a, nsvgs=40, ncomponents=3)
    spanorm_pca(b, nsvgs=40, ncomponents=3, svd_solver="randomized")
    # approximate solver on a nearly flat spectrum: leading variance tight, the rest within 1%
    va, vb = a.uns["X_spanorm_pca"]["variance"], b.uns["X_spanorm_pca"]["variance"]
    np.testing.assert_allclose(vb, va, rtol=1e-2)
    c = np.abs(np.sum(a.obsm["X_spanorm_pca"][:, 0] * b.obsm["X_spanorm_pca"][:, 0]))
    assert (
        c
        / (
            np.linalg.norm(a.obsm["X_spanorm_pca"][:, 0])
            * np.linalg.norm(b.obsm["X_spanorm_pca"][:, 0])
        )
        > 0.999
    )


def test_calculate_pca_centres_not_scales():
    rng = np.random.default_rng(1)
    e = rng.standard_normal((6, 30)) * np.arange(1, 7)[:, None]
    p = calculate_pca(e, 6)
    xc = e.T - e.T.mean(axis=0)
    np.testing.assert_allclose(p["x"] @ p["rotation"].T, xc, atol=1e-10)
    np.testing.assert_allclose(p["variance"].sum(), xc.var(axis=0, ddof=1).sum(), rtol=1e-10)


def test_errors(ref):
    a = seeded(ref)
    for kw in (
        {"nsvgs": 0},
        {"ncomponents": 0},
        {"ncomponents": 501},
        {"residuals": "x"},
        {"svd_solver": "x"},
        {"ncomponents": 41},
    ):
        with pytest.raises(ValueError):
            spanorm_pca(a, **kw)
    with pytest.raises(ValueError, match="spanorm_null model not found"):
        b = seeded(ref)
        del b.uns["spanorm"]["spanorm_null"]
        for k in [k for k in b.var.columns if k.startswith("spanorm_null")]:
            del b.var[k]
        spanorm_pca(b)
    c = make_adata(ref["Y"], ref["coords"], ref["genes"])
    with pytest.raises(ValueError, match="not found"):
        spanorm_pca(c)
    d = seeded(ref)
    with pytest.raises(ValueError, match="do not match"):
        spanorm_pca(d[:300].copy())
    e = seeded(ref)
    e.var.loc[e.var_names[0], "svg_fdr"] = np.nan
    with pytest.raises(ValueError, match="smaller nsvgs"):
        spanorm_pca(e, nsvgs=40, ncomponents=3)


def test_nsvgs_inf_uses_all_svgs_passing_fdr(ref):
    # R: nsvgs = Inf is no cap, so every gene passing svg_fdr is used
    a = seeded(ref)
    spanorm_pca(a, nsvgs=np.inf, svg_fdr=0.2, ncomponents=3)
    expected = int((a.var["svg_fdr"] <= 0.2).sum())
    assert a.uns["X_spanorm_pca"]["params"]["nsvgs"] == expected
    assert len(a.uns["X_spanorm_pca"]["svgs"]) == expected
