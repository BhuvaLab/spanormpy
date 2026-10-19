"""polish_spanorm: R parity and the ported testthat cases.

R sources: tests/testthat/test-polishSpaNorm.R, test-polishSpaNormModel.R and
test-polishSpaNormJoint.R (SpaNorm c12e5e5). The fixtures are in tests/ref/make_polish_spanorm.R:
Python seeds ``spanorm`` / ``spanorm_null`` from R's *input* fits (``spanorm_svg`` arrives in a
later task) and compares every fit and the logcounts that ``polish_spanorm`` stores.
"""

import re
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import optimize
from tests._ref import load_ref

from spanormpy import _polish_model as pm
from spanormpy import polish_spanorm, spanorm
from spanormpy._fit import (
    SpaNormFit,
    fit_is_stale_for_spots,
    get_spanorm_fit,
    is_polished,
    load_fit,
    store_fit,
)
from spanormpy._nbdist import dnbinom
from spanormpy._polish_engine import nb_mu_floor
from spanormpy._solver import nb_newton_solver
from spanormpy.polish import polish_nb

RTOL = 1e-6
ATOL = 1e-10  # entries that are exactly 0 up to rounding (e.g. a coefficient at 1e-14)


@pytest.fixture(scope="module")
def ref():
    return load_ref("polish_spanorm")


# ------------------------------------------------------------------------------ helpers


def fit_from_ref(d, batch=None) -> SpaNormFit:
    lam = np.atleast_1d(d["lambda_a"]).astype(float)
    return SpaNormFit(
        ngenes=int(d["ngenes"]),
        ncells=int(d["ncells"]),
        gene_model="nb",
        df_tps=np.atleast_1d(d["df_tps"]).astype(int),
        sample_p=0.25,
        lambda_a=lam,
        batch=batch,
        W=np.asarray(d["W"]),
        alpha=np.asarray(d["alpha"]),
        gmean=np.asarray(d["gmean"]),
        psi=np.asarray(d["psi"]),
        wtype=np.atleast_1d(d["wtype"]).astype(str),
        loglik=np.atleast_1d(d["loglik"]).astype(float),
        sampling=np.atleast_1d(d["sampling"]).astype(str),
        colnames=np.atleast_1d(d["W_colnames"]).astype(str),
    )


def make_adata(Y, coords, genes=None):
    a = anndata.AnnData(
        X=np.asarray(Y, dtype=float).T,
        obs=pd.DataFrame(index=[f"c{i + 1}" for i in range(Y.shape[1])]),
        var=pd.DataFrame(index=list(genes) if genes is not None else [f"g{i}" for i in range(Y.shape[0])]),
    )
    a.obsm["spatial"] = np.asarray(coords)
    return a


def seeded(ref, with_null=True, with_svg=True):
    """Fixture a: counts + R's input SpaNorm (and null) fit stored in an AnnData."""
    a = make_adata(ref["Y"], ref["coords"], ref["genes"])
    store_fit(a, fit_from_ref(ref["a"]["input"]), "spanorm")
    if with_null:
        store_fit(a, fit_from_ref(ref["a"]["input_null"]), "spanorm_null")
    if with_svg:
        a.var["svg_F"] = ref["a"]["svg_F"]
        a.var["svg_p"] = 0.5
        a.var["svg_fdr"] = 0.5
    return a


def fresh(ref, **kw):
    return seeded(ref, **kw)


def quiet(adata, **kw):
    """polish_spanorm with the expected SVG warning swallowed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polish_spanorm(adata, verbose=False, **kw)
    return adata


def compare_fit(adata, name, r, rtol=RTOL):
    fit = load_fit(adata, name)
    np.testing.assert_allclose(fit.alpha, r["alpha"], rtol=rtol, atol=ATOL)
    np.testing.assert_allclose(fit.gmean, r["gmean"], rtol=rtol, atol=ATOL)
    np.testing.assert_allclose(fit.psi, r["psi"], rtol=rtol)
    np.testing.assert_allclose(fit.W, r["W"], rtol=1e-12)
    assert is_polished(fit) == bool(r["polished"])
    if not is_polished(fit):
        return fit
    g = fit.polish["genes"]
    for c in ("restarted", "capped", "singular", "psi_bound", "polished"):
        np.testing.assert_array_equal(g[c].to_numpy().astype(bool), np.asarray(r["genes"][c]) != 0, c)
    np.testing.assert_array_equal(g["iterations"].to_numpy(), np.asarray(r["genes"]["iterations"]))
    np.testing.assert_allclose(g["psi_fitnb"], r["genes"]["psi_fitnb"], rtol=rtol)
    np.testing.assert_allclose(g["loglik"], r["genes"]["loglik"], rtol=rtol, equal_nan=True)
    held = np.asarray([("" if pd.isna(x) else x) for x in g["held_out"]])
    np.testing.assert_array_equal(held, np.atleast_1d(r["genes"]["held_out"]))
    s, rs = fit.polish["settings"], r["settings"]
    for k, v in rs.items():
        if k in ("psi_method", "ls", "cells"):
            assert s[k] == v
        elif k in ("ls_converged",):
            assert bool(s[k]) == bool(v)
        elif k in ("maxit", "ls_iterations", "ls_maxit", "ls_singular"):
            assert int(s[k]) == int(v), k
        elif k in ("ls_score",):
            np.testing.assert_allclose(s[k], v, rtol=1e-4, atol=1e-5)  # score at the stop: noise level
        else:
            np.testing.assert_allclose(s[k], v, rtol=rtol, err_msg=k, atol=0 if k != "ls_se" else 0)
    return fit


def compare_run(adata, r, null=True, rtol=RTOL):
    compare_fit(adata, "spanorm", r["SpaNorm"], rtol)
    compare_fit(adata, "spanorm_unpolished", r["SpaNormUnpolished"], rtol)
    if "SpaNormNull" in r and null:
        compare_fit(adata, "spanorm_null", r["SpaNormNull"], rtol)
        compare_fit(adata, "spanorm_null_unpolished", r["SpaNormNullUnpolished"], rtol)
    np.testing.assert_allclose(adata.layers["logcounts"], r["logcounts"].T, rtol=rtol, atol=1e-9)


RUNS = {
    "default": {},
    "tight": dict(tol=1e-12),
    "joint": dict(ls="joint"),
    "profile": dict(psi_method="profile"),
    "profile_joint": dict(psi_method="profile", ls="joint"),
    "cells_fit": dict(cells="fit"),
    "no_null": dict(null=False),
    "pearson": dict(adj_method="pearson", scale_factor=2.0),
    "maxit3": dict(maxit=3, tol=1e-12),
}


@pytest.mark.parametrize("name", list(RUNS))
def test_polish_spanorm_matches_r(ref, name):
    a = seeded(ref)
    with pytest.warns(UserWarning, match="SVG results"):
        polish_spanorm(a, verbose=False, **RUNS[name])
    r = ref["a"][name]
    compare_run(a, r, null=RUNS[name].get("null", True))
    assert not any(c in a.var for c in ("svg_F", "svg_p", "svg_fdr"))
    if name == "no_null":
        assert "spanorm_null_unpolished" not in a.uns["spanorm"]
        # null = False leaves a stored null exactly as it was
        compare_fit(a, "spanorm_null", ref["a"]["input_null"])


@pytest.mark.parametrize("name", ["zero_gene", "zero_gene_joint"])
def test_all_zero_gene_matches_r(ref, name):
    Y = ref["Y"].copy()
    Y[0, :] = 0
    a = make_adata(Y, ref["coords"], ref["genes"])
    store_fit(a, fit_from_ref(ref["a"]["input"]), "spanorm")
    kw = dict(ls="joint") if "joint" in name else {}
    # held out quietly: a verbose message only, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polish_spanorm(a, verbose=False, null=False, **kw)
    r = ref["a"][name]
    compare_run(a, r, null=False)
    g = load_fit(a).polish["genes"]
    assert not g["polished"].iloc[0] and g["polished"].iloc[1:].all()
    assert g["held_out"].iloc[0] == "all-zero" and g["held_out"].iloc[1:].isna().all()
    assert g["iterations"].iloc[0] == 0 and np.isnan(g["loglik"].iloc[0])
    f0 = load_fit(a, "spanorm_unpolished")
    f = load_fit(a)
    assert f.gmean[0] == f0.gmean[0] and f.psi[0] == f0.psi[0]
    np.testing.assert_array_equal(f.alpha[0, 1:], f0.alpha[0, 1:])
    assert np.isfinite(f.gmean).all() and np.isfinite(f.alpha).all() and np.isfinite(f.psi).all()


def test_batch_level_holdout_matches_r(ref):
    b = ref["b"]
    g = int(b["gene"]) - 1
    batch = np.asarray(b["batch"]).astype(str)
    for name, kw in (("default", {}), ("joint", dict(ls="joint")), ("tight", dict(tol=1e-12))):
        a = make_adata(b["Y"], ref["coords"], ref["genes"])
        store_fit(a, fit_from_ref(b["input"], batch=batch), "spanorm")
        with pytest.warns(UserWarning, match="^1 gene with no counts in some batch level"):
            polish_spanorm(a, verbose=False, null=False, **kw)
        compare_run(a, b[name], null=False)
        f = load_fit(a)
        f0 = load_fit(a, "spanorm_unpolished")
        pol = f.polish["genes"]
        assert not pol["polished"].iloc[g] and pol["held_out"].iloc[g] == "zero-batch-level"
        assert pol["iterations"].iloc[g] == 0 and np.isnan(pol["loglik"].iloc[g])
        assert f.gmean[g] == f0.gmean[g] and f.psi[g] == f0.psi[g]
        np.testing.assert_array_equal(f.alpha[g, 1:], f0.alpha[g, 1:])
        assert pol["polished"].drop(pol.index[g]).all()
        assert pol["held_out"].drop(pol.index[g]).isna().all()
        s = f.polish["settings"]
        assert (s["pen"][np.r_[False, f.wtype[1:] == "batch"]] == 0).all()  # batch is unpenalised
        if kw.get("ls") == "joint":
            assert s["ls_converged"] and s["a1"] != s["a1_input"]
            assert len(np.unique(f.alpha[:, 0])) == 1 and f.alpha[g, 0] == s["a1"]
        else:
            # the level it was never seen in keeps the unpolished normalised values
            lvl = batch == "s2"
            l1 = a.layers["logcounts"][:, g]
            assert l1[lvl].mean() < l1[~lvl].mean() - 1  # not imputed at full strength


def test_collinear_unpenalised_block_matches_r(ref, monkeypatch):
    c = ref["c"]
    batch = np.asarray(c["batch"])
    real = pm.polish_nb

    def boom(*a, **k):
        raise AssertionError("polish_nb reached on a collinear unpenalised block")

    monkeypatch.setattr(pm, "polish_nb", boom)
    for name, kw in (("default", {}), ("joint", dict(ls="joint"))):
        a = make_adata(c["Y"], c["coords"], ref["genes"][:20])
        store_fit(a, fit_from_ref(c["input"], batch=batch), "spanorm")
        f0 = load_fit(a)
        assert (f0.wtype == "batch").sum() == 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            polish_spanorm(a, verbose=False, null=False, **kw)
        msgs = [str(x.message) for x in w]
        assert any(
            re.match(
                r"^20 of the 20 genes passed to the 'spanorm' polish could not be polished "
                r"\(20 with a singular", m)
            for m in msgs
        ), msgs
        f = load_fit(a)
        pol = f.polish["genes"]
        assert is_polished(f) and not pol["polished"].any() and pol["singular"].all()
        assert pol["held_out"].isna().all()
        np.testing.assert_array_equal(f.alpha, f0.alpha)
        np.testing.assert_array_equal(f.gmean, f0.gmean)
        if name == "joint":
            assert f.polish["settings"]["ls_iterations"] == 0
            assert any("nothing informs" in m for m in msgs)
        compare_run(a, c[name], null=False)
    monkeypatch.setattr(pm, "polish_nb", real)


# ------------------------------------------------------------------- ported testthat cases


def scaled_score(fit, Y, cells="all"):
    """Largest absolute penalised score per gene over total count (helper-polish.R)."""
    p = pm.spanorm_polish_problem(fit, cells)
    Y = Y[:, p["cells_idx"]]
    A = np.column_stack([fit.gmean, fit.alpha[:, 1:]])
    out = []
    for g in range(Y.shape[0]):
        mu = np.maximum(np.exp(p["X"] @ A[g] + p["offset"]), nb_mu_floor())
        sc = p["X"].T @ ((Y[g] - mu) / (1 + fit.psi[g] * mu)) - p["pen"] * A[g]
        out.append(np.abs(sc).max() / Y[g].sum())
    return np.array(out)


def total_loglik(fit, Y, genes=None):
    p = pm.spanorm_polish_problem(fit)
    A = np.column_stack([fit.gmean, fit.alpha[:, 1:]])
    genes = range(Y.shape[0]) if genes is None else genes
    tot = 0.0
    for g in genes:
        mu = np.maximum(np.exp(p["X"] @ A[g] + p["offset"]), nb_mu_floor())
        tot += float(dnbinom(Y[g], mu=mu, size=1 / fit.psi[g], log=True).sum()) - 0.5 * float(
            (p["pen"] * A[g] ** 2).sum()
        )
    return tot


def a1_score(fit, Y, genes=None):
    p = pm.spanorm_polish_problem(fit)
    A = np.column_stack([fit.gmean, fit.alpha[:, 1:]])
    genes = range(Y.shape[0]) if genes is None else genes
    u = 0.0
    for g in genes:
        mu = np.maximum(np.exp(p["X"] @ A[g] + p["offset"]), nb_mu_floor())
        u += float((p["w1"] * (Y[g] - mu) / (1 + fit.psi[g] * mu)).sum())
    return u


def test_polished_fit_is_at_each_genes_optimum(ref):
    a = quiet(seeded(ref), tol=1e-12)
    f, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    assert is_polished(f) and not is_polished(f0)
    np.testing.assert_array_equal(f.alpha[:, 0], f0.alpha[:, 0])  # a1 untouched
    assert scaled_score(f, ref["Y"]).max() < 1e-6
    np.testing.assert_array_equal(f.psi, f0.psi)  # psi_method = "fixed"
    # the null is at its own optimum too
    assert scaled_score(load_fit(a, "spanorm_null"), ref["Y"]).max() < 1e-6


def test_polishing_raises_every_genes_likelihood(ref):
    a = quiet(seeded(ref))
    f, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    for g in range(a.n_vars):
        l1, l0 = total_loglik(f, ref["Y"], [g]), total_loglik(f0, ref["Y"], [g])
        assert l1 >= l0 - 1e-8 * abs(l0)


def test_logcounts_rewritten_and_consistent_with_spanorm_reuse(ref):
    a = seeded(ref)
    before = np.array(a.layers["logcounts"]) if "logcounts" in a.layers else None
    assert before is None
    quiet(a)
    lc = a.layers["logcounts"].copy()
    assert np.isfinite(lc).all()
    # the same result as spanorm() re-normalising with the polished fit cached
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(a, df_tps=2, sample_p=0.25, verbose=False)
    np.testing.assert_allclose(a.layers["logcounts"], lc, rtol=1e-10)
    assert is_polished(load_fit(a))


def test_non_integer_counts_refused(ref):
    a = seeded(ref)
    a.X = a.X + 0.5
    with pytest.raises(ValueError, match="integer counts"):
        polish_spanorm(a, verbose=False)


def test_unshared_library_size_refused_and_object_untouched(ref):
    a = seeded(ref, with_null=False)
    f = load_fit(a)
    alpha = f.alpha.copy()
    alpha[:, 0] += np.linspace(0, 0.1, alpha.shape[0])
    a.varm["spanorm_alpha"] = alpha
    before = a.copy()
    with pytest.raises(ValueError, match=r"column 1 .*not shared"):
        polish_spanorm(a, verbose=False)
    assert "spanorm_unpolished" not in a.uns["spanorm"] and "logcounts" not in a.layers
    np.testing.assert_array_equal(a.varm["spanorm_alpha"], before.varm["spanorm_alpha"])


def test_settings_and_diagnostics_recorded(ref):
    a = quiet(seeded(ref))
    f, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    s, genes = f.polish["settings"], f.polish["genes"]
    assert {"psi_method", "ls", "cells", "maxit", "tol", "pen", "a1_input", "a1", "spanorm"} <= set(s)
    assert (s["psi_method"], s["ls"], s["cells"]) == ("fixed", "fixed", "all")
    np.testing.assert_allclose(s["pen"], pm.spanorm_penalty(f0))
    assert s["a1"] == f0.alpha[0, 0] and s["a1_input"] == s["a1"]
    assert list(genes.index) == list(a.var_names)
    assert {"iterations", "psi_fitnb", "restarted", "capped", "singular", "psi_bound",
            "polished", "loglik", "held_out"} <= set(genes.columns)
    assert np.isfinite(genes["loglik"]).all()
    np.testing.assert_array_equal(f.loglik, f0.loglik)  # the fit's own trace is left alone
    f.validate()
    # joint-only keys absent; profile records psi_range
    assert "ls_iterations" not in s and "psi_range" not in s
    a2 = quiet(seeded(ref, with_null=False), psi_method="profile", psi_range=(1e-2, 1e2))
    np.testing.assert_array_equal(load_fit(a2).polish["settings"]["psi_range"], [1e-2, 1e2])


def test_already_polished_refused_unless_overwrite(ref):
    a = seeded(ref)
    f_in = load_fit(a)
    quiet(a, psi_method="profile")
    with pytest.raises(ValueError, match="already polished.*overwrite"):
        polish_spanorm(a, verbose=False)
    f1 = load_fit(a)
    quiet(a, overwrite=True)
    f2 = load_fit(a)
    # the unpolished fit still holds the original, never a polished one
    u = load_fit(a, "spanorm_unpolished")
    assert not is_polished(u)
    np.testing.assert_array_equal(u.alpha, f_in.alpha)
    assert f2.polish["settings"]["psi_method"] == "fixed"
    assert not np.allclose(f1.psi, f_in.psi)
    np.testing.assert_array_equal(f2.psi, f_in.psi)  # "fixed" keeps the ORIGINAL dispersion
    np.testing.assert_array_equal(f2.polish["genes"]["psi_fitnb"], f_in.psi)
    # and the result is the one-shot polish of the original (and R's overwrite run)
    b = quiet(seeded(ref))
    np.testing.assert_array_equal(load_fit(b).alpha, f2.alpha)
    compare_run(a, ref["a"]["overwrite"])
    assert "already polished" in ref["a"]["refused_msg"]


def test_overwrite_without_matching_unpolished_is_refused(ref):
    a = quiet(seeded(ref, with_null=False))
    store_fit(a, load_fit(a), "spanorm_unpolished")  # a polished fit under the unpolished name
    with pytest.raises(ValueError, match="nothing to polish from"):
        polish_spanorm(a, overwrite=True, verbose=False)


def test_null_polished_alike_and_svg_columns_dropped(ref):
    a = seeded(ref)
    n0 = load_fit(a, "spanorm_null")
    with pytest.warns(UserWarning, match="SVG"):
        polish_spanorm(a, tol=1e-12, verbose=False)
    nul, full = load_fit(a, "spanorm_null"), load_fit(a)
    assert is_polished(nul)
    np.testing.assert_array_equal(load_fit(a, "spanorm_null_unpolished").alpha, n0.alpha)
    keys = ("psi_method", "ls", "cells")
    assert [nul.polish["settings"][k] for k in keys] == [full.polish["settings"][k] for k in keys]
    assert not any(c in a.var for c in ("svg_F", "svg_p", "svg_fdr"))


def test_unknown_kwargs_warn_and_args_validated(ref):
    a = seeded(ref, with_null=False, with_svg=False)
    with pytest.warns(UserWarning, match="unrecognised"):
        polish_spanorm(a, verbose=False, bogus=1)
    with pytest.raises(ValueError, match="psi_method"):
        polish_spanorm(a, psi_method="x", verbose=False)
    with pytest.raises(ValueError, match="'ls'"):
        polish_spanorm(a, ls="x", verbose=False)
    with pytest.raises(ValueError, match="'cells'"):
        polish_spanorm(a, cells="x", verbose=False)


def test_missing_fit_error(ref):
    a = make_adata(ref["Y"], ref["coords"], ref["genes"])
    with pytest.raises(ValueError, match="spanorm model not found"):
        polish_spanorm(a, verbose=False)


# --------------------------------------------------------- containers, views, storage


def test_copy_view_and_h5ad_round_trip(ref, tmp_path):
    a = seeded(ref)
    out = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = polish_spanorm(a, verbose=False, copy=True)
    assert "logcounts" not in a.layers and not is_polished(load_fit(a))  # input untouched
    assert is_polished(load_fit(out))
    # a gene-subset view is converted to an actual AnnData without ImplicitModificationWarning
    b = seeded(ref, with_svg=False)
    view = b[:, :25]
    assert view.is_view
    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        polish_spanorm(view, verbose=False, null=False)
    assert not view.is_view and is_polished(load_fit(view)) and not is_polished(load_fit(b))
    # held_out has missing values; the polished adata must survive an h5ad round trip
    out.write_h5ad(tmp_path / "p.h5ad")
    back = anndata.read_h5ad(tmp_path / "p.h5ad")
    fb, fo = load_fit(back), load_fit(out)
    np.testing.assert_array_equal(fb.alpha, fo.alpha)
    assert fb.polish["settings"]["ls"] == "fixed"
    assert pd.isna(fb.polish["genes"]["held_out"]).all()


def test_sparse_counts_match_dense(ref):
    d = quiet(seeded(ref, with_null=False))
    s = seeded(ref, with_null=False)
    s.X = sp.csr_matrix(s.X)
    quiet(s)
    np.testing.assert_allclose(
        load_fit(s).alpha, load_fit(d).alpha, rtol=1e-10, atol=1e-12
    )
    assert sp.issparse(s.layers["logcounts"])
    np.testing.assert_allclose(s.layers["logcounts"].toarray(), d.layers["logcounts"], rtol=1e-10)


# ------------------------------------------------------------ staleness (ledger rulings)


def test_gene_subset_reuses_stored_fits(ref):
    a = seeded(ref)
    keep = np.arange(0, 40, 2)
    sub = a[:, keep].copy()
    assert not fit_is_stale_for_spots(sub, "spanorm")
    quiet(sub)
    f = load_fit(sub)
    assert is_polished(f) and f.ngenes == len(keep)
    # the genes are polished independently, so they equal the full run's genes
    full = quiet(seeded(ref))
    np.testing.assert_allclose(f.alpha, load_fit(full).alpha[keep], rtol=1e-8, atol=1e-12)


def test_spot_subset_is_stale_and_refused_like_r(ref):
    a = seeded(ref)
    sub = a[np.arange(0, 400)].copy()
    assert fit_is_stale_for_spots(sub, "spanorm") and fit_is_stale_for_spots(sub, "spanorm_null")
    assert not fit_is_stale_for_spots(a, "spanorm") and not fit_is_stale_for_spots(a, "absent")
    with pytest.raises(ValueError, match=r"spanorm model dimensions do not match the data\. Please rerun 'spanorm'"):
        get_spanorm_fit(sub)
    with pytest.raises(ValueError, match="dimensions do not match"):
        polish_spanorm(sub, verbose=False)
    # nothing was written
    assert "logcounts" not in sub.layers and "spanorm_unpolished" not in sub.uns["spanorm"]


def test_gene_plus_spot_subset_is_stale(ref):
    a = seeded(ref)
    sub = a[np.arange(0, 400), :20].copy()
    assert fit_is_stale_for_spots(sub, "spanorm")
    with pytest.raises(ValueError, match="dimensions do not match"):
        polish_spanorm(sub, verbose=False)


def test_stale_null_refused_when_polishing_null_only(ref):
    a = seeded(ref)
    # only the null is stale (e.g. refitted after a spot subset elsewhere)
    a.uns["spanorm"]["spanorm_null"]["ncells"] = a.n_obs + 7
    with pytest.raises(ValueError, match=r"spanorm_null model dimensions do not match.*spanorm_svg"):
        polish_spanorm(a, verbose=False)
    quiet(a, null=False)  # null = False never validates the null
    assert is_polished(load_fit(a))


def test_spot_reorder_reuses_the_fit(ref):
    a = seeded(ref)
    perm = np.random.default_rng(3).permutation(a.n_obs)
    b = a[perm].copy()
    assert not fit_is_stale_for_spots(b, "spanorm")
    quiet(b)
    base = quiet(seeded(ref))
    fb, f0 = load_fit(b), load_fit(base)
    # genes are converged over the same set of spots, so only summation order differs
    np.testing.assert_allclose(fb.alpha, f0.alpha, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fb.polish["genes"]["loglik"], f0.polish["genes"]["loglik"], rtol=1e-9)
    np.testing.assert_allclose(b.layers["logcounts"], base.layers["logcounts"][perm], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------- model mapping (ModelR)


def mk_fit(shared=True):
    rng = np.random.default_rng(5)
    n, G = 50, 4
    W = rng.normal(size=(n, 4))
    al = rng.normal(size=(G, 4))
    al[:, 0] = 1.02 if shared else np.linspace(0.9, 1.1, G)
    return SpaNormFit(
        ngenes=G, ncells=n, gene_model="nb", df_tps=np.array([2, 2, 2, 2]), sample_p=0.5,
        lambda_a=np.array([1e-4, 2e-4]), batch=None, W=W, alpha=al, gmean=rng.normal(size=G),
        psi=np.full(G, 0.2), wtype=np.array(["ls", "biology", "biology", "ls"]), loglik=np.zeros(1),
        sampling=np.repeat(["glm", "all"], n // 2),
    )


def test_problem_maps_spanorm_model():
    f = mk_fit()
    p = pm.spanorm_polish_problem(f)
    np.testing.assert_array_equal(p["X"][:, 0], np.ones(50))
    np.testing.assert_allclose(p["pen"], [0, 1e-4 * 50, 1e-4 * 50, 2e-4 * 50])
    np.testing.assert_allclose(p["offset"], 1.02 * f.W[:, 0])
    np.testing.assert_allclose(p["A0"], np.column_stack([f.gmean, f.alpha[:, 1:]]))
    assert pm.spanorm_polish_problem(f, "fit")["cells_idx"].sum() == 25
    np.testing.assert_array_equal(
        pm.spanorm_polish_problem(f, "fit")["cells_idx"], f.sampling != "all"
    )
    assert pm.spanorm_polish_problem(f, "fit")["X"].shape[0] == 25
    with pytest.raises(ValueError, match="not .*shared"):
        pm.spanorm_polish_problem(mk_fit(shared=False))


def test_validity_of_polish_slot():
    f = mk_fit()
    assert not is_polished(f)
    f.polish = {"settings": {"psi_method": "fixed"}, "genes": pd.DataFrame()}
    assert f.validate()
    f.polish = {"settings": {"psi_method": "fixed"}}
    with pytest.raises(ValueError, match="polish"):
        f.validate()
    f.polish = {"nonsense": 1}
    with pytest.raises(ValueError, match="polish"):
        f.validate()


def test_hold_out_groups_from_unpenalised_indicator_columns():
    X = np.column_stack([
        np.ones(6), [-1.2, 0.3, 0.8, -0.4, 1.1, -0.6], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]
    ])
    pen = np.array([0, 1, 0, 0.0])
    np.testing.assert_array_equal(pm.unpenalised_groups(X, pen), [1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(pm.unpenalised_groups(X, np.array([0, 1, 1, 1.0])), np.ones(6))
    assert pm.unpenalised_groups(X, np.zeros(4)) is None
    Y = np.array([[1, 2, 3, 4, 5, 6], [0] * 6, [1, 1, 0, 0, 2, 0], [0, 0, 1, 0, 0, 1]], float)
    want = [None, "all-zero", "zero-batch-level", "zero-batch-level"]
    assert list(pm.polish_hold_out(Y, X, pen)) == want
    assert list(pm.polish_hold_out(sp.csr_matrix(Y), X, pen)) == want
    assert list(pm.polish_hold_out(Y, X, np.zeros(4))) == [None, "all-zero", None, None]
    assert list(pm.polish_hold_out(Y, X, np.array([0, 1, 1, 1.0]))) == [None, "all-zero", None, None]


def test_rank_deficiency():
    n = 30
    b = np.tile([0.0, 1.0], n // 2)
    X = np.column_stack([np.ones(n), np.arange(n), b, 1 - b])  # indicators sum to the intercept
    assert pm.unpenalised_rank_deficient(X, np.array([0, 1, 0, 0.0]))
    assert not pm.unpenalised_rank_deficient(X, np.array([0, 1, 0, 1.0]))  # ridge on one
    assert not pm.unpenalised_rank_deficient(X[:, :2], np.zeros(2))
    assert not pm.unpenalised_rank_deficient(X[:, :1], np.zeros(1))  # intercept alone
    Z = np.column_stack([np.ones(n), np.zeros(n)])  # an all-zero unpenalised column
    assert pm.unpenalised_rank_deficient(Z, np.zeros(2))


def test_psi_range_default():
    np.testing.assert_array_equal(pm.polish_psi_range({}), [1e-3, 1e3])
    np.testing.assert_array_equal(pm.polish_psi_range({"psi_range": [1e-2, 5]}), [1e-2, 5])


# ----------------------------------------------------------------------- joint a1 (JointR)


def sim_problem(seed=30, n=120, G=5):
    rng = np.random.default_rng(seed)
    w1 = np.log(rng.uniform(0.5, 2, n))
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x2])
    pen = np.array([0, 2.0])
    mu = np.exp(0.3 + (np.arange(1, G + 1) / 5)[:, None] + 0.8 * w1 + 0.3 * x2)
    Y = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
    prob = dict(X=X, pen=pen, a1=1.0, offset=1.0 * w1, w1=w1, A0=np.zeros((G, 2)),
                psi=np.full(G, 0.2), cells_idx=np.ones(n, bool))
    return Y, prob


def test_joint_a1_matches_full_optimisation():
    Y, prob = sim_problem()
    G, n = Y.shape
    X, pen, w1, psi = prob["X"], prob["pen"], prob["w1"], prob["psi"]

    def negobj(par):
        a1, A = par[0], par[1:].reshape(G, 2)
        tot = 0.0
        for g in range(G):
            mu = np.exp(X @ A[g] + a1 * w1)
            tot += float(dnbinom(Y[g], mu=mu, size=1 / psi[g], log=True).sum()) - 0.5 * (pen * A[g] ** 2).sum()
        return -tot

    def grad(par):
        a1, A = par[0], par[1:].reshape(G, 2)
        ga, gA = 0.0, np.zeros((G, 2))
        for g in range(G):
            mu = np.exp(X @ A[g] + a1 * w1)
            r = (Y[g] - mu) / (1 + psi[g] * mu)
            ga += (w1 * r).sum()
            gA[g] = X.T @ r - pen * A[g]
        return -np.concatenate([[ga], gA.ravel()])

    opt = optimize.minimize(negobj, np.r_[1, np.zeros(2 * G)], jac=grad, method="BFGS",
                            options=dict(maxiter=5000, gtol=1e-10))
    assert np.abs(grad(opt.x)).max() < 1e-3  # the oracle itself converged
    pol = polish_nb(Y, X, prob["A0"], psi, lambda_a=pen, offset=prob["offset"], psi_method="fixed")
    j = pm.polish_shared_ls(Y, prob, pol, psi_method="fixed", tol_ls=1e-8)
    np.testing.assert_allclose(j["a1"], opt.x[0], rtol=1e-4)
    np.testing.assert_allclose(j["pol"]["alpha"], opt.x[1:].reshape(G, 2), rtol=1e-3, atol=1e-6)
    assert abs(j["score"]) < 1e-6 * Y.sum()
    assert j["converged"] and j["iterations"] > 0 and j["singular"] == 0
    assert np.isfinite(j["se"]) and j["se"] > 0


def test_joint_never_lowers_total_and_is_stationary(ref):
    fx = load_fit(quiet(seeded(ref, with_null=False), ls="fixed", tol=1e-12))
    a = quiet(seeded(ref, with_null=False), ls="joint", tol=1e-12)
    jt, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    Y = ref["Y"]
    assert total_loglik(jt, Y) >= total_loglik(fx, Y) - 1e-8 * abs(total_loglik(fx, Y))
    assert len(np.unique(jt.alpha[:, 0])) == 1
    s = jt.polish["settings"]
    assert s["ls"] == "joint" and s["a1_input"] == f0.alpha[0, 0] and s["a1"] == jt.alpha[0, 0]
    assert s["a1"] != s["a1_input"]
    U = a1_score(jt, Y)
    assert abs(U) * s["ls_se"] < 1e-3
    assert abs(s["ls_score"] - U) < 1e-6 * abs(U) + 1e-8
    assert abs(s["ls_score"]) * s["ls_se"] < 1e-3  # the stopping rule
    assert s["ls_converged"] and 1 <= s["ls_iterations"] <= 10 and s["ls_singular"] == 0
    assert scaled_score(jt, Y).max() < 1e-6
    np.testing.assert_array_equal(jt.psi, f0.psi)
    genes = jt.polish["genes"]
    np.testing.assert_allclose(genes["loglik"].sum(), total_loglik(jt, Y), rtol=1e-10)
    assert genes["polished"].all()


def test_ls_se_is_profiled_standard_error(ref):
    a = quiet(seeded(ref, with_null=False), ls="joint", tol=1e-12)
    f = load_fit(a)
    s = f.polish["settings"]
    p = pm.spanorm_polish_problem(f)
    A = np.column_stack([f.gmean, f.alpha[:, 1:]])

    def prof(a1):
        pl = polish_nb(ref["Y"], p["X"], A, f.psi, lambda_a=p["pen"], offset=a1 * p["w1"],
                       psi_method="fixed", tol=1e-14)
        return float(np.sum(pl["loglik"]))

    h = 5 * s["ls_se"]
    curv = -(prof(s["a1"] + h) - 2 * prof(s["a1"]) + prof(s["a1"] - h)) / h**2
    # Fisher (expected) against observed information: close, not equal
    np.testing.assert_allclose(1 / s["ls_se"] ** 2, curv, rtol=0.05)


def test_joint_all_zero_gene_shares_new_a1(ref):
    Y = ref["Y"].copy()
    Y[0, :] = 0
    a = make_adata(Y, ref["coords"], ref["genes"])
    store_fit(a, fit_from_ref(ref["a"]["input"]), "spanorm")
    polish_spanorm(a, ls="joint", verbose=False, null=False)
    f, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    s, pol = f.polish["settings"], f.polish["genes"]
    assert not pol["polished"].iloc[0] and pol["polished"].iloc[1:].all()
    assert f.gmean[0] == f0.gmean[0] and f.psi[0] == f0.psi[0]
    np.testing.assert_array_equal(f.alpha[0, 1:], f0.alpha[0, 1:])
    assert len(np.unique(f.alpha[:, 0])) == 1 and f.alpha[0, 0] == s["a1"] != s["a1_input"]
    U = a1_score(f, Y, genes=range(1, Y.shape[0]))
    assert abs(U) * s["ls_se"] < 1e-3
    assert abs(s["ls_score"] - U) < 1e-6 * abs(U) + 1e-8


def test_joint_with_no_polished_gene_warns_and_keeps_a1(ref):
    f0 = fit_from_ref(ref["a"]["input"])
    Y0 = np.zeros((f0.ngenes, f0.ncells))
    with pytest.warns(UserWarning, match="nothing informs"):
        f = _polish_fit(f0, Y0, ls="joint")
    s = f.polish["settings"]
    assert s["ls"] == "joint" and s["ls_iterations"] == 0 and s["a1"] == s["a1_input"] == f0.alpha[0, 0]
    assert {"ls_score", "ls_se", "ls_singular", "ls_converged", "ls_maxit", "ls_tol"} <= set(s)
    assert s["ls_maxit"] == 10 and s["ls_tol"] == 1e-3 and s["ls_singular"] == 0 and not s["ls_converged"]
    np.testing.assert_array_equal(f.alpha, f0.alpha)
    np.testing.assert_array_equal(f.gmean, f0.gmean)
    np.testing.assert_array_equal(f.psi, f0.psi)
    assert not f.polish["genes"]["polished"].any()

    # and the same when the genes are passed but the cold polish converged none
    rng = np.random.default_rng(4)
    n = 50
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    w1 = rng.normal(scale=0.3, size=n)
    Y = rng.poisson(5, size=(3, n)).astype(float)
    prob = dict(X=X, pen=np.array([0, 1.0]), a1=0.7, offset=0.7 * w1, w1=w1, A0=np.zeros((3, 2)),
                psi=np.full(3, 0.1), cells_idx=np.ones(n, bool))
    pol = dict(alpha=prob["A0"], psi=prob["psi"], loglik=np.full(3, np.nan),
               polish=pd.DataFrame(dict(iterations=0, psi_fitnb=prob["psi"], restarted=False,
                                        capped=False, singular=True, psi_bound=False, polished=False)))
    with pytest.warns(UserWarning, match="nothing informs"):
        j = pm.polish_shared_ls(Y, prob, pol)
    assert j["a1"] == 0.7 and j["iterations"] == 0 and j["pol"] is pol


def _polish_fit(f0, Y, **kw):
    from spanormpy.polish_spanorm import polish_spanorm_fit

    return polish_spanorm_fit(f0, Y, **kw)


def test_singular_gene_left_out_of_both_U_and_I():
    rng = np.random.default_rng(8)
    n = 80
    ind = np.repeat([0.0, 1.0], n // 2)
    X = np.column_stack([np.ones(n), ind])
    pen = np.zeros(2)
    w1 = rng.normal(scale=0.3, size=n)
    A = np.array([[-40, 54], [1, 0.2], [0.5, -0.1]], float)
    psi = np.array([0, 0.1, 0.2])
    mu1 = np.maximum(np.exp(A[0, 0] + A[0, 1] * ind + 0.9 * w1), nb_mu_floor())
    Y = np.vstack([
        rng.poisson(mu1),
        rng.negative_binomial(10, 10 / (10 + np.exp(A[1, 0] + A[1, 1] * ind + 0.9 * w1))),
        rng.negative_binomial(5, 5 / (5 + np.exp(A[2, 0] + A[2, 1] * ind + 0.9 * w1))),
    ]).astype(float)
    solver = nb_newton_solver(X, pen)
    import torch

    mu1t = torch.as_tensor(np.maximum(np.exp(X @ A[0] + 0.9 * w1), nb_mu_floor()))
    assert solver.solve(solver.factor(mu1t), torch.ones(2, dtype=torch.float64)) is None  # premise

    si = pm.ls_score_info(Y, A, psi, 0.9, X, w1, pen, solver)
    ref_ = pm.ls_score_info(Y[1:], A[1:], psi[1:], 0.9, X, w1, pen, solver)
    assert si["excluded"] == [0] and si["n"] == 2 and ref_["n"] == 2
    np.testing.assert_allclose(si["U"], ref_["U"], rtol=1e-12)
    np.testing.assert_allclose(si["I"], ref_["I"], rtol=1e-12)
    sb = pm.ls_score_info(Y, A, psi, 0.9, X, w1, pen, solver, block=1)
    for k in ("U", "I", "n"):
        np.testing.assert_allclose(sb[k], si[k], rtol=1e-12)
    assert sb["excluded"] == si["excluded"]

    prob = dict(X=X, pen=pen, a1=0.9, offset=0.9 * w1, w1=w1, A0=A, psi=psi, cells_idx=np.ones(n, bool))

    def mkpol(rows):
        return dict(alpha=A[rows].copy(), psi=psi[rows].copy(), loglik=np.zeros(len(A[rows])),
                    polish=pd.DataFrame(dict(iterations=1, psi_fitnb=psi[rows], restarted=False,
                                             capped=False, singular=False, psi_bound=False,
                                             polished=True), index=range(len(A[rows]))))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        j = pm.polish_shared_ls(Y, prob, mkpol(slice(None)), psi_method="fixed")
    assert j["converged"] and j["singular"] == 1
    np.testing.assert_array_equal(j["pol"]["polish"]["singular"].to_numpy(), [True, False, False])
    np.testing.assert_array_equal(j["pol"]["alpha"][0], A[0])  # held where it was
    prob2 = dict(prob, A0=A[1:], psi=psi[1:])
    j2 = pm.polish_shared_ls(Y[1:], prob2, mkpol(slice(1, 3)), psi_method="fixed")
    np.testing.assert_allclose(j["a1"], j2["a1"], rtol=1e-10)
    np.testing.assert_allclose(j["pol"]["alpha"][1:], j2["pol"]["alpha"], rtol=1e-10)
    assert j["iterations"] == j2["iterations"]
    np.testing.assert_allclose(j["score"], j2["score"], rtol=1e-8, atol=1e-10)


def test_joint_reports_cap():
    Y, prob = sim_problem()
    prob = dict(prob, offset=prob["w1"])
    pol = polish_nb(Y, prob["X"], prob["A0"], prob["psi"], lambda_a=prob["pen"],
                    offset=prob["offset"], psi_method="fixed")
    with pytest.warns(UserWarning, match="maxit_ls = 1 reached"):
        j = pm.polish_shared_ls(Y, prob, pol, maxit_ls=1)
    assert j["iterations"] == 1 and not j["converged"]
    assert np.sum(j["pol"]["loglik"]) > np.sum(pol["loglik"])  # one step is still an ascent
    assert (j["pol"]["polish"]["iterations"].to_numpy() > pol["polish"]["iterations"].to_numpy()).all()


def test_profiled_information_matches_schur_complement():
    rng = np.random.default_rng(12)
    n, G = 60, 7
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    pen = np.array([0, 0.5, 3.0])
    w1 = rng.normal(scale=0.4, size=n)
    A = np.column_stack([rng.normal(1, 1, G), rng.normal(0, 0.2, G), rng.normal(0, 0.2, G)])
    psi = rng.uniform(0.05, 0.5, G)
    mu = np.exp(A @ X.T + 0.8 * w1)
    Y = rng.negative_binomial(1 / psi[:, None], 1 / (1 + psi[:, None] * mu)).astype(float)
    si = pm.ls_score_info(Y, A, psi, 0.8, X, w1, pen, nb_newton_solver(X, pen), block=3)
    U = I = 0.0
    for g in range(G):
        m = np.exp(X @ A[g] + 0.8 * w1)
        d = m / (1 + psi[g] * m)
        U += (w1 * (Y[g] - m) / (1 + psi[g] * m)).sum()
        Xa = np.column_stack([X, w1])
        M = (Xa * d[:, None]).T @ Xa + np.diag(np.r_[pen, 0])
        I += M[3, 3] - M[3, :3] @ np.linalg.solve(M[:3, :3], M[:3, 3])
    np.testing.assert_allclose(si["U"], U, rtol=1e-10)
    np.testing.assert_allclose(si["I"], I, rtol=1e-10)
    assert si["n"] == 7


def test_joint_profile_is_stationary_at_reported_psi(ref):
    base = seeded(ref, with_null=False)
    a = quiet(base, ls="joint", psi_method="profile", tol=1e-12)
    f, f0 = load_fit(a), load_fit(a, "spanorm_unpolished")
    s = f.polish["settings"]
    Y = ref["Y"]
    assert s["ls_converged"]
    U = a1_score(f, Y)
    assert abs(U) * s["ls_se"] < 1e-3
    assert abs(s["ls_score"] - U) < 1e-6 * abs(U) + 1e-8
    assert scaled_score(f, Y).max() < 1e-6
    assert not np.allclose(f.psi, f0.psi)
    fx = load_fit(quiet(seeded(ref, with_null=False), psi_method="profile", tol=1e-12))
    assert total_loglik(f, Y) >= total_loglik(fx, Y) - 1e-8 * abs(total_loglik(fx, Y))


def test_null_polished_jointly_at_its_own_a1(ref):
    a = seeded(ref)
    with pytest.warns(UserWarning, match="SVG"):
        polish_spanorm(a, ls="joint", verbose=False)
    sf = load_fit(a).polish["settings"]
    sn = load_fit(a, "spanorm_null").polish["settings"]
    assert sn["ls"] == "joint" and sn["a1"] != sn["a1_input"] and sn["a1"] != sf["a1"]
    assert len(np.unique(load_fit(a, "spanorm_null").alpha[:, 0])) == 1


def test_joint_stop_is_its_own_not_the_per_gene_one(ref):
    a = quiet(seeded(ref, with_null=False), ls="joint", psi_method="profile", tol=1e-12)
    s = load_fit(a).polish["settings"]
    assert s["ls_tol"] == 1e-3 and s["ls_maxit"] == 10 and s["ls_iterations"] <= 10
    assert s["tol"] == 1e-12 and s["ls_converged"]
    assert abs(s["ls_score"]) * s["ls_se"] < 1e-3


def test_warm_repolishes_get_callers_maxit_and_tol(ref, monkeypatch):
    real = pm.polish_nb
    seen = []

    def spy(*a, **k):
        seen.append(dict(warm=bool(k.get("warm")), maxit=k.get("maxit"), tol=k.get("tol")))
        return real(*a, **k)

    monkeypatch.setattr(pm, "polish_nb", spy)
    a = quiet(seeded(ref, with_null=False), ls="joint", psi_method="profile", maxit=30, tol=1e-5)
    warm = [x for x in seen if x["warm"]]
    cold = [x for x in seen if not x["warm"]]
    assert warm and cold
    assert all(x["maxit"] == 30 and x["tol"] == 1e-5 for x in warm + cold)
    s = load_fit(a).polish["settings"]
    assert (s["maxit"], s["tol"], s["ls_maxit"], s["ls_tol"]) == (30, 1e-5, 10, 1e-3)


def test_non_positive_information_warns_and_keeps_fixed_polish(ref, monkeypatch):
    Y, prob = sim_problem()
    pol = polish_nb(Y, prob["X"], prob["A0"], prob["psi"], lambda_a=prob["pen"],
                    offset=prob["offset"], psi_method="fixed")
    real = pm.ls_score_info
    with monkeypatch.context() as m:
        m.setattr(pm, "ls_score_info", lambda Y_, A, *a, **k: dict(U=5.0, I=-1.0, excluded=[], n=len(A)))
        with pytest.warns(UserWarning, match="not positive"):
            j = pm.polish_shared_ls(Y, prob, pol)
        assert j["a1"] == 1.0 and j["pol"] is pol and j["iterations"] == 0
        assert not j["converged"] and np.isnan(j["se"])
    # non-positive after a step: the cold pass is still what is returned
    calls = []

    def later(Y_, A, *a, **k):
        calls.append(1)
        return real(Y_, A, *a, **k) if len(calls) == 1 else dict(U=5.0, I=-1.0, excluded=[], n=len(A))

    with monkeypatch.context() as m:
        m.setattr(pm, "ls_score_info", later)
        with pytest.warns(UserWarning, match="not positive"):
            j = pm.polish_shared_ls(Y, prob, pol)
        assert j["a1"] == 1.0 and j["pol"] is pol and j["iterations"] == 1 and not j["converged"]

    # through polish_spanorm: the fit is the fixed-a1 polish, with the record
    fx = load_fit(quiet(seeded(ref, with_null=False)))
    a = seeded(ref, with_null=False, with_svg=False)
    with monkeypatch.context() as m:
        m.setattr(pm, "ls_score_info", lambda Y_, A, *a_, **k: dict(U=5.0, I=-1.0, excluded=[], n=len(A)))
        with pytest.warns(UserWarning, match="not positive"):
            polish_spanorm(a, ls="joint", verbose=False)
    f = load_fit(a)
    s = f.polish["settings"]
    assert s["ls"] == "joint" and not s["ls_converged"] and s["ls_iterations"] == 0
    assert s["a1"] == s["a1_input"]
    for k in ("gmean", "alpha", "psi"):
        np.testing.assert_array_equal(getattr(f, k), getattr(fx, k))
    pd.testing.assert_frame_equal(f.polish["genes"], fx.polish["genes"])


def test_default_tol_ls_converges_and_matches_tight_stop(ref, monkeypatch):
    a = quiet(seeded(ref, with_null=False), ls="joint")
    s = load_fit(a).polish["settings"]
    assert s["ls_tol"] == 1e-3 and s["ls_converged"] and s["ls_iterations"] < 10
    monkeypatch.setattr(pm, "LS_TOL", 1e-9)
    t = quiet(seeded(ref, with_null=False), ls="joint")
    st = load_fit(t).polish["settings"]
    assert st["ls_converged"]
    assert abs(s["a1"] - st["a1"]) < 1e-3 * s["ls_se"]
