"""spanorm_svg / top_svgs / svg_test: R parity and the ported testthat cases.

R sources: tests/testthat/test-mainSpaNormSVG.R (SpaNorm c12e5e5). References are in
tests/ref/make_svg.R. Python seeds ``spanorm`` / ``spanorm_null`` from R's *input* fits and
injects R's dispersion subset into the null fit, so the null fit and every statistic compare
at the iterative-fit tolerance (rtol 1e-6); p-values and FDRs compare on the log scale.
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats
from tests._ref import load_ref
from tests.test_polish_spanorm import compare_fit, fit_from_ref, make_adata

import spanormpy._fitnb as fitnb_mod
from spanormpy import polish_spanorm, spanorm, spanorm_svg, top_svgs
from spanormpy._fit import SpaNormFit, get_spanorm_fit, is_polished, load_fit, store_fit
from spanormpy.polish_spanorm import SVG_COLUMNS
from spanormpy.svg import (
    _p_adjust_fdr,
    _polished_alike,
    fit_spanorm_technical,
    svg_test,
)

RTOL = 1e-6  # iterative fits given identical inputs and R's sampling (spec tolerance)
ATOL = 1e-10  # entries that are exactly 0 up to rounding


@pytest.fixture(scope="module")
def ref():
    return load_ref("svg")


def inject_psi_sampling(monkeypatch, sampling):
    """Replace the dispersion-subset draw by R's, rebuilt from R's ``fit$sampling``."""
    sampling = np.asarray(sampling).astype(str)
    mask = sampling[sampling != "all"] == "dispersion"
    monkeypatch.setattr(fitnb_mod, "_sample_psi_idx", lambda idx, maxn, rng: mask)


def assert_logclose(py, r, rtol=RTOL):
    """p-values and FDRs compare on the log scale (they span many orders of magnitude)."""
    np.testing.assert_allclose(np.log(py), np.log(r), rtol=rtol, atol=1e-12)


def assert_svg_matches(adata, r, rtol=RTOL):
    np.testing.assert_allclose(adata.var["svg_F"], r["svg_F"], rtol=rtol, atol=ATOL)
    assert_logclose(adata.var["svg_p"], r["svg_p"], rtol)
    assert_logclose(adata.var["svg_fdr"], r["svg_fdr"], rtol)


def assert_null_matches(adata, r, rtol=RTOL):
    fit = load_fit(adata, "spanorm_null")
    np.testing.assert_allclose(fit.alpha, r["alpha"], rtol=rtol, atol=ATOL)
    np.testing.assert_allclose(fit.gmean, r["gmean"], rtol=rtol, atol=ATOL)
    np.testing.assert_allclose(fit.psi, r["psi"], rtol=rtol)
    # atol: with size factors recovered as exp(W[:, 0]) the design is rebuilt to ~1e-16 absolute
    np.testing.assert_allclose(fit.W, r["W"], rtol=1e-12, atol=1e-12)
    assert list(fit.wtype) == list(np.atleast_1d(r["wtype"]))
    assert list(fit.colnames) == list(np.atleast_1d(r["W_colnames"]))
    np.testing.assert_array_equal(fit.sampling, r["sampling"])
    np.testing.assert_allclose(fit.loglik, np.atleast_1d(r["loglik"]), rtol=rtol)


def seed_dlpfc(adata, r):
    store_fit(adata, fit_from_ref(r["input"]), "spanorm")
    return adata


# ------------------------------------------------------------------ R parity


def test_svg_matches_r(dlpfc_small, ref, monkeypatch, device):
    backend = "cpu" if device == "cpu" else "gpu"
    r = ref["base"]
    assert list(dlpfc_small.var_names) == list(ref["genes"])
    seed_dlpfc(dlpfc_small, r)
    inject_psi_sampling(monkeypatch, r["null"]["sampling"])
    spanorm_svg(dlpfc_small, backend=backend, verbose=False)
    assert_null_matches(dlpfc_small, r["null"])
    assert_svg_matches(dlpfc_small, r)
    # the unclamped statistic: svg_F = pmax(F_raw, 0)
    np.testing.assert_allclose(dlpfc_small.var["svg_F_raw"], r["F_raw"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        dlpfc_small.var["svg_F"], np.maximum(dlpfc_small.var["svg_F_raw"], 0), rtol=0, atol=0
    )
    # the full fit is untouched and no pre-polish null is stored for an unpolished pair
    assert not is_polished(load_fit(dlpfc_small, "spanorm"))
    assert load_fit(dlpfc_small, "spanorm_null_unpolished", validate=False) is None


def test_svg_test_matches_r_statistic(dlpfc_small, ref):
    """svg_test alone, on R's own pair of fits (no fitting involved): closed-form tolerance."""
    r = ref["base"]
    full, nul = fit_from_ref(r["input"]), fit_from_ref(r["null"])
    res = svg_test(np.asarray(dlpfc_small.X).T, full, nul)
    # F is a difference of two log-likelihood sums of order 1e4: summation order (torch vs R
    # rowSums) leaves ~1e-12 absolute noise, i.e. up to 5e-10 relative on an F of 0.01
    np.testing.assert_allclose(res["svg_F"], r["svg_F"], rtol=1e-8, atol=1e-10)
    assert_logclose(res["svg_p"], r["svg_p"], 1e-8)
    assert_logclose(res["svg_fdr"], r["svg_fdr"], 1e-8)
    np.testing.assert_allclose(res.attrs["F_raw"], r["F_raw"], rtol=1e-8, atol=1e-10)
    assert list(res.columns) == ["svg_F", "svg_p", "svg_fdr"]


def test_top_svgs_matches_r(dlpfc_small, ref, monkeypatch):
    r = ref["base"]
    seed_dlpfc(dlpfc_small, r)
    inject_psi_sampling(monkeypatch, r["null"]["sampling"])
    spanorm_svg(dlpfc_small, verbose=False)
    top = top_svgs(dlpfc_small, n=20, fdr=0.05)
    assert list(top.index) == list(np.atleast_1d(r["top_genes"]))
    assert list(top.columns) == ["svg_F", "svg_p", "svg_fdr"]
    np.testing.assert_allclose(top["svg_F"], r["top_F"], rtol=RTOL)
    assert_logclose(top["svg_p"], r["top_p"])
    assert_logclose(top["svg_fdr"], r["top_fdr"])
    assert list(top_svgs(dlpfc_small).index) == list(np.atleast_1d(r["top10_genes"]))


def polished_adata(ref, null_input=None, **polish_kw):
    """The 40 x 500 polish fixture: R's unpolished input fit, then Python's polish_spanorm."""
    pol = ref["pol"]
    a = make_adata(pol["Y"], pol["coords"], pol["genes"])
    store_fit(a, fit_from_ref(pol["input"]), "spanorm")
    if null_input is not None:
        store_fit(a, fit_from_ref(null_input), "spanorm_null")
    return a


def quiet_polish(a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polish_spanorm(a, verbose=False, **kw)
    return a


def check_polished_run(a, r, monkeypatch):
    inject_psi_sampling(monkeypatch, r["null"]["sampling"])
    spanorm_svg(a, backend="cpu", verbose=False)
    compare_fit(a, "spanorm_null", r["null"])
    assert_null_matches(a, r["null"])
    assert_svg_matches(a, r)
    return a


def test_paired_null_matches_r(ref, monkeypatch):
    """A polished full fit and no stored null: fit the null, then polish it alike."""
    r = ref["pol"]["p1"]
    a = quiet_polish(polished_adata(ref), null=False)
    check_polished_run(a, r, monkeypatch)
    # the pre-polish null is kept, as polish_spanorm keeps spanorm_null_unpolished
    compare_fit(a, "spanorm_null_unpolished", r["null_unpolished"])
    assert not is_polished(load_fit(a, "spanorm_null_unpolished"))
    assert _polished_alike(load_fit(a, "spanorm"), load_fit(a, "spanorm_null"))


def test_paired_null_stored_unpolished_null(ref, monkeypatch, capsys):
    """A stored unpolished null next to a polished full fit is polished to match."""
    pol = ref["pol"]
    a = polished_adata(ref, null_input=pol["unpolished"]["null"])
    quiet_polish(a, null=False)
    assert not is_polished(load_fit(a, "spanorm_null"))
    check_polished_run(a, pol["p2"], monkeypatch)
    err = capsys.readouterr().err
    assert "the stored null SpaNorm model is not polished; polishing it to match" in err


def test_paired_null_settings_mismatch(ref, monkeypatch, capsys):
    """The reviewer's sequence: polish both (fixed), then the full fit alone (profile)."""
    pol = ref["pol"]
    a = polished_adata(ref, null_input=pol["unpolished"]["null"])
    quiet_polish(a)
    quiet_polish(a, overwrite=True, null=False, psi_method="profile")
    assert load_fit(a, "spanorm_null").polish["settings"]["psi_method"] == "fixed"
    inject_psi_sampling(monkeypatch, pol["p3"]["null"]["sampling"])
    spanorm_svg(a, backend="cpu", verbose=False)
    err = capsys.readouterr().err
    assert "polished with different settings than the full model ('psi_method' differs" in err
    full, nul = load_fit(a, "spanorm"), load_fit(a, "spanorm_null")
    for k in ("psi_method", "ls", "cells"):
        assert nul.polish["settings"][k] == full.polish["settings"][k]
    assert nul.polish["settings"]["psi_method"] == "profile"
    compare_fit(a, "spanorm_null", pol["p3"]["null"])
    assert_svg_matches(a, pol["p3"])
    # the pre-polish null is the one the public sequence left behind, restored
    assert not is_polished(load_fit(a, "spanorm_null_unpolished"))
    # a pair polished consistently from scratch gives the same statistic
    np.testing.assert_allclose(a.var["svg_F"], pol["p3_ref"]["svg_F"], rtol=1e-8, atol=1e-8)


def test_psi_range_reaches_the_null(ref, monkeypatch):
    pol = ref["pol"]
    a = quiet_polish(polished_adata(ref), psi_method="profile", psi_range=(0.02, 0.2), null=False)
    check_polished_run(a, pol["p4"], monkeypatch)
    for name in ("spanorm", "spanorm_null"):
        np.testing.assert_array_equal(
            load_fit(a, name).polish["settings"]["psi_range"], [0.02, 0.2]
        )


def test_polish_maxit_tol_forwarded_to_the_null(ref, monkeypatch):
    a = quiet_polish(polished_adata(ref), null=False, maxit=7, tol=1e-5)
    check_polished_run(a, ref["pol"]["p5"], monkeypatch)
    fs = load_fit(a, "spanorm").polish["settings"]
    ns = load_fit(a, "spanorm_null").polish["settings"]
    assert fs["maxit"] == 7 and fs["tol"] == 1e-5
    assert ns["maxit"] == fs["maxit"] and ns["tol"] == fs["tol"]


def test_batch_design_zero_batch_level_gene(ref, monkeypatch):
    """A batch design with a gene absent from one level: held out of the null polish too."""
    bt = ref["batch"]
    g = int(bt["gene"]) - 1
    a = make_adata(bt["Y"], ref["pol"]["coords"], ref["pol"]["genes"])
    store_fit(a, fit_from_ref(bt["input"], batch=np.asarray(bt["batch"]).astype(str)), "spanorm")
    with pytest.warns(UserWarning, match="batch level"):
        polish_spanorm(a, null=False, verbose=False)
    inject_psi_sampling(monkeypatch, bt["run"]["null"]["sampling"])
    with pytest.warns(
        UserWarning, match=r"^1 gene with no counts in some batch level.*'spanorm_null'"
    ):
        spanorm_svg(a, backend="cpu", verbose=False)
    full, nul = load_fit(a, "spanorm"), load_fit(a, "spanorm_null")
    assert "batch" in nul.wtype and is_polished(nul) and _polished_alike(full, nul)
    assert not is_polished(load_fit(a, "spanorm_null_unpolished"))
    g_df = nul.polish["genes"]
    assert g_df["held_out"].iloc[g] == "zero-batch-level" and not g_df["polished"].iloc[g]
    assert g_df["polished"].drop(g_df.index[g]).all()
    assert np.isfinite(a.var["svg_F"]).all()
    compare_fit(a, "spanorm_null", bt["run"]["null"])
    assert_svg_matches(a, bt["run"])


# ------------------------------------------------------------------ default pipeline


def inject_default_sampling(monkeypatch, r):
    import importlib

    sn = importlib.import_module("spanormpy.spanorm")
    s = np.asarray(r["input"]["sampling"]).astype(str)
    glm = np.flatnonzero(s != "all")
    psi_mask = s[glm] == "dispersion"
    monkeypatch.setattr(sn, "_sample_random", lambda coords, nsub, rng: glm)
    monkeypatch.setattr(fitnb_mod, "_sample_psi_idx", lambda idx, maxn, rng: psi_mask)


def test_default_pipeline_svg_with_r_inputs(dlpfc_small, ref, monkeypatch):
    """spanorm() + spanorm_svg() with R's size factors and sampling injected, defaults otherwise."""
    r = ref["default"]
    inject_default_sampling(monkeypatch, r)
    spanorm(dlpfc_small, size_factors=r["sf"], verbose=False)
    inject_psi_sampling(monkeypatch, r["null"]["sampling"])
    spanorm_svg(dlpfc_small, backend="cpu", verbose=False)
    assert_null_matches(dlpfc_small, r["null"])
    assert_svg_matches(dlpfc_small, r)


def _spearman(a, b):
    return stats.spearmanr(a, b).statistic


def test_default_pipeline_svg_close(dlpfc_small, ref):
    """A fully independent run (own size factors AND own subsampling) against R's default run.

    R run against itself under another seed already differs by the 25% random sampling, so the
    fully independent Python run need only be as close to R's default as R's other default
    runs are (the R-vs-R seed-to-seed floor of svg_F), within a small margin.
    """
    runs = [np.asarray(ref[k]["svg_F"]) for k in ("default", "seed11", "seed12")]
    floor = np.mean([_spearman(runs[i], runs[j]) for i, j in ((0, 1), (0, 2), (1, 2))])
    spanorm(dlpfc_small, verbose=False)
    spanorm_svg(dlpfc_small, backend="cpu", verbose=False)
    py = dlpfc_small.var["svg_F"].to_numpy()
    assert np.isfinite(py).all()
    rho = np.mean([_spearman(py, r) for r in runs])
    # Both means are over three correlations, and the R-vs-R floor is itself low (~0.65: the
    # statistic of the many weakly spatial genes is dominated by the sampling noise), so allow
    # 0.07 below it (Task 10's noise-floor test allows 0.05). Python seeds 0-3 give mean
    # Spearman 0.607, 0.608, 0.634, 0.666 against a floor of 0.664: a single Python draw has
    # the same spread as R has against itself, so the margin covers one draw.
    assert rho > floor - 0.07, f"mean Spearman {rho:.4f} vs R-vs-R floor {floor:.4f}"


# ------------------------------------------------------------------ ported testthat cases


def create_test_spe(n_genes=100, n_spots=50, seed=0):
    rng = np.random.default_rng(seed)
    a = anndata.AnnData(
        X=rng.poisson(5, size=(n_spots, n_genes)).astype(float),
        obs=pd.DataFrame(index=[f"spot{i + 1}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene{i + 1}" for i in range(n_genes)]),
    )
    a.obsm["spatial"] = rng.uniform(size=(n_spots, 2))
    return a


def create_test_spanorm(n_genes=100, n_spots=50, n_covariates=4, seed=1) -> SpaNormFit:
    rng = np.random.default_rng(seed)
    return SpaNormFit(
        ngenes=n_genes,
        ncells=n_spots,
        gene_model="nb",
        df_tps=np.array([6]),
        sample_p=0.25,
        lambda_a=np.array([0.0001, 0.0002]),
        batch=None,
        W=rng.normal(size=(n_spots, n_covariates)),
        alpha=rng.normal(size=(n_genes, n_covariates)),
        gmean=np.ones(n_genes),
        psi=np.ones(n_genes),
        wtype=np.array(["ls", "ls", *["biology"] * (n_covariates - 2)]),
        loglik=-np.ones(n_spots),
        sampling=np.array(["dispersion"] * n_spots),
    )


def with_model(n_genes=100, n_spots=50, **kw):
    a = create_test_spe(n_genes, n_spots)
    store_fit(a, create_test_spanorm(n_genes, n_spots, **kw), "spanorm")
    return a


def test_fails_without_spanorm_model():
    with pytest.raises(ValueError, match="spanorm model"):
        spanorm_svg(create_test_spe(), verbose=False)


def test_warns_when_overwriting_existing_results():
    a = with_model()
    for c in ("svg_F", "svg_p", "svg_fdr"):
        a.var[c] = np.nan
    with pytest.warns(UserWarning, match="SVG results exist"):
        spanorm_svg(a, verbose=False)
    # old results are replaced
    assert not np.isnan(a.var[["svg_F", "svg_p", "svg_fdr"]].to_numpy()).all()


def test_fit_technical_handles_inputs():
    rng = np.random.default_rng(3)
    Y = rng.poisson(5, size=(100, 50)).astype(float)
    fit = create_test_spanorm()
    res = fit_spanorm_technical(Y, fit, lambda m: None)
    assert isinstance(res, SpaNormFit)
    assert res.W.shape[1] == 2  # only technical covariates
    assert (res.wtype != "biology").all() and (res.wtype == "ls").all()


def test_svg_test_validates_input_dimensions():
    rng = np.random.default_rng(4)
    Y = rng.poisson(5, size=(100, 50)).astype(float)
    full = create_test_spanorm(100, 50, 4)
    with pytest.raises(ValueError, match="number of genes differ between SpaNorm fits and/or data"):
        svg_test(Y, full, create_test_spanorm(99, 50, 3))
    with pytest.raises(ValueError, match="number of cells differ between SpaNorm fits and/or data"):
        svg_test(Y, full, create_test_spanorm(100, 49, 3))
    with pytest.raises(ValueError, match="technical model is not nested in the full model"):
        svg_test(Y, full, create_test_spanorm(100, 50, 4))
    with pytest.raises(TypeError, match="Y must be a matrix"):
        svg_test(list(Y), full, create_test_spanorm(100, 50, 2))
    with pytest.raises(TypeError, match="fit_spanorm must be a SpaNormFit"):
        svg_test(Y, "x", create_test_spanorm(100, 50, 2))
    with pytest.raises(TypeError, match="fit_technical must be a SpaNormFit"):
        svg_test(Y, full, "x")


def test_svg_test_nesting_uses_stored_column_names():
    rng = np.random.default_rng(5)
    Y = rng.poisson(5, size=(30, 40)).astype(float)
    full, nul = create_test_spanorm(30, 40, 4), create_test_spanorm(30, 40, 2)
    full.colnames = np.array(["logLS", "a", "b", "c"])
    nul.colnames = np.array(["logLS", "a"])
    assert len(svg_test(Y, full, nul)) == 30
    nul.colnames = np.array(["logLS", "zzz"])
    with pytest.raises(ValueError, match="not nested"):
        svg_test(Y, full, nul)


def test_svg_test_output_format():
    rng = np.random.default_rng(6)
    Y = rng.poisson(5, size=(100, 50)).astype(float)
    res = svg_test(Y, create_test_spanorm(100, 50, 4), create_test_spanorm(100, 50, 2))
    assert isinstance(res, pd.DataFrame) and len(res) == 100
    assert list(res.columns) == ["svg_F", "svg_p", "svg_fdr"]
    assert ((res["svg_p"] >= 0) & (res["svg_p"] <= 1)).all()
    assert ((res["svg_fdr"] >= 0) & (res["svg_fdr"] <= 1)).all()
    np.testing.assert_allclose(res["svg_fdr"], _p_adjust_fdr(res["svg_p"]))
    assert (res["svg_F"] >= 0).all() and (res.attrs["F_raw"] <= res["svg_F"] + 1e-12).all()


def test_p_adjust_fdr_matches_bh():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan, 0.5, 0.03])
    ok = ~np.isnan(p)
    expected = np.full(p.shape, np.nan)
    expected[ok] = stats.false_discovery_control(p[ok], method="bh")
    np.testing.assert_allclose(_p_adjust_fdr(p), expected)
    np.testing.assert_allclose(_p_adjust_fdr(np.array([0.3])), [0.3])


def test_edge_cases_single_gene_and_zero_gene():
    a = create_test_spe(n_genes=1, n_spots=50)
    store_fit(a, create_test_spanorm(1, 50), "spanorm")
    spanorm_svg(a, verbose=False)
    assert len(a.var) == 1 and np.isfinite(a.var["svg_F"]).all()

    a = create_test_spe(n_genes=10, n_spots=10)
    a.X[:, 0] = 0  # first gene has zero expression
    store_fit(a, create_test_spanorm(10, 10), "spanorm")
    spanorm_svg(a, verbose=False)
    assert len(a.var) == 10
    for c in ("svg_F", "svg_p", "svg_fdr", "svg_F_raw"):
        assert np.isfinite(a.var[c]).all(), c
    assert ((a.var["svg_p"] >= 0) & (a.var["svg_p"] <= 1)).all()
    assert (a.var["svg_F"] >= 0).all()


def test_results_stable_across_runs():
    a = with_model(50, 30)
    b = a.copy()
    spanorm_svg(a, verbose=False)
    spanorm_svg(b, verbose=False)
    pd.testing.assert_frame_equal(
        a.var[["svg_F", "svg_p", "svg_fdr"]], b.var[["svg_F", "svg_p", "svg_fdr"]]
    )


def test_validates_input_types():
    with pytest.raises(TypeError):
        spanorm_svg([], verbose=False)
    a = with_model()
    with pytest.raises(ValueError, match="layer 'nope' not found"):
        spanorm_svg(a, layer="nope", verbose=False)


def test_checks_model_compatibility():
    a = create_test_spe()
    with pytest.raises(ValueError):  # a fit for one more gene than the data has
        store_fit(a, create_test_spanorm(101, 50), "spanorm")
    with pytest.raises(ValueError, match="spanorm model"):
        spanorm_svg(a, verbose=False)


def test_spot_subset_of_a_fitted_object_is_an_error():
    a = with_model()
    sub = a[:30].copy()
    with pytest.raises(ValueError, match="dimensions do not match"):
        spanorm_svg(sub, verbose=False)


def test_top_svgs_results():
    a = with_model()
    spanorm_svg(a, verbose=False)
    res = top_svgs(a, n=10, fdr=0.05)
    assert isinstance(res, pd.DataFrame) and res.shape[1] == 3
    assert list(res.columns) == ["svg_F", "svg_p", "svg_fdr"]
    assert len(res) <= 10 and (np.diff(res["svg_fdr"]) >= 0).all()
    assert (res["svg_fdr"] <= 0.05).all()


def test_top_svgs_edge_cases():
    a = with_model()
    spanorm_svg(a, verbose=False)
    assert len(top_svgs(a, n=1000, fdr=0.001)) < 1000
    assert len(top_svgs(a, n=1)) == 1
    assert len(top_svgs(a, fdr=1e-10)) >= 0
    full = top_svgs(a, n=1000)
    assert len(full) == 100
    assert (np.diff(full["svg_fdr"]) >= 0).all()
    assert list(full.index[:3]) == list(a.var["svg_fdr"].sort_values(kind="stable").index[:3])


def test_top_svgs_validates_inputs():
    a = create_test_spe()
    with pytest.raises(ValueError, match="SVG results not found"):
        top_svgs(a)
    store_fit(a, create_test_spanorm(), "spanorm")
    spanorm_svg(a, verbose=False)
    for kw in ({"n": 0}, {"n": -1}, {"fdr": -0.1}, {"fdr": 1.1}):
        with pytest.raises(ValueError):
            top_svgs(a, **kw)


def test_penalises_library_size_terms_as_the_full_fit_does(monkeypatch):
    """The null receives lambda_a * ncells for its library-size columns, as the full fit does."""
    ng, nc = 30, 60
    rng = np.random.default_rng(1712)
    Y = rng.poisson(10, size=(ng, nc)).astype(float)
    coords = rng.uniform(size=(nc, 2))
    from spanormpy.spanorm import fit_spanorm

    seen = []
    orig = fitnb_mod.fit_nb_given_psi

    def spy(Ysub, Wsub, psi, **kw):
        seen.append(np.asarray(kw["lambda_a"]).copy())
        return orig(Ysub, Wsub, psi, **kw)

    monkeypatch.setattr(fitnb_mod, "fit_nb_given_psi", spy)
    quiet = lambda m: None
    full, _ = fit_spanorm(
        Y,
        coords,
        1.0,
        "nb",
        df_tps=2,
        lambda_a=[1e-4, 2e-4],
        size_factors=Y.sum(0) / Y.sum(0).mean(),
        msgfun=quiet,
        maxit_psi=1,
        backend="cpu",
    )
    pen_full = seen[0]
    seen.clear()
    tech = fit_spanorm_technical(Y, full, quiet, maxit_psi=1, backend="cpu")
    pen_tech = seen[0]
    names_full = list(full.colnames)[1:]
    wt_full = np.asarray(full.wtype)[1:]
    ls_cols = [n for n, w in zip(names_full, wt_full, strict=True) if w == "ls"]
    assert len(ls_cols) > 0
    names_tech = list(tech.colnames)[1:]
    for c in ls_cols:
        assert pen_tech[names_tech.index(c)] == pen_full[names_full.index(c)] == 2e-4 * nc
    # the null must be fitted to the same cells as the full model
    with pytest.raises(ValueError, match="number of cells"):
        fit_spanorm_technical(Y[:, 1:], full, quiet)


# ------------------------------------------------------------------ polished pairing (testthat)


def small_unpolished_pair(ref):
    pol = ref["pol"]
    a = polished_adata(ref, null_input=pol["unpolished"]["null"])
    return a


def test_svg_test_refuses_a_mixed_pair(ref):
    a = small_unpolished_pair(ref)
    full, nul = load_fit(a, "spanorm"), load_fit(a, "spanorm_null")
    quiet_polish(a, null=False)
    full_p = load_fit(a, "spanorm")
    Y = np.asarray(a.X).T
    with pytest.raises(ValueError, match="polished alike"):
        svg_test(Y, full_p, nul)
    assert not _polished_alike(full_p, nul) and _polished_alike(full, nul)


def test_refuses_polished_null_next_to_unpolished_full(ref):
    a = small_unpolished_pair(ref)
    quiet_polish(a)  # polishes both
    nul_p = load_fit(a, "spanorm_null")
    b = polished_adata(ref)
    store_fit(b, nul_p, "spanorm_null")
    with pytest.raises(ValueError, match="polish_spanorm"):
        spanorm_svg(b, verbose=False)


def test_svg_test_refuses_settings_mismatched_pair(ref):
    a = small_unpolished_pair(ref)
    quiet_polish(a)
    full_fixed = load_fit(a, "spanorm")
    b = small_unpolished_pair(ref)
    quiet_polish(b, psi_method="profile")
    null_profile = load_fit(b, "spanorm_null")
    Y = np.asarray(a.X).T
    with pytest.raises(ValueError, match="polished alike: 'psi_method' differs"):
        svg_test(Y, full_fixed, null_profile)


def test_psi_range_compared_under_profile_only(ref):
    a = small_unpolished_pair(ref)
    quiet_polish(a, psi_method="profile")
    full_p = load_fit(a, "spanorm")
    b = small_unpolished_pair(ref)
    quiet_polish(b, psi_method="profile", psi_range=(0.02, 0.2))
    null_p = load_fit(b, "spanorm_null")
    np.testing.assert_array_equal(full_p.polish["settings"]["psi_range"], [1e-3, 1e3])
    assert not _polished_alike(full_p, null_p)
    Y = np.asarray(a.X).T
    with pytest.raises(ValueError, match="polished alike: 'psi_range' differs"):
        svg_test(Y, full_p, null_p)
    # a profile fit polished before psi_range was recorded reads as the default
    old = load_fit(b, "spanorm_null")
    del old.polish["settings"]["psi_range"]
    assert _polished_alike(full_p, old)
    old_full = load_fit(a, "spanorm")
    del old_full.polish["settings"]["psi_range"]
    assert not _polished_alike(old_full, null_p)
    # under "fixed" the range is not part of the objective, so not recorded
    c = small_unpolished_pair(ref)
    quiet_polish(c, psi_method="fixed", psi_range=(0.02, 0.2))
    assert "psi_range" not in load_fit(c, "spanorm").polish["settings"]


# ------------------------------------------------------------------ storage and API


def test_polish_spanorm_removes_exactly_the_columns_spanorm_svg_writes(ref):
    a = polished_adata(ref)
    before = set(a.var.columns)
    spanorm_svg(a, backend="cpu", verbose=False)
    written = (set(a.var.columns) - before) & {c for c in a.var.columns if c.startswith("svg_")}
    assert written == set(SVG_COLUMNS)
    svg_cols_present = {c for c in a.var.columns if c.startswith("svg_")}
    assert svg_cols_present == set(SVG_COLUMNS)
    with pytest.warns(UserWarning, match="SVG results .* removed"):
        polish_spanorm(a, verbose=False)
    assert not ({c for c in a.var.columns if c.startswith("svg_")})
    assert set(a.var.columns) >= before


def test_view_and_gene_subset_reuse_the_stored_fits(ref, recwarn):
    a = polished_adata(ref)
    spanorm_svg(a, backend="cpu", verbose=False)
    full_f = a.var["svg_F"].to_numpy().copy()
    sub = a[:, :25]
    assert sub.is_view
    n_before = len(recwarn)
    spanorm_svg(sub, backend="cpu", verbose=False)
    assert not [w for w in recwarn.list[n_before:] if "ImplicitModification" in str(w.category)]
    assert not sub.is_view
    assert "svg_F_raw" in sub.var
    # the stored (gene-subset) fits are reused and keep the whole fit's dispersion threshold
    np.testing.assert_allclose(sub.var["svg_F"], full_f[:25], rtol=1e-10, atol=1e-12)
    assert get_spanorm_fit(sub, null=True) is not None


def test_sparse_counts_equal_dense(ref):
    dense = polished_adata(ref)
    sparse = polished_adata(ref)
    sparse.X = sp.csr_matrix(sparse.X)
    spanorm_svg(dense, backend="cpu", verbose=False)
    spanorm_svg(sparse, backend="cpu", verbose=False)
    for c in ("svg_F", "svg_p", "svg_fdr", "svg_F_raw"):
        np.testing.assert_allclose(sparse.var[c], dense.var[c], rtol=1e-8, atol=1e-12)


def test_copy_layer_and_messages(ref, capsys):
    a = polished_adata(ref)
    a.layers["raw"] = a.X.copy()
    a.X = np.zeros_like(a.X)  # X is not counts; the layer is
    out = spanorm_svg(a, layer="raw", copy=True, verbose=True)
    assert isinstance(out, anndata.AnnData) and "svg_F" in out.var and "svg_F" not in a.var
    err = capsys.readouterr().err
    for msg in (
        "(1/3) Retrieving SpaNorm model",
        "(2/3) Fitting Null SpaNorm model",
        "(3/3) Finding SVGs",
        "SVGs found (FDR < 0.05)",
        "cells/spots sampled to fit model",
    ):
        assert msg in err
    assert out.n_obs == a.n_obs and "spanorm_null_alpha" in out.varm
    # second run reuses the stored null
    with pytest.warns(UserWarning, match="SVG results exist"):
        spanorm_svg(out, backend="cpu", verbose=True)
    assert "(2/3) Retrieving Null SpaNorm model" in capsys.readouterr().err


def test_svg_keeps_var_columns_and_overwrites_in_place(ref):
    a = polished_adata(ref)
    a.var["other"] = 1.0
    spanorm_svg(a, backend="cpu", verbose=False)
    assert "other" in a.var
    first = a.var["svg_F"].to_numpy().copy()
    with pytest.warns(UserWarning, match="SVG results exist"):
        spanorm_svg(a, backend="cpu", verbose=False)
    np.testing.assert_allclose(a.var["svg_F"], first, rtol=0, atol=0)


def test_top_svgs_nan_fdr_rows_match_r():
    """R keeps one all-NA row per NaN FDR, last in order and counted toward n."""
    r = load_ref("svg_top")
    fdr = np.asarray(r["fdr"])
    a = create_test_spe(n_genes=len(fdr), n_spots=5)
    a.var["svg_F"] = np.arange(1, len(fdr) + 1) * 1.5
    a.var["svg_p"] = fdr / 2
    a.var["svg_fdr"] = fdr
    for k in "abcd":
        c = r[k]
        res = top_svgs(a, n=int(c["n"]), fdr=float(c["fdr"]))
        assert len(res) == int(c["nrow"]), k
        np.testing.assert_array_equal(res["svg_F"].to_numpy(), np.asarray(c["F"]))  # NaN == NaN
        isna = np.isnan(res["svg_fdr"].to_numpy())
        rn = np.asarray(c["rownames"])
        np.testing.assert_array_equal(isna, np.char.startswith(rn.astype(str), "NA"))
        np.testing.assert_array_equal([g.replace("gene", "g") for g in res.index[~isna]], rn[~isna])
        # the NaN rows keep their real gene names, in gene order
        assert list(res.index[isna]) == [g for g in a.var_names[np.isnan(fdr)]][: isna.sum()]


def test_top_svgs_checks_counts_like_checkspe():
    a = with_model()
    spanorm_svg(a, verbose=False)
    with pytest.raises(ValueError, match="layer 'nope' not found"):
        top_svgs(a, layer="nope")
    a.X[0, 0] = 0.5
    with pytest.raises(ValueError):
        top_svgs(a)
    a.layers["raw"] = np.round(np.abs(a.X))
    assert len(top_svgs(a, layer="raw")) > 0
