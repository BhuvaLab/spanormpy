import importlib
import warnings

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

import spanormpy._fitnb as fitnb_mod
from spanormpy._fit import load_fit
from spanormpy import spanorm
from spanormpy.spanorm import match_df_tps, match_lambda
from tests._ref import load_ref

# `spanormpy.spanorm` the function shadows the module attribute; fetch the module itself
sn = importlib.import_module("spanormpy.spanorm")

# Iterative fits given identical inputs and R's sampling: spec tolerance (rtol 1e-6).
RTOL = 1e-6
COMMON = dict(df_tps=2, tol=1e-3, sample_p=0.25)


@pytest.fixture(scope="module")
def ref():
    return load_ref("spanorm")


def inject_r_sampling(monkeypatch, sampling):
    """Replace both subsampling hooks by R's draw, rebuilt from R's ``fit$sampling``."""
    sampling = np.asarray(sampling).astype(str)
    glm = np.flatnonzero(sampling != "all")
    psi_mask = sampling[glm] == "dispersion"
    monkeypatch.setattr(sn, "_sample_random", lambda coords, nsub, rng: glm)
    monkeypatch.setattr(fitnb_mod, "_sample_psi_idx", lambda idx, maxn, rng: psi_mask)


def run(adata, ref_part, monkeypatch, **kw):
    inject_r_sampling(monkeypatch, ref_part["sampling"])
    spanorm(adata, verbose=False, **{**COMMON, **kw})
    return adata


def assert_fit_matches(adata, r, sub=None):
    np.testing.assert_allclose(adata.varm["spanorm_alpha"], r["alpha"], rtol=RTOL)
    np.testing.assert_allclose(adata.var["spanorm_psi"], r["psi"], rtol=RTOL)
    np.testing.assert_allclose(adata.var["spanorm_gmean"], r["gmean"], rtol=RTOL)
    np.testing.assert_allclose(adata.obsm["spanorm_W"], r["W"], rtol=RTOL)
    lc = adata.layers["logcounts"]
    if sub is not None:
        lc = lc[:, sub]
    np.testing.assert_allclose(lc, r["logcounts"].T, rtol=RTOL)


def test_spanorm_matches_r(dlpfc_small, ref, monkeypatch, device):
    backend = "cpu" if device == "cpu" else "gpu"
    r = ref["base"]
    a = run(dlpfc_small, r, monkeypatch, size_factors=ref["sf"], backend=backend)
    assert_fit_matches(a, r)
    fit = load_fit(a)
    np.testing.assert_allclose(fit.loglik, np.atleast_1d(r["loglik"]), rtol=RTOL)
    assert list(fit.wtype) == list(r["wtype"])
    assert list(fit.colnames) == list(r["W_colnames"])
    np.testing.assert_array_equal(fit.sampling, r["sampling"])
    np.testing.assert_array_equal(fit.df_tps, r["df_tps"])
    np.testing.assert_array_equal(np.atleast_1d(fit.lambda_a), np.atleast_1d(r["lambda_a"]))
    assert a.uns["spanorm"]["adj_method"] == "auto"
    assert "spanorm_size_factors" not in a.obs  # user-supplied, not computed


def test_psi_not_nan_report(ref):
    # Documents R's reference: a NaN dispersion would turn every logPAC into NaN (Task 8 quirk).
    assert np.isfinite(ref["base"]["psi"]).all()
    assert np.isfinite(ref["base"]["logcounts"]).all()


def test_df_tps_pair(dlpfc_small, ref, monkeypatch):
    r = ref["df42"]
    a = run(dlpfc_small, r, monkeypatch, size_factors=ref["sf"], df_tps=(4, 2))
    fit = load_fit(a)
    np.testing.assert_array_equal(fit.df_tps, r["df_tps"])
    sub = np.asarray(ref["gsub_idx"]).astype(int) - 1
    assert_fit_matches(a, r, sub)


def test_batch_matches_r(dlpfc_small, ref, monkeypatch):
    r = ref["batch"]
    x = dlpfc_small.obsm["spatial"].iloc[:, 0].to_numpy()
    dlpfc_small.obs["slide"] = np.where(x > np.median(x), "b2", "b1")
    assert (dlpfc_small.obs["slide"].to_numpy() == np.asarray(ref["batch_vec"])).all()
    a = run(dlpfc_small, r, monkeypatch, size_factors=ref["sf"], batch_key="slide")
    fit = load_fit(a)
    assert list(fit.wtype) == list(r["wtype"]) and fit.wtype[-1] == "batch"
    assert list(fit.colnames) == list(r["W_colnames"])
    sub = np.asarray(ref["gsub_idx"]).astype(int) - 1
    assert_fit_matches(a, r, sub)


@pytest.fixture
def fitted(dlpfc_small, ref, monkeypatch):
    return run(dlpfc_small, ref["base"], monkeypatch, size_factors=ref["sf"])


@pytest.mark.parametrize("method", ["pearson", "medbio", "meanbio", "logpac"])
def test_adj_method_reuse_no_refit(fitted, ref, method, capsys):
    before = {
        "alpha": fitted.varm["spanorm_alpha"].copy(),
        "psi": fitted.var["spanorm_psi"].to_numpy().copy(),
        "W": fitted.obsm["spanorm_W"].copy(),
        "samp": fitted.obs["spanorm_sampling"].copy(),
    }
    with pytest.warns(UserWarning, match="'logcounts' exists and will be overwritten"):
        spanorm(fitted, adj_method=method, **COMMON)
    err = capsys.readouterr().err
    assert "(1/2) Retrieve precomputed SpaNorm model" in err
    assert "Fitting SpaNorm model" not in err
    np.testing.assert_array_equal(fitted.varm["spanorm_alpha"], before["alpha"])
    np.testing.assert_array_equal(fitted.var["spanorm_psi"], before["psi"])
    np.testing.assert_array_equal(fitted.obsm["spanorm_W"], before["W"])
    assert (fitted.obs["spanorm_sampling"] == before["samp"]).all()
    assert fitted.uns["spanorm"]["adj_method"] == method
    sub = np.asarray(ref["gsub_idx"]).astype(int) - 1
    np.testing.assert_allclose(
        fitted.layers["logcounts"][:, sub], ref["reuse"][method]["logcounts"].T, rtol=RTOL
    )


def test_overwrite_refits(fitted, ref, monkeypatch, capsys):
    r = ref["overwrite"]
    old_alpha = fitted.varm["spanorm_alpha"].copy()
    inject_r_sampling(monkeypatch, r["sampling"])
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(fitted, overwrite=True, size_factors=ref["sf"], **COMMON)
    assert "(1/2) Fitting SpaNorm model" in capsys.readouterr().err
    assert not np.allclose(fitted.varm["spanorm_alpha"], old_alpha, rtol=1e-3)
    sub = np.asarray(ref["gsub_idx"]).astype(int) - 1
    assert_fit_matches(fitted, r, sub)


def _r_flags(msgs):
    msgs = np.atleast_1d(msgs)
    return [m for m in msgs if m.startswith("(")]


@pytest.mark.parametrize(
    ("name", "kw", "refit"),
    [
        ("df3", dict(df_tps=3), True),
        ("lambda", dict(lambda_a=1e-3), True),
        ("df21", dict(df_tps=(2, 1)), False),
        ("batch_change", dict(batch="BATCH"), True),
    ],
)
def test_mismatch_messages(fitted, ref, monkeypatch, capsys, name, kw, refit):
    r = ref[name]
    # R's own messages say whether it refitted
    assert ("(1/2) Fitting SpaNorm model" in _r_flags(r["msgs"])) == refit
    if kw.get("batch") == "BATCH":
        x = fitted.obsm["spatial"].iloc[:, 0].to_numpy()
        kw = dict(batch=np.where(x > np.median(x), "b2", "b1"))
    inject_r_sampling(monkeypatch, r["sampling"])
    with pytest.warns(UserWarning, match="overwritten"):
        spanorm(fitted, size_factors=ref["sf"], **{**COMMON, **kw})
    err = capsys.readouterr().err
    py_flags = [ln for ln in err.splitlines() if ln.startswith("(")]
    assert py_flags == _r_flags(r["msgs"])
    if refit:
        np.testing.assert_array_equal(load_fit(fitted).df_tps, r["df_tps"])
        np.testing.assert_allclose(fitted.var["spanorm_psi"], r["psi"], rtol=RTOL)
        np.testing.assert_allclose(fitted.varm["spanorm_alpha"], r["alpha"], rtol=RTOL)
        sub = np.asarray(ref["gsub_idx"]).astype(int) - 1
        np.testing.assert_allclose(
            fitted.layers["logcounts"][:, sub], r["logcounts"].T, rtol=RTOL
        )


def test_sampled_message_matches_r(fitted, ref, capsys):
    msgs = [m for m in np.atleast_1d(ref["base"]["msgs"]) if "sampled" in m]
    assert msgs == ["1004 cells/spots sampled to fit model"]


# ------------------------------------------------------------- default pipeline


def _gene_cors(py, rr):
    return np.array([np.corrcoef(py[:, j], rr[:, j])[0, 1] for j in range(py.shape[1])])


def test_default_pipeline_close(dlpfc_small, ref, monkeypatch):
    """Python's own size factors (no injection of them) against R's default ``SpaNorm()``.

    Cell/dispersion sampling is R's (injected): independent random subsets cannot reach 0.99
    per gene, because R run against itself with another seed already drops to ~0.87
    (see ``test_default_pipeline_sampling_noise_floor``).
    """
    rr = np.asarray(ref["default_logcounts"]).T  # cells x genes
    inject_r_sampling(monkeypatch, ref["default_sampling"])
    spanorm(dlpfc_small, verbose=False)
    py = dlpfc_small.layers["logcounts"]
    assert py.shape == rr.shape and np.isfinite(py).all()
    sf = dlpfc_small.obs["spanorm_size_factors"].to_numpy()
    assert np.corrcoef(sf, ref["default_sf"])[0, 1] > 0.99
    cors = _gene_cors(py, rr)
    assert not np.isnan(cors).any()
    assert cors.min() > 0.99, f"min per-gene correlation {cors.min():.4f}"


def test_default_pipeline_sampling_noise_floor(dlpfc_small, ref):
    """Fully independent run (own size factors AND own subsampling) is as close to R as R is
    to itself under another seed (same data, only the RNG draw differs)."""
    rr = np.asarray(ref["default_logcounts"]).T
    floor = _gene_cors(np.asarray(ref["default_logcounts_seed11"]).T, rr)
    spanorm(dlpfc_small, verbose=False)
    cors = _gene_cors(dlpfc_small.layers["logcounts"], rr)
    assert cors.min() > floor.min() - 0.05
    assert np.median(cors) > np.median(floor) - 0.02


# ------------------------------------------------------------- arguments and behaviour


def test_param_errors(dlpfc_small):
    for kw, msg in [
        (dict(lambda_a=[-1, 1]), "positive"),
        (dict(lambda_a=[-1, -1]), "positive"),
        (dict(sample_p=0), "0,1"),
        (dict(sample_p=1.1), "0,1"),
        (dict(sample_p=-1), "0,1"),
        (dict(lambda_a=[1, 1, 1]), "length 2"),
        (dict(df_tps=[1, 2, 3]), "length 2"),
        (dict(adj_method="abc"), "adj_method"),
        (dict(gene_model="abc"), "gene_model"),
        (dict(backend="abc"), "backend"),
    ]:
        with pytest.raises(ValueError, match=msg):
            spanorm(dlpfc_small, size_factors=np.linspace(0.5, 1.5, dlpfc_small.n_obs), verbose=False, **kw)
    with pytest.raises(ValueError, match="too small"):
        spanorm(dlpfc_small, sample_p=1e-6, size_factors=np.linspace(0.5, 1.5, dlpfc_small.n_obs), verbose=False)
    with pytest.raises(ValueError, match="only one of"):
        spanorm(dlpfc_small, batch_key="AnnotatedCluster", batch=np.zeros(4015))


def test_sample_p_too_large_warns():
    rng = np.random.default_rng(1)
    n = 3200
    a = anndata.AnnData(rng.poisson(5, size=(n, 8)).astype(float))
    a.obsm["spatial"] = rng.uniform(size=(n, 2))
    with pytest.warns(UserWarning, match="consider reducing 'sample_p' to 0.93"):
        spanorm(
            a, sample_p=1, df_tps=2, tol=1, maxit_psi=1, maxit_nb=1,
            size_factors=np.linspace(0.5, 1.5, n), verbose=False,
        )


def test_sample_random_properties():
    coords = np.c_[np.arange(6), np.arange(6)]
    idx = sn._sample_random(coords, 3, np.random.default_rng(0))
    assert len(np.unique(idx)) == 3 and idx.max() < 6


def test_match_df_tps_cases():  # test-mainSpaNorm.R
    assert match_df_tps([6, 4, 3, 2], 6)
    assert match_df_tps([6, 5, 3, 2], 6)
    assert match_df_tps([4, 3, 2, 1], 4)
    assert not match_df_tps([5, 4, 3, 2], 6)
    assert not match_df_tps([6, 5, 4, 4], 6)
    assert not match_df_tps([6, 4, 3, 2], 5)
    assert match_df_tps([6, 4, 3, 2], [6, 3])
    assert match_df_tps([6, 5, 4, 3], [6, 4])
    assert match_df_tps([4, 3, 2, 1], [4, 2])
    assert not match_df_tps([5, 4, 3, 2], [6, 3])
    assert not match_df_tps([6, 5, 2, 1], [6, 3])
    assert not match_df_tps([6, 4, 3, 2], [5, 3])
    with pytest.raises(ValueError):
        match_df_tps([6], [6])
    with pytest.raises(ValueError):
        match_df_tps([6, 4, 3, 2], [6, 3, 1])


def test_match_lambda_cases():
    assert match_lambda([1e-4, 1e-4], 1e-4)
    assert match_lambda([1e-4, 1e-3], [1e-4, 1e-3])
    assert not match_lambda([1e-4, 1e-3], 1e-4)
    with pytest.raises(ValueError):
        match_lambda([1e-4], 1e-4)


def small_adata(seed=0, n=120, g=10):
    rng = np.random.default_rng(seed)
    a = anndata.AnnData(rng.poisson(8, size=(n, g)).astype(float))
    a.obsm["spatial"] = rng.uniform(size=(n, 2))
    a.obs_names = [f"s{i}" for i in range(n)]
    return a


FAST = dict(df_tps=2, sample_p=0.8, tol=1e-1, maxit_psi=3, maxit_nb=5, verbose=False)


def test_copy_leaves_input_untouched():
    a = small_adata()
    out = spanorm(a, copy=True, **FAST)
    assert "logcounts" in out.layers and "logcounts" not in a.layers
    assert "spanorm_alpha" not in a.varm and "spanorm_size_factors" in out.obs


def test_view_is_converted_without_warning():
    a = small_adata()
    spanorm(a, **FAST)
    view = a[:, :7]
    assert view.is_view
    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        # logcounts exists -> R's overwrite warning is expected, nothing else
        with pytest.warns(UserWarning, match="overwritten"):
            spanorm(view, adj_method="meanbio", **FAST)
    assert not view.is_view
    assert view.n_vars == 7 and view.layers["logcounts"].shape == (120, 7)


def test_gene_subset_reuses_fit_with_global_psi_max(capsys):
    a = small_adata()
    spanorm(a, **FAST)
    full = a.layers["logcounts"].copy()
    sub = a[:, [0, 3, 4, 9]].copy()
    del sub.layers["logcounts"]
    spanorm(sub, **{**FAST, "verbose": True})
    assert "Retrieve precomputed" in capsys.readouterr().err
    np.testing.assert_allclose(sub.layers["logcounts"], full[:, [0, 3, 4, 9]], rtol=1e-10)


def test_sparse_matches_dense():
    a = small_adata()
    b = a.copy()
    b.X = sp.csr_matrix(b.X)
    spanorm(a, **FAST)
    spanorm(b, **FAST)
    assert sp.issparse(b.layers["logcounts"])
    np.testing.assert_allclose(b.layers["logcounts"].toarray(), a.layers["logcounts"], rtol=1e-10)


def test_layer_counts_and_size_factor_obs_key():
    a = small_adata()
    a.layers["raw"] = a.X.copy()
    a.X = np.zeros_like(a.X)  # X must not be read
    a.obs["mysf"] = np.linspace(0.5, 1.5, a.n_obs)
    spanorm(a, layer="raw", size_factors="mysf", **FAST)
    assert "spanorm_size_factors" not in a.obs
    assert np.isfinite(a.layers["logcounts"]).all()
    # `size_factors` obs column is never read implicitly
    b = small_adata()
    b.obs["size_factors"] = np.full(b.n_obs, np.nan)
    spanorm(b, **FAST)
    assert np.isfinite(b.layers["logcounts"]).all()


def test_spatialdata_input(dlpfc_small, ref, monkeypatch):
    sd = pytest.importorskip("spatialdata")
    r = ref["base"]
    inject_r_sampling(monkeypatch, r["sampling"])
    plain = dlpfc_small.copy()
    spanorm(plain, size_factors=ref["sf"], verbose=False, **COMMON)
    sdata = sd.SpatialData(tables={"table": dlpfc_small.copy()})
    spanorm(sdata, size_factors=ref["sf"], verbose=False, **COMMON)
    got = sdata.tables["table"].layers["logcounts"]
    np.testing.assert_allclose(got, plain.layers["logcounts"], rtol=1e-10)
    np.testing.assert_allclose(got, r["logcounts"].T, rtol=RTOL)
    with pytest.raises(ValueError, match="not found"):
        spanorm(sdata, table_key="nope", verbose=False)
