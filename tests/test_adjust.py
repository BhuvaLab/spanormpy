"""Tests for the adjustments and blocked normalisation against R references."""

import numpy as np
import pytest
import scipy.sparse as sp
from tests._ref import load_ref

from spanormpy._adjust import (
    get_adjustment_fun,
    normalise_blocked,
    normalise_logpac,
    normalise_meanbatch,
    normalise_meanbio,
    normalise_meanls,
    normalise_medbio,
    normalise_pearson,
)
from spanormpy._fit import SpaNormFit

FUNS = {
    "logpac": normalise_logpac,
    "pearson": normalise_pearson,
    "medbio": normalise_medbio,
    "meanbio": normalise_meanbio,
    "meanls": normalise_meanls,
    "meanbatch": normalise_meanbatch,
}
SF = {"sf1": 1.0, "sf2p5": 2.5}
# Deterministic given the fit; scipy NB cdf/ppf agree with R's closed forms to rounding error.
RTOL, ATOL = 1e-10, 1e-12


@pytest.fixture(scope="module")
def ref():
    return load_ref("adjust")


def _fit(r) -> SpaNormFit:
    return SpaNormFit(
        ngenes=int(r["ngenes"]),
        ncells=int(r["ncells"]),
        gene_model=r["gene_model"],
        df_tps=np.atleast_1d(r["df_tps"]).astype(int),
        sample_p=r["sample_p"],
        lambda_a=np.atleast_1d(r["lambda_a"]),
        batch=r.get("batch"),
        W=r["W"],
        alpha=r["alpha"],
        gmean=np.atleast_1d(r["gmean"]),
        psi=np.atleast_1d(r["psi"]),
        wtype=np.atleast_1d(r["wtype"]),
        loglik=np.atleast_1d(r["loglik"]),
        sampling=np.atleast_1d(r["sampling"]),
    )


def _cases():
    out = []
    for m in FUNS:
        for s in SF:
            out.append((("batch" if m == "meanbatch" else "nobatch"), m, s))
    for m in ("logpac", "pearson", "medbio", "meanbio", "meanls"):
        out.append(("batch", m, "sf2p5"))
    return out


@pytest.mark.parametrize("which,method,sf", _cases())
def test_adjustment_matches_r(ref, device, which, method, sf):
    r = ref[which]
    fit = _fit(r)
    backend = "cpu" if device == "cpu" else "gpu"
    got = FUNS[method](ref["Y"], SF[sf], fit, backend=backend)
    want = r["adj"][f"{method}_{sf}"]
    assert got.shape == want.shape
    assert np.isfinite(want).all()  # guard against a degenerate (vacuous) reference
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("block_size", [2000, 8000, 20000])
@pytest.mark.parametrize("method", ["logpac", "pearson", "medbio", "meanbio"])
@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
def test_blocked_equals_whole(ref, method, block_size, kind):
    fit = _fit(ref["nobatch"])
    Y = ref["Y"]
    Yin = {"dense": Y, "csr": sp.csr_matrix(Y), "csc": sp.csc_matrix(Y)}[kind]
    whole = FUNS[method](Y, 2.5, fit)
    blocked = normalise_blocked(FUNS[method], Yin, 2.5, fit, block_size=block_size)
    np.testing.assert_allclose(blocked, whole, rtol=1e-12, atol=1e-12)


def test_blocked_matches_r_blocked(ref):
    fit = _fit(ref["nobatch"])
    b = ref["blocked"]
    got = normalise_blocked(normalise_logpac, ref["Y"], 1.0, fit, block_size=8e3)
    np.testing.assert_allclose(got, b["logpac"], rtol=RTOL, atol=ATOL)
    got = normalise_blocked(normalise_pearson, ref["Y"], 2.5, fit, block_size=8e3)
    np.testing.assert_allclose(got, b["pearson"], rtol=RTOL, atol=ATOL)


def test_blocked_threaded_equals_whole(ref):
    # tile cells so the problem passes the 1e6-element threshold of the parallel branch
    fit = _fit(ref["nobatch"])
    fit.W = np.tile(fit.W, (20, 1))
    big = np.tile(ref["Y"], (1, 20))
    whole = normalise_pearson(big, 1.0, fit)
    got = normalise_blocked(normalise_pearson, big, 1.0, fit, n_jobs=2)
    np.testing.assert_allclose(got, whole, rtol=1e-12, atol=1e-12)


def test_blocked_unblocked_path_is_whole(ref):
    fit = _fit(ref["nobatch"])
    got = normalise_blocked(normalise_meanbio, ref["Y"], 1.0, fit)
    np.testing.assert_array_equal(got, normalise_meanbio(ref["Y"], 1.0, fit))


def test_auto_resolution():
    assert get_adjustment_fun("nb", "auto") is normalise_logpac
    assert get_adjustment_fun("nb", "logpac") is normalise_logpac
    assert get_adjustment_fun("nb", "pearson") is normalise_pearson
    assert get_adjustment_fun("nb", "medbio") is normalise_medbio
    assert get_adjustment_fun("nb", "meanbio") is normalise_meanbio
    with pytest.raises(ValueError, match="invalid argument for 'adj.method'"):
        get_adjustment_fun("nb", "bogus")
    with pytest.raises(ValueError, match="'zinb' gene model not supported"):
        get_adjustment_fun("zinb", "auto")


def test_blocked_backed_h5py(ref, tmp_path):
    import h5py

    fit = _fit(ref["nobatch"])
    with h5py.File(tmp_path / "y.h5", "w") as f:
        f["Y"] = ref["Y"]
        got = normalise_blocked(normalise_pearson, f["Y"], 1.0, fit, block_size=5000)
    np.testing.assert_allclose(got, normalise_pearson(ref["Y"], 1.0, fit), rtol=1e-12, atol=1e-12)
