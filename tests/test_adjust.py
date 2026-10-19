"""Tests for the adjustments and blocked normalisation against R references."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from tests._ref import load_ref

from spanormpy import _adjust
from spanormpy._adjust import (
    get_adjustment_fun,
    normalise_block_size,
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


def test_normalise_block_size(ref, tmp_path):
    import h5py

    Y = ref["Y"]
    assert normalise_block_size(Y) == np.inf
    assert normalise_block_size(sp.csr_matrix(Y)) == np.inf
    with h5py.File(tmp_path / "y.h5", "w") as f:
        f["Y"] = Y
        assert normalise_block_size(f["Y"]) == 1e8 / 8  # DelayedArray auto block size, in doubles
    pytest.importorskip("dask.array")
    import dask.array as da

    assert normalise_block_size(da.from_array(Y)) == 1e8 / 8


class _SpyArray:
    """Row-sliceable array that records every read; never hands out the whole matrix."""

    def __init__(self, a):
        self.a, self.shape, self.reads = a, a.shape, []

    def __getitem__(self, key):
        rows = key[0]
        self.reads.append(rows.stop - rows.start)
        return self.a[key]


def test_default_block_size_reads_backed_input_in_blocks(ref, monkeypatch):
    # the default budget comes from normalise_block_size; shrink it so the small fixture splits
    fit = _fit(ref["nobatch"])
    spy = _SpyArray(ref["Y"])
    monkeypatch.setattr(_adjust, "normalise_block_size", lambda Y: 10_000.0)
    got = normalise_blocked(normalise_pearson, spy, 1.0, fit)
    assert len(spy.reads) > 1
    assert max(spy.reads) < ref["Y"].shape[0]
    assert sum(spy.reads) == ref["Y"].shape[0]
    np.testing.assert_allclose(got, normalise_pearson(ref["Y"], 1.0, fit), rtol=1e-12, atol=1e-12)


def test_in_memory_default_is_single_pass(ref, monkeypatch):
    fit = _fit(ref["nobatch"])
    calls = []
    normalise_blocked(lambda *a: calls.append(1) or normalise_meanbio(*a), ref["Y"], 1.0, fit)
    assert len(calls) == 1


def test_tensor_dtype_follows_backend_dtype(monkeypatch):
    # MPS has no float64: tensors on an accelerator must be built with the backend dtype
    monkeypatch.setattr(_adjust, "get_backend_dtype", lambda: torch.float32)
    t = _adjust._t(np.ones(3), torch.device("meta"))
    assert t.dtype == torch.float32
    assert _adjust._t(np.ones(3), torch.device("cpu")).dtype == torch.float64


@pytest.mark.parametrize("case", ["degen_zero", "degen_mixed"])
@pytest.mark.parametrize("method", ["logpac", "pearson", "medbio", "meanbio"])
def test_degenerate_psi_matches_r(ref, case, method):
    import warnings

    fit = _fit(ref["nobatch"])
    fit.psi = np.asarray(ref[case]["psi"])
    want = ref[case][method]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # psi = 0 must not warn (R is silent)
        got = FUNS[method](ref["Y"], 2.5, fit)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    # fixed in SpaNorm 1.7.16: the cap uses finite log-dispersions only, so a NaN psi
    # (gene 2 in degen_mixed) is NaN for its own gene alone; psi-free meanbio is finite.
    bad = np.zeros(got.shape[0], dtype=bool)
    if case == "degen_mixed" and method != "meanbio":
        bad[1] = True
    assert np.isnan(got).all(axis=1).tolist() == bad.tolist()
    assert np.isfinite(got[~bad]).all()


def test_psi_max_finite_only():
    from spanormpy._adjust import global_psi_max
    from spanormpy._nbdist import compute_psi_max, winsorise_psi

    psi = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 100.0])
    lp = np.log(psi)
    ref = np.exp(np.median(lp) + 4 * 1.4826 * np.median(np.abs(lp - np.median(lp))))
    assert compute_psi_max(psi) == pytest.approx(ref)
    assert global_psi_max(np.r_[psi, np.nan, 0.0, np.inf]) == pytest.approx(ref)
    assert compute_psi_max(np.array([np.nan, 0.0])) == np.inf
    assert compute_psi_max(np.array([])) == np.inf
    w = winsorise_psi(np.r_[psi, np.nan])
    assert np.isnan(w[-1]) and np.allclose(w[:-1], np.minimum(psi, ref))
