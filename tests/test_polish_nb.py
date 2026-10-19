"""polish_nb / nb_profile_psi: R parity and the ported testthat cases.

R sources: tests/testthat/test-polishNB.R and test-polishEngine.R (SpaNorm c12e5e5).
The reference fixtures are in tests/ref/make_polish_nb.R.
"""

import math
import re

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy import optimize
from tests._ref import load_ref

import spanormpy
from spanormpy import _polish_batch as pb
from spanormpy import polish as pol
from spanormpy._backend import as_tensor
from spanormpy._block_design import nb_block_design
from spanormpy._nbdist import dnbinom
from spanormpy._polish_engine import MU_FLOOR, brent_fmin, polish_gene
from spanormpy._solver import nb_newton_solver, nb_newton_solver_batch
from spanormpy.polish import nb_mu_floor, nb_profile_psi, polish_nb

CASES = ["plain", "offvec", "offmat", "absorb", "absgrp", "block"]
FLAGS = ["restarted", "capped", "singular", "psi_bound", "polished"]


@pytest.fixture(scope="module")
def ref():
    return load_ref("polish_nb")


def case_args(ref, name):
    """polish_nb keyword arguments (and the counts/start) of one reference case."""
    c = ref[name]
    kw = {"lambda_a": c.get("pen", 0.0)}
    if name in ("offvec", "offmat"):
        kw["offset"] = c["offset"]
    if name in ("absorb", "absgrp"):
        kw["start_cols"] = c["start"].astype(bool)
        if name == "absorb":
            kw["absorb"] = c["nested"].astype(bool)
        else:
            g = c["group"].astype(float)
            kw["absorb"] = np.where(g < 0, np.nan, g)
    W = c.get("W")
    if name == "block":
        W = nb_block_design(c["X"], c["Z"], c["block"])
        kw["start_cols"] = c["start"].astype(bool)
    return c["Y"], W, c["A0"], c["psi0"], kw


# ---- R parity --------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("psi_method", ["profile", "fixed"])
@pytest.mark.parametrize("engine", ["batch", "gene"])
def test_polish_nb_matches_r(ref, engine, psi_method, case, device):
    if device != "cpu" and (engine != "batch" or case == "block"):
        pytest.skip("a device needs engine='batch' and a dense design")
    Y, W, A0, psi0, kw = case_args(ref, case)
    got = polish_nb(
        Y, W, A0, psi0, psi_method=psi_method, engine=engine,
        backend="cpu" if device == "cpu" else "gpu", **kw,
    )  # fmt: skip
    want = ref[case][f"{engine}_{psi_method}"]
    # the shared-factor device path refreshes its information on its own schedule
    rtol = 1e-6 if device == "cpu" else 1e-5
    for f in FLAGS:
        np.testing.assert_array_equal(got["polish"][f].to_numpy(), want[f].astype(bool), err_msg=f)
    np.testing.assert_array_equal(got["polish"]["iterations"], want["iterations"])
    np.testing.assert_allclose(got["alpha"], want["alpha"], rtol=rtol, atol=1e-9)
    np.testing.assert_allclose(got["psi"], want["psi"], rtol=rtol)
    np.testing.assert_allclose(got["loglik"], want["loglik"], rtol=rtol, equal_nan=True)
    assert want["polished"].all() or case == "plain"
    np.testing.assert_allclose(got["polish"]["psi_fitnb"], np.resize(psi0, len(got["psi"])))


def test_plain_case_exercises_restart_and_bound(ref):
    # the parity above is worthless if the fixture takes only the easy path
    w = ref["plain"]["batch_profile"]
    assert w["restarted"].any()
    assert ref["plain"]["batch_fixed"]["polished"].all()


@pytest.mark.parametrize("psi_method", ["profile", "fixed"])
def test_batch_equals_gene_engine(ref, psi_method):
    Y, W, A0, psi0, kw = case_args(ref, "plain")
    a = polish_nb(Y, W, A0, psi0, psi_method=psi_method, engine="gene", **kw)
    b = polish_nb(Y, W, A0, psi0, psi_method=psi_method, engine="batch", **kw)
    # at a held dispersion the two engines take the same Newton path; the profile
    # dispersion is Brent per gene and a bisection in the batch (optimize()'s tolerance)
    tol = 1e-8 if psi_method == "fixed" else 1e-4
    np.testing.assert_allclose(a["alpha"], b["alpha"], rtol=tol, atol=1e-7)
    np.testing.assert_allclose(a["psi"], b["psi"], rtol=tol)
    for f in FLAGS:
        np.testing.assert_array_equal(a["polish"][f], b["polish"][f])


def test_warm_matches_r(ref):
    w = ref["warm"]
    for eng in ("batch", "gene"):
        got = polish_nb(
            w["Y"], w["W"], w["A0"], w["psi0"], lambda_a=w["pen_warm"], warm=True, engine=eng
        )
        np.testing.assert_allclose(got["alpha"], w[eng]["alpha"], rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(got["psi"], w["psi0"])
        np.testing.assert_allclose(got["loglik"], w[eng]["loglik"], rtol=1e-6)
        np.testing.assert_array_equal(got["polish"]["iterations"], w[eng]["iterations"])
        np.testing.assert_array_equal(got["polish"]["capped"], w[eng]["capped"].astype(bool))


def test_profile_psi_matches_r(ref):
    p = ref["profile"]
    got = nb_profile_psi(p["Y"], p["W"], p["A"], p["psi"])
    np.testing.assert_allclose(got, p["plain"], rtol=1e-8)
    assert got[2] == p["psi"][2]  # no spread at all: kept at the input
    got = nb_profile_psi(p["Y"], p["W"], p["A"], p["psi"], offset=p["offset"])
    np.testing.assert_allclose(got, p["off"], rtol=1e-8)
    O = np.vstack([p["offset"], p["offset"] + 0.5, p["offset"], p["offset"]])
    got = nb_profile_psi(p["Y"], p["W"], p["A"], p["psi"], offset=O, block_size=1)
    np.testing.assert_allclose(got, p["offmat"], rtol=1e-8)


def test_mu_floor(ref):
    assert nb_mu_floor() == ref["mu_floor"] == MU_FLOOR == pytest.approx(math.exp(-30), rel=1e-15)


def test_settings_exported():
    assert spanormpy.settings.polish_mem_budget == 1e9
    assert spanormpy.polish_nb is polish_nb


# ---- helpers ---------------------------------------------------------------------


def nbinom(rng, mu, size, shape=None):
    """numpy negative binomial with R's (mu, size) parametrisation."""
    mu = np.asarray(mu, dtype=float)
    p = size / (size + mu)
    return rng.negative_binomial(size, p, size=shape if shape is not None else mu.shape)


def score(y, W, a, psi, pen, off=0.0):
    mu = np.maximum(np.exp(W @ a + off), MU_FLOOR)
    return W.T @ ((y - mu) / (1 + psi * mu)) - pen * a


def spy_batch(monkeypatch):
    """Record the keyword arguments polish_nb hands to polish_batch."""
    seen = []
    real = pol.polish_batch

    def wrapped(Yb, *a, **kw):
        seen.append({"Yb": Yb.cpu().numpy(), **kw})
        return real(Yb, *a, **kw)

    monkeypatch.setattr(pol, "polish_batch", wrapped)
    return seen


def pretend_gpu(monkeypatch):
    """Reach the device (shared-factor) path on the CPU: report a GPU, same device."""
    monkeypatch.setattr(pol, "check_gpu", lambda: True)
    monkeypatch.setattr(pol, "resolve_device", lambda backend="auto": torch.device("cpu"))
    monkeypatch.setattr(pol, "require_float64", lambda *a, **k: None)


# ---- test-polishNB.R ---------------------------------------------------------------


def test_polish_converges_each_gene_to_its_own_optimum():
    rng = np.random.default_rng(10)
    G, n = 20, 400
    W = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    Y = np.vstack([nbinom(rng, np.exp(0.5 + g / 10 + 0.3 * W[:, 1]), 3) for g in range(1, G + 1)])
    A0 = np.zeros((G, 3))
    A0[:, 0] = np.log(Y.mean(1))
    # tol is the relative log-likelihood gain at which Newton stops; at the default the
    # score is left at up to ~3e-6 * sum(y), so a 1e-6 zero-score check needs 1e-12
    pol_ = polish_nb(Y, W, A0, np.full(G, 0.5), lambda_a=0.1, psi_method="fixed", tol=1e-12)
    for g in range(G):
        sc = score(Y[g], W, pol_["alpha"][g], pol_["psi"][g], 0.1)
        assert np.abs(sc).max() / Y[g].sum() < 1e-6
    assert pol_["polish"]["polished"].all()
    np.testing.assert_array_equal(pol_["psi"], 0.5)  # "fixed" keeps the dispersion


def test_gene_and_batch_engines_agree():
    rng = np.random.default_rng(11)
    G, n = 8, 250
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = np.vstack([nbinom(rng, np.exp(1 + 0.2 * W[:, 1]), 2) for _ in range(G)])
    a0, p0 = np.zeros((G, 2)), np.full(G, 0.5)
    kw = {"lambda_a": [0, 1]}
    a = polish_nb(Y, W, a0, p0, engine="gene", psi_method="fixed", **kw)
    b = polish_nb(Y, W, a0, p0, engine="batch", psi_method="fixed", **kw)
    np.testing.assert_allclose(a["alpha"], b["alpha"], rtol=1e-8)
    np.testing.assert_allclose(a["psi"], b["psi"], rtol=1e-6)
    a = polish_nb(Y, W, a0, p0, engine="gene", **kw)
    b = polish_nb(Y, W, a0, p0, engine="batch", **kw)
    np.testing.assert_allclose(a["alpha"], b["alpha"], rtol=1e-6)
    np.testing.assert_allclose(a["psi"], b["psi"], rtol=1e-4)


def test_refuses_non_integer_counts():
    with pytest.raises(ValueError, match="needs integer counts"):
        polish_nb(np.array([[1.5, 2, 3, 4]]), np.ones((4, 1)), np.zeros((1, 1)), 0.1)


@pytest.mark.parametrize("engine", ["batch", "gene"])
def test_gene_with_no_counts_is_flagged_not_fatal(engine):
    rng = np.random.default_rng(12)
    n = 100
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = np.vstack([nbinom(rng, np.full(n, 5.0), 3), np.zeros(n)])
    r = polish_nb(Y, W, np.zeros((2, 2)), [0.3, 0.3], lambda_a=[0, 1], engine=engine)
    assert r["polish"]["polished"].iloc[0]
    assert np.isfinite(r["alpha"]).all() and np.isfinite(r["psi"]).all()


def slope_design(G=5, seed=20):
    """Cell-type intercepts, a covariate, nested sample x cell-type indicators and slopes."""
    rng = np.random.default_rng(seed)
    n = 240
    smp = np.repeat(np.arange(1, 5), 60)
    ct = np.tile(["A", "B"], 120)
    x = rng.normal(size=n)
    Zn = np.column_stack(
        [((smp == s) & (ct == c)).astype(float) for s in range(1, 5) for c in "AB"]
    )
    Zs = np.column_stack([x * (smp == s) for s in range(1, 5)])
    W = np.column_stack([(ct == "A").astype(float), (ct == "B").astype(float), x, Zn, Zs])
    p = W.shape[1]
    u = rng.normal(0, 0.3, 8)
    mu = lambda g: np.exp(0.5 + 0.2 * g * (ct == "A") + 0.3 * x + u[(smp - 1) * 2 + (ct == "B")])
    Y = np.vstack([nbinom(rng, mu(g), 4) for g in range(1, G + 1)])
    return {
        "Y": Y, "W": W, "A0": np.zeros((G, p)), "psi": np.full(G, 0.3),
        "pen": np.r_[0, 0, 0.01, np.full(8, 2.0), np.full(4, 5.0)],
        "start": np.r_[True, True, np.zeros(p - 2, bool)],
        "nested": np.r_[np.zeros(3, bool), np.ones(8, bool), np.zeros(4, bool)],
        "group": np.r_[np.full(3, np.nan), np.repeat(np.arange(1.0, 5), 2), np.arange(1.0, 5)],
    }  # fmt: skip


def test_absorb_batch_reaches_shared_factor_solver_and_grouping_goes_dense(monkeypatch):
    d = slope_design()
    pen_t = as_tensor(d["pen"], "cpu")
    with pytest.raises(ValueError, match="1x1 blocks only"):
        nb_newton_solver_batch(as_tensor(d["W"], "cpu"), pen_t, d["group"])
    base = {"lambda_a": d["pen"], "start_cols": d["start"]}
    cpu = polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb=d["group"], backend="cpu", **base)
    assert cpu["polish"]["polished"].all()

    pretend_gpu(monkeypatch)
    seen = spy_batch(monkeypatch)
    dev = polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb=d["group"],
                    absorb_batch=d["nested"], backend="gpu", **base)  # fmt: skip
    assert seen
    for s in seen:
        assert s["shared_factor"]
        np.testing.assert_array_equal(s["nested"], d["nested"])
    assert dev["polish"]["polished"].all()
    # a shared factorisation refreshes on a different schedule from the per-gene one:
    # the same optimum, not the same path
    np.testing.assert_allclose(dev["alpha"], cpu["alpha"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dev["psi"], cpu["psi"], rtol=1e-5)

    # without absorb_batch the grouping falls back to the dense batched solver
    seen.clear()
    dns = polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb=d["group"], backend="gpu", **base)
    assert seen
    for s in seen:
        assert s["shared_factor"] and not s["nested"].any()
    assert dns["polish"]["polished"].all()
    np.testing.assert_allclose(dns["alpha"], cpu["alpha"], rtol=1e-5, atol=1e-8)

    # and a boolean absorb with no absorb_batch is passed through unchanged
    seen.clear()
    polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb=d["nested"], backend="gpu", **base)
    for s in seen:
        np.testing.assert_array_equal(s["nested"], d["nested"])


def test_absorb_batch_must_be_boolean_over_columns():
    d = slope_design(G=2)
    kw = {"lambda_a": d["pen"], "absorb": d["group"]}
    with pytest.raises(ValueError, match="absorb_batch"):
        polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb_batch=d["group"], **kw)
    with pytest.raises(ValueError, match="absorb_batch"):
        polish_nb(d["Y"], d["W"], d["A0"], d["psi"], absorb_batch=d["nested"][1:], **kw)


def ml_psi(y, mu, rng_=(1e-3, 1e3)):
    """Profile-ML dispersion of one gene at a fixed mean (the oracle for nb_profile_psi)."""
    yt, mt = torch.as_tensor(y, dtype=torch.float64), torch.as_tensor(mu, dtype=torch.float64)

    def f(lp):
        return -float(dnbinom(yt, mu=mt, size=math.exp(-lp), log=True).sum())

    return math.exp(brent_fmin(f, math.log(rng_[0]), math.log(rng_[1]), tol=1e-12))


def test_nb_profile_psi_is_profile_ml_and_keeps_input_on_bound():
    rng = np.random.default_rng(30)
    n = 400
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    A = np.array([[1, 0.3], [0.5, -0.2], [np.log(5), 0]])
    Y = np.vstack([nbinom(rng, np.exp(W @ A[0]), 2), nbinom(rng, np.exp(W @ A[1]), 5),
                   np.full(n, 5)])  # fmt: skip
    psi_in = np.array([0.7, 0.7, 0.123])
    got = nb_profile_psi(Y, W, A, psi_in)
    for g in range(2):
        assert got[g] == pytest.approx(ml_psi(Y[g], np.exp(W @ A[g])), rel=1e-6)
    assert got[2] == 0.123
    # gene blocking is exact
    np.testing.assert_allclose(nb_profile_psi(Y, W, A, psi_in, block_size=1), got, rtol=1e-12)


def test_nb_profile_psi_profiles_at_the_offset_mean():
    rng = np.random.default_rng(31)
    n = 400
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = np.log(rng.uniform(0.3, 3, n))
    A = np.array([[1, 0.3], [0.2, 0.5]])
    Y = np.vstack([nbinom(rng, np.exp(W @ A[g] + off), 3) for g in range(2)])
    with_ = nb_profile_psi(Y, W, A, [0.5, 0.5], offset=off)
    without = nb_profile_psi(Y, W, A, [0.5, 0.5])
    for g in range(2):
        assert with_[g] == pytest.approx(ml_psi(Y[g], np.exp(W @ A[g] + off)), rel=1e-6)
    assert np.abs(np.log(with_) - np.log(without)).min() > 0.01
    # a genes x cells offset gives each gene its own row, also across blocks
    O = np.vstack([off, off + 0.5])
    m = nb_profile_psi(Y, W, A, [0.5, 0.5], offset=O, block_size=1)
    assert m[0] == pytest.approx(with_[0], rel=1e-12)
    assert m[1] == pytest.approx(ml_psi(Y[1], np.exp(W @ A[1] + O[1])), rel=1e-6)


def test_nb_profile_psi_uses_the_same_kernel_as_the_batched_engine():
    rng = np.random.default_rng(32)
    n, p, b = 150, 3, 5
    W = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    alpha = rng.normal(0, 0.2, (b, p))
    alpha[:, 0] = 2
    mu = np.exp(alpha @ W.T)
    Y = np.zeros((b, n))
    for i in range(b - 1):
        Y[i] = nbinom(rng, mu[i], 3)
    Y[b - 1] = rng.poisson(mu[b - 1])  # at_bound: keeps its incoming psi
    psi_in = np.full(b, 0.35)
    got = nb_profile_psi(Y, W, alpha, psi_in)
    assert len(got) == b and np.isfinite(got).all()
    kpsi, kbound = pb.psi_profile_batch(
        torch.as_tensor(Y), torch.as_tensor(np.maximum(np.exp(alpha @ W.T), MU_FLOOR))
    )
    np.testing.assert_array_equal(got[kbound], psi_in[kbound])
    # one optimiser, not two: a free gene's value is the kernel's, exactly
    assert (~kbound).any()
    np.testing.assert_allclose(got[~kbound], kpsi.numpy()[~kbound], rtol=1e-12)


@pytest.mark.parametrize("engine", ["batch", "gene"])
def test_polish_reaches_each_genes_optimum_with_an_offset(engine):
    rng = np.random.default_rng(40)
    G, n = 6, 300
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = np.log(rng.uniform(0.3, 3, n))
    O = np.vstack([off + g / 4 for g in range(1, G + 1)])
    Y = np.vstack([nbinom(rng, np.exp(0.5 + 0.3 * W[:, 1] + O[g]), 4) for g in range(G)])
    pen = np.array([0, 0.5])
    kw = {"lambda_a": pen, "psi_method": "fixed", "tol": 1e-12, "engine": engine}
    v = polish_nb(Y, W, np.zeros((G, 2)), np.full(G, 0.3), offset=off, **kw)
    m = polish_nb(Y, W, np.zeros((G, 2)), np.full(G, 0.3), offset=O, **kw)
    assert v["polish"]["polished"].all() and m["polish"]["polished"].all()
    for g in range(G):
        assert np.abs(score(Y[g], W, v["alpha"][g], 0.3, pen, off)).max() / Y[g].sum() < 1e-6
        assert np.abs(score(Y[g], W, m["alpha"][g], 0.3, pen, O[g])).max() / Y[g].sum() < 1e-6
    # the per-gene offset shifts each intercept by its own g / 4
    np.testing.assert_allclose(m["alpha"][:, 0] - v["alpha"][:, 0], -np.arange(1, G + 1) / 4,
                               atol=1e-4)  # fmt: skip


def test_per_gene_offset_matrix_is_sliced_across_blocks_and_batches():
    rng = np.random.default_rng(41)
    G, n = 7, 150
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    O = np.vstack([np.log(rng.uniform(0.5, 2, n)) + (g - 4) / 3 for g in range(1, G + 1)])
    Y = np.vstack([nbinom(rng, np.exp(0.5 + 0.2 * g * W[:, 1] + O[g - 1]), 4)
                   for g in range(1, G + 1)])  # fmt: skip
    Y[2] = 0  # a gene off the common path
    A0 = np.zeros((G, 2))
    A0[4, 0] = -40  # a degenerate start: restarted
    pen = np.array([0, 0.5])
    kw = {"lambda_a": pen, "offset": O}
    whole = polish_nb(Y, W, A0, np.full(G, 0.3), **kw)
    alone = [
        polish_nb(Y[g : g + 1], W, A0[g : g + 1], 0.3, lambda_a=pen, offset=O[g : g + 1])
        for g in range(G)
    ]
    assert whole["polish"]["restarted"].iloc[4]
    # a batch's GEMMs round differently with its shape, so layouts agree to rounding;
    # a mis-sliced offset row moves an intercept by (g - h) / 3
    np.testing.assert_allclose(whole["alpha"], np.vstack([a["alpha"] for a in alone]), rtol=1e-10,
                               atol=1e-10)  # fmt: skip
    np.testing.assert_allclose(whole["psi"], [a["psi"][0] for a in alone], rtol=1e-10)
    for f in FLAGS:
        np.testing.assert_array_equal(whole["polish"][f], [a["polish"][f][0] for a in alone])
    for bs, bt in [(2, 1), (3, 2), (7, 3)]:
        r = polish_nb(Y, W, A0, np.full(G, 0.3), block_size=bs, batch_size=bt, **kw)
        np.testing.assert_allclose(r["alpha"], whole["alpha"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r["psi"], whole["psi"], rtol=1e-10)
        for f in FLAGS:
            np.testing.assert_array_equal(r["polish"][f], whole["polish"][f])
    g = polish_nb(Y, W, A0, np.full(G, 0.3), engine="gene", psi_method="fixed", block_size=3, **kw)
    b = polish_nb(Y, W, A0, np.full(G, 0.3), psi_method="fixed", block_size=3, **kw)
    np.testing.assert_allclose(g["alpha"], b["alpha"], rtol=1e-8)
    # and a warm pass reads the offset too: from the converged fit it stays put (the
    # all-zero gene is left out, its intercept's optimum being -Inf)
    w = polish_nb(Y, W, whole["alpha"], whole["psi"], warm=True, block_size=2, batch_size=2, **kw)
    keep = np.arange(G) != 2
    # rtol 1e-4: the cold fit stopped at tol 1e-8 (relative log-likelihood gain), so a warm
    # step still moves a coefficient by ~4e-6 on this numpy draw; a dropped offset would
    # move it by O(1)
    np.testing.assert_allclose(w["alpha"][keep], whole["alpha"][keep], rtol=1e-4, atol=1e-9)


def test_bad_offset_is_refused_naming_the_argument_and_shape():
    rng = np.random.default_rng(42)
    n = 30
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((2, n), 4.0), 3)
    A0 = np.zeros((2, 2))

    def run(off):
        return polish_nb(Y, W, A0, [0.3, 0.3], offset=off)

    bad_nan = np.zeros(n)
    bad_nan[2] = np.nan
    bad_inf = np.zeros(n)
    bad_inf[4] = np.inf
    bad_na = np.zeros((2, n))
    bad_na[1, 6] = np.nan
    with pytest.raises(ValueError, match=r"'offset' must be finite.*one value per cell \(30\)"):
        run(bad_nan)
    with pytest.raises(ValueError, match="'offset' must be finite"):
        run(bad_inf)
    with pytest.raises(ValueError, match="'offset' must be finite"):
        run(bad_na)
    with pytest.raises(
        ValueError, match=r"'offset' must be None, a numeric vector with one value per cell \(30\)"
    ):
        run(np.zeros(n - 1))
    with pytest.raises(ValueError, match="'offset' must be None"):
        run(np.zeros(2))
    with pytest.raises(ValueError, match=r"genes x cells matrix \(2 x 30\)"):
        run(np.zeros((3, n)))
    with pytest.raises(ValueError, match=r"genes x cells matrix \(2 x 30\)"):
        run(np.zeros((2, n - 1)))
    with pytest.raises(ValueError, match="'offset' must be None"):
        run(np.zeros((2, n, 1)))
    with pytest.raises(ValueError, match="'offset' must be None"):
        run(np.array([str(v) for v in np.zeros(n)]))
    # nb_profile_psi applies the same check
    with pytest.raises(ValueError, match="'offset' must be finite"):
        nb_profile_psi(Y, W, A0, [0.3, 0.3], offset=bad_nan)
    with pytest.raises(ValueError, match="genes x cells matrix"):
        nb_profile_psi(Y, W, A0, [0.3, 0.3], offset=np.zeros((3, n)))


def test_sparse_offset_made_dense_and_torch_offset_refused():
    rng = np.random.default_rng(43)
    n = 80
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    O = np.vstack([np.log(rng.uniform(0.5, 2, n)), np.full(n, 0.3)])
    Y = np.vstack([nbinom(rng, np.exp(1 + O[g]), 4) for g in range(2)])
    A0 = np.zeros((2, 2))
    base = polish_nb(Y, W, A0, [0.3, 0.3], offset=O)
    sparse = polish_nb(Y, W, A0, [0.3, 0.3], offset=sp.csr_matrix(O))
    np.testing.assert_array_equal(sparse["alpha"], base["alpha"])
    np.testing.assert_array_equal(
        nb_profile_psi(Y, W, base["alpha"], base["psi"], offset=sp.csr_matrix(O)),
        nb_profile_psi(Y, W, base["alpha"], base["psi"], offset=O),
    )
    with pytest.raises(ValueError, match="not a torch tensor"):
        polish_nb(Y, W, A0, [0.3, 0.3], offset=torch.as_tensor(O))


def test_offset_reaches_the_device_path_sliced_per_sub_batch(monkeypatch):
    rng = np.random.default_rng(44)
    G, n = 5, 160
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    O = np.vstack([np.log(rng.uniform(0.5, 2, n)) + g / 3 for g in range(1, G + 1)])
    Y = np.vstack([nbinom(rng, np.exp(0.5 + 0.3 * W[:, 1] + O[g]), 4) for g in range(G)])
    A0 = np.zeros((G, 2))
    A0[1, 0] = np.nan  # a sane start on one gene
    kw = {"lambda_a": [0, 0.5]}
    offs = [O, O[0]]
    cpu = [polish_nb(Y, W, A0, np.full(G, 0.3), offset=o, backend="cpu", **kw) for o in offs]
    pretend_gpu(monkeypatch)
    seen = spy_batch(monkeypatch)
    dev = [polish_nb(Y, W, A0, np.full(G, 0.3), offset=o, backend="gpu", batch_size=2, **kw)
           for o in offs]  # fmt: skip
    # three sub-batches per run (2 + 2 + 1 genes); a matrix offset arrives as exactly the
    # rows of the genes in the sub-batch, a vector as itself
    assert len(seen) == 6
    rows = [[0, 1], [2, 3], [4]]
    for k in range(3):
        assert seen[k]["shared_factor"]
        np.testing.assert_array_equal(seen[k]["Yb"], Y[rows[k]])
        np.testing.assert_array_equal(seen[k]["offset"], O[rows[k]])
        np.testing.assert_array_equal(seen[k + 3]["offset"], O[0])
    for k in range(2):
        assert dev[k]["polish"]["polished"].all()
        np.testing.assert_array_equal(dev[k]["polish"]["restarted"], cpu[k]["polish"]["restarted"])
        np.testing.assert_allclose(dev[k]["alpha"], cpu[k]["alpha"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dev[k]["psi"], cpu[k]["psi"], rtol=1e-5)


def test_batch_size_counts_the_offsets_matrices(capsys):
    rng = np.random.default_rng(45)
    n = 240
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((12, n), 4.0), 3)
    A0 = np.zeros((12, 2))
    old = spanormpy.settings.polish_mem_budget
    try:
        # a budget of exactly ten genes at the no-offset count of six matrices
        spanormpy.settings.polish_mem_budget = 8 * n * 6 * 10

        def opening(off):
            polish_nb(Y, W, A0, np.full(12, 0.3), offset=off, psi_method="fixed", verbose=True)
            return capsys.readouterr().err.splitlines()[0]

        assert "in batches of 10 " in opening(None)
        assert "in batches of 8 " in opening(np.full(n, 0.1))  # a vector: 7 matrices
        assert "in batches of 7 " in opening(np.full((12, n), 0.1))  # a matrix: 8 matrices
    finally:
        spanormpy.settings.polish_mem_budget = old
    assert pb.polish_batch_size(n, budget=1e6) == max(
        1, math.floor(1e6 / (8 * n * pb.POLISH_GENE_CELL_MATS))
    )


@pytest.mark.parametrize("engine", ["batch", "gene"])
def test_psi_range_bounds_the_profile_search(engine):
    rng = np.random.default_rng(46)
    G, n = 4, 300
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = np.vstack([nbinom(rng, np.exp(1 + 0.2 * W[:, 1]), 3) for _ in range(G)])  # psi ~ 1/3
    A0 = np.zeros((G, 2))
    free = polish_nb(Y, W, A0, np.full(G, 0.9), engine=engine)
    assert not free["polish"]["psi_bound"].any()
    # a range excluding the optimum: every search lands on the bound and keeps its input
    tight = polish_nb(Y, W, A0, np.full(G, 0.9), psi_range=(2, 50), engine=engine)
    assert tight["polish"]["psi_bound"].all()
    np.testing.assert_array_equal(tight["psi"], 0.9)


def test_psi_range_validation():
    rng = np.random.default_rng(46)
    n = 60
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((2, n), 4.0), 3)
    for bad in [(5, 1), (0, 1), (1, np.inf), (1, 2, 3), (np.nan, 2)]:
        with pytest.raises(ValueError, match="psi_range"):
            polish_nb(Y, W, np.zeros((2, 2)), [0.9, 0.9], psi_range=bad)


@pytest.mark.parametrize("engine", ["gene", "batch"])
def test_shape_mismatches_are_refused(engine):
    rng = np.random.default_rng(50)
    n = 50
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((2, n), 4.0), 3)
    A0 = np.zeros((2, 2))
    W100 = np.column_stack([np.ones(100), rng.normal(size=100)])
    # on the per-gene engine this used to pass silently: each gene's 50 counts were
    # recycled against a 100-cell mean and came back polished
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'W' must have one row per cell (column of Y): nrow(W) = 100, ncol(Y) = 50"
        ),
    ):
        polish_nb(Y, W100, A0, [0.3, 0.3], engine=engine)  # fmt: skip
    f = {"engine": engine}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'alpha' must have one row per gene (row of Y): nrow(alpha) = 3, nrow(Y) = 2"
        ),
    ):
        polish_nb(Y, W, np.zeros((3, 2)), [0.3, 0.3], **f)  # fmt: skip
    with pytest.raises(ValueError, match=re.escape("nrow(alpha) = 1, nrow(Y) = 2")):
        polish_nb(Y, W, np.zeros((1, 2)), 0.3, **f)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'alpha' must have one column per column of W: ncol(alpha) = 3, ncol(W) = 2"
        ),
    ):
        polish_nb(Y, W, np.zeros((2, 3)), [0.3, 0.3], **f)  # fmt: skip
    with pytest.raises(
        ValueError,
        match=re.escape("'psi' must be one value or one per gene: length(psi) = 3, nrow(Y) = 2"),
    ):
        polish_nb(Y, W, A0, [0.3, 0.3, 0.3], **f)  # fmt: skip
    with pytest.raises(ValueError, match=re.escape("nrow(W) = 100, ncol(Y) = 50")):
        nb_profile_psi(Y, W100, A0, [0.3, 0.3])
    with pytest.raises(ValueError, match=re.escape("nrow(alpha) = 3, nrow(Y) = 2")):
        nb_profile_psi(Y, W, np.zeros((3, 2)), [0.3, 0.3])
    with pytest.raises(ValueError, match=re.escape("ncol(alpha) = 3, ncol(W) = 2")):
        nb_profile_psi(Y, W, np.zeros((2, 3)), [0.3, 0.3])
    with pytest.raises(ValueError, match=re.escape("length(psi) = 3, nrow(Y) = 2")):
        nb_profile_psi(Y, W, A0, [0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match=r"'lambda_a' must be a single value or one per column"):
        polish_nb(Y, W, A0, 0.3, lambda_a=[0, 1, 2])


def test_nb_profile_psi_recycles_a_scalar_psi():
    rng = np.random.default_rng(51)
    n = 100
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    # genes 2 and 3 have no spread, so their search runs to the lower bound and they keep
    # the input psi, which a scalar used to give only gene 1
    Y = np.vstack([nbinom(rng, np.full(n, 5.0), 3), np.full(n, 5), np.full(n, 5)])
    A = np.tile([np.log(5), 0.0], (3, 1))
    got = nb_profile_psi(Y, W, A, 0.2)
    assert not np.isnan(got).any()
    np.testing.assert_array_equal(got, nb_profile_psi(Y, W, A, np.full(3, 0.2)))
    np.testing.assert_array_equal(got[1:], [0.2, 0.2])


def test_threaded_dispatch_equals_serial(monkeypatch):
    rng = np.random.default_rng(52)
    n = 120
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = np.vstack([nbinom(rng, np.exp(1 + 0.2 * W[:, 1]), 3) for _ in range(4)])
    A0 = np.zeros((4, 2))
    serial = polish_nb(Y, W, A0, 0.3, block_size=2)
    for have in (True, False):  # with and without threadpoolctl limiting BLAS threads
        monkeypatch.setattr(pol, "_has_threadpoolctl", lambda have=have: have)
        assert pol._single_blas(2) is have
        assert not pol._single_blas(1)
        assert pol.bplapply_single_blas(range(3), lambda i: i**2, 2) == [0, 1, 4]
        par = polish_nb(Y, W, A0, 0.3, block_size=2, n_jobs=2)
        np.testing.assert_array_equal(par["alpha"], serial["alpha"])
        np.testing.assert_array_equal(par["psi"], serial["psi"])
        np.testing.assert_array_equal(
            nb_profile_psi(Y, W, A0, np.full(4, 0.3), block_size=2, n_jobs=2),
            nb_profile_psi(Y, W, A0, np.full(4, 0.3), block_size=2),
        )


def test_no_gpu_mem_budget_argument():
    import inspect

    assert "gpu_mem_budget" not in inspect.signature(polish_nb).parameters


def test_unfactorable_block_flags_gene_singular_in_both_engines():
    # Sample 4's slope column is zero and unpenalised, so its block of C = Z' diag(w) Z has
    # a zero row and its Cholesky fails; the grouped solver's factor() returns None. One
    # unfactorable gene must not stop every gene's polish: it falls back to its input fit,
    # flagged singular.
    d = slope_design(G=3)
    d["W"][:, 14] = 0
    d["pen"][14] = 0
    sol = nb_newton_solver(as_tensor(d["W"], "cpu"), as_tensor(d["pen"], "cpu"), d["group"])
    st = sol.factor(torch.ones(d["W"].shape[0], dtype=torch.float64))
    assert st is None
    assert sol.solve(st, torch.ones(d["W"].shape[1], dtype=torch.float64)) is None
    assert sol.xcov(st) is None
    for eng in ("gene", "batch"):
        r = polish_nb(d["Y"], d["W"], d["A0"], d["psi"], lambda_a=d["pen"], absorb=d["group"],
                      start_cols=d["start"], engine=eng)  # fmt: skip
        assert r["polish"]["singular"].all(), eng
        assert not r["polish"]["polished"].any(), eng
        np.testing.assert_allclose(r["alpha"], d["A0"])


def test_block_design_rejects_absorb_and_device(monkeypatch):
    rng = np.random.default_rng(53)
    n = 60
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    Z = np.ones((n, 1))
    D = nb_block_design(X, Z, np.repeat(["s1", "s2"], 30))
    Y = nbinom(rng, np.full((2, n), 4.0), 3)
    A0 = np.zeros((2, D.shape[1]))
    with pytest.raises(ValueError, match="must be None for an nbBlockDesign"):
        polish_nb(Y, D, A0, 0.3, absorb=np.zeros(D.shape[1], bool))
    with pytest.raises(ValueError, match="must be None for an nbBlockDesign"):
        polish_nb(Y, D, A0, 0.3, absorb_batch=np.zeros(D.shape[1], bool))
    pretend_gpu(monkeypatch)
    with pytest.raises(ValueError, match="CPU only"):
        polish_nb(Y, D, A0, 0.3, backend="gpu")


def test_gpu_needs_batch_engine_and_float64(monkeypatch):
    rng = np.random.default_rng(54)
    n = 40
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((2, n), 4.0), 3)
    pretend_gpu(monkeypatch)
    with pytest.raises(ValueError, match='needs engine = "batch"'):
        polish_nb(Y, W, np.zeros((2, 2)), 0.3, engine="gene", backend="gpu")
    with pytest.warns(UserWarning, match="forcing serial dispatch"):
        polish_nb(Y, W, np.zeros((2, 2)), 0.3, backend="gpu", n_jobs=2)
    with pytest.raises(ValueError, match="single precision"):
        pb.require_float64(torch.float32)
    pb.require_float64(torch.float64)


def test_sparse_counts_equal_dense():
    rng = np.random.default_rng(55)
    n = 120
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((4, n), 4.0), 3)
    a = polish_nb(Y, W, np.zeros((4, 2)), 0.3)
    b = polish_nb(sp.csr_matrix(Y), W, np.zeros((4, 2)), 0.3)
    np.testing.assert_array_equal(a["alpha"], b["alpha"])


# ---- test-polishEngine.R: kernels ----------------------------------------------------


def nb_fixture(n=50, p=4, b=5, seed=17):
    rng = np.random.default_rng(seed)
    W = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    A = rng.normal(0, 0.3, (b, p))
    A[:, 0] += 1.5
    Y = nbinom(rng, np.exp(A @ W.T), 2)
    return {"W": W, "A": A, "Y": Y.astype(float), "psi": rng.uniform(0.1, 1.5, b),
            "pen": rng.uniform(0, 0.4, p)}  # fmt: skip


def mu_ref(f):
    return np.maximum(np.exp(f["A"] @ f["W"].T), MU_FLOOR)


def ll_ref(f, M):
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    size = t(np.divide(1.0, f["psi"], out=np.full_like(f["psi"], np.inf), where=f["psi"] > 0))
    ll = dnbinom(t(f["Y"]), mu=t(M), size=size[:, None], log=True).sum(1).numpy()
    return ll - 0.5 * (f["A"] ** 2) @ f["pen"]


def test_mu_batch_reproduces_the_mean_including_the_floor():
    f = nb_fixture()
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    np.testing.assert_allclose(pb.mu_batch(t(f["A"]), t(f["W"])).numpy(), mu_ref(f), rtol=1e-12)
    # drive the INTERCEPT down so the floor bites (all -50 would give large positive eta
    # wherever rowSums(W) is negative)
    f["A"][0] = np.r_[-50, np.zeros(f["A"].shape[1] - 1)]
    got = pb.mu_batch(t(f["A"]), t(f["W"])).numpy()
    assert (got[0] == MU_FLOOR).all()
    np.testing.assert_allclose(got, mu_ref(f), rtol=1e-12)


def test_mu_batch_refuses_a_bare_per_cell_offset():
    t = torch.ones(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="offset_rows"):
        pb.mu_batch(t, torch.ones(10, 2, dtype=torch.float64), torch.zeros(10, dtype=torch.float64))


def test_nb_loglik_batch_matches_dnbinom_all_zero_gene_and_large_count():
    f = nb_fixture()
    f["Y"][0] = 0
    f["Y"][1] *= 100
    M = mu_ref(f)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    got = pb.nb_loglik_batch(t(f["Y"]), t(M), f["psi"], t(f["A"]), f["pen"]).numpy()
    np.testing.assert_allclose(got, ll_ref(f, M), rtol=1e-10)
    assert np.isfinite(got).all()


def test_nb_loglik_batch_zero_dispersion_is_poisson_and_mixes_with_nb():
    # psi = 0 is size = Inf, which dnbinom() treats as Poisson. A written-out log-pmf gives
    # Inf - Inf = NaN there, and the engine would silently polish nothing.
    f = nb_fixture()
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    for zero in (slice(None), [1, 3]):
        f["psi"][zero] = 0
        M = mu_ref(f)
        got = pb.nb_loglik_batch(t(f["Y"]), t(M), f["psi"], t(f["A"]), f["pen"]).numpy()
        assert np.isfinite(got).all()
        pois = (f["Y"] * np.log(M) - M - np.vectorize(math.lgamma)(f["Y"] + 1)).sum(1)
        want = pois - 0.5 * (f["A"] ** 2) @ f["pen"]
        np.testing.assert_allclose(got[zero], want[zero], rtol=1e-10)
        nb = ~np.isin(np.arange(5), np.arange(5)[zero])
        np.testing.assert_allclose(got[nb], ll_ref(f, M)[nb], rtol=1e-10)


# ---- test-polishEngine.R: profile dispersion ---------------------------------------


def psi_fixture(n=200, b=6, seed=5):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(2, 0.3, (b, n)))
    Y = np.zeros((b, n))
    psis = [0.05, 0.2, 0.8, 2, 10][: b - 1]
    for i in range(b - 1):
        Y[i] = nbinom(rng, mu[i], 1 / psis[i])
    Y[b - 1] = rng.poisson(mu[b - 1])  # exactly Poisson: sits below the range, at_bound
    return Y, mu, (1e-3, 1e3)


def nb_ll(y, mu, psi):
    return float(dnbinom(torch.as_tensor(y), mu=torch.as_tensor(mu), size=1 / psi, log=True).sum())


def psi_ref(Y, Mu, rng_):
    """optimize() per gene, as the per-gene engine does."""
    lo, hi = math.log(rng_[0]), math.log(rng_[1])
    est, bnd = [], []
    for y, mu in zip(Y, Mu, strict=True):
        m = brent_fmin(lambda lp, y=y, mu=mu: -nb_ll(y, mu, math.exp(lp)), lo, hi)
        bnd.append((m - lo) < 1e-3 * (hi - lo) or (hi - m) < 1e-3 * (hi - lo))
        est.append(math.exp(m))
    return np.array(est), np.array(bnd)


def test_psi_profile_batch_agrees_with_optimize_and_is_never_worse():
    Y, Mu, rng_ = psi_fixture()
    ref_psi, ref_bnd = psi_ref(Y, Mu, rng_)
    got, bnd = pb.psi_profile_batch(torch.as_tensor(Y), torch.as_tensor(Mu), rng_)
    got = got.numpy()
    free = ~ref_bnd
    assert free.any()
    assert np.abs(np.log(got[free]) - np.log(ref_psi[free])).max() < 2e-4
    np.testing.assert_array_equal(bnd, ref_bnd)  # the at_bound rule
    assert bnd[-1]  # the Poisson gene is the one flagged
    for i in range(len(Y)):  # the gate: the bisection must not find a worse optimum
        ll_old = nb_ll(Y[i], Mu[i], ref_psi[i])
        assert nb_ll(Y[i], Mu[i], got[i]) >= ll_old - 1e-8 * abs(ll_old)


def test_psi_profile_batch_is_fixed_iteration_so_more_steps_only_refine():
    Y, Mu, rng_ = psi_fixture()
    ty, tm = torch.as_tensor(Y), torch.as_tensor(Mu)
    a, ba = pb.psi_profile_batch(ty, tm, rng_, maxit=50)
    b, bb = pb.psi_profile_batch(ty, tm, rng_, maxit=60)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-8)
    np.testing.assert_array_equal(ba, bb)


# ---- test-polishEngine.R: the batched Newton against the per-gene oracle ---------------


def toy_batch(n=240, G=6, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    ct = np.tile([1, 2], n // 2)
    X = np.column_stack([np.ones(n), (ct == 2).astype(float), x])
    grp = np.arange(n) % G
    Z = np.zeros((n, G))
    Z[np.arange(n), grp] = 1
    W = np.column_stack([X, Z])
    pen = np.r_[0, 0, 0, np.full(G, 1 / 0.05)]
    return {
        "W": as_tensor(W, "cpu"), "Wn": W, "pen": as_tensor(pen, "cpu"),
        "nested": np.r_[np.zeros(3, bool), np.ones(G, bool)],
        "ct_cols": np.r_[True, True, np.zeros(1 + G, bool)],
        "rng": rng,
    }  # fmt: skip


def gene_by_gene(Yb, d, A0, psi0, solver, **kw):
    out = [polish_gene(Yb[i], d["W"], A0[i], psi0[i], d["pen"], solver,
                       start_cols=d["ct_cols"], **kw) for i in range(len(Yb))]  # fmt: skip
    res = {k: np.array([o[k] for o in out]) for k in out[0] if k != "alpha"}
    res["alpha"] = np.vstack([o["alpha"] for o in out])
    return res


def same_fit(b, g, tol=1e-12):
    for f in ("alpha", "psi", "loglik"):
        np.testing.assert_allclose(b[f], g[f], rtol=tol, atol=1e-14, err_msg=f)
    for f in FLAGS:
        np.testing.assert_array_equal(b[f], g[f], err_msg=f)


def no_worse(b, g):
    assert (b["loglik"] >= g["loglik"] - 1e-9 * np.abs(g["loglik"])).all()


def batch_of(d, Yb, A0, psi0, solver, **kw):
    return pb.polish_batch(torch.as_tensor(Yb, dtype=torch.float64), d["W"], A0, psi0, d["pen"],
                           solver, start_cols=d["ct_cols"], **kw)  # fmt: skip


def test_a_batch_of_one_reproduces_polish_gene_exactly():
    d = toy_batch()
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    y = nbinom(np.random.default_rng(1), np.full((1, 240), 6.0), 3).astype(float)
    A0 = np.zeros((1, d["W"].shape[1]))
    A0[0, 0] = np.log(y.mean())
    # "fixed" takes the one deliberate divergence (bisection vs optimize) out of the
    # comparison and keeps this a strict test of the Newton machinery
    b = batch_of(d, y, A0, 0.4, solver, psi_method="fixed")
    g = gene_by_gene(y, d, A0, [0.4], solver, psi_method="fixed")
    same_fit(b, g)


def test_a_batch_reproduces_the_same_genes_one_at_a_time():
    d = toy_batch()
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    rng = np.random.default_rng(2)
    B = 13
    Yb = np.vstack([nbinom(rng, np.full(240, [2, 8, 30][i % 3]), 2) for i in range(B)]).astype(
        float
    )
    A0 = np.zeros((B, d["W"].shape[1]))
    A0[:, 0] = np.log(np.maximum(Yb.mean(1), 0.1))
    psi0 = np.full(B, 0.4)
    b = batch_of(d, Yb, A0, psi0, solver, psi_method="fixed")
    g = gene_by_gene(Yb, d, A0, psi0, solver, psi_method="fixed")
    same_fit(b, g)


def test_the_batch_boundary_does_not_move_a_genes_answer():
    # GEMMs round with batch shape, so this is a tolerance, not machine precision
    d = toy_batch()
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    rng = np.random.default_rng(3)
    B = 13
    Yb = np.vstack([nbinom(rng, np.full(240, [3, 12][i % 2]), 2) for i in range(B)]).astype(float)
    A0 = np.zeros((B, d["W"].shape[1]))
    A0[:, 0] = np.log(np.maximum(Yb.mean(1), 0.1))
    full = batch_of(d, Yb, A0, np.full(B, 0.4), solver)
    for bs in (1, 2, 5):
        parts = [batch_of(d, Yb[i : i + bs], A0[i : i + bs], np.full(len(Yb[i : i + bs]), 0.4),
                          solver) for i in range(0, B, bs)]  # fmt: skip
        for f in ("alpha", "psi", "loglik"):
            np.testing.assert_allclose(np.concatenate([p[f] for p in parts]), full[f], rtol=1e-9,
                                       atol=1e-12, err_msg=f"{f} batch size {bs}")  # fmt: skip


def test_one_batch_holds_genes_taking_every_different_path():
    # a converged gene, a degenerate start, an all-zero gene, a hard line search and a gene
    # whose psi sits on a bound, in one batch against the same five run singly: this is what
    # catches per-gene state leaking across slices
    d = toy_batch()
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    rng = np.random.default_rng(5)
    n = 240
    Yb = np.vstack([nbinom(rng, np.full(n, 8.0), 3), nbinom(rng, np.full(n, 40.0), 5),
                    np.zeros(n), rng.poisson(4, n), nbinom(rng, np.full(n, 2.0), 1)]).astype(float)  # fmt: skip
    B = len(Yb)
    A0 = np.zeros((B, d["W"].shape[1]))
    A0[:, 0] = np.log(np.maximum(Yb.mean(1), 0.1))
    A0[1] = -30  # a degenerate start: forces a restart
    psi0 = np.array([0.4, 0.2, 0.5, 0.3, 0.8])
    b = batch_of(d, Yb, A0, psi0, solver)
    g = gene_by_gene(Yb, d, A0, psi0, solver)
    # the profile dispersion is Brent per gene and a bisection in the batch: numeric fields
    # agree at the search's accuracy, flags and iteration counts exactly
    same_fit(b, g, tol=1e-4)
    no_worse(b, g)
    np.testing.assert_array_equal(b["iterations"], g["iterations"])
    assert g["restarted"].any() and g["psi_bound"].any()  # the fixture must exercise them


def test_a_gene_that_cannot_be_polished_does_not_poison_its_batch():
    d = toy_batch()
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    rng = np.random.default_rng(6)
    B = 4
    Yb = nbinom(rng, np.full((B, 240), 7.0), 3).astype(float)
    A0 = np.zeros((B, d["W"].shape[1]))
    A0[:, 0] = np.log(Yb.mean(1))
    A0[1, 2] = np.nan
    psi0 = np.full(B, 0.4)
    b = batch_of(d, Yb, A0, psi0, solver, warm=True)
    g = gene_by_gene(Yb, d, A0, psi0, solver, warm=True)
    assert not b["polished"][1]
    np.testing.assert_array_equal(b["alpha"][1], A0[1])  # the input fit is kept, not the sane start
    assert b["polished"][[0, 2, 3]].all()
    same_fit(b, g)
    keep = [0, 2, 3]  # identical to those run without the failing gene present at all
    ok = batch_of(d, Yb[keep], A0[keep], psi0[keep], solver, warm=True)
    np.testing.assert_allclose(b["alpha"][keep], ok["alpha"], rtol=1e-12)


def test_polish_batch_reports_what_a_shared_factorisation_would_cost():
    d = toy_batch()
    rng = np.random.default_rng(4)
    B = 8
    mu = np.exp(d["Wn"] @ np.r_[1.2, 0.4, 0.3, np.zeros(6)])
    Yb = np.vstack([nbinom(rng, mu, 1 / 0.4) for _ in range(B)]).astype(float)
    A0 = np.zeros((B, 9))
    A0[:, 0] = np.log(np.maximum(Yb.mean(1), 0.1))
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    nf = batch_of(d, Yb, A0, np.full(B, 0.4), solver)["factorisations"]
    assert set(nf) == {"pergene", "sync"}
    assert nf["pergene"] >= B  # every gene is factorised once before its first step
    assert nf["sync"] >= nf["pergene"]  # a shared factorisation never builds fewer


def test_a_shared_factorisation_reaches_the_same_optimum_as_the_per_gene_one():
    d = toy_batch()
    rng = np.random.default_rng(21)
    B = 12
    mu = np.exp(d["Wn"] @ np.r_[1.2, 0.4, 0.3, np.zeros(6)])
    Yb = np.vstack([nbinom(rng, mu, 1 / 0.4) for _ in range(B)]).astype(float)
    A0 = np.zeros((B, 9))
    A0[:, 0] = np.log(np.maximum(Yb.mean(1), 0.1))
    solver = nb_newton_solver(d["W"], d["pen"], d["nested"])
    per = batch_of(d, Yb, A0, np.full(B, 0.4), solver)
    shd = batch_of(d, Yb, A0, np.full(B, 0.4), solver, shared_factor=True, nested=d["nested"])
    for f in ("polished", "singular", "capped"):
        np.testing.assert_array_equal(shd[f], per[f])
    np.testing.assert_allclose(shd["alpha"], per["alpha"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(shd["psi"], per["psi"], rtol=1e-5)
    # the gate: the penalised log-likelihood is never worse
    assert (shd["loglik"] - per["loglik"]).min() > -1e-7 * np.abs(per["loglik"]).max()


# ---- test-polishEngine.R: the optional per-cell offset ---------------------------------


def test_polish_gene_reaches_a_zero_penalised_score_with_an_offset():
    rng = np.random.default_rng(1)
    n = 300
    W = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    off = np.log(rng.uniform(0.5, 2, n))
    y = nbinom(rng, np.exp(1 + 0.4 * W[:, 1] + off), 5).astype(float)
    pen = np.array([0, 1.0, 1.0])
    Wt, pt = as_tensor(W, "cpu"), as_tensor(pen, "cpu")
    r = polish_gene(y, Wt, np.zeros(3), 0.2, pt, nb_newton_solver(Wt, pt, None),
                    psi_method="fixed", offset=off)  # fmt: skip
    assert np.abs(score(y, W, r["alpha"], r["psi"], pen, off)).max() / y.sum() < 1e-6
    assert r["polished"]


def test_far_start_converges_to_the_intercept_net_of_a_constant_offset():
    rng = np.random.default_rng(2)
    n = 200
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = np.full(n, 3.0)
    y = nbinom(rng, np.full(n, np.exp(0.5 + 3.0)), 10).astype(float)
    Wt, pt = as_tensor(W, "cpu"), as_tensor([0.0, 0.0], "cpu")
    # a0 = 40 is far from the optimum but not degenerate (its fitted log-mean is 43)
    r = polish_gene(y, Wt, [40.0, 0.0], 0.1, pt, nb_newton_solver(Wt, pt, None),
                    psi_method="fixed", offset=off)  # fmt: skip
    assert not r["restarted"]
    assert r["alpha"][0] == pytest.approx(0.5, abs=0.1)


def test_batch_and_gene_engines_agree_with_a_vector_and_a_matrix_offset():
    rng = np.random.default_rng(3)
    G, n = 6, 150
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = np.log(rng.uniform(0.5, 2, n))
    Y = np.vstack([nbinom(rng, np.exp(0.5 + 0.2 * g * W[:, 1] + off), 4)
                   for g in range(1, G + 1)]).astype(float)  # fmt: skip
    pen = np.array([0, 0.5])
    Wt, pt = as_tensor(W, "cpu"), as_tensor(pen, "cpu")
    s = nb_newton_solver(Wt, pt, None)
    per = np.vstack([polish_gene(Y[g], Wt, np.zeros(2), 0.25, pt, s, psi_method="fixed",
                                 offset=off)["alpha"] for g in range(G)])  # fmt: skip
    ty = torch.as_tensor(Y)
    bv = pb.polish_batch(ty, Wt, np.zeros((G, 2)), np.full(G, 0.25), pt, s, psi_method="fixed",
                         offset=off)  # fmt: skip
    bm = pb.polish_batch(ty, Wt, np.zeros((G, 2)), np.full(G, 0.25), pt, s, psi_method="fixed",
                         offset=np.tile(off, (G, 1)))  # fmt: skip
    np.testing.assert_allclose(bv["alpha"], per, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(bm["alpha"], bv["alpha"])


def test_offset_none_equals_no_offset():
    rng = np.random.default_rng(4)
    n = 120
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = nbinom(rng, np.exp(1 + 0.3 * W[:, 1]), 3).astype(float)
    Wt, pt = as_tensor(W, "cpu"), as_tensor([0, 0.1], "cpu")
    s = nb_newton_solver(Wt, pt, None)
    a = polish_gene(y, Wt, [0, 0], 0.3, pt, s)
    b = polish_gene(y, Wt, [0, 0], 0.3, pt, s, offset=None)
    c = polish_gene(y, Wt, [0, 0], 0.3, pt, s, offset=np.zeros(n))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a[k], c[k])


def test_degenerate_start_check_reads_the_linear_predictor_with_its_offset():
    rng = np.random.default_rng(5)
    n = 150
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Wt, pt = as_tensor(W, "cpu"), as_tensor([0.0, 0.0], "cpu")
    s = nb_newton_solver(Wt, pt, None)
    kw = {"psi_method": "fixed"}
    # -11 + 12 = 1 is a fitted log-mean at the optimum, so not degenerate; read without the
    # offset it is -11 < -10 and would restart
    up = np.full(n, 12.0)
    y = nbinom(rng, np.full(n, np.exp(-11 + 12.0)), 5).astype(float)
    g = polish_gene(y, Wt, [-11, 0], 0.2, pt, s, offset=up, **kw)
    b = pb.polish_batch(torch.as_tensor(y[None]), Wt, [[-11, 0]], 0.2, pt, s, offset=up, **kw)
    assert not g["restarted"] and not b["restarted"][0]
    # -5 - 8 = -13 is below -10: degenerate, restarted from the sane start
    down = np.full(n, -8.0)
    y = nbinom(rng, np.full(n, np.exp(9 - 8.0)), 5).astype(float)
    g = polish_gene(y, Wt, [-5, 0], 0.2, pt, s, offset=down, **kw)
    b = pb.polish_batch(torch.as_tensor(y[None]), Wt, [[-5, 0]], 0.2, pt, s, offset=down, **kw)
    assert g["restarted"] and b["restarted"][0]


def test_sane_start_is_the_log_mean_net_of_the_offset_in_both_engines():
    rng = np.random.default_rng(6)
    n = 200
    ct = np.tile([True, False], n // 2)
    W = np.column_stack([ct.astype(float), (~ct).astype(float), rng.normal(size=n)])
    pen = np.zeros(3)
    Wt, pt = as_tensor(W, "cpu"), as_tensor(pen, "cpu")
    s = nb_newton_solver(Wt, pt, None)
    G = 3
    O = np.vstack([np.log(rng.uniform(0.5, 2, n)) + np.where(ct, g, -g) for g in range(1, G + 1)])
    Y = np.vstack([nbinom(rng, np.exp(0.3 * W[:, 2] + O[g] + 0.5), 5)
                   for g in range(G)]).astype(float)  # fmt: skip
    A0 = np.zeros((G, 3))
    A0[:, 0] = np.nan  # a non-finite start: always the sane start
    for sc in (np.array([True, True, False]), None):
        # maxit = 0 returns the start itself, so the formula is checked directly
        for g in range(G):
            r = polish_gene(Y[g], Wt, A0[g], 0.2, pt, s, maxit=0, start_cols=sc,
                            psi_method="fixed", offset=O[g])  # fmt: skip
            if sc is None:
                want = [np.log(Y[g].mean() + 1e-3) - O[g].mean(), 0, 0]
            else:
                want = [np.log(Y[g, ct].mean() + 1e-3) - O[g, ct].mean(),
                        np.log(Y[g, ~ct].mean() + 1e-3) - O[g, ~ct].mean(), 0]  # fmt: skip
            assert r["restarted"]
            np.testing.assert_allclose(r["alpha"], want, rtol=1e-12, atol=1e-14)
        # the batched engine takes at least one step: compare one step from each engine's start
        A1 = A0.copy()
        A1[0] = [0.5, 0.5, 0]
        per = np.vstack([polish_gene(Y[g], Wt, A1[g], 0.2, pt, s, maxit=1, start_cols=sc,
                                     psi_method="fixed", offset=O[g])["alpha"]
                         for g in range(G)])  # fmt: skip
        bat = pb.polish_batch(torch.as_tensor(Y), Wt, A1, np.full(G, 0.2), pt, s, maxit=1,
                              start_cols=sc, psi_method="fixed", offset=O)  # fmt: skip
        np.testing.assert_array_equal(bat["restarted"], [False] + [True] * (G - 1))
        np.testing.assert_allclose(bat["alpha"], per, rtol=1e-10, atol=1e-12)


def test_per_gene_offset_matrix_reaches_each_gene_on_every_path():
    # the all-zero first gene's dispersion runs to a bound, so the profile re-polish runs on
    # rows 2..B: positions in `rows` are then not batch rows, where a mis-sliced offset would
    # show. Each gene alone in its own batch is the reference.
    rng = np.random.default_rng(7)
    G, n = 4, 160
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    pen = np.array([0, 0.5])
    Wt, pt = as_tensor(W, "cpu"), as_tensor(pen, "cpu")
    s = nb_newton_solver(Wt, pt, None)
    O = np.vstack([np.log(rng.uniform(0.5, 2, n)) + g - 2 for g in range(1, G + 1)])
    Y = np.vstack([nbinom(rng, np.exp(0.5 + 0.3 * W[:, 1] + O[g]), 4)
                   for g in range(G)]).astype(float)  # fmt: skip
    Y[0] = 0
    A0 = np.column_stack([np.log(np.maximum(Y.mean(1), 0.1)) - O.mean(1), np.zeros(G)])
    ty = torch.as_tensor(Y)
    full = pb.polish_batch(ty, Wt, A0, np.full(G, 0.3), pt, s, offset=O)
    one = [pb.polish_batch(ty[g : g + 1], Wt, A0[g : g + 1], 0.3, pt, s, offset=O[g : g + 1])
           for g in range(G)]  # fmt: skip
    assert full["psi_bound"][0] and full["polished"][1:].all()
    np.testing.assert_allclose(full["alpha"], np.vstack([o["alpha"] for o in one]), rtol=1e-9,
                               atol=1e-10)  # fmt: skip
    np.testing.assert_allclose(full["psi"], [o["psi"][0] for o in one], rtol=1e-9)
    np.testing.assert_array_equal(full["iterations"], [o["iterations"][0] for o in one])


def test_mis_shaped_offset_is_refused_not_recycled():
    rng = np.random.default_rng(8)
    n = 40
    W = np.column_stack([np.ones(n), rng.normal(size=n)])
    Y = nbinom(rng, np.full((3, n), 4.0), 3).astype(float)
    Wt, pt = as_tensor(W, "cpu"), as_tensor([0.0, 0.0], "cpu")
    s = nb_newton_solver(Wt, pt, None)
    with pytest.raises(ValueError, match="'offset' must have one value per cell"):
        polish_gene(Y[0], Wt, [1, 0], 0.3, pt, s, offset=np.zeros(n - 1))
    ty = torch.as_tensor(Y)
    for bad in (np.zeros(3), np.zeros((2, n))):
        with pytest.raises(ValueError, match="'offset' must have one value per cell"):
            pb.polish_batch(ty, Wt, np.ones((3, 2)), np.full(3, 0.3), pt, s, offset=bad)
    with pytest.raises(ValueError, match="offset_rows"):
        pb.mu_batch(torch.ones(3, 2, dtype=torch.float64), Wt, torch.zeros(n, dtype=torch.float64))


def test_brent_fmin_matches_r_optimize_on_a_known_case():
    # R: optimize(function(x) (x - 1/3)^2, c(0, 1))$minimum = 0.3333333
    x = brent_fmin(lambda t: (t - 1 / 3) ** 2, 0.0, 1.0)
    assert x == pytest.approx(1 / 3, abs=1e-6)
    # and agrees with scipy's bounded Brent to the shared tolerance on a non-quadratic
    f = lambda t: math.cosh(t - 0.7) + 0.1 * t
    assert brent_fmin(f, -3, 3, tol=1e-10) == pytest.approx(
        optimize.minimize_scalar(f, bounds=(-3, 3), method="bounded",
                                 options={"xatol": 1e-10}).x, abs=1e-6)  # fmt: skip
