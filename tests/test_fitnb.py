"""Tests for the NB fitting core (fitSpaNormNB.R) against R references."""

import re

import numpy as np
import pytest
from tests._ref import load_ref

from spanormpy import _fitnb
from spanormpy._backend import gene_block_indices
from spanormpy._edger import EdgeRError
from spanormpy._fitnb import (
    calculate_mu,
    fit_nb,
    fit_nb_given_psi,
    fit_spanorm_nb,
)

RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture(scope="module")
def ref():
    return load_ref("fitnb")


def _backend(device):
    return "cpu" if device == "cpu" else "gpu"


@pytest.fixture
def inject_psi_idx(monkeypatch, ref):
    """Replace the dispersion-subset draw by R's recorded draw."""
    pick = np.asarray(ref["psi_pick"]).astype(int) - 1

    def fake(idx, maxn_psi, rng):
        nsub = int(np.sum(idx))
        out = np.zeros(nsub, dtype=bool)
        out[pick] = True
        return out

    monkeypatch.setattr(_fitnb, "_sample_psi_idx", fake)


def _check_fit(got, exp, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(got["alpha"], exp["alpha"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(got["gmean"], exp["gmean"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(got["psi"], exp["psi"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.atleast_1d(got["loglik"]), np.atleast_1d(exp["loglik"]), rtol=rtol
    )
    assert list(got["sampling"]) == list(np.atleast_1d(exp["sampling"]))


def test_fit_spanorm_nb_matches_r(ref, device, inject_psi_idx):
    exp = ref["spanorm"]
    got = fit_spanorm_nb(
        ref["Y"],
        ref["W"],
        ref["idx"].astype(bool),
        maxit_psi=3,
        maxit_nb=10,
        tol=1e-4,
        is_spanorm=True,
        lambda_a=ref["lambda_a"],
        backend=_backend(device),
        msgfun=lambda m: None,
    )
    _check_fit(got, exp)
    # the shared logLS coefficient is identical across genes
    assert np.unique(got["alpha"][:, 0]).size == 1


def test_fit_spanorm_nb_messages_match_r(ref, inject_psi_idx):
    msgs = []
    fit_spanorm_nb(
        ref["Y"],
        ref["W"],
        ref["idx"].astype(bool),
        maxit_psi=2,
        maxit_nb=10,
        tol=1e-4,
        is_spanorm=True,
        lambda_a=ref["lambda_a"],
        backend="cpu",
        msgfun=msgs.append,
    )
    exp = [str(m) for m in np.atleast_1d(ref["spanorm_msgs"])]
    assert len(msgs) == len(exp)
    num = re.compile(r"-?\d+\.\d+")
    for g, e in zip(msgs, exp, strict=True):
        # same text; log-likelihood numbers agree to the printed (6 d.p.) precision of R
        assert num.sub("#", g) == num.sub("#", e)
        for a, b in zip(num.findall(g), num.findall(e), strict=True):
            assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_fit_nb_offset_matches_r(ref, device, inject_psi_idx):
    exp = ref["offset"]
    got = fit_nb(
        ref["Y"],
        ref["Wg"],
        idx=ref["idx"].astype(bool),
        maxit_psi=3,
        maxit_nb=10,
        offset=ref["O"],
        backend=_backend(device),
        verbose=False,
    )
    _check_fit(got, exp)
    assert np.all(got["gmean"] == 0)


def test_fit_nb_generic_ridge_winsor_matches_r(ref, device, inject_psi_idx):
    got = fit_nb(
        ref["Y"],
        ref["Wg"],
        idx=ref["idx"].astype(bool),
        maxit_psi=3,
        maxit_nb=10,
        lambda_a=0.5,
        winsor=2,
        backend=_backend(device),
        verbose=False,
    )
    _check_fit(got, ref["generic"])


def test_fit_nb_fixed_psi_matches_r(ref, device):
    got = fit_nb(
        ref["Y"],
        ref["Wg"],
        idx=ref["idx"].astype(bool),
        psi=ref["p0"],
        offset=ref["O"],
        maxit_nb=10,
        backend=_backend(device),
        verbose=False,
    )
    _check_fit(got, ref["fixed_psi"])
    assert np.isscalar(got["loglik"]) or np.ndim(got["loglik"]) == 0
    assert "dispersion" not in set(got["sampling"])
    np.testing.assert_array_equal(got["psi"], ref["p0"])

    got0 = fit_nb(
        ref["Y"], ref["Wg"], psi=ref["p0"], maxit_nb=10, backend=_backend(device), verbose=False
    )
    _check_fit(got0, ref["fixed_psi_noidx"])


@pytest.mark.parametrize("nb", [1, 3])
def test_fit_nb_given_psi_blocks_match_r(ref, device, nb):
    idx = ref["idx"].astype(bool)
    exp = ref[f"givenpsi_b{nb}"]
    ng = ref["Y"].shape[0]
    got = fit_nb_given_psi(
        ref["Y"][:, idx],
        ref["W"][idx],
        ref["givenpsi_psi"],
        lambda_a=ref["lambda_a"],
        is_spanorm=True,
        maxit_nb=10,
        blocks=gene_block_indices(ng, nb),
        backend=_backend(device),
        msgfun=lambda m: None,
    )
    np.testing.assert_allclose(got["alpha"], exp["alpha"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got["gmean"], exp["gmean"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got["loglik"], exp["loglik"], rtol=RTOL)
    np.testing.assert_allclose(got["loglik_iter"], np.atleast_1d(exp["loglik_iter"]), rtol=RTOL)


@pytest.mark.parametrize("is_spanorm", [True, False])
def test_blocked_equals_unblocked(ref, device, is_spanorm):
    idx = ref["idx"].astype(bool)
    Y, W = ref["Y"][:, idx], (ref["W"] if is_spanorm else ref["Wg"])[idx]
    ng = Y.shape[0]
    psi = ref["givenpsi_psi"]
    lam = ref["lambda_a"] if is_spanorm else 0.1
    off = None if is_spanorm else ref["O"][:, idx]
    res = {}
    for nb in (1, 3, 7):
        blocks = gene_block_indices(ng, nb)
        assert len(blocks) > 1 or nb == 1
        res[nb] = fit_nb_given_psi(
            Y,
            W,
            psi,
            lambda_a=lam,
            is_spanorm=is_spanorm,
            offset=off,
            maxit_nb=6,
            blocks=blocks,
            backend=_backend(device),
            msgfun=lambda m: None,
        )
    for nb in (3, 7):
        np.testing.assert_allclose(res[nb]["alpha"], res[1]["alpha"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(res[nb]["gmean"], res[1]["gmean"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(res[nb]["loglik"], res[1]["loglik"], rtol=1e-10)


def test_forced_gpu_budget_blocks_cpu_unchanged(ref):
    # R: backend = "cpu" never engages blocking, whatever the budget (even an invalid one)
    kw = {
        "idx": ref["idx"].astype(bool),
        "psi": ref["p0"],
        "maxit_nb": 10,
        "backend": "cpu",
        "verbose": False,
    }
    a = fit_nb(ref["Y"], ref["Wg"], gpu_mem_budget=1, **kw)
    b = fit_nb(ref["Y"], ref["Wg"], gpu_mem_budget=-5, **kw)
    np.testing.assert_array_equal(a["alpha"], b["alpha"])


def test_calculate_mu_matches_r(ref):
    for key, W, gm_al, kw in [
        ("spanorm", ref["W"], ref["spanorm"], {}),
        ("offset", ref["Wg"], ref["offset"], {"offset": ref["O"]}),
    ]:
        mu = calculate_mu(gm_al["gmean"], gm_al["alpha"], W, **kw)
        np.testing.assert_allclose(mu, gm_al["mu"], rtol=1e-10, atol=1e-12)
        mu = calculate_mu(gm_al["gmean"], gm_al["alpha"], W, winsor=np.inf, **kw)
        np.testing.assert_allclose(mu, gm_al["mu_inf"], rtol=1e-10, atol=1e-12)
    mu = calculate_mu(ref["spanorm"]["gmean"], ref["spanorm"]["alpha"], ref["W"], winsor=2)
    np.testing.assert_allclose(mu, ref["spanorm"]["mu_w2"], rtol=1e-10, atol=1e-12)


def test_edger_failure_gives_nan_then_r_fallback(ref, monkeypatch, inject_psi_idx):
    """R: tryCatch maps an edgeR error to NA, so psi stays at its initial 0 (Poisson)."""

    def boom(*a, **k):
        raise EdgeRError("boom")

    monkeypatch.setattr(_fitnb, "estimate_disp_tagwise", boom)
    got = fit_spanorm_nb(
        ref["Y"],
        ref["W"],
        ref["idx"].astype(bool),
        maxit_psi=2,
        maxit_nb=10,
        is_spanorm=True,
        lambda_a=ref["lambda_a"],
        backend="cpu",
        msgfun=lambda m: None,
    )
    np.testing.assert_array_equal(got["psi"], 0.0)
    assert np.all(np.isfinite(got["alpha"]))
    if not ref["edger_fail_is_error"]:
        _check_fit(got, ref["edger_fail"])


def test_sample_psi_idx_draws_subset():
    idx = np.zeros(50, dtype=bool)
    idx[::2] = True
    m = _fitnb._sample_psi_idx(idx, 10, np.random.default_rng(0))
    assert m.dtype == bool and m.size == 25 and m.sum() == 10
    assert _fitnb._sample_psi_idx(idx, 100, np.random.default_rng(0)).all()


# ---- testthat error / warning cases ----------------------------------------
def test_fit_spanorm_nb_param_checks():
    emat = np.random.default_rng(36).poisson(10, (6, 4000))
    with pytest.raises(ValueError, match="greater"):
        fit_spanorm_nb(emat, None, None, maxit_psi=0)
    with pytest.raises(ValueError, match="greater"):
        fit_spanorm_nb(emat, None, None, maxit_psi=-1)
    with pytest.raises(ValueError, match="integer"):
        fit_spanorm_nb(emat, None, None, maxit_psi=1.1)


def test_fit_nb_given_psi_param_checks():
    Y = np.zeros((6, 10))
    W = np.zeros((10, 3))
    kw = {"lambda_a": 0.1, "gmean": np.zeros(6), "alpha": np.zeros((6, 3))}
    psi = np.zeros(6)
    for v, m in [(0, "greater"), (-1, "greater"), (1.1, "integer")]:
        with pytest.raises(ValueError, match=m):
            fit_nb_given_psi(Y, W, psi, maxit_nb=v, **kw)
    for v in [0, 1, 1.1, -1.1]:
        with pytest.raises(ValueError, match=r"0,1"):
            fit_nb_given_psi(Y, W, psi, step_factor=v, **kw)
    with pytest.raises(ValueError, match="positive"):
        fit_nb_given_psi(Y, W, psi, lambda_a=-1)
    with pytest.raises(ValueError, match="length equal"):
        fit_nb_given_psi(Y, W, psi, lambda_a=[1, 2])


def _toy(ng=8, ns=30, seed=110):
    rng = np.random.default_rng(seed)
    W = np.column_stack([np.ones(ns), (np.arange(ns) - ns / 2) / ns])
    Y = rng.poisson(5, (ng, ns))
    return Y, W


def test_malformed_offset_rejected():
    Y, W = _toy()
    ng, ns = Y.shape
    for off, msg in [
        (np.zeros((ng, ns - 1)), "genes x cells"),
        (np.zeros((ng + 1, ns)), "genes x cells"),
        (np.zeros(ns), "genes x cells"),
        (np.full((ng, ns), np.inf), "finite"),
        (np.full((ng, ns), np.nan), "finite"),
    ]:
        with pytest.raises(ValueError, match=msg):
            fit_nb(Y, W, offset=off, backend="cpu", verbose=False)
    with pytest.raises(ValueError, match="genes x cells"):
        calculate_mu(np.zeros(ng), np.zeros((ng, 2)), W, offset=np.zeros((2, 2)))


def test_malformed_psi_rejected():
    Y, W = _toy()
    ng = Y.shape[0]
    p0 = np.linspace(0.05, 0.6, ng)
    for psi, msg in [
        (p0[1:], "length 8"),
        (np.array(list("abcdefgh")), "length 8"),
        (np.zeros(ng), "strictly positive"),
        (-p0, "strictly positive"),
        (np.r_[np.nan, p0[1:]], "strictly positive"),
        (np.r_[np.inf, p0[1:]], "strictly positive"),
    ]:
        with pytest.raises(ValueError, match=msg):
            fit_nb(Y, W, psi=psi, backend="cpu", verbose=False)


def test_winsor_validated():
    _, W = _toy()
    for w in [0, -1, np.nan]:
        with pytest.raises(ValueError, match="positive"):
            calculate_mu(np.zeros(8), np.zeros((8, 2)), W, winsor=w)


def test_fit_nb_extra_params_forwarded():
    rng = np.random.default_rng(12)
    W = np.column_stack([np.ones(40), np.linspace(-1.5, 1.5, 40)])
    Y = rng.poisson(6, (10, 40))
    fit_nb(Y, W, maxn_psi=20, backend="cpu", verbose=False)
    fit_nb(Y, W, step_factor=0.3, backend="cpu", verbose=False)
    fit_nb(Y, W, gpu_mem_budget=1e9, backend="cpu", verbose=False)
    fit_nb(Y, W, gpu_mem_budget=1, backend="cpu", verbose=False)


# ---- ported testthat property tests (offset / psi) --------------------------
def _scale(v):
    return (v - v.mean()) / v.std(ddof=1)


def _nbdata(seed, ns):
    rng = np.random.default_rng(seed)
    return rng, _scale(rng.normal(size=ns)), _scale(rng.normal(size=ns))


def test_offset_equals_pinned_column():
    rng, z, x = _nbdata(102, 80)
    ng, ns, cc = 12, 80, 0.5
    WB = np.column_stack([z, np.ones(ns), x])
    WA = np.column_stack([(1 + cc) * z, np.ones(ns), x])
    a_true = np.column_stack(
        [rng.normal(0.3, 0.15, ng), rng.normal(1.2, 0.3, ng), rng.normal(0, 0.5, ng)]
    )
    mu = np.exp(a_true @ WB.T)
    Y = rng.negative_binomial(5, 5 / (5 + mu))
    O = np.tile(cc * z, (ng, 1))
    kw = {"maxit_psi": 4, "backend": "cpu", "verbose": False}
    fA = fit_nb(Y, WA, rng=np.random.default_rng(23), **kw)
    fB = fit_nb(Y, WB, offset=O, rng=np.random.default_rng(23), **kw)
    assert np.abs(fA["alpha"] - np.array([1, 0, 0])).max() > 0.5
    assert np.abs(fA["alpha"][:, 0] * (1 + cc) - (fB["alpha"][:, 0] + cc)).max() < 1e-10
    assert np.abs(fA["alpha"][:, 1:] - fB["alpha"][:, 1:]).max() < 1e-10
    assert np.abs(fA["psi"] - fB["psi"]).max() < 1e-10
    np.testing.assert_allclose(fA["loglik"], fB["loglik"], rtol=1e-10)
    muA = calculate_mu(fA["gmean"], fA["alpha"], WA, winsor=np.inf)
    muB = calculate_mu(fB["gmean"], fB["alpha"], WB, winsor=np.inf, offset=O)
    assert (np.abs(muA - muB) / muA).max() < 1e-10


def test_offset_subset_by_idx_in_lockstep():
    rng = np.random.default_rng(105)
    ng, ns = 12, 80
    W = np.column_stack([np.ones(ns), rng.normal(size=ns)])
    Y = rng.poisson(8, (ng, ns))
    O = np.outer(np.arange(1, ng + 1) / ng, np.cos(np.arange(1, ns + 1) / 5))
    idx = np.resize([True, True, False, True], ns)
    kw = {
        "maxit_psi": 3,
        "lambda_a": 0.0,
        "is_spanorm": False,
        "backend": "cpu",
        "msgfun": lambda m: None,
    }
    u1 = fit_spanorm_nb(Y, W, idx, offset=O, rng=np.random.default_rng(31), **kw)
    u2 = fit_spanorm_nb(
        Y[:, idx],
        W[idx],
        np.ones(idx.sum(), bool),
        offset=O[:, idx],
        rng=np.random.default_rng(31),
        **kw,
    )
    assert np.abs(u1["alpha"] - u2["alpha"]).max() < 1e-10
    assert np.abs(u1["psi"] - u2["psi"]).max() < 1e-10
    u3 = fit_spanorm_nb(
        Y[:, idx],
        W[idx],
        np.ones(idx.sum(), bool),
        offset=O[:, : idx.sum()],
        rng=np.random.default_rng(31),
        **kw,
    )
    assert np.abs(u3["alpha"] - u1["alpha"]).max() > 1e-3


def test_psi_used_as_is_and_loglik_identity():
    from spanormpy._nbdist import dnbinom

    rng = np.random.default_rng(202)
    ng, ns = 15, 120
    W = np.column_stack([np.ones(ns), rng.normal(size=ns)])
    a = np.column_stack([rng.normal(1.2, 0.3, ng), rng.normal(0, 0.5, ng)])
    mu = np.exp(a @ W.T)
    Y = rng.negative_binomial(5, 5 / (5 + mu))
    p0 = rng.uniform(0.05, 0.6, ng)
    fp = fit_nb(Y, W, psi=p0, backend="cpu", verbose=False)
    fp2 = fit_nb(Y, W, psi=2 * p0, backend="cpu", verbose=False)
    np.testing.assert_array_equal(fp["psi"], p0)
    assert np.ndim(fp["loglik"]) == 0
    ll = dnbinom(Y, mu=np.exp(fp["alpha"] @ W.T), size=1 / p0[:, None], log=True).sum().item()
    assert abs(fp["loglik"] - ll) < 1e-6
    assert np.abs(fp["alpha"] - fp2["alpha"]).max() > 1e-3
    # psi = None is a strict no-op
    f0 = fit_nb(Y, W, maxit_psi=3, backend="cpu", verbose=False, rng=np.random.default_rng(9))
    fn = fit_nb(
        Y, W, maxit_psi=3, psi=None, backend="cpu", verbose=False, rng=np.random.default_rng(9)
    )
    np.testing.assert_array_equal(f0["alpha"], fn["alpha"])
    np.testing.assert_array_equal(f0["psi"], fn["psi"])


def test_psi_with_idx_matches_presubset():
    rng = np.random.default_rng(205)
    ng, ns = 12, 90
    W = np.column_stack([np.ones(ns), rng.normal(size=ns)])
    Y = rng.poisson(8, (ng, ns))
    p0 = rng.uniform(0.05, 0.6, ng)
    idx = np.resize([True, True, False], ns)
    u1 = fit_nb(Y, W, idx=idx, psi=p0, backend="cpu", verbose=False)
    u2 = fit_nb(Y[:, idx], W[idx], psi=p0, backend="cpu", verbose=False)
    np.testing.assert_array_equal(u1["alpha"], u2["alpha"])
    assert (u1["sampling"] == "all").sum() == (~idx).sum()
    assert (u1["sampling"] == "glm").sum() == idx.sum()


def test_fit_nb_recovers_generic_glm():
    rng = np.random.default_rng(10)
    ng, ns = 40, 250
    W = np.column_stack([np.ones(ns), rng.normal(size=ns), rng.normal(size=ns)])
    a = np.column_stack([rng.normal(1, 0.4, ng), rng.normal(0, 0.7, ng), rng.normal(0, 0.7, ng)])
    mu = np.exp(a @ W.T)
    Y = rng.negative_binomial(5, 5 / (5 + mu))
    fit = fit_nb(Y, W, backend="cpu", verbose=False)
    assert np.all(fit["gmean"] == 0)
    assert fit["alpha"].shape == (ng, 3)
    assert np.corrcoef(fit["alpha"].ravel(), a.ravel())[0, 1] > 0.9


# ---- convergence / step-halving references ----------------------------------
def _numfmt(m):
    return re.sub(r"-?\d+\.\d+", "#", m)


def _nums(m):
    return [float(x) for x in re.findall(r"-?\d+\.\d+", m)]


def _assert_msgs(got, exp):
    exp = [str(m) for m in np.atleast_1d(exp)]
    assert [_numfmt(m) for m in got] == [_numfmt(m) for m in exp]
    for g, e in zip(got, exp, strict=True):
        np.testing.assert_allclose(_nums(g), _nums(e), rtol=1e-6)


def test_converged_generic_matches_r(ref, device):
    exp = ref["conv_generic"]
    msgs = []
    got = fit_spanorm_nb(
        exp["Y"],
        exp["W"],
        np.ones(exp["Y"].shape[1], bool),
        is_spanorm=False,
        lambda_a=0.0,
        backend=_backend(device),
        msgfun=msgs.append,
    )
    _check_fit(got, exp)
    _assert_msgs(msgs, exp["msgs"])
    assert any(m.endswith("(converged)") for m in msgs)


def test_converged_spanorm_matches_r(ref, device):
    exp = ref["conv_spanorm"]
    msgs = []
    got = fit_spanorm_nb(
        exp["Y"],
        exp["W"],
        np.ones(exp["Y"].shape[1], bool),
        is_spanorm=True,
        lambda_a=exp["lambda_a"],
        backend=_backend(device),
        msgfun=msgs.append,
    )
    _check_fit(got, exp)
    _assert_msgs(msgs, exp["msgs"])
    assert msgs[-1].endswith("(converged)")
    assert any("(did not converged)" in m for m in msgs)  # first inner fit hits maxit_nb


@pytest.mark.parametrize("sd", [1, 2])
@pytest.mark.parametrize("sp", [0, 1])
def test_step_halving_matches_r(ref, device, sd, sp):
    exp = ref[f"halv_s{sd}_sp{sp}"]
    c = ref["conv_generic"]
    msgs = []
    got = fit_nb_given_psi(
        c["Y"],
        c["W"],
        ref["halv_psi"],
        lambda_a=[0.1, 0.1] if sp else 0.1,
        gmean=np.zeros(c["Y"].shape[0]),
        alpha=exp["A0"],
        is_spanorm=bool(sp),
        winsor=np.inf,
        backend=_backend(device),
        msgfun=msgs.append,
    )
    np.testing.assert_allclose(got["alpha"], exp["alpha"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got["gmean"], exp["gmean"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got["loglik"], exp["loglik"], rtol=RTOL)
    np.testing.assert_allclose(got["loglik_iter"], np.atleast_1d(exp["loglik_iter"]), rtol=RTOL)
    _assert_msgs(msgs, exp["msgs"])


def test_blocked_offs_psi_with_offset(ref):
    import torch

    idx = ref["idx"].astype(bool)
    alpha = torch.as_tensor(ref["offset"]["alpha"])
    Wp = torch.as_tensor(ref["Wg"][idx][:200])
    off = ref["O"][:, idx][:, :200]
    exp = ref["offset"]["alpha"] @ ref["Wg"][idx][:200].T + off
    blocks = gene_block_indices(alpha.shape[0], 3)
    assert len(blocks) == 3
    got = _fitnb._blocked_offs_psi(alpha, Wp, blocks, off)
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)


def test_budget_forced_blocks_equal_single_block(ref, monkeypatch):
    """Force the budget-driven route (as on an accelerator) on the CPU device."""
    import spanormpy._backend as be

    idx = ref["idx"].astype(bool)
    kw = {
        "maxit_psi": 2,
        "maxit_nb": 5,
        "lambda_a": 0.5,
        "offset": ref["O"],
        "backend": "gpu",
        "msgfun": lambda m: None,
    }
    single = fit_spanorm_nb(ref["Y"], ref["Wg"], idx, rng=np.random.default_rng(1), **kw)

    seen = []
    real = be.gene_block_count

    def counting(*a, **k):
        n = real(*a, **k)
        seen.append(n)
        return n

    monkeypatch.setattr(_fitnb, "check_gpu", lambda: True)
    monkeypatch.setattr(_fitnb, "get_gpu_memory_budget", lambda b=None: 1000.0)
    monkeypatch.setattr(be, "check_gpu", lambda: True)
    monkeypatch.setattr(be, "gpu_dtype_bytes", lambda: 8)
    monkeypatch.setattr(_fitnb, "gene_block_count", counting)
    blocked = fit_spanorm_nb(ref["Y"], ref["Wg"], idx, rng=np.random.default_rng(1), **kw)
    assert (
        seen
        and max(seen) == ref["Y"].shape[0]
        and len(gene_block_indices(ref["Y"].shape[0], max(seen))) > 1
    )
    # Blocked and unblocked offsets for edgeR differ by ~1e-16 (matmul summation order); edgeR's
    # dispersion optimiser is flat near the optimum and amplifies that to ~4e-9 relative in psi
    # for low-dispersion genes, so the outer-loop quantities are compared at 1e-7, not 1e-10.
    for k in ("alpha", "gmean", "psi"):
        np.testing.assert_allclose(blocked[k], single[k], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(blocked["loglik"], single["loglik"], rtol=1e-9)


def test_tol_reaches_only_outer_loop(ref):
    c = ref["conv_generic"]
    idx = np.ones(c["Y"].shape[1], bool)
    out = {}
    for tol in (1e-4, 1e9):
        msgs = []
        fit_spanorm_nb(
            c["Y"],
            c["W"],
            idx,
            lambda_a=0.0,
            tol=tol,
            maxit_psi=25,
            rng=np.random.default_rng(7),
            backend="cpu",
            msgfun=msgs.append,
        )
        out[tol] = msgs
    outer = lambda ms: [m for m in ms if "estimating gene-wise" in m]
    assert len(outer(out[1e9])) == 2
    assert len(outer(out[1e-4])) > 2
    # inner loop is unaffected: its first inner fit is identical under both tol values
    first = lambda ms: [m for m in ms if m.startswith("iter:  1, iter:")]
    assert first(out[1e9]) == first(out[1e-4])


def test_edger_any_error_is_caught(ref, monkeypatch, inject_psi_idx):
    def boom(*a, **k):
        raise RuntimeError("anything")

    monkeypatch.setattr(_fitnb, "estimate_disp_tagwise", boom)
    got = fit_spanorm_nb(
        ref["Y"],
        ref["W"],
        ref["idx"].astype(bool),
        maxit_psi=1,
        maxit_nb=2,
        is_spanorm=True,
        lambda_a=ref["lambda_a"],
        backend="cpu",
        msgfun=lambda m: None,
    )
    np.testing.assert_array_equal(got["psi"], 0.0)
