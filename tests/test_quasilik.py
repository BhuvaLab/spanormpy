"""Tests for quasi-likelihood dispersion (quasilik.R) against R references."""

import numpy as np
import pytest
import torch
from tests._ref import load_ref

from spanormpy._quasilik import (
    _ql_interp,
    nb_deviance_moments,
    nb_unit_deviance,
    ql_dispersion,
    ql_moment_table,
)

RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture(scope="module")
def ref():
    return load_ref("quasilik")


def _tab(d):
    t = dict(d)
    for k in ("K", "J"):
        t[k] = int(t[k])
    for k in ("lo", "h", "plo", "ph"):
        t[k] = float(t[k])
    return t


def _close(got, exp, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=rtol, atol=atol)


def _cmp(got, exp, n=None, **kw):
    for k in ("deviance", "df", "s2"):
        _close(got[k], np.asarray(exp[k])[:n], **kw)



# --- nb_unit_deviance --------------------------------------------------------
def test_unit_deviance_vs_r(ref):
    u = ref["ud"]
    _close(nb_unit_deviance(u["y"], u["mu"], u["phi"]), u["dev"])
    _close(nb_unit_deviance(u["y"], u["mu"], 0.3), u["dev_scalar"])
    edge = nb_unit_deviance(
        np.array([[0, 0], [3, 2.0]]), np.array([[0, 1], [0, 1.5]]), [0.5, 2.0]
    )
    _close(edge, u["edge"])


def test_unit_deviance_matches_edger(ref):
    # edgeR's C unit deviance carries series approximations (observed max abs gap 3e-7);
    # R's testthat compares the mean relative difference at 1e-6, so elementwise this is looser
    u = ref["ud"]
    _close(nb_unit_deviance(u["y"], u["mu"], u["phi"]), u["edger"], rtol=1e-4, atol=1e-6)
    one = nb_unit_deviance(u["y"][1:2], u["mu"][1:2], 0.2)
    assert one.shape == (1, u["y"].shape[1])
    _close(one[0], nb_unit_deviance(u["y"], u["mu"], u["phi"])[1])


def test_unit_deviance_torch_equals_numpy(ref):
    u = ref["ud"]
    got = nb_unit_deviance(torch.as_tensor(u["y"]), torch.as_tensor(u["mu"]), u["phi"])
    assert isinstance(got, torch.Tensor)
    _close(got.numpy(), nb_unit_deviance(u["y"], u["mu"], u["phi"]))


# --- nb_deviance_moments -----------------------------------------------------
def test_moments_vs_r(ref):
    m = ref["mom"]
    got = nb_deviance_moments(m["mu"], m["phi"])
    _close(got["w0"], m["w0"])
    _close(got["w1"], m["w1"])
    tight = nb_deviance_moments(m["mu"], m["phi"], eps=1e-13)
    # eps = 1e-13 puts the window ends at 1 - 1e-13, where R's own CDF is quantised at
    # 1.1e-16 (a few counts, ~1e-18 of mass, in the upper end); the far tail carries
    # ~3e-10 of the variance, so only 1e-8 is attainable there
    _close(tight["w0"], m["w0_tight"], rtol=1e-8, atol=0)
    _close(tight["w1"], m["w1_tight"], rtol=1e-8, atol=0)
    # converged at the default window (testthat: 1e-4)
    _close(got["w0"], tight["w0"], rtol=1e-4, atol=0)
    _close(got["w1"], tight["w1"], rtol=1e-4, atol=0)


def test_moments_chisq_limit_and_sparse_point(ref):
    m = nb_deviance_moments(500, 1e-6)  # near-Poisson: unit deviance ~ chisq_1
    assert abs(m["w1"][0] - 1) < 0.02 and abs(m["w0"][0] - 1) < 0.02
    _close([m["w0"][0], m["w1"][0]], ref["mom"]["near_poisson"])
    s = nb_deviance_moments(0.1, 5)  # about half a df per observation
    assert 0.4 < s["w1"][0] < 0.6 and s["w0"][0] > 1.5
    _close([s["w0"][0], s["w1"][0]], ref["mom"]["sparse"])


def test_moments_invalid_entries_and_caps(ref):
    b = ref["mom_bad"]
    got = nb_deviance_moments(b["mu"], b["phi"])
    _close(got["w0"], b["w0"])
    _close(got["w1"], b["w1"])
    assert got["w0"][0] == 0 and got["w1"][1] == 0
    c = nb_deviance_moments([50, 300], [0.5, 20], maxterms=25)
    _close(c["w0"], ref["mom_cap"]["w0"])
    _close(c["w1"], ref["mom_cap"]["w1"])
    v = ref["mom_vec"]
    got = nb_deviance_moments(v["mu"], 2.5)
    _close(got["w0"], v["w0"])
    _close(got["w1"], v["w1"])
    empty = nb_deviance_moments([0.0, np.nan], 1.0)
    assert (empty["w0"] == 0).all() and (empty["w1"] == 0).all()


# --- ql_moment_table and interpolation ---------------------------------------
_TABS = {
    "default": dict(lmu_range=np.log([1e-6, 1e3]), lphi_range=np.log([0.1, 10])),
    "one_phi": dict(lmu_range=np.log([0.01, 50]), lphi_range=np.log([2.0, 2.0]), step_mu=0.3),
    "steps": dict(
        lmu_range=np.log([0.05, 20]), lphi_range=np.log([0.05, 8]), step_mu=0.5, step_phi=0.7
    ),
    "grid": dict(lmu_range=np.log([0.05, 20]), lphi_range=np.log([0.05, 8]), ngrid=17, nphi=5),
    "degenerate": dict(lmu_range=np.log([2.0, 2.0]), lphi_range=np.log([1, 1.5]), ngrid=4, nphi=3),
}


@pytest.mark.parametrize("name", list(_TABS))
def test_moment_table_and_interp_vs_r(ref, name):
    exp = _tab(ref[f"tab_{name}"])
    tab = ql_moment_table(**_TABS[name])
    for k in ("K", "J"):
        assert tab[k] == exp[k]
    for k in ("lo", "h", "plo", "ph"):
        _close(tab[k], exp[k])
    assert tab["w0"].shape == (exp["J"], exp["K"])
    _close(tab["w0"], exp["w0"])
    _close(tab["w1"], exp["w1"])
    ip = ref[f"interp_{name}"]
    got = _ql_interp(exp, ip["lmu"], ip["lphi"])  # R's table: isolates the lookup rule
    _close(got["w0"], ip["w0"])
    _close(got["w1"], ip["w1"])
    gt = _ql_interp(exp, torch.as_tensor(ip["lmu"]), ip["lphi"])
    _close(gt["w0"].numpy(), ip["w0"])
    _close(gt["w1"].numpy(), ip["w1"])


# --- ql_dispersion on a dlpfc_small fit --------------------------------------
@pytest.mark.parametrize("leverage", ["trace", "exact"])
@pytest.mark.parametrize("moments", ["table", "grid", "cell"])
def test_ql_dispersion_dlpfc_vs_r(ref, leverage, moments):
    q = ref["ql"]
    n = None
    got = ql_dispersion(
        q["Y"][:n], q["mu"][:n], q["phi"][:n], design=q["design"], leverage=leverage, moments=moments
    )
    _cmp(got, q[f"{leverage}_{moments}"], n=n)


def test_ql_dispersion_options_vs_r(ref):
    q = ref["ql"]
    Y, mu, phi, des = q["Y"], q["mu"], q["phi"], q["design"]
    p = des.shape[1]
    g = slice(None)
    _cmp(
        ql_dispersion(Y[g], mu[g], phi[g], design=des, prior=1.7, leverage="exact", moments="cell"),
        q["prior"],
        n=None,
    )
    _cmp(ql_dispersion(Y, mu, phi, design=des, prior=1.7), q["prior_table"])
    _cmp(
        ql_dispersion(Y[g], mu[g], phi[g], p=p, prior=1.7, moments="grid", ngrid=64),
        q["prior_grid"],
        n=None,
    )
    _cmp(ql_dispersion(Y, mu, phi, p=p, ngrid=32, nphi=8), q["small_grid"])
    _cmp(ql_dispersion(Y, mu, phi, p=p, table=_tab(q["full_table"])), q["with_table"])
    _cmp(ql_dispersion(Y, mu, 2.0, p=3), q["scalar_phi"])
    tiny = ql_dispersion(Y[:, :3], mu[:, :3], phi, p=3)  # n == p: early return of zeros
    _cmp(tiny, q["tiny_n"])
    assert (tiny["s2"] == 0).all()


def test_ql_dispersion_torch_dlpfc(ref):
    q = ref["ql"]
    cpu = ql_dispersion(q["Y"], q["mu"], q["phi"], p=q["design"].shape[1])
    dev = ql_dispersion(torch.as_tensor(q["Y"]), torch.as_tensor(q["mu"]), q["phi"], p=q["design"].shape[1])
    _cmp(dev, cpu, rtol=1e-8)
    _cmp(dev, q["trace_table"], rtol=1e-8)


# --- testthat cases ----------------------------------------------------------
def test_ql_matches_edger_glmqlfit(ref):
    e = ref["edger"]
    got = ql_dispersion(
        e["y"], e["mu"], 0.2, design=e["design"], prior=e["prior"], leverage="exact", moments="cell"
    )
    _cmp(got, e["got"])
    # the slack to edgeR is edgeR's Chebyshev fast path, not ours (testthat: 3e-3)
    _close(got["df"], e["df"], rtol=3e-3, atol=0)
    _close(got["deviance"], e["deviance"], rtol=3e-3, atol=0)
    _close(got["s2"], e["deviance"] / e["df"], rtol=3e-3, atol=0)


def test_ql_trace_tracks_exact_leverage(ref):
    t = ref["t12"]
    ex = ql_dispersion(t["y"], t["mu"], 0.3, design=t["design"], leverage="exact", moments="cell")
    ap = ql_dispersion(t["y"], t["mu"], 0.3, design=t["design"], leverage="trace", moments="cell")
    _cmp(ex, t["ex"])
    _cmp(ap, t["ap"])
    _close(ap["df"], ex["df"], rtol=1e-3, atol=0)
    _close(ap["s2"], ex["s2"], rtol=1e-3, atol=0)


def test_ql_shared_table_matches_cell_moments(ref):
    t = ref["t14"]
    ex = ql_dispersion(t["y"], t["mu"], t["phi"], design=t["design"], moments="cell")
    tb = ql_dispersion(t["y"], t["mu"], t["phi"], design=t["design"], moments="table")
    _cmp(ex, t["ex"])
    _cmp(tb, t["tb"])
    _close(tb["s2"], ex["s2"], rtol=2e-3, atol=0)
    _close(tb["df"], ex["df"], rtol=2e-3, atol=0)
    gr = ql_dispersion(t["y"], t["mu"], 5.0, design=t["design"], moments="grid")
    tb1 = ql_dispersion(t["y"], t["mu"], 5.0, design=t["design"], moments="table")
    _cmp(gr, t["gr"])
    _cmp(tb1, t["tb1"])
    _close(tb1["s2"], gr["s2"], rtol=2e-3, atol=0)


def test_ql_accepts_p_in_place_of_design():
    rng = np.random.default_rng(15)
    ng, n = 8, 300
    design = np.column_stack([np.ones(n), rng.normal(size=n)])
    mu = np.exp(rng.normal(0, 1, (ng, n)))
    y = rng.negative_binomial(0.5, 0.5 / (0.5 + mu)).astype(float)
    a = ql_dispersion(y, mu, 2.0, design=design)
    b = ql_dispersion(y, mu, 2.0, p=2)
    np.testing.assert_array_equal(a["s2"], b["s2"])
    np.testing.assert_array_equal(a["df"], b["df"])


def test_ql_argument_errors():
    y = np.ones((2, 20))
    with pytest.raises(ValueError, match="supply `design` or its column count `p`"):
        ql_dispersion(y, y, 1.0)
    with pytest.raises(ValueError, match="needs `design`"):
        ql_dispersion(y, y, 1.0, p=2, leverage="exact", moments="cell")
    with pytest.raises(ValueError, match="should be one of"):
        ql_dispersion(y, y, 1.0, p=2, leverage="nope")
    with pytest.raises(ValueError, match="should be one of"):
        ql_dispersion(y, y, 1.0, p=2, moments="nope")
    yt = torch.as_tensor(y)
    with pytest.raises(ValueError, match="supports leverage"):
        ql_dispersion(yt, yt, 1.0, p=2, moments="cell")
    with pytest.raises(ValueError, match="supports leverage"):
        ql_dispersion(yt, yt, 1.0, p=2, leverage="exact")


def test_ql_caller_table_makes_result_invariant_to_gene_split(ref):
    t = ref["t17"]
    y, mu, phi = t["y"], t["mu"], t["phi"]
    tab = ql_moment_table(np.log([1e-8, y.max() + 1]), np.log([phi.min(), phi.max()]))
    exp = _tab(t["tab"])
    for k in ("w0", "w1"):
        _close(tab[k], exp[k])
    whole = ql_dispersion(y, mu, phi, p=3, table=tab)
    _cmp(whole, t["whole"])
    parts = [ql_dispersion(y[i], mu[i], phi[i], p=3, table=tab) for i in (slice(0, 7), slice(7, 20))]
    np.testing.assert_array_equal(whole["s2"], np.concatenate([p["s2"] for p in parts]))
    np.testing.assert_array_equal(whole["df"], np.concatenate([p["df"] for p in parts]))
    ex = ql_dispersion(y, mu, phi, p=3, moments="cell")
    _cmp(ex, t["ex"])
    _close(whole["s2"], ex["s2"], rtol=2e-3, atol=0)
    dev = ql_dispersion(torch.as_tensor(y), torch.as_tensor(mu), phi, p=3, table=tab)
    _close(dev["s2"], whole["s2"], rtol=1e-8, atol=0)
