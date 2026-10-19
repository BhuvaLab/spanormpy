import numpy as np
import pytest
import torch
from tests._ref import load_ref

from spanormpy._backend import as_tensor
from spanormpy._solver import (
    _absorb_blocks,
    _chol_batch,
    _chol_solve_batch,
    _segment_sum,
    _subset_state,
    nb_absorb_gram_batch,
    nb_gram_batch,
    nb_newton_solver,
    nb_newton_solver_batch,
)

RTOL, ATOL = 1e-10, 1e-12


@pytest.fixture(scope="module")
def ref():
    return load_ref("solver")


def T(x, device="cpu"):
    return as_tensor(x, device)


def N(t):
    return t.detach().cpu().numpy()


def close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(N(a) if isinstance(a, torch.Tensor) else a, b, rtol=rtol, atol=atol)


# ---- agreement with R ------------------------------------------------------


def test_dense_solver_matches_r(ref, device):
    W, pen, wt0, Sc = T(ref["W"], device), ref["pen"], ref["wt0"], ref["Sc"]
    sol = nb_newton_solver(W, pen)
    for g in range(wt0.shape[0]):
        step, cov = sol.solve(wt0[g], Sc[g]), sol.xcov(wt0[g])
        if ref["dense_step_null"][g]:
            assert step is None and cov is None and ref["dense_xcov_null"][g]
        else:
            close(step, ref["dense_step"][g])
            close(cov, ref["dense_xcov"][g])
    assert list(ref["dense_step_null"]) == [0, 1, 0, 0, 0]
    for g in range(wt0.shape[0]):
        close(sol.factor(ref["wt"][g]).info, ref["dense_info"][g])
    # a state from factor() is reusable by solve() and xcov()
    st = sol.factor(ref["wt"][0])
    close(sol.solve(st, Sc[0]), ref["dense_step_state"])
    close(sol.xcov(st), ref["dense_xcov_state"])


def test_nested_solver_matches_r(ref, device):
    W, pen, nested = T(ref["W"], device), ref["pen"], ref["nested"].astype(bool)
    wt, Sc = ref["wt"], ref["Sc"]
    sol = nb_newton_solver(W, pen, nested)
    for g in range(wt.shape[0]):
        close(sol.solve(wt[g], Sc[g]), ref["nested_step"][g])
        close(sol.xcov(wt[g]), ref["nested_xcov"][g])
        st = sol.factor(wt[g])
        close(st.S, ref["nested_S"][g])
        close(st.B, ref["nested_B"][g])
        close(st.cvec, ref["nested_cvec"][g])
    # an integer grouping of 1x1 blocks goes through the blocked path
    ids = np.array([None] * 5 + [0, 1, 2], dtype=object)
    close(nb_newton_solver(W, pen, ids).solve(wt[2], Sc[2]), ref["nested_step_blk"])


def test_blocked_solver_matches_r(ref):
    b = ref["blocked"]
    group = np.where(b["group"] == 0, None, b["group"]).astype(object)
    sol = nb_newton_solver(T(b["W"]), b["pen"], group)
    close(sol.solve(b["w"], b["s"]), b["step"])
    close(sol.xcov(b["w"]), b["xcov"])
    st = sol.factor(b["w"])
    close(st.S, b["S"])
    close(st.B, b["B"])


def test_batch_solver_matches_r(ref, device):
    W, pen, nested = T(ref["W"], device), ref["pen"], ref["nested"].astype(bool)
    wt, wt0, Sc = T(ref["wt"], device), T(ref["wt0"], device), T(ref["Sc"], device)
    sb = nb_newton_solver_batch(W, pen, nested)
    st = sb.factor(wt)
    assert st.ok.all()
    close(sb.solve(st, Sc), ref["batch_step"])
    close(sb.xcov(st), ref["batch_xcov"])
    close(st.S, ref["batch_S"])
    close(st.B, ref["batch_B"])
    close(st.cvec, ref["batch_cvec"])
    # a zero-weight gene is flagged, NaN, and does not poison its neighbours
    st0 = sb.factor(wt0)
    np.testing.assert_array_equal(st0.ok, ref["batch0_ok"].astype(bool))
    for got, want in ((sb.solve(st0, Sc), ref["batch0_step"]), (sb.xcov(st0), ref["batch0_xcov"])):
        got = N(got)
        assert np.isnan(got[1]).all() and np.isnan(want[1]).all()
        np.testing.assert_allclose(
            np.delete(got, 1, 0), np.delete(want, 1, 0), rtol=RTOL, atol=ATOL
        )
    # no nested block
    sbd = nb_newton_solver_batch(W, pen)
    std = sbd.factor(wt)
    close(sbd.solve(std, Sc), ref["batchd_step"])
    close(sbd.xcov(std), ref["batchd_xcov"])
    std0 = sbd.factor(wt0)
    np.testing.assert_array_equal(std0.ok, ref["batchd0_ok"].astype(bool))
    got, want = N(sbd.solve(std0, Sc)), ref["batchd0_step"]
    assert np.isnan(got[1]).all() and np.isnan(want[1]).all()
    np.testing.assert_allclose(np.delete(got, 1, 0), np.delete(want, 1, 0), rtol=RTOL, atol=ATOL)


def test_gram_batch_matches_r(ref, device):
    W, wt, pen = T(ref["W"], device), T(ref["wt"], device), ref["pen"]
    close(nb_gram_batch(W, wt, pen), ref["gram_pen"])
    close(nb_gram_batch(W, wt), ref["gram_nopen"])
    close(nb_gram_batch(W, wt, pen, cell_tile=7), ref["gram_tile"])
    # each slice is exactly symmetric (the Cholesky relies on it)
    G = N(nb_gram_batch(W, wt, pen))
    np.testing.assert_array_equal(G, G.transpose(0, 2, 1))


def test_absorb_gram_batch_matches_r(ref, device):
    W, wt, pen = T(ref["W"], device), T(ref["wt"], device), ref["pen"]
    nested = ref["nested"].astype(bool)
    close(nb_absorb_gram_batch(W, pen, nested, wt), ref["absorb_S"])
    close(nb_absorb_gram_batch(W, pen, nested, wt, cell_tile=7), ref["absorb_tile"])
    pp = nb_absorb_gram_batch(W, pen, nested, wt, parts=True)
    assert set(pp) == {"S", "B", "cvec", "xi", "zi"}
    close(pp["S"], ref["absorb_parts_S"])
    close(pp["B"], ref["absorb_parts_B"])
    close(pp["cvec"], ref["absorb_parts_cvec"])
    np.testing.assert_array_equal(pp["xi"], ref["absorb_parts_xi"] - 1)
    np.testing.assert_array_equal(pp["zi"], ref["absorb_parts_zi"] - 1)
    # a group holding no cells keeps its slot
    pe = nb_absorb_gram_batch(T(ref["We"], device), pen, nested, wt, parts=True)
    close(pe["S"], ref["absorb_empty_S"])
    close(pe["cvec"], ref["absorb_empty_cvec"])
    close(pe["B"], ref["absorb_empty_B"])


# ---- ported from test-nbSolver.R (oracles are dense solves) ----------------


def _nested_fixture(n=80, px=4, G=5, b=4, seed=3, nested=True):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, px - 1))])
    wt = rng.uniform(0.2, 2, (b, n))
    if not nested:
        return {
            "W": X,
            "px": px,
            "G": 0,
            "nested": np.zeros(px, bool),
            "pen": np.zeros(px),
            "wt": wt,
        }
    grp = np.concatenate([np.arange(G), rng.integers(0, G, n - G)])
    Z = np.zeros((n, G))
    Z[np.arange(n), grp] = 1
    return {
        "W": np.hstack([X, Z]),
        "px": px,
        "G": G,
        "nested": np.r_[np.zeros(px, bool), np.ones(G, bool)],
        "pen": np.r_[np.zeros(px), np.full(G, 0.3)],
        "wt": wt,
    }


def _scores(f, seed=99):
    return np.random.default_rng(seed).standard_normal((f["wt"].shape[0], f["W"].shape[1]))


def _dense_solve(W, pen, w, s):
    info = (W * w[:, None]).T @ W + np.diag(pen)
    return np.linalg.solve(info, s)


def test_absorb_batch_returns_parts_matching_per_gene_solver():
    f = _nested_fixture()
    got = nb_absorb_gram_batch(T(f["W"]), f["pen"], f["nested"], T(f["wt"]), parts=True)
    b = f["wt"].shape[0]
    assert tuple(got["S"].shape) == (b, f["px"], f["px"])
    assert tuple(got["B"].shape) == (b, f["G"], f["px"])
    assert tuple(got["cvec"].shape) == (b, f["G"])
    sol = nb_newton_solver(T(f["W"]), f["pen"], f["nested"])
    for g in range(b):
        ref = sol.factor(f["wt"][g])
        close(got["S"][g], N(ref.S), rtol=1e-10)
        close(got["B"][g].T, N(ref.B), rtol=1e-10)
        close(got["cvec"][g], N(ref.cvec), rtol=1e-10)
    close(nb_absorb_gram_batch(T(f["W"]), f["pen"], f["nested"], T(f["wt"])), N(got["S"]))


@pytest.mark.parametrize("nested", [True, False])
def test_batch_solver_matches_per_gene_solver(nested):
    f = _nested_fixture(nested=nested)
    Sc = _scores(f)
    sb = nb_newton_solver_batch(T(f["W"]), f["pen"], f["nested"])
    st = sb.factor(T(f["wt"]))
    assert st.ok.all()
    D = N(sb.solve(st, T(Sc)))
    V = N(sb.xcov(st))
    sol = nb_newton_solver(T(f["W"]), f["pen"], f["nested"])
    for g in range(f["wt"].shape[0]):
        np.testing.assert_allclose(D[g], N(sol.solve(f["wt"][g], Sc[g])), rtol=1e-9)
        np.testing.assert_allclose(V[g], N(sol.xcov(f["wt"][g])), rtol=1e-9, atol=1e-12)


def test_singular_gene_does_not_poison_batch():
    f = _nested_fixture()
    f["wt"][1] = 0
    Sc = _scores(f)
    sb = nb_newton_solver_batch(T(f["W"]), f["pen"], f["nested"])
    st = sb.factor(T(f["wt"]))
    assert not st.ok[1] and st.ok[[0, 2, 3]].all()
    D = N(sb.solve(st, T(Sc)))
    assert np.isnan(D[1]).all()
    sol = nb_newton_solver(T(f["W"]), f["pen"], f["nested"])
    for g in (0, 2, 3):
        np.testing.assert_allclose(D[g], N(sol.solve(f["wt"][g], Sc[g])), rtol=1e-9)
    # a subset of the state keeps the verdicts of its genes
    sub = _subset_state(st, [1, 2])
    np.testing.assert_array_equal(sub.ok, [False, True])
    assert sub.S.shape[0] == 2 and sub.B.shape[0] == 2


def _slope_fixture(n=120, px=3, S=4, K=2, seed=11):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, px - 1))])
    smp = np.concatenate([np.arange(S), rng.integers(0, S, n - S)])
    Zint = np.eye(S)[smp]
    bases = rng.standard_normal((n, K))
    Z = np.hstack([Zint] + [Zint * bases[:, [k]] for k in range(K)])
    blk = np.concatenate([np.arange(S), np.tile(np.arange(S), K)])
    return {
        "W": np.hstack([X, Z]),
        "px": px,
        "group": np.array([None] * px + list(blk), dtype=object),
        "pen": np.r_[np.zeros(px), np.full(Z.shape[1], 0.4)],
        "w": rng.uniform(0.2, 2, n),
        "score": rng.standard_normal(px + Z.shape[1]),
    }


def test_blocked_solver_absorbs_per_sample_block_exactly():
    f = _slope_fixture()
    zi = np.array([g is not None for g in f["group"]])
    Zw = f["W"][:, zi] * np.sqrt(f["w"])[:, None]
    C = Zw.T @ Zw
    assert np.abs(C[np.triu_indices_from(C, 1)]).max() > 1e-6  # not the diagonal case
    sol = nb_newton_solver(T(f["W"]), f["pen"], f["group"])
    np.testing.assert_allclose(
        N(sol.solve(f["w"], f["score"])),
        _dense_solve(f["W"], f["pen"], f["w"], f["score"]),
        rtol=1e-8,
    )
    info = (f["W"] * f["w"][:, None]).T @ f["W"] + np.diag(f["pen"])
    np.testing.assert_allclose(
        N(sol.xcov(f["w"])), np.linalg.inv(info)[: f["px"], : f["px"]], rtol=1e-8, atol=1e-12
    )


def test_logical_nested_absorbs_indicator_block_exactly():
    f = _nested_fixture(n=90, px=3, G=5, seed=5)
    rng = np.random.default_rng(1)
    w, s = rng.uniform(0.2, 2, 90), rng.standard_normal(8)
    sol = nb_newton_solver(T(f["W"]), f["pen"], f["nested"])
    np.testing.assert_allclose(N(sol.solve(w, s)), _dense_solve(f["W"], f["pen"], w, s), rtol=1e-8)


def test_mixed_indicator_and_slope_grouping_absorbs_exactly():
    rng = np.random.default_rng(7)
    n, px, S, K, nct = 150, 3, 3, 2, 2
    X = np.column_stack([np.ones(n), rng.standard_normal((n, px - 1))])
    smp = rng.integers(0, S, n)
    ct = rng.integers(0, nct, n)
    Zint = np.eye(S)[smp]
    bases = rng.standard_normal((n, K))
    Zsl = np.hstack([Zint * bases[:, [k]] for k in range(K)])
    combos = sorted(set(zip(smp, ct)))
    Zct = np.column_stack([(smp == s_) & (ct == c_) for s_, c_ in combos]).astype(float)
    W = np.hstack([X, Zint, Zsl, Zct])
    pen = np.r_[np.zeros(px), np.full(W.shape[1] - px, 0.5)]
    group = np.array(
        [None] * px + list(range(S)) + list(range(S)) * K + [s_ for s_, _ in combos], dtype=object
    )
    w, s = rng.uniform(0.2, 2, n), rng.standard_normal(W.shape[1])
    sol = nb_newton_solver(T(W), pen, group)
    np.testing.assert_allclose(N(sol.solve(w, s)), _dense_solve(W, pen, w, s), rtol=1e-8)


def test_batched_solver_refuses_multicolumn_block():
    f = _slope_fixture()
    with pytest.raises(ValueError, match="1x1|one column|block"):
        nb_newton_solver_batch(T(f["W"]), f["pen"], f["group"])


def test_batched_solver_accepts_logical_indicator():
    f = _nested_fixture(n=60, px=3, G=4, seed=5)
    nb_newton_solver_batch(T(f["W"]), f["pen"], f["nested"])


def test_blocked_solver_rejects_cells_loading_on_two_blocks():
    f = _slope_fixture()
    W = f["W"].copy()
    zi = np.nonzero([g is not None for g in f["group"]])[0]
    # cell 0 loads on sample 0's columns; also give it a column of sample 1
    W[0, zi[1]] = 1.0
    with pytest.raises(ValueError, match="not block-orthogonal"):
        nb_newton_solver(T(W), f["pen"], f["group"])


def test_dense_nested_rejects_non_partition():
    f = _nested_fixture()
    W = f["W"].copy()
    W[0, f["px"] + 1] = 1  # cell 0 is in group 0 by construction
    with pytest.raises(ValueError, match="partitioning"):
        nb_newton_solver(T(W), f["pen"], f["nested"])
    with pytest.raises(ValueError, match="partition"):
        nb_absorb_gram_batch(T(W), f["pen"], f["nested"], T(f["wt"]))


def test_absorb_spec_validation_and_ordering():
    with pytest.raises(ValueError, match="one entry per column"):
        _absorb_blocks([True, False], 3)
    np.testing.assert_array_equal(_absorb_blocks(None, 3), [-1, -1, -1])
    np.testing.assert_array_equal(_absorb_blocks(np.array([False, True, True]), 3), [-1, 1, 2])
    # ids are ranked like factor(as.character(.)): "10" sorts before "2"
    np.testing.assert_array_equal(
        _absorb_blocks(np.array([None, 10, 2], dtype=object), 3), [-1, 0, 1]
    )
    np.testing.assert_array_equal(_absorb_blocks(np.array([np.nan, 1.0, 1.0]), 3), [-1, 0, 0])
    # all-NA and all-FALSE select the dense path
    W = _nested_fixture(nested=False)
    for spec in (np.full(4, np.nan), np.zeros(4, bool)):
        sol = nb_newton_solver(T(W["W"]), W["pen"] + 0.1, spec)
        assert hasattr(sol.factor(W["wt"][0]), "info")


# ---- ported from test-nbArrayBatch.R (the helpers that live in the solver) ---


def test_segment_sum_matches_rowsum_and_keeps_empty_groups():
    rng = np.random.default_rng(7)
    n, k, G = 200, 5, 9
    M = rng.standard_normal((n, k))
    gidx = rng.integers(0, G, n)
    want = np.zeros((G, k))
    np.add.at(want, gidx, M)
    close(_segment_sum(T(M), torch.as_tensor(gidx), G), want, rtol=1e-12)
    g2 = gidx.copy()
    g2[g2 == 3] = 4
    got = N(_segment_sum(T(M), torch.as_tensor(g2), G))
    assert got.shape == (G, k) and (got[3] == 0).all()


def test_chol_batch_flags_each_slice():
    S = np.stack([np.eye(3), -np.eye(3), 2 * np.eye(3)])
    L, ok = _chol_batch(T(S))
    np.testing.assert_array_equal(ok, [True, False, True])
    out = N(_chol_solve_batch(L, ok, T(np.ones((3, 3)))))
    assert np.isnan(out[1]).all() and not np.isnan(out[[0, 2]]).any()
    np.testing.assert_allclose(out[2], 0.5)


# ---- block-design guards and tile invariance --------------------------------


def test_gram_and_absorb_are_invariant_to_cell_tile():
    f = _nested_fixture(n=60, px=4, G=5, b=3, seed=7)
    whole = N(nb_absorb_gram_batch(T(f["W"]), f["pen"], f["nested"], T(f["wt"])))
    for tile in (1, 7, 59, 60, 1000):
        close(
            nb_absorb_gram_batch(T(f["W"]), f["pen"], f["nested"], T(f["wt"]), cell_tile=tile),
            whole,
            1e-12,
        )
        close(
            nb_gram_batch(T(f["W"]), T(f["wt"]), cell_tile=tile),
            N(nb_gram_batch(T(f["W"]), T(f["wt"]))),
            1e-12,
        )


def test_absorb_batch_keeps_group_holding_no_cells():
    rng = np.random.default_rng(7)
    n, px, G, b = 60, 4, 5, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, px - 1))])
    grp = rng.choice([0, 2, 3, 4], n)
    Z = np.zeros((n, G))
    Z[np.arange(n), grp] = 1
    W, nested = np.hstack([X, Z]), np.r_[np.zeros(px, bool), np.ones(G, bool)]
    pen = np.r_[np.zeros(px), np.full(G, 0.3)]
    wt = rng.uniform(0.2, 2, (b, n))
    want = np.zeros((b, px, px))
    for g in range(b):
        Cd = (Z * wt[g][:, None]).sum(0) + 0.3
        B = (X * wt[g][:, None]).T @ Z
        want[g] = (X * wt[g][:, None]).T @ X - B @ (B.T / Cd[:, None])
    close(nb_absorb_gram_batch(T(W), pen, nested, T(wt)), want, 1e-10)
