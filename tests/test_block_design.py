import numpy as np
import pandas as pd
import pytest
import torch
from tests._ref import load_ref

from spanormpy._backend import as_tensor
from spanormpy._block_design import (
    BLOCK_BATCH_DENSE_MIN,
    _block_col_cells,
    _block_dense_absorb,
    _block_eta_batch,
    _block_eta_vec,
    _block_expand,
    _block_score_batch,
    _block_score_vec,
    _design_col_nonzero,
    _design_cross,
    _design_eta,
    nb_block_design,
)
from spanormpy._solver import (
    nb_absorb_gram_batch,
    nb_gram_batch,
    nb_newton_solver,
    nb_newton_solver_batch,
)

RTOL, ATOL = 1e-10, 1e-12


@pytest.fixture(scope="module")
def ref():
    return load_ref("solver")


def N(t):
    return t.detach().cpu().numpy()


def close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(N(a) if isinstance(a, torch.Tensor) else a, b, rtol=rtol, atol=atol)


def build(r):
    return nb_block_design(
        pd.DataFrame(r["X"], columns=["n1", "n2", "n3"]),
        pd.DataFrame(r["Z"], columns=["int", "ls1"]),
        pd.Categorical(r["block"], categories=list(np.atleast_1d(r["levels"]))),
    )


CASES = ["blk_main", "blk_aliased", "blk_empty"]


@pytest.mark.parametrize("case", CASES)
def test_dense_expansion_and_helpers_match_r(ref, case):
    r = ref[case]
    D = build(r)
    assert D.shape == r["dense"].shape
    assert D.colnames == list(r["colnames"])
    np.testing.assert_allclose(D.to_dense(), r["dense"], rtol=0, atol=0)
    close(_block_eta_vec(D, r["a"]), r["eta_vec"], 1e-12)
    close(_block_score_vec(D, r["r"]), r["score_vec"], 1e-12)
    close(_block_eta_batch(D, r["A3"]), r["eta3"], 1e-12)
    close(_block_score_batch(r["R3"], D), r["score3"], 1e-12)
    # at or above BLOCK_BATCH_DENSE_MIN genes the group-by-group branch runs
    assert r["A8"].shape[0] >= BLOCK_BATCH_DENSE_MIN
    close(_block_eta_batch(D, r["A8"]), r["eta8"], 1e-12)
    close(_block_score_batch(r["R8"], D), r["score8"], 1e-12)
    # design-agnostic helpers: same answer from the compact and the dense design
    W = as_tensor(r["dense"], "cpu")
    close(_design_eta(D, r["a"]), N(_design_eta(W, r["a"])), 1e-12)
    close(_design_cross(D, r["r"]), N(_design_cross(W, r["r"])), 1e-12)
    for j in range(D.shape[1]):
        np.testing.assert_array_equal(
            N(_block_col_cells(D, j)), np.nonzero(r["dense"][:, j] != 0)[0], err_msg=str(j)
        )
        np.testing.assert_array_equal(N(_design_col_nonzero(D, j)), r["dense"][:, j] != 0)
        np.testing.assert_array_equal(N(_design_col_nonzero(W, j)), r["dense"][:, j] != 0)


@pytest.mark.parametrize("case", CASES)
def test_compact_solver_matches_r(ref, case):
    r = ref[case]
    D = build(r)
    sol = nb_newton_solver(D, r["pen_xz"])  # [X | Z] penalty is expanded to every group
    wts, Sc = np.atleast_2d(r["wts"]), np.atleast_2d(r["Sc"])
    for g in range(wts.shape[0]):
        st = sol.factor(wts[g])
        np.testing.assert_array_equal(st.rank, np.atleast_2d(r["rank"])[g])
        close(st.S, r["S"][g])
        close(st.B, r["B"][g])
        for k in range(D.G):
            close(st.H[k] @ st.H[k].T, r["Cinv"][g, k])
        close(sol.solve(st, Sc[g]), r["step"][g])
        close(sol.xcov(st), r["xcov"][g])
        # a weight vector is accepted wherever a state is
        close(sol.solve(wts[g], Sc[g]), r["step"][g])
        close(sol.xcov(wts[g]), r["xcov"][g])


def test_compact_solver_matches_dense_oracles(ref):
    r = ref["blk_main"]
    D = build(r)
    pen = _block_expand(D, r["pen_xz"], "pen")
    sc = nb_newton_solver(D, r["pen_xz"])
    W = r["dense"]
    w, s = r["wts"][0], r["Sc"][0]
    info = (W * w[:, None]).T @ W + np.diag(pen)
    close(sc.solve(w, s), np.linalg.solve(info, s), 1e-9)
    close(sc.xcov(w), np.linalg.inv(info)[:3, :3], 1e-9)
    # the dense grouped solver on the expanded design agrees
    group = np.where(_block_dense_absorb(D) < 0, None, _block_dense_absorb(D)).astype(object)
    sd = nb_newton_solver(as_tensor(W, "cpu"), pen, group)
    close(sc.solve(w, s), N(sd.solve(w, s)), 1e-10)
    np.testing.assert_array_equal(sc.factor(w).rank, [2] * D.G)


def test_blocked_dense_path_matches_r(ref):
    b = ref["blocked"]
    D = build(ref["blk_main"])
    np.testing.assert_array_equal(
        _block_dense_absorb(D) + 1, b["group"]
    )  # R ids are 1-based with NA -> 0 in the file
    group = np.where(b["group"] == 0, None, b["group"] - 1).astype(object)
    sol = nb_newton_solver(as_tensor(b["W"], "cpu"), b["pen"], group)
    close(sol.solve(b["w"], b["s"]), b["step"])


def test_empty_group_unpenalised_keeps_start_and_reduced_design_agrees():
    # unpenalised block columns of a group with no cells: the generalised inverse
    # leaves them out, so the step equals the step of the design without that group
    rng = np.random.default_rng(5)
    sizes = [80, 0, 70, 50]
    block = pd.Categorical(
        np.repeat([f"P{i}" for i in range(4)], sizes), categories=[f"P{i}" for i in range(4)]
    )
    n = sum(sizes)
    X, Z = rng.standard_normal((n, 3)), np.column_stack([np.ones(n), rng.standard_normal(n)])
    D = nb_block_design(X, Z, block)
    assert "1 empty" in repr(D)
    w, s = rng.uniform(0.1, 2, n), rng.standard_normal(D.shape[1])
    pen0 = np.zeros(5)
    step = N(nb_newton_solver(D, pen0).solve(w, s))
    empty = slice(3 + 2, 3 + 4)
    assert (step[empty] == 0).all()
    keep = np.r_[np.arange(5), np.arange(7, 11)]
    Wd = D.to_dense()[:, keep]
    info = (Wd * w[:, None]).T @ Wd
    dx = np.linalg.solve(info, s[keep])
    np.testing.assert_allclose(step[keep], dx, rtol=1e-8)


def test_block_design_validation():
    rng = np.random.default_rng(0)
    X, Z, b = (
        rng.standard_normal((10, 2)),
        np.column_stack([np.ones(10), rng.standard_normal(10)]),
        np.tile(["a", "b"], 5),
    )
    assert nb_block_design(X, Z, b).G == 2
    assert nb_block_design(X[:, 0], Z, b).px == 1  # a vector X
    with pytest.raises(ValueError, match="one row per cell"):
        nb_block_design(X, Z[1:], b)
    with pytest.raises(ValueError, match="one value per cell"):
        nb_block_design(X, Z, b[1:])
    bn = b.astype(object)
    bn[2] = None
    with pytest.raises(ValueError, match="missing"):
        nb_block_design(X, Z, bn)
    Xn = X.copy()
    Xn[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        nb_block_design(Xn, Z, b)
    with pytest.raises(ValueError, match="at least one column"):
        nb_block_design(X, Z[:, :0], b)
    with pytest.raises(ValueError, match="at least one column"):
        nb_block_design(X[:, :0], Z, b)
    with pytest.raises(ValueError, match="rank_tol"):
        nb_block_design(X, Z, b, rank_tol=2)
    with pytest.raises(ValueError, match="numeric"):
        nb_block_design(X, np.full((10, 1), "a"), b)


def test_block_expand():
    D = nb_block_design(np.ones((6, 3)), np.ones((6, 2)), np.tile(["a", "b", "c"], 2))
    q, G = 2, 3
    np.testing.assert_array_equal(
        _block_expand(D, [1, 2, 3, 11, 12], "v"), [1, 2, 3] + [11, 12] * G
    )
    np.testing.assert_array_equal(_block_expand(D, 7, "v"), np.full(3 + G * q, 7))
    full = np.arange(3 + G * q)
    np.testing.assert_array_equal(_block_expand(D, full, "v"), full)
    with pytest.raises(ValueError, match="lambda_a"):
        _block_expand(D, [1, 2, 3, 4], "lambda_a")


def test_repr_reports_groups_and_columns(ref):
    D = build(ref["blk_main"])
    assert "5 groups x 2 block columns" in repr(D)
    assert D.colnames[:5] == ["n1", "n2", "n3", "P01:int", "P01:ls1"]


def test_solver_rejects_absorb_for_block_design_and_batched_helpers_refuse_it(ref):
    D = build(ref["blk_main"])
    group = _block_dense_absorb(D)
    with pytest.raises(ValueError, match="must be None"):
        nb_newton_solver(D, [0, 0, 0, 0, 0], absorb=group >= 0)
    nb_newton_solver(D, [0, 0, 0, 0, 0], absorb=np.zeros(D.shape[1], bool))
    wt = np.ones((2, D.n))
    for call in (
        lambda: nb_newton_solver_batch(D, 0),
        lambda: nb_gram_batch(D, wt),
        lambda: nb_absorb_gram_batch(D, 0, None, wt),
    ):
        with pytest.raises(ValueError, match="dense design"):
            call()


def test_sorted_and_unsorted_cells_agree():
    # cells not sorted by group: compact helpers must follow the cell order
    rng = np.random.default_rng(3)
    n = 40
    block = rng.choice(["a", "b", "c"], n)
    X, Z = rng.standard_normal((n, 2)), np.column_stack([np.ones(n), rng.standard_normal(n)])
    D = nb_block_design(X, Z, block)
    a = rng.standard_normal(D.shape[1])
    close(_block_eta_vec(D, a), D.to_dense() @ a, 1e-12)
