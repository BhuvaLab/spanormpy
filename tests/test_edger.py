"""edgeR::estimateDisp port (tests/ref/make_edger.R)."""

import numpy as np
import pytest
from tests._ref import load_ref

from spanormpy import _edger as edger
from spanormpy._edger import EdgeRError, estimate_disp, estimate_disp_tagwise

REF = load_ref("edger")
CASES = sorted(k for k in REF if k != "z")

# rtol=1e-8 (looser than the 1e-10 closed-form default): the dispersions come out of a
# spline-maximised adjusted profile likelihood built from ~1e3-magnitude log-likelihood sums
# (lgamma differences at ~1e-12 relative), a Newton fit with tol 1e-10, locfit's tree
# interpolation and Brent's uniroot/optimize (tol 1e-8 / 1e-4 on iterates that agree to ~1e-13).


def _run(case):
    c = REF[case]
    return estimate_disp(c["y"], c["offset"], robust=bool(c["robust"]))


@pytest.fixture(scope="module", params=CASES)
def case_result(request):
    return request.param, _run(request.param)


def test_tagwise_matches_r(case_result):
    case, out = case_result
    np.testing.assert_allclose(out["tagwise_dispersion"], REF[case]["tagwise"], rtol=1e-8, atol=0)


def test_tagwise_helper_matches_estimate_disp():
    c = REF["a"]
    np.testing.assert_array_equal(
        estimate_disp_tagwise(c["y"], c["offset"], robust=True), _run("a")["tagwise_dispersion"]
    )


def test_common_trended_match_r(case_result):
    case, out = case_result
    np.testing.assert_allclose(out["common_dispersion"], REF[case]["common"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(out["trended_dispersion"], REF[case]["trended"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(out["span"], REF[case]["span"], rtol=1e-12)


def test_prior_df_matches_r(case_result):
    case, out = case_result
    # non-robust R returns a scalar prior.df (one value shared by all genes); we return it per gene
    for key in ("prior_df", "prior_n"):
        got, ref = np.broadcast_arrays(
            np.asarray(out[key], dtype=float), np.atleast_1d(REF[case][key])
        )
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=0)


def test_apl_grid_and_ave_log_cpm_match_r(case_result):
    case, out = case_result
    ref = REF[case]
    sel = ref["sel"].astype(bool)
    np.testing.assert_array_equal(out["sel"], sel)
    scale = np.abs(ref["apl"]).max()
    # APL values are ~1e3 and carry ~1e-12 relative lgamma round-off -> 1e-8 relative to the scale
    np.testing.assert_allclose(out["apl_grid"], ref["apl"], rtol=1e-8, atol=1e-8 * scale)
    np.testing.assert_allclose(out["ave_log_cpm"], ref["ave_log_cpm"], rtol=1e-10, atol=1e-12)


def test_unequal_residual_df_branch_is_taken():
    # case e has cells with zero count and fitted mean < 1e-4, so residual df differ by gene
    assert _run("e")["unequal_df"]
    assert not _run("a")["unequal_df"]


def test_nonconverged_newton_case_is_exercised(monkeypatch):
    # cases j and k must hit the non-converged branch of the one-group fit (matched to R in
    # test_tagwise_matches_r[j/k]); guard against the case silently losing that property
    flags = []
    orig = edger._glm_one_group

    def spy(*args, **kwargs):
        beta, conv = orig(*args, **{**kwargs, "return_conv": True})
        flags.append(conv)
        return beta

    monkeypatch.setattr(edger, "_glm_one_group", spy)
    _run("j")
    _run("k")
    assert any(not c.all() for c in flags)


def test_zero_library_cell_raises_like_r():
    # R: aveLogCPM offset log(lib.size) = -Inf -> "offsets must be finite values"
    c = REF["z"]
    assert "finite" in c["error"]
    with pytest.raises(EdgeRError, match="offsets must be finite values"):
        estimate_disp_tagwise(c["y"], c["offset"], robust=bool(c["robust"]))


def test_degenerate_genes_get_trended_value():
    # case c: genes 1-3 (all zero, sum 1, sum 5 handled as selected) -- unselected genes use the
    # trended dispersion at the lowest AveLogCPM
    out = _run("c")
    sel = out["sel"]
    assert not sel[0] and not sel[1] and sel[2]
    np.testing.assert_array_equal(out["tagwise_dispersion"][~sel], out["trended_dispersion"][~sel])


def test_vector_offset_equals_matrix_offset():
    c = REF["h"]
    off_vec = c["offset"][0]
    a = estimate_disp(c["y"], off_vec)["tagwise_dispersion"]
    b = estimate_disp(c["y"], np.tile(off_vec, (c["y"].shape[0], 1)))["tagwise_dispersion"]
    np.testing.assert_array_equal(a, b)


def test_non_finite_offset_raises():
    c = REF["a"]
    off = c["offset"].copy()
    off[0, 0] = -np.inf
    with pytest.raises(EdgeRError, match="finite"):
        estimate_disp_tagwise(c["y"], off)


def test_wrong_offset_shape_raises():
    c = REF["a"]
    with pytest.raises(EdgeRError):
        estimate_disp_tagwise(c["y"], c["offset"][:, :10])
    with pytest.raises(EdgeRError):
        estimate_disp_tagwise(c["y"], np.zeros(7))


def test_no_gene_with_enough_counts_raises():
    with pytest.raises(EdgeRError):
        estimate_disp_tagwise(np.zeros((5, 20)), np.zeros((5, 20)))


def test_single_cell_returns_nan_with_warning():
    with pytest.warns(UserWarning, match="No residual df"):
        out = estimate_disp_tagwise(np.full((4, 1), 7.0), np.zeros((4, 1)))
    assert np.all(np.isnan(out))
