import numpy as np
import pytest
from tests._ref import load_ref

from spanormpy._tps import bs_tps, tps_basis

RTOL, ATOL = 1e-10, 1e-12


@pytest.fixture(scope="module")
def ref():
    return load_ref("tps")


def _same(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("df", [2, 6, 10])
@pytest.mark.parametrize("center", [True, False])
def test_tps_basis_matches_r(ref, df, center):
    r = ref[f"tps_df{df}_c{int(center)}"]
    B = tps_basis(ref["x"], ref["y"], df=df, center=center)
    assert B.shape == r["B"].shape
    assert B.dtype == np.float64
    _same(B, r["B"])


def test_tps_basis_pair_df_matches_r(ref):
    B = tps_basis(ref["x"], ref["y"], df=(4, 3))
    assert B.shape[1] == 12
    _same(B, ref["tps_pair"]["B"])


def test_tps_basis_reference_subset_and_extrapolation_match_r(ref):
    x, y = ref["x"], ref["y"]
    sub = np.asarray(ref["sub_idx"]).astype(int) - 1
    B = tps_basis(x[sub], y[sub], df=6, ref_x=x, ref_y=y)
    _same(B, ref["tps_ref"]["B"])
    full = tps_basis(x, y, df=6)
    _same(B, full[sub])
    assert np.abs(B.mean(axis=0)).max() > 1e-6  # centred over the reference, not the subset
    _same(full.mean(axis=0), 0)
    ex = tps_basis(
        [x.min() - 500, x.max() + 800, x.mean()],
        [y.max() + 300, y.min() - 100, y.mean()],
        df=6, ref_x=x, ref_y=y,
    )
    assert np.isfinite(ex).all()
    _same(ex, ref["extrap"]["B"])


def test_center_false_differs_by_reference_means(ref):
    x, y = ref["x"], ref["y"]
    U = tps_basis(x, y, df=(3, 3), center=False)
    B = tps_basis(x, y, df=(3, 3))
    assert B.shape[1] == 9
    _same(U - U.mean(axis=0), B)


@pytest.mark.parametrize("df", [1, 3, 6, 10])
def test_bs_tps_matches_r(ref, df):
    x, y = ref["x"], ref["y"]
    sx = (x - x.min()) / (x.max() - x.min()) - 0.5
    sy = (y - y.min()) / (y.max() - y.min()) - 0.5
    r = ref[f"bs_df{df}"]
    B, dfs = bs_tps(sx, sy, df)
    assert tuple(dfs) == tuple(np.atleast_1d(r["df"]))
    _same(B, r["B"])
    _same(tps_basis(sx, sy, df), B)  # tpsBasis on its own coordinates is bs.tps


def test_bs_tps_raw_coordinates_aspect_rule(ref):
    B, dfs = bs_tps(ref["x"], ref["y"], 6)
    assert tuple(dfs) == tuple(np.atleast_1d(ref["bs_raw6"]["df"]))
    _same(B, ref["bs_raw6"]["B"])


def test_tps_basis_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        tps_basis(np.arange(1, 4), np.arange(1, 5))
    with pytest.raises(ValueError, match="finite"):
        tps_basis([1, np.nan, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="positive integers"):
        tps_basis(np.arange(1, 6), np.arange(1, 6), df=0)
    with pytest.raises(ValueError, match="positive integers"):
        tps_basis(np.arange(1, 6), np.arange(1, 6), df=2.5)
    with pytest.raises(ValueError, match="positive integers"):
        tps_basis(np.arange(1, 6), np.arange(1, 6), df=(1, 2, 3))
    with pytest.raises(ValueError, match="same length"):
        tps_basis(np.arange(1, 6), np.arange(1, 6), ref_x=np.arange(1, 4), ref_y=np.arange(1, 5))
    with pytest.raises(ValueError, match="no range"):
        tps_basis(np.arange(1, 6), np.ones(5), df=3)


def test_bs_tps_validates_inputs():
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="greater than 0"):
        bs_tps(x, x, 0)
    with pytest.raises(ValueError, match="integer"):
        bs_tps(x, x, 2.5)
    with pytest.raises(ValueError, match="length"):
        bs_tps(x, x[:5], 3)
    with pytest.raises(ValueError, match="length"):
        bs_tps(x, x, [2, 3])
