import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from _ref import load_ref
from spanormpy.size_factors import (
    _col_sums,
    _fast_size_factors,
    compute_size_factors,
    fast_size_factors,
    quick_cluster,
    sum_factors,
)

REF = load_ref("sizefactors")


def test_col_sums_float32_sparse_accumulates_in_float64():
    # 1000 x (1e4 + 0.1): float32 accumulation gives 10000025, exact is 10000100
    Y = sp.csc_matrix(np.full((1000, 3), 1e4 + 0.1, dtype=np.float32))
    exp = float(np.float32(1e4 + 0.1)) * 1000
    got = _col_sums(Y)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, exp, rtol=1e-12)
    np.testing.assert_allclose(_col_sums(Y.toarray()), exp, rtol=1e-12)


@pytest.fixture(scope="module")
def datasets():
    ad = anndata.read_h5ad(Path(__file__).parent / "ref/data/dlpfc_small.h5ad")
    return {"dl": np.asarray(ad.X, dtype=float).T, "sim": REF["sim"]["counts"].astype(float)}


@pytest.fixture(scope="module")
def py_clusters(datasets):
    return {k: quick_cluster(Y) for k, Y in datasets.items()}


def _adata(Y):
    ad = anndata.AnnData(Y.T.copy())
    ad.obs_names = [f"c{i}" for i in range(ad.n_obs)]
    ad.var_names = [f"g{i}" for i in range(ad.n_vars)]
    return ad


@pytest.mark.parametrize("name", ["dl", "sim"])
def test_sum_factors_given_r_clusters_exact(datasets, name):
    sf = sum_factors(datasets[name], REF[name]["quick_cluster"])
    np.testing.assert_allclose(sf, REF[name]["sf_given_clusters"], rtol=1e-10)


@pytest.mark.parametrize("name", ["dl", "sim"])
def test_sum_factors_no_clusters_exact(datasets, name):
    # dl has 4015 cells > max.cluster.size (3000), so the cluster-splitting path is exercised
    sf = sum_factors(datasets[name])
    np.testing.assert_allclose(sf, REF[name]["sf_no_clusters"], rtol=1e-10)


def test_sum_factors_small_cluster_and_ref_clust(datasets):
    # cluster 4 has 15 cells: fewer than the smallest pool size (21), so only the ridge rows act
    d, Y = REF["sim"], datasets["sim"]
    np.testing.assert_allclose(sum_factors(Y, d["manual_clusters"]), d["sf_manual"], rtol=1e-10)
    np.testing.assert_allclose(
        sum_factors(Y, d["manual_clusters"], ref_clust=2), d["sf_ref_clust"], rtol=1e-10
    )


def test_sum_factors_positive_branch():
    Y = REF["low_counts"].astype(float)
    cl = REF["low_clusters"]
    with pytest.warns(UserWarning, match="non-positive size factor"):
        sf = sum_factors(Y, cl)
    assert REF["low_warn"] == "encountered non-positive size factor estimates"
    assert np.all(sf > 0)
    # the uncorrected estimate (positive = FALSE) is exact and has a non-positive entry
    with pytest.warns(UserWarning, match="non-positive"):
        raw = sum_factors(Y, cl, positive=False)
    np.testing.assert_allclose(raw, REF["low_sf_nopos"], rtol=1e-10)
    assert (raw <= 0).any()
    # cleanSizeFactors refits a nonlinear (nls) trend that stops at R's 1e-5 relative-offset
    # criterion; the replaced entry therefore agrees to ~1e-9 while the others stay at 1e-10.
    bad = raw <= 0
    np.testing.assert_allclose(sf[~bad], REF["low_sf"][~bad], rtol=1e-10)
    np.testing.assert_allclose(sf[bad], REF["low_sf"][bad], rtol=1e-7)


@pytest.mark.parametrize("name", ["dl", "sim"])
def test_quick_cluster_close(py_clusters, name):
    ari = adjusted_rand_score(REF[name]["quick_cluster"], py_clusters[name])
    assert ari >= 0.9


@pytest.mark.parametrize("name", ["dl", "sim"])
def test_default_size_factors_close(datasets, py_clusters, name):
    sf = sum_factors(datasets[name], py_clusters[name])
    r = np.corrcoef(np.log(sf), np.log(REF[name]["sf_given_clusters"]))[0, 1]
    assert r > 0.99


def test_quick_cluster_requires_min_size(datasets):
    with pytest.raises(ValueError, match="fewer cells than the minimum cluster size"):
        quick_cluster(datasets["sim"][:, :80])


def test_fast_size_factors_exact(datasets, dlpfc_small):
    np.testing.assert_allclose(
        _fast_size_factors(datasets["dl"]), REF["dl"]["fast"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _fast_size_factors(sp.csr_matrix(datasets["sim"])), REF["sim"]["fast"], rtol=1e-10
    )
    fast_size_factors(dlpfc_small)
    np.testing.assert_allclose(
        dlpfc_small.obs["spanorm_size_factors"], REF["dl"]["fast"], rtol=1e-10
    )
    assert (
        len(dlpfc_small.obs["spanorm_size_factors"]) == dlpfc_small.n_obs
    )  # testthat length check


def test_fast_size_factors_checks_counts(dlpfc_small):
    with pytest.raises(ValueError, match="nrow.+ncol"):
        fast_size_factors(dlpfc_small[:, :0].copy())


def test_zero_count_spot(datasets):
    Y = datasets["sim"].copy()
    Y[:, 6] = 0
    # R: calculateSumFactors errors on a cell with zero library size ...
    assert "cells should have non-zero library sizes or 'scaling' values" in REF["zero_msg"]
    with pytest.raises(ValueError, match="non-zero library sizes"):
        sum_factors(Y, np.ones(Y.shape[1], dtype=int))
    # ... and so does quickCluster (librarySizeFactors gives a zero size factor)
    assert "size factors should be positive" in REF["qc_zero_msg"]
    with pytest.raises(ValueError, match="size factors should be positive"):
        quick_cluster(Y)
    # fastSizeFactors gives the spot a zero factor, which the caller floors with pmax(1e-08, .)
    fast = _fast_size_factors(Y)
    assert fast[6] == 0
    assert np.maximum(1e-8, fast)[6] == 1e-8


def test_obs_size_factors_column_not_read(datasets):
    Y = datasets["sim"]
    cl = REF["sim"]["quick_cluster"]
    a, b = _adata(Y), _adata(Y)
    b.obs["size_factors"] = 1.0
    sa = compute_size_factors(a, clusters=cl, copy=True)
    sb = compute_size_factors(b, clusters=cl, copy=True)
    np.testing.assert_array_equal(sa, sb)
    np.testing.assert_allclose(sa, REF["sim"]["sf_given_clusters"], rtol=1e-10)
    np.testing.assert_array_equal(b.obs["size_factors"], 1.0)
    np.testing.assert_array_equal(b.obs["spanorm_size_factors"], sb)


def test_compute_size_factors_default_and_options(datasets):
    Y = datasets["sim"]
    ad = _adata(Y)
    assert compute_size_factors(ad) is None
    r = np.corrcoef(np.log(ad.obs["spanorm_size_factors"]), np.log(REF["sim"]["sf_given_clusters"]))
    assert r[0, 1] > 0.99
    # obs column name and key_added / layer
    ad.layers["raw"] = ad.X.copy()
    ad.X = np.ones(ad.shape)
    ad.obs["cl"] = pd.Categorical(REF["sim"]["quick_cluster"])
    out = compute_size_factors(ad, clusters="cl", layer="raw", key_added="sf2", copy=True)
    np.testing.assert_allclose(out, REF["sim"]["sf_given_clusters"], rtol=1e-10)
    np.testing.assert_array_equal(ad.obs["sf2"], out)


def test_sparse_matches_dense(datasets):
    Y = datasets["sim"]
    cl = REF["sim"]["quick_cluster"]
    for fmt in (sp.csr_matrix, sp.csc_matrix):
        S = fmt(Y.astype(np.float32))
        np.testing.assert_allclose(sum_factors(S, cl), REF["sim"]["sf_given_clusters"], rtol=1e-10)
        assert adjusted_rand_score(quick_cluster(S), quick_cluster(Y)) == 1.0
    ad = anndata.AnnData(sp.csr_matrix(Y.T.astype(np.int32)))
    compute_size_factors(ad, clusters=cl)
    np.testing.assert_allclose(
        ad.obs["spanorm_size_factors"], REF["sim"]["sf_given_clusters"], rtol=1e-10
    )


def test_view_is_converted_without_warning(datasets):
    ad = _adata(datasets["sim"])
    view = ad[:, :150]
    assert view.is_view
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_size_factors(view, clusters=np.ones(view.n_obs, dtype=int))
    assert "spanorm_size_factors" in view.obs
    assert "spanorm_size_factors" not in ad.obs
