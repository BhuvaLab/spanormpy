import numpy as np
import pytest
import scipy.sparse as sp

from _ref import load_ref
from spanormpy._filter import _filter_genes, filter_genes

REF = load_ref("sizefactors")


def _counts(dlpfc_small):
    return np.asarray(dlpfc_small.X).T


@pytest.mark.parametrize("prop", ["0.1", "0.2", "0.5", "0", "1"])
def test_filter_genes_exact(dlpfc_small, prop):
    expected = REF["dl"][f"filter_{prop}"].astype(bool)
    Y = _counts(dlpfc_small)
    np.testing.assert_array_equal(_filter_genes(Y, float(prop)), expected)
    np.testing.assert_array_equal(_filter_genes(sp.csr_matrix(Y), float(prop)), expected)
    np.testing.assert_array_equal(filter_genes(dlpfc_small, float(prop), inplace=False), expected)


def test_filter_genes_simulated_exact():
    Y = REF["sim"]["counts"]
    for prop in ("0.1", "0.2", "0.5"):
        np.testing.assert_array_equal(
            _filter_genes(Y, float(prop)), REF["sim"][f"filter_{prop}"].astype(bool)
        )


def test_filter_genes_testthat_semantics(dlpfc_small):
    # port of test-filterGenes.R
    keep = filter_genes(dlpfc_small, inplace=False)
    assert keep.dtype == bool
    assert keep.shape == (dlpfc_small.n_vars,)
    with pytest.raises(ValueError, match="prop"):
        filter_genes(dlpfc_small, 1.1, inplace=False)
    with pytest.raises(ValueError, match="prop"):
        filter_genes(dlpfc_small, -0.1, inplace=False)
    assert filter_genes(dlpfc_small, 0, inplace=False).sum() == dlpfc_small.n_vars
    with pytest.raises(ValueError, match="nrow.+ncol"):
        filter_genes(dlpfc_small[:0].copy(), inplace=False)
    with pytest.raises(ValueError, match="nrow.+ncol"):
        filter_genes(dlpfc_small[:, :0].copy(), inplace=False)


def test_filter_genes_rejects_bad_counts(dlpfc_small):
    dlpfc_small.X = np.asarray(dlpfc_small.X, dtype=float)
    dlpfc_small.X[0, 0] = 0.1
    with pytest.raises(ValueError, match="non-integer"):
        filter_genes(dlpfc_small, inplace=False)
    dlpfc_small.X[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        filter_genes(dlpfc_small, inplace=False)


def test_filter_genes_layer_sparse_and_inplace(dlpfc_small):
    expected = REF["dl"]["filter_0.5"].astype(bool)
    dlpfc_small.layers["raw"] = sp.csr_matrix(np.asarray(dlpfc_small.X))
    dlpfc_small.X = np.zeros(dlpfc_small.shape)  # X must not be used when a layer is given
    np.testing.assert_array_equal(
        filter_genes(dlpfc_small, 0.5, layer="raw", inplace=False), expected
    )
    n_before = dlpfc_small.n_vars
    assert filter_genes(dlpfc_small, 0.5, layer="raw") is None
    assert dlpfc_small.n_vars == expected.sum() < n_before
