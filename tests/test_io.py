import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spanormpy._io import check_batch, get_coords, get_counts, resolve_input


def _adata(xy, n=6):
    a = anndata.AnnData(np.ones((n, 3)))
    a.obsm["spatial"] = xy
    return a


# ---------------------------------------------------------------- coordinates
def test_coords_int_array_become_float():
    xy = get_coords(_adata(np.arange(12).reshape(6, 2)))
    assert xy.dtype == np.float64 and xy.shape == (6, 2)


def test_coords_dataframe():
    df = pd.DataFrame({"a": np.arange(6), "b": np.arange(6) * 2.0}, index=[str(i) for i in range(6)])
    xy = get_coords(_adata(df))
    assert xy.dtype == np.float64
    np.testing.assert_array_equal(xy[:, 1], np.arange(6) * 2.0)


def test_coords_three_columns_use_first_two():
    raw = np.random.default_rng(0).normal(size=(6, 3))
    np.testing.assert_array_equal(get_coords(_adata(raw)), raw[:, :2])


def test_coords_errors():
    with pytest.raises(ValueError, match="at least 2 columns"):
        get_coords(_adata(np.arange(6.0)[:, None]))
    bad = np.arange(12.0).reshape(6, 2)
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        get_coords(_adata(bad))
    bad[2, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        get_coords(_adata(bad))
    with pytest.raises(ValueError, match="not found"):
        get_coords(_adata(np.zeros((6, 2))), "xy")


# ---------------------------------------------------------------- counts
def test_get_counts_orientation_and_checks():
    a = anndata.AnnData(np.arange(12.0).reshape(4, 3))
    assert get_counts(a).shape == (3, 4)
    a.layers["raw"] = sp.csr_matrix(a.X)
    Y = get_counts(a, "raw")
    assert sp.issparse(Y) and Y.shape == (3, 4)
    a.X[1, 1] = 1.5
    with pytest.raises(ValueError, match="non-integer"):
        get_counts(a)
    a.X[1, 1] = -1
    with pytest.raises(ValueError, match="negative"):
        get_counts(a)
    with pytest.raises(ValueError, match="layer"):
        get_counts(a, "nope")


def test_resolve_input_rejects_other_types():
    with pytest.raises(TypeError):
        resolve_input(np.zeros((2, 2)))
    a = anndata.AnnData(np.ones((2, 2)))
    got, write_back = resolve_input(a)
    assert got is a and write_back(a) is None


# ---------------------------------------------------------------- check_batch (test-checkBatch.R)
def test_batch_none():
    assert check_batch(None, 5) is None


def test_batch_vector_checks():
    b = np.repeat(list("ABC"), 2)
    with pytest.raises(ValueError, match="length"):
        check_batch(b, 5)
    m = check_batch(b, 6)
    assert m.shape == (6, 2)
    # treatment contrasts: first level is the reference
    np.testing.assert_array_equal(m[:, 0], [0, 0, 1, 1, 0, 0])
    # numeric vector is a single covariate column
    assert check_batch(np.arange(1, 7), 6).shape == (6, 1)
    # numeric factor
    assert check_batch(pd.Categorical(np.tile([1, 2, 3], 2)), 6).shape == (6, 2)
    # NaN
    bad = b.astype(object)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        check_batch(bad, 6)
    with pytest.raises(ValueError, match="missing"):
        check_batch(np.array([1.0, np.nan, 2, 3, 4, 5]), 6)


def test_batch_levels_follow_r_collation_not_codepoint():
    # R (en_US.UTF-8): a < A < b < B; codepoint order would give A < B < a < b
    b = np.array(["B", "a", "b", "A", "a", "B"])
    m = check_batch(b, 6)
    assert m.shape == (6, 3)
    # reference level is "a": column 0 flags "A"
    np.testing.assert_array_equal(m[:, 0], [0, 0, 0, 1, 0, 0])


def test_batch_unused_categorical_level_keeps_zero_column():
    # R: model.matrix keeps columns for unused factor levels
    cat = pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"])
    m = check_batch(cat, 4)
    assert m.shape == (4, 2)
    assert np.all(m[:, 1] == 0)


def test_batch_single_level_errors_like_r():
    with pytest.raises(ValueError, match="2 or more levels"):
        check_batch(np.array(["a"] * 4), 4)
    with pytest.raises(ValueError, match="2 or more levels"):
        check_batch(pd.Categorical(["a"] * 4, categories=["a", "b"][:1]), 4)


def test_batch_matrix_checks():
    b = np.repeat(list("ABC"), 2)
    with pytest.raises(ValueError, match="numeric matrix"):
        check_batch(b[:, None], 6)
    design = pd.get_dummies(pd.Series(b), dtype=float)
    design.insert(0, "(Intercept)", 1.0)
    design = design.iloc[:, [0, 2, 3]]  # (Intercept), B, C
    with pytest.raises(ValueError, match="number of rows"):
        check_batch(design.to_numpy()[:5], 5 + 1)
    with pytest.warns(UserWarning, match="intercept"):
        out = check_batch(design, 6)
    assert out.shape == (6, 2)
    # intercept detected from all-ones even without a name
    with pytest.warns(UserWarning, match="intercept"):
        check_batch(design.to_numpy(), 6)
    named = design.rename(columns={"(Intercept)": "column1"})
    with pytest.warns(UserWarning, match="intercept"):
        check_batch(named, 6)
    # NaN
    nan = design.to_numpy().copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        check_batch(nan, 6)
    # no intercept: kept as is
    noint = design.iloc[:, 1:]
    assert check_batch(noint, 6).shape == (6, 2)
