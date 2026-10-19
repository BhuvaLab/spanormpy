import warnings
from dataclasses import fields

import anndata
import numpy as np
import pandas as pd
import pytest

from spanormpy._adjust import global_psi_max, subset_fit_genes
from spanormpy._fit import (
    SpaNormFit,
    get_spanorm_fit,
    is_polished,
    load_fit,
    store_fit,
    subset_genes,
)

NG, NC, P = 12, 30, 3


def make_fit(batch="none", polish=False, ngenes=NG, ncells=NC, seed=0):
    rng = np.random.default_rng(seed)
    b = {
        "none": None,
        "vector": np.array(["A", "B"] * (ncells // 2)),
        "numvec": np.arange(ncells) % 3,
        "matrix": rng.normal(size=(ncells, 2)),
        "strmatrix": np.array([["A", "x"], ["B", "y"]] * (ncells // 2)),
    }[batch]
    fit = SpaNormFit(
        ngenes=ngenes,
        ncells=ncells,
        gene_model="nb",
        df_tps=np.array([6, 6, 2, 2]),
        sample_p=0.25,
        lambda_a=np.array([0.5, 0.1]),
        batch=b,
        W=rng.normal(size=(ncells, P)),
        alpha=rng.normal(size=(ngenes, P)),
        gmean=rng.normal(size=ngenes),
        psi=rng.gamma(2, 1, size=ngenes),
        wtype=np.array(["ls", "biology", "biology"]),
        loglik=-np.array([5.0, 4.0, 3.5]),
        sampling=np.array((["all", "glm", "dispersion"] * ncells)[:ncells]),
    )
    fit.colnames = np.array(["(Intercept)", "bio1", "bio2"])
    if polish:
        fit.polish = {
            "settings": {
                "psi_method": "profile",
                "ls": "joint",
                "maxit": 50,
                "tol": 1e-8,
                "pen": np.array([0.0, 1.5, 1.5]),
                "psi_range": np.array([1e-3, 1e3]),
                "ls_converged": True,
                "SpaNorm": "1.7.15",
                "cells": None,
            },
            "genes": pd.DataFrame(
                {
                    "iterations": rng.integers(1, 9, ngenes),
                    "restarted": rng.random(ngenes) > 0.5,
                    "psi_fitnb": rng.gamma(2, 1, ngenes),
                    "polished": np.ones(ngenes, dtype=bool),
                    "loglik": np.where(np.arange(ngenes) == 2, np.nan, -rng.random(ngenes)),
                },
                index=[f"g{i}" for i in range(ngenes)],
            ),
        }
    return fit


def make_adata(fit):
    ad = anndata.AnnData(np.ones((fit.ncells, fit.ngenes)))
    ad.obs_names = [f"c{i}" for i in range(fit.ncells)]
    ad.var_names = [f"g{i}" for i in range(fit.ngenes)]
    return ad


def assert_fits_equal(a, b):
    for f in fields(SpaNormFit):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "polish":
            assert set(x) == set(y)
            for k in x:
                if k == "genes":
                    pd.testing.assert_frame_equal(
                        x[k].reset_index(drop=True), y[k].reset_index(drop=True), check_dtype=False
                    )
                elif k == "settings":
                    assert set(x[k]) == set(y[k])
                    for kk in x[k]:
                        np.testing.assert_array_equal(x[k][kk], y[k][kk])
                else:
                    np.testing.assert_array_equal(x[k], y[k])
        elif x is None or y is None:
            assert x is None and y is None, f.name
        elif f.name == "psi_max":
            assert x == y
        elif isinstance(x, (str, int, float)) and not isinstance(x, bool):
            assert x == y, f.name
        else:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=f.name)


@pytest.mark.parametrize("batch", ["none", "vector", "numvec", "matrix", "strmatrix"])
@pytest.mark.parametrize("polish", [False, True])
def test_store_load_roundtrip(batch, polish):
    fit = make_fit(batch, polish)
    ad = make_adata(fit)
    store_fit(ad, fit)
    assert_fits_equal(load_fit(ad), fit)
    assert is_polished(load_fit(ad)) == polish
    # slots per the spec table
    assert ad.varm["spanorm_alpha"].shape == (NG, P)
    assert ad.obsm["spanorm_W"].shape == (NC, P)
    assert {"spanorm_gmean", "spanorm_psi"} <= set(ad.var.columns)
    assert "spanorm_sampling" in ad.obs.columns


def test_none_setting_and_colnames_and_missing_batch():
    fit = make_fit("vector", polish=True)
    ad = make_adata(fit)
    store_fit(ad, fit)
    back = load_fit(ad)
    assert back.polish["settings"]["cells"] is None
    np.testing.assert_array_equal(back.colnames, fit.colnames)
    assert load_fit(ad[:, :3].copy()).colnames.tolist() == fit.colnames.tolist()
    fit.colnames = None
    store_fit(ad, fit)
    assert load_fit(ad).colnames is None
    fit.colnames = np.array(["a", "b"])
    with pytest.raises(ValueError, match="colnames"):
        fit.validate()
    store_fit(ad, make_fit("vector"))
    del ad.obs["spanorm_batch"]
    with pytest.raises(ValueError, match="incomplete"):
        load_fit(ad)


def test_name_prefix_and_missing():
    fit = make_fit()
    ad = make_adata(fit)
    assert load_fit(ad) is None
    store_fit(ad, fit, name="spanorm_null")
    assert "spanorm_null_alpha" in ad.varm
    assert load_fit(ad) is None
    assert get_spanorm_fit(ad, null=True) is not None
    assert get_spanorm_fit(ad, null=False, validate=False) is None
    with pytest.raises(ValueError, match="not found"):
        get_spanorm_fit(ad)


def test_restore_replaces_polish():
    ad = make_adata(make_fit())
    store_fit(ad, make_fit(polish=True))
    assert "spanorm_polish_iterations" in ad.var
    store_fit(ad, make_fit())
    assert not any(c.startswith("spanorm_polish_") for c in ad.var.columns)
    assert not is_polished(load_fit(ad))


@pytest.mark.parametrize("polish", [False, True])
def test_subset_after_store_equals_subset_genes(polish):
    fit = make_fit("vector", polish)
    ad = make_adata(fit)
    store_fit(ad, fit)
    mask = np.zeros(NG, dtype=bool)
    mask[[1, 2, 5, 7]] = True
    sub = load_fit(ad[:, mask].copy())
    want = subset_genes(fit, mask)
    assert_fits_equal(sub, want)
    assert sub.ngenes == 4
    assert sub.psi_max == global_psi_max(fit.psi) != global_psi_max(fit.psi[mask])
    # consistent with the adjustment layer's subsetFitGenes
    ref = subset_fit_genes(fit, np.flatnonzero(mask))
    assert ref.psi_max == sub.psi_max
    np.testing.assert_array_equal(ref.psi, sub.psi)
    # a subset of a loaded subset keeps the global threshold
    assert subset_fit_genes(sub, np.array([0, 1])).psi_max == sub.psi_max
    # a view works too
    assert_fits_equal(load_fit(ad[:, mask]), want)


def test_spot_subset_roundtrip():
    fit = make_fit("vector")
    ad = make_adata(fit)
    store_fit(ad, fit)
    idx = np.array([0, 3, 4, 10, 29])
    sub = load_fit(ad[idx].copy())
    assert sub.ncells == 5
    np.testing.assert_array_equal(sub.W, fit.W[idx])
    np.testing.assert_array_equal(sub.sampling, fit.sampling[idx])
    np.testing.assert_array_equal(sub.batch, fit.batch[idx])
    np.testing.assert_array_equal(sub.alpha, fit.alpha)
    assert sub.psi_max is None


def _arrays(d, out):
    for v in d.values():
        if isinstance(v, dict):
            _arrays(v, out)
        elif isinstance(v, np.ndarray) and v.ndim >= 1:
            out.append(v)
    return out


@pytest.mark.parametrize("polish", [False, True])
def test_uns_holds_no_per_gene_or_per_spot_arrays(polish):
    fit = make_fit("vector", polish)
    ad = make_adata(fit)
    store_fit(ad, fit)
    for a in _arrays(ad.uns["spanorm"], []):
        assert len(a) not in (ad.n_obs, ad.n_vars)


def test_polish_routing_by_shape():
    fit = make_fit(polish=True)
    rng = np.random.default_rng(1)
    fit.polish.update(
        gene_vec=rng.random(NG),
        gene_mat=rng.random((NG, 2)),
        cell_vec=rng.random(NC),
        cell_mat=rng.random((NC, 4)),
        total=3.5,
        label="x",
    )
    ad = make_adata(fit)
    store_fit(ad, fit)
    assert "spanorm_polish_gene_vec" in ad.var
    assert ad.varm["spanorm_polish_gene_mat"].shape == (NG, 2)
    assert "spanorm_polish_cell_vec" in ad.obs
    assert ad.obsm["spanorm_polish_cell_mat"].shape == (NC, 4)
    assert_fits_equal(load_fit(ad), fit)
    sub = load_fit(ad[:NC // 2, :6].copy())
    np.testing.assert_array_equal(sub.polish["cell_vec"], fit.polish["cell_vec"][: NC // 2])
    np.testing.assert_array_equal(sub.polish["gene_mat"], fit.polish["gene_mat"][:6])
    assert sub.polish["total"] == 3.5
    bad = make_fit(polish=True, ngenes=NC)
    bad.polish["x"] = np.zeros(NC)
    with pytest.raises(ValueError, match="ambiguous"):
        store_fit(make_adata(bad), bad)


@pytest.mark.parametrize("batch", ["none", "vector", "strmatrix"])
def test_h5ad_roundtrip(tmp_path, batch):
    fit = make_fit(batch, polish=True)
    ad = make_adata(fit)
    store_fit(ad, fit)
    store_fit(ad, make_fit(batch, seed=2), name="spanorm_null")
    path = tmp_path / "x.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ad.write_h5ad(path)
        back = anndata.read_h5ad(path)
    assert_fits_equal(load_fit(back), fit)
    assert_fits_equal(load_fit(back, "spanorm_null"), make_fit(batch, seed=2))
    mask = np.arange(NG) % 2 == 0
    assert_fits_equal(load_fit(back[:, mask].copy()), subset_genes(fit, mask))


def test_store_on_view_converts_without_warning():
    fit = make_fit(ngenes=6, ncells=10)
    ad = make_adata(make_fit())
    view = ad[:10, :6]
    assert view.is_view
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store_fit(view, fit)
    assert not view.is_view
    assert_fits_equal(load_fit(view), fit)
    assert "spanorm_alpha" not in ad.varm


def test_store_dimension_mismatch():
    ad = make_adata(make_fit())
    with pytest.raises(ValueError, match="does not match"):
        store_fit(ad, make_fit(ngenes=5))


def test_is_polished():
    fit = make_fit()
    assert not is_polished(fit)
    fit.polish = {"settings": {}, "genes": pd.DataFrame()}
    assert is_polished(fit)
    assert not is_polished(object())


def _mod(**kw):
    f = make_fit()
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_validate_ok():
    assert make_fit("vector").validate() is True
    assert make_fit("matrix", polish=True).validate() is True
    f = _mod(lambda_a=0.5, batch=np.full((NC, 2), "A"))
    assert f.validate()


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(df_tps=np.array([0, 6, 2, 2])), "greater"),
        (dict(gene_model="gamma"), "nb"),
        (dict(sample_p=0.0), "interval"),
        (dict(sample_p=1.1), "interval"),
        (dict(sample_p=-1.0), "interval"),
        (dict(lambda_a=-1.0), "positive"),
        (dict(loglik=np.array([1.0, -1, -2])), "less"),
        (dict(W=np.zeros((NC, P - 1))), "not match"),
        (dict(W=np.zeros((NC - 1, P))), "not match"),
        (dict(W=np.zeros(P)), "class"),
        (dict(W=np.where(np.arange(NC * P).reshape(NC, P) == 0, np.nan, 0.0)), "missing"),
        (dict(alpha=np.zeros((NG, P - 1))), "not match"),
        (dict(alpha=np.zeros((NG - 1, P))), "not match"),
        (dict(alpha=np.zeros(NG)), "class"),
        (dict(alpha=np.full((NG, P), np.nan)), "missing"),
        (dict(gmean=np.zeros(NG - 1)), "not match"),
        (dict(gmean=np.zeros(1)), "match"),
        (dict(gmean=np.r_[np.nan, np.zeros(NG - 1)]), "missing"),
        (dict(psi=np.zeros(NG - 1)), "not match"),
        (dict(psi=np.zeros(1)), "match"),
        (dict(psi=np.r_[np.nan, np.zeros(NG - 1)]), "missing"),
        (dict(wtype=np.array(["ls", "biology"])), "not match"),
        (dict(wtype=np.array(["ls"])), "match"),
        (dict(wtype=np.array(["ls", "biology", None], dtype=object)), "missing"),
        (dict(wtype=np.array(["ls", "biology", "foo"])), "biology"),
        (dict(sampling=np.array(["all"] * (NC - 1))), "not match"),
        (dict(sampling=np.array(["all"] * (NC - 1) + ["foo"])), "sampling"),
        (dict(batch=np.array(["A"] * (NC - 1))), "not match"),
        (dict(batch=np.array(["A"])), "match"),
        (dict(batch=np.array(["A"] * (NC - 1) + [None], dtype=object)), "missing"),
        (dict(batch=np.full((NC - 1, 2), "A")), "not match"),
        (dict(batch=np.where(np.arange(NC * 2).reshape(NC, 2) == 0, np.nan, 1.0)), "missing"),
        (dict(polish={"settings": {}}), "polish"),
    ],
)
def test_validate_errors(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _mod(**kw).validate()


def test_validate_nb_psi_only():
    f = _mod(gene_model="nb", psi=np.full(NG, np.nan))
    with pytest.raises(ValueError, match="missing"):
        f.validate()
