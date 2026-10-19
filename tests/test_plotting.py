"""pl.plot_spatial / pl.plot_covariate: R parity of the covariate surfaces (tests/ref/make_plot.R) and plot content."""

import matplotlib

matplotlib.use("Agg")

import anndata
import matplotlib.image as mimg
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from tests._ref import fit_from_ref, load_ref

from spanormpy import pl
from spanormpy._fit import store_fit


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def ref():
    return load_ref("plot")


@pytest.fixture
def fitted(ref):
    xy = np.asarray(ref["coords"])
    n = xy.shape[0]
    g = ref["biology"].shape[0]
    a = anndata.AnnData(
        X=np.random.default_rng(0).poisson(2.0, (n, g)).astype(float),
        obs=pd.DataFrame({"grp": pd.Categorical(np.where(xy[:, 0] > np.median(xy[:, 0]), "a", "b"))},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
    )
    a.obsm["spatial"] = xy
    a.obsm["X_pca"] = np.random.default_rng(1).normal(size=(n, 3))
    a.layers["lc"] = np.log1p(a.X)
    store_fit(a, fit_from_ref(ref["fit"]), "spanorm")
    return a


def n_points(ax):
    cols = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(cols) == 1
    return len(cols[0].get_offsets())


@pytest.mark.parametrize("cov", ["biology", "ls", "batch"])
def test_covariate_values_match_r(ref, fitted, cov):
    key = "batch_effect" if cov == "batch" else cov
    got = pl.covariate_values(fitted, cov)
    np.testing.assert_allclose(got.T, ref[key], rtol=1e-6)
    ax = pl.plot_covariate(fitted, cov, color="g3", show=False)
    np.testing.assert_allclose(ax.collections[0].get_array(), ref[key][3], rtol=1e-6)
    assert ax.get_title() == cov


def test_covariate_errors(fitted):
    with pytest.raises(ValueError, match="no genes selected"):
        pl.plot_covariate(fitted)
    with pytest.raises(ValueError, match="no genes selected"):
        pl.plot_covariate(fitted, color="nope")
    with pytest.raises(ValueError, match="fit not found"):
        pl.plot_covariate(anndata.AnnData(np.ones((3, 2))), color="0")
    with pytest.raises(ValueError, match="one of"):
        pl.plot_covariate(fitted, "foo", color="g1")


def test_covariate_no_batch(ref, fitted):
    fit = fit_from_ref(ref["fit"])
    keep = fit.wtype != "batch"
    fit.W, fit.alpha, fit.wtype = fit.W[:, keep], fit.alpha[:, keep], fit.wtype[keep]
    fit.colnames = fit.colnames[keep]
    store_fit(fitted, fit, "spanorm")
    with pytest.raises(ValueError, match="no batch covariates"):
        pl.plot_covariate(fitted, "batch", color="g1")


def test_what_variants(fitted):
    n = fitted.n_obs
    assert n_points(pl.plot_spatial(fitted, "annotation", "grp")) == n
    assert n_points(pl.plot_spatial(fitted, "expression", "g2")) == n
    assert n_points(pl.plot_spatial(fitted, "expression", "g2", layer="lc")) == n
    ax = pl.plot_spatial(fitted, "reduceddim", 1, basis="pca")
    assert n_points(ax) == n
    np.testing.assert_allclose(ax.collections[0].get_array(), fitted.obsm["X_pca"][:, 1])
    assert n_points(pl.plot_spatial(fitted)) == n


def test_expression_values(fitted):
    ax = pl.plot_spatial(fitted, "expression", "g2", layer="lc")
    np.testing.assert_allclose(ax.collections[0].get_array(), fitted[:, "g2"].layers["lc"].ravel())


def test_bad_args(fitted):
    with pytest.raises(ValueError, match="'what'"):
        pl.plot_spatial(fitted, "foo")
    with pytest.raises(ValueError, match="obs"):
        pl.plot_spatial(fitted, "annotation", "zzz")
    with pytest.raises(ValueError, match="gene"):
        pl.plot_spatial(fitted, "expression", "zzz")
    with pytest.raises(ValueError, match="basis"):
        pl.plot_spatial(fitted, "reduceddim", 0)
    with pytest.raises(ValueError, match="img_alpha"):
        pl.plot_spatial(fitted, img_alpha=2)


def add_image(a):
    img = np.random.default_rng(2).random((20, 30, 3))
    a.uns["spatial"] = {"lib": {"images": {"hires": img},
                                "scalefactors": {"tissue_hires_scalef": 0.05, "spot_diameter_fullres": 100.0}}}
    return img


def test_img_requires_spatial(fitted):
    with pytest.raises(ValueError, match="no image data"):
        pl.plot_spatial(fitted, img=True)


def test_img_added_and_scaled(fitted):
    add_image(fitted)
    ax = pl.plot_spatial(fitted, "annotation", "grp", img=True, img_alpha=0.5)
    assert len(ax.images) == 1 and ax.images[0].get_alpha() == 0.5
    assert n_points(ax) == fitted.n_obs
    off = ax.collections[0].get_offsets()
    np.testing.assert_allclose(off, fitted.obsm["spatial"] * 0.05)
    with pytest.raises(ValueError, match="lowres"):
        pl.plot_spatial(fitted, img=True, img_key="lowres")


def test_crop_limits(fitted):
    add_image(fitted)
    xy = fitted.obsm["spatial"] * 0.05
    pad = 100.0 / 2 * 0.05
    ax = pl.plot_spatial(fitted, img=True, crop=True)
    np.testing.assert_allclose(ax.get_xlim(), [xy[:, 0].min() - pad, xy[:, 0].max() + pad])
    np.testing.assert_allclose(sorted(ax.get_ylim()), [xy[:, 1].min() - pad, xy[:, 1].max() + pad])
    # R: crop only acts with img
    a1 = pl.plot_spatial(fitted, crop=True)
    a2 = pl.plot_spatial(fitted, crop=False)
    assert a1.get_xlim() == a2.get_xlim() and a1.get_ylim() == a2.get_ylim()


def test_string_annotation_and_title(fitted):
    fitted.obs["s"] = pd.Series(np.where(fitted.obsm["spatial"][:, 0] > 0, "u", "v"), index=fitted.obs_names, dtype=object)
    assert n_points(pl.plot_spatial(fitted, "annotation", "s")) == fitted.n_obs
    fitted.obs["s2"] = fitted.obs["s"].astype("string")
    assert n_points(pl.plot_spatial(fitted, "annotation", "s2")) == fitted.n_obs
    ax = pl.plot_covariate(fitted, color="g1", title="mine")
    assert ax.get_title() == "mine"
    with pytest.raises(TypeError, match="single gene"):
        pl.plot_covariate(fitted, color=["g1"])


def test_circles_and_ax(fitted):
    fig, ax0 = plt.subplots()
    ax = pl.plot_spatial(fitted, "expression", "g1", circles=True, size=50, ax=ax0)
    assert ax is ax0
    assert len(ax.collections[0].get_paths()) == fitted.n_obs
