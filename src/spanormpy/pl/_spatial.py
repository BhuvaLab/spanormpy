"""Spatial scatter plots (R ``plotSpatial``, ``R/plotSpatial.R``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .._io import get_coords

_WHAT = ("annotation", "expression", "reduceddim")


def _image(adata, library_id, img_key):
    """Tissue image and its scale factors from the scanpy Visium ``uns["spatial"]`` layout.

    Returns ``(image, scale_factor, spot_diameter_fullres)``; mirrors R ``extractImage``
    (an object without image data is an error).
    """
    spatial = adata.uns.get("spatial") if hasattr(adata.uns, "get") else None
    if not spatial:
        raise ValueError("no image data found: 'adata.uns[\"spatial\"]' is missing or empty")
    if library_id is None:
        library_id = next(iter(spatial))
    if library_id not in spatial:
        raise ValueError(f"library_id '{library_id}' not found in adata.uns['spatial']")
    lib = spatial[library_id]
    images = lib.get("images", {}) if hasattr(lib, "get") else {}
    if img_key not in images:
        raise ValueError(
            f"no '{img_key}' image in adata.uns['spatial']['{library_id}']['images']"
        )
    sfs = lib.get("scalefactors", {})
    sf = sfs.get(f"tissue_{img_key}_scalef", 1.0)
    return np.asarray(images[img_key]), float(sf), sfs.get("spot_diameter_fullres")


def _gene_values(adata, gene, layer):
    if gene not in adata.var_names:
        raise ValueError(f"gene '{gene}' not found in 'adata.var_names'")
    if (adata.var_names == gene).sum() != 1:
        raise ValueError(f"gene '{gene}' is duplicated in 'adata.var_names'")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"layer '{layer}' not found in 'adata'")
    x = adata[:, gene].layers[layer] if layer is not None else adata[:, gene].X
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    return np.asarray(x, dtype=float).ravel()


def _component(adata, basis, color):
    if basis is None:
        raise ValueError("'basis' is required when what='reduceddim'")
    key = basis if basis in adata.obsm else f"X_{basis}"
    if key not in adata.obsm:
        raise ValueError(f"'{basis}' not found in adata.obsm")
    m = adata.obsm[key]
    if isinstance(m, pd.DataFrame):
        if color not in m.columns:
            raise ValueError(f"component '{color}' not found in adata.obsm['{key}']")
        return m[color].to_numpy(dtype=float)
    m = np.asarray(m)
    if isinstance(color, bool) or not isinstance(color, (int, np.integer)):
        raise ValueError("'color' must be an integer column index for an array in 'obsm'")
    if not -m.shape[1] <= color < m.shape[1]:
        raise ValueError(f"component {color} out of range for adata.obsm['{key}']")
    return m[:, color].astype(float)


def _values(adata, what, color, layer, basis):
    if what not in _WHAT:
        raise ValueError(f"'what' must be one of: {', '.join(_WHAT)}")
    if color is None:
        return None, None
    if what == "annotation":
        if color not in adata.obs.columns:
            raise ValueError(f"'{color}' not found in 'adata.obs'")
        return adata.obs[color], str(color)
    if what == "expression":
        return _gene_values(adata, color, layer), str(color)
    return _component(adata, basis, color), str(color)


def _draw(adata, values, label, *, title=None, img=False, crop=False, img_alpha=1.0,
          size=None, circles=False, ax=None, show=None, library_id=None, img_key="hires",
          spatial_key="spatial", cmap="viridis", **kwargs) -> Axes:
    if not 0 <= img_alpha <= 1:
        raise ValueError("'img_alpha' must be in [0, 1]")
    xy = get_coords(adata, spatial_key)
    if ax is None:
        _, ax = plt.subplots()

    sf, diam = 1.0, None
    if img:
        image, sf, diam = _image(adata, library_id, img_key)
        ax.imshow(image, alpha=img_alpha)
    xy = xy * sf

    legend = None
    cb_values = None
    if values is None:
        c, extra = "tab:blue", {}
    elif isinstance(values, pd.Series) and (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
    ):
        cat = values.astype("category")
        cats = list(cat.cat.categories)
        stored = adata.uns.get(f"{label}_colors")
        pal = list(stored) if stored is not None and len(stored) >= len(cats) else None
        if pal is None:
            base = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            pal = [base(i % base.N) for i in range(len(cats))]
        lut = dict(zip(cats, pal))
        c = [lut[v] if not pd.isna(v) else "lightgrey" for v in cat.astype(object)]
        extra = {}
        legend = [Line2D([], [], marker="o", ls="", color=lut[k], label=str(k)) for k in cats]
    else:
        cb_values = np.asarray(values, dtype=float)
        c, extra = cb_values, {"cmap": cmap}

    if circles:
        r = size if size is not None else 100.0
        pc = PatchCollection([Circle((x, y), r * sf) for x, y in xy], **kwargs)
        if cb_values is not None:
            pc.set_array(cb_values)
            pc.set_cmap(cmap)
        else:
            pc.set_facecolor(c)
        ax.add_collection(pc)
        ax.autoscale_view()
        mappable = pc
    else:
        s = size if size is not None else 120000 / max(adata.n_obs, 1)
        mappable = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=s, linewidths=0, **extra, **kwargs)

    if cb_values is not None:
        ax.figure.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.1, label=label)
    if legend is not None:
        ax.legend(handles=legend, title=label, loc="upper center",
                  bbox_to_anchor=(0.5, -0.05), ncol=min(len(legend), 4), frameon=False)
    if crop and img:  # R: `if (crop & img)`
        pad = (r * sf if circles else (diam or 0.0) / 2 * sf)
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)  # y points down
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_spatial(adata, what="annotation", color=None, layer=None, basis=None, img=False,
                 crop=False, img_alpha=1.0, size=None, circles=False, ax=None, show=None,
                 library_id=None, img_key="hires", spatial_key="spatial", **kwargs) -> Axes:
    """Plot spots at their spatial coordinates (R ``plotSpatial``).

    Parameters
    ----------
    adata
        Annotated data matrix with ``obsm[spatial_key]``.
    what
        ``"annotation"`` (``color`` is an ``obs`` column), ``"expression"`` (``color`` is a
        gene, read from ``layer`` or ``X``) or ``"reduceddim"`` (``color`` is a column name or
        index of ``obsm[basis]``).
    color
        What to colour spots by; ``None`` draws plain spots.
    layer, basis
        Expression layer (default ``X``) and the ``obsm`` key (``"X_"`` prefix optional).
    img
        Draw the tissue image from ``uns["spatial"][library_id]["images"][img_key]``
        (scanpy Visium layout); coordinates are scaled by ``tissue_<img_key>_scalef``.
        An error is raised when no image is present.
    crop
        Limit the axes to the extent of the spots (padded by the spot radius). Requires
        ``img=True`` and is ignored otherwise, as in R.
    img_alpha
        Image opacity in [0, 1].
    size
        Marker area (points^2), or the circle radius in full-resolution pixels when
        ``circles=True`` (default 100, as in R; the radius is multiplied by the image scale factor).
    circles
        Draw spots as true circles in data units (slower for large data).
    ax
        Axes to draw on; a new figure is created when ``None``.
    show
        Call ``plt.show()`` when true (``None`` is the same as ``False``). The axes are always returned.
    library_id, img_key, spatial_key
        Entry of ``uns["spatial"]`` (default: the first), image resolution
        (``"hires"``/``"lowres"``) and ``obsm`` key of the coordinates.
    **kwargs
        Passed to the scatter (or ``PatchCollection`` when ``circles=True``).

    Examples
    --------
    >>> import spanormpy as sn
    >>> # adata: the HumanDLPFC AnnData (raw counts in X, coordinates in obsm["spatial"])
    >>> # colour spots by an annotation
    >>> ax = sn.pl.plot_spatial(adata, color="AnnotatedCluster", size=3)

    >>> # colour spots by the expression of a gene
    >>> ax = sn.pl.plot_spatial(adata, what="expression", color="MOBP", size=3)

    >>> # use a layer (here the normalised values written by sn.spanorm) and a different colour map
    >>> ax = sn.pl.plot_spatial(adata, what="expression", color="MOBP", layer="logcounts",
    ...                         size=3, cmap="magma")
    """
    values, label = _values(adata, what, color, layer, basis)
    return _draw(adata, values, label, img=img, crop=crop, img_alpha=img_alpha, size=size,
                 circles=circles, ax=ax, show=show, library_id=library_id, img_key=img_key,
                 spatial_key=spatial_key, **kwargs)
