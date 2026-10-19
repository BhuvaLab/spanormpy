# spanormpy

[![PyPI version](https://img.shields.io/pypi/v/spanormpy)](https://pypi.org/project/spanormpy/)
[![PyPI monthly downloads](https://static.pepy.tech/badge/spanormpy/month)](https://pepy.tech/project/spanormpy)
[![PyPI total downloads](https://static.pepy.tech/badge/spanormpy)](https://pepy.tech/project/spanormpy)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/spanormpy)](https://anaconda.org/conda-forge/spanormpy)

A faithful Python / scverse port of the R package
[SpaNorm](https://github.com/bhuvad/SpaNorm) (spatially-aware normalisation for spatial
transcriptomics data). It operates on `AnnData` (and optionally `SpatialData`) with a PyTorch
backend that runs on CPU and CUDA (Apple Metal in float32).

Status: under development (`0.1.0.dev0`). Ports SpaNorm 1.7.15 (R commit `c12e5e5`).

## Install

```bash
pip install git+https://github.com/BhuvaLab/spanormpy   # for now
pip install spanormpy                                    # once published on PyPI
```

Optional extras: `pip install "spanormpy[spatial]"` for `SpatialData` input and
`pip install "spanormpy[dev]"` for the test and tutorial tools. For a GPU, install the CUDA build of
PyTorch first (see <https://pytorch.org>), then use `backend="gpu"` or the default `backend="auto"`.
See [docs/benchmark.md](docs/benchmark.md) (about 2.7x faster `spanorm` on an H100 MIG slice).

## Quickstart

```python
import anndata as ad
import spanormpy as sn

adata = ad.read_h5ad("visium.h5ad")        # raw counts in X, coordinates in obsm["spatial"]
sn.filter_genes(adata, prop=0.2)           # keep genes expressed in >= 20% of spots (in place)

sn.spanorm(adata)                          # normalised data -> adata.layers["logcounts"]
sn.pl.plot_spatial(adata, what="expression", color="MOBP", layer="logcounts")

sn.spanorm_svg(adata)                      # spatially variable genes -> adata.var["svg_*"]
sn.top_svgs(adata, n=10)
sn.spanorm_pca(adata)                      # GLM-PCA approximation -> adata.obsm["X_spanorm_pca"]
sn.polish_spanorm(adata)                   # optional: converge every gene separately
```

The fit is stored in the object (`var`, `varm`, `obsm`, and scalars in `uns["spanorm"]`), so
re-running with another `adj_method` only redoes the adjustment. A full walk-through is in
[docs/tutorial.ipynb](docs/tutorial.ipynb) (port of the R vignette). Every R export and its Python
name are listed in [docs/r-to-python.md](docs/r-to-python.md).

## Differences from R

The full list is in [docs/r-discrepancies.md](docs/r-discrepancies.md). The intentional ones:

- **Gene-subset reuse of stored fits.** After `adata[:, genes]`, `spanorm`, `spanorm_svg` and
  `polish_spanorm` reuse the matching part of the stored fit exactly as R does after `subsetFitGenes`;
  functions given a view convert it to an actual object without warning.
- **Explicit `size_factors` argument.** Pass size factors directly; `obs["size_factors"]` is never read
  implicitly.
- **Default size factors** are computed by the built-in scran-like method (clustered sum factors) and
  written to `obs["spanorm_size_factors"]`.
- **`logcounts` follows the input.** `layers["logcounts"]` is dense for dense counts and sparse for
  sparse counts.
- **`n_jobs` replaces `BPPARAM`**, and `random_state` replaces `set.seed`.

## Download statistics

The badges above show download counts from PyPI (via pepy.tech) and conda-forge. They appear
once the package is published there. spanormpy contains no telemetry and makes no network calls.

## Citation

Salim A, Bhuva DD, Chen C, Yang P, Davis MJ, Yang JYH. SpaNorm: spatially-aware normalisation for
spatial transcriptomics data. bioRxiv 2024, 2024-05.
[doi:10.1101/2024.05.31.596908](https://doi.org/10.1101/2024.05.31.596908)

## Licence

GPL-3.0-or-later, as for SpaNorm.
