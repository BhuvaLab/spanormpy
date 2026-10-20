# spanormpy

[![CI](https://github.com/BhuvaLab/spanormpy/actions/workflows/ci.yaml/badge.svg)](https://github.com/BhuvaLab/spanormpy/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/spanormpy)](https://pypi.org/project/spanormpy/)
[![PyPI monthly downloads](https://static.pepy.tech/badge/spanormpy/month)](https://pepy.tech/project/spanormpy)
[![PyPI total downloads](https://static.pepy.tech/badge/spanormpy)](https://pepy.tech/project/spanormpy)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/spanormpy)](https://anaconda.org/conda-forge/spanormpy)

A faithful Python / scverse port of the R package
[SpaNorm](https://github.com/BhuvaLab/SpaNorm) (spatially-aware normalisation for spatial
transcriptomics data). It operates on `AnnData` (and optionally `SpatialData`) with a PyTorch
backend that runs on CPU and CUDA (Apple Metal in float32).

Documentation: <https://bhuvalab.github.io/SpaNorm/> (one site covering both the R package and spanormpy,
with side-by-side R and Python tutorials and the API reference for each).

Version 0.1.0 (beta). Ports SpaNorm 1.7.16 (R commit `f233b7b`, Bioconductor devel); see [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install spanormpy
pip install git+https://github.com/BhuvaLab/spanormpy   # development version
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
[docs/tutorial.ipynb](docs/tutorial.ipynb) (a squidpy mouse-brain Visium walk-through of the R vignette; the HumanDLPFC version identical to the R vignette is on the website). Every R export and its Python
name are listed in [docs/r-to-python.md](docs/r-to-python.md).

## Differences from R

The full list, with reasons, is in [docs/r-discrepancies.md](docs/r-discrepancies.md). The intentional ones:

- **Gene subsets reuse the stored fit (a deviation).** R refits when the object is subset by genes
  (`fit$ngenes != nrow`). After `adata[:, genes]`, `spanorm` instead reuses the matching slice of the stored
  fit (as R's `subsetFitGenes` would), which equals R's adjustment of the full fit on those genes;
  `overwrite=True` reproduces R's refit. A spot subset refits, as in R. `polish_spanorm` and `spanorm_svg`
  have their own subset rules (see the discrepancies page). Views are converted to real objects without a warning.
- **Explicit `size_factors` argument.** Pass size factors directly; `obs["size_factors"]` is never read implicitly.
- **Default pipeline matches R only approximately.** Default size factors come from a built-in scran-like
  method (own `quick_cluster`) written to `obs["spanorm_size_factors"]`, and spot sampling uses a numpy RNG
  (`random_state` replaces `set.seed`). Results agree with R to numerical closeness; exact parity is obtained
  when R's size factors and sampling are supplied.
- **Adjustment honours `backend`.** R's adjustment step uses any available accelerator regardless of the
  requested backend; here `backend="cpu"` keeps everything on the CPU.
- **`logcounts` follows the input.** `layers["logcounts"]` is dense for dense counts and sparse for sparse counts
  (R always stores sparse).
- **`n_jobs` replaces `BPPARAM`.**

## Download statistics

The badges above show download counts from PyPI (via pepy.tech) and conda-forge. They appear
once the package is published there. spanormpy contains no telemetry and makes no network calls.

## Citation

Salim A, Bhuva DD, Chen C, Yang P, Davis MJ, Yang JYH. SpaNorm: spatially-aware normalisation for
spatial transcriptomics data. bioRxiv 2024, 2024-05.
[doi:10.1101/2024.05.31.596908](https://doi.org/10.1101/2024.05.31.596908)

## Licence

GPL-3.0-or-later, as for SpaNorm.
