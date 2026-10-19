# spanormpy

[![PyPI version](https://img.shields.io/pypi/v/spanormpy)](https://pypi.org/project/spanormpy/)
[![PyPI monthly downloads](https://static.pepy.tech/badge/spanormpy/month)](https://pepy.tech/project/spanormpy)
[![PyPI total downloads](https://static.pepy.tech/badge/spanormpy)](https://pepy.tech/project/spanormpy)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/spanormpy)](https://anaconda.org/conda-forge/spanormpy)

Python / scverse port of the R package SpaNorm: spatially-aware library size normalisation for spatial
transcriptomics data, on `AnnData` with a PyTorch (CPU / CUDA) backend.

```bash
pip install git+https://github.com/BhuvaLab/spanormpy   # for now
pip install spanormpy                                    # once published on PyPI
```

For a GPU, install the CUDA build of PyTorch first. Download counts come from PyPI and conda-forge
and appear once the package is published; spanormpy has no telemetry.

```{toctree}
:maxdepth: 1

tutorial
api
r-to-python
r-discrepancies
benchmark
```
