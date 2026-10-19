# spanormpy

A faithful Python / scverse port of the R package
[SpaNorm](https://github.com/bhuvad/SpaNorm) (spatially-aware normalisation for spatial
transcriptomics data). Operates on `AnnData` (and optionally `SpatialData`) with a PyTorch
backend that runs on CPU and CUDA.

Status: under development (`0.1.0.dev0`).

## Install

```bash
pip install -e ".[dev,spatial]"
```

## Licence

GPL-3.0-or-later, as for SpaNorm.
