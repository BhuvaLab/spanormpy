from pathlib import Path

import anndata
import pytest

DATA = Path(__file__).parent / "ref" / "data"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a CUDA device")


@pytest.fixture
def dlpfc_small():
    """Fresh copy of the reference DLPFC AnnData (cells x genes) for each test."""
    return anndata.read_h5ad(DATA / "dlpfc_small.h5ad")


def _devices():
    import torch

    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture(params=_devices())
def device(request):
    return request.param
