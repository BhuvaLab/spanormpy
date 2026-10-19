"""Device/dtype resolution, accelerator memory budget and gene blocking.

Port of ``R/gpuFunctions.R``. Only the helpers the Python port needs are kept;
the R-side tensor wrappers (``toGPUMatrix``, ``rowMeans_gpu``, ...) are plain
torch/numpy operations here.
"""

from __future__ import annotations

import math
import subprocess
import warnings

import numpy as np
import torch

DEFAULT_WINSOR = 4
# Conservative estimate of how many (genes_block x cells) tensors are co-resident
# on the device in one block-processing pass (see R's GPU_BLOCK_TENSOR_MULTIPLIER).
GPU_BLOCK_TENSOR_MULTIPLIER = 10
# Budget (bytes) used when free accelerator memory cannot be detected.
GPU_MEM_BUDGET_FALLBACK_BYTES = 2 * 1024**3
# Fraction of total system memory taken as the MPS budget (unified memory).
_MPS_SAFETY_FRACTION = 0.5
# R's constant for ``mad()`` (matrixStats::colMads uses the same).
_MAD_CONSTANT = 1.4826

_cache: dict = {}


def reset_gpu_cache() -> None:
    """Clear the cached device and memory budget (R ``resetGPUCache``)."""
    _cache.clear()


def get_backend_device() -> str:
    """``"cuda"`` if a CUDA GPU is available, else ``"mps"``, else ``"cpu"`` (cached)."""
    if "device" in _cache:
        return _cache["device"]
    dev = "cpu"
    try:
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
    except Exception:  # noqa: BLE001 - mirror R's tryCatch(..., "cpu")
        dev = "cpu"
    _cache["device"] = dev
    return dev


def get_backend_dtype() -> torch.dtype:
    """float32 on MPS, float64 elsewhere."""
    return torch.float32 if get_backend_device() == "mps" else torch.float64


def gpu_dtype_bytes() -> int:
    return 4 if get_backend_device() == "mps" else 8


def check_gpu() -> bool:
    """True when an accelerator (CUDA or MPS) is in use."""
    return get_backend_device() != "cpu"


def resolve_device(backend: str = "auto") -> torch.device:
    """Resolve R's ``backend`` argument to a torch device.

    ``"cpu"`` is always the CPU; ``"gpu"`` and ``"auto"`` use the accelerator
    when one is available and fall back to the CPU otherwise.
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("'backend' should be one of 'auto', 'cpu', 'gpu'")
    if backend != "cpu" and check_gpu():
        return torch.device(get_backend_device())
    return torch.device("cpu")


def _validate_budget(value, argname: str) -> float:
    ok = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not math.isnan(value)
        and value > 0
    )
    if not ok:
        raise ValueError(
            f"'{argname}' should be a single positive number (bytes), "
            "or Inf to disable blocking"
        )
    return value


def set_gpu_memory_budget(bytes: int | None) -> None:
    """Cache a memory budget (bytes; ``inf`` disables blocking). ``None`` clears it."""
    if bytes is None:
        _cache.pop("gpu_mem_budget", None)
        return
    _cache["gpu_mem_budget"] = _validate_budget(bytes, "bytes")


def _cuda_mem_free_bytes() -> float:
    try:
        return float(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        return math.nan


def _mps_mem_budget_bytes() -> float:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        total = float(out)
    except Exception:  # noqa: BLE001
        return math.nan
    return total * _MPS_SAFETY_FRACTION if total >= 0 else math.nan


def get_gpu_memory_budget(gpu_mem_budget: int | None = None) -> float:
    """Memory budget in bytes for gene blocking.

    A per-call override wins; otherwise the cached value, otherwise detected
    (``inf`` when no accelerator is in use) and cached for the session.
    """
    if gpu_mem_budget is not None:
        return _validate_budget(gpu_mem_budget, "gpu_mem_budget")
    if "gpu_mem_budget" in _cache:
        return _cache["gpu_mem_budget"]
    if not check_gpu():
        budget = math.inf
    else:
        dev = get_backend_device()
        detected = _cuda_mem_free_bytes() if dev == "cuda" else _mps_mem_budget_bytes()
        if math.isnan(detected):
            warnings.warn(
                f"Could not auto-detect available {dev} memory; using a conservative "
                f"{GPU_MEM_BUDGET_FALLBACK_BYTES / 1024**3:.1f} GiB budget. "
                "Override with 'gpu_mem_budget' (bytes).",
                stacklevel=2,
            )
            budget = GPU_MEM_BUDGET_FALLBACK_BYTES
        else:
            budget = detected
    _cache["gpu_mem_budget"] = budget
    return budget


def gene_block_count(
    n_genes: int,
    n_cells: int,
    n_cov: int,
    backend: str = "auto",
    budget_bytes: float | None = None,
    tensor_multiplier: float = GPU_BLOCK_TENSOR_MULTIPLIER,
) -> int:
    """Number of contiguous gene blocks keeping a genes x cells fit within the budget.

    Always 1 for ``backend="cpu"`` or when no accelerator is in use. ``n_cov``
    does not enter the formula (only genes x cells intermediates matter).
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("'backend' should be one of 'auto', 'cpu', 'gpu'")
    if backend == "cpu" or not check_gpu():
        return 1
    if budget_bytes is None:
        budget_bytes = get_gpu_memory_budget()
    if not math.isfinite(budget_bytes):
        return 1
    needed = float(n_genes) * float(n_cells) * gpu_dtype_bytes() * tensor_multiplier
    # Clamp in floating point: R converts to integer first and returns NA when
    # the ratio exceeds 2^31 (see docs/r-discrepancies.md).
    nblocks = math.ceil(needed / budget_bytes) if needed > budget_bytes else 1
    return int(max(1, min(int(n_genes), nblocks)))


def gene_block_indices(n_genes: int, nblocks: int) -> list[np.ndarray]:
    """Split ``range(n_genes)`` into contiguous blocks (0-based), as R's ``geneBlockIndices``.

    Like R, fewer than ``nblocks`` blocks are returned when the block size
    rounds up (e.g. 10 genes in 6 blocks gives 5 blocks of 2).
    """
    idx = np.arange(n_genes)
    size = math.ceil(n_genes / nblocks)
    return [idx[i : i + size] for i in range(0, n_genes, size)]


def as_tensor(x, device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert an array-like to a tensor on ``device`` (float64, or float32 on MPS)."""
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    arr = np.asarray(x)
    if not arr.flags.writeable:  # e.g. arrays of an AnnData view; torch warns on read-only input
        arr = arr.copy()
    return torch.as_tensor(arr, dtype=dtype, device=device)


def check_winsor(winsor) -> None:
    """Validate a winsorisation strength: a single positive number (``inf`` allowed)."""
    ok = (
        isinstance(winsor, (int, float, np.integer, np.floating))
        and not isinstance(winsor, (bool, np.bool_))
        and not math.isnan(winsor)
        and winsor > 0
    )
    if not ok:
        raise ValueError("'winsor' should be a single positive number")


def _col_median(x):
    """Median over axis 0 (average of the two middle values, like R)."""
    if isinstance(x, torch.Tensor):
        s = torch.sort(x, dim=0).values
        n = s.shape[0]
        return 0.5 * (s[(n - 1) // 2] + s[n // 2])
    return np.median(x, axis=0)


def winsorise_cols(alpha, k: float = DEFAULT_WINSOR):
    """Clamp each column of ``alpha`` to median +/- k*MAD (numpy or torch, same type out)."""
    check_winsor(k)
    if math.isinf(k):  # avoids k * mad == NaN when mad == 0
        return alpha
    is_t = isinstance(alpha, torch.Tensor)
    if not is_t:
        alpha = np.asarray(alpha, dtype=float)
    med = _col_median(alpha)
    mad = _MAD_CONSTANT * _col_median((alpha - med).abs() if is_t else np.abs(alpha - med))
    hi, lo = med + k * mad, med - k * mad
    if is_t:
        return torch.maximum(torch.minimum(alpha, hi), lo)
    return np.maximum(np.minimum(alpha, hi), lo)


def has_bad_values(x) -> bool:
    """True if ``x`` (array or tensor) contains NaN or +/-Inf."""
    if isinstance(x, torch.Tensor):
        return bool((~torch.isfinite(x)).any().item())
    return bool(not np.isfinite(np.asarray(x, dtype=float)).all())
