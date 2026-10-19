"""Load reference values written by tests/ref/make_*.R (rhdf5 HDF5 files)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from spanormpy._fit import SpaNormFit

REF_DIR = Path(__file__).parent / "ref" / "data"


def _decode(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind in ("S", "O"):
        return np.array(
            [s.decode() if isinstance(s, bytes) else s for s in a.ravel()], dtype=str
        ).reshape(a.shape)
    return a


def _read(obj):
    if isinstance(obj, h5py.Group):
        return {k: _read(obj[k]) for k in obj}
    a = obj[()]
    if isinstance(a, bytes):
        return a.decode()
    a = np.asarray(a)
    if a.ndim == 1 and a.size == 1:
        a = a.reshape(())  # length-1 R vector -> scalar (1x1 matrices are 2-D and stay arrays)
    if a.ndim == 0:
        return _decode(a.reshape(1))[0].item() if a.dtype.kind in "SO" else a.item()
    # rhdf5 writes R's column-major array reversed; h5py sees dims reversed, so transpose back.
    return _decode(a.T if a.ndim > 1 else a)


def load_ref(topic: str) -> dict:
    """Read ``tests/ref/data/<topic>.h5`` with R's array shapes.

    Length-1 R vectors become Python scalars; use ``np.atleast_1d`` if you need an array.
    1x1 matrices keep shape (1, 1). R ``NA`` becomes ``nan``.
    """
    with h5py.File(REF_DIR / f"{topic}.h5", "r") as f:
        return {k: _read(f[k]) for k in f}


def fit_from_ref(d, batch=None) -> SpaNormFit:
    lam = np.atleast_1d(d["lambda_a"]).astype(float)
    return SpaNormFit(
        ngenes=int(d["ngenes"]),
        ncells=int(d["ncells"]),
        gene_model="nb",
        df_tps=np.atleast_1d(d["df_tps"]).astype(int),
        sample_p=0.25,
        lambda_a=lam,
        batch=batch,
        W=np.asarray(d["W"]),
        alpha=np.asarray(d["alpha"]),
        gmean=np.asarray(d["gmean"]),
        psi=np.asarray(d["psi"]),
        wtype=np.atleast_1d(d["wtype"]).astype(str),
        loglik=np.atleast_1d(d["loglik"]).astype(float),
        sampling=np.atleast_1d(d["sampling"]).astype(str),
        colnames=np.atleast_1d(d["W_colnames"]).astype(str),
    )
