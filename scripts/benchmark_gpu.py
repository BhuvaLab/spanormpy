"""Time spanorm and spanorm_svg on the full HumanDLPFC data, CPU vs GPU.

Usage: python scripts/benchmark_gpu.py [--data PATH] [--backends cpu gpu] [--profile] [--out JSON]
The data (tests/ref/data/dlpfc_full.h5ad) is exported by tests/ref/make_dlpfc.R and git-ignored.
CPU runs use torch threads = the allocated cores. With --profile the slowest functions
(cProfile, cumulative time) of each stage are printed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import time

import anndata as ad
import torch

import spanormpy as sp


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, profile: bool):
    prof = cProfile.Profile() if profile else None
    _sync()
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    fn()
    _sync()
    if prof:
        prof.disable()
    dt = time.perf_counter() - t0
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
        print(s.getvalue())
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tests/ref/data/dlpfc_full.h5ad")
    ap.add_argument("--backends", nargs="+", default=["cpu", "gpu"])
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cores = _cores()
    torch.set_num_threads(cores)
    base = ad.read_h5ad(a.data)
    print(f"data {base.shape} (cells x genes); cores {cores}; cuda {torch.cuda.is_available()}")
    res = {"shape": list(base.shape), "cores": cores}
    for backend in a.backends:
        n_jobs = 1
        adata = base.copy()
        t_fit = _timed(
            lambda: sp.spanorm(adata, backend=backend, n_jobs=n_jobs, verbose=False),
            a.profile,
        )
        t_svg = _timed(lambda: sp.spanorm_svg(adata, backend=backend, verbose=False), a.profile)
        res[backend] = {"spanorm_s": t_fit, "spanorm_svg_s": t_svg}
        print(f"{backend}: spanorm {t_fit:.1f} s, spanorm_svg {t_svg:.1f} s", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
