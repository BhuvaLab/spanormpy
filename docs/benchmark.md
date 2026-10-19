# GPU verification and benchmark

Run on Bunya, partition `gpu_cuda`, one NVIDIA H100 PCIe (MIG 4g.40gb slice), 8 CPU cores, 64 GB,
torch 2.6.0+cu124. Scripts: `scripts/slurm_gpu_tests.sh`, `scripts/slurm_gpu_benchmark.sh`,
`scripts/benchmark_gpu.py`.

## Test suite on GPU

`pytest -q -m "gpu or not gpu"` (job 30149258, after the polish fix below):

```
696 passed, 14 skipped in 189.15s (0:03:09)
```

The 14 skips are the `[cuda]` cases of `tests/test_polish_nb.py` that request the per-gene engine
or a compact block design, which R also rejects on a device. CPU only: 621 passed. Every
`[cuda]`-parametrised test passed on the first GPU run (job 30148378, same counts); no CUDA failure
was found.

Fix made while auditing the CUDA paths: `polish_nb` built the per-gene solver from the device
tensor `W_use`; R builds it from the host design (the batched engine never uses it), so it is now
built from the host design on the device path.

## R GPU path versus CPU path

Read at c12e5e5 (`fitSpaNormNB.R`, `polishNB.R`, `polishEngineBatch.R`, `mainSpaNormSVG.R`,
`gpuFunctions.R`). Same algorithm as the CPU path, differing only in device and rounding:

- `fitSpaNormNB`: the `*_gpu` helpers (`tcrossprod_gpu`, `dnbinom_gpu`, ...) are the same
  operations on torch tensors; gene blocking only bounds device memory and does not change results.
- `polishNB`: a device requires `engine = "batch"` (the batched engine is also available on CPU and
  is what the CPU reference tests cover), serial dispatch, no compact block design. The shared-factor
  solver is exact (`absorb_batch`).
- `SpaNormSVG`: only forwards `backend`.

So no R GPU reference was produced (the R torch install has no CUDA either: `cuda_is_available()` is
FALSE); the CPU references apply to both devices, and the `[cuda]` tests compare to them.

## Timings (full HumanDLPFC: 4015 spots x 2508 genes after filterGenes(prop = 0.2))

Wall time in seconds, `n_jobs = 1`; CPU uses 8 torch threads. Job 30149257 (the test job ran
concurrently on the same node, so treat as indicative).

| function | cpu | gpu | speed-up |
|---|---|---|---|
| `spanorm` | 77.7 | 28.9 | 2.7x |
| `spanorm_svg` | 13.8 | 10.8 | 1.3x |

GPU was faster for both, so no profile of a slow function was needed. Where GPU time goes
(cProfile run, job 30148983): the remaining `spanorm` time on GPU is host work that has no device
path in R either (`quick_cluster` size factors, the edgeR dispersion estimation, `qnbinom`).
