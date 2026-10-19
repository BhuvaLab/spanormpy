# Discrepancies from R SpaNorm

## geneBlockCount integer overflow

`geneBlockCount()` (R/gpuFunctions.R) does `as.integer(ceiling(needed.bytes / budget.bytes))`
before clamping to `n_genes`. When the ratio exceeds 2^31 (for example a 1-byte budget on a
large problem) `as.integer` yields `NA` with a warning and the function returns `NA`.
`spanormpy._backend.gene_block_count` clamps in floating point and returns `n_genes` in that case.
The two agree everywhere R returns a value.

## pnbinom / qnbinom on accelerators

On CPU tensors `pnbinom`/`qnbinom` are exact (scipy incomplete beta; for size >= 1000 and
q <= 5000 the CDF is summed from the pmf in extended precision because scipy's `betainc`
loses ~5e-11 relative accuracy at size 1e6), and `qnbinom` uses R's 64*eps fuzz. On
accelerators they use R's GPU algorithm (cumulative pmf sum), whose CDF is accurate to ~1e-11
relative in far tails, so `qnbinom` there uses a 1e-10 relative fuzz instead. A p lying exactly
on a CDF step can therefore differ from R by one count on accelerators, never on CPU.
If the search cap (1e6) is reached, unfinished elements are NaN with a warning.

## Nested absorption with a group holding no cells (`.newtonSolver`)

In the diagonal nested path `.newtonSolver()` sums weights and `X*w` per group with `rowsum()` on a
factor, which drops a group with no cells. `cvec` and `B` then come back short, `pen_z` recycles
against them and the Schur complement is misaligned (or errors). The batched `.absorbBatch()`
already keeps the slot through `.segmentSum()`; R's own test notes the case is unreachable from the
package's designs. `spanormpy._solver._newton_solver` uses the same zero-filled segment sum for both,
so an empty group keeps its (penalty-only) slot. Results agree with R wherever R returns a valid answer.

## Singularity test in the dense solvers

R's `solve()` signals a singular system through LAPACK `dgesv` (exact zero pivot) or a reciprocal
condition number below `.Machine$double.eps` (estimated by `dgecon`). `_solver._lu_checked` applies
the same two tests with the exact 1-norm condition number, so a matrix whose estimated and exact
rcond straddle machine epsilon could in principle be classified differently.

## Batched Cholesky on a failed slice

R's torch branch of `.newtonSolverBatch()$xcov` calls `torch_cholesky_inverse()` on the whole stack,
which raises on a singular slice; its base-R branch returns `NA` for that slice. Python follows the
base-R behaviour: failed slices are NaN and their neighbours are unaffected.

## edgeR `compute_unit_nb_deviance` integer division (matched, not fixed)

edgeR's C routine evaluates the small-dispersion (phi < 1e-4) Poisson-like branch with
`2/3*resid`, which is C integer division and equals 0. `spanormpy._edger._unit_nb_deviance`
reproduces this exactly (`0.0 * resid`), so deviances and the residual variances fed to
`squeezeVar` agree with R. It only matters for trended dispersions below 1e-4.

## edgeR one-group Newton fit: unassigned coefficient on non-convergence (matched)

In `glm_one_group_vec` the output `*beta` is only assigned when the Newton iteration converges
within `maxit` (50); `fit_one_group_mat` / `average_log_cpm` then copy the uninitialised local
`ocoef` into the result. R observes ~4.6e-310 (effectively 0) for the first such row and the
previous row's value afterwards (same stack slot). `spanormpy._edger._glm_one_group` reproduces
that (0.0 for a leading row, otherwise the previous Newton-fitted row's coefficient). This is
undefined behaviour in edgeR, matched empirically on the R-vs-Python fuzz (all non-converging
scenarios reproduce); it affects very sparse, strongly overdispersed genes at large grid
dispersions.

## quickCluster: deterministic substitutes for irlba and the KMKNN search

`quick_cluster` follows `scran::quickCluster(Y)` step for step, but the centred SVD uses an exact
thin SVD (`numpy.linalg.svd`; `scipy.sparse.linalg.svds` only above 1000 retained genes) instead of
`irlba`, and the SNN graph uses an exact `cKDTree` search. Principal components agree up to sign
and `irlba` tolerance; exact distance ties in the neighbour search may be broken differently.
Clusters are compared to R by adjusted Rand index (>= 0.9, 1.0 on the test data), not exactly.

## cleanSizeFactors: nls stopping point

`scuttle::cleanSizeFactors` (the `positive = TRUE` repair of non-positive pooled estimates) fits a
nonlinear trend with `nls`, which stops at a 1e-5 relative-offset criterion. The Python Gauss-Newton
uses LAPACK QR where R uses LINPACK `dqrdc2`, so the replaced entries agree to ~1e-9 relative
rather than 1e-10; all other size factors agree to 1e-10.

## fitNB / fitSpaNormNB: `tol` reaches only the outer loop

`fitSpaNormNB(..., tol = )` has `tol` as a formal argument, so it is not part of the `...` forwarded to
`fitNBGivenPsi`; the inner IRLS loop therefore always uses its default `tol = 1e-4` and `fitNB(tol = )`
only changes the outer dispersion loop. Matched in `fit_spanorm_nb` (pinned by
`test_tol_reaches_only_outer_loop`).

## fitNBGivenPsi / fitSpaNormNB: NaN in step-halving and convergence comparisons

R evaluates `if (is.na(degener) | degener)` (NaN log-likelihood sums are treated as degenerate, matched),
but `step[check.gene] <- ...` with an `NA` in the per-cell logical index, and
`ifelse((l1 - l2) / abs(l2) < tol, TRUE, FALSE)` with NaN inputs, error or give `NA` in R. Python treats NaN
comparisons as False: such cells keep their step size, and a NaN convergence ratio counts as not converged.
