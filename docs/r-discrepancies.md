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
