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
