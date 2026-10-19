# Discrepancies from R SpaNorm

## geneBlockCount integer overflow

`geneBlockCount()` (R/gpuFunctions.R) does `as.integer(ceiling(needed.bytes / budget.bytes))`
before clamping to `n_genes`. When the ratio exceeds 2^31 (for example a 1-byte budget on a
large problem) `as.integer` yields `NA` with a warning and the function returns `NA`.
`spanormpy._backend.gene_block_count` clamps in floating point and returns `n_genes` in that case.
The two agree everywhere R returns a value.
