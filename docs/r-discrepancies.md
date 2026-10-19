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

## normalise*: a single NaN dispersion turns logPAC, Pearson and medbio into all-NaN (R behaviour, reproduced)

`winsorisePsi` caps dispersions at `exp(median(log psi) + 4 * mad(log psi))`. A NaN in `psi` makes the median and
MAD NaN, so `psi.max` is NaN and `pmin(psi, psi.max)` is NaN for every gene: `normaliseLogPAC`,
`normalisePearson` and `normaliseMedianBio` return NaN for the whole matrix, while `normaliseMeanBio` (which
does not use `psi`) stays finite. A zero `psi` is harmless (`log(0) = -Inf` is absorbed by the median/MAD and
the outputs stay finite), and R emits no warning. `winsorise_psi` and `normalise_*` reproduce this exactly
(pinned by `test_degenerate_psi_matches_r`).

## Batch factor levels: locale collation

R's `factor()` sorts character batch labels with the session locale (`en_US.UTF-8`:
`a < A < b < B`, punctuation < digits < letters); pandas and numpy sort by codepoint
(`A < B < a < b`). `spanormpy._io.check_batch` sorts strings with an approximation of the
glibc `en_US.UTF-8` collation (punctuation < digits < letters, case-insensitive first, lower
case before upper case on ties). Exotic characters (accents, non-Latin scripts) may still
order differently from R. Which level is the reference only changes the parametrisation of
the batch columns, not the fitted values or the normalised data. For full control pass a
`pandas.Categorical`, whose category order is used as given. Numeric vectors are one
covariate column and booleans are factors, as in `model.matrix`. A single-level batch raises
R's "contrasts can be applied only to factors with 2 or more levels"; unused categorical
levels give all-zero columns, as in R.

## Adjustment backend

R's `SpaNorm()` calls the `normalise*` functions without a backend (they use the accelerator
whenever one is available). `spanorm(backend=...)` passes the user's `backend` to the
adjustment too, so `backend="cpu"` really keeps everything on the CPU.

## Output sparsity

R always stores `logcounts` as a sparse matrix. `spanorm` writes a dense float64 layer for
dense input and a CSR layer for sparse input; values are identical.

## Default pipeline (own size factors and sampling)

Python's `quick_cluster` and RNG differ from R's, so the default pipeline is checked for
closeness only (see `tests/test_spanorm.py::test_default_pipeline_close`).

## Rerun after a gene or spot subset

R keeps `metadata(spe)$SpaNorm` when an SPE is subset, and `.spaNormCore` refits because
`fit$ngenes != nrow(emat)` (genes) or `fit$ncells != ncol(emat)` (spots). `spanorm` on AnnData
differs for genes only: per-gene fit data live in `var`/`varm`, so a gene subset carries its
slice of the fit and is **reused** (with the whole fit's `psi_max`, as R's `subsetFitGenes`), which
equals R's adjustment of the full fit on those genes. R's refit behaviour is reachable with
`overwrite=True` (pinned against R in `tests/test_api_robustness.py`). A spot subset **refits**,
as R does (detected through the `ncells` stored in `uns["spanorm"]["spanorm"]`).

## Degenerate inputs

An all-zero gene and a zero-total spot with supplied size factors are handled by R (finite
logcounts, size factor floored at 1e-8) and identically here. Without supplied size factors R's
`calculateSumFactors` fails on a zero-total spot ("size factors should be positive"); so does
`spanorm`. A batch factor with an unused level gives an all-zero unregularised column and a
singular system in both (R: `dgesv ... exactly singular`; here torch's `LinAlgError`, a different
exception type).

## qnbinom in the far upper tail (CPU)

R >= 4.2 fuzzes `qnbinom` much less than the older `64*eps` when the tail probability is tiny:
at `p = 1 - 1e-10` the 64*eps fuzz (1.4e-14) is ~1e-4 of the tail and moves the quantile by
2-3 counts. `qnbinom` on CPU now searches the survival function with `sf(k) <= (1-p) + 8*eps*p`
when `1 - p < 1e-6`, and keeps the `64*eps` CDF fuzz elsewhere (the CDF R feeds back through `p`
carries errors up to ~1e-14 that only the larger fuzz absorbs). This is needed for
`nb_deviance_moments` to reproduce R's summation window to 1e-10. At `1 - p ~ 1e-13` R's own CDF is
quantised at 1.1e-16 and its quantile can differ from the exact one by a few counts; not reproducible.

## polish_spanorm after a gene or spot subset

`polishSpaNorm` calls `getSpaNormFit`, which in R errors when `fit$ngenes`/`fit$ncells` differ from
the object. `polish_spanorm` follows R for **spot** subsets (detected through the `ncells` stored in
`uns["spanorm"][name]`, `_fit.fit_is_stale_for_spots`): it raises "... model dimensions do not match
the data. Please rerun 'spanorm'." (or `'spanorm_svg'` for the null fit), also for a gene plus spot
subset, and writes nothing. A **gene** subset reuses the stored fits (see "Rerun after a gene or spot
subset"): the genes are polished independently, so the result equals the full run's genes. The
polished subset fit keeps the whole fit's `psi_max` (threshold from the unpolished dispersions),
because the other genes' polished dispersions are not available. A spot reorder keeps the count and
reuses the fit. `get_spanorm_fit(validate=True)` raises the same error.

## polish_spanorm bookkeeping

`adata.uns["spanorm"]["adj_method"]` is updated to the method used to rewrite `logcounts`. SVG
columns removed on polishing are `svg_F`, `svg_p`, `svg_fdr`, `svg_F_raw` (R: `svg.F`, `svg.p`,
`svg.fdr`). The polish settings key `SpaNorm` (R package version) is `spanorm` (spanormpy version).
Per-gene `held_out` is stored as a categorical `var` column (h5ad cannot hold missing strings in an
object column). R's Seurat/DelayedArray branches and the pre-`polish`-slot RDS fixture have no
Python counterpart.

## spanorm_svg bookkeeping and subsets

`SpaNormSVG` writes `svg.F`, `svg.p`, `svg.fdr` and returns the unclamped statistic only as an attribute
of `svgTest()`'s table, which `rowData<-` drops; `spanorm_svg` keeps it as `var["svg_F_raw"]`
(`svg_test(...).attrs["F_raw"]` for the table). The "SVG results exist" warning fires when all three of
`svg_F`, `svg_p`, `svg_fdr` are present, and all four columns are replaced. Like `polish_spanorm`, a
**spot** subset of an object carrying a fit is an error ("... model dimensions do not match ..."), and a
**gene** subset reuses the stored full and null fits (the null of a gene subset keeps the whole fit's
dispersion-winsorisation threshold), so the subset's statistics equal the full run's rows. A null that
has to be fitted on a gene subset winsorises with the subset's own dispersions. `spanorm_svg` gained a
`random_state` argument (seed of the null's dispersion subsampling, R: the global RNG). The paired
null reuses `polish_spanorm_fit`, with the full fit's recorded `psi_method`, `ls`, `cells`, `maxit`,
`tol` and `psi_range`, on the CPU backend (R does not pass `backend` to the polish either).
`top_svgs` does not run `checkSPE` (R validates the counts); it only needs the SVG columns. R's
`topSVGs` keeps rows whose FDR is `NA` when filtering (`NA <= fdr` subsets to `NA` rows); `top_svgs`
drops them.

## top_svgs and NaN FDRs

R's `topSVGs` subsets with `results[results$svg.fdr <= fdr, ]`; a `NA` FDR makes that an all-NA row
(named `NA`, `NA.1`, ...), which `order()` puts last and `min(n, nrow)` counts. This is an indexing
artefact, but `top_svgs` reproduces the row count and order: genes with a NaN FDR are returned after
the sorted finite rows, in gene order, counted toward `n`, with all statistics NaN. Only the names
differ: they keep their real gene names instead of R's `NA` junk names. `top_svgs` also takes `layer`
and runs the counts check R's `checkSPE` does (it supersedes the earlier note that counts are not checked).

## SpaNormPCA with NaN-FDR genes

`topSVGs` returns all-NA rows named `NA`, `NA.1`, ... for genes whose FDR is `NaN` (see above), and R's
`getResiduals(...)[svgs, ]` then fails with "subscript out of bounds". `spanorm_pca` raises a `ValueError`
in the same situation (use a smaller `nsvgs` or set the FDRs). `percentVar`/`variance` cover only the
`ncomponents` computed components (as in R), so `percent_var` sums to 100 over them, not over all components.

`spanorm_pca(n_jobs=...)` stands in for R's `BPPARAM` but has no effect (the decomposition uses BLAS threads).
`random_state` seeds the optional randomised SVD (R's `BSPARAM` algorithms have their own RNG use).
