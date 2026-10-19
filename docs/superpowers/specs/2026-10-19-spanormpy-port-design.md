# spanormpy: Python port of SpaNorm — design

Date: 2026-10-19
Status: draft for approval

## Intent

SpaNorm (R/Bioconductor, `../SpaNorm`) normalises spatial transcriptomics
counts by modelling library size and biology as smooth spatial functions in a
gene- and spot-specific negative binomial (NB) model. Python users in the
scverse ecosystem (scanpy, squidpy, spatialdata) cannot use it today.
`spanormpy` is a faithful Python port of SpaNorm master that fits naturally into scverse
workflows and will be released publicly on PyPI under the name `spanormpy`
(repo: https://github.com/BhuvaLab/spanormpy).

The overriding requirement is **numerical fidelity to the R package**: a
user should get the same normalised values, fits and SVG statistics from
either implementation. The second requirement is **feature completeness**:
everything SpaNorm `master` ships, spanormpy ships. Users run SpaNorm on
GPUs, so the GPU path is first-class, not an afterthought.

Credits are limited, so the work is organised for efficient execution:
Opus plans, Sonnet subagents execute well-specified tasks, each with an
objective pass/fail check against R reference outputs.

## Source of truth

- SpaNorm **`master` at commit `c12e5e5`** (DESCRIPTION version 1.7.15, 2026-09-30; includes `tpsBasis` and `nbBlockDesign`).
  Unmerged branches (`fix/polish-collinear-block`,
  `fix/technical-lambda-scaling`, anything else) are **out of scope**.
- Read R sources with `git -C ../SpaNorm show master:<path>`; the working
  tree of `../SpaNorm` is on a different branch and must not be changed.
- R reference outputs are generated from that commit installed into a
  **separate R library** (e.g. `spanormpy/tests/ref/Rlib`, git-ignored),
  using the existing `latest-r` conda env (edgeR 4.8.0, scran 1.38.0). The
  user's installed SpaNorm (1.7.10) must not be touched.

## Goals

1. Port every user-facing feature of SpaNorm master:
   - `SpaNorm` — model fit + adjustment, all `adj.method`s (logpac,
     pearson, meanbio, medbio) and gene models, batch support, `df.tps`
     as scalar or length-2, `sample.p` subsampling, `lambda.a`, reuse of a
     stored fit (with the `overwrite` semantics and the df.tps/lambda
     match checks), blocked normalisation for large data.
   - `SpaNormSVG` / `topSVGs` — technical (null) fit, SVG test, paired-null
     / polished-alike logic.
   - `SpaNormPCA` — deviance/pearson residual PCA.
   - Polishing: `polishNB`, `polishSpaNorm` (incl. joint variant and model
     helpers), `isPolished`, `nbProfilePsi`, hold-out / rank-deficiency
     checks.
   - Quasi-likelihood: `qlDispersion`, `qlMomentTable`, `nbUnitDeviance`,
     `nbDevianceMoments`.
   - Solvers and design: `fitNB`, `nbNewtonSolver`, `nbNewtonSolverBatch`,
     `nbGramBatch`, `nbAbsorbGramBatch`, `nbMuFloor`, `nbBlockDesign`,
     `tpsBasis`, `calculateMu`.
   - Utilities: `fastSizeFactors`, `filterGenes`, `hasBadValues`.
   - Backend control: `checkGPU`, `getBackendDevice`, `getBackendDtype`,
     `get/setGPUMemoryBudget`, `resetGPUCache`, `gpuDtypeBytes`.
   - Plotting: `plotSpatial`, `plotCovariate`.
2. Numerical agreement with R on shared inputs (see Accuracy).
3. GPU support via PyTorch (CUDA), float64 like the R torch backend
   (float32 only on MPS, as in R).
4. scverse-native API and data layout (below).
5. Port the R test suite's intent (23 testthat files) plus R-parity tests.
6. Release-ready packaging (not released yet).

## Non-goals

- Seurat and SpatialFeatureExperiment support (AnnData replaces them).
- Reading or writing R `.rds` `SpaNormFit` objects.
- Features on unmerged R branches.
- Publishing to PyPI / scverse listing / pushing to GitHub in this phase
  (only on the user's explicit say-so).
- Speed-ups or algorithmic changes that alter results relative to R. If a
  bug is found in R, record it (`docs/r-discrepancies.md`) and match R
  anyway, unless the user decides otherwise.

## Anti-goals (failure even if it "works")

- Results that quietly differ from R beyond tolerance.
- Per-gene or per-spot fit data in `adata.uns`, which breaks when the
  AnnData is subset.
- A GPU path that is slower than or less accurate than the CPU path, or
  that only works on CPU in practice.
- Tests that only check "it runs" for algorithms that have an R reference.

## scverse API and data layout

Input: an `AnnData` with raw counts in `adata.X` or a named layer
(`layer=` argument), spatial coordinates in `adata.obsm["spatial"]` (the
squidpy/scanpy convention; `spatial_key=` to override). A `SpatialData`
object is accepted by taking its table (`table_key=`) and writing results
back to it. Batch is an `obs` column name (`batch_key=`).

Function names follow Python convention, mirroring R names so R users can
find them: e.g. `spanormpy.spanorm(adata, ...)` (R `SpaNorm`),
`spanormpy.spanorm_svg`, `spanormpy.top_svgs`, `spanormpy.spanorm_pca`,
`spanormpy.polish_spanorm`, `spanormpy.pl.plot_spatial`,
`spanormpy.pl.plot_covariate`; low-level solvers under a submodule (exact
module split is the builder's call). Argument names are the R names in
snake_case (`sample_p`, `df_tps`, `lambda_a`, `adj_method`, `gene_model`,
`scale_factor`, `maxit_nb`, `maxit_psi`, `backend`, `overwrite`, ...),
with the same defaults. A module docstring / docs page tabulates R → Python
names.

Storage (user decision: anything per gene or per spot lives in axis-aligned
slots so subsetting stays valid without validity checks):

| SpaNormFit slot / output | AnnData location |
|---|---|
| `alpha` (genes × covariates) | `varm["spanorm_alpha"]` |
| `gmean`, `psi` | `var["spanorm_gmean"]`, `var["spanorm_psi"]` |
| `W` (spots × covariates) | `obsm["spanorm_W"]` |
| `sampling` (per spot) | `obs["spanorm_sampling"]` |
| size factors | `obs["size_factors"]` (scanpy-compatible name; builder may align with R `sizeFactor`) |
| normalised values (logPAC etc.) | `layers["logcounts"]` (R assay name), adj. method recorded in `uns` |
| technical/null fit (SVG) | same scheme with `spanorm_null_` prefix |
| SVG statistics | `var` columns |
| polish results | per-gene → `var`/`varm`, per-spot → `obs`/`obsm` |
| gene×gene quantities, if any | `varp` |
| scalars only: `ngenes`, `ncells`, `gene.model`, `df.tps`, `sample.p`, `lambda.a`, `wtype` (column labels of W/alpha), `loglik` trace, batch levels, polish settings | `uns["spanorm"]` |

Because `uns` holds no per-gene/per-spot data, `adata[:, genes]` and
`adata[spots, :]` keep the fit consistent. `ngenes`/`ncells` in `uns` are
informational and must not be used for validation. Reuse of a stored fit
after subsetting must behave as R does after `subsetFitGenes`.

Parallelism and scale: R's `BPPARAM` becomes an `n_jobs` argument (thread
or process pool); disk-backed `DelayedArray` support becomes block-wise
processing over sparse (`scipy.sparse`) and backed/dask arrays, bounding
peak memory the same way `normaliseBlocked` does.

Plotting: matplotlib in scanpy/squidpy style, returning figures/axes; the
tissue image comes from `adata.uns["spatial"]` (scanpy Visium convention).
Visual parity with ggplot output is not required — same content is.

## Backend (my call)

A single implementation written against PyTorch tensors serves both
`backend="cpu"` (torch CPU, float64) and `backend="gpu"` (CUDA/MPS),
replacing R's split between base-R CPU code and torch GPU code; `"auto"`
picks GPU when available, as in R. torch is a required dependency. The GPU
memory budget, gene-blocking (`geneBlockCount`) and cache-reset semantics
of `R/gpuFunctions.R` are reproduced; R's thin wrappers (`matmul_gpu`,
`toGPUMatrix`, ...) become internal helpers rather than public API, since
in Python they are one-line torch calls. If the builder finds the R CPU and
GPU paths are numerically different algorithms (not just different
devices), both must be preserved.

## External R dependencies that must be ported

- **`edgeR::estimateDisp(..., tagwise = TRUE, robust = TRUE)`** with an
  intercept-only design and offsets (`R/fitSpaNormNB.R`). This is
  deterministic and on the main path: port exactly the code path SpaNorm
  exercises (adjusted profile likelihood, weighted likelihood empirical
  Bayes, trend, robust prior handling) and match R to tight tolerance.
  Existing Python ports (e.g. inmoose/edgepy) may be used only if they
  pass the parity tests; otherwise port from edgeR 4.8.0 source.
- **`scran::calculateSumFactors(Y, clusters = cl)`** — deterministic given
  clusters; port and match exactly.
- **`scran::quickCluster(Y)`** — involves approximate PCA and graph
  clustering; exact match is not expected (user agreed). Reimplement the
  same recipe (log-normalise, PCA, SNN graph, walktrap, `min.size`) and
  verify closeness (high ARI with R clusters and size factors correlated
  > 0.99 on the reference data).
- `matrixStats` row/col medians/MADs, `stats::dnbinom/qnbinom/pnbinom`,
  `optimize`, `approx`, `p.adjust`, `pf`, `model.matrix` — use scipy/numpy
  equivalents; verify where R semantics differ (MAD constant 1.4826,
  `optimize`'s golden-section tolerance, `approx` rules, quantile types).

## Accuracy

- R reference generation: R scripts in `tests/ref/` run SpaNorm master on
  fixed inputs (HumanDLPFC subset from `../SpaNorm/data/HumanDLPFC.rda`
  plus small simulated cases mirroring the testthat tests) and write
  inputs and outputs to files Python can read (h5ad/npz). Reference files
  are committed if small; a script regenerates them.
- Parity tests compare Python to R per component (basis, design matrix,
  size factors, dispersions, alpha, mu, each adj. method, SVG stats, PCA,
  polish outputs, QL dispersions).
- Tolerances (my call; tighten where possible): deterministic closed-form
  pieces `rtol ≤ 1e-10`; iterative fits with identical inputs and
  stopping rules `rtol ≤ 1e-6` on fitted values and normalised data;
  GPU float64 same as CPU; MPS float32 looser and documented. Any looser
  tolerance must be justified in the test.
- Subsampling (`sample.p`) uses R's RNG in R; to compare, the reference
  scripts export the sampled indices and the Python tests inject them.
  Python's own sampling uses numpy's RNG with a `random_state` argument.
- GPU tests run as SLURM GPU jobs on Bunya; they are skipped when no GPU is
  present.

## Constraints

- Python ≥ 3.11 (my call); env: a new conda env `spanormpy` with PyTorch +
  CUDA, anndata, scanpy, scipy, numpy, matplotlib, python-igraph,
  spatialdata (optional extra).
- scverse cookiecutter-style layout: `pyproject.toml` (hatch), `src/`
  layout, pytest, pre-commit/ruff, Sphinx docs skeleton, GPL-3.0-or-later
  (matches R).
- Port the vignette as a tutorial notebook.
- Commits go to the local repo; nothing is pushed without the user's OK.

## Execution approach

- Opus writes the implementation plan (task-sized, ordered by dependency:
  scaffolding → R reference harness → basis/design → solvers → edgeR/scran
  ports → core fit → adjustment → SVG → PCA → polish/QL → backend/GPU
  blocking → AnnData/SpatialData glue → plotting → docs).
- Sonnet subagents execute tasks; each task names the R source file(s) to
  port and the parity test that defines done.
- Opus is used only for planning and resolving failures Sonnet cannot.

## Decisions settled after review (my calls)

- **Precomputed size factors.** R uses `sizeFactors(spe)` when set. Python
  takes an explicit `size_factors=` argument (array, or the name of an
  `obs` column). It never reads a `size_factors` column implicitly, because
  other scanpy tools write columns with that name. Passing them is mainly
  for comparison with R.
- **Default size factors (user decision).** spanormpy has its own built-in
  size factor computation that works like scran: quickCluster-style
  clustering followed by pooled deconvolution sum factors. It runs in pure
  Python with no R needed, and it is the default whenever `size_factors=`
  is not given. It is also a public function (e.g.
  `spanormpy.compute_size_factors(adata, clusters=None, ...)`) so users
  can run it on its own or pass their own clusters. Results go to
  `obs["spanorm_size_factors"]`. Given R's clusters, it must match
  `scran::calculateSumFactors` exactly; with its own clustering, it must
  meet the "close" check below.
- **Reference backend.** R references come from R's CPU path. Python's CPU
  and GPU paths (both float64) are tested against those references with the
  same tolerances. If R's GPU path turns out to be a different algorithm
  (not just a different device), the builder adds R GPU references from a
  Bunya GPU job, provided R torch with CUDA works there. Otherwise the
  builder reports the problem.
- **End-to-end parity.** Strict parity (tolerances above) is required with
  R's size factors injected. On the full default pipeline, where Python
  clusters on its own, the check is "close": size factors and logPAC
  values correlated > 0.99 per gene, and SVG rankings with
  Spearman > 0.95 (thresholds may be tightened, not loosened, by the
  builder). No port of irlba or R's RNG.
- **GPU speed.** There is one benchmark task on the full HumanDLPFC data:
  GPU vs CPU (torch CPU, all allocated cores) on a Bunya GPU node, with the
  result recorded. Pass means GPU wall time below CPU for `spanorm` and
  `spanorm_svg`. `"auto"` chooses the device the same way R does, with no
  extra size-based fallback.

## Left to the builder

Module layout, internal helper names, choice of h5ad vs npz for
references, exact quickCluster closeness thresholds within the bounds
above, docs tooling details, how `n_jobs` is implemented.

## Addendum (2026-10-19): one documentation website for SpaNorm and spanormpy

**User decisions:** a single Quarto website serves both packages. It replaces SpaNorm's pkgdown site, lives in the SpaNorm repo and deploys to the same GitHub Pages URL (bhuvad.github.io/spaNorm). Code examples sit in R | Python tabs (`panel-tabset group="language"`), so picking one language switches every tab on the site and the choice is remembered.

**Design (my calls):**
- **Location:** `site/` in the SpaNorm repo on a branch `docs/quarto-site`, created from `master` in a separate worktree at `../SpaNorm-site`. The user's own SpaNorm checkout is never touched. `site/` goes in `.Rbuildignore`, and Bioconductor vignettes are unchanged.
- **Pages:**
  - Home: what SpaNorm is, the workflow figure, install instructions for R (Bioconductor/GitHub) and Python (pip from GitHub, CUDA torch note), citation.
  - Get started: the SpaNorm vignette, with every step in R | Python tabs.
  - Articles: SVGs and PCA; polishing; large data and GPU (the benchmark).
  - Reference: separate R and Python sections. R pages come from the Rd files via `altdoc`, installed into a project-local R library rather than the user's. Python pages come from `quartodoc`.
  - R vs Python: the name table plus intentional differences, taken from spanormpy's `docs/r-to-python.md` and `docs/r-discrepancies.md`.
  - News from NEWS.md, and GitHub links to both repos.
- **Look:** Bootstrap 5, as SpaNorm's pkgdown uses, with a navbar and sidebar layout similar to pkgdown.
- **Execution:** R chunks run with knitr and Python chunks with reticulate, using the spanormpy env. Outputs are stored with `freeze: auto` and the `_freeze/` directory is committed, so CI only renders.
- **CI:** a new `.github/workflows/quarto-site.yaml` replaces `pkgdown.yaml`. On pushes to master it installs R + SpaNorm + altdoc and Python + spanormpy (git) + quartodoc, renders, and deploys to `gh-pages`.
- **Release gate:** nothing is merged into SpaNorm master and nothing is deployed until the user says so. Pushing the `docs/quarto-site` branch is fine.
