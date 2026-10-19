# spanormpy Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A faithful, GPU-capable Python port of SpaNorm master (`c12e5e5`) for scverse, numerically matching R.

**Architecture:** Internal algorithms are line-by-line ports of the R functions, kept in R's orientation (Y is genes × cells) and written once against PyTorch tensors, so one code path serves CPU and CUDA. A thin glue layer reads from and writes to AnnData/SpatialData. Every algorithmic task is gated by a parity test against reference outputs that R scripts generate from SpaNorm master.

**Tech Stack:** Python ≥ 3.11, PyTorch, numpy, scipy, anndata, scanpy (tests/tutorial), python-igraph, matplotlib, h5py, pytest; optional spatialdata. R 4.5 (`latest-r` env) with rhdf5 for references.

**Spec:** `docs/superpowers/specs/2026-10-19-spanormpy-port-design.md`. Read it before any task. It decides layout, tolerances and scope, and this plan does not restate it.

## Global Constraints

- Source of truth: `git -C ../SpaNorm show master:<path>` at commit `c12e5e5` (version 1.7.15). Never modify `../SpaNorm` or the user's installed R packages.
- R references come from SpaNorm master installed into `tests/ref/Rlib/` (git-ignored), called with `Rscript` from `~/miniconda3/envs/latest-r/bin`.
- Python env: conda env `spanormpy` (Python 3.12, `torch` with CUDA 12.x wheels). All commands below assume `conda activate spanormpy`.
- Internal orientation: every ported algorithm takes and returns arrays in R's orientation (genes × cells). Only `_io.py` and the storage layer transpose to AnnData's cells × genes.
- Default dtype is float64 on CPU and CUDA, and float32 only on MPS (R `getBackendDtype`).
- Tolerances: deterministic closed-form `rtol=1e-10, atol=1e-12`. Iterative fits given identical inputs `rtol=1e-6`. GPU float64 uses the same values as CPU. Any looser tolerance carries a comment justifying it.
- R argument names become snake_case with the same defaults (`sample.p` → `sample_p`). R's `BPPARAM` becomes `n_jobs: int = 1`. `backend` takes `"auto" | "cpu" | "gpu"`, with each R function's own default.
- AnnData storage: per-gene data goes in `var`/`varm`/`varp`, per-spot data in `obs`/`obsm`, and only scalars and labels go in `uns["spanorm"]` (spec table). Fit name prefixes map R object names: `SpaNorm` → `spanorm_`, `SpaNormNull` → `spanorm_null_`, `SpaNormUnpolished` → `spanorm_unpolished_`, `SpaNormNullUnpolished` → `spanorm_null_unpolished_`.
- Normalised output goes to `layers["logcounts"]`. Size factors computed by spanormpy go to `obs["spanorm_size_factors"]`. `obs["size_factors"]` is never read implicitly.
- Where R has a bug, match R and log it in `docs/r-discrepancies.md`.
- Commit after every task with a message ending `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Never push.
- Each task also ports the assertions of the R testthat file(s) named in its **R tests** line (behaviour, errors and warnings), skipping Seurat- and SPE-class-specific ones.

## Review Focus

1. **Sparse counts** (`scipy.sparse` CSR/CSC in `X` or a layer, int or float32 dtype) must give results identical to the same matrix dense. Test owner: Task 11.
2. **AnnData views and subsets after fitting** (`adata[:, genes]`, `adata[spots]`). Rerunning `spanorm`, `spanorm_svg` or `polish_spanorm` must reuse the stored fit exactly as R does after `subsetFitGenes`, and must not raise `ImplicitModificationWarning` or write into a view. Test owners: Tasks 10–11.
3. **Coordinates in non-standard form**: `obsm["spatial"]` as an int array, as a DataFrame, or with more than 2 columns. Use the first two columns as floats, and raise a clear error for fewer than 2 or for non-finite values. Test owner: Task 11.
4. **Batch column edge cases**: categorical with unused levels, a single level, NaN entries, and numeric covariate matrices. Behaviour must match R `checkBatch`. Test owner: Task 11.
5. **Genes with all-zero counts or spots with zero total counts** must follow R's behaviour: the same error, or the same `pmax(1e-08, ·)` flooring. They must never produce silent NaN in `logcounts`. Test owners: Tasks 6 and 11.

---

## File Structure

```
pyproject.toml, README.md, LICENSE (GPL-3.0-or-later), .gitignore, .pre-commit-config.yaml
docs/ (Sphinx skeleton, r-to-python.md, r-discrepancies.md, tutorial notebook)
src/spanormpy/
  __init__.py            public API re-exports
  _backend.py            gpuFunctions.R: device/dtype, memory budget, gene blocking, tensor helpers
  _nbdist.py             dnbinom/pnbinom/qnbinom (torch), nb log-likelihood
  _tps.py                tpsBasis.R + bs.tps (mainSpaNorm.R)
  _block_design.py       nbBlockDesign.R
  _solver.py             nbSolver.R + nbSolverBatch.R
  _edger.py              port of edgeR::estimateDisp path used by SpaNorm
  size_factors.py        scran sum factors + quickCluster, fastSizeFactors
  _filter.py             filterGenes
  _fitnb.py              fitSpaNormNB.R (fitNB, fitSpaNormNB, fitNBGivenPsi, calculateMu, winsorise*)
  _adjust.py             normalise* adjustments + normaliseBlocked
  _fit.py                SpaNormFit dataclass + AnnData store/load + subsetFitGenes semantics
  _io.py                 resolve counts/coords/batch from AnnData or SpatialData; checkBatch
  spanorm.py             SpaNorm()/.spaNormCore/fitSpaNorm/sampleRandom/match* checks
  _quasilik.py           quasilik.R
  _polish_engine.py      polishEngine.R
  _polish_batch.py       polishEngineBatch.R
  polish.py              polishNB.R (polish_nb, nb_profile_psi)
  _polish_model.py       polishSpaNormModel.R + polishSpaNormJoint.R
  polish_spanorm.py      polishSpaNorm.R
  svg.py                 mainSpaNormSVG.R
  pca.py                 mainSpaNormPCA.R
  pl/__init__.py, pl/_spatial.py, pl/_covariate.py
tests/
  ref/install_spanorm.sh, ref/_common.R, ref/make_<topic>.R, ref/make_all.sh, ref/data/<topic>.h5
  _ref.py                load_ref(topic) helper
  conftest.py            fixtures: dlpfc_small (AnnData), devices param (cpu, cuda-if-available)
  test_<topic>.py
scripts/slurm_gpu_tests.sh, scripts/benchmark_gpu.py
```

---

### Task 1: Scaffold, environment and R reference harness

**Files:**
- Create: `pyproject.toml`, `.gitignore` (adds `tests/ref/Rlib/`, `__pycache__`, `.pytest_cache`), `LICENSE`, `README.md`, `.pre-commit-config.yaml` (ruff), `src/spanormpy/__init__.py` (`__version__ = "0.1.0.dev0"`), `docs/` Sphinx skeleton (`docs/conf.py`, `docs/index.md`), `docs/r-discrepancies.md` (heading only)
- Create: `tests/ref/install_spanorm.sh`, `tests/ref/_common.R`, `tests/ref/make_all.sh`, `tests/ref/make_dlpfc.R`, `tests/_ref.py`, `tests/conftest.py`, `tests/test_harness.py`

**Interfaces:**
- Produces: `tests/_ref.py::load_ref(topic: str) -> dict[str, np.ndarray | str | float]`. It reads `tests/ref/data/<topic>.h5`, undoes rhdf5's column-major transposition so arrays have R's shape, and returns scalars as Python scalars and character vectors as `np.ndarray[str]`. R `NA` becomes `nan`.
- Produces: `tests/ref/_common.R` defining `library(SpaNorm, lib.loc = "tests/ref/Rlib")`, `write_ref(x: named list, topic)` (writes each element, with nested lists as groups, logical as int8, and factors as character plus a `levels` attribute), and `sim_counts(ngenes, ncells, seed)` (a deterministic NB simulation on a grid with a spatial and a library-size effect, used by many tasks).
- Produces: `tests/ref/data/dlpfc_small.h5ad`, written by `make_dlpfc.R` via zellkonverter. It holds `HumanDLPFC` from `../SpaNorm/data/HumanDLPFC.rda`, filtered with `filterGenes(prop = 0.2)`, the first 300 genes by row order, all spots, counts in `X` (cells × genes), coordinates in `obsm["spatial"]`, and the `AnnotatedCluster` column in `obs`.
- Produces: `conftest.py` fixtures `dlpfc_small` (fresh `anndata.read_h5ad` copy per test) and `device` (parametrised over `"cpu"` and, when `torch.cuda.is_available()`, `"cuda"`), plus marker `gpu`.

- [ ] **Step 1:** Create the env: `conda create -n spanormpy python=3.12 -y`, then `pip install -e ".[dev,spatial]"` once `pyproject.toml` exists. `pyproject.toml` uses hatchling and has runtime deps `numpy>=1.26, scipy>=1.11, torch>=2.3, anndata>=0.10, pandas, python-igraph>=0.11, matplotlib>=3.8, h5py`, extra `spatial = ["spatialdata>=0.2"]`, and extra `dev = ["pytest", "scanpy>=1.10", "ruff", "pre-commit"]`.
- [ ] **Step 2:** `install_spanorm.sh` runs `git -C ../SpaNorm archive c12e5e5 | tar -x -C $TMP` and then `R CMD INSTALL --library=tests/ref/Rlib $TMP`. Run it. Expected: `Rscript -e 'library(SpaNorm, lib.loc="tests/ref/Rlib"); packageVersion("SpaNorm")'` prints `1.7.15`.
- [ ] **Step 3:** Write `tests/test_harness.py::test_roundtrip`. `make_all.sh` runs every `tests/ref/make_*.R`, and a `make_harness.R` writes `list(m = matrix(1:6 + 0.5, 2, 3), v = c(1, NA), s = c("a","b"), x = 3.25, f = factor(c("u","v","u")))`. Assert `ref["m"].shape == (2, 3)`, `ref["m"][1, 2] == 6.5`, `np.isnan(ref["v"][1])`, `list(ref["s"]) == ["a", "b"]`, `ref["x"] == 3.25`, and `list(ref["f"]) == ["u", "v", "u"]`.
- [ ] **Step 4:** Implement `_common.R`, `_ref.py` and `make_dlpfc.R`. Run `bash tests/ref/make_all.sh && pytest tests/test_harness.py -v`. Expected: PASS, with `dlpfc_small.h5ad` present.
- [ ] **Step 5:** Commit (`chore: scaffold package, env and R reference harness`).

### Task 2: Backend, NB distribution helpers, winsorisation

**R sources:** `R/gpuFunctions.R`; `winsorisePsi`, `.checkWinsor` and `calculateMu` signatures from `R/fitSpaNormNB.R`. **R tests:** `test-gpuFunctions.R`.

**Files:** Create `src/spanormpy/_backend.py`, `src/spanormpy/_nbdist.py`, `tests/ref/make_backend.R`, `tests/test_backend.py`

**Interfaces:**
- Produces (`_backend.py`): `get_backend_device() -> str` (`"cuda"|"mps"|"cpu"`), `get_backend_dtype() -> torch.dtype`, `gpu_dtype_bytes() -> int`, `check_gpu() -> bool`, `set_gpu_memory_budget(bytes: int | None) -> None`, `get_gpu_memory_budget(gpu_mem_budget: int | None = None) -> int`, `reset_gpu_cache() -> None`, `resolve_device(backend: str) -> torch.device` (R's `"auto"` and `"gpu"` resolution, including the fallback to CPU when no GPU is available), `gene_block_count(n_genes, n_cells, n_cov, backend="auto", budget_bytes=None, tensor_multiplier=10) -> int`, `gene_block_indices(n_genes, nblocks) -> list[np.ndarray]` (0-based), `as_tensor(x, device, dtype=None) -> torch.Tensor`, `winsorise_cols(alpha, k=4)`, `has_bad_values(x) -> bool`. Constants: `DEFAULT_WINSOR = 4` and `GPU_BLOCK_TENSOR_MULTIPLIER = 10`.
- The CUDA free-memory budget uses `torch.cuda.mem_get_info()` in place of parsing nvidia-smi. Keep R's safety fractions and its MPS rule.
- Produces (`_nbdist.py`): `dnbinom(x, mu, size, log=False)`, `pnbinom(q, mu, size, log=False)` and `qnbinom(p, mu, size)`, all tensor-in/tensor-out on any device with R semantics (including `size=Inf` → Poisson and `mu=0`). Also `winsorise_psi(psi, winsor=4)`.

- [ ] **Step 1: Write the failing tests.** `make_backend.R` writes `geneBlockCount` for a grid of `(n_genes, n_cells, n_cov, budget)` values and the corresponding `geneBlockIndices`. It also writes `dnbinom`/`pnbinom`/`qnbinom` on a grid of x ∈ {0,1,5,50,1000}, mu ∈ {1e-8, 0.5, 3, 200}, size ∈ {0.01, 1, 10, 1e6, Inf}, and `winsorisePsi` / `winsoriseCols` on a fixed random input (`set.seed(1)`). Tests: `test_gene_block_count_matches_r` (exact ints), `test_gene_block_indices_matches_r` (R indices minus 1), `test_dnbinom_matches_r[device]` (log and non-log, `rtol=1e-10`), `test_pnbinom_qnbinom_match_r[device]`, `test_winsorise_matches_r`, `test_has_bad_values` (True for nan/inf, False otherwise), `test_auto_backend_falls_back_to_cpu` (monkeypatch `torch.cuda.is_available` to return False, then `resolve_device("gpu").type == "cpu"`), and `test_set_get_budget_roundtrip` plus the invalid-budget errors from `test-gpuFunctions.R`.
- [ ] **Step 2:** `pytest tests/test_backend.py -v`. Expected: FAIL (import errors).
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** `pytest tests/test_backend.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 3: Thin-plate spline basis

**R sources:** `R/tpsBasis.R`; `bs.tps` in `R/mainSpaNorm.R`. **R tests:** `test-tpsBasis.R` (from master).

**Files:** Create `src/spanormpy/_tps.py`, `tests/ref/make_tps.R`, `tests/test_tps.py`

**Interfaces:**
- Produces: `tps_basis(x, y, df=6, ref_x=None, ref_y=None, center=True) -> np.ndarray` (cells × basis, float64, carrying the same column order as R). Also `bs_tps(x, y, df_tps=6) -> tuple[np.ndarray, int]`, which returns the basis and the effective `df.tps` (R's `attr(, "df.tps")`).

- [ ] **Step 1:** `make_tps.R` writes `tpsBasis` and `bs.tps` outputs on the `dlpfc_small` coordinates for `df ∈ {2, 6, 10}`, with and without `center`, and with a `ref.x/ref.y` subset. Tests: `test_tps_basis_matches_r` (`rtol=1e-10`), `test_bs_tps_matches_r` (basis and df), plus the testthat error cases.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with numpy/scipy. Match R's linear-algebra choices (for example `qr` vs `svd`) so that the basis matches, not just its span. If R uses an eigendecomposition whose sign is arbitrary, normalise signs in both R and Python using the rule R applies. If R applies none, compare `abs` and record that in `docs/r-discrepancies.md`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 4: Newton solvers (dense, blocked, batch) and block design

**R sources:** `R/nbSolver.R`, `R/nbSolverBatch.R`, `R/nbBlockDesign.R`. **R tests:** `test-nbSolver.R`, `test-nbArrayBatch.R`, `test-nbBlockDesign.R`.

**Files:** Create `src/spanormpy/_solver.py`, `src/spanormpy/_block_design.py`, `tests/ref/make_solver.R`, `tests/test_solver.py`, `tests/test_block_design.py`

**Interfaces:**
- Produces: `nb_newton_solver(W, pen, absorb=None) -> Callable` and `nb_newton_solver_batch(W, pen, absorb=None) -> Callable`, each returning a solver closure with the same call contract as R's returned function (document the argument and return shapes in the docstring). Also `nb_gram_batch(W, wt_block, penalty_diag=None, backend="cpu", cell_tile=None) -> torch.Tensor`, `nb_absorb_gram_batch(W, pen, absorb, wt_block, cell_tile=None, parts=False)`, and `class NBBlockDesign` built by `nb_block_design(X, Z, block, rank_tol=1e-10) -> NBBlockDesign`, with `.shape`, `.colnames`, `to_dense() -> np.ndarray` and `__repr__`. The internal helpers `_design_eta(W, a)`, `_design_cross(W, r)` and `_design_col_nonzero(W, j)` accept either a dense tensor or an `NBBlockDesign`. All are torch, device-agnostic.

- [ ] **Step 1:** `make_solver.R` builds small designs with `set.seed(2)` (n=200 cells, p=8, a 3-level nested block, a penalty vector with zeros for unpenalised columns), weight blocks for 5 genes, and score vectors. It writes the outputs of the dense, blocked and batch solvers, `nbGramBatch`, `nbAbsorbGramBatch` (`parts` FALSE and TRUE) and `nbBlockDesign` (`as.matrix`, `.newtonSolverCompact`). Tests compare each against R at `rtol=1e-10` on `device`, and port the error cases from the three testthat files.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement, porting batched Cholesky (`.cholBatch`, `.cholSolveBatch`) with `torch.linalg.cholesky`/`cholesky_solve`. Keep R's jitter and failure handling.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 5: edgeR `estimateDisp` port

**R source:** edgeR 4.8.0 (`~/miniconda3/envs/latest-r/lib/R/library/edgeR`; read the functions with `Rscript -e 'edgeR:::fn'`). It covers exactly the call path `estimateDisp(y, design = matrix(1, n), offset = offs, tagwise = TRUE, robust = TRUE)$tagwise.dispersion` on a numeric matrix: `estimateDisp.default`, `adjustedProfileLik`, `maximizeInterpolant`, `WLEB`, `locfitByCol`, `movingAverageByCol`, the `aveLogCPM` used, `glmFit` / `mglmOneGroup` for the one-group case, and `systematicSubset` if reached. Port C code paths (for example `mglmOneGroup`, `compute_apl`) from the edgeR C sources in its GitHub tag `edgeR_4.8.0` if they are not in the installed package.

**Files:** Create `src/spanormpy/_edger.py`, `tests/ref/make_edger.R`, `tests/test_edger.py`

**Interfaces:**
- Produces: `estimate_disp_tagwise(Y: np.ndarray, offset: np.ndarray, robust: bool = True) -> np.ndarray`, where Y is genes × cells and `offset` is genes × cells or length cells. It returns tagwise dispersions and raises `EdgeRError` where edgeR errors (the caller maps that to `nan`, as R's `tryCatch` does).

- [ ] **Step 1:** `make_edger.R` writes inputs and `tagwise.dispersion` for: (a) `sim_counts(100, 300, 3)` with a zero offset; (b) the same with a genes × cells offset taken from a real SpaNorm iteration (call `SpaNorm:::.blockedOffsPsi` on a fit of `dlpfc_small`); (c) a case including all-zero and near-constant genes; (d) `robust = FALSE`. Also write the intermediate `common.dispersion`, `trended.dispersion`, `prior.df` and the APL grid. Tests: `test_tagwise_matches_r[case]` (`rtol=1e-8`, to allow locfit and optimiser round-off; keep a comment), and the intermediates at the same tolerance.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement in numpy/scipy (CPU only; R calls it on CPU too).
- [ ] **Step 4:** Run. Expected: PASS. If a tolerance cannot be met, stop and report the largest deviation and the function where it begins. Do not loosen the tolerance.
- [ ] **Step 5:** Commit.

### Task 6: Size factors and gene filtering

**R sources:** `scran::calculateSumFactors`, `scran::quickCluster` and their scuttle helpers (scran 1.38.0 installed; read with `Rscript -e`), `R/fastSizeFactors.R`, `R/filterGenes.R`. **R tests:** `test-fastSizeFactors.R`, `test-filterGenes.R`.

**Files:** Create `src/spanormpy/size_factors.py`, `src/spanormpy/_filter.py`, `tests/ref/make_sizefactors.R`, `tests/test_size_factors.py`, `tests/test_filter.py`

**Interfaces:**
- Produces: `compute_size_factors(adata, clusters=None, layer=None, key_added="spanorm_size_factors", copy=False, random_state=0)`, which writes `obs[key_added]` and returns the array when `copy=True`. Also the internal `sum_factors(Y: np.ndarray|sparse (genes × cells), clusters: np.ndarray|None) -> np.ndarray` (exact port of `calculateSumFactors`: pooling, ref.clust, min.mean, positivity handling), `quick_cluster(Y, random_state=0) -> np.ndarray` (same recipe: scuttle log-normalisation, `min.size = 100`, approximate PCA with `d = 50`, SNN graph with `k = 10`, `type = "rank"`, igraph walktrap, and R's block/merging behaviour), `fast_size_factors(adata, layer=None) -> None` (writes `obs["spanorm_size_factors"]`), `_fast_size_factors(Y) -> np.ndarray`, `filter_genes(adata, prop=0.1, layer=None, inplace=True)` (R semantics; returns a boolean mask when `inplace=False`), and `_filter_genes(Y, prop) -> np.ndarray[bool]`.

- [ ] **Step 1:** `make_sizefactors.R` writes, for `dlpfc_small` and for `sim_counts(200, 600, 4)`: `quickCluster` clusters, `calculateSumFactors(Y, clusters = cl)`, `calculateSumFactors(Y)` without clusters, `fastSizeFactors_intl`, and `filterGenes_intl` for `prop ∈ {0.1, 0.2, 0.5}`. Tests:
  - `test_sum_factors_given_r_clusters_exact` (`rtol=1e-10`)
  - `test_sum_factors_no_clusters_exact`
  - `test_quick_cluster_close` (adjusted Rand index ≥ 0.9 against R clusters)
  - `test_default_size_factors_close` (Pearson correlation of logs > 0.99 against R `calculateSumFactors(Y, clusters = quickCluster(Y))`)
  - `test_fast_size_factors_exact`, `test_filter_genes_exact`
  - `test_zero_count_spot` (a spot with zero total gives a positive floored factor after `pmax(1e-08, ·)` in the caller, and `sum_factors` matches R's behaviour on it)
  - `test_obs_size_factors_column_not_read` (`obs["size_factors"]` set to 1 does not change `compute_size_factors` output)
  - the testthat cases.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with numpy/scipy sparse, `scipy.sparse.linalg.svds` or randomised SVD for PCA, and python-igraph `community_walktrap`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 7: NB model fitting core

**R sources:** `R/fitSpaNormNB.R` (all of it except the `normalise*` functions). **R tests:** `test-fitSpaNormNB.R`, `test-fitNB-offset.R`, `test-fitNB-psi.R`.

**Files:** Create `src/spanormpy/_fitnb.py`, `tests/ref/make_fitnb.R`, `tests/test_fitnb.py`

**Interfaces:**
- Consumes: `_backend` (Task 2), `_nbdist` (Task 2), `_edger.estimate_disp_tagwise` (Task 5).
- Produces:
  - `fit_nb(Y, W, idx=None, lambda_a=0.0, winsor=4, maxit_psi=25, maxit_nb=50, tol=1e-4, offset=None, psi=None, backend="auto", verbose=True, **kwargs) -> dict`, with R's returned list fields under the same names in snake_case: `alpha`, `gmean`, `psi`, `loglik`, `sampling`, and others as R returns them.
  - `fit_spanorm_nb(Y, W, idx, maxit_psi=25, tol=1e-4, maxn_psi=500, offset=None, psi=None, is_spanorm=False, winsor=4, backend="auto", gpu_mem_budget=None, msgfun=None, **kwargs) -> dict`
  - `fit_nb_given_psi(...)`, mirroring R's signature.
  - `calculate_mu(gmean, alpha, W, winsor=4, backend="cpu", offset=None) -> np.ndarray | torch.Tensor`
  - `_sample_psi_idx(idx: np.ndarray[bool], maxn_psi: int, rng: np.random.Generator) -> np.ndarray[bool]`, the only place the dispersion subset is drawn.
  - `idx` is a boolean mask over cells. The gene-blocked (`blocks`) path uses `gene_block_count`/`gene_block_indices`.

- [ ] **Step 1:** `make_fitnb.R` uses `dlpfc_small` with the W that `fitSpaNorm` builds (df.tps 2), an R-chosen `idx` and psi subset (recorded), `maxit.psi = 3`, and `tol = 1e-4`. It writes the inputs, every iteration's psi, the final `alpha`, `gmean`, `psi`, `loglik` and `calculateMu` output. It repeats this with an offset matrix (`test-fitNB-offset`), with fixed psi (`test-fitNB-psi`), and with a forced small GPU budget, so the blocked path runs with 3 gene blocks. Tests:
  - `test_fit_spanorm_nb_matches_r[device]` (psi injected through monkeypatched `_sample_psi_idx`, `rtol=1e-6`)
  - `test_fit_nb_offset_matches_r[device]`, `test_fit_nb_fixed_psi_matches_r[device]`
  - `test_blocked_equals_unblocked[device]` (`rtol=1e-10`)
  - `test_calculate_mu_matches_r`
  - `test_edger_failure_gives_nan_then_r_fallback` (make `estimate_disp_tagwise` raise and assert R's handling)
  - the testthat error and warning cases.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement on torch tensors. edgeR is called on host numpy arrays, as R does.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 8: Adjustments (logPAC, Pearson, meanbio, medbio, meanbatch, meanls) and blocked normalisation

**R sources:** the `normalise*` functions in `R/fitSpaNormNB.R`; `getAdjustmentFun`, `getGeneModels`, `normaliseBlocked` and `.normaliseBlockSize` in `R/mainSpaNorm.R`.

**Files:** Create `src/spanormpy/_adjust.py`, `tests/ref/make_adjust.R`, `tests/test_adjust.py`

**Interfaces:**
- Consumes: `calculate_mu`, `winsorise_psi`, `qnbinom`/`pnbinom` (Tasks 2 and 7).
- Produces: `normalise_logpac`, `normalise_pearson`, `normalise_meanbio`, `normalise_medbio`, `normalise_meanbatch` and `normalise_meanls`, each `(Y, scale_factor, fit: SpaNormFit) -> np.ndarray` (genes × cells). Also `get_adjustment_fun(gene_model: str, adj_method: str) -> Callable`, with R's `"auto"` resolution, and `normalise_blocked(adj_fun, Y, scale_factor, fit, n_jobs=1, block_size=None) -> np.ndarray` (row blocks; dense, sparse and backed inputs). This task creates `src/spanormpy/_fit.py` containing only the `SpaNormFit` dataclass, with the fields listed in Task 9. Task 9 adds validation and storage to the same file. Task 8 tests construct the dataclass directly from R reference arrays.

- [ ] **Step 1:** `make_adjust.R` writes, for a fitted `dlpfc_small` model, every adjustment method's output with `scale.factor ∈ {1, 2.5}`, plus a batch fit for `meanbatch`. Tests: `test_adjustment_matches_r[method][device]` (`rtol=1e-6` relative to R outputs computed from the same fit arrays; deterministic given the fit, so tighten to `1e-10` if achieved), `test_blocked_equals_whole[block_size]`, and `test_auto_resolution`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 9: `SpaNormFit` object and AnnData storage

**R sources:** `R/AllClasses.R`; `subsetFitGenes` and `getSpaNormFit` in `R/mainSpaNorm.R`; storage writes in `mainSpaNorm.R`, `polishSpaNorm.R` and `mainSpaNormSVG.R`. **R tests:** `test-SpaNormFit.R`.

**Files:** Create `src/spanormpy/_fit.py`, `tests/test_fit_store.py`

**Interfaces:**
- Produces: `@dataclass class SpaNormFit` with fields `ngenes:int, ncells:int, gene_model:str, df_tps:np.ndarray[int], sample_p:float, lambda_a:float, batch:np.ndarray|None, W:np.ndarray (cells × p), alpha:np.ndarray (genes × p), gmean:np.ndarray, psi:np.ndarray, wtype:np.ndarray[str] (len p), loglik:np.ndarray, sampling:np.ndarray[str] (len cells), polish:dict` and `.validate()` (R `validSpaNormFit`). Also:
  - `is_polished(fit) -> bool`
  - `subset_genes(fit, mask) -> SpaNormFit`
  - `store_fit(adata, fit, name="spanorm") -> None`, which writes per the spec table: `varm[f"{name}_alpha"]`, `var[f"{name}_gmean"]`, `var[f"{name}_psi"]`, `obsm[f"{name}_W"]`, `obs[f"{name}_sampling"]`, `uns["spanorm"][name] = {scalars, wtype, W/alpha colnames, loglik, batch levels, polish scalars}`, and the per-gene and per-spot polish arrays into `var`/`varm`/`obs`/`obsm` with the `f"{name}_polish_"` prefix.
  - `load_fit(adata, name="spanorm", validate=True) -> SpaNormFit | None`, which rebuilds the fit from whatever genes and spots `adata` currently holds, so subsets are consistent by construction.
  - `get_spanorm_fit(adata, null=False, validate=True) -> SpaNormFit` (public, R `getSpaNormFit`).
- Storage must work on a non-view AnnData. Callers make the copy when given a view (Task 11).

- [ ] **Step 1:** Tests: `test_store_load_roundtrip` (all fields equal), `test_subset_after_store_equals_subset_genes` (`load_fit(adata[:, mask].copy())` equals `subset_genes(fit, mask)`), `test_spot_subset_roundtrip`, `test_uns_holds_no_per_gene_or_per_spot_arrays` (every array in `uns["spanorm"]` has length ≠ n_obs and ≠ n_vars, or is labels of length p), `test_validate_errors` (the testthat cases), and `test_is_polished`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 10: `spanorm()` main entry point

**R sources:** `R/mainSpaNorm.R` (SpatialExperiment method, `.spaNormCore`, `fitSpaNorm`, `sampleRandom`, `matchDftps`, `matchLambda`), `R/checkers.R` (`checkBatch`, `checkSPE` coordinate rules). **R tests:** `test-mainSpaNorm.R`, `test-checkBatch.R`, `test-checkers.R`.

**Files:** Create `src/spanormpy/_io.py`, `src/spanormpy/spanorm.py`, `tests/ref/make_spanorm.R`, `tests/test_spanorm.py`, `tests/test_io.py`; modify `src/spanormpy/__init__.py`

**Interfaces:**
- Consumes: Tasks 3, 6, 7, 8 and 9.
- Produces (`_io.py`):
  - `resolve_input(data, layer=None, spatial_key="spatial", table_key=None) -> tuple[AnnData, Callable[[AnnData], Any]]`, which accepts AnnData or SpatialData and returns the table plus a write-back function
  - `get_counts(adata, layer) -> np.ndarray|sparse (genes × cells)`
  - `get_coords(adata, spatial_key) -> np.ndarray (cells × 2)`
  - `check_batch(batch, nobs) -> np.ndarray|None`, where `batch` is an obs key, an array or a matrix, ported from R `checkBatch`.
- Produces (`spanorm.py`): `spanorm(data, sample_p=0.25, gene_model="nb", adj_method="auto", scale_factor=1.0, df_tps=6, lambda_a=1e-4, batch_key=None, batch=None, size_factors=None, tol=1e-4, step_factor=0.5, maxit_nb=50, maxit_psi=25, maxn_psi=500, overwrite=False, backend="auto", n_jobs=1, verbose=True, layer=None, spatial_key="spatial", table_key=None, random_state=0, copy=False, **kwargs)`.
  - Returns `None`, or an AnnData when `copy=True`.
  - Writes `layers["logcounts"]` (dense float64, or sparse if R would return sparse), the fit via `store_fit(name="spanorm")`, `obs["spanorm_size_factors"]` when computed, and `uns["spanorm"]["adj_method"]`.
  - `size_factors` is an array or an obs key. When it is None and no fit is stored, it calls `size_factors.sum_factors(Y, quick_cluster(Y))`, as R calls scran.
  - Internal `_sample_random(coords, nsub, rng) -> np.ndarray[int]` is the only place the GLM cell subset is drawn.

- [ ] **Step 1: Write the failing tests.** `make_spanorm.R` runs `SpaNorm` on `dlpfc_small` as an SPE, with `sizeFactors` set to `scran::calculateSumFactors` output (exported), `set.seed(10)`, `sample.p = 0.25`, `df.tps = 2` and `tol = 1e-3`. It exports `fit$sampling`, `alpha`, `gmean`, `psi`, `W`, `loglik` and `logcounts`. It repeats this with `df.tps = c(4, 2)`, with a batch covariate (a 2-level split by x coordinate), with each `adj.method`, and with the reuse paths (a second call with a different `adj.method`, `overwrite = TRUE`, and mismatched `df.tps`/`lambda.a`, recording R's errors or messages). Tests:
  - `test_spanorm_matches_r[device]`: inject R's sampling by monkeypatching `_sample_random` and `_sample_psi_idx` from `fit$sampling`, pass `size_factors=` from R, then compare `varm["spanorm_alpha"]`, `var["spanorm_psi"]`, `var["spanorm_gmean"]`, `obsm["spanorm_W"]` and `layers["logcounts"]` at `rtol=1e-6`.
  - `test_df_tps_pair`, `test_batch_matches_r`, `test_adj_method_reuse_no_refit` (the fit arrays are unchanged and logcounts match R), `test_overwrite_refits`, `test_mismatch_messages`.
  - `test_spatialdata_input` (a SpatialData wrapping `dlpfc_small` gives the same logcounts as AnnData), with `pytest.importorskip("spatialdata")`.
  - `test_default_pipeline_close`: no injection, Python's own size factors and sampling. The per-gene Pearson correlation of logcounts against R's default `SpaNorm(dlpfc_small)` exported by `make_spanorm.R` must be > 0.99 for every gene, as the spec requires. Never loosen this.
  - `test_io.py`: coordinates as an int array, as a DataFrame and with 3 columns; errors for 1 column and for NaN; `check_batch` cases from `test-checkBatch.R` (unused categorical levels, a single level, NaN, numeric matrix).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 11: Input robustness of the AnnData API (Review Focus 1–5)

**Files:** Create `tests/test_api_robustness.py`; modify `src/spanormpy/_io.py` and `src/spanormpy/spanorm.py` as the tests require

**Interfaces:**
- Consumes: `spanorm`, `load_fit` (Tasks 9 and 10).

- [ ] **Step 1:** Tests:
  - `test_sparse_equals_dense[csr|csc][int64|float32]`: logcounts and fit equal at `rtol=1e-10`.
  - `test_counts_in_layer`.
  - `test_view_input_is_copied_not_written` (passing `adata[:, :200]` raises no `ImplicitModificationWarning`, and the result is a new object or an error matching scanpy's convention of refusing views, whichever is clearer; document the choice).
  - `test_rerun_after_gene_subset_reuses_fit` (fit, subset genes, rerun with `adj_method="pearson"`; equals R `SpaNorm` on the subsetted SPE, which `make_spanorm.R` adds).
  - `test_rerun_after_spot_subset_matches_r` (whatever R does on a spot-subsetted SPE carrying a fit: reuse, refit or error).
  - `test_all_zero_gene` and `test_zero_total_spot`, each matching R (added to `make_spanorm.R`): the same error type and message intent, or the same finite output; no NaN in logcounts.
- [ ] **Step 2:** Run. Expected: some FAIL.
- [ ] **Step 3:** Fix the glue code.
- [ ] **Step 4:** Run the full suite with `pytest -q`. Expected: all PASS.
- [ ] **Step 5:** Commit.

### Task 12: Quasi-likelihood dispersion

**R sources:** `R/quasilik.R`. **R tests:** `test-quasilik.R`.

**Files:** Create `src/spanormpy/_quasilik.py`, `tests/ref/make_quasilik.R`, `tests/test_quasilik.py`

**Interfaces:**
- Produces: `nb_unit_deviance(y, mu, phi)`, `nb_deviance_moments(mu, phi, eps=1e-10, maxterms=20000)`, `ql_moment_table(lmu_range, lphi_range, step_mu=0.08, step_phi=0.12, ngrid=None, nphi=None) -> dict`, and `ql_dispersion(y, mu, phi, design=None, p=None, prior=1.0, leverage="trace", moments="table", ngrid=256, nphi=64, table=None) -> dict` (R's list fields).

- [ ] **Step 1:** `make_quasilik.R` writes outputs for a grid of `mu` and `phi`, the moment table, `.qlInterp` at off-grid points, and `qlDispersion` for every `leverage` × `moments` combination on a `dlpfc_small` fit. Tests at `rtol=1e-10`, except the `"table"` interpolation: if R uses `approx` or bilinear interpolation, match it exactly. Port the testthat cases.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 13: Polish engines and `polish_nb` / `nb_profile_psi`

**R sources:** `R/polishEngine.R`, `R/polishEngineBatch.R`, `R/polishNB.R`. **R tests:** `test-polishEngine.R`, `test-polishNB.R` (plus `helper-polish.R` fixtures).

**Files:** Create `src/spanormpy/_polish_engine.py`, `src/spanormpy/_polish_batch.py`, `src/spanormpy/polish.py`, `tests/ref/make_polish_nb.R`, `tests/test_polish_nb.py`

**Interfaces:**
- Consumes: `_solver` (Task 4), `_block_design` (Task 4), `_nbdist` (Task 2).
- Produces:
  - `nb_mu_floor() -> float`
  - `polish_nb(Y, W, alpha, psi, lambda_a=0.0, offset=None, absorb=None, absorb_batch=None, start_cols=None, psi_method="profile", psi_range=(1e-3, 1e3), warm=False, maxit=50, tol=1e-8, engine="batch", batch_size=None, block_size=None, backend="cpu", n_jobs=1, verbose=False) -> dict` (R's list fields)
  - `nb_profile_psi(Y, W, alpha, psi, psi_range=(1e-3, 1e3), block_size=None, n_jobs=1, offset=None) -> np.ndarray`
  - The batch memory budget is read from `spanormpy.settings.polish_mem_budget` (default `1e9`), in place of R's `options(SpaNorm.polish.mem.budget)`.
  - `.bplapplySingleBLAS` becomes a thread pool with BLAS threads limited through `threadpoolctl` when available.

- [ ] **Step 1:** `make_polish_nb.R` ports the `helper-polish.R` fixtures and writes `polishNB` outputs for engine ∈ {batch, gene}, psi.method ∈ {profile, fixed}, with and without offset, with absorb (a nested block), with an `nbBlockDesign` W, with `warm` TRUE, and `nbProfilePsi`. Tests: `test_polish_nb_matches_r[engine][psi_method][device]` (`rtol=1e-6`), `test_batch_equals_gene_engine`, `test_profile_psi_matches_r`, and the testthat cases (shape errors, `psi_range` validation).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The batch engine requires float64 (`.requireFloat64`) and errors on MPS, as R does.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 14: `polish_spanorm`

**R sources:** `R/polishSpaNorm.R`, `R/polishSpaNormModel.R`, `R/polishSpaNormJoint.R`. **R tests:** `test-polishSpaNorm.R`, `test-polishSpaNormModel.R`, `test-polishSpaNormJoint.R`.

**Files:** Create `src/spanormpy/_polish_model.py`, `src/spanormpy/polish_spanorm.py`, `tests/ref/make_polish_spanorm.R`, `tests/test_polish_spanorm.py`

**Interfaces:**
- Consumes: `polish_nb` (Task 13), `load_fit`/`store_fit` (Task 9), adjustments (Task 8), `_io` (Task 10).
- Produces: `polish_spanorm(data, adj_method="auto", scale_factor=1.0, psi_method="fixed", ls="fixed", cells="all", null=True, maxit=50, tol=1e-8, engine="batch", batch_size=None, backend="cpu", n_jobs=1, overwrite=False, verbose=True, layer=None, table_key=None, copy=False, **kwargs)`.
  - It stores polished fits under `spanorm`/`spanorm_null`, and the originals under `spanorm_unpolished`/`spanorm_null_unpolished`.
  - It rewrites `layers["logcounts"]` and drops stale SVG columns from `var` (R line 218).
  - It ports `.unpenalisedRankDeficient`, `.polishHoldOut`, `.unpenalisedGroups` and `.polishPsiRange`, and the shared-LS joint path `ls="joint"`.

- [ ] **Step 1:** `make_polish_spanorm.R` runs on the Task 10 fit (with SVG run first, so the null fit exists): `ls ∈ {fixed, joint}`, `psi.method ∈ {fixed, profile}`, `cells ∈ {all, fit}`, `null` TRUE and FALSE, and the overwrite and re-polish paths. It exports R's null fit too. Python tests seed `spanorm_null` via `store_fit` from those arrays, because `spanorm_svg` arrives in Task 15. Tests compare every stored fit array and logcounts at `rtol=1e-6`, and port the three testthat files (rank-deficiency errors, hold-out, uninformed-LS warnings).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 15: SVG detection

**R sources:** `R/mainSpaNormSVG.R`. **R tests:** `test-mainSpaNormSVG.R`.

**Files:** Create `src/spanormpy/svg.py`, `tests/ref/make_svg.R`, `tests/test_svg.py`

**Interfaces:**
- Consumes: `fit_spanorm_nb` (Task 7), storage (Task 9), `polish_spanorm` internals (Task 14) for the paired-null path.
- Produces: `spanorm_svg(data, backend="auto", verbose=True, layer=None, table_key=None, copy=False)` and `top_svgs(data, n=10, fdr=1.0) -> pd.DataFrame` (columns `svg_F`, `svg_p`, `svg_fdr`, sorted by FDR, indexed by gene name). `spanorm_svg` writes `var["svg_F"]`, `var["svg_p"]` and `var["svg_fdr"]`, keeps `F.raw` as `var["svg_F_raw"]` (R attribute), and stores the technical fit as `spanorm_null`. Internal: `fit_spanorm_technical(Y, fit, msgfun, **kw)`, `svg_test(Y, fit, fit_technical) -> pd.DataFrame`, `_polished_alike`, `_svg_paired_null`.

- [ ] **Step 1:** `make_svg.R` runs `SpaNormSVG` on the Task 10 fit (unpolished), and on a polished fit (the paired-null path). It writes the technical fit arrays, the SVG columns, `F.raw` and `topSVGs(n = 20, fdr = 0.05)`. Tests: `test_svg_matches_r[device]` (`rtol=1e-6` on F and the technical fit, with p and FDR compared on log scale, `rtol=1e-6`), `test_paired_null_matches_r`, `test_top_svgs_matches_r` (same gene order), `test_default_pipeline_svg_close` (with no injection, the Spearman correlation of `svg_F` against R's default-pipeline SVG run exported by `make_svg.R` is > 0.95), and the testthat cases (no fit stored, mismatched fits).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 16: SpaNorm PCA

**R sources:** `R/mainSpaNormPCA.R`. **R tests:** `test-mainSpaNormPCA.R`.

**Files:** Create `src/spanormpy/pca.py`, `tests/ref/make_pca.R`, `tests/test_pca.py`

**Interfaces:**
- Consumes: `top_svgs`, `load_fit(name="spanorm_null")`.
- Produces: `spanorm_pca(data, nsvgs=3000, ncomponents=50, svg_fdr=1.0, residuals="deviance", key_added="X_spanorm_pca", svd_solver="exact", n_jobs=1, copy=False)`. It writes `obsm[key_added]` and `uns[key_added]` (variance, and rotation for the SVGs used, stored in `varm[key_added + "_loadings"]` with zeros for genes not used). Internal: `get_residuals(Y, fit_technical, type) -> np.ndarray` and `deviance_residuals(Y, fit_technical)`.

- [ ] **Step 1:** `make_pca.R` writes deviance and Pearson residuals for the top SVGs, and `SpaNormPCA(..., BSPARAM = BiocSingular::ExactParam())` scores. Tests: `test_residuals_match_r` (`rtol=1e-10`) and `test_pca_matches_r_up_to_sign` (align the sign of each component by the sign of its largest-magnitude loading, then `rtol=1e-6`), plus the testthat cases.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. `svd_solver="exact"` uses `np.linalg.svd`, and `"randomized"` is offered for large data.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 17: Plotting

**R sources:** `R/plotSpatial.R`, `R/plotCovariates.R`.

**Files:** Create `src/spanormpy/pl/__init__.py`, `src/spanormpy/pl/_spatial.py`, `src/spanormpy/pl/_covariate.py`, `tests/test_plotting.py`

**Interfaces:**
- Produces: `pl.plot_spatial(adata, what="annotation", color=None, layer=None, basis=None, img=False, crop=False, img_alpha=1.0, size=None, circles=False, ax=None, show=None) -> matplotlib.axes.Axes`. `what` is one of `"annotation" | "expression" | "reduceddim"`. `color` names an obs column, a gene, or a component of `obsm[basis]`. The image comes from `uns["spatial"][library_id]["images"]["hires"|"lowres"]` and its scale factors. Also `pl.plot_covariate(adata, covariate="biology", **kwargs) -> Axes`, which plots the fitted covariate surface exactly as R computes it (`biology`, `ls`, `batch`).

- [ ] **Step 1:** Tests (Agg backend): each `what` returns an Axes with one scatter collection of n_obs points; `img=True` adds an image when `uns["spatial"]` exists and raises a clear error when it does not; `crop=True` sets the axis limits to the spot extent; `plot_covariate` values equal R's covariate values (exported by a new `make_plot.R`, `rtol=1e-6`).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with matplotlib, in scanpy style (`show` semantics as in `sc.pl`).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 18: GPU verification and benchmark on Bunya

**Files:** Create `scripts/slurm_gpu_tests.sh`, `scripts/benchmark_gpu.py`, `docs/benchmark.md`

**Interfaces:**
- Consumes: all public functions.

- [ ] **Step 1:** `slurm_gpu_tests.sh` requests one GPU (`--partition=gpu_cuda --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00`; check the Bunya partition names with the `hpc-job-submission` skill), activates `spanormpy` and runs `pytest -q -m "gpu or not gpu"`. Expected: every `[cuda]`-parametrised test runs and passes. Paste the summary line into `docs/benchmark.md`.
- [ ] **Step 2:** Check that R's GPU path is the same algorithm as its CPU path. Read the `backend == "gpu"` branches in `fitSpaNormNB.R`, `polishNB.R` and `mainSpaNormSVG.R`. If any branch differs beyond the device, write an R GPU reference (with `make_*_gpu.R`, run inside the same SLURM job, needing R torch with CUDA) and a matching test. Otherwise record "same algorithm" in `docs/benchmark.md`.
- [ ] **Step 3:** `benchmark_gpu.py` times `spanorm` and `spanorm_svg` on the full HumanDLPFC data (exported by `make_dlpfc.R` as `dlpfc_full.h5ad`, git-ignored) for `backend="cpu"` (torch threads = allocated cores) and `"gpu"`. Expected: GPU wall time is lower than CPU for both. Record the times in `docs/benchmark.md`. If the GPU is slower, stop and report the profile of the slowest function.
- [ ] **Step 4:** Commit.

### Task 19: Documentation and tutorial

**Files:** Create `docs/r-to-python.md` (a table mapping every R export in `NAMESPACE` at `c12e5e5` to its Python name, or to "internal: <module>" for the R GPU helper wrappers), `docs/tutorial.ipynb` (port of `vignettes/SpaNorm.Rmd` using `dlpfc_full.h5ad` and squidpy/scanpy loading), `docs/api.md` (autosummary); modify `README.md` (install, quickstart, citation from `../SpaNorm/inst/CITATION`, and download-tracking badges: PyPI version `https://img.shields.io/pypi/v/spanormpy`, PyPI monthly downloads `https://static.pepy.tech/badge/spanormpy/month` linking to `https://pepy.tech/project/spanormpy`, total downloads `https://static.pepy.tech/badge/spanormpy` and conda-forge downloads `https://img.shields.io/conda/dn/conda-forge/spanormpy`. Mirror them in `docs/index.md`. Add a "Download statistics" note to the README saying the counts come from PyPI/conda-forge and appear once the package is published. Add no telemetry or network code to the package — user decision 2026-10-19)

- [ ] **Step 1:** Write `tests/test_api_coverage.py::test_every_r_export_mapped`. It parses `NAMESPACE` from `git -C ../SpaNorm show c12e5e5:NAMESPACE` (skip the test if `../SpaNorm` is missing) and asserts that every export appears in `docs/r-to-python.md`, and that every Python name listed there is importable from `spanormpy` or `spanormpy.pl`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write the docs. Execute the notebook with `jupyter nbconvert --execute --to notebook docs/tutorial.ipynb` (on a GPU node or with `backend="cpu"` and `sample_p=0.05`).
- [ ] **Step 4:** Run `pytest -q`. Expected: all PASS. The notebook runs without error.
- [ ] **Step 5:** Commit.

---

## Website tasks (added 2026-10-19; see spec Addendum)

Global for Tasks 20–22:
- Work in the SpaNorm worktree `../SpaNorm-site` on branch `docs/quarto-site`. Quarto is `~/miniconda3/envs/latest-r/bin/quarto` (1.8). The Python env is `/scratch/user/uqdbhuva/envs/spanormpy`.
- Never modify `../SpaNorm`. Never push to SpaNorm `master` or `gh-pages`.

### Task 20: Site scaffold and tabbed Get started

**Files:** Create `site/_quarto.yml`, `site/index.qmd`, `site/get-started.qmd`, `site/styles.scss`, `site/.gitignore`; modify `.Rbuildignore` (`^site$`).

- [ ] **Step 1:** Create the worktree: `git -C ../SpaNorm worktree add ../SpaNorm-site -b docs/quarto-site master`.
- [ ] **Step 2:** Write `_quarto.yml` with `project: type: website`, the navbar (Home, Get started, Articles, Reference ▸ R / Python, R vs Python, News, GitHub), the Bootstrap 5 cosmo theme close to pkgdown, and `execute: freeze: auto`. Set reticulate to `RETICULATE_PYTHON=/scratch/user/uqdbhuva/envs/spanormpy/bin/python` through a `site/_environment` file.
- [ ] **Step 3:** Write `get-started.qmd` as a port of `vignettes/SpaNorm.Rmd`. Every computational step is a `::: {.panel-tabset group="language"}` block with an R tab (knitr chunk) and a Python tab (python chunk via reticulate, spanormpy API on the same data loaded from an h5ad export of HumanDLPFC). Keep the computation small (`sample.p` / `sample_p` = 0.05, `df.tps` = 2).
- [ ] **Step 4:** Run `quarto render site` and confirm it exits 0 and that `_site/get-started.html` contains both tabs with outputs and figures. Commit, including `_freeze/`.

### Task 21: References, articles and the R vs Python page

**Files:** Create `site/reference/r/` (generated), `site/reference/python/` (generated), `site/scripts/build_reference.sh`, `site/articles/{svg-pca,polishing,large-data-gpu}.qmd`, `site/r-vs-python.qmd`, `site/news.qmd`.

- [ ] **Step 1:** `build_reference.sh` installs `altdoc` into `site/_rlib` (git-ignored) and converts SpaNorm's Rd files to Quarto pages under `reference/r/`, with an index grouped like pkgdown's reference. It also runs `quartodoc build` with the config in `_quarto.yml` (package `spanormpy`, sections Normalisation / SVG & PCA / Polishing / Low-level / Plotting / Backend), producing `reference/python/`.
- [ ] **Step 2:** Write the articles, each with R | Python tabs, frozen. Large-data-GPU summarises `docs/benchmark.md` from spanormpy and shows `backend="gpu"` alongside `backend = "gpu"`.
- [ ] **Step 3:** `r-vs-python.qmd` includes spanormpy's `docs/r-to-python.md` and `docs/r-discrepancies.md`. A pre-render script copies them from `$SPANORMPY_DIR` (default `../spanormpy`) into `site/_includes/`.
- [ ] **Step 4:** Run `bash site/scripts/build_reference.sh && quarto render site`. Expected: it exits 0, every navbar link resolves (check with a small link check over `_site/*.html`), and every exported R function and every Python name in `spanormpy.__all__` has a page. Commit.

### Task 22: CI workflow and cross-links

**Files:** Create `.github/workflows/quarto-site.yaml`; delete `.github/workflows/pkgdown.yaml` and `_pkgdown.yml` on this branch; modify spanormpy `README.md` and `docs/index.md` (in the spanormpy repo) to link the site.

- [ ] **Step 1:** Write the workflow:
  - Trigger: `push: master`, `workflow_dispatch`, and `pull_request` (render only, no deploy on PRs).
  - Setup: `r-lib/actions/setup-r` + `setup-r-dependencies` (SpaNorm + altdoc + reticulate), `actions/setup-python` + `pip install git+https://github.com/BhuvaLab/spanormpy quartodoc`, and `quarto-dev/quarto-actions/setup`.
  - Build: run `build_reference.sh`, then `quarto render site`. Frozen outputs mean nothing re-executes.
  - Deploy: `quarto-dev/quarto-actions/publish` with `target: gh-pages` and `path: site`.
  - Pin action versions.
- [ ] **Step 2:** Validate the YAML with `python -c "import yaml; yaml.safe_load(open(...))"` and `actionlint` if available. Re-run `quarto render site` from a clean clone of the branch to prove CI-equivalence offline.
- [ ] **Step 3:** In spanormpy, point the README "Documentation" link at https://bhuvalab.github.io/SpaNorm/ and note that the site covers both languages. Commit in each repo, and push only the `docs/quarto-site` branch of SpaNorm and `port/spanorm` of spanormpy.
