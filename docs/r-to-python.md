# R to Python mapping

Every export of R SpaNorm 1.7.15 (commit `c12e5e5`, `NAMESPACE`) and its spanormpy counterpart.
"internal" rows are building blocks that R exports for the GPU and polishing engines; they live in
private modules of spanormpy (no stability guarantee), or are plain `torch` operations where R needed
a wrapper around `torch` for CPU/GPU dispatch.

Argument names are snake_case with R's defaults (`sample.p` is `sample_p`); R's `BPPARAM` is
`n_jobs`. Intentional deviations are listed in [r-discrepancies](r-discrepancies.md) and summarised in the
README ("Differences from R").

| R export | Python | Notes |
|---|---|---|
| `SpaNorm` | `spanormpy.spanorm` | in place on AnnData (`copy=True` to return a copy); output in `layers["logcounts"]` |
| `SpaNormPCA` | `spanormpy.spanorm_pca` | result in `obsm["X_spanorm_pca"]` |
| `SpaNormSVG` | `spanormpy.spanorm_svg` | statistics in `var["svg_F"]`, `var["svg_p"]`, `var["svg_fdr"]` |
| `add_vec_mat_gpu` | internal: `torch` (broadcast addition) | R wrapper not needed; torch tensors are used directly |
| `calculateMu` | `spanormpy.calculate_mu` |  |
| `checkGPU` | internal: `spanormpy._backend.check_gpu` |  |
| `diag_mat` | internal: `torch` (`torch.diag_embed`) | R wrapper not needed |
| `dnbinom_gpu` | internal: `spanormpy._nbdist.dnbinom` |  |
| `fastSizeFactors` | `spanormpy.fast_size_factors` | writes `obs["spanorm_size_factors"]` |
| `filterGenes` | `spanormpy.filter_genes` | `inplace=False` returns R's logical vector |
| `fitNB` | `spanormpy.fit_nb` |  |
| `getBackendDevice` | internal: `spanormpy._backend.get_backend_device` |  |
| `getBackendDtype` | internal: `spanormpy._backend.get_backend_dtype` |  |
| `getGPUMemoryBudget` | internal: `spanormpy._backend.get_gpu_memory_budget` |  |
| `gpuDtypeBytes` | internal: `spanormpy._backend.gpu_dtype_bytes` |  |
| `hasBadValues` | internal: `spanormpy._backend.has_bad_values` |  |
| `invert_mat` | internal: `torch` (`torch.linalg.inv`) | R wrapper not needed |
| `invert_mat_batched` | internal: `torch` (`torch.linalg.inv`, batched) | R wrapper not needed |
| `isPolished` | `spanormpy.is_polished` | takes a stored `SpaNormFit` |
| `is_torch_tensor` | internal: `torch` (`isinstance(x, torch.Tensor)`) | R wrapper not needed |
| `matmul_gpu` | internal: `torch` (`torch.matmul`) | R wrapper not needed |
| `mult_vec_mat_gpu` | internal: `torch` (broadcast multiplication) | R wrapper not needed |
| `nbAbsorbGramBatch` | internal: `spanormpy._solver.nb_absorb_gram_batch` |  |
| `nbBlockDesign` | internal: `spanormpy._block_design.nb_block_design` | S3 methods (`as.matrix`, `dim`, `dimnames`, `print`) are methods of `NBBlockDesign` |
| `nbDevianceMoments` | internal: `spanormpy._quasilik.nb_deviance_moments` |  |
| `nbGramBatch` | internal: `spanormpy._solver.nb_gram_batch` |  |
| `nbMuFloor` | `spanormpy.nb_mu_floor` |  |
| `nbNewtonSolver` | internal: `spanormpy._solver.nb_newton_solver` |  |
| `nbNewtonSolverBatch` | internal: `spanormpy._solver.nb_newton_solver_batch` |  |
| `nbProfilePsi` | `spanormpy.nb_profile_psi` |  |
| `nbUnitDeviance` | internal: `spanormpy._quasilik.nb_unit_deviance` |  |
| `plotCovariate` | `spanormpy.pl.plot_covariate` | matplotlib; returns `Axes` |
| `plotSpatial` | `spanormpy.pl.plot_spatial` | matplotlib; returns `Axes` |
| `polishNB` | `spanormpy.polish_nb` |  |
| `polishSpaNorm` | `spanormpy.polish_spanorm` |  |
| `qlDispersion` | internal: `spanormpy._quasilik.ql_dispersion` |  |
| `qlMomentTable` | internal: `spanormpy._quasilik.ql_moment_table` |  |
| `resetGPUCache` | internal: `spanormpy._backend.reset_gpu_cache` |  |
| `rowSums_gpu` | internal: `torch` (`Tensor.sum(dim=1)`) | R wrapper not needed |
| `setGPUMemoryBudget` | internal: `spanormpy._backend.set_gpu_memory_budget` | also `spanormpy.settings` for the polish memory budget |
| `tcrossprod_gpu` | internal: `torch` (`A @ B.T`) | R wrapper not needed |
| `toGPUMatrix` | internal: `spanormpy._backend.as_tensor` |  |
| `toGPUVector` | internal: `spanormpy._backend.as_tensor` |  |
| `toRMatrix` | internal: `torch` (`Tensor.cpu().numpy()`) | R wrapper not needed |
| `topSVGs` | `spanormpy.top_svgs` | returns a `DataFrame` |
| `tpsBasis` | internal: `spanormpy._tps.tps_basis` |  |
| `SpaNormFit` | internal: `spanormpy._fit.SpaNormFit` | exported class; stored in AnnData `var`/`varm`/`obsm`/`uns["spanorm"]`, the `$` method is attribute access |
