# Changelog

## 0.1.0

First release. A Python / scverse port of the R package SpaNorm 1.7.16 (Bioconductor devel).

- `spanorm`, `spanorm_svg`, `top_svgs`, `spanorm_pca`, `polish_spanorm`, `polish_nb`,
  `nb_profile_psi`, quasi-likelihood dispersion, `fit_nb`, `filter_genes`, size factors
  (built-in scran-style pooling), `tps_basis`, `nb_block_design` and plotting
  (`pl.plot_spatial`, `pl.plot_covariate`).
- `AnnData` input (and `SpatialData` through its table). Per-gene fit components live in
  `var`/`varm`, per-spot components in `obs`/`obsm`, so subsetting genes keeps the fit consistent.
- PyTorch backend for CPU and CUDA (float64); Apple Metal runs in float32.
- Outputs agree with R at rtol 1e-6 (closed-form steps 1e-10) when R's size factors and spot
  sampling are supplied; see `docs/r-discrepancies.md` for every intentional difference.
