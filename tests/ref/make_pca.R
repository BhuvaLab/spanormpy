# Reference values for SpaNormPCA (R/mainSpaNormPCA.R): residuals and ExactParam PCA.
# 40 gene x 500 spot fixture (the polish fixture of make_polish_spanorm.R), SpaNorm then SpaNormSVG.
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
})
sce <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
Y <- as.matrix(assay(sce, "counts"))
xy <- as.matrix(reducedDim(sce, "spatial"))
NG <- 40L; NC <- 500L
top <- order(-rowSums(Y))[seq_len(NG)]
Ys <- Y[top, seq_len(NC)]
xys <- xy[seq_len(NC), , drop = FALSE]
storage.mode(Ys) <- "integer"
colnames(Ys) <- paste0("c", seq_len(NC))
spe <- SpatialExperiment(assays = list(counts = Ys), spatialCoords = xys)
colnames(spe) <- colnames(Ys); rownames(spe) <- rownames(Ys)

pack_fit <- function(f) list(
  ngenes = f@ngenes, ncells = f@ncells, df_tps = f@df.tps, lambda_a = f@lambda.a,
  W = f@W, W_colnames = colnames(f@W), alpha = f@alpha, gmean = f@gmean, psi = f@psi,
  wtype = as.character(f@wtype), loglik = f@loglik, sampling = as.character(f@sampling))

set.seed(20260925)
spe <- suppressMessages(suppressWarnings(
  SpaNorm(spe, df.tps = 2L, sample.p = 0.25, verbose = FALSE, backend = "cpu")))
spe <- suppressMessages(SpaNormSVG(spe, backend = "cpu", verbose = FALSE))
md <- S4Vectors::metadata(spe)
ft <- md$SpaNormNull

out <- list(Y = Ys, coords = xys, genes = rownames(Ys),
            input = pack_fit(md$SpaNorm), null = pack_fit(ft),
            svg_F = rowData(spe)$svg.F, svg_p = rowData(spe)$svg.p, svg_fdr = rowData(spe)$svg.fdr)

# residuals on all genes (R internals)
out$dev_all <- SpaNorm:::getResiduals(Ys, ft, "deviance")
out$pearson_all <- SpaNorm:::getResiduals(Ys, ft, "pearson")

run <- function(nsvgs, ncomp, fdr, res) {
  sv <- rownames(topSVGs(spe, nsvgs, fdr))
  r <- SpaNormPCA(spe, nsvgs = nsvgs, ncomponents = ncomp, svg.fdr = fdr,
                  BSPARAM = BiocSingular::ExactParam(), residuals = res)
  pc <- reducedDim(r, "PCA")
  list(nsvgs = nsvgs, ncomp = ncomp, fdr = fdr, svgs = sv, x = unname(pc),
       percentVar = attr(pc, "percentVar"), varExplained = attr(pc, "varExplained"),
       rotation = unname(attr(pc, "rotation")), rot_rownames = rownames(attr(pc, "rotation")),
       dim = dim(pc), dimnames = colnames(pc) %||% "")
}
out$dev <- run(25, 5, 1, "deviance")
out$pearson <- run(25, 5, 1, "pearson")
out$fdr <- run(40, 4, 0.5, "deviance")
write_ref(out, "pca")
