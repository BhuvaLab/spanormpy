# Reference values for plotCovariate (R/plotCovariates.R): the covariate surfaces
# normaliseMeanBio / normaliseMeanLS / normaliseMeanBatch evaluated at scale factor 1.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SingleCellExperiment))
ns <- function(n) get(n, envir = asNamespace("SpaNorm"))

spe <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
cells <- seq(1, ncol(spe), by = 4)
Y <- as.matrix(assay(spe, "counts"))[round(seq(1, 300, length.out = 20)), cells]
dimnames(Y) <- NULL
xy <- reducedDim(spe, "spatial")[cells, ]
LS <- colSums(Y) / mean(colSums(Y))
batch <- factor(ifelse(xy[, 1] > median(xy[, 1]), "b2", "b1"))

set.seed(3)
f <- ns("fitSpaNorm")(Y = Y, coords = xy, sample.p = 0.25, gene.model = "nb", df.tps = 2,
                      lambda.a = 1e-4, batch = batch, LS = LS, msgfun = function(...) NULL,
                      tol = 1e-2, maxit.nb = 10, maxit.psi = 3, backend = "cpu")
out <- list(
  coords = unname(xy), batch = as.character(batch),
  fit = list(ngenes = f@ngenes, ncells = f@ncells, df_tps = f@df.tps, lambda_a = f@lambda.a,
             W = f@W, W_colnames = colnames(f@W), alpha = f@alpha, gmean = f@gmean, psi = f@psi,
             wtype = as.character(f@wtype), loglik = f@loglik, sampling = as.character(f@sampling)),
  biology = unname(ns("normaliseMeanBio")(Y, 1, f)),
  ls = unname(ns("normaliseMeanLS")(Y, 1, f)),
  batch_effect = unname(ns("normaliseMeanBatch")(Y, 1, f))
)
write_ref(out, "plot")
