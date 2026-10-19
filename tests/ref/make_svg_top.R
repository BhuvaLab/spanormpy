# topSVGs on a table with NaN FDRs: R's `results[results$svg.fdr <= fdr, ]` turns each NaN into an
# all-NA row ("NA", "NA.1", ...), which order() puts last and min(n, nrow) counts.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SpatialExperiment))
ng <- 12L; nc <- 5L
counts <- matrix(rpois(ng * nc, 5), ng, nc, dimnames = list(paste0("g", 1:ng), paste0("c", 1:nc)))
spe <- SpatialExperiment(assays = list(counts = counts), spatialCoords = cbind(runif(nc), runif(nc)))
fdr <- c(0.5, 0.01, NaN, 0.2, 0.01, NaN, 0.04, 0.9, 0.03, NaN, 0.2, 0.6)
rowData(spe)$svg.F <- seq_len(ng) * 1.5
rowData(spe)$svg.p <- fdr / 2
rowData(spe)$svg.fdr <- fdr
out <- list(fdr = fdr)
cases <- list(a = list(n = 10, fdr = 1), b = list(n = 5, fdr = 1), c = list(n = 20, fdr = 0.05),
              d = list(n = 8, fdr = 0.5))
for (k in names(cases)) {
  res <- topSVGs(spe, n = cases[[k]]$n, fdr = cases[[k]]$fdr)
  out[[k]] <- list(n = cases[[k]]$n, fdr = cases[[k]]$fdr, nrow = nrow(res), F = res$svg.F,
                   rownames = rownames(res))
}
write_ref(out, "svg_top")
