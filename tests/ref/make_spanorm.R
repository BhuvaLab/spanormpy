# Reference values for spanorm() end to end (R SpaNorm on the dlpfc_small SPE).
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
})

sce <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
Y <- as.matrix(assay(sce, "counts"))
xy <- as.matrix(reducedDim(sce, "spatial"))
spe <- SpatialExperiment(assays = list(counts = Y), spatialCoords = xy)
colnames(spe) <- colnames(sce)
rownames(spe) <- rownames(sce)

# exact size factors as the pipeline would compute them (exported for injection)
set.seed(1)
sf <- as.numeric(scran::calculateSumFactors(Y, clusters = scran::quickCluster(Y)))
spe_sf <- spe
sizeFactors(spe_sf) <- sf

gsub_idx <- seq(1, nrow(Y), by = 4)  # export subset of genes for the smaller scenarios

run <- function(spe, seed, ...) {
  msgs <- character(); warns <- character()
  set.seed(seed)
  out <- withCallingHandlers(
    SpaNorm(spe, ...),
    message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") },
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  list(spe = out, msgs = msgs[grepl("^\\(|sampled", msgs)], warns = warns)
}
pack <- function(res, full = FALSE) {
  f <- S4Vectors::metadata(res$spe)$SpaNorm
  lc <- as.matrix(assay(res$spe, "logcounts"))
  list(ngenes = f@ngenes, ncells = f@ncells, df_tps = f@df.tps, lambda_a = f@lambda.a,
       W = f@W, W_colnames = colnames(f@W), alpha = f@alpha, gmean = f@gmean, psi = f@psi,
       wtype = as.character(f@wtype), loglik = f@loglik, sampling = as.character(f@sampling),
       logcounts = if (full) unname(lc) else unname(lc[gsub_idx, ]),
       msgs = res$msgs, warns = if (length(res$warns)) res$warns else "")
}
common <- list(df.tps = 2, tol = 1e-3, sample.p = 0.25, verbose = TRUE)
call <- function(spe, seed, ...) do.call(run, c(list(spe, seed), modifyList(common, list(...))))

out <- list(sf = sf, gsub_idx = gsub_idx)

base <- call(spe_sf, 10)
out$base <- pack(base, full = TRUE)

# reuse: other adjustment methods on the stored fit, no refit
reuse <- list()
for (m in c("pearson", "medbio", "meanbio", "logpac")) {
  r <- call(base$spe, 99, adj.method = m)
  reuse[[m]] <- pack(r)
}
out$reuse <- reuse

# overwrite = TRUE refits (new seed => new sample)
out$overwrite <- pack(call(base$spe, 11, overwrite = TRUE))
# mismatched df.tps / lambda.a / batch refit; matching pair (2, 1) reuses
out$df3 <- pack(call(base$spe, 12, df.tps = 3))
out$lambda <- pack(call(base$spe, 13, lambda.a = 1e-3))
out$df21 <- pack(call(base$spe, 14, df.tps = c(2, 1)))
bt <- factor(ifelse(xy[, 1] > median(xy[, 1]), "b2", "b1"))
out$batch_change <- pack(call(base$spe, 15, batch = bt))

# df.tps pair
out$df42 <- pack(call(spe_sf, 10, df.tps = c(4, 2)))

# batch covariate
out$batch <- pack(call(spe_sf, 10, batch = bt))
out$batch_vec <- as.character(bt)

# R's own default pipeline (own quickCluster size factors, df.tps = 6, tol = 1e-4)
set.seed(10)
dflt <- suppressMessages(suppressWarnings(SpaNorm(spe)))
out$default_logcounts <- unname(as.matrix(assay(dflt, "logcounts")))
dfit <- S4Vectors::metadata(dflt)$SpaNorm
out$default_sampling <- as.character(dfit@sampling)  # R's cell / dispersion subsets
out$default_sf <- exp(dfit@W[, 1])                    # R's own quickCluster size factors
# a second default run with another seed: the sampling-noise floor of R against itself
set.seed(11)
dflt2 <- suppressMessages(suppressWarnings(SpaNorm(spe)))
out$default_logcounts_seed11 <- unname(as.matrix(assay(dflt2, "logcounts")))
write_ref(out, "spanorm")
