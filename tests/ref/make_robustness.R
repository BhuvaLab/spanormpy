# Reference behaviour of R SpaNorm on awkward inputs (Task 11): rerun after a gene or spot
# subset of an SPE carrying a fit, an all-zero gene, a zero-total spot. Errors are recorded
# with tryCatch (status 0 = ok, 1 = error) so the Python tests can match "error vs handled".
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
})

sce <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
Yall <- as.matrix(assay(sce, "counts"))
gidx <- sort(order(-rowSums(Yall))[1:60])          # 60 well-expressed genes: every spot has counts
Y <- as.matrix(assay(sce, "counts"))[gidx, ]
xy <- as.matrix(reducedDim(sce, "spatial"))
mk <- function(Y, xy, sf = NULL) {
  spe <- SpatialExperiment(assays = list(counts = Y), spatialCoords = xy)
  colnames(spe) <- paste0("s", seq_len(ncol(Y)))
  rownames(spe) <- rownames(Y)
  if (!is.null(sf)) sizeFactors(spe) <- sf
  spe
}
stopifnot(all(colSums(Y) > 0))
set.seed(1)
sf <- as.numeric(scran::calculateSumFactors(Y, clusters = scran::quickCluster(Y)))
common <- list(df.tps = 2, tol = 1e-3, sample.p = 0.25, verbose = TRUE)

# status 0 + the object, or status 1 + the error message; messages/warnings collected
attempt <- function(spe, seed, ...) {
  msgs <- character(); warns <- character()
  set.seed(seed)
  args <- c(list(spe), modifyList(common, list(...)))
  res <- tryCatch(
    withCallingHandlers(
      do.call(SpaNorm, args),
      message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") },
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
    ),
    error = function(e) conditionMessage(e)
  )
  ok <- !is.character(res)
  out <- list(status = if (ok) 0L else 1L, error = if (ok) "" else res,
              msgs = if (length(msgs)) msgs else "", warns = if (length(warns)) warns else "")
  if (ok) {
    f <- S4Vectors::metadata(res)$SpaNorm
    lc <- as.matrix(assay(res, "logcounts"))
    out <- c(out, list(logcounts = unname(lc), alpha = f@alpha, psi = f@psi, gmean = f@gmean,
                       W = f@W, sampling = as.character(f@sampling),
                       ngenes = f@ngenes, ncells = f@ncells, nan = sum(is.nan(lc) | is.na(lc))))
  }
  list(out = out, spe = if (ok) res else NULL)
}

out <- list(gidx = gidx, sf = sf)
base <- attempt(mk(Y, xy, sf), 10)
out$base <- base$out

# ---- gene subset of an SPE carrying the fit (metadata is kept by `[`)
sub_g <- seq(2, 60, by = 3)
out$sub_g <- sub_g
out$full_pearson <- attempt(base$spe, 99, adj.method = "pearson")$out   # fit reuse on all genes
out$gene_subset <- attempt(base$spe[sub_g, ], 21, adj.method = "pearson")$out

# ---- spot subset of an SPE carrying the fit
sub_s <- seq(1, ncol(Y), by = 2)
out$sub_s <- sub_s
out$spot_subset <- attempt(base$spe[, sub_s], 22, adj.method = "pearson")$out

# ---- all-zero gene (gene 5), size factors supplied
Yz <- Y; Yz[5, ] <- 0
out$zero_gene <- attempt(mk(Yz, xy, sf), 10)$out
out$zero_gene_idx <- 5L

# ---- zero-total spot (spot 7): size factors supplied, supplied as 0, and computed by R
Ys <- Y; Ys[, 7] <- 0
out$zero_spot_sf <- attempt(mk(Ys, xy, sf), 10)$out
sf0 <- sf; sf0[7] <- 0
out$zero_spot_sf0 <- attempt(mk(Ys, xy, sf0), 10)$out
out$zero_spot_nosf <- attempt(mk(Ys, xy, NULL), 10)$out
out$zero_spot_idx <- 7L
write_ref(out, "robustness")
